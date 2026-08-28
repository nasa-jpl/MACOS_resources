function out = dw_dsurf_multi(session, rx_path, opts)
%MACOS.DW_DSURF_MULTI  Multi-field dw/dKr + dw/dKc supervisor.
%   out = macos.dw_dsurf_multi(SESSION, RX_PATH, ...) loads RX_PATH on
%   SESSION, snapshots the nominal source FoV, then loops field points --
%   each iteration absolutely sets ChfRayDir via set_src_fov and runs
%   dw_dsurf at the new field.  The per-field results are tiled into one
%   big OPDall canvas and scattered into one big dwdsall (canonical
%   state-vector form):
%
%       wall = dwdsall * x + w0_stacked
%
%   where wall is the column-major vectorisation of OPDall via m2v, and
%   x is the stacked (Kr, Kc) state of every POWERED optic (Element=
%   Reflector / Refractor with |Kr| << the flat sentinel).
%
%   The powered-surface (radius + conic) companion to
%   macos.dw_dz_zernike_multi, built on the same field-loop + tiling
%   machinery, a different DOF set.
%
%   REQUIRED USER INPUTS:
%     'field_x_rad'   half-FoV in x (direction cosine added to ChfRayDir)
%     'field_y_rad'   half-FoV in y (independent of x)
%
%   FIELD SET (one of):
%     default 5-field (center + 4 corners)
%     'grid' 'NxM'    auto-generate uniform N x M grid (center counted
%                     once when both N and M are odd)
%     'fields' FILE   override: rows of 'name dx_rad dy_rad tile_row tile_col'
%
%   OTHER NAME-VALUE PAIRS (forwarded to dw_dsurf):
%     'params' (cellstr subset of {'Kr','Kc'}), 'elts' (vector of element
%     IDs to include, default [] = all powered optics), 'delta', 'method',
%     'exit_pupil_elt', 'verbose'.
%
%   'ngridpts'  (default [] = keep the .in value) ray-grid sampling
%               override, applied once right after the Rx load; persists
%               across the per-field calls.  Clamped by the engine to
%               [3, model-size limit] (warns).
%   'reset_xp'  (default true) re-find the exit pupil (FEX, chief ray) for
%               EACH field before differencing, so the nominal wavefront is
%               referenced to that field's own chief ray and the gross field
%               TILT is removed (off-axis fields otherwise carry a large
%               linear-in-field tilt that swamps the OPD canvas).  A poke's
%               OWN tilt is retained -- the reference is fixed per field, not
%               re-fit after each poke.  Requires a STOP set and > 3 elements.
%               Restore scope: the pre-loop EP is snapshotted/restored via
%               get_xp/set_xp -- vpt/psi/rad (VptElt/PsiElt/KrElt at nElt-1)
%               only.  FEX-written auxiliary fields on the EP element
%               (RptElt, zElt, fElt, eElt, KcElt) are left as re-derived;
%               callers who hand-author those own re-asserting them.
%
%   'configs'   (default [] = today's single-block call, byte-identical)
%               a 1xNc struct array of CONFIGURATIONS -- named sets of
%               element setting overrides ("zoom positions"; in our
%               systems more often a COMPENSATION state).  Each entry is
%               .name + .set, a cell of setter invocations
%               {fname, elt, args...} dispatched against the Session.
%               The Jacobian is then evaluated per (configuration, field)
%               and the blocks stack as extra ROWS.  v1 accepts only the
%               Row COUNT: a configuration that changes ray survival
%               (a tilt can vignette a field) contributes a different
%               number of rows, so the stack is sum-over-configurations,
%               not exactly Nc*Nw.  Slice a block with
%               out.indxall.config == c -- the blocks are contiguous.
%               pose setters perturb / set_elt_vpt / set_elt_psi /
%               set_elt_rpt / set_elt_csys; the runner owns the
%               modify()-after-setters rule and the snapshot / restore /
%               ASSERT cycle.  See macos.dw_dx_multi,
%               private/config_axis.m, design/PLAN_CONFIGURATIONS.md.
%
%   OUTPUT STRUCT FIELDS:
%     dwdsall       Nw x Ns canonical state-vector Jacobian
%     dwdxall       alias (== dwdsall) -- canonical-form consumers
%     w0_stacked    Nw x 1 stacked nominal OPDs (m2v of OPDall)
%     indxall       struct with i, j, size -- m2v round-trip metadata
%     OPDall        full tiled OPD canvas (un-vectorised)
%     channel_names Ns x 1 cell array  ('Elt N Kr' / 'Elt N Kc')
%     iElt / param  Ns x 1 per-channel element id + param ('Kr'|'Kc')
%     field_table   Nfields x 4: [dx_rad, dy_rad, tile_row, tile_col]
%     field_names   Nfields x 1 cell array
%     chfraydir_nom 3 x 1 nominal ChfRayDir
%     per_field_dwds      Nfields x 1 cell of single-field dwds blocks
%     per_field_w_nom_2d  Nfields x 1 cell of single-field nominal OPDs
%     rx_path / delta / method / wf_elt / params  -- echoed inputs
%
%   See also: macos.dw_dsurf, macos.dw_dz_zernike_multi, macos.dw_dx_multi.

arguments
    session
    rx_path                     (1,:) char {mustBeNonempty}
    opts.field_x_rad            (1,1) double = NaN
    opts.field_y_rad            (1,1) double = NaN
    opts.fields                 (1,:) char = ''
    opts.grid                   (1,:) char = ''
    opts.params                 cell = {'Kr','Kc'}
    opts.elts                   (:,1) double = []
    opts.delta                  (1,1) double = 1e-6
    opts.method                 (1,:) char {mustBeMember(opts.method, ...
                                  {'central','forward'})} = 'central'
    opts.exit_pupil_elt         (1,1) double {mustBeInteger} = -1
    opts.reset_xp               (1,1) logical = true
    opts.reset_xp_method        (1,:) char {mustBeMember( ...
        opts.reset_xp_method, {'fex','sxp','pupil_find'})} = 'fex'
    opts.pupil_find_opts        cell = {}
    opts.pf_scope               (1,:) char {mustBeMember( ...
        opts.pf_scope, {'config','field'})} = 'field'
    opts.pf_probe_rad           (1,1) double = NaN
    opts.verbose                (1,1) logical = false
    opts.ngridpts               double {mustBeScalarOrEmpty} = []
    opts.stop_elt               double {mustBeScalarOrEmpty} = []
    opts.configs                          = []
    opts.src_samp               double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.compute_los            (1,1) logical = false
    opts.spot_elt               double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.orient (1,:) char {mustBeMember(opts.orient, {'raw','xy'})} = 'raw'   % OPD array orientation (doc/opd_conventions.md)
    opts.sign   (1,:) char {mustBeMember(opts.sign, {'opl','wavefront'})} = 'opl' % OPD sign convention
    opts.remove_ptt (1,1) logical = false   % piston+tip+tilt removed per
                                            % response column (see dw_dsurf)
end

if isnan(opts.field_x_rad) || isnan(opts.field_y_rad)
    error('macos:dw_dsurf_multi:fov', ...
        'field_x_rad and field_y_rad are required');
end

% ---- Field set ----------------------------------------------------
if ~isempty(opts.fields)
    fields = load_field_file(opts.fields);
elseif ~isempty(opts.grid)
    [nx, ny] = parse_grid_spec(opts.grid);
    fields = make_grid_field_set(nx, ny, opts.field_x_rad, opts.field_y_rad);
else
    fields = make_5field_set(opts.field_x_rad, opts.field_y_rad);
end
n_fields = numel(fields);
tile_rows = max(arrayfun(@(s) s.tile_row, fields)) + 1;
tile_cols = max(arrayfun(@(s) s.tile_col, fields)) + 1;
fprintf('[setup] %d field points, tile grid %dx%d\n', ...
    n_fields, tile_rows, tile_cols);
for k = 1:n_fields
    fprintf('  field %-8s: dir-offset=(%+.3e,%+.3e) rad  tile=(%d,%d)\n', ...
        fields(k).name, fields(k).dx, fields(k).dy, ...
        fields(k).tile_row, fields(k).tile_col);
end

% ---- Load + snapshot nominal source -------------------------------
session.load_rx(rx_path);
apply_ngridpts(session, opts.ngridpts, 'dw_dsurf_multi');

% Apply source sampling if specified
if ~isempty(opts.src_samp)
    session.set_src_sampling(opts.src_samp);
    session.modify();  % Flush cache so the new sampling takes effect
end

% Apply the aperture stop here so it survives across the per-field calls
% (they run reload_rx=false and never touch the stop state).  Mirrors
% macos.dw_dx_multi's 'stop_elt': a deck that carries no header ApStop=
% -- the zoom fixture is one -- has nowhere else to get one, and the
% exit-pupil machinery (reset_xp / FEX) requires a stop.
if ~isempty(opts.stop_elt)
    session.stop(int32(opts.stop_elt));
end

% ---- Configuration axis -------------------------------------------
% Validated AFTER the load (element ids can be range-checked) and BEFORE
% anything is applied.  Absent/empty => n_cfg == 1 and every line below
% degenerates to the pre-configs single-block path.
ep_elt_chk = [];
if opts.reset_xp, ep_elt_chk = session.num_elt() - 1; end
cfgs = config_axis('validate', opts.configs, session.num_elt(), ...
                    'dw_dsurf_multi', ep_elt_chk);
has_cfg = ~isempty(cfgs);
n_cfg   = max(1, numel(cfgs));
if has_cfg
    fprintf('[setup] %d configuration(s):\n', n_cfg);
    for ci = 1:n_cfg
        fprintf('  config %-8s: %d setter(s) on elt %s\n', cfgs(ci).name, ...
            numel(cfgs(ci).set), mat2str(cfgs(ci).elts));
    end
end

nom = session.get_src_fov();
fprintf('[setup] nominal ChfRayDir = [%g %g %g]; zSrc = %.3e\n', ...
    nom.src_dir, nom.zSrc);

% Snapshot the prescription's exit-pupil reference (elt nElt-1 geometry)
% so the per-field FEX resets can be undone before returning.  Track
% whether FEX actually moves the EP (writes only into a Return/Reference
% nElt-1; no-pupil decks are silent no-ops) + guard a powered-optic
% clobber -- see private/reset_xp_guard.
reset_ep_moved = false;
use_pf = strcmp(opts.reset_xp_method, 'pupil_find');
% pf_scope='field' (Dave, 2026-08-25): one mini-cone fit PER (config,
% field) block -- a 3x3 probe grid of half-width pf_delta centered on
% each field, so the placed sphere's axis parallels that combo's OWN
% chief ray and the field tilt is absorbed per block (as fex does), with
% the bundle-fit vertex instead of the chief crossing.  Residual tilt in
% w_nom is then the coma channel -- signal, not artifact.
% pf_scope='config' keeps the field-set-wide fit: one frozen sphere per
% configuration, per-field tilt retained.
pf_fld = use_pf && strcmp(opts.pf_scope, 'field');
pf_delta = opts.pf_probe_rad;
if pf_fld && isnan(pf_delta)
    rmax = max(hypot([fields.dx], [fields.dy]));
    if rmax == 0
        error('macos:dw_dsurf_multi:pfProbe', ...
            ['pf_scope=''field'' with an all-on-axis field set: pass ' ...
             '''pf_probe_rad'' (the 3x3 probe half-width, rad) explicitly.']);
    end
    % 0.15x the field half-width: local enough that every combo's fit
    % runs in the same small-cone regime, large enough that the probe
    % chief-ray crossings stay well-conditioned (vertex noise scales as
    % crossing scatter / delta -- the same near-parallel degeneracy
    % FEX's telecentric guard exists for).
    pf_delta = 0.15 * rmax;
end
if pf_fld
    fprintf('[setup] pupil_find scope=field: 3x3 probe cone, half-width %.3g rad\n', ...
        pf_delta);
end
pf_out = struct([]);
if opts.reset_xp
    xp0 = macos.get_xp();
    ep_is_powered = reset_xp_guard('is_powered', session);
end

% ---- Per-field loop -----------------------------------------------
% Each iteration uses set_src_fov ABSOLUTELY (no perturb-undo needed) +
% modify() to flush the trace cache so the new ChfRayDir takes effect,
% and passes reload_rx=false to dw_dsurf so it does NOT call load_rx
% (which would reset ChfRayDir back to the prescription nominal).
per_field_dwds   = cell(n_cfg, n_fields);
per_field_w_nom  = cell(n_cfg, n_fields);
per_field_struct = cell(n_cfg, n_fields);
if opts.compute_los
    per_field_dcdx = cell(n_cfg, n_fields);
end
names = {};  iElt_out = [];  param_out = {};
for ic = 1:n_cfg
% Order (PLAN_CONFIGURATIONS 2.1): apply the configuration -> modify()
% once -> run the field loop, whose per-field reset_xp then derives every
% field's exit pupil FROM THE CONFIGURED GEOMETRY -> restore -> assert.
% pupil_find round-trip hygiene (2026-08-27 w_nom audit): the guard's
% save_rx -> load_rx round trip compounds across a sequential
% multi-config call (dw_dx_multi's config loop has the full story) --
% reload fresh at the top of every configuration after the first so the
% sequential call matches the checkpointed path by construction.
if has_cfg && use_pf && ic > 1
    session.load_rx(rx_path);
    apply_ngridpts(session, opts.ngridpts, 'dw_dsurf_multi');
    if ~isempty(opts.src_samp)
        session.set_src_sampling(opts.src_samp);
        session.modify();
    end
    if ~isempty(opts.stop_elt)
        session.stop(int32(opts.stop_elt));
    end
end
if has_cfg
    snap = config_axis('snapshot', session, cfgs(ic).elts);
    config_axis('apply', session, cfgs(ic));
    fprintf('[config %s] applied (%d setter(s))\n', cfgs(ic).name, ...
        numel(cfgs(ic).set));
end
if opts.reset_xp && use_pf && ~pf_fld
    % reset_xp_method='pupil_find': place the cone-convergence best-fit
    % sphere ONCE for this configuration (field-set-wide fit -- the cone
    % aperture IS the field set), then run the field loop with the
    % per-field FEX reset OFF.  A frozen, best-fit exit pupil: the
    % per-field tilt reference behaves as reset_xp=false always has.
    % Restore the NOMINAL source before the fit: the previous
    % configuration's field loop leaves the session at its LAST field, and
    % save_rx would bake that direction into the temp deck -- pupil_map
    % derives its probe standoff from the saved deck's source, so every
    % configuration after the first got one identical, polluted placement
    % (measured: cfgs 2..5 dep_rms 5.5e-3 vs 1.9e-3 clean, nominal rows
    % identical).  Gated by tPupilFindMethod/
    % test_config_sphere_is_independent_of_predecessors.
    session.set_src_fov('src_pos', nom.src_pos, 'src_dir', nom.src_dir, ...
                        'zSrc', nom.zSrc);
    session.modify();
    Fpf = zeros(n_fields, 2);
    for kf = 1:n_fields, Fpf(kf,:) = [fields(kf).dx, fields(kf).dy]; end
    pf_ic = reset_xp_guard('pupil_find', session, Fpf, opts.stop_elt, ...
                           session.num_elt() - 1, opts.pupil_find_opts, xp0);
    reset_ep_moved = true;              % placed by construction
    fprintf(['[pupil_find] sphere placed: vtx moved %.3g from FEX, ' ...
             'dep_rms %.3g, conv R %.4g\n'], ...
            norm(pf_ic.vtx(:) - pf_ic.fex.vpt(:)), pf_ic.dep_rms, ...
            pf_ic.conv_radius);
    m0 = struct('scope', 'config', 'config', ic, 'field', 0, ...
                'vtx', pf_ic.vtx_written(:).', 'bundle_vtx', pf_ic.vtx(:).', ...
                'rad', pf_ic.rad, ...
                'fex_vpt', pf_ic.fex.vpt(:).', ...
                'vtx_minus_fex', norm(pf_ic.vtx(:) - pf_ic.fex.vpt(:)), ...
                'dep_rms', pf_ic.dep_rms, 'conv_radius', pf_ic.conv_radius);
    if isempty(pf_out), pf_out = m0; else, pf_out(end+1) = m0; end %#ok<AGROW>
end
for k = 1:n_fields
    if opts.reset_xp && pf_fld
        % Per-combo placement: restore the NOMINAL source first (the deck
        % standoff derivation -- same hygiene as the per-config scope; the
        % guard restores the pristine EP into the saved deck each time),
        % then fit the 3x3 mini-cone centered on THIS field and place.
        % The sphere persists as element geometry across the block's poke
        % traces, so the poke's own tilt is retained -- the family
        % convention.
        session.set_src_fov('src_pos', nom.src_pos, 'src_dir', nom.src_dir, ...
                            'zSrc', nom.zSrc);
        session.modify();
        [pgx, pgy] = ndgrid([-1 0 1] * pf_delta);
        Fprobe = [fields(k).dx + pgx(:), fields(k).dy + pgy(:)];
        pf_ic = reset_xp_guard('pupil_find', session, Fprobe, opts.stop_elt, ...
                               session.num_elt() - 1, ...
                               [{'vertex', 'chief'}, opts.pupil_find_opts], xp0);
        reset_ep_moved = true;          % placed by construction
        fprintf(['[cfg %d field %s pupil_find] sphere placed: vtx moved ' ...
                 '%.3g from FEX, dep_rms %.3g\n'], ic, fields(k).name, ...
                norm(pf_ic.vtx(:) - pf_ic.fex.vpt(:)), pf_ic.dep_rms);
        m0 = struct('scope', 'field', 'config', ic, 'field', k, ...
                    'vtx', pf_ic.vtx_written(:).', 'bundle_vtx', pf_ic.vtx(:).', ...
                    'rad', pf_ic.rad, ...
                    'fex_vpt', pf_ic.fex.vpt(:).', ...
                    'vtx_minus_fex', norm(pf_ic.vtx(:) - pf_ic.fex.vpt(:)), ...
                    'dep_rms', pf_ic.dep_rms, 'conv_radius', pf_ic.conv_radius);
        if isempty(pf_out), pf_out = m0; else, pf_out(end+1) = m0; end %#ok<AGROW>
    end
    new_dir = field_to_chfraydir(nom.src_dir, fields(k).dx, fields(k).dy);
    session.set_src_fov('src_pos', nom.src_pos, 'src_dir', new_dir, ...
                        'zSrc', nom.zSrc);
    session.modify();   % flush trace cache so the new dir takes effect
    fprintf('[field %s] ChfRayDir = [%g %g %g]\n', ...
        fields(k).name, new_dir);
    if opts.reset_xp && ~use_pf
        % Re-reference this field's exit pupil to its OWN chief ray (FEX
        % writes the reference into elt nElt-1 = wf_elt) so the nominal
        % wavefront is tilt-removed.  It persists as element geometry across
        % the poke traces, and a poke's own tilt is retained (the reference
        % is NOT re-fit after poking).  Kr/Kc pokes act on the powered optics
        % (Reflector/Refractor); elt nElt-1 is a Return/Reference and is NOT
        % in the powered set, so the pokes never touch it.  See dw_dgrid_multi.
        % shared guard: raises the supervisor-level no-stop error and
        % absorbs the no-pupil-element FAIL -- see private/reset_xp_guard.
        reset_xp_guard('fex', session);
        reset_ep_moved = reset_xp_guard('check', session, xp0, ...
            reset_ep_moved, ep_is_powered);
    end
    sf = macos.dw_dsurf(session, rx_path, ...
        'params', opts.params, ...
        'elts', opts.elts, ...
        'delta', opts.delta, ...
        'method', opts.method, ...
        'exit_pupil_elt', opts.exit_pupil_elt, ...
        'verbose', opts.verbose, ...
        'reload_rx', false, ...
        'remove_ptt', opts.remove_ptt, ...
        'compute_los', opts.compute_los, ...
        'spot_elt', opts.spot_elt);    % keep current src_fov state
    per_field_dwds{ic, k} = sf.dwds;
    per_field_w_nom{ic, k} = sf.w_nom_2d;
    per_field_struct{ic, k} = sf;
    if opts.compute_los
        per_field_dcdx{ic, k} = sf.dcdx;
    end
    if isempty(names)
        names = sf.channel_names;  iElt_out = sf.iElt;  param_out = sf.param;
    elseif ~isequal(names, sf.channel_names)
        error('macos:dw_dsurf_multi:channelMismatch', ...
            ['field %s: channel_names differ from the first block ' ...
             '(%d vs %d channels) -- a configuration must not change ' ...
             'the channel list.'], fields(k).name, ...
            numel(sf.channel_names), numel(names));
    end
    col_rms_mean = mean(sqrt(mean(sf.dwds.^2, 1)));
    fprintf('[field %s] dwds shape [%d %d], mean col-RMS %.3e', ...
        fields(k).name, size(sf.dwds, 1), size(sf.dwds, 2), col_rms_mean);
    if opts.compute_los
        los_rms_mean = mean(sqrt(sum(sf.dcdx.^2, 2)));
        fprintf('  mean LOS-RMS %.3e', los_rms_mean);
    end
    fprintf('\n');
end
% Restore AFTER the channel loop has undone its own pokes, never
% interleaved with it.  The assertion is the load-bearing part.
if has_cfg
    config_axis('undo', session, cfgs(ic), snap);
    drift = config_axis('assert', session, snap, cfgs(ic).name, 'dw_dsurf_multi');
    fprintf(['[config %s] restored + verified '  ...
             '(worst pose drift %.1f%% of tolerance)\n'], ...
        cfgs(ic).name, 100 * drift);
end
end

% Restore source back to nominal.
session.set_src_fov('src_pos', nom.src_pos, 'src_dir', nom.src_dir, ...
                    'zSrc', nom.zSrc);
session.modify();

% Restore the prescription's exit-pupil reference (undo the per-field FEX
% writes to elt nElt-1) so the session is left as loaded.
if opts.reset_xp
    macos.set_xp(xp0.vpt, xp0.psi, xp0.rad);
    session.modify();
end
reset_xp_stamp = reset_xp_guard('finalize', opts.reset_xp, ...
    reset_ep_moved, session.num_elt() - 1);

% ---- Tile OPDall + scatter dwdsall --------------------------------
N = size(per_field_w_nom{1, 1}, 1);
% Each configuration gets its OWN field canvas first; macos.config_canvas
% then places those canvases -- at their geometric tile positions when the
% schedule declares them, so a zoom schedule reads as its corners and its
% centre exactly as the field set does inside one cell -- and builds a
% CONFIGURATION-MAJOR index.  So w for one configuration stacks its
% FIELDS, and w for the run stacks the CONFIGURATIONS.  Vectorising the
% assembled canvas directly would not do that: m2v walks column-major, so
% any outer layout that varies down a column interleaves the blocks.
canv = cell(1, n_cfg);
for ic = 1:n_cfg
    Cc = zeros(tile_rows * N, tile_cols * N);
    for k = 1:n_fields
        assert(size(per_field_w_nom{ic, k}, 1) == N, ...
            'macos:dw_dsurf_multi:gridSize', ...
            'block (%d,%d) has a different OPD grid size', ic, k);
        r0 = fields(k).tile_row * N;
        c0 = fields(k).tile_col * N;
        Cc(r0+1:r0+N, c0+1:c0+N) = per_field_w_nom{ic, k};
    end
    canv{ic} = Cc;
end
cfg_tiles = [];
if n_cfg >= 2, cfg_tiles = config_axis('tiles', cfgs); end
[OPDall, indxall] = macos.config_canvas(canv, cfg_tiles);
if ~has_cfg
    % preserved surface: no configuration field on the index struct
    indxall = rmfield(indxall, 'config');
end
w0_stacked = macos.m2v(OPDall, indxall);
Nw = numel(w0_stacked);
Ns = size(per_field_dwds{1, 1}, 2);
fprintf('[stack] OPDall [%d %d]; non-zero pixels = %d\n', ...
    size(OPDall, 1), size(OPDall, 2), Nw);

dwdsall = zeros(Nw, Ns);
row0 = 0;
for ic = 1:n_cfg
    % this configuration's own rows, in the order config_canvas used
    [~, ixc] = macos.m2v(canv{ic});
    ic_i = ixc.i(:);  ic_j = ixc.j(:);
for k = 1:n_fields
    tr = fields(k).tile_row;
    tc = fields(k).tile_col;
    in_tile = (ic_i > tr*N) & (ic_i <= (tr+1)*N) ...
            & (ic_j > tc*N) & (ic_j <= (tc+1)*N);
    i_local = ic_i(in_tile) - tr * N;
    j_local = ic_j(in_tile) - tc * N;
    % Build field-local m2v of this tile so we can map global rows back
    % to the per-field dwds rows.
    [~, field_indx] = macos.m2v(per_field_w_nom{ic, k});
    field_i = field_indx.i(:);
    field_j = field_indx.j(:);
    flat_local  = (j_local - 1) * N + i_local;
    flat_field  = (field_j  - 1) * N + field_i;
    [tf, loc] = ismember(flat_local, flat_field);
    if ~all(tf)
        error('macos:dw_dsurf_multi:scatter', ...
            'field %s: indxall references pixels outside per-field mask', ...
            fields(k).name);
    end
    global_rows = row0 + find(in_tile);
    dwdsall(global_rows, :) = per_field_dwds{ic, k}(loc, :);
    fprintf('[stack] field %s: scattered %d rows into dwdsall\n', ...
        fields(k).name, numel(global_rows));
end
    row0 = row0 + numel(ic_i);
end

fprintf('[stack] dwdsall shape [%d %d]; |dwdsall| max = %.3e\n', ...
    size(dwdsall, 1), size(dwdsall, 2), max(abs(dwdsall(:))));

% Center-tile sanity check: the (0,0) field's rows in dwdsall must
% exactly match its per_field_dwds block (max|diff| = 0 in practice).
ctr_idx = find_center_field_index(fields);
if ~isempty(ctr_idx)
  row0 = 0;
  for ic = 1:n_cfg
      [~, ixc] = macos.m2v(canv{ic});
    tr = fields(ctr_idx).tile_row;
    tc = fields(ctr_idx).tile_col;
    in_ctr = (ixc.i(:) > tr*N) & (ixc.i(:) <= (tr+1)*N) ...
           & (ixc.j(:) > tc*N) & (ixc.j(:) <= (tc+1)*N);
    dwdsall_ctr = dwdsall(row0 + find(in_ctr), :);
    dwds_C = per_field_dwds{ic, ctr_idx};
    max_diff = max(abs(dwdsall_ctr(:) - dwds_C(:)));
    fprintf('[check] dwdsall@center-tile vs per_field_dwds[center]: ');
    fprintf('max|diff| = %.3e ([%d %d])\n', ...
        max_diff, size(dwdsall_ctr, 1), size(dwdsall_ctr, 2));
    assert(max_diff == 0, ...
        'scatter bug: dwdsall@center-tile differs from per_field_dwds[center]');
      row0 = row0 + numel(ixc.i);
  end
else
    fprintf('[check] no (0,0)-offset field -- skipping center-tile check\n');
end

% ---- Pack output struct -------------------------------------------
out = struct();
out.dwdsall              = dwdsall;
out.dwdxall              = dwdsall;     % canonical-form alias
out.w0_stacked           = w0_stacked;
out.indxall              = indxall;
out.OPDall               = OPDall;
out.channel_names        = names;
out.iElt                 = iElt_out;
out.param                = param_out;
out.field_table          = arrayfun( ...
    @(s) [s.dx, s.dy, s.tile_row, s.tile_col], fields, ...
    'UniformOutput', false);
out.field_table          = vertcat(out.field_table{:});
out.field_names          = {fields.name}.';
out.chfraydir_nom        = nom.src_dir(:);
if has_cfg
    out.config_table       = cfgs(:);
    out.config_names       = {cfgs.name}.';
    out.per_field_dwds       = per_field_dwds;      % Nc x Nf
    out.per_field_w_nom_2d = per_field_w_nom;     % Nc x Nf
else
    % preserved surface: without 'configs' the cells keep their Nf x 1
    % shape and no configuration fields are added
    out.per_field_dwds       = per_field_dwds(1, :).';
    out.per_field_w_nom_2d = per_field_w_nom(1, :).';
end
out.rx_path              = rx_path;
out.delta                = opts.delta;
out.method               = opts.method;
out.wf_elt               = per_field_struct{1, 1}.wf_elt;
out.params               = opts.params;
out = apply_opd_convention(out, opts.orient, opts.sign);
out.reset_xp             = reset_xp_stamp;   % true | false | 'no-effect'
out.reset_xp_method      = opts.reset_xp_method;
if use_pf, out.pf_scope = opts.pf_scope; end
if ~isempty(pf_out), out.pupil_find = pf_out; end   % per-config metrics

% Add per-field LOS if SPOT was computed
if opts.compute_los
    out.dcdx_per_field = per_field_dcdx;
    if isempty(opts.spot_elt)
        out.spot_elt = session.num_elt();  % Default focal plane
    else
        out.spot_elt = opts.spot_elt;
    end
end
end


% =====================================================================
function fields = make_5field_set(field_x_rad, field_y_rad)
% Default 5-field: center + 4 corners.
fields = struct('name', {}, 'dx', {}, 'dy', {}, ...
                 'tile_row', {}, 'tile_col', {});
fields(end+1) = field_entry('C',  0,            0,            1, 1);
fields(end+1) = field_entry('UL', -field_x_rad, +field_y_rad, 2, 0);
fields(end+1) = field_entry('UR', +field_x_rad, +field_y_rad, 2, 2);
fields(end+1) = field_entry('LL', -field_x_rad, -field_y_rad, 0, 0);
fields(end+1) = field_entry('LR', +field_x_rad, -field_y_rad, 0, 2);
end


function fields = make_grid_field_set(nx, ny, field_x_rad, field_y_rad)
% Uniform NxM grid covering [-field_x_rad..+field_x_rad] x
% [-field_y_rad..+field_y_rad].
if nx > 1, dx_axis = linspace(-field_x_rad, +field_x_rad, nx); else, dx_axis = 0; end
if ny > 1, dy_axis = linspace(-field_y_rad, +field_y_rad, ny); else, dy_axis = 0; end
fields = struct('name', {}, 'dx', {}, 'dy', {}, ...
                 'tile_row', {}, 'tile_col', {});
for ir = 1:numel(dy_axis)
    for ic = 1:numel(dx_axis)
        dy = dy_axis(ir);  dx = dx_axis(ic);
        if (abs(dx) < 1e-30) && (abs(dy) < 1e-30)
            nm = 'C';
        else
            nm = sprintf('F_r%d_c%d', ir-1, ic-1);
        end
        fields(end+1) = field_entry(nm, dx, dy, ir-1, ic-1); %#ok<AGROW>
    end
end
end


function e = field_entry(name, dx, dy, tr, tc)
e.name = name;  e.dx = dx;  e.dy = dy;  e.tile_row = tr;  e.tile_col = tc;
end


function idx = find_center_field_index(fields)
idx = [];
for k = 1:numel(fields)
    if abs(fields(k).dx) < 1e-30 && abs(fields(k).dy) < 1e-30
        idx = k;  return;
    end
end
end


function new_dir = field_to_chfraydir(dir_nom, dx_rad, dy_rad)
% Direction-cosine offset on top of the nominal ChfRayDir, renormalised.
v = dir_nom(:) + [dx_rad; dy_rad; 0];
n = norm(v);
if n == 0
    error('macos:dw_dsurf_multi:zerodir', ...
        'zero-magnitude direction after field offset');
end
new_dir = v / n;
end


function [nx, ny] = parse_grid_spec(spec)
toks = regexp(lower(spec), 'x', 'split');
if numel(toks) ~= 2
    error('macos:dw_dsurf_multi:grid', ...
        '''grid'' must be ''NxM'' (e.g. ''3x3''); got %s', spec);
end
nx = str2double(toks{1});  ny = str2double(toks{2});
if isnan(nx) || isnan(ny) || nx < 1 || ny < 1
    error('macos:dw_dsurf_multi:grid', ...
        '''grid'' must be ''NxM'' with positive integers; got %s', spec);
end
end


function fields = load_field_file(fname)
% Free-form list: lines of 'name dx_rad dy_rad tile_row tile_col'.
fid = fopen(fname, 'r');
if fid < 0
    error('macos:dw_dsurf_multi:fields', ...
        'cannot open fields file: %s', fname);
end
c = onCleanup(@() fclose(fid)); %#ok<NASGU>
fields = struct('name', {}, 'dx', {}, 'dy', {}, ...
                 'tile_row', {}, 'tile_col', {});
while true
    ln = fgetl(fid);
    if ~ischar(ln); break; end
    s = strtrim(ln);
    if isempty(s) || startsWith(s, '#'), continue; end
    toks = regexp(s, '\s+', 'split');
    if numel(toks) < 5
        error('macos:dw_dsurf_multi:fields', ...
            'fields-file row needs 5 columns: %s', s);
    end
    fields(end+1) = field_entry(toks{1}, ...
        str2double(toks{2}), str2double(toks{3}), ...
        str2double(toks{4}), str2double(toks{5})); %#ok<AGROW>
end
end
