function out = dw_multi_core(session, rx_path, opts, F)
%DW_MULTI_CORE  The ONE multi-field/multi-config sensitivity supervisor.
%   out = dw_multi_core(SESSION, RX_PATH, OPTS, F) is the shared core
%   behind macos.dw_dx_multi / dw_dsurf_multi / dw_dz_zernike_multi /
%   dw_dgrid_multi (BRIEF_luis_round3 S3, 2026-09-05): the field-set
%   construction, load/stop/re-aim, configuration axis, reset_xp /
%   pupil_find machinery, per-(configuration, field) loop, canvas tiling
%   + Jacobian scatter, and output packing existed as four ~700-line
%   near-copies -- the "fix one, siblings stay broken" architecture.
%   Each public front now validates its own arguments (signatures
%   unchanged) and hands this core a family descriptor F:
%
%     F.name       front name, e.g. 'dw_dx_multi' (error ids + log tags)
%     F.jac        Jacobian field in the single-field out, e.g. 'dwdx'
%     F.all_names  cellstr of out names for the stacked Jacobian, in
%                  pack order (aliases included), e.g. {'dwdsall','dwdxall'}
%     F.per_name   out name for the per-field cell, e.g. 'per_field_dwdx'
%     F.single     @(session, rx_path, opts, hoist) -> sf  (the family's
%                  single-field driver, called with reload_rx=false)
%     F.hoist      [] | @(session, rx_path, opts) -> hoist  (parse-once
%                  work after the load, before the loop; may print)
%     F.meta       @(sf) -> struct of per-channel bookkeeping copied
%                  verbatim into out (iElt/param/kind/dof_idx/map_idx...)
%     F.extras     @(out, opts, sf1) -> out  (family echoes, applied
%                  BEFORE apply_opd_convention -- the incumbent order)
%     F.cfg_tag_fieldlog  dx-style "config X / " prefixes on field logs
%     F.los_reshape       conditional Nc reshape of dcdx_per_field (dx)
%     F.deprecate_sxp     warn once on reset_xp_method='sxp' (grid)
%
%   Behavior is the incumbents', verified byte-identical on the A/B
%   harness (all four families x {defaults, configs+pupil_find,
%   frozen-EP/orient} on e5hex1) -- see the sens-core commit message.
%   The empty-OPD guard (formerly dw_dx_multi only) now protects every
%   family.  OPTS is the front's validated struct; fields only some
%   fronts declare (stop_obj_pos, reload_rx) are read defensively.

stop_obj_pos = getf_(opts, 'stop_obj_pos', []);
do_reload    = getf_(opts, 'reload_rx', true);

if isnan(opts.field_x_rad) || isnan(opts.field_y_rad)
    error(eid_(F, 'fov'), 'field_x_rad and field_y_rad are required');
end

% ---- Field set ----------------------------------------------------
if ~isempty(opts.fields)
    fields = load_field_file(opts.fields, F);
elseif ~isempty(opts.grid)
    [nx, ny] = parse_grid_spec(opts.grid, F);
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

% ---- Load + stop + re-aim -----------------------------------------
% dw_dgrid_multi's 'reload_rx' lets a caller pre-install engine state
% (macos.elt_grid_add figures) a fresh load would wipe.
if do_reload
    session.load_rx(rx_path);
end
apply_ngridpts(session, opts.ngridpts, F.name);
if ~isempty(opts.src_samp)
    session.set_src_sampling(opts.src_samp);
    session.modify();  % Flush cache so the new sampling takes effect
end
% Apply the stop here so it survives across the per-field calls (they
% run reload_rx=false and never touch the stop state).  A deck with no
% header ApStop= has nowhere else to get one, and the exit-pupil
% machinery (reset_xp / FEX) requires a stop.
if ~isempty(opts.stop_elt) && ~isempty(stop_obj_pos)
    error(eid_(F, 'stop'), ...
        'stop_elt and stop_obj_pos are mutually exclusive');
end
if ~isempty(opts.stop_elt)
    session.stop(int32(opts.stop_elt));
elseif ~isempty(stop_obj_pos)
    session.stop_obj(stop_obj_pos(1), stop_obj_pos(2), stop_obj_pos(3));
end
% Per-field stop re-aim (Dave 2026-08-28): the stop-enforced chief IS
% the field's chief ray -- the stop is re-issued after EVERY per-field
% (and nominal-restore) set_src_fov below.  See private/field_stop_reaim.
reaim = field_stop_reaim(session, rx_path, opts.stop_elt, stop_obj_pos);

% ---- Family hoist (parse-once work: groups map, influence basis) ---
hoist = [];
if ~isempty(F.hoist)
    hoist = F.hoist(session, rx_path, opts);
end

% ---- Configuration axis -------------------------------------------
% Validated AFTER the load (element ids can be range-checked) and BEFORE
% anything is applied.  Absent/empty => n_cfg == 1 and every line below
% degenerates to the pre-configs single-block path.
ep_elt_chk = [];
if opts.reset_xp, ep_elt_chk = session.num_elt() - 1; end
cfgs = config_axis('validate', opts.configs, session.num_elt(), ...
                    F.name, ep_elt_chk);
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

if F.deprecate_sxp && strcmp(opts.reset_xp_method, 'sxp')
    warn_reset_xp_method_deprecated_(F);
end

% Snapshot the prescription's exit-pupil reference (elt nElt-1 geometry)
% so the per-field FEX resets can be undone before returning.  FEX
% writes only into a Return/Reference nElt-1 (no-pupil decks are silent
% no-ops); track whether it actually moved the EP + guard a
% powered-optic clobber -- see private/reset_xp_guard.
reset_ep_moved = false;
use_pf = strcmp(opts.reset_xp_method, 'pupil_find');
if strcmp(opts.fex_axis, 'centroid') && use_pf
    error(eid_(F, 'fexAxisScope'), ...
        ['fex_axis=''centroid'' applies to the FEX per-field reset ' ...
         'only; the pupil_find written reference is chief-tied by ' ...
         'doctrine.  Drop fex_axis or use reset_xp_method=''fex''.']);
end
% pf_scope='field' (Dave, 2026-08-25): one mini-cone fit PER (config,
% field) block -- a 3x3 probe grid of half-width pf_delta centered on
% each field, so the placed sphere's axis parallels that combo's OWN
% chief ray and the field tilt is absorbed per block (as fex does).
% pf_scope='config' keeps the field-set-wide fit: one frozen sphere per
% configuration, per-field tilt retained (diagnostic mode).
pf_fld = use_pf && strcmp(opts.pf_scope, 'field');
pf_delta = opts.pf_probe_rad;
if pf_fld && isnan(pf_delta)
    rmax = max(hypot([fields.dx], [fields.dy]));
    if rmax == 0
        error(eid_(F, 'pfProbe'), ...
            ['pf_scope=''field'' with an all-on-axis field set: pass ' ...
             '''pf_probe_rad'' (the 3x3 probe half-width, rad) explicitly.']);
    end
    % 0.15x the field half-width: local enough that every combo's fit
    % runs in the same small-cone regime, large enough that the probe
    % chief-ray crossings stay well-conditioned.
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

% ---- Read-surface preflight (Dave 2026-09-08: the wavefront is read at
% the PUPIL by default -- it is the OPD every PSF / diffraction calc uses).
% With no explicit exit_pupil_elt the read lands on nElt-1, which is a
% valid reference ONLY when it is a Return/Reference surface (the
% add_pupil pair, a bench pupil, a collimated-pupil Reference).  A powered
% optic there gives the one-signed 'dome' (path to a curved mirror), and
% the FocalPlane is no alternative: its OPD is the path to each ray's
% landing point and is BLIND to tilt (segment tilt -> segment piston;
% macos/REPORT_ep_dome_review.md).  Refuse up front, once, with the
% remedies, instead of letting the per-field single-DOF call throw
% mid-loop or reset_xp report a soft 'no-effect'.  ERROR, not warn: a
% column read there is not a mis-scaled sensitivity, it is the wrong
% basis vector (Dave's ruling supersedes the warn-and-read path).
if opts.exit_pupil_elt < 0
    ep_rd  = session.num_elt() - 1;
    ep_inf = macos.get_elt_info(ep_rd);
    if ~any(ep_inf.elt_id == [3, 8])
        error(eid_(F, 'noPupil'), ...
            ['%s: no exit-pupil element -- nElt-1 (elt %d) is a %s, not a ' ...
             'Return/Reference surface, so there is no pupil to read the ' ...
             'wavefront at (a powered optic there yields a one-signed ' ...
             'dome; the FocalPlane is blind to tilt).  Remedies: place one ' ...
             'with macos.design.Telescope.add_pupil or the ' ...
             'FP_return/ExitPupil recipe (mmacos/tools/ep_dome_probe/' ...
             'make_pupil_deck.py), or pass ''exit_pupil_elt'' naming a ' ...
             'Reference at a COLLIMATED pupil plane.'], ...
            F.name, ep_rd, ep_inf.type);
    elseif abs(session.get_elt_kr(ep_rd)) >= 1e22
        warning(eid_(F, 'flatPupil'), ...
            ['%s: the read surface nElt-1 (elt %d) is a FLAT %s.  A flat ' ...
             'reference is a valid wavefront read only in COLLIMATED ' ...
             'space (a pupil-plane Reference); in converging space the ' ...
             'read must be a sphere centred on the image (add_pupil).'], ...
            F.name, ep_rd, ep_inf.type);
    end
end

% ---- Per-(configuration, field) loop ------------------------------
% Each iteration sets the source ABSOLUTELY (set_src_fov + reaim +
% modify), then calls the family single with reload_rx=false so the
% field direction survives.
per_field_jac    = cell(n_cfg, n_fields);
per_field_w_nom  = cell(n_cfg, n_fields);
per_field_struct = cell(n_cfg, n_fields);
if opts.compute_los
    per_field_dcdx = cell(n_cfg, n_fields);
end
names = {};  meta = [];
n_empty = 0;
for ic = 1:n_cfg
% Order (PLAN_CONFIGURATIONS 2.1): apply the configuration -> modify()
% once -> field loop (per-field reset_xp derives every field's exit
% pupil FROM THE CONFIGURED GEOMETRY) -> restore -> assert.
% pupil_find round-trip hygiene (2026-08-27 w_nom audit): the guard's
% save_rx -> load_rx round trip compounds across a sequential
% multi-config call -- reload fresh at the top of every configuration
% after the first so the sequential call matches the checkpointed path
% by construction.  Gated on reload_rx (dw_dgrid_multi): pre-installed
% engine state must survive.
if has_cfg && use_pf && ic > 1 && do_reload
    session.load_rx(rx_path);
    apply_ngridpts(session, opts.ngridpts, F.name);
    if ~isempty(opts.src_samp)
        session.set_src_sampling(opts.src_samp);
        session.modify();
    end
    if ~isempty(opts.stop_elt)
        session.stop(int32(opts.stop_elt));
    elseif ~isempty(stop_obj_pos)
        session.stop_obj(stop_obj_pos(1), stop_obj_pos(2), stop_obj_pos(3));
    end
end
if has_cfg
    snap = config_axis('snapshot', session, cfgs(ic).elts);
    config_axis('apply', session, cfgs(ic));
    fprintf('[config %s] applied (%d setter(s))\n', cfgs(ic).name, ...
        numel(cfgs(ic).set));
end
if opts.reset_xp && use_pf && ~pf_fld
    % One field-set-wide sphere per configuration (the cone aperture IS
    % the field set); the field loop then runs with the per-field FEX
    % reset OFF.  Restore the NOMINAL source first: the previous
    % configuration's loop leaves the session at its LAST field, and
    % save_rx would bake that direction into the guard's temp deck
    % (measured: cfgs 2..5 dep_rms 5.5e-3 vs 1.9e-3 clean).  Gated by
    % tPupilFindMethod/test_config_sphere_is_independent_of_predecessors.
    session.set_src_fov('src_pos', nom.src_pos, 'src_dir', nom.src_dir, ...
                        'zSrc', nom.zSrc);
    reaim();   % stop-enforced chief (field_stop_reaim)
    session.modify();
    Fpf = zeros(n_fields, 2);
    for kf = 1:n_fields, Fpf(kf,:) = [fields(kf).dx, fields(kf).dy]; end
    pf_ic = reset_xp_guard('pupil_find', session, Fpf, opts.stop_elt, ...
                           session.num_elt() - 1, opts.pupil_find_opts, xp0);
    reset_ep_moved = true;              % placed by construction
    fprintf(['[%spupil_find] sphere placed: vtx moved %.3g from FEX, ' ...
             'dep_rms %.3g, conv R %.4g\n'], cfg_tag(has_cfg, cfgs, ic, F), ...
            norm(pf_ic.vtx(:) - pf_ic.fex.vpt(:)), pf_ic.dep_rms, ...
            pf_ic.conv_radius);
    pf_out = pf_append_(pf_out, 'config', ic, 0, pf_ic);
end
for k = 1:n_fields
    if opts.reset_xp && pf_fld
        % Per-combo placement: restore the NOMINAL source first (same
        % hygiene as the per-config scope), then fit the 3x3 mini-cone
        % centered on THIS field and place.  The sphere persists as
        % element geometry across the block's poke traces, so the poke's
        % own tilt is retained -- the family convention.
        session.set_src_fov('src_pos', nom.src_pos, 'src_dir', nom.src_dir, ...
                            'zSrc', nom.zSrc);
        reaim();   % stop-enforced chief (field_stop_reaim)
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
        pf_out = pf_append_(pf_out, 'field', ic, k, pf_ic);
    end
    new_dir = field_to_chfraydir(nom.src_dir, fields(k).dx, fields(k).dy, F);
    session.set_src_fov('src_pos', nom.src_pos, 'src_dir', new_dir, ...
                        'zSrc', nom.zSrc);
    reaim();   % stop-enforced chief (field_stop_reaim)
    session.modify();   % flush trace cache so the new dir takes effect
    fprintf('[%sfield %s] ChfRayDir = [%g %g %g]\n', ...
        cfg_tag(has_cfg, cfgs, ic, F), fields(k).name, new_dir);
    if opts.reset_xp && ~use_pf
        % Re-reference this field's exit pupil to its OWN chief ray (FEX
        % writes the reference into elt nElt-1 = wf_elt) so the nominal
        % wavefront is tilt-removed.  It persists as element geometry
        % across the poke traces; a poke's own tilt is retained (the
        % reference is NOT re-fit after poking).  FEX and SXP are merged
        % in the engine -- FEX alone is well-posed for all placements.
        % Shared guard: supervisor-level no-stop error + absorbs the
        % no-pupil-element FAIL -- see private/reset_xp_guard.
        reset_xp_guard('fex', session, opts.fex_axis);
        reset_ep_moved = reset_xp_guard('check', session, xp0, ...
            reset_ep_moved, ep_is_powered);
    end
    sf = F.single(session, rx_path, opts, hoist);
    % An empty OPD at the read surface (no surviving rays at this field)
    % contributes ZERO ROWS to the stacked Jacobian (m2v keeps only
    % non-zero pixels -- 0 is the no-ray mask sentinel).  WARN once per
    % run, never error, never flood (Dave 2026-09-07): long-standing
    % practice completes such runs, and zero OPDs can be physical; the
    % run proceeds exactly as the pre-guard tools did, and further empty
    % blocks are counted into ONE end-of-run note.  (dw_dx_multi's
    % former HARD error is deliberately softened by this ruling.)  The
    % distinct case -- a zero RESPONSE column, e.g. a rotation about
    % psi of a symmetric shape -- was never gated and stays in the
    % matrices.
    if nnz(sf.w_nom_2d) == 0
        n_empty = n_empty + 1;
        if n_empty == 1
            warning(eid_(F, 'emptyOPD'), ...
                ['%sfield %s: OPD at the read surface (elt %d) has no ' ...
                 'non-zero samples -- 0 rays survived there; this block ' ...
                 'contributes 0 rows to the stacked Jacobian.  Usual ' ...
                 'cause if unintended: a tight clip aperture or full ' ...
                 'vignetting at this field.  Further empty blocks in ' ...
                 'this run are counted, not re-warned.'], ...
                cfg_tag(has_cfg, cfgs, ic, F), fields(k).name, sf.wf_elt);
        end
    end
    per_field_jac{ic, k}    = sf.(F.jac);
    per_field_w_nom{ic, k}  = sf.w_nom_2d;
    per_field_struct{ic, k} = sf;
    if opts.compute_los
        per_field_dcdx{ic, k} = sf.dcdx;
    end
    if isempty(names)
        names = sf.channel_names;  meta = F.meta(sf);
    elseif ~isequal(names, sf.channel_names)
        % Column identity is ASSERTED, not assumed: the channel list is
        % built once; a configuration that changed it would silently
        % misalign the stacked Jacobian's columns.
        error(eid_(F, 'channelMismatch'), ...
            ['%sfield %s: channel_names differ from the first block ' ...
             '(%d vs %d channels) -- a configuration must not change ' ...
             'the channel list.'], cfg_tag(has_cfg, cfgs, ic, F), ...
            fields(k).name, numel(sf.channel_names), numel(names));
    end
    col_rms_mean = mean(sqrt(mean(sf.(F.jac).^2, 1)));
    fprintf('[%sfield %s] %s shape [%d %d], mean col-RMS %.3e', ...
        cfg_tag(has_cfg, cfgs, ic, F), fields(k).name, F.jac, ...
        size(sf.(F.jac), 1), size(sf.(F.jac), 2), col_rms_mean);
    if opts.compute_los
        los_rms_mean = mean(sqrt(sum(sf.dcdx.^2, 2)));
        fprintf('  mean LOS-RMS %.3e', los_rms_mean);
    end
    fprintf('\n');
end
% Restore AFTER the channel loop has undone its own pokes, never
% interleaved with it (an element can be BOTH a configuration element
% and a Jacobian channel).  The assertion is the load-bearing part.
if has_cfg
    config_axis('undo', session, cfgs(ic), snap);
    drift = config_axis('assert', session, snap, cfgs(ic).name, F.name);
    fprintf(['[config %s] restored + verified '  ...
             '(worst pose drift %.1f%% of tolerance)\n'], ...
        cfgs(ic).name, 100 * drift);
end
end

if n_empty > 0
    fprintf(['[note] %d of %d (configuration, field) block(s) had empty ' ...
             'OPD support and contribute 0 rows (first occurrence warned ' ...
             'above)\n'], n_empty, n_cfg*n_fields);
end

% Restore source back to nominal.
session.set_src_fov('src_pos', nom.src_pos, 'src_dir', nom.src_dir, ...
                    'zSrc', nom.zSrc);
reaim();   % stop-enforced chief (field_stop_reaim)
session.modify();

% Restore the prescription's exit-pupil reference (undo the per-field
% FEX writes to elt nElt-1) so the session is left as loaded.
if opts.reset_xp
    macos.set_xp(xp0.vpt, xp0.psi, xp0.rad);
    session.modify();
end
% NO-PUPIL GUARD: reset_xp requested but FEX never moved the EP element
% -- the harvest is really FROZEN-EP; stamp the truth ('no-effect').
reset_xp_stamp = reset_xp_guard('finalize', opts.reset_xp, ...
    reset_ep_moved, session.num_elt() - 1);

% ---- Tile OPDall + scatter the stacked Jacobian -------------------
N = size(per_field_w_nom{1, 1}, 1);
% Each configuration gets its OWN field canvas; macos.config_canvas
% places those canvases and builds a CONFIGURATION-MAJOR index, so w for
% one configuration stacks its FIELDS and w for the run stacks the
% CONFIGURATIONS.  (m2v on the assembled canvas directly would
% interleave the blocks -- it walks column-major.)
canv = cell(1, n_cfg);
for ic = 1:n_cfg
    Cc = zeros(tile_rows * N, tile_cols * N);
    for k = 1:n_fields
        assert(size(per_field_w_nom{ic, k}, 1) == N, ...
            eid_(F, 'gridSize'), ...
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
Nz = size(per_field_jac{1, 1}, 2);
fprintf('[stack] OPDall [%d %d]; non-zero pixels = %d\n', ...
    size(OPDall, 1), size(OPDall, 2), Nw);

jac_all = zeros(Nw, Nz);
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
    % Field-local m2v of this tile maps global rows back to the
    % per-field Jacobian rows.
    [~, field_indx] = macos.m2v(per_field_w_nom{ic, k});
    field_i = field_indx.i(:);
    field_j = field_indx.j(:);
    flat_local = (j_local - 1) * N + i_local;
    flat_field = (field_j  - 1) * N + field_i;
    [tf, loc] = ismember(flat_local, flat_field);
    if ~all(tf)
        error(eid_(F, 'scatter'), ...
            '%sfield %s: indxall references pixels outside per-field mask', ...
            cfg_tag(has_cfg, cfgs, ic, F), fields(k).name);
    end
    global_rows = row0 + find(in_tile);
    jac_all(global_rows, :) = per_field_jac{ic, k}(loc, :);
    fprintf('[stack] %sfield %s: scattered %d rows into %s\n', ...
        cfg_tag(has_cfg, cfgs, ic, F), fields(k).name, ...
        numel(global_rows), F.all_names{1});
end
    row0 = row0 + numel(ic_i);
end

fprintf('[stack] %s shape [%d %d]; |%s| max = %.3e\n', ...
    F.all_names{1}, size(jac_all, 1), size(jac_all, 2), ...
    F.all_names{1}, max(abs(jac_all(:))));

% Center-tile sanity check: the (0,0) field's rows in the stacked
% Jacobian must exactly match its per-field block.
ctr_idx = find_center_field_index(fields);
if ~isempty(ctr_idx)
    row0 = 0;
    for ic = 1:n_cfg
        [~, ixc] = macos.m2v(canv{ic});
        tr = fields(ctr_idx).tile_row;
        tc = fields(ctr_idx).tile_col;
        in_ctr = (ixc.i(:) > tr*N) & (ixc.i(:) <= (tr+1)*N) ...
               & (ixc.j(:) > tc*N) & (ixc.j(:) <= (tc+1)*N);
        jac_ctr = jac_all(row0 + find(in_ctr), :);
        jac_C   = per_field_jac{ic, ctr_idx};
        d_ctr = abs(jac_ctr(:) - jac_C(:));
        % An EMPTY centre block (0 rows: the emptyOPD case, warned above)
        % is consistent with itself; max([]) is [] and would trip the
        % scalar-logical assert that the warn-and-complete ruling forbids.
        if isempty(d_ctr), max_diff = 0; else, max_diff = max(d_ctr); end
        fprintf('[check] %s@center-tile vs per-field[center]: ', F.all_names{1});
        fprintf('max|diff| = %.3e ([%d %d])\n', ...
            max_diff, size(jac_ctr, 1), size(jac_ctr, 2));
        assert(max_diff == 0, ...
            'scatter bug: %s@center-tile differs from per-field[center]', ...
            F.all_names{1});
        row0 = row0 + numel(ixc.i);
    end
else
    fprintf('[check] no (0,0)-offset field -- skipping center-tile check\n');
end

% ---- Pack output struct -------------------------------------------
out = struct();
for a = 1:numel(F.all_names)
    out.(F.all_names{a}) = jac_all;
end
out.w0_stacked           = w0_stacked;
out.indxall              = indxall;
out.OPDall               = OPDall;
out.channel_names        = names;
mf = fieldnames(meta);
for a = 1:numel(mf)
    out.(mf{a}) = meta.(mf{a});
end
out.field_table          = arrayfun( ...
    @(s) [s.dx, s.dy, s.tile_row, s.tile_col], fields, ...
    'UniformOutput', false);
out.field_table          = vertcat(out.field_table{:});
out.field_names          = {fields.name}.';
out.chfraydir_nom        = nom.src_dir(:);
if has_cfg
    out.config_table       = cfgs(:);
    out.config_names       = {cfgs.name}.';
    out.(F.per_name)       = per_field_jac;         % Nc x Nf
    out.per_field_w_nom_2d = per_field_w_nom;       % Nc x Nf
else
    % preserved surface: without 'configs' the cells keep their Nf x 1
    % shape and no configuration fields are added
    out.(F.per_name)       = per_field_jac(1, :).';
    out.per_field_w_nom_2d = per_field_w_nom(1, :).';
end
out.rx_path              = rx_path;
out.delta                = opts.delta;
out.method               = opts.method;
out.wf_elt               = per_field_struct{1, 1}.wf_elt;
out = F.extras(out, opts, per_field_struct{1, 1});
out = apply_opd_convention(out, opts.orient, opts.sign);
out.reset_xp             = reset_xp_stamp;   % true | false | 'no-effect'
out.reset_xp_method      = opts.reset_xp_method;
out.fex_axis             = opts.fex_axis;
if use_pf, out.pf_scope = opts.pf_scope; end
if ~isempty(pf_out), out.pupil_find = pf_out; end   % per-config metrics

% Add per-field LOS if SPOT was computed
if opts.compute_los
    if F.los_reshape && ~has_cfg
        out.dcdx_per_field = per_field_dcdx(1, :).';
    else
        out.dcdx_per_field = per_field_dcdx;
    end
    if isempty(opts.spot_elt)
        out.spot_elt = session.num_elt();  % Default focal plane
    else
        out.spot_elt = opts.spot_elt;
    end
end
end


% =====================================================================
function v = getf_(s, name, dflt)
if isfield(s, name), v = s.(name); else, v = dflt; end
end

function id = eid_(F, tail)
id = ['macos:' F.name ':' tail];
end

function t = cfg_tag(has_cfg, cfgs, ic, F)
% "config <name> / " prefix for progress lines; empty without 'configs'
% or for the families whose historic logs did not carry it.
if has_cfg && F.cfg_tag_fieldlog
    t = sprintf('config %s / ', cfgs(ic).name);
else
    t = '';
end
end

function pf_out = pf_append_(pf_out, scope, ic, k, pf_ic)
m0 = struct('scope', scope, 'config', ic, 'field', k, ...
            'vtx', pf_ic.vtx_written(:).', 'bundle_vtx', pf_ic.vtx(:).', ...
            'rad', pf_ic.rad, ...
            'fex_vpt', pf_ic.fex.vpt(:).', ...
            'vtx_minus_fex', norm(pf_ic.vtx(:) - pf_ic.fex.vpt(:)), ...
            'dep_rms', pf_ic.dep_rms, 'conv_radius', pf_ic.conv_radius);
if isempty(pf_out), pf_out = m0; else, pf_out(end+1) = m0; end
end

function warn_reset_xp_method_deprecated_(F)
% One-time-per-session deprecation notice for reset_xp_method='sxp'.
persistent warned
if isempty(warned)
    warning(eid_(F, 'resetXpMethodDeprecated'), ...
        ['reset_xp_method is deprecated: FEX and SXP are merged in the ' ...
         'engine, so FEX is used for the per-field exit-pupil reset ' ...
         'regardless of this option.  ''sxp'' is accepted as an alias.']);
    warned = true;
end
end

% ---- field-set helpers (formerly duplicated in all four fronts) -----
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
% Uniform NxM grid covering the +-field rectangle.
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

function new_dir = field_to_chfraydir(dir_nom, dx_rad, dy_rad, F)
% Direction-cosine offset on top of the nominal ChfRayDir, renormalised.
v = dir_nom(:) + [dx_rad; dy_rad; 0];
n = norm(v);
if n == 0
    error(eid_(F, 'zerodir'), ...
        'zero-magnitude direction after field offset');
end
new_dir = v / n;
end

function [nx, ny] = parse_grid_spec(spec, F)
toks = regexp(lower(spec), 'x', 'split');
if numel(toks) ~= 2
    error(eid_(F, 'grid'), ...
        '''grid'' must be ''NxM'' (e.g. ''3x3''); got %s', spec);
end
nx = str2double(toks{1});  ny = str2double(toks{2});
if isnan(nx) || isnan(ny) || nx < 1 || ny < 1
    error(eid_(F, 'grid'), ...
        '''grid'' must be ''NxM'' with positive integers; got %s', spec);
end
end

function fields = load_field_file(fname, F)
% Free-form list: lines of 'name dx_rad dy_rad tile_row tile_col'.
fid = fopen(fname, 'r');
if fid < 0
    error(eid_(F, 'fields'), 'cannot open fields file: %s', fname);
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
        error(eid_(F, 'fields'), ...
            'fields-file row needs 5 columns: %s', s);
    end
    fields(end+1) = field_entry(toks{1}, ...
        str2double(toks{2}), str2double(toks{3}), ...
        str2double(toks{4}), str2double(toks{5})); %#ok<AGROW>
end
end
