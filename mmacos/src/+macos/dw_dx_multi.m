function out = dw_dx_multi(session, rx_path, opts)
%MACOS.DW_DX_MULTI  Multi-field dw/dx rigid-body supervisor.
%   Mirror of macos.dw_dz_zernike_multi but for the rigid-body
%   Jacobian.  Loads Rx once, builds source / per-element channels
%   once, then loops field points -- each iteration absolutely sets
%   ChfRayDir via set_src_fov + modify(), runs dw_dx with
%   reload_rx=false, and stacks per-field dwdx into the canonical
%   state-vector form  wall = dwdxall * x + w0_stacked.
%
%   REQUIRED:
%     'field_x_rad', 'field_y_rad'
%
%   FIELD SET (one of):
%     default 5-field (C + 4 corners)
%     'grid' 'NxM'
%     'fields' FILE
%
%   FORWARDED TO dw_dx:  dofs, elts, fp_mode, ep_elt, include_source,
%     src_stop_mode, src_stop_pos, src_stop_elt, include_non_optics,
%     stop_elt, stop_obj_pos, groups, groups_auto, group_coords,
%     group_fp_mode, group_stop_mode, group_stop_pos, rot_output,
%     delta, method, exit_pupil_elt, verbose.
%
%   ELEMENT GROUPS ('groups' / 'groups_auto' + the four group_* knobs)
%   declare RIGID-BODY groups -- sets of elements perturbed as one unit
%   via the engine's GPERTURB -- exactly as in macos.dw_dx.  Their
%   channels are APPENDED AFTER the per-element block in every field's
%   block, so the stacked column order is
%       [source] [per-element] [group]
%   and every field block carries the SAME group columns in the SAME
%   order.  That is not an accident of the loop: the group map is
%   materialized ONCE here (PARSE-ONCE HOIST -- 'groups_auto' reads the
%   Rx FILE, and forwarding it per field would re-parse it per field),
%   merged auto-then-explicit with dw_dx's semantics (an explicit entry
%   overrides an auto one of the same name), and handed down as
%   'groups' with groups_auto=false.  So every field sees the identical
%   map object; the group channel COUNT cannot drift between fields
%   ('group_fp_mode' selects each channel's post-perturbation follow-up,
%   never how many channels exist).  The column-identity assertion below
%   gates it anyway.
%
%   Group channels carry no single element id: out.iElt is 0 for them
%   (as it is for source channels) and out.kind is 'Group' -- section
%   them on kind, not on iElt.  Units: group and per-element columns
%   share one convention -- OPD-per-metre for translations, OPD-per-rad
%   for rotations -- so one numeric 'delta' is one physical poke for
%   either.
%
%   'delta' can be (1,1) for uniform step or (1,6) for per-DOF steps
%     [Rx Ry Rz Tx Ty Tz]. Rotations in rad. Translation units set by
%     'delta_units' ('si' metres, default | 'base' BaseUnits). Default 1e-8.
%
%   'ngridpts' (default [] = keep the .in value) overrides the ray-grid
%   sampling once, right after the Rx load; it persists across the
%   per-field calls (they run reload_rx=false).  Clamped by the engine
%   to [3, model-size limit] (warns).
%
%   'reset_xp'  (default true) re-find the exit pupil for EACH field
%               before differencing, so the nominal wavefront is
%               referenced to that field's own chief ray and the gross
%               field TILT is removed.  A poke's OWN tilt is retained --
%               the reference is fixed per field, not re-fit after each
%               poke.  With a frozen EP (false) the field tilt is
%               common-mode between w_nom and every poked w, so it
%               cancels in the FD columns; what reset_xp changes is the
%               first-order residual d(frame term)/dx -- negligible at
%               arcminute fields, percent-level on tilt-coupled DOFs at
%               wide fields.  Matches dw_dz_zernike_multi / dw_dsurf_multi
%               / dw_dgrid_multi (family alignment).  Requires a STOP set
%               and > 3 elements.  Set false to keep the prescription's
%               elt nElt-1 reference unchanged (frozen EP).
%               The re-find uses FEX (macos.fex, chief-ray centred);
%               FEX and SXP are merged in the engine, so FEX alone is
%               the well-posed re-reference for all placements.
%               Composes with fp_mode='track': the per-field EP is
%               written into elt nElt-1 BEFORE the FocalPlaneChannel
%               builds its columns, so 'track' saves/restores the
%               post-reset EP pose.
%               Restore scope: the pre-loop EP is snapshotted/restored via
%               get_xp/set_xp -- vpt/psi/rad (VptElt/PsiElt/KrElt at
%               nElt-1) only.  FEX-written auxiliary fields on the EP
%               element (RptElt, zElt, fElt, eElt, KcElt) are left as
%               re-derived; callers who hand-author those own re-asserting
%               them.
%
%   'reset_xp_method'  'fex' (default) | 'sxp' (alias of fex; the engine
%               merged them) | 'pupil_find'.  'pupil_find' places the
%               cone-convergence best-fit exit-pupil sphere
%               (design/src/pupil_find) ONCE per configuration -- the
%               cone aperture IS the field set (>=3 fields) -- and runs
%               the field loop with the per-field FEX reset OFF: a
%               frozen, field-set-wide best-fit reference, an upgrade of
%               the frozen-EP mode rather than a per-field re-reference
%               (the per-field tilt term behaves as reset_xp=false).
%               REQUIRES 'stop_elt' (pupil_find sets the engine stop at
%               that element).  'pupil_find_opts' forwards extra
%               name-values to the finder (anchor/nodes/...).  Per-config
%               fit metrics return in out.pupil_find.
%   'configs'   (default [] = today's single-block call, byte-identical)
%               a 1xNc struct array of CONFIGURATIONS -- named sets of
%               element setting overrides ("zoom positions"; in our
%               systems more often a COMPENSATION state, e.g. a steering
%               mirror at a pupil fold re-pointed to cancel pointing
%               drift).  Each entry is
%                   .name  char
%                   .set   cell array of setter invocations, each itself
%                          a cell {fname, elt, args...} dispatched
%                          against the Session
%               e.g.  struct('name','zUR', 'set', {{ ...
%                       {'perturb', 25, 'rotation', [t;t;0], ...
%                        'frame','local'} }})
%               The Jacobian is then evaluated per (configuration, field)
%               block and the blocks stack as extra ROWS -- a
%               configuration adds observations of the SAME state vector
%               x, exactly as a field point does, so every downstream
%               consumer (run_compare, the MET optimiser, the simulator)
%               keeps working unchanged.
%               Row COUNT: a configuration that changes ray survival
%               (a tilt can vignette a field) contributes a different
%               number of rows, so the stack is sum-over-configurations,
%               not exactly Nc*Nw.  Slice a block with
%               out.indxall.config == c -- the blocks are contiguous.
%               v1 accepts ONLY the pose setters perturb / set_elt_vpt /
%               set_elt_psi / set_elt_rpt / set_elt_csys; anything else
%               is a loud validation error BEFORE anything is applied.
%               The runner owns the modify()-after-setters rule, and
%               snapshots / restores / ASSERTS the touched elements
%               around each block, so a configuration that fails to
%               restore is a hard error rather than silent contamination
%               of the next block.  See private/config_axis.m and
%               design/PLAN_CONFIGURATIONS.md.
%
%   OUTPUT STRUCT FIELDS:
%     dwdxall            Nw x Nz canonical state-vector Jacobian
%     w0_stacked         Nw x 1 stacked nominal OPDs (m2v of OPDall)
%     indxall            i, j, size struct
%     OPDall             full tiled OPD canvas
%     channel_names      Nz x 1 cell
%     iElt / kind / dof_idx  Nz x 1 per-channel bookkeeping (iElt = 0
%                        for source AND group channels -- section on
%                        kind)
%     field_table        Nfields x 4
%     field_names        Nfields x 1 cell
%     chfraydir_nom      3 x 1
%     per_field_dwdx     Nfields x 1 cell of single-field blocks
%                        (Nconfigs x Nfields with 'configs')
%     per_field_w_nom_2d Nfields x 1 cell of single-field nominal OPDs
%                        (Nconfigs x Nfields with 'configs')
%     config_table       (with 'configs' only) Nc x 1 struct: name +
%                        the setter list, verbatim
%     indxall.config     (with 'configs' only) per-row configuration index
%     rx_path / delta / method / wf_elt / rot_output / cbm

arguments
    session
    rx_path                  (1,:) char {mustBeNonempty}
    opts.field_x_rad         (1,1) double = NaN
    opts.field_y_rad         (1,1) double = NaN
    opts.fields              (1,:) char = ''
    opts.grid                (1,:) char = ''
    opts.dofs                (:,1) double = (0:5).'
    opts.elts                (:,1) double = []
    opts.fp_mode             (1,:) char {mustBeMember( ...
        opts.fp_mode, {'track','srs','sxp','none'})} = 'track'
    opts.ep_elt              (1,1) double {mustBeInteger} = -1
    opts.include_source      (1,1) logical = false
    opts.src_stop_mode       (1,:) char {mustBeMember( ...
        opts.src_stop_mode, {'obj','elt','none'})} = 'obj'
    opts.src_stop_pos        (1,3) double = [0 0 0]
    opts.src_stop_elt        (1,1) double {mustBeInteger} = 0
    opts.include_non_optics  (1,1) logical = false
    opts.stop_elt            double = []
    opts.stop_obj_pos        double = []
    opts.groups              = []   % containers.Map name -> col vec,
                                    % or [] = no extras
    opts.groups_auto         (1,1) logical = false
    opts.group_coords        (1,:) char {mustBeMember( ...
                                opts.group_coords, ...
                                {'global','local'})} = 'global'
    opts.group_fp_mode       (1,:) char {mustBeMember( ...
                                opts.group_fp_mode, ...
                                {'auto','none','sxp','srs'})} = 'auto'
    opts.group_stop_mode     (1,:) char {mustBeMember( ...
                                opts.group_stop_mode, ...
                                {'obj','elt','none'})} = 'obj'
    opts.group_stop_pos      (1,3) double = [0 0 0]
    opts.rot_output          (1,:) char {mustBeMember( ...
        opts.rot_output, {'natural','base-per-rad'})} = 'natural'
    opts.delta               (:,:) double {mustBeDeltaSize} = 1e-8
    opts.delta_units         (1,:) char {mustBeMember(opts.delta_units, ...
                                {'si','base'})} = 'si'
    opts.method              (1,:) char {mustBeMember(opts.method, ...
                                {'central','forward'})} = 'central'
    opts.exit_pupil_elt      (1,1) double {mustBeInteger} = -1
    opts.reset_xp            (1,1) logical = true
    opts.reset_xp_method     (1,:) char {mustBeMember( ...
        opts.reset_xp_method, {'fex','sxp','pupil_find'})} = 'fex'
    opts.pupil_find_opts     cell = {}
    opts.configs                          = []
    opts.verbose             (1,1) logical = false
    opts.ngridpts            double {mustBeScalarOrEmpty} = []
    opts.src_samp            double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.compute_los         (1,1) logical = false
    opts.spot_elt            double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.orient (1,:) char {mustBeMember(opts.orient, {'raw','xy'})} = 'raw'   % OPD array orientation (doc/opd_conventions.md)
    opts.sign   (1,:) char {mustBeMember(opts.sign, {'opl','wavefront'})} = 'opl' % OPD sign convention
end

if isnan(opts.field_x_rad) || isnan(opts.field_y_rad)
    error('macos:dw_dx_multi:fov', ...
        'field_x_rad and field_y_rad are required');
end

% ---- Field set ----------------------------------------------------
if ~isempty(opts.fields)
    fields = load_field_file(opts.fields);
elseif ~isempty(opts.grid)
    [nx, ny] = parse_grid_spec(opts.grid);
    fields = make_grid_field_set(nx, ny, opts.field_x_rad, ...
                                  opts.field_y_rad);
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

% ---- Load + snapshot nominal --------------------------------------
session.load_rx(rx_path);
apply_ngridpts(session, opts.ngridpts, 'dw_dx_multi');

% Apply source sampling if specified
if ~isempty(opts.src_samp)
    session.set_src_sampling(opts.src_samp);
    session.modify();  % Flush cache so the new sampling takes effect
end

% Apply stop here so it survives across per-field calls (dw_dx with
% reload_rx=false won't touch the stop state).
if ~isempty(opts.stop_elt) && ~isempty(opts.stop_obj_pos)
    error('macos:dw_dx_multi:stop', ...
        'stop_elt and stop_obj_pos are mutually exclusive');
end
if ~isempty(opts.stop_elt)
    session.stop(int32(opts.stop_elt));
elseif ~isempty(opts.stop_obj_pos)
    session.stop_obj(opts.stop_obj_pos(1), opts.stop_obj_pos(2), ...
                      opts.stop_obj_pos(3));
end

% ---- Element groups: PARSE-ONCE HOIST ------------------------------
% 'groups_auto' reads the Rx FILE, so forwarding it to every per-field
% dw_dx call would re-parse the same file per field.  Materialize the
% merged map ONCE here -- auto first, explicit entries overriding on a
% name collision, which is dw_dx's own merge order -- and hand it down
% as 'groups' with groups_auto=false.  Same map object for every field,
% so the group channel COUNT cannot drift between blocks; the
% channel-identity assertion in the loop gates that regardless.
grp_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
if opts.groups_auto
    grp_map = macos.channels.parse_rx_groups(rx_path);
end
if isa(opts.groups, 'containers.Map')
    gk = keys(opts.groups);
    for kk = 1:numel(gk)
        grp_map(gk{kk}) = opts.groups(gk{kk});
    end
elseif ~isempty(opts.groups)
    error('macos:dw_dx_multi:groups', ...
        'groups must be a containers.Map (name -> member id column) or []');
end
if grp_map.Count > 0
    gk = keys(grp_map);
    fprintf('[setup] %d element group(s):\n', grp_map.Count);
    for kk = 1:numel(gk)
        fprintf('  group %-10s: elts %s\n', gk{kk}, ...
            mat2str(reshape(double(grp_map(gk{kk})), 1, [])));
    end
else
    % preserved surface: with no groups the per-field call gets [],
    % exactly the argument it got before the group opts existed.
    grp_map = [];
end

% ---- Configuration axis -------------------------------------------
% Validated AFTER the load (so element ids can be range-checked) and
% BEFORE anything is applied.  Absent/empty => n_cfg == 1 and every line
% below degenerates to the pre-configs single-block path.
ep_elt_chk = [];
if opts.reset_xp, ep_elt_chk = session.num_elt() - 1; end
cfgs = config_axis('validate', opts.configs, session.num_elt(), ...
                    'dw_dx_multi', ep_elt_chk);
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
% so the per-field FEX resets can be undone before returning.
% reset_xp acts by writing the pupil reference into nElt-1 -- but the
% engine FEX only writes when nElt-1 is a Return/Reference surface;
% on any other type it silently declines (xp_fnd still returns PASS),
% so reset_xp is a no-op there.  Track whether ANY field's FEX actually
% moved the EP element, and refuse to let it CLOBBER a powered optic.
reset_ep_moved = false;   % did any field's fex() change nElt-1 geometry?
use_pf = strcmp(opts.reset_xp_method, 'pupil_find');
pf_out = struct([]);
if opts.reset_xp
    xp0 = macos.get_xp();
    ep_is_powered = reset_xp_guard('is_powered', session);
end

% ---- Per-(configuration, field) loop ------------------------------
per_field_dwdx   = cell(n_cfg, n_fields);
per_field_w_nom  = cell(n_cfg, n_fields);
per_field_struct = cell(n_cfg, n_fields);
if opts.compute_los
    per_field_dcdx = cell(n_cfg, n_fields);
end
names = {};
iElt_out = [];
kind_out = {};
dof_out  = [];
for ic = 1:n_cfg
% Order (PLAN_CONFIGURATIONS §2.1): apply the configuration -> modify()
% once -> run the field loop, whose per-field reset_xp then derives every
% field's exit pupil FROM THE CONFIGURED GEOMETRY (a pupil-fold tilt
% moves the EP; that is physics, not drift) -> restore -> assert.
if has_cfg
    snap = config_axis('snapshot', session, cfgs(ic).elts);
    config_axis('apply', session, cfgs(ic));
    fprintf('[config %s] applied (%d setter(s))\n', cfgs(ic).name, ...
        numel(cfgs(ic).set));
end
if opts.reset_xp && use_pf
    % reset_xp_method='pupil_find': place the cone-convergence best-fit
    % sphere ONCE for this configuration (field-set-wide fit -- the cone
    % aperture IS the field set), then run the field loop with the
    % per-field FEX reset OFF.  A frozen, best-fit exit pupil: the
    % per-field tilt reference behaves as reset_xp=false always has.
    Fpf = zeros(n_fields, 2);
    for kf = 1:n_fields, Fpf(kf,:) = [fields(kf).dx, fields(kf).dy]; end
    pf_ic = reset_xp_guard('pupil_find', session, Fpf, opts.stop_elt, ...
                           session.num_elt() - 1, opts.pupil_find_opts);
    reset_ep_moved = true;              % placed by construction
    fprintf(['[%spupil_find] sphere placed: vtx moved %.3g from FEX, ' ...
             'dep_rms %.3g, conv R %.4g\n'], cfg_tag(has_cfg, cfgs, ic), ...
            norm(pf_ic.vtx(:) - pf_ic.fex.vpt(:)), pf_ic.dep_rms, ...
            pf_ic.conv_radius);
    m0 = struct('vtx', pf_ic.vtx(:).', 'rad', pf_ic.rad, ...
                'fex_vpt', pf_ic.fex.vpt(:).', ...
                'vtx_minus_fex', norm(pf_ic.vtx(:) - pf_ic.fex.vpt(:)), ...
                'dep_rms', pf_ic.dep_rms, 'conv_radius', pf_ic.conv_radius);
    if isempty(pf_out), pf_out = m0; else, pf_out(end+1) = m0; end %#ok<AGROW>
end
for k = 1:n_fields
    new_dir = field_to_chfraydir(nom.src_dir, fields(k).dx, fields(k).dy);
    session.set_src_fov('src_pos', nom.src_pos, 'src_dir', new_dir, ...
                        'zSrc', nom.zSrc);
    session.modify();
    fprintf('[%sfield %s] ChfRayDir = [%g %g %g]\n', ...
        cfg_tag(has_cfg, cfgs, ic), fields(k).name, new_dir);
    if opts.reset_xp && ~use_pf
        % Re-reference this field's exit pupil to its OWN chief ray: FEX
        % writes the reference sphere into elt nElt-1 (= wf_elt), so the
        % nominal (unpoked) wavefront there is tilt-removed.  That
        % reference is element geometry, so it persists across the poke
        % traces below and the rigid-body pokes act on elts 1..nElt-2,
        % never touching elt nElt-1.  Net: the FIELD tilt is removed from
        % the nominal, but a POKE's own tilt is retained.  Writing the EP
        % here -- BEFORE dw_dx builds its channels -- lets a
        % FocalPlaneChannel ('track') save/restore the POST-reset EP pose.
        % FEX and SXP are merged in the engine, so FEX alone is well-posed
        % for all exit-pupil placements.  The shared guard raises the
        % supervisor-level no-stop error and absorbs the no-pupil-element
        % FAIL -- see private/reset_xp_guard.
        reset_xp_guard('fex', session);
        % Did FEX actually write? (engine writes nElt-1 only for a
        % Return/Reference surface; elsewhere it declines.)  The
        % shared guard also ERRORS if a write landed on a powered optic.
        reset_ep_moved = reset_xp_guard('check', session, xp0, ...
            reset_ep_moved, ep_is_powered);
    end
    sf = macos.dw_dx(session, rx_path, ...
        'dofs', opts.dofs, ...
        'elts', opts.elts, ...
        'fp_mode', opts.fp_mode, ...
        'ep_elt', opts.ep_elt, ...
        'include_source', opts.include_source, ...
        'src_stop_mode', opts.src_stop_mode, ...
        'src_stop_pos', opts.src_stop_pos, ...
        'src_stop_elt', opts.src_stop_elt, ...
        'include_non_optics', opts.include_non_optics, ...
        'groups', grp_map, ...
        'groups_auto', false, ...
        'group_coords', opts.group_coords, ...
        'group_fp_mode', opts.group_fp_mode, ...
        'group_stop_mode', opts.group_stop_mode, ...
        'group_stop_pos', opts.group_stop_pos, ...
        'rot_output', opts.rot_output, ...
        'delta', opts.delta, ...
        'delta_units', opts.delta_units, ...
        'method', opts.method, ...
        'exit_pupil_elt', opts.exit_pupil_elt, ...
        'verbose', opts.verbose, ...
        'reload_rx', false, ...
        'compute_los', opts.compute_los, ...
        'spot_elt', opts.spot_elt);
    % Guard: an empty OPD at the read surface (no surviving rays -- e.g.
    % the beam footprint overflows a tight clip aperture at that field, or
    % the trace is fully lost) yields a zero-row per-field block that
    % otherwise scatters silently to nothing and later trips the
    % center-tile check with an opaque scalar-logical error.  Fail loudly
    % and actionably here instead.
    if nnz(sf.w_nom_2d) == 0
        error('macos:dw_dx_multi:emptyOPD', ...
            ['field %s: OPD at the read surface (elt %d) has no non-zero ' ...
             'samples -- 0 rays survived there.  Likely the beam footprint ' ...
             'overflows a tight clip aperture at this field (strip the ' ...
             'ApType= clips or widen the field/grid), or the trace is ' ...
             'fully vignetted.'], fields(k).name, sf.wf_elt);
    end
    per_field_dwdx{ic, k}   = sf.dwdx;
    per_field_w_nom{ic, k}  = sf.w_nom_2d;
    per_field_struct{ic, k} = sf;
    if opts.compute_los
        per_field_dcdx{ic, k} = sf.dcdx;
    end
    if isempty(names)
        names = sf.channel_names; iElt_out = sf.iElt;
        kind_out = sf.kind; dof_out = sf.dof_idx;
    elseif ~isequal(names, sf.channel_names)
        % Column identity is ASSERTED, not assumed: the channel list is
        % built once, before the configuration loop, so a configuration
        % that changed the element count would silently MISALIGN the
        % stacked Jacobian's columns.
        error('macos:dw_dx_multi:channelMismatch', ...
            ['%sfield %s: channel_names differ from the first block ' ...
             '(%d vs %d channels) -- a configuration must not change ' ...
             'the channel list.'], cfg_tag(has_cfg, cfgs, ic), ...
            fields(k).name, numel(sf.channel_names), numel(names));
    end
    col_rms_mean = mean(sqrt(mean(sf.dwdx.^2, 1)));
    fprintf('[%sfield %s] dwdx shape [%d %d], mean col-RMS %.3e', ...
        cfg_tag(has_cfg, cfgs, ic), fields(k).name, ...
        size(sf.dwdx, 1), size(sf.dwdx, 2), col_rms_mean);
    if opts.compute_los
        los_rms_mean = mean(sqrt(sum(sf.dcdx.^2, 2)));
        fprintf('  mean LOS-RMS %.3e', los_rms_mean);
    end
    fprintf('\n');
end
% Restore AFTER the channel loop has finished undoing its own pokes
% (dw_dx restores every channel before returning), never interleaved with
% it -- element 25 of the zoom fixture is BOTH the configuration element
% and a Jacobian channel.  The assertion is the load-bearing part.
if has_cfg
    config_axis('undo', session, cfgs(ic), snap);
    drift = config_axis('assert', session, snap, cfgs(ic).name, 'dw_dx_multi');
    fprintf(['[config %s] restored + verified '  ...
             '(worst pose drift %.1f%% of tolerance)\n'], ...
        cfgs(ic).name, 100 * drift);
end
end

session.set_src_fov('src_pos', nom.src_pos, 'src_dir', nom.src_dir, ...
                    'zSrc', nom.zSrc);
session.modify();

% Restore the prescription's exit-pupil reference (undo the per-field FEX
% writes to elt nElt-1) so the session is left as loaded.
if opts.reset_xp
    macos.set_xp(xp0.vpt, xp0.psi, xp0.rad);
    session.modify();
end

% NO-PUPIL GUARD: reset_xp was requested but FEX never moved the EP
% element at any field -- this Rx has no exit-pupil element at nElt-1, so
% the engine declined to write and the harvest is really FROZEN-EP.  Warn
% once and stamp the truth so downstream convention asserts (run_compare)
% see 'no-effect', not a false 'true'.
reset_xp_stamp = reset_xp_guard('finalize', opts.reset_xp, ...
    reset_ep_moved, session.num_elt() - 1);

% ---- Tile OPDall + scatter dwdxall --------------------------------
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
            'macos:dw_dx_multi:gridSize', ...
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
Nz = size(per_field_dwdx{1, 1}, 2);
fprintf('[stack] OPDall [%d %d]; non-zero pixels = %d\n', ...
    size(OPDall, 1), size(OPDall, 2), Nw);

dwdxall = zeros(Nw, Nz);
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
    [~, field_indx] = macos.m2v(per_field_w_nom{ic, k});
    field_i = field_indx.i(:);
    field_j = field_indx.j(:);
    flat_local = (j_local - 1) * N + i_local;
    flat_field = (field_j  - 1) * N + field_i;
    [tf, loc] = ismember(flat_local, flat_field);
    if ~all(tf)
        error('macos:dw_dx_multi:scatter', ...
            '%sfield %s: indxall references pixels outside per-field mask', ...
            cfg_tag(has_cfg, cfgs, ic), fields(k).name);
    end
    global_rows = row0 + find(in_tile);
    dwdxall(global_rows, :) = per_field_dwdx{ic, k}(loc, :);
    fprintf('[stack] %sfield %s: scattered %d rows into dwdxall\n', ...
        cfg_tag(has_cfg, cfgs, ic), fields(k).name, numel(global_rows));
end
    row0 = row0 + numel(ic_i);
end

fprintf('[stack] dwdxall shape [%d %d]; |dwdxall| max = %.3e\n', ...
    size(dwdxall, 1), size(dwdxall, 2), max(abs(dwdxall(:))));

% Center-tile sanity check.
ctr_idx = find_center_field_index(fields);
if ~isempty(ctr_idx)
    row0 = 0;
    for ic = 1:n_cfg
        [~, ixc] = macos.m2v(canv{ic});
        tr = fields(ctr_idx).tile_row;
        tc = fields(ctr_idx).tile_col;
        in_ctr = (ixc.i(:) > tr*N) & (ixc.i(:) <= (tr+1)*N) ...
               & (ixc.j(:) > tc*N) & (ixc.j(:) <= (tc+1)*N);
        max_diff = max(abs(dwdxall(row0 + find(in_ctr), :) - per_field_dwdx{ic, ctr_idx}), ...
                        [], 'all');
        fprintf('[check] %sdwdxall@center-tile vs per_field_dwdx[center]: ', ...
            cfg_tag(has_cfg, cfgs, ic));
        fprintf('max|diff| = %.3e\n', max_diff);
        assert(max_diff == 0, ...
            'scatter bug: dwdxall@center-tile differs from per_field_dwdx[center]');
        row0 = row0 + numel(ixc.i);
    end
end

out = struct();
out.dwdxall              = dwdxall;
out.w0_stacked           = w0_stacked;
out.indxall              = indxall;
out.OPDall               = OPDall;
out.channel_names        = names;
out.iElt                 = iElt_out;
out.kind                 = kind_out;   % 'Source'|'RigidBody'|
                                       % 'FocalPlane'|'Group'
out.dof_idx              = dof_out;
out.field_table          = arrayfun( ...
    @(s) [s.dx, s.dy, s.tile_row, s.tile_col], fields, ...
    'UniformOutput', false);
out.field_table          = vertcat(out.field_table{:});
out.field_names          = {fields.name}.';
out.chfraydir_nom        = nom.src_dir(:);
if has_cfg
    out.config_table     = cfgs(:);   % name + setter list (+ .raw verbatim)
    out.config_names     = {cfgs.name}.';
    out.per_field_dwdx     = per_field_dwdx;        % Nc x Nf
    out.per_field_w_nom_2d = per_field_w_nom;       % Nc x Nf
else
    % preserved surface: without 'configs' the cells keep their Nf x 1
    % shape and no configuration fields are added
    out.per_field_dwdx     = per_field_dwdx(1, :).';
    out.per_field_w_nom_2d = per_field_w_nom(1, :).';
end
out.rx_path              = rx_path;
out.delta                = opts.delta;
out.method               = opts.method;
out.wf_elt               = per_field_struct{1, 1}.wf_elt;
out.rot_output           = opts.rot_output;
out.cbm                  = per_field_struct{1, 1}.cbm;
out = apply_opd_convention(out, opts.orient, opts.sign);
out.reset_xp             = reset_xp_stamp;   % true | false | 'no-effect'
out.reset_xp_method      = opts.reset_xp_method;
if ~isempty(pf_out), out.pupil_find = pf_out; end   % per-config metrics

% Add per-field LOS if SPOT was computed
if opts.compute_los
    if has_cfg, out.dcdx_per_field = per_field_dcdx;
    else,       out.dcdx_per_field = per_field_dcdx(1, :).'; end
    if isempty(opts.spot_elt)
        out.spot_elt = session.num_elt();  % Default focal plane
    else
        out.spot_elt = opts.spot_elt;
    end
end
end


% =====================================================================
function t = cfg_tag(has_cfg, cfgs, ic)
% "config <name> / " prefix for the progress lines; EMPTY without
% 'configs', so the log of a no-configs run is unchanged.
if has_cfg, t = sprintf('config %s / ', cfgs(ic).name);
else,       t = ''; end
end


% =====================================================================
function fields = make_5field_set(field_x_rad, field_y_rad)
fields = struct('name', {}, 'dx', {}, 'dy', {}, ...
                 'tile_row', {}, 'tile_col', {});
fields(end+1) = field_entry('C',  0,            0,            1, 1);
fields(end+1) = field_entry('UL', -field_x_rad, +field_y_rad, 2, 0);
fields(end+1) = field_entry('UR', +field_x_rad, +field_y_rad, 2, 2);
fields(end+1) = field_entry('LL', -field_x_rad, -field_y_rad, 0, 0);
fields(end+1) = field_entry('LR', +field_x_rad, -field_y_rad, 0, 2);
end


function fields = make_grid_field_set(nx, ny, field_x_rad, field_y_rad)
if nx > 1, dx_axis = linspace(-field_x_rad, +field_x_rad, nx);
else, dx_axis = 0; end
if ny > 1, dy_axis = linspace(-field_y_rad, +field_y_rad, ny);
else, dy_axis = 0; end
fields = struct('name', {}, 'dx', {}, 'dy', {}, ...
                 'tile_row', {}, 'tile_col', {});
for ir = 1:numel(dy_axis)
    for ic = 1:numel(dx_axis)
        dx = dx_axis(ic); dy = dy_axis(ir);
        is_center = (abs(dx) < 1e-30) && (abs(dy) < 1e-30);
        if is_center, nm = 'C';
        else, nm = sprintf('F_r%d_c%d', ir-1, ic-1); end
        fields(end+1) = field_entry(nm, dx, dy, ir-1, ic-1); %#ok<AGROW>
    end
end
end


function e = field_entry(name, dx, dy, tr, tc)
e.name = name; e.dx = dx; e.dy = dy; e.tile_row = tr; e.tile_col = tc;
end


function idx = find_center_field_index(fields)
idx = [];
for k = 1:numel(fields)
    if abs(fields(k).dx) < 1e-30 && abs(fields(k).dy) < 1e-30
        idx = k; return;
    end
end
end


function new_dir = field_to_chfraydir(dir_nom, dx_rad, dy_rad)
v = dir_nom(:) + [dx_rad; dy_rad; 0];
n = norm(v);
if n == 0
    error('macos:dw_dx_multi:zerodir', ...
        'zero-magnitude direction after field offset');
end
new_dir = v / n;
end


function [nx, ny] = parse_grid_spec(spec)
toks = regexp(lower(spec), 'x', 'split');
if numel(toks) ~= 2
    error('macos:dw_dx_multi:grid', ...
        '''grid'' must be ''NxM''; got %s', spec);
end
nx = str2double(toks{1});
ny = str2double(toks{2});
if isnan(nx) || isnan(ny) || nx < 1 || ny < 1
    error('macos:dw_dx_multi:grid', ...
        '''grid'' must be ''NxM'' with positive integers; got %s', spec);
end
end


function fields = load_field_file(fname)
fid = fopen(fname, 'r');
if fid < 0
    error('macos:dw_dx_multi:fields', ...
        'cannot open fields file: %s', fname);
end
c = onCleanup(@() fclose(fid));
fields = struct('name', {}, 'dx', {}, 'dy', {}, ...
                 'tile_row', {}, 'tile_col', {});
while true
    ln = fgetl(fid);
    if ~ischar(ln); break; end
    s = strtrim(ln);
    if isempty(s) || startsWith(s, '#'), continue; end
    toks = regexp(s, '\s+', 'split');
    if numel(toks) < 5
        error('macos:dw_dx_multi:fields', ...
            'fields-file row needs 5 columns: %s', s);
    end
    fields(end+1) = field_entry(toks{1}, ...
        str2double(toks{2}), str2double(toks{3}), ...
        str2double(toks{4}), str2double(toks{5})); %#ok<AGROW>
end
end

function mustBeDeltaSize(d)
    if ~(isequal(size(d), [1 1]) || isequal(size(d), [1 6]))
        error('macos:dw_dx_multi:deltaSize', ...
            'delta must be (1,1) or (1,6)');
    end
end


