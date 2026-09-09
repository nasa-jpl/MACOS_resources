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
%   share one convention -- the OPD numerator in the deck's BaseUnits
%   (matching w0/opd() and the figure rungs, Dave 2026-08-25), per SI
%   METRE for translations and per rad for rotations -- so one numeric
%   'delta' is one physical poke for either.
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
%               common-mode between w_nom and every poked w, but it does
%               NOT fully cancel in the FD columns: a poke remaps rays
%               slightly, and the remapped rays sample the frame-tilt
%               gradient, so the columns pick up an error PROPORTIONAL
%               to the tilt retained in w_nom.  MEASURED (2026-08-27
%               w_nom audit, zoom fixture, +-1 arcmin): rigid-body
%               columns off by 3-5% vs a chief-tied reference at
%               0.64 mm of retained tilt; 1e-6 at 4e-5 mm.
%               Matches dw_dz_zernike_multi / dw_dsurf_multi
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
%               (design/src/pupil_find); WHERE it places is set by
%               'pf_scope' (default 'field': one placement per
%               (configuration, field) block, so each combo's nominal
%               is chief-tied exactly as under fex).
%               Needs a stop: 'stop_elt', or a deck-declared ApStop=
%               (object-space header form included -- the segmented-
%               primary idiom; pupil_find leaves a deck stop in force).  'pupil_find_opts' forwards extra
%               name-values to the finder (anchor/nodes/...).  Per-block
%               fit metrics return in out.pupil_find.
%   'fex_axis'  'chief' (default) | 'centroid'.  The FEX pupil-sphere
%               AXIS for the per-field reset (engine CHIEFRAY/CENTROID
%               toggle; api xp_fnd mode 1 | 0).  The vertex and radius
%               are axis-invariant -- only psi moves -- and on an
%               obscured or segmented beam the downstream centroid
%               walks off the chief, so 'centroid' tilts the reference
%               sphere and leaves pure tip/tilt (+piston) FRAME terms
%               in w_nom (Luis 2026-08-27; the FEX-axis ruling made
%               chief the default on every platform).  A diagnostic
%               opt-in, echoed in out.fex_axis.  fex/sxp only:
%               combining it with reset_xp_method='pupil_find' errors
%               -- the pupil_find written reference is chief-tied by
%               doctrine.
%   'pf_scope'  'field' (default; flipped from 'config' 2026-08-27,
%               the w_nom audit) | 'config'.  Scope of the pupil_find
%               placement.  'field': one MINI-CONE fit per
%               (configuration, field) block -- a 3x3 probe grid of
%               half-width 'pf_probe_rad' centered on that field.  The
%               WRITTEN sphere is that combo's own chief-crossing vertex
%               + axis + radius (pupil_find 'vertex','chief'), so the
%               field tilt is absorbed per block and w_nom sits at fex
%               scale (measured == fex to 4e-11 mm on the zoom and
%               e5hex1 fixtures); the mini-cone BUNDLE fit is kept as
%               the pupil-wander diagnostic (out.pupil_find: bundle_vtx,
%               vtx_minus_fex, dep_rms) -- writing the bundle vertex
%               itself injects a pure-tilt frame term (0.38 mm lateral
%               offset -> 4.4e-3 mm RMS of tilt, zero aberration
%               content; measured 2026-08-25).  Each block harvests, and
%               subtracts, its OWN w_nom against its OWN sphere.
%               'config': ONE field-set-wide sphere per configuration
%               (frozen best-fit EP) -- a DIAGNOSTIC mode for frozen-
%               reference / pupil-wander studies: the per-field tilt
%               stays in w_nom (0.64 mm RMS at +-1 arcmin on the zoom
%               fixture) and leaks 3-5% into the rigid-body dwdx
%               columns (see 'reset_xp' above).
%   'pf_probe_rad'  probe half-width in rad for pf_scope='field'
%               (default NaN = 0.15x the field half-width).  Too small
%               is ill-conditioned: the probe chief-ray crossings
%               degenerate as the cone closes.
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
    opts.fex_axis            (1,:) char {mustBeMember( ...
        opts.fex_axis, {'chief','centroid'})} = 'chief'
    opts.pupil_find_opts     cell = {}
    opts.pf_scope            (1,:) char {mustBeMember( ...
        opts.pf_scope, {'config','field'})} = 'field'
    opts.pf_probe_rad        (1,1) double = NaN
    opts.configs                          = []
    opts.verbose             (1,1) logical = false
    opts.ngridpts            double {mustBeScalarOrEmpty} = []
    opts.src_samp            double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.compute_los         (1,1) logical = false
    opts.spot_elt            double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.orient (1,:) char {mustBeMember(opts.orient, {'raw','xy'})} = 'raw'   % OPD array orientation (doc/opd_conventions.md)
    opts.sign   (1,:) char {mustBeMember(opts.sign, {'opl','wavefront'})} = 'opl' % OPD sign convention
    opts.opd_ref (1,:) char {mustBeMember(opts.opd_ref, {'mean','chief'})} = 'mean'
                                     % OPD reference (macos.opd_ref): 'mean' =
                                     % whole-aperture mean (engine default);
                                     % 'chief' = the chief ray -- on SEGMENTED
                                     % decks a single-segment poke under 'mean'
                                     % pistons EVERY other segment by
                                     % -(N_k/N)*mean(poked response) (PLAN 0.x);
                                     % under 'chief' they read exactly 0.
                                     % Re-applied after every Rx (re)load.
end

F = struct();
F.name       = 'dw_dx_multi';
F.jac        = 'dwdx';
F.all_names  = {'dwdxall'};
F.per_name   = 'per_field_dwdx';
F.hoist      = @(s, rx, o) dx_groups_hoist_(rx, o);
F.single     = @(s, rx, o, h) macos.dw_dx(s, rx, ...
    'dofs', o.dofs, 'elts', o.elts, 'fp_mode', o.fp_mode, ...
    'ep_elt', o.ep_elt, 'include_source', o.include_source, ...
    'src_stop_mode', o.src_stop_mode, 'src_stop_pos', o.src_stop_pos, ...
    'src_stop_elt', o.src_stop_elt, ...
    'include_non_optics', o.include_non_optics, ...
    'groups', h, 'groups_auto', false, ...
    'group_coords', o.group_coords, 'group_fp_mode', o.group_fp_mode, ...
    'group_stop_mode', o.group_stop_mode, ...
    'group_stop_pos', o.group_stop_pos, ...
    'rot_output', o.rot_output, 'delta', o.delta, ...
    'delta_units', o.delta_units, 'method', o.method, ...
    'exit_pupil_elt', o.exit_pupil_elt, 'verbose', o.verbose, ...
    'reload_rx', false, 'compute_los', o.compute_los, ...
    'spot_elt', o.spot_elt);
F.meta       = @(sf) struct('iElt', sf.iElt, 'kind', {sf.kind}, ...
                            'dof_idx', sf.dof_idx);
F.extras     = @(out, o, sf1) dx_extras_(out, o, sf1);
F.cfg_tag_fieldlog = true;
F.los_reshape      = true;
F.deprecate_sxp    = false;
out = dw_multi_core(session, rx_path, opts, F);
end


% =====================================================================
function grp_map = dx_groups_hoist_(rx_path, opts)
% Element groups: PARSE-ONCE HOIST.  'groups_auto' reads the Rx FILE, so
% forwarding it to every per-field dw_dx call would re-parse it per
% field.  Materialize the merged map ONCE (auto first, explicit entries
% overriding on a name collision -- dw_dx's own merge order) and hand it
% down as 'groups' with groups_auto=false.  Same map object for every
% field, so the group channel COUNT cannot drift between blocks; the
% channel-identity assertion in the core gates that regardless.
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
end

function out = dx_extras_(out, opts, sf1)
out.rot_output = opts.rot_output;
out.cbm        = sf1.cbm;
end

function mustBeDeltaSize(d)
    if ~(isequal(size(d), [1 1]) || isequal(size(d), [1 6]))
        error('macos:dw_dx_multi:deltaSize', ...
            'delta must be (1,1) or (1,6)');
    end
end
