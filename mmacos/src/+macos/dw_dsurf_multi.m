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
%   Reflector / Refractor / NSReflector / NSRefractor / Segment with
%   |Kr| << the flat sentinel; engine-queried -- macos.find_powered_elts).
%   An id passed in 'elts' that is not powered-capable ERRORS with a named
%   reason rather than being dropped silently.
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
    opts.fex_axis               (1,:) char {mustBeMember( ...
        opts.fex_axis, {'chief','centroid'})} = 'chief'
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

F = struct();
F.name       = 'dw_dsurf_multi';
F.jac        = 'dwds';
F.all_names  = {'dwdsall','dwdxall'};
F.per_name   = 'per_field_dwds';
F.hoist      = [];
F.single     = @(s, rx, o, h) macos.dw_dsurf(s, rx, ...
    'params', o.params, 'elts', o.elts, 'delta', o.delta, ...
    'method', o.method, 'exit_pupil_elt', o.exit_pupil_elt, ...
    'verbose', o.verbose, 'reload_rx', false, ...
    'remove_ptt', o.remove_ptt, 'compute_los', o.compute_los, ...
    'spot_elt', o.spot_elt);
F.meta       = @(sf) struct('iElt', sf.iElt, 'param', {sf.param});
F.extras     = @(out, o, sf1) setfield(out, 'params', o.params); %#ok<SFLD>
F.cfg_tag_fieldlog = false;
F.los_reshape      = false;
F.deprecate_sxp    = false;
out = dw_multi_core(session, rx_path, opts, F);
end
