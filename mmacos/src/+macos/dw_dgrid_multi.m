function out = dw_dgrid_multi(session, rx_path, opts)
%MACOS.DW_DGRID_MULTI  Multi-field dw/d(grid-data) sensitivity supervisor.
%   out = macos.dw_dgrid_multi(SESSION, RX_PATH, ...) loads RX_PATH on
%   SESSION, snapshots the nominal source FoV, then loops field points --
%   each iteration absolutely sets ChfRayDir via set_src_fov and runs
%   dw_dgrid at the new field.  The per-field results are tiled into one
%   big OPDall canvas and scattered into one big dwdgall (canonical
%   state-vector form):
%
%       wall = dwdgall * x + w0_stacked
%
%   where wall is the column-major vectorisation of OPDall via m2v, and
%   x is the stacked influence-map amplitude of every grid poke on every
%   grid-bearing surface.
%
%   The grid-data (GMI pgrid) companion to macos.dw_dz_zernike_multi and
%   macos.dw_dsurf_multi, built on the same field-loop + tiling machinery,
%   a different DOF set.
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
%   OTHER NAME-VALUE PAIRS (forwarded to dw_dgrid):
%     'influence'  [N x N x K] maps for every grid element, OR a per-segment
%                  basis (a macos.segment_grid_basis struct, or a cell per grid
%                  element).  Default: a low-order Zernike-on-grid basis.
%     'zmodes'     Noll/ANSI modes for the default basis.  Default [4 5 6 7 8 11].
%     'elts'       Vector of element IDs to include.  Default [] (auto-detect all
%                  grid-bearing elements from the loaded prescription).
%     'delta', 'method', 'exit_pupil_elt', 'verbose'.
%
%   'ngridpts'  (default [] = keep the .in value) ray-grid sampling
%               override, applied once right after the Rx load; persists
%               across the per-field calls.  Clamped by the engine to
%               [3, model-size limit] (warns).
%   'reset_xp'  (default true) re-find the exit pupil (FEX, chief ray) for
%               EACH field before differencing, so the nominal wavefront is
%               referenced to that field's own chief ray and the gross field
%               TILT is removed (without it, an off-axis field carries a
%               large linear-in-field tilt that swamps the OPD canvas).  A
%               poke's OWN tilt is retained -- the reference is fixed per
%               field, not re-fit after each poke.  Requires a STOP set and
%               > 3 elements.  Set false to keep the prescription's
%               elt nElt-1 reference unchanged.
%               Restore scope: the pre-loop EP is snapshotted and restored
%               via get_xp/set_xp, i.e. vpt/psi/rad (VptElt/PsiElt/KrElt at
%               nElt-1) only.  FEX-written auxiliary fields on the EP
%               element (RptElt, zElt, fElt, eElt, KcElt) are left as
%               re-derived, not rolled back -- callers who hand-author
%               those on the EP element own re-asserting them afterward.
%   'reset_xp_method'  'fex' (default; 'sxp' = deprecated alias -- FEX and
%                SXP are merged in the engine) | 'pupil_find' (cone-
%                convergence sphere placement; see dw_dx_multi's help --
%                needs a stop: 'stop_elt' or a deck-declared ApStop=,
%                object-space header form included.  'pf_scope' 'field'
%                (default: per-(config, field) mini-cone, chief-tied
%                like fex) | 'config' (frozen field-set-wide sphere --
%                diagnostic; the retained field tilt leaks into the
%                columns) + 'pf_probe_rad'; dw_dx_multi's help
%                has the full story).  Historical
%                note: the alias exists because FEX/SXP were merged
%               (FEX radius = chief-ray distance to iElt+1 = the FP), so
%               FEX is the only path.  'sxp' is accepted as an alias with
%               a one-time warning; do not rely on it.
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
%     dwdgall       Nw x Ng canonical state-vector Jacobian
%     dwdxall       alias (== dwdgall) -- canonical-form consumers
%     w0_stacked    Nw x 1 stacked nominal OPDs (m2v of OPDall)
%     indxall       struct with i, j, size -- m2v round-trip metadata
%     OPDall        full tiled OPD canvas (un-vectorised)
%     channel_names Ng x 1 cell array
%     iElt / map_idx  Ng x 1 per-channel element id + influence-map index
%     field_table   Nfields x 4: [dx_rad, dy_rad, tile_row, tile_col]
%     field_names   Nfields x 1 cell array
%     chfraydir_nom 3 x 1 nominal ChfRayDir
%     per_field_dwdg      Nfields x 1 cell of single-field dwdg blocks
%     per_field_w_nom_2d  Nfields x 1 cell of single-field nominal OPDs
%     rx_path / delta / method / wf_elt / zmodes  -- echoed inputs
%
%   See also: macos.dw_dgrid, macos.dw_dz_zernike_multi, macos.dw_dsurf_multi.

arguments
    session
    rx_path                     (1,:) char {mustBeNonempty}
    opts.field_x_rad            (1,1) double = NaN
    opts.field_y_rad            (1,1) double = NaN
    opts.fields                 (1,:) char = ''
    opts.grid                   (1,:) char = ''
    opts.influence              = []   % [NxNxK] | per-segment struct | cell
    opts.zmodes                 (1,:) double = [4 5 6 7 8 11]
    opts.elts                   (:,1) double = []
    opts.delta                  (1,1) double = 1e-6
    opts.method                 (1,:) char {mustBeMember(opts.method, ...
                                  {'central','forward'})} = 'central'
    opts.exit_pupil_elt         (1,1) double {mustBeInteger} = -1
    opts.reset_xp               (1,1) logical = true
    opts.verbose                (1,1) logical = false
    opts.reload_rx              (1,1) logical = true
    opts.reset_xp_method        (1,:) char {mustBeMember(opts.reset_xp_method, ...
                                  {'fex','sxp','pupil_find'})} = 'fex'
    opts.fex_axis               (1,:) char {mustBeMember( ...
        opts.fex_axis, {'chief','centroid'})} = 'chief'
    opts.pupil_find_opts        cell = {}
    opts.pf_scope               (1,:) char {mustBeMember( ...
        opts.pf_scope, {'config','field'})} = 'field'
    opts.pf_probe_rad           (1,1) double = NaN
    opts.ngridpts               double {mustBeScalarOrEmpty} = []
    opts.stop_elt               double {mustBeScalarOrEmpty} = []
    opts.configs                          = []
    opts.src_samp               double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.compute_los            (1,1) logical = false
    opts.spot_elt               double {mustBeScalarOrEmpty, mustBeInteger} = []
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
F.name       = 'dw_dgrid_multi';
F.jac        = 'dwdg';
F.all_names  = {'dwdgall','dwdxall'};
F.per_name   = 'per_field_dwdg';
F.hoist      = @(s, rx, o) grid_influence_hoist_(o);
F.single     = @(s, rx, o, h) macos.dw_dgrid(s, rx, ...
    'influence', h, 'elts', o.elts, 'delta', o.delta, ...
    'method', o.method, 'exit_pupil_elt', o.exit_pupil_elt, ...
    'verbose', o.verbose, 'reload_rx', false, ...
    'compute_los', o.compute_los, 'spot_elt', o.spot_elt);
F.meta       = @(sf) struct('iElt', sf.iElt, 'map_idx', sf.map_idx);
F.extras     = @(out, o, sf1) setfield(out, 'zmodes', o.zmodes); %#ok<SFLD>
F.cfg_tag_fieldlog = false;
F.los_reshape      = false;
F.deprecate_sxp    = true;
out = dw_multi_core(session, rx_path, opts, F);
end


% =====================================================================
function infl = grid_influence_hoist_(opts)
% Resolve the influence basis ONCE so every field shares identical
% columns (same channel order, same map amplitudes).  When the caller
% supplies 'influence' it is used verbatim; otherwise a default
% Zernike-on-grid basis is built at the first grid element's sampling.
infl = opts.influence;
if isempty(infl)
    g = macos.find_grid_elts();
    if ~isempty(opts.elts)
        g = intersect(g, opts.elts);
    end
    if isempty(g)
        error('macos:dw_dgrid_multi:nogrid', ...
            'no grid-bearing elements in the loaded prescription');
    end
    nsz  = double(mmacos('elt_srf_grid_size', g(1), 1));
    infl = macos.zernike_grid_basis(nsz, opts.zmodes);
end
end
