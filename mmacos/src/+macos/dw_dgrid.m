function out = dw_dgrid(session, rx_path, opts)
%MACOS.DW_DGRID  Finite-difference dW/d(grid-data) sensitivity Jacobian.
%   out = macos.dw_dgrid(SESSION, RX_PATH) perturbs each grid-bearing surface
%   (ANY GridData-enabled SrfType -- GridData/AsGrData/MonGrData/ZrnGridData/
%   FreeForm) by a set of influence-function maps and returns the OPD-wavefront
%   Jacobian dW/d(map amplitude), one column per (element, influence map).
%
%   The maps are ADDED to the surface's grid data (GridMat) in place, so the
%   element KEEPS its SrfType and any conic/Zernike/monomial components --
%   unlike GMI, which forces SrfType->9 (GridData) and clobbers a composite
%   surface.  This is the grid-data leg of the GMI sensitivity migration; it
%   mirrors macos.dw_dz_zernike (rigid-body = dw_dx, Zernike = dw_dz_zernike).
%
%   Name-value pairs:
%     'influence'       [N×N×K] maps applied to every grid element; OR a
%                       per-segment basis (a macos.segment_grid_basis struct, or
%                       a cell per grid element -- see grid_channels).  Default:
%                       a low-order Zernike-on-grid basis (zernike_grid_basis)
%                       at the elements' grid size.  N = surface grid sampling.
%     'zmodes'          Noll modes for the default basis.  Default [4 5 6 7 8 11].
%     'elts'            Vector of element IDs to include in the sensitivity
%                       calculation.  Only grid-bearing elements that are also
%                       in this list will be perturbed.  Default [] (auto-detect
%                       all grid-bearing elements from the loaded prescription).
%                       Example: 'elts', [3 5 7] includes only elements 3, 5, 7.
%     'exit_pupil_elt'  surface the wavefront is read at.  Default -1 = nElt-1.
%     'delta'           finite-difference step (map amplitude).  Default 1e-6.
%     'method'          'central' (default) | 'forward'.
%     'verbose'         logical.  Default false.
%     'reload_rx'       reload RX_PATH first.  Default true.
%     'ngridpts'        ray-grid sampling override (nGridPts).  Default [] =
%                       keep the .in-file value.  Clamped by the engine to
%                       [3, model-size limit] (warns).  This is the RAY grid;
%                       it is independent of the surface GridMat sampling.
%
%   Returns a struct: dwdg (Nw×Nz Jacobian), w_nom_2d, w_nom_vec, indx,
%   channel_names (Nz×1), iElt, map_idx, rx_path, wf_elt, delta, method.
%
%   See also: macos.dw_dz_zernike, macos.find_grid_elts,
%             macos.channels.grid_channels, macos.zernike_grid_basis.
arguments
    session
    rx_path (1,:) char = ''
    opts.influence              = []   % [NxNxK] | per-segment struct | cell
    opts.zmodes         (1,:) double = [4 5 6 7 8 11]
    opts.elts           (:,1) double = []
    opts.exit_pupil_elt (1,1) double = -1
    opts.delta          (1,1) double = 1e-6
    opts.method         (1,:) char {mustBeMember(opts.method,{'central','forward'})} = 'central'
    opts.verbose        (1,1) logical = false
    opts.reload_rx      (1,1) logical = true
    opts.ngridpts       double {mustBeScalarOrEmpty} = []
    opts.src_samp       double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.compute_los    (1,1) logical = false
    opts.spot_elt       double {mustBeScalarOrEmpty, mustBeInteger} = []
end
if opts.reload_rx && ~isempty(rx_path)
    session.load_rx(rx_path);
end
apply_ngridpts(session, opts.ngridpts, 'dw_dgrid');

% Apply source sampling if specified
if ~isempty(opts.src_samp)
    session.set_src_sampling(opts.src_samp);
    session.modify();  % Flush cache so the new sampling takes effect
end

wf_elt = opts.exit_pupil_elt;
if wf_elt < 0
    wf_elt = session.num_elt() - 1;
end

g = macos.find_grid_elts();
if ~isempty(opts.elts)
    g = intersect(g, opts.elts);
end
if isempty(g)
    error('macos:dw_dgrid:nogrid', ...
        'no grid-bearing elements in the loaded prescription');
end

% Influence basis: caller-supplied, else a default Zernike-on-grid basis at
% the first eligible element's grid size (all eligible elements must share it).
infl = opts.influence;
if isempty(infl)
    nsz  = double(mmacos('elt_srf_grid_size', g(1), 1));
    infl = macos.zernike_grid_basis(nsz, opts.zmodes);
end

channels = macos.channels.grid_channels(session, infl);
wf_func  = @() local_wf(session, wf_elt);

% Create spot_func if LOS computation requested
n_elt = session.num_elt();
if opts.compute_los
    if isempty(opts.spot_elt)
        spot_elt_use = n_elt;  % Default to focal plane
    else
        spot_elt_use = opts.spot_elt;
    end
    spot_func = @() local_spot(spot_elt_use);
else
    spot_func = [];
end

[dwdg, w_nom_2d, w_nom_vec, indx, names, dcdx, spot_pos, spot_neg, spot_nom, spot_pert] = ...
    macos.dwdz_for_current_source(channels, wf_func, opts.delta, ...
        'method', opts.method, 'verbose', opts.verbose, 'spot_func', spot_func);

iElt_out = zeros(numel(channels), 1);
idx_out  = zeros(numel(channels), 1);
for k = 1:numel(channels)
    iElt_out(k) = channels{k}.iElt;
    idx_out(k)  = channels{k}.idx;
end

out = struct();
out.dwdg          = dwdg;
out.w_nom_2d      = w_nom_2d;
out.w_nom_vec     = w_nom_vec;
out.indx          = indx;
out.channel_names = names;
out.iElt          = iElt_out;
out.map_idx       = idx_out;
out.rx_path       = rx_path;
out.wf_elt        = wf_elt;
out.delta         = opts.delta;
out.method        = opts.method;

% Add LOS fields if SPOT was computed
if opts.compute_los
    out.dcdx      = dcdx;
    out.spot_elt  = spot_elt_use;
    if strcmp(opts.method, 'central')
        out.spot_pos = spot_pos;
        out.spot_neg = spot_neg;
    else  % forward
        out.spot_nom  = spot_nom;
        out.spot_pert = spot_pert;
    end
end
end

% ---------------------------------------------------------------------------
function W = local_wf(session, wf_elt)
session.trace(wf_elt);
W = session.opd();
end

function S = local_spot(spot_elt)
S = macos.spot(spot_elt, 'ref', 'tout', 'at', 'chief');
end
