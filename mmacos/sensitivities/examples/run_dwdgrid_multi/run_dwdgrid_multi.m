% run_dwdgrid_multi.m -- multi-field dw/d(grid-data) sensitivity (example).
% =====================================================================
%  Multi-field GRID-DATA (the GMI pgrid channel) wavefront-sensitivity
%  Jacobian: each influence-function "poke" ADDED to a surface's grid data
%  (in place, ANY GridData-enabled SrfType -- so the element keeps its
%  conic/Zernike/monomial parts) has a wavefront sensitivity dW/d(poke).
%  Result in canonical state-vector form:
%
%      wall = dwdgall * x + w0_stacked
%
%  SETUP: run `mmacos_setup` once per MATLAB session first (it puts the
%  +macos package, the mmacos mex, and the plot/save helpers on the path).
%
%  This example is self-contained -- it ships e5hex1_grid.in alongside the
%  script.  TO RUN ON YOUR OWN SYSTEM, point RX (CONFIG block) at your own
%  .in (needs a STOP set and at least one grid-bearing element); everything
%  below the CONFIG block is generic.
%
%  Outputs (this directory):
%    *_OPDall.png    nominal OPD at every field point (field canvas)
%    *_channels.png  EACH channel's MULTI-FIELD dW (one subplot per grid
%                    element x influence poke) -- the per-channel sensitivity
%    *_<rx>.mat      dwdgall (= dwdxall) + w0_stacked + indxall + ...
%
%  Per-field exit-pupil reset (reset_xp=true, the default) re-references each
%  field's nominal to its OWN chief ray (FEX) so the gross off-axis field
%  tilt is removed; a poke's own tilt is retained.  Requires a STOP + >3 elts.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX     = fullfile(here, 'e5hex1_grid.in'); % self-contained demo Rx (point
                                           % elsewhere to run on your own system)
MODEL  = 256;           % model size: >= the Rx's nGridMat (256 for e5hex1_grid)
NGRIDPTS = 63;         % ray-grid sampling override ([] = keep the .in value)
FOV    = 1e-4;          % half-field (rad) for the 4 corner field points
DELTA  = 1e-6;          % finite-difference step (grid-map amplitude, BaseUnits)
ZMODES = [4 5 6 7 8 9]; % default poke shapes: MACOS ANSI Zernike indices
INFL   = [];            % optional [N x N x K] influence maps (DM actuators,
                        % measured figure...) -- overrides ZMODES when set
% --------------------------------------------------------------------
% For a SEGMENTED aperture this script auto-confines each default-basis
% Zernike to the segment aperture (lMon) when the Rx declares lMon +
% GridSrfdx; otherwise the basis spans the full grid.
% =====================================================================

[~, rxstem] = fileparts(RX);
fprintf('=== dw/d(grid) multi-field: %s (model %d) ===\n', rxstem, MODEL);
m = macos.Session(MODEL);  m.load_rx(RX);

% Build the influence basis (unless caller supplied INFL).
if isempty(INFL)
    g = macos.find_grid_elts();
    if isempty(g)
        error('run_dwdgrid_multi:nogrid', 'no grid-bearing elements in %s', RX);
    end
    N    = double(mmacos('elt_srf_grid_size', g(1), 1));
    txt  = fileread(RX);
    % str2num (not str2double) so Fortran D-exponents parse, e.g. 1.0D-02
    lMon = str2num(regexp(txt, '(?<=lMon=)\s*[\d.eEdD+-]+',      'match', 'once'));  %#ok<ST2NM>
    gdx  = str2num(regexp(txt, '(?<=GridSrfdx=)\s*[\d.eEdD+-]+', 'match', 'once'));  %#ok<ST2NM>
    if ~isempty(lMon) && ~isempty(gdx) && gdx > 0
        ap_frac = lMon / (((N-1)/2) * gdx);   % confine to the segment aperture
        fprintf('aperture-confined basis: %.0f%% of grid half-width\n', 100*ap_frac);
        INFL = macos.zernike_grid_basis(N, ZMODES, ap_frac);
    else
        INFL = macos.zernike_grid_basis(N, ZMODES);   % full-grid basis
    end
end

out = macos.dw_dgrid_multi(m, RX, ...
    'ngridpts', NGRIDPTS, ...
    'field_x_rad', FOV, 'field_y_rad', FOV, ...
    'influence', INFL, 'delta', DELTA);

% ---- Figures -------------------------------------------------------
plot_opd_canvas(out, sprintf('dw/d(grid) %s -- nominal OPD, %d fields', ...
    rxstem, numel(out.field_names)), here, ['dwdgrid_multi_' rxstem '_OPDall.png']);
plot_dw_channels(out, sprintf('dW/d(grid) -- each poke, %d fields (%s)', ...
    numel(out.field_names), rxstem), here, ['dwdgrid_multi_' rxstem '_channels.png']);

% ---- Per-element single-page plots: center-field AND multi-field ----
plot_dw_per_element(out, 'center', here, ['dwdgrid_multi_' rxstem]);
plot_dw_per_element(out, 'multi',  here, ['dwdgrid_multi_' rxstem]);

% ---- Save canonical state-vector .mat ------------------------------
save_dw_multi(out, MODEL, fullfile(here, ['dwdgrid_multi_' rxstem '.mat']));
fprintf('=== dw/d(grid) multi: %d channels x %d fields ===\n', ...
    numel(out.channel_names), numel(out.field_names));
