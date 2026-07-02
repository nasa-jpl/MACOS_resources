% run_dwdgrid_multi_e5hex2.m -- PART 2: multi-field dw/d(grid-data) on the
% e5hex2 segmented telescope, from the SAVED per-segment GS Zernike basis.
% =====================================================================
%  Consumes the basis written by make_gs_basis_e5hex2.m (PART 1) and assembles
%  the multi-field grid-data (GMI pgrid) wavefront-sensitivity Jacobian by
%  poking each (segment, mode).  Splitting the (slow) basis build from the dW
%  assembly lets the dW step be re-run cheaply.
%
%  Run PART 1 first:   run make_gs_basis_e5hex2      % -> gridmat_*.mat
%  then this:          run run_dwdgrid_multi_e5hex2
%
%  Outputs (this directory):
%    *_OPDall.png        nominal OPD at every field point (field canvas)
%    *_channels.png      each (segment x mode) channel's MULTI-field dW
%    *_elt<E>_multi.png  per segment: each mode's MULTI-field dW
%    *_elt<E>_center.png per segment: each mode's CENTER-field dW
%    *_e5hex2grid.mat    dwdgall + w0_stacked + indxall + the basis
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
cd(here);

% ===================  CONFIG  =======================================
RX        = fullfile(here, 'e5hex2grid.in');                  % 19-hex + conf. Ref
BASIS_MAT = fullfile(here, 'gridmat_e5hex2grid_ansi_gs.mat'); % from PART 1
MODEL     = 512;        % model size (>= nGridMat=256; e5hex2 loads at 512)
FOV       = 1e-4;       % half-field (rad) for the 4 corner field points
DELTA     = 1e-6;       % finite-difference step (grid-map amplitude)
% =====================================================================

if exist(BASIS_MAT, 'file') ~= 2
    error(['basis not found: %s\n  Run PART 1 first:  run make_gs_basis_e5hex2'], ...
          BASIS_MAT);
end
[~, rxstem] = fileparts(RX);
fprintf('=== dw/d(grid) multi-field [saved GS basis]: %s (model %d) ===\n', ...
        rxstem, MODEL);

% (1) Load the per-segment GS basis struct saved by PART 1.
basis = load(BASIS_MAT);            % -struct save -> fields seg, modes, N, ...
fprintf('loaded GS basis: %d segments x %d modes (N=%d grid, dx=%.4g)\n', ...
        numel(basis.seg), numel(basis.modes), basis.N, basis.gdx);

% (2) Multi-field dw/d(grid) sensitivities, each segment's own modes routed to
%     that segment (grid_channels keys the basis struct by iElt).
m   = macos.Session(MODEL);
out = macos.dw_dgrid_multi(m, RX, ...
    'field_x_rad', FOV, 'field_y_rad', FOV, ...
    'influence', basis, 'delta', DELTA);

% ---- Figures -------------------------------------------------------
plot_opd_canvas(out, sprintf('dw/d(grid) %s -- nominal OPD, %d fields', ...
    rxstem, numel(out.field_names)), here, ['dwdgrid_multi_' rxstem '_OPDall.png']);
plot_dw_channels(out, sprintf('dW/d(grid) per-segment GS basis -- each (seg,mode), %d fields (%s)', ...
    numel(out.field_names), rxstem), here, ['dwdgrid_multi_' rxstem '_channels.png']);
plot_dw_per_element(out, 'center', here, ['dwdgrid_multi_' rxstem]);
plot_dw_per_element(out, 'multi',  here, ['dwdgrid_multi_' rxstem]);

% ---- Save canonical state-vector .mat ------------------------------
save_dw_multi(out, MODEL, fullfile(here, ['dwdgrid_multi_' rxstem '.mat']));
fprintf('=== dw/d(grid) e5hex2: %d channels x %d fields ===\n', ...
    numel(out.channel_names), numel(out.field_names));

exit(0);
