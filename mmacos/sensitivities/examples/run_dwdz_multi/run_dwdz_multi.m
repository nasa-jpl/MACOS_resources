% run_dwdz_multi.m -- multi-field dw/dz ZERNIKE-COEFFICIENT sensitivity (example).
% =====================================================================
%  Multi-field Zernike-coefficient wavefront-sensitivity Jacobian: per
%  (element, Zernike mode), for the MonZern / FFZern / element-Zern surface
%  components.  Canonical state-vector form:
%
%      wall = dwdzall (= dwdxall) * x + w0_stacked
%
%  SETUP: run `mmacos_setup` once per MATLAB session first (it puts the
%  +macos package, the mmacos mex, and the plot/save helpers on the path).
%
%  This example is self-contained -- it ships e5hex1.in alongside the script.
%  TO RUN ON YOUR OWN SYSTEM, point RX (CONFIG block) at your own .in;
%  everything below the CONFIG block is generic.
%
%  Outputs (this directory):
%    *_OPDall.png    nominal OPD at every field point (field canvas)
%    *_channels.png  EACH channel's MULTI-FIELD dW (one subplot per
%                    element x Zernike mode) -- the per-channel sensitivity
%    *_<rx>.mat      dwdzall (= dwdxall) + w0_stacked + indxall + ...
%
%  Per-field exit-pupil reset (reset_xp=true, the default) re-references each
%  field's nominal to its own chief ray (FEX) so the gross field tilt is
%  removed; a poke's own tilt is retained.  Requires a STOP + >3 elts.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX     = fullfile(here, 'e5hex1.in');  % self-contained demo Rx (point elsewhere
                                       % to run on your own system)
MODEL  = 128;           % model size (>= your aperture grid sampling)
NGRIDPTS = 63;         % ray-grid sampling override ([] = keep the .in value)
FOV    = 1e-4;          % half-field (rad) for the 4 corner field points
DELTA  = 1e-6;          % finite-difference step (Zernike coefficient)
KINDS  = {'monzern','zern'};  % subset of {'monzern','ffzern','zern'}
ZSTART = 4;             % lowest Zernike mode (4 skips piston/tip/tilt)
ZEND   = 9;             % highest Zernike mode -- this is the END mode, NOT a
                        % count: modes ZSTART:ZEND are taken (4:9 = Z4..Z9).
% =====================================================================

[~, rxstem] = fileparts(RX);
fprintf('=== dw/dz (Zernike) multi-field: %s (model %d) ===\n', rxstem, MODEL);
m   = macos.Session(MODEL);
out = macos.dw_dz_zernike_multi(m, RX, ...
    'ngridpts', NGRIDPTS, ...
    'field_x_rad', FOV, 'field_y_rad', FOV, ...
    'kinds', KINDS, 'zmode_start', ZSTART, 'n_zcoef', ZEND, 'delta', DELTA);

% ---- Figures -------------------------------------------------------
plot_opd_canvas(out, sprintf('dw/dz %s -- nominal OPD, %d fields', ...
    rxstem, numel(out.field_names)), here, ['dwdz_multi_' rxstem '_OPDall.png']);
plot_dw_channels(out, sprintf('dW/dz (Zernike) -- each mode, %d fields (%s)', ...
    numel(out.field_names), rxstem), here, ['dwdz_multi_' rxstem '_channels.png']);

% ---- Per-element single-page plots: center-field AND multi-field ----
plot_dw_per_element(out, 'center', here, ['dwdz_multi_' rxstem]);
plot_dw_per_element(out, 'multi',  here, ['dwdz_multi_' rxstem]);

% ---- Save canonical state-vector .mat ------------------------------
save_dw_multi(out, MODEL, fullfile(here, ['dwdz_multi_' rxstem '.mat']));
fprintf('=== dw/dz multi: %d channels x %d fields ===\n', ...
    numel(out.channel_names), numel(out.field_names));
