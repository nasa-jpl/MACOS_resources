% run_dwdz_multi.m -- multi-field dw/dz ZERNIKE-COEFFICIENT sensitivity (GENERIC).
% =====================================================================
%  Multi-field Zernike-coefficient wavefront-sensitivity Jacobian: per
%  (element, Zernike mode), for the MonZern / FFZern / element-Zern surface
%  components.  Canonical state-vector form:
%
%      wall = dwdzall (= dwdxall) * x + w0_stacked
%
%  TO RUN ON YOUR OWN SYSTEM: edit the CONFIG block ("YOUR .in FILE GOES
%  HERE") -- everything below it is generic.
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

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX     = '';            % <-- YOUR .in FILE GOES HERE (absolute path)
MODEL  = 128;           % model size (>= your aperture grid sampling)
FOV    = 1e-4;          % half-field (rad) for the 4 corner field points
DELTA  = 1e-6;          % finite-difference step (Zernike coefficient)
KINDS  = {'monzern','zern'};  % subset of {'monzern','ffzern','zern'}
ZSTART = 4;             % lowest Zernike mode (4 skips piston/tip/tilt)
ZEND   = 9;             % highest Zernike mode -- this is the END mode, NOT a
                        % count: modes ZSTART:ZEND are taken (4:9 = Z4..Z9).
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'src'));     % +macos package + mmacos mex
addpath(here);                            % plot_* / save_dw_multi helpers

if isempty(RX)
    RX = fullfile(here, '..', 'examples', 'sensitivities', 'e5hex1', 'e5hex1.in');
    fprintf('[demo] RX not set -- using bundled example: %s\n', RX);
end
[~, rxstem] = fileparts(RX);

fprintf('=== dw/dz (Zernike) multi-field: %s (model %d) ===\n', rxstem, MODEL);
m   = macos.Session(MODEL);
out = macos.dw_dz_zernike_multi(m, RX, ...
    'field_x_rad', FOV, 'field_y_rad', FOV, ...
    'kinds', KINDS, 'zmode_start', ZSTART, 'n_zcoef', ZEND, 'delta', DELTA);

% ---- Figures -------------------------------------------------------
plot_opd_canvas(out, sprintf('dw/dz %s -- nominal OPD, %d fields', ...
    rxstem, numel(out.field_names)), here, ['dwdz_multi_' rxstem '_OPDall.png']);
plot_dw_channels(out, sprintf('dW/dz (Zernike) -- each mode, %d fields (%s)', ...
    numel(out.field_names), rxstem), here, ['dwdz_multi_' rxstem '_channels.png']);

% ---- Save canonical state-vector .mat ------------------------------
save_dw_multi(out, MODEL, fullfile(here, ['dwdz_multi_' rxstem '.mat']));
fprintf('=== dw/dz multi: %d channels x %d fields ===\n', ...
    numel(out.channel_names), numel(out.field_names));
