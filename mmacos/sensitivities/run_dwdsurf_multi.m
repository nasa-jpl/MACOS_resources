% run_dwdsurf_multi.m -- multi-field dw/d(Kr,Kc) POWERED-SURFACE sensitivity (GENERIC).
% =====================================================================
%  Multi-field powered-surface wavefront-sensitivity Jacobian: per
%  (powered optic, parameter), parameter in {Kr (base radius), Kc (conic)},
%  for every Element= Reflector / Refractor with finite curvature.  Canonical
%  state-vector form:
%
%      wall = dwdsall (= dwdxall) * x + w0_stacked
%
%  TO RUN ON YOUR OWN SYSTEM: edit the CONFIG block ("YOUR .in FILE GOES
%  HERE") -- everything below it is generic.
%
%  Outputs (this directory):
%    *_OPDall.png    nominal OPD at every field point (field canvas)
%    *_channels.png  EACH channel's MULTI-FIELD dW (one subplot per
%                    optic x {Kr,Kc}) -- the per-channel sensitivity
%    *_<rx>.mat      dwdsall (= dwdxall) + w0_stacked + indxall + ...
%
%  Per-field exit-pupil reset (reset_xp=true, the default) re-references each
%  field's nominal to its own chief ray (FEX) so the gross field tilt is
%  removed; a poke's own tilt is retained.  Requires a STOP + >3 elts.
% =====================================================================

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX     = '';            % <-- YOUR .in FILE GOES HERE (absolute path)
MODEL  = 128;           % model size (>= your aperture grid sampling)
FOV    = 1e-4;          % half-field (rad) for the 4 corner field points
DELTA  = 1e-6;          % finite-difference step (Kr / Kc)
PARAMS = {'Kr','Kc'};   % subset of {'Kr','Kc'}
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'src'));     % +macos package + mmacos mex
addpath(here);                            % plot_* / save_dw_multi helpers

if isempty(RX)
    RX = fullfile(here, '..', 'examples', 'sensitivities', 'e5hex1', 'e5hex1.in');
    fprintf('[demo] RX not set -- using bundled example: %s\n', RX);
end
[~, rxstem] = fileparts(RX);

fprintf('=== dw/d(Kr,Kc) multi-field: %s (model %d) ===\n', rxstem, MODEL);
m   = macos.Session(MODEL);
out = macos.dw_dsurf_multi(m, RX, ...
    'field_x_rad', FOV, 'field_y_rad', FOV, ...
    'params', PARAMS, 'delta', DELTA);

% ---- Figures -------------------------------------------------------
plot_opd_canvas(out, sprintf('dw/d(Kr,Kc) %s -- nominal OPD, %d fields', ...
    rxstem, numel(out.field_names)), here, ['dwdsurf_multi_' rxstem '_OPDall.png']);
plot_dw_channels(out, sprintf('dW/d(Kr,Kc) -- each param, %d fields (%s)', ...
    numel(out.field_names), rxstem), here, ['dwdsurf_multi_' rxstem '_channels.png']);

% ---- Save canonical state-vector .mat ------------------------------
save_dw_multi(out, MODEL, fullfile(here, ['dwdsurf_multi_' rxstem '.mat']));
fprintf('=== dw/d(Kr,Kc) multi: %d channels x %d fields ===\n', ...
    numel(out.channel_names), numel(out.field_names));
