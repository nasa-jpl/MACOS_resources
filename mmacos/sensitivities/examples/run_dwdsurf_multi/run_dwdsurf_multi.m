% run_dwdsurf_multi.m -- multi-field dw/d(Kr,Kc) POWERED-SURFACE sensitivity (example).
% =====================================================================
%  Multi-field powered-surface wavefront-sensitivity Jacobian: per
%  (powered optic, parameter), parameter in {Kr (base radius), Kc (conic)},
%  for every Element= Reflector / Refractor with finite curvature.  Canonical
%  state-vector form:
%
%      wall = dwdsall (= dwdxall) * x + w0_stacked
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
%                    optic x {Kr,Kc}) -- the per-channel sensitivity
%    *_<rx>.mat      dwdsall (= dwdxall) + w0_stacked + indxall + ...
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
FOV    = 1e-4;          % half-field (rad) for the 4 corner field points
DELTA  = 1e-6;          % finite-difference step (Kr / Kc)
PARAMS = {'Kr','Kc'};   % subset of {'Kr','Kc'}
% =====================================================================

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

% ---- Per-element single-page plots: center-field AND multi-field ----
plot_dw_per_element(out, 'center', here, ['dwdsurf_multi_' rxstem]);
plot_dw_per_element(out, 'multi',  here, ['dwdsurf_multi_' rxstem]);

% ---- Save canonical state-vector .mat ------------------------------
save_dw_multi(out, MODEL, fullfile(here, ['dwdsurf_multi_' rxstem '.mat']));
fprintf('=== dw/d(Kr,Kc) multi: %d channels x %d fields ===\n', ...
    numel(out.channel_names), numel(out.field_names));
