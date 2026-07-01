% run_dwdx_multi.m -- multi-field dw/dx RIGID-BODY sensitivity (example).
% =====================================================================
%  Multi-field rigid-body (6-DOF: Rx Ry Rz Tx Ty Tz) wavefront-sensitivity
%  Jacobian for every actual optic, in canonical state-vector form:
%
%      wall = dwdxall * x + w0_stacked
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
%                    element x DOF) -- the per-channel sensitivity
%    *_<rx>.mat      dwdxall + w0_stacked + indxall + ...
%
%  FIELD REFERENCING: dw_dx's FocalPlaneChannel re-references the exit pupil
%  per perturbation (fp_mode='track' -> sxp), so the off-axis field tilt is
%  already handled -- no separate per-field exit-pupil reset is needed here
%  (unlike dw_dz/dw_dsurf/dw_dgrid, which take reset_xp).
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX     = fullfile(here, 'e5hex1.in');  % self-contained demo Rx (point elsewhere
                                       % to run on your own system)
MODEL  = 128;           % model size (>= your aperture grid sampling)
FOV    = 1e-4;          % half-field (rad) for the 4 corner field points
DELTA  = 1e-8;          % finite-difference step (rigid-body)
DOFS   = (0:5).';       % 0=Rx 1=Ry 2=Rz 3=Tx 4=Ty 5=Tz  (subset allowed)
% =====================================================================

[~, rxstem] = fileparts(RX);
fprintf('=== dw/dx multi-field: %s (model %d) ===\n', rxstem, MODEL);
m   = macos.Session(MODEL);
out = macos.dw_dx_multi(m, RX, ...
    'field_x_rad', FOV, 'field_y_rad', FOV, ...
    'dofs', DOFS, 'delta', DELTA);

% ---- Figures -------------------------------------------------------
plot_opd_canvas(out, sprintf('dw/dx %s -- nominal OPD, %d fields', ...
    rxstem, numel(out.field_names)), here, ['dwdx_multi_' rxstem '_OPDall.png']);
plot_dw_channels(out, sprintf('dW/dx -- each channel, %d fields (%s)', ...
    numel(out.field_names), rxstem), here, ['dwdx_multi_' rxstem '_channels.png']);

% ---- Per-element single-page plots: center-field AND multi-field ----
plot_dw_per_element(out, 'center', here, ['dwdx_multi_' rxstem]);
plot_dw_per_element(out, 'multi',  here, ['dwdx_multi_' rxstem]);

% ---- Save canonical state-vector .mat ------------------------------
save_dw_multi(out, MODEL, fullfile(here, ['dwdx_multi_' rxstem '.mat']));
fprintf('=== dw/dx multi: %d channels x %d fields ===\n', ...
    numel(out.channel_names), numel(out.field_names));
