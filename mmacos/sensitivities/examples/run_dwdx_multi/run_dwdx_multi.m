% run_dwdx_multi.m -- multi-field dw/dx RIGID-BODY sensitivity (example).
% =====================================================================
%  Thin driver over the general sensitivity stage runner
%  design/runners/run_sensitivities.m ('dwdx' channel only): rigid-body
%  6-DOF (Rx Ry Rz Tx Ty Tz) wavefront Jacobian for every optic, in
%  canonical state-vector form  wall = dwdxall * x + w0_stacked.
%  (Single source of truth -- the per-example runner copies retired
%  2026-07-19 per the runners doctrine.)
%
%  SETUP: run `mmacos_setup` once per MATLAB session first.
%  Self-contained: ships e5hex1.in beside the script.  TO RUN ON YOUR
%  OWN SYSTEM, point RX at your .in -- everything else is generic.
%
%  Outputs (this directory): <name>_sens_report.txt + _sens.mat +
%  _opdall/_svspec/_dwdx_channels.png + per-element pages.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX     = fullfile(here, 'e5hex1.in');  % your .in goes here
MODEL  = 128;           % model size (>= your aperture grid sampling)
NGRIDPTS = 63;          % ray-grid override ([] = keep the .in value)
FOV    = 1e-4;          % half-field (rad) for the 4 corner field points
DELTA  = 1e-8;          % finite-difference step (rigid-body)
% =====================================================================

[~, rxstem] = fileparts(RX);
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdx", ...
    'ngridpts', NGRIDPTS, 'model_size', MODEL, 'delta_x', DELTA, ...
    'out_dir', here, 'name', ['dwdx_multi_' rxstem]);
fprintf('=== dw/dx multi: %d channels x %d fields ===\n', ...
    numel(art.ox.channel_names), size(art.ox.field_table, 1));
