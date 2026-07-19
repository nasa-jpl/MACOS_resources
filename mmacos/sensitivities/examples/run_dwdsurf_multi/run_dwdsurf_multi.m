% run_dwdsurf_multi.m -- multi-field dw/d(Kr,Kc) POWERED-SURFACE sensitivity (example).
% =====================================================================
%  Thin driver over design/runners/run_sensitivities.m ('dwdsurf'
%  channel only): per-(powered optic, {Kr, Kc}) radius/conic wavefront
%  Jacobian, canonical form  wall = dwdsall * x + w0_stacked.
%  (Single source of truth -- the per-example runner copies retired
%  2026-07-19.)
%
%  SETUP: run `mmacos_setup` once per MATLAB session first.
%  Self-contained: ships e5hex1.in beside the script.  TO RUN ON YOUR
%  OWN SYSTEM, point RX at your .in -- everything else is generic.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX     = fullfile(here, 'e5hex1.in');
MODEL  = 128;
NGRIDPTS = 63;
FOV    = 1e-4;
PARAMS = {'Kr','Kc'};   % surface parameters to perturb
% =====================================================================

[~, rxstem] = fileparts(RX);
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdsurf", ...
    'ngridpts', NGRIDPTS, 'model_size', MODEL, 'surf_params', PARAMS, ...
    'out_dir', here, 'name', ['dwdsurf_multi_' rxstem]);
fprintf('=== dw/dsurf multi: %d channels x %d fields ===\n', ...
    numel(art.os.channel_names), size(art.os.field_table, 1));
