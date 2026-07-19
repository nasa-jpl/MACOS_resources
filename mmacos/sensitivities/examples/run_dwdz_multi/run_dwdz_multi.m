% run_dwdz_multi.m -- multi-field dw/dz ZERNIKE-COEFFICIENT sensitivity (example).
% =====================================================================
%  Thin driver over design/runners/run_sensitivities.m ('dwdz' channel
%  only): per-(element, Zernike-mode) wavefront Jacobian for the
%  MonZern / element-Zern surface components, canonical form
%  wall = dwdzall * x + w0_stacked.  (Single source of truth -- the
%  per-example runner copies retired 2026-07-19.)
%
%  SETUP: run `mmacos_setup` once per MATLAB session first.
%  Self-contained: ships e5hex1.in beside the script.  TO RUN ON YOUR
%  OWN SYSTEM, point RX at your .in -- everything else is generic.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX     = fullfile(here, 'e5hex1.in');
MODEL  = 128;
NGRIDPTS = [];          % keep the .in ray grid
FOV    = 1e-4;
DELTA  = 1e-6;          % FD step (Zernike coefficient, BaseUnits)
KINDS  = {'monzern','zern'};  % subset of {'monzern','ffzern','zern'}
ZMODES = 4:9;           % modes (4 skips piston/tip/tilt)
% =====================================================================

[~, rxstem] = fileparts(RX);
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdz", ...
    'ngridpts', NGRIDPTS, 'model_size', MODEL, 'delta_z', DELTA, ...
    'zkinds', KINDS, 'zmodes_fig', ZMODES, ...
    'out_dir', here, 'name', ['dwdz_multi_' rxstem]);
fprintf('=== dw/dz multi: %d channels x %d fields ===\n', ...
    numel(art.oz.channel_names), size(art.oz.field_table, 1));
