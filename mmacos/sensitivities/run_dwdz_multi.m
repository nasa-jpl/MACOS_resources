% run_dwdz_multi.m -- multi-field dw/dz ZERNIKE-COEFFICIENT sensitivity (GENERIC).
% =====================================================================
%  Multi-field Zernike-coefficient wavefront-sensitivity Jacobian: per
%  (element, Zernike mode), for the MonZern / element-Zern surface
%  components.  Canonical state-vector form:
%
%      wall = dwdzall (= dwdxall) * x + w0_stacked
%
%  TO RUN ON YOUR OWN SYSTEM: edit the CONFIG block.
%
%  NOTE (2026-07-19): this script is now a thin wrapper over the
%  sensitivity STAGE RUNNER design/runners/run_sensitivities.m (single
%  algorithm source).  The CONFIG interface is unchanged.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'design', 'runners'));

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX     = '';            % <-- YOUR .in FILE GOES HERE (absolute path)
MODEL  = 128;           % model size (>= your aperture grid sampling)
NGRIDPTS = [];          % ray-grid sampling override ([] = keep the .in value)
FOV    = 1e-4;          % half-field (rad) for the 4 corner field points
DELTA  = 1e-6;          % finite-difference step (Zernike coefficient)
KINDS  = {'monzern','zern'};  % subset of {'monzern','ffzern','zern'}
ZSTART = 4;             % lowest Zernike mode (4 skips piston/tip/tilt)
ZEND   = 9;             % highest Zernike mode (END mode, not a count)
% =====================================================================

if isempty(RX)
    RX = fullfile(here, 'examples', 'run_dwdz_multi', 'e5hex1.in');
    fprintf('[demo] RX not set -- using bundled example: %s\n', RX);
end
[~, rxstem] = fileparts(RX);
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdz", ...
    'ngridpts', NGRIDPTS, 'model_size', MODEL, 'delta_z', DELTA, ...
    'zkinds', KINDS, 'zmodes_fig', ZSTART:ZEND, ...
    'out_dir', here, 'name', ['dwdz_multi_' rxstem]);
fprintf('=== dw/dz multi: %d channels x %d fields ===\n', ...
    numel(art.oz.channel_names), size(art.oz.field_table, 1));
