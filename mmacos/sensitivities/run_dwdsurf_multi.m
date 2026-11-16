% run_dwdsurf_multi.m -- multi-field dw/d(Kr,Kc) POWERED-SURFACE sensitivity (GENERIC).
% =====================================================================
%  Multi-field powered-surface wavefront-sensitivity Jacobian: per
%  (powered optic, parameter), parameter in {Kr (base radius), Kc (conic)}.
%  Canonical state-vector form:
%
%      wall = dwdsall (= dwdxall) * x + w0_stacked
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
NGRIDPTS = 63;          % ray-grid sampling override ([] = keep the .in value)
FOV    = 1e-4;          % half-field (rad) for the 4 corner field points
PARAMS = {'Kr','Kc'};   % surface parameters to perturb
%
%  Bundled demo deck, used when RX is empty.  EXPLICIT path -- the
%  runner used to reach for examples/<its own name>/, so moving the
%  asset directory broke it silently.  It is one CONFIG line now.
DEMO_RX = fullfile(here, '..', 'templates', '50_sensitivities', 'run_dwdsurf_multi', ...
                   'e5hex1.in');
% =====================================================================

if isempty(RX)
    RX = DEMO_RX;
    fprintf('[demo] RX not set -- using bundled example: %s\n', RX);
end
assert(isfile(RX), 'run_dwd:noDeck', ...
    'prescription not found: %s\n(set RX, or fix DEMO_RX in the CONFIG block)', RX);
[~, rxstem] = fileparts(RX);
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdsurf", ...
    'ngridpts', NGRIDPTS, 'model_size', MODEL, 'surf_params', PARAMS, ...
    'out_dir', here, 'name', ['dwdsurf_multi_' rxstem]);
fprintf('=== dw/dsurf multi: %d channels x %d fields ===\n', ...
    numel(art.os.channel_names), size(art.os.field_table, 1));
