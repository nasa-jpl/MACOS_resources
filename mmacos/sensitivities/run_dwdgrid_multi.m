% run_dwdgrid_multi.m -- multi-field dw/d(grid-data) sensitivity (GENERIC).
% =====================================================================
%  Multi-field GRID-DATA (the GMI pgrid channel) wavefront-sensitivity
%  Jacobian: each influence-function "poke" ADDED to a surface's grid
%  data has a wavefront sensitivity dW/d(poke).  Canonical form:
%
%      wall = dwdgall (= dwdxall) * x + w0_stacked
%
%  TO RUN ON YOUR OWN SYSTEM: edit the CONFIG block.
%
%  NOTE (2026-07-19): this script is now a thin wrapper over the
%  sensitivity STAGE RUNNER design/runners/run_sensitivities.m (single
%  algorithm source).  Default (INFL empty): the runner grid-augments
%  the Rx in each segment's CLOCKED Mon frame (replacing stale
%  parent-frame grid lines -- the e5-corpus central-dot trap), span
%  from the parent Aperture so the maps FILL each segment, and builds
%  a per-segment Gram-Schmidt basis over MODES.  With INFL set (e.g.
%  DM actuator maps), your maps are used VERBATIM on the Rx's
%  EXISTING grids -- no augmentation.  CONFIG interface unchanged.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'design', 'runners'));

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX     = '';            % <-- YOUR .in FILE GOES HERE (absolute path)
MODEL  = 256;           % model size: >= the grid size in play
NG     = 128;           % augmentation nGridMat (INFL empty)
NGRIDPTS = [];          % ray-grid sampling override ([] = keep the .in value)
FOV    = 1e-4;          % half-field (rad) for the 4 corner field points
DELTA  = 1e-6;          % finite-difference step (grid-map amplitude, BaseUnits)
ZMODES = [4 5 6 7 8 9]; % default poke shapes: MACOS ANSI Zernike indices
INFL   = [];            % optional [N x N x K] influence maps (DM actuators,
                        % measured figure...) -- overrides ZMODES when set
%
%  Bundled demo deck, used when RX is empty.  EXPLICIT path -- the
%  runner used to reach for examples/<its own name>/, so moving the
%  asset directory broke it silently.  It is one CONFIG line now.
DEMO_RX = fullfile(here, 'examples', 'run_dwdgrid_multi', ...
                   'e5hex1_grid.in');
% =====================================================================

if isempty(RX)
    RX = DEMO_RX;
    fprintf('[demo] RX not set -- using bundled example: %s\n', RX);
end
assert(isfile(RX), 'run_dwd:noDeck', ...
    'prescription not found: %s\n(set RX, or fix DEMO_RX in the CONFIG block)', RX);
[~, rxstem] = fileparts(RX);
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdgrid", ...
    'ngridpts', NGRIDPTS, 'model_size', MODEL, 'ng', NG, ...
    'zmodes_grid', ZMODES, 'delta_g', DELTA, 'influence', INFL, ...
    'out_dir', here, 'name', ['dwdgrid_multi_' rxstem]);
fprintf('=== dw/dgrid multi: %d channels x %d fields ===\n', ...
    numel(art.og.channel_names), size(art.og.field_table, 1));
