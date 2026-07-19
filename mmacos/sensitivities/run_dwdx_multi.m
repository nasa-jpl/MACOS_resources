% run_dwdx_multi.m -- multi-field dw/dx RIGID-BODY sensitivity (GENERIC).
% =====================================================================
%  Multi-field rigid-body (6-DOF: Rx Ry Rz Tx Ty Tz) wavefront-sensitivity
%  Jacobian for every actual optic, in canonical state-vector form:
%
%      wall = dwdxall * x + w0_stacked
%
%  TO RUN ON YOUR OWN SYSTEM: edit the CONFIG block ("YOUR .in FILE GOES
%  HERE") -- everything below it is generic.
%
%  NOTE (2026-07-19): this script is now a thin wrapper over the
%  sensitivity STAGE RUNNER design/runners/run_sensitivities.m (single
%  algorithm source; per-element pages land in <name>_pages/, plots
%  piston-removed).  The CONFIG interface is unchanged.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'design', 'runners'));

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX     = '';            % <-- YOUR .in FILE GOES HERE (absolute path)
MODEL  = 128;           % model size (>= your aperture grid sampling)
NGRIDPTS = 63;          % ray-grid sampling override ([] = keep the .in value)
FOV    = 1e-4;          % half-field (rad) for the 4 corner field points
DELTA  = 1e-8;          % finite-difference step (rigid-body)
DOFS   = (0:5).';       % 0=Rx 1=Ry 2=Rz 3=Tx 4=Ty 5=Tz  (subset allowed)
% =====================================================================

if isempty(RX)
    RX = fullfile(here, 'examples', 'run_dwdx_multi', 'e5hex1.in');
    fprintf('[demo] RX not set -- using bundled example: %s\n', RX);
end
[~, rxstem] = fileparts(RX);
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdx", ...
    'ngridpts', NGRIDPTS, 'model_size', MODEL, 'delta_x', DELTA, ...
    'dofs', DOFS, 'out_dir', here, 'name', ['dwdx_multi_' rxstem]);
fprintf('=== dw/dx multi: %d channels x %d fields ===\n', ...
    numel(art.ox.channel_names), size(art.ox.field_table, 1));
