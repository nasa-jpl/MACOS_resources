% run_dwdgrid_multi_singlesegbasis.m -- multi-field dw/d(grid-data) on a
% SEGMENTED aperture, using a SINGLE shared Zernike basis across all segments
% (macos.gs_zernike_segment_basis).
% =====================================================================
%  The grid-data wavefront-sensitivity Jacobian on a segmented primary,
%  where ONE reference segment's Gram-Schmidt-orthonormalized Zernike
%  basis is reused for EVERY segment -- each segment's local (xMon,yMon)
%  frame is already clocked to its orientation, so the same modes apply
%  in the local frame.  Cheaper than the per-segment build, exact when
%  the segments are congruent.  Contrast run_dwdgrid_multi_multisegbasis.
%
%  Canonical state-vector form:  wall = dwdgall * x + w0_stacked
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
RX         = '';         % <-- YOUR SEGMENTED .in FILE GOES HERE
MODEL      = 256;        % model size: >= the augmentation grid size NG
NG         = 128;        % augmentation nGridMat
NGRIDPTS   = [];         % ray-grid sampling override ([] = keep)
FOV        = 1e-5;       % half-field (rad) for the 4 corner field points
DELTA      = 1e-6;       % FD step (grid-map amplitude, BaseUnits)
PM_REF_ELT = 1;          % near-pupil Reference (footprint trace target)
REF_SEG    = 4;          % segment whose aperture defines the shared basis
MODES      = 4:15;       % Zernike figure modes
XP_METHOD  = 'sxp';      % per-field exit-pupil reset: 'fex' | 'sxp'
%
%  Bundled demo deck, used when RX is empty.  EXPLICIT path -- the
%  runner used to reach for examples/<its own name>/, so moving the
%  asset directory broke it silently.  It is one CONFIG line now.
DEMO_RX = fullfile(here, '..', 'templates', '50_sensitivities', 'run_dwdgrid_multi_singlesegbasis', ...
                   'SegDemo3conic.in');
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
    'zmodes_grid', MODES, 'delta_g', DELTA, 'grid_basis', 'single', ...
    'ref_seg', REF_SEG, 'pm_ref_elt', PM_REF_ELT, ...
    'reset_xp_method', XP_METHOD, ...
    'out_dir', here, 'name', ['dwdgrid_singlesegbasis_' rxstem]);
fprintf('=== dw/dgrid shared basis: %d channels x %d fields ===\n', ...
    numel(art.og.channel_names), size(art.og.field_table, 1));
