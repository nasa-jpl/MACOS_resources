% run_dwdgrid_multi_multisegbasis.m -- multi-field dw/d(grid-data) on a
% SEGMENTED aperture, using a PER-SEGMENT Zernike basis (one basis PER segment,
% macos.segment_grid_basis).
% =====================================================================
%  The grid-data wavefront-sensitivity Jacobian on a segmented primary,
%  where EACH segment gets its OWN bespoke aperture mask + Zernike mode
%  stack in its own clocked frame.  The general case -- matters for the
%  clipped EDGE segments of a real aperture.  Contrast
%  run_dwdgrid_multi_singlesegbasis (one shared reference basis;
%  cheaper, exact when segments are congruent).
%
%  Canonical state-vector form:  wall = dwdgall * x + w0_stacked
%
%  TO RUN ON YOUR OWN SYSTEM: edit the CONFIG block.
%
%  NOTE (2026-07-19): this script is now a thin wrapper over the
%  sensitivity STAGE RUNNER design/runners/run_sensitivities.m (single
%  algorithm source).  The runner grid-augments the Rx in each
%  segment's CLOCKED Mon frame (replacing stale parent-frame grid
%  lines -- the e5-corpus central-dot trap) with the span sized from
%  the parent Aperture so the influence maps FILL each segment.
%  The CONFIG interface is unchanged.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'design', 'runners'));

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX            = '';         % <-- YOUR SEGMENTED .in FILE GOES HERE
MODEL         = 256;        % model size: >= the augmentation grid size NG
NG            = 128;        % augmentation nGridMat
NGRIDPTS      = [];         % ray-grid sampling override ([] = keep)
FOV           = 1e-5;       % half-field (rad) for the 4 corner field points
DELTA         = 1e-6;       % FD step (grid-map amplitude, BaseUnits)
PM_REF_ELT    = 1;          % near-pupil Reference (footprint trace target)
MODES         = 4:15;       % Zernike figure modes per segment
XP_METHOD     = 'sxp';      % per-field exit-pupil reset: 'fex' | 'sxp'
                            % (SegDemo3-style near-EP layouts need 'sxp')
%
%  Bundled demo deck, used when RX is empty.  EXPLICIT path -- the
%  runner used to reach for examples/<its own name>/, so moving the
%  asset directory broke it silently.  It is one CONFIG line now.
DEMO_RX = fullfile(here, 'examples', 'run_dwdgrid_multi_multisegbasis', ...
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
    'zmodes_grid', MODES, 'delta_g', DELTA, 'grid_basis', 'multi', ...
    'pm_ref_elt', PM_REF_ELT, 'reset_xp_method', XP_METHOD, ...
    'out_dir', here, 'name', ['dwdgrid_multisegbasis_' rxstem]);
fprintf('=== dw/dgrid per-segment basis: %d channels x %d fields ===\n', ...
    numel(art.og.channel_names), size(art.og.field_table, 1));
