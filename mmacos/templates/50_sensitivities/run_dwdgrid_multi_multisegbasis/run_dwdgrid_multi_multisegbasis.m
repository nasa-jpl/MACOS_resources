% run_dwdgrid_multi_multisegbasis.m -- dw/d(grid) with a PER-SEGMENT basis.
% =====================================================================
%  Thin driver over design/runners/run_sensitivities.m ('dwdgrid',
%  grid_basis='multi' -- the default): EACH segment gets its own
%  bespoke aperture mask + Zernike mode stack in its own clocked frame
%  (macos.segment_grid_basis).  The general case -- matters for the
%  clipped EDGE segments of a real aperture.  Contrast
%  run_dwdgrid_multi_singlesegbasis (one shared reference basis;
%  cheaper, exact when segments are congruent).
%  (Single source of truth -- the per-example runner copies retired
%  2026-07-19.)
%
%  SETUP: run `mmacos_setup` once per MATLAB session first.
%  SegDemo3: FEX collapses to a ~0.1 m EP sphere -> per-field
%  exit-pupil reset uses SXP.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX         = fullfile(here, 'SegDemo3conic.in');
MODEL      = 256;
NG         = 128;
NGRIDPTS   = [];
FOV        = 1e-5;
PM_REF_ELT = 1;         % near-pupil Reference (footprint trace target)
MODES      = 4:15;
XP_METHOD  = 'sxp';     % 'fex' | 'sxp' (near-EP layouts need 'sxp')
% =====================================================================

[~, rxstem] = fileparts(RX);
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdgrid", ...
    'ngridpts', NGRIDPTS, 'model_size', MODEL, 'ng', NG, ...
    'zmodes_grid', MODES, 'grid_basis', 'multi', ...
    'pm_ref_elt', PM_REF_ELT, 'reset_xp_method', XP_METHOD, ...
    'out_dir', here, 'name', ['dwdgrid_multisegbasis_' rxstem]);
fprintf('=== dw/dgrid per-segment basis: %d channels x %d fields ===\n', ...
    numel(art.og.channel_names), size(art.og.field_table, 1));
