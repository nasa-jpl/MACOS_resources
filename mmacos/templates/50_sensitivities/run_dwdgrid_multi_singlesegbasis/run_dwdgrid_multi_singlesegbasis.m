% run_dwdgrid_multi_singlesegbasis.m -- dw/d(grid) with ONE shared basis.
% =====================================================================
%  Thin driver over design/runners/run_sensitivities.m ('dwdgrid',
%  grid_basis='single'): ONE reference segment's Gram-Schmidt Zernike
%  basis (macos.gs_zernike_segment_basis) shared by every segment --
%  each segment's local (xMon,yMon) frame is already clocked to its
%  orientation, so the same modes apply in the local frame.  Cheaper
%  than the per-segment build, exact when the segments are congruent.
%  Contrast run_dwdgrid_multi_multisegbasis (bespoke basis per
%  segment; the general case).
%  (Single source of truth -- the per-example runner copies retired
%  2026-07-19.)
%
%  SETUP: run `mmacos_setup` once per MATLAB session first.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX         = fullfile(here, 'SegDemo3conic.in');
MODEL      = 256;
NG         = 128;
NGRIDPTS   = [];
FOV        = 1e-5;
PM_REF_ELT = 1;         % near-pupil Reference (footprint trace target)
REF_SEG    = 4;         % segment whose aperture defines the shared basis
MODES      = 4:15;
XP_METHOD  = 'sxp';
% =====================================================================

[~, rxstem] = fileparts(RX);
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdgrid", ...
    'ngridpts', NGRIDPTS, 'model_size', MODEL, 'ng', NG, ...
    'zmodes_grid', MODES, 'grid_basis', 'single', 'ref_seg', REF_SEG, ...
    'pm_ref_elt', PM_REF_ELT, 'reset_xp_method', XP_METHOD, ...
    'out_dir', here, 'name', ['dwdgrid_singlesegbasis_' rxstem]);
fprintf('=== dw/dgrid shared basis: %d channels x %d fields ===\n', ...
    numel(art.og.channel_names), size(art.og.field_table, 1));
