% run_dwdgrid_multi_singlesegbasis.m -- multi-field dw/d(grid-data) on a
% SEGMENTED aperture, using a SINGLE shared Zernike basis across all segments
% (macos.gs_zernike_segment_basis).
% =====================================================================
%  The grid-data (GMI pgrid) wavefront-sensitivity Jacobian on a segmented
%  primary, where ONE reference segment's Gram-Schmidt-orthonormalized Zernike
%  basis is reused for EVERY segment -- each segment's local (xMon,yMon) frame
%  is already clocked to its orientation, so the same modes apply in the local
%  frame everywhere.  Contrast the two segmented-aperture drivers:
%
%    singlesegbasis (this):   ONE reference basis shared by every segment;
%                             cheaper, exact when the segments are congruent.
%    multisegbasis:           one bespoke basis PER segment
%                             (macos.segment_grid_basis); the general case --
%                             matters for the clipped EDGE segments of a real
%                             aperture.
%
%  Canonical state-vector form:  wall = dwdgall * x + w0_stacked
%
%  TO RUN ON YOUR OWN SYSTEM: edit the CONFIG block ("YOUR .in FILE GOES HERE").
%  Needs a SEGMENTED Rx with a near-pupil Reference (a valid trace target ahead
%  of the grid segments) + a STOP set.  gs_zernike_segment_basis uses Noll
%  `zernike_mode` from ~/matlab.  Everything below CONFIG is generic.
% =====================================================================

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX         = '';         % <-- YOUR SEGMENTED .in FILE GOES HERE (absolute path)
MODEL      = 256;        % model size: >= the Rx's nGridMat
NGRIDPTS   = [];         % ray-grid sampling override, e.g. 63 ([] = keep the .in value)
FOV        = 1e-5;       % half-field (rad) for the 4 corner field points
DELTA      = 1e-6;       % finite-difference step (grid-map amplitude, BaseUnits)
PM_REF_ELT = 1;          % near-pupil Reference (trace target for the footprint)
REF_SEG    = 4;          % reference segment whose aperture defines the basis
SEG_ELTS   = 2:8;        % ALL segment elements (Voronoi centres)
MODES      = 4:15;       % Zernike figure modes
XP_METHOD  = 'sxp';      % per-field exit-pupil reset: 'fex' | 'sxp'
                         % (SegDemo3-style near-EP layouts need 'sxp')
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'src'));     % +macos package + mmacos mex
addpath(here);                            % plot_* / save_dw_multi helpers

if isempty(RX)
    RX = fullfile(here, 'examples', 'run_dwdgrid_multi_singlesegbasis', 'SegDemo3conic.in');
    fprintf('[demo] RX not set -- using bundled example: %s\n', RX);
end
[~, rxstem] = fileparts(RX);

fprintf('=== dw/d(grid) multi-field [single shared basis]: %s (model %d) ===\n', ...
        rxstem, MODEL);
m = macos.Session(MODEL);

% Build ONE shared basis over the reference segment's aperture; it is applied
% to every segment (in that segment's own clocked local frame).
[INFL, ~, binfo] = macos.gs_zernike_segment_basis(m, RX, ...
    'pm_ref_elt', PM_REF_ELT, 'ref_seg', REF_SEG, ...
    'seg_elts', SEG_ELTS, 'modes', MODES);
fprintf('single shared GS basis: %d modes over %d-px aperture (ref seg elt %d)\n', ...
        numel(MODES), binfo.mask_px, binfo.ref_seg);

% Multi-field response.  SegDemo3: FEX collapses to a ~0.1 m EP sphere, so use
% SXP (EP radius = EP->FP distance) for the per-field exit-pupil reset.
out = macos.dw_dgrid_multi(m, RX, ...
    'ngridpts', NGRIDPTS, ...
    'field_x_rad', FOV, 'field_y_rad', FOV, ...
    'influence', INFL, 'delta', DELTA, ...
    'reset_xp_method', XP_METHOD);

% ---- Figures -------------------------------------------------------
plot_opd_canvas(out, sprintf('dw/d(grid) %s -- nominal OPD, %d fields', ...
    rxstem, numel(out.field_names)), here, ['dwdgrid_singlesegbasis_' rxstem '_OPDall.png']);
plot_dw_channels(out, sprintf('dW/d(grid) single shared basis -- each (seg,mode), %d fields (%s)', ...
    numel(out.field_names), rxstem), here, ['dwdgrid_singlesegbasis_' rxstem '_channels.png']);

% ---- Per-element single-page plots: center-field AND multi-field ----
plot_dw_per_element(out, 'center', here, ['dwdgrid_singlesegbasis_' rxstem]);
plot_dw_per_element(out, 'multi',  here, ['dwdgrid_singlesegbasis_' rxstem]);

% ---- Save canonical state-vector .mat ------------------------------
save_dw_multi(out, MODEL, fullfile(here, ['dwdgrid_singlesegbasis_' rxstem '.mat']));
fprintf('=== dw/d(grid) single shared basis: %d channels x %d fields ===\n', ...
    numel(out.channel_names), numel(out.field_names));
