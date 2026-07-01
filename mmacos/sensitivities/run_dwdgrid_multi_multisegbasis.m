% run_dwdgrid_multi_multisegbasis.m -- multi-field dw/d(grid-data) on a
% SEGMENTED aperture, using a PER-SEGMENT Zernike basis (one basis PER segment,
% macos.segment_grid_basis).
% =====================================================================
%  The grid-data (GMI pgrid) wavefront-sensitivity Jacobian on a segmented
%  primary, where EACH segment gets its OWN bespoke aperture mask + stack of
%  Zernike figure modes -- built in that segment's clocked (xData,yData) frame
%  by macos.segment_grid_basis.  Contrast the two segmented-aperture drivers:
%
%    multisegbasis (this):    one basis PER segment; the general case -- matters
%                             for the clipped EDGE segments of a real aperture.
%    singlesegbasis:          ONE reference basis shared by every segment
%                             (macos.gs_zernike_segment_basis); cheaper, exact
%                             when the segments are congruent.
%
%  Canonical state-vector form:  wall = dwdgall * x + w0_stacked
%
%  TO RUN ON YOUR OWN SYSTEM: edit the CONFIG block ("YOUR .in FILE GOES HERE").
%  Needs a SEGMENTED Rx with a near-pupil Reference (a valid trace target ahead
%  of the grid segments) + a STOP set.  Everything below CONFIG is generic.
% =====================================================================

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX            = '';         % <-- YOUR SEGMENTED .in FILE GOES HERE (absolute path)
MODEL         = 256;        % model size: >= the Rx's nGridMat
FOV           = 1e-5;       % half-field (rad) for the 4 corner field points
DELTA         = 1e-6;       % finite-difference step (grid-map amplitude, BaseUnits)
PM_REF_ELT    = 1;          % near-pupil Reference (trace target for the footprint)
MODES         = 4:15;       % Zernike figure modes per segment
ORTHOGONALIZE = true;       % true = Gram-Schmidt per segment; false = circular
ZERN_TYPE     = 'ansi';     % 'ansi' (engine ZerntoMon1/NormANSI) | 'noll'
XP_METHOD     = 'sxp';      % per-field exit-pupil reset: 'fex' | 'sxp'
                            % (SegDemo3-style near-EP layouts need 'sxp')
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'src'));     % +macos package + mmacos mex
addpath(here);                            % plot_* / save_dw_multi helpers

if isempty(RX)
    RX = fullfile(here, 'examples', 'run_dwdgrid_multi_multisegbasis', 'SegDemo3conic.in');
    fprintf('[demo] RX not set -- using bundled example: %s\n', RX);
end
[~, rxstem] = fileparts(RX);

fprintf('=== dw/d(grid) multi-field [per-segment basis]: %s (model %d) ===\n', ...
        rxstem, MODEL);
m = macos.Session(MODEL);

% Build the PER-SEGMENT influence basis: bespoke mask + Zernike modes in each
% segment's own clocked frame.  (Same struct gen_segment_gridmat saves to .mat.)
basis = macos.segment_grid_basis(m, RX, ...
    'pm_ref_elt', PM_REF_ELT, 'modes', MODES, ...
    'orthogonalize', ORTHOGONALIZE, 'zern_type', ZERN_TYPE);
fprintf('per-segment basis: %d segments x %d modes (N=%d grid, dx=%.4g)\n', ...
        numel(basis.seg), numel(basis.modes), basis.N, basis.gdx);

% Multi-field response.  SegDemo3: FEX collapses to a ~0.1 m EP sphere, so use
% SXP (EP radius = EP->FP distance) for the per-field exit-pupil reset.
out = macos.dw_dgrid_multi(m, RX, ...
    'field_x_rad', FOV, 'field_y_rad', FOV, ...
    'influence', basis, 'delta', DELTA, ...
    'reset_xp_method', XP_METHOD);

% ---- Figures -------------------------------------------------------
plot_opd_canvas(out, sprintf('dw/d(grid) %s -- nominal OPD, %d fields', ...
    rxstem, numel(out.field_names)), here, ['dwdgrid_multisegbasis_' rxstem '_OPDall.png']);
plot_dw_channels(out, sprintf('dW/d(grid) per-segment basis -- each (seg,mode), %d fields (%s)', ...
    numel(out.field_names), rxstem), here, ['dwdgrid_multisegbasis_' rxstem '_channels.png']);

% ---- Per-element single-page plots: center-field AND multi-field ----
plot_dw_per_element(out, 'center', here, ['dwdgrid_multisegbasis_' rxstem]);
plot_dw_per_element(out, 'multi',  here, ['dwdgrid_multisegbasis_' rxstem]);

% ---- Save canonical state-vector .mat ------------------------------
save_dw_multi(out, MODEL, fullfile(here, ['dwdgrid_multisegbasis_' rxstem '.mat']));
fprintf('=== dw/d(grid) per-segment: %d channels x %d fields ===\n', ...
    numel(out.channel_names), numel(out.field_names));
