% run_dwdgrid_multi.m -- multi-field dw/d(grid-data) sensitivity (example).
% =====================================================================
%  Thin driver over design/runners/run_sensitivities.m ('dwdgrid'
%  channel only) on the segmented e5hex1 fixture.  The runner
%  grid-augments the Rx in each segment's CLOCKED Mon frame
%  (macos.design.grid_augment_rx -- REPLACING any stale parent-frame
%  grid lines SegMirMaker replicates into segment blocks; poking those
%  paints about the aperture center and rank-collapses the Jacobian),
%  builds a per-segment Gram-Schmidt Zernike influence basis, and
%  harvests  wall = dwdgall * x + w0_stacked.  The grid SPAN comes
%  from the parent Aperture (span_frac 0.7) so the influence maps
%  FILL each segment -- never size it from lMon (wedge trap).
%  (Single source of truth -- the per-example runner copies retired
%  2026-07-19.)
%
%  SETUP: run `mmacos_setup` once per MATLAB session first.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX     = fullfile(here, 'e5hex1_grid.in');  % segmented Rx (existing grid
                                            % lines are RE-FRAMED by the runner)
MODEL  = 256;           % >= the augmentation grid size NG
NG     = 128;           % augmentation nGridMat
NGRIDPTS = 63;
FOV    = 1e-4;
MODES  = [4 5 6 7 8 9];    % Zernike figure modes per segment
% =====================================================================

[~, rxstem] = fileparts(RX);
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdgrid", ...
    'ngridpts', NGRIDPTS, 'model_size', MODEL, 'ng', NG, ...
    'zmodes_grid', MODES, ...
    'out_dir', here, 'name', ['dwdgrid_multi_' rxstem]);
fprintf('=== dw/dgrid multi: %d channels x %d fields ===\n', ...
    numel(art.og.channel_names), size(art.og.field_table, 1));
