%ZEMAX_CASSEGRAIN  A Zemax telescope into the segmentation -> sensitivities -> MET chain.
%
% The bridge from a converted prescription to the design pipeline
% (design/runners/README.md).  Zemax (and CODE V) decks have no exit-pupil
% element, and every wavefront read in the chain takes the OPD at element
% nElt-1 -- so the first step after the converter is
% macos.design.add_exit_pupil, and from there the stage runners take the
% .in from stage to stage unchanged:
%
%   zmx2macos.py            zemax_cassegrain.zmx  ->  zemax_cassegrain.in
%   [1] add_exit_pupil      ->  zc_xp.in            (FP_return + ExitPupil before the FP)
%   [2] run_segmentation    ->  zc_hex1.in + Hx.m   (the primary -> 7 hex segments)
%   [3] run_sensitivities   ->  zc_hex1_sens.mat    (dwdx rigid, dwdz segment figure)
%   [4] run_met             ->  zc_hex1_met.in / _metopt.in / _met.mat (edge + laser truss)
%
% The telescope: the converter's synthetic f/6 Cassegrain (500 mm, primary
% R -1000 K -1 = the stop, secondary R -240 K -1.96 400 mm in front, image
% 200 mm behind the primary) with a 25 mm hole in the primary (Zemax OBSC),
% written by OpticStudio-style `.zmx` text.  Regenerate the deck with
%   python3 ../../../../rx_converter/zemax/zmx2macos.py zemax_cassegrain.zmx -o zemax_cassegrain.in
% (numpy only; see rx_converter/zemax/README.md).
%
% Modify the knobs and re-run.  Outputs land beside this script; the
% committed ones are the deck, the pupil deck and the report.  Needs the
% SegMirMaker binary (segmirmaker/README.md) and a built mmacos mex.

%% ------------------------------- knobs ---------------------------------
MODEL   = 256;            % engine model size (nGridpts 65 in the deck)
RINGS   = 1;  GRID = 'Hex';                 % 1 ring -> 7 hexagonal segments
GAP     = 5;              % inter-segment gap, mm (the deck is BaseUnits mm)
FOV     = 2.908882e-4;    % half-field for the Jacobians, rad (1 arcmin)
R_FID   = 45;             % fiducial ring on the secondary (55 mm semi-diameter), mm
RFID_GRID = [35 40 45];   % radii the layout optimizer may put the fiducials at, mm
MIN_SEP = 15;             % launcher separation gate, mm (the runner's 50 mm default
                          % is for metre-class segments; these are 167 mm across)
OPTIMIZE = false;         % met_layout_opt search: true costs about an hour at the
                          % default grids (4 sweeps x 68880 layouts, measured
                          % 2026-11-04) and did not beat the as-built truss here
                          % (the Zemax deck declares no aperture on the secondary:
                          % DIAM is a drawing size, so the optimizer cannot size
                          % the rim itself)
NAME    = 'zc';

here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mmacos_setup.m'));
rx0 = fullfile(here, 'zemax_cassegrain.in');

%% [1] the exit pupil the chain reads at
xp = macos.design.add_exit_pupil(rx0, fullfile(here, [NAME '_xp.in']), 'model_size', MODEL);

%% [2] segment the primary
seg = run_segmentation(xp.rx_out, 'elt', 1, 'rings', RINGS, 'grid', GRID, 'gap', GAP, ...
    'model_size', MODEL, 'out_dir', here, 'name', [NAME '_hex1']);

%% [3] sensitivities over centre + 4 corners
% The converter put the stop ON the primary (element-form ApStop=); segmenting
% that element drops it (segment_rx warns), so the stage is told the stop
% POSITION, the primary's vertex in object space.
STOP = [0 0 0];
sens = run_sensitivities(seg.in, 'fov_rad', FOV, 'channels', ["dwdx" "dwdz"], 'stop', STOP, ...
    'model_size', MODEL, 'out_dir', here, 'name', [NAME '_hex1']);

%% [4] metrology: fiducials on the secondary, launchers on the segments
% Bodies = the 7 segments + the secondary (hub).  A two-mirror telescope has
% no third OPTIC to carry an aft launcher ring (the image reference is a
% Return, not a body the sensitivities harvest perturbs); the e2e template's
% s5_met.m shows the aft leg on a fold.
hub = seg.seg.nseg + 1;          % the secondary follows the segments (read order)
met = run_met(seg.in, 'hx', seg.hx, 'hub', hub, 'r_fid', R_FID, 'jac', sens.mat, ...
    'min_sep', MIN_SEP, 'optimize', OPTIMIZE, 'opt', struct('rfid_grid', RFID_GRID), ...
    'model_size', MODEL, 'out_dir', here, 'name', [NAME '_hex1']);

%% summary
fprintf('\nzemax_cassegrain: %s -> %d elements with the pupil (EP radius %.4f mm) -> %d segments -> %s\n', ...
    rx0, xp.n_out, xp.ep_radius, seg.seg.nseg, met.met_in);
