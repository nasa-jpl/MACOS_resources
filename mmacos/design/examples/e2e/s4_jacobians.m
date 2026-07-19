%S4_JACOBIANS  Stage 4: sensitivity Jacobians on the segmented system.
%
% Stage 4 of the end-to-end worked example -- now a THIN DRIVER over
% the general stage runner design/runners/run_sensitivities.m (the
% runners doctrine, Dave 2026-07-19: the product is a small set of
% reusable stage runners handing the .in from stage to stage; this
% script only wires the e2e parameters and the s5 handoff).
%
% The runner harvests the three channels on e2e_<P.seg.variant>.in
% over the +-2' science field (center + 4 corners), each in canonical
% state-vector form  wall = J*x + w0_stacked:
%   dwdx     rigid-body 6-DOF per element (LOCAL/TElt triads)
%   dwdz     segment-LOCAL MonZernike modes 4..11 (kinds={'monzern'})
%   dwdgrid  per-segment G-S Zernike influence pokes on the
%            grid-augmented Rx (grids in the CLOCKED Mon frames --
%            macos.design.grid_augment_rx)
% and emits the conditioning report (all-column AND segment-only),
% per-segment column norms, the nominal-OPD canvas, SV spectra,
% per-channel overviews, and the e5-style PER-ELEMENT pages.
%
% Artifacts (beside this script): s4_sens_report.txt (+ s4_report.txt
% alias), s4_sens.mat, s4_grid.in + flat256.txt, s4_*.png, and the s5
% handoff s4_jacobians.mat (ox/oz/og + P -- consumed by s5_met.m).
%
% Run AFTER s3_segmentation.m.

addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/design/src'));
addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/design/runners'));
addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/sensitivities'));
P = e2e_params();
here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
v  = char(P.seg.variant);
rx = fullfile(here, sprintf('e2e_%s.in', v));
assert(isfile(rx), 's4 needs e2e_%s.in -- run s3_segmentation.m first', v);

FOV = P.inst.fov_arcmin * pi/180/60;      % +-2' corners (science patch)

art = run_sensitivities(rx, ...
    'fov_rad', FOV, ...
    'zmodes_fig', 4:11, ...               % dwdz figure modes / segment
    'zmodes_grid', 4:9, ...               % dwdgrid pokes / segment
    'ng', 256, ...                        % grid size (model 512)
    'model_size', P.seg.model_size, ...
    'out_dir', here, 'name', 's4');

% s5 handoff: run_met consumes s4_jacobians.mat (ox/oz/og + P)
ox = art.ox;  oz = art.oz;  og = art.og; %#ok<NASGU>
save(fullfile(here, 's4_jacobians.mat'), 'ox', 'oz', 'og', 'P', '-v7.3');
copyfile(fullfile(here, 's4_sens_report.txt'), ...
         fullfile(here, 's4_report.txt'));
fprintf('\nStage 4 complete: s4_jacobians.mat + s4_report.txt\n');
fprintf('Next: s5_met.m (shape-class launcher patterns; dedx/dldx join dwdx here).\n');
