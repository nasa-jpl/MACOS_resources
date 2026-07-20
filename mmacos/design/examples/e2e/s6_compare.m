%S6_COMPARE  Stage 6: mmacos engine vs linear model, one poke at a time.
%
% Stage 6 of the end-to-end worked example -- a THIN DRIVER over the
% general compare stage runner (design/runners/run_compare.m).  Per
% Dave's spec (2026-07-19): poke each rigid-body DOF of the sensed
% bodies in turn (100 nrad / 100 nm) and display, side by side, the
% mmacos ENGINE response and the LINEAR MODEL prediction
%
%     w = dwdx*x + dwdz*z + dwdgrid*g + dwdu*u + w0
%     m = [l; e] = [dldx; dedx]*x + m0
%
% each graphic = the center-field OPD change above stacked bar charts
% of l, e_piston, e_gap, e_shear.  Each poke dwells 0.25 s on screen;
% every pair is saved as a frame and the sequence becomes
% s6_compare.gif.  The per-poke agreement table (engine vs linear =
% the linearization error at the poke size) goes to s6_report.txt,
% and the control subset dwdu (u = segments + SM) is exported in
% s6_compare.mat for the stage-7 simulator.
%
% Run AFTER s3_segmentation.m, s4_jacobians.m and s5_met.m.

addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/design/src'));
addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/design/runners'));
P = e2e_params();
here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
v = char(P.seg.variant);
rx  = fullfile(here, sprintf('e2e_%s.in', v));
hx  = fullfile(here, sprintf('e2e_%sHx.m', v));
jac = fullfile(here, 's4_jacobians.mat');
met = fullfile(here, sprintf('e2e_%s_met.mat', v));
assert(isfile(rx) && isfile(hx), 's6 needs e2e_%s.in + Hx -- run s3 first', v);
assert(isfile(jac), 's6 needs s4_jacobians.mat -- run s4 first');
assert(isfile(met), 's6 needs e2e_%s_met.mat -- run s5 first', v);

% batch runs (matlab -batch) have no screen: skip the dwell + live figs
vis = usejava('desktop');

art = run_compare(rx, 'hx', hx, 'jac', jac, 'met', met, ...
    'poke_rot', 1e-7, 'poke_trans', 1e-7, ...       % 100 nrad / 100 nm
    'dwell', 1.6, 'visible', vis, ...      % Dave 2026-07-19: 0.25 too fast
    'model_size', P.seg.model_size, ...
    'out_dir', here, 'name', 's6');

copyfile(art.report, fullfile(here, 's6_report.txt'));
fprintf('\nStage 6 complete: s6_compare.mat + s6_compare.gif + s6_report.txt\n');
fprintf('worst engine-vs-linear: w %.2e, l %.2e, e %.2e\n', ...
    art.worst.w, art.worst.l, art.worst.e);
fprintf('Next: s7 run_simulator (estimator/controller over a time history).\n');
