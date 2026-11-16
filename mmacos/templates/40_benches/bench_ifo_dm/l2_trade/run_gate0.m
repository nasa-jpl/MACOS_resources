% run_gate0.m -- Gate 0 for the L2 pupil-relay trade (PLAN sec.4).
%   The metric harness, run on the UNMODIFIED singlet rig, must reproduce
%   the phase-1 committed numbers before any design work:
%     M1 residual   6.758 +/- 0.1 nm rms   (detector-mode retrace term)
%     pupil map     mag 0.8101, anam ~0%, rot 180.000 deg, nl 0.0205 mm rms
%   plus the generated DM map must equal the committed ../dm_map.txt.
%   On failure: reconcile against ../run_final.log / ../bench_ifo_dm.mat --
%   do NOT proceed on a broken harness.
%
%   Run:  matlab -batch "run('.../run_gate0.m'); exit(0)"

addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/src'));
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end

out = ifo_l2_metric({}, 'workdir', here);

% DM map identity vs the committed phase-1 map
M1c = macos.read_grid_file(fullfile(here, '..', 'dm_map.txt'));
M1g = macos.read_grid_file(fullfile(here, 'dm_p50_s7_c.txt'));
dmax = max(abs(M1c(:) - M1g(:)));

fprintf('\n=== GATE 0 ===\n');
chk = @(name, val, ref, tol) fprintf('  %-28s %12.6g  (ref %g, tol %g)  %s\n', ...
    name, val, ref, tol, string(ternary(abs(val-ref) <= tol, 'PASS', 'FAIL')));
chk('M1 residual (nm rms)', out.m1_resid_nm, 6.758, 0.1);
chk('M1 corr',              out.m1_corr,     0.935135, 0.005);
chk('map mag',              out.map.mag,     0.8101, 0.001);
chk('map |rot| (mrad)',     abs(out.map.rot_mrad), 3141.593, 1.0);
chk('map nonlinear (mm rms)', out.map.nl_rms_mm, 0.0205, 0.001);
chk('DM map max|diff| (mm)', dmax, 0, 1e-15);
g0 = abs(out.m1_resid_nm - 6.758) <= 0.1 && abs(out.map.mag - 0.8101) <= 0.001 ...
    && abs(out.map.nl_rms_mm - 0.0205) <= 0.001 && dmax <= 1e-15 && out.pass;
fprintf('  guards overall: %s\n', string(ternary(out.pass, 'PASS', 'FAIL')));
fprintf('GATE 0: %s\n', string(ternary(g0, 'PASS -- proceed to mechanism analysis', ...
    'FAIL -- reconcile against ../run_final.log before any design work')));
save(fullfile(here, 'gate0.mat'), 'out', 'g0');

function y = ternary(c, a, b)
    if c, y = a; else, y = b; end
end
