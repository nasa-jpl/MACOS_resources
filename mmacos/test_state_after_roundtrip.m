function test_state_after_roundtrip()
%TEST_STATE_AFTER_ROUNDTRIP  Pin down the perturb-round-trip ULP residual.
%
%   Sister test to test_macos_pkg.m, kept for regression purposes.
%   Probes the FP-precision artifact in CPERTURB_PROG that shows up
%   when a +θ axis rotation is followed by its -θ inverse.
%
%   Finding: psi(3) lands 1 ULP off (-1 + 1.11e-16 instead of -1)
%   because sin(θ)² + cos(θ)² evaluates to 1 ± eps for some specific θ
%   values in IEEE 754.  Path F (cumulative +1e-6 + 1e-6 then -2e-6
%   inverse) takes a different FP path and lands back at exact -1.
%
%   Translation round-trips have no sin/cos and are bitwise exact.
%
%   The artifact is at the eps × |coord| floor — far below any
%   practical OPD signal — but propagates to ~3e-14 OPD residual
%   that briefly confused the +macos smoke-test author into thinking
%   perturb compositions were broken.  They are not.
%
%   Cosmetic follow-up (low priority): renormalize psi inside
%   CPERTURB_PROG after the rotation, so the round-trip is bitwise
%   identity and psi can't slowly drift under repeated perturbs.

rx_path = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
    'pymacos', 'tests', 'Rx', 'Rx_Cass_FarField.in');

addpath(fileparts(mfilename('fullpath')));

macos.init(128); macos.load_rx(rx_path);

% Snapshot clean state
vpt0 = macos.get_elt_vpt(2);
psi0 = macos.get_elt_psi(2);
rpt0 = macos.get_elt_rpt(2);
fprintf('Clean state of Elt 2:\n');
fprintf('  vpt = [%g %g %g]\n', vpt0);
fprintf('  psi = [%g %g %g]\n', psi0);
fprintf('  rpt = [%g %g %g]\n', rpt0);

% Path D: +1e-6, -1e-6
macos.load_rx(rx_path);
macos.perturb(2, 'rotation', [1e-6;0;0], 'frame', 'global');
macos.perturb(2, 'rotation', [-1e-6;0;0], 'frame', 'global');
vptD = macos.get_elt_vpt(2);
psiD = macos.get_elt_psi(2);
rptD = macos.get_elt_rpt(2);
fprintf('\nAfter D (+1e-6, -1e-6) round trip:\n');
fprintf('  Δvpt = [%g %g %g]\n', vptD - vpt0);
fprintf('  Δpsi = [%g %g %g]\n', psiD - psi0);
fprintf('  Δrpt = [%g %g %g]\n', rptD - rpt0);

% Path F: +1e-6, +1e-6, -2e-6
macos.load_rx(rx_path);
macos.perturb(2, 'rotation', [1e-6;0;0], 'frame', 'global');
macos.perturb(2, 'rotation', [1e-6;0;0], 'frame', 'global');
macos.perturb(2, 'rotation', [-2e-6;0;0], 'frame', 'global');
vptF = macos.get_elt_vpt(2);
psiF = macos.get_elt_psi(2);
rptF = macos.get_elt_rpt(2);
fprintf('\nAfter F (+1e-6, +1e-6, -2e-6):\n');
fprintf('  Δvpt = [%g %g %g]\n', vptF - vpt0);
fprintf('  Δpsi = [%g %g %g]\n', psiF - psi0);
fprintf('  Δrpt = [%g %g %g]\n', rptF - rpt0);

% Path G: nothing (sanity)
macos.load_rx(rx_path);
vptG = macos.get_elt_vpt(2);
psiG = macos.get_elt_psi(2);
fprintf('\nAfter G (just reload):\n');
fprintf('  Δvpt = [%g %g %g]\n', vptG - vpt0);
fprintf('  Δpsi = [%g %g %g]\n', psiG - psi0);

end
