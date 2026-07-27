function probe_sp_sign()
%PROBE_SP_SIGN  Reproducer for the reflected-p-hat / Fresnel-r_p sign conflict.
%
%   Run:  matlab -batch "addpath('<this dir>'); probe_sp_sign"
%   (or from the mmacos root, after mmacos_setup.m).
%
%   WHAT IT SHOWS.  A single on-axis mirror rotates a uniformly x-polarized
%   input into a ~50/50 x/y mixture in the GLOBAL frame; two on-axis mirrors
%   cancel it exactly.  A single mirror at <2 deg AOI cannot do that: the
%   physical cross-polarization from an isotropic surface is O(sin^2 beta)
%   with beta the local surface slope, i.e. ~1e-3 in amplitude here.
%
%   The observable is ray-side (RayE via macos.ray_field), so it is
%   independent of the diffraction layer, and it is measured on
%   Rx_Cass_FarField -- the same fixture the Phase-2 gates use.
%
%   WHY EVERY EXISTING GATE PASSES ANYWAY.  The defect acts as a reflection
%   of the transverse field about the local p-hat axis.  A reflection is an
%   involution (M^2 = I) AND is unitary, so:
%     * Rx_Cass_FarField has exactly TWO mirrors -> it cancels exactly;
%     * Rx_VecChain has no reflectors at all;
%     * the unitarity gate cannot see a unitary error;
%     * the Fresnel-analytic fold gate compares |RS/RP| and D (magnitudes)
%       and checks the RS/RP phase against an analytic RPa transcribed FROM
%       the engine's own expression -- circular with respect to this sign.
%
%   EXPECTED OUTPUT (model 256, gfortran, engine at pol-core 7bbaad7):
%       after 1 mirror : Py/Px = 1.0163e+00      <-- ~50/50, unphysical
%       after 2 mirrors: Py/Px = 7.0612e-07
%   With the standard Fresnel r_p restored in elemsub.F (the commented-out
%   "dcr's original" line, elemsub.F:454) these become
%       after 1 mirror : Py/Px = 2.0724e-04      <-- 4900x smaller
%       after 2 mirrors: Py/Px = 7.0612e-07      <-- BIT-IDENTICAL
%   i.e. the candidate fix changes the odd-mirror case by three and a half
%   orders of magnitude and leaves the even-mirror case untouched, which is
%   exactly why the change is invisible to the current suite.
%
%   See macos/REVIEW_POL_SP_SIGN_2026-07-27.md for the full analysis.

    here = fileparts(mfilename('fullpath'));
    rx   = fullfile(here, '..', '..', '..', 'pymacos', 'tests', 'Rx', ...
                    'Rx_Cass_FarField.in');
    if ~isfile(rx)
        error('probe_sp_sign:fixture', 'Rx_Cass_FarField.in not found at %s', rx);
    end

    macos.init(256);
    macos.load_rx(rx);
    macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    macos.vector_diffraction(true);

    % elt 1 = SecMirObs (Obscuring), 2 = Primary, 3 = Secondary.  So
    % ray_field(2) is the field after ONE mirror and ray_field(3) after TWO.
    fprintf('\n  mirrors    Py/Px        max|Ey/Ex|\n');
    fprintf(  '  -------    ----------   ----------\n');
    for e = [2 3]
        macos.trace(e);
        rf = macos.ray_field(e);
        ok = rf.status == 0;
        Px = sum(abs(rf.Ex(ok)).^2);
        Py = sum(abs(rf.Ey(ok)).^2);
        fprintf('  %7d    %10.4e   %10.4e\n', e - 1, Py/Px, ...
                max(abs(rf.Ey(ok) ./ rf.Ex(ok))));
    end

    % Radial independence: a physical AOI-driven effect must vanish on axis
    % and grow with radius.  This one does not, which is the second,
    % fixture-independent reason it cannot be physics.
    macos.trace(2);
    rf = macos.ray_field(2);
    ok = rf.status == 0;
    N  = size(rf.Ex, 1);
    c  = (N + 1) / 2;
    [jj, ii] = meshgrid(1:N, 1:N);
    rho = hypot(ii - c, jj - c);
    rmax = max(rho(ok));
    fprintf('\n  after ONE mirror, |Ey/Ex| vs radius (flat => not AOI-driven):\n');
    for f = [0.25 0.5 0.75 1.0]
        sel = ok & abs(rho - f*rmax) < 2;
        if nnz(sel) > 20
            fprintf('    rho = %5.1f px (n=%5d):  median |Ey/Ex| = %.4e\n', ...
                    f*rmax, nnz(sel), median(abs(rf.Ey(sel) ./ rf.Ex(sel))));
        end
    end
    fprintf('\n');
end
