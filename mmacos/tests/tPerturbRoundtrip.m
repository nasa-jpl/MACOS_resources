classdef tPerturbRoundtrip < matlab.unittest.TestCase
%TPERTURBROUNDTRIP  Regression for the CPERTURB_PROG round-trip
%   ULP-residual artifact (see macos/PLAN.md §0 follow-up).
%
%   Today's observed behavior:
%     - Translation +Δ then -Δ: bitwise identity in element state.
%     - Rotation +θ then -θ around a single axis: psi can land 1 ULP
%       off for some specific θ values (e.g. 1e-6, 3e-5) because
%       sin²(θ) + cos²(θ) ≠ 1 exactly in IEEE 754 at those values.
%     - All other θ values tested produce bitwise identity.
%
%   The tests below encode this as upper-bound invariants:
%   - Translation round-trip must be bitwise identity (assertEqual).
%   - Rotation round-trip must leave psi within 4*eps of its original
%     direction (one ULP per component at |psi|=1).  That holds today
%     AND after the prospective psi-renormalize fix in CPERTURB_PROG.

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
        TestElt   = 2     % M2 in Cass
    end

    properties
        rx_path
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.rx_path = rx_fixture_path(testCase.RxName);
            macos.init(testCase.ModelSize);
        end
    end

    methods (TestMethodSetup)
        function setupMethod(testCase)
            macos.load_rx(testCase.rx_path);
        end
    end

    methods (Test)
        function test_translation_roundtrip_bitwise_identity(testCase)
            % SI metres; perturb converts to BaseUnits internally.
            vpt0 = macos.get_elt_vpt(testCase.TestElt);
            psi0 = macos.get_elt_psi(testCase.TestElt);
            rpt0 = macos.get_elt_rpt(testCase.TestElt);

            macos.perturb(testCase.TestElt, ...
                'translation', [0; 0; 1e-9], 'frame', 'global');
            macos.perturb(testCase.TestElt, ...
                'translation', [0; 0; -1e-9], 'frame', 'global');

            testCase.verifyEqual(macos.get_elt_vpt(testCase.TestElt), vpt0);
            testCase.verifyEqual(macos.get_elt_psi(testCase.TestElt), psi0);
            testCase.verifyEqual(macos.get_elt_rpt(testCase.TestElt), rpt0);
        end

        function test_rotation_roundtrip_psi_within_4eps(testCase)
            % Sweep a set of θ values that includes the known-spiky 1e-6.
            % All must leave psi within 4*eps of original direction.
            psi0 = macos.get_elt_psi(testCase.TestElt);
            test_thetas = [1e-9, 1e-7, 1e-6, 3e-5, 1e-3, 0.1, 1];

            for k = 1:numel(test_thetas)
                th = test_thetas(k);
                macos.load_rx(testCase.rx_path);
                macos.perturb(testCase.TestElt, ...
                    'rotation', [th; 0; 0], 'frame', 'global');
                macos.perturb(testCase.TestElt, ...
                    'rotation', [-th; 0; 0], 'frame', 'global');
                psi_after = macos.get_elt_psi(testCase.TestElt);
                err = max(abs(psi_after - psi0));
                testCase.verifyLessThanOrEqual(err, 4 * eps(1.0), ...
                    sprintf('theta=%g: psi residual %g exceeds 4*eps', ...
                        th, err));
            end
        end

        function test_compound_perturb_path_F_is_exact(testCase)
            % Path F from the investigation: +1e-6, +1e-6, then -2e-6
            % inverse.  All three perturbs use the "spiky" 1e-6
            % magnitude, but the FP arithmetic accumulates to exact
            % identity (in contrast to two-step +θ, -θ).  Pinned here
            % so a future Qform / CPERTURB_PROG refactor that breaks it
            % surfaces immediately.
            psi0 = macos.get_elt_psi(testCase.TestElt);
            macos.perturb(testCase.TestElt, ...
                'rotation', [1e-6; 0; 0], 'frame', 'global');
            macos.perturb(testCase.TestElt, ...
                'rotation', [1e-6; 0; 0], 'frame', 'global');
            macos.perturb(testCase.TestElt, ...
                'rotation', [-2e-6; 0; 0], 'frame', 'global');
            testCase.verifyEqual(macos.get_elt_psi(testCase.TestElt), ...
                psi0);
        end
    end
end
