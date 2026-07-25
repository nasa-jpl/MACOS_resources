classdef tPolarization < matlab.unittest.TestCase
%TPOLARIZATION  Phase-1 polarization exposure (PLAN_POLARIZATION).
%   Exercises the engine polarization physics newly exposed through the
%   bindings: macos.polarization (pol_set/pol_get), macos.vector_diffraction
%   (vecdif_set), macos.coating (coat_set/coat_get, Model A), and
%   macos.ray_field (rayfield_get).  These are the state/round-trip/geometry
%   gates; the Jones-pupil physics lands in Phase 2.

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
        Det       = 6        % FocalPlane
        Fold      = 3        % a reflector to coat
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            macos.init(testCase.ModelSize);
        end
    end

    methods (TestMethodSetup)
        function loadRx(testCase)
            macos.load_rx(rx_fixture_path(testCase.RxName));
        end
    end

    methods (Test)
        % ---- polarization on/off + source state round-trip -----------
        function test_pol_on_off_roundtrip(testCase)
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
            s = macos.polarization();
            testCase.verifyTrue(s.on);
            testCase.verifyEqual(s.Ex, complex(1,0), 'AbsTol', 1e-12);
            testCase.verifyEqual(s.Ey, complex(0,0), 'AbsTol', 1e-12);

            macos.polarization('off');
            s = macos.polarization();
            testCase.verifyFalse(s.on);
        end

        function test_pol_circular_state(testCase)
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 1]);
            s = macos.polarization();
            testCase.verifyEqual(s.Ey, complex(0,1), 'AbsTol', 1e-12);
        end

        function test_pol_enables_vector(testCase)
            % ModelSize 128 has mWF=3, so POLARIZATION enables vector diff.
            macos.polarization('on');
            s = macos.polarization();
            testCase.verifyTrue(s.vector);
        end

        % ---- vector / scalar toggle + ordering guard -----------------
        function test_vecdif_toggle(testCase)
            macos.polarization('on');
            macos.vector_diffraction(false);
            s = macos.polarization();
            testCase.verifyFalse(s.vector);
            macos.vector_diffraction(true);
            s = macos.polarization();
            testCase.verifyTrue(s.vector);
        end

        function test_vector_requires_polarization(testCase)
            macos.polarization('off');
            % VECTOR with polarization off must error (not silently revert).
            % mexErrMsgTxt throws without a stable identifier, so just
            % assert that an error is raised.
            threw = false;
            try
                macos.vector_diffraction(true);
            catch
                threw = true;
            end
            testCase.verifyTrue(threw, ...
                'vector_diffraction(true) should error when polarization is off');
        end

        % ---- coating set/get round-trip (Model A) --------------------
        function test_coat_roundtrip_identity(testCase)
            n = [1.38 2.30];
            k = [0.0  0.10];
            t = [1.0e-7 5.0e-8];   % physical thickness (BaseUnits = m)
            macos.coating(testCase.Fold, 'index', n, 'extinc', k, ...
                          'thickness', t);
            s = macos.coating(testCase.Fold);
            testCase.verifyEqual(s.n_layer, 2);
            testCase.verifyEqual(s.index,     n, 'AbsTol', 1e-12);
            testCase.verifyEqual(s.extinc,    k, 'AbsTol', 1e-12);
            testCase.verifyEqual(s.thickness, t, 'RelTol', 1e-9);
        end

        function test_coat_query_uncoated(testCase)
            s = macos.coating(testCase.Det);   % focal plane, no coating
            testCase.verifyEqual(s.n_layer, 0);
        end

        function test_coat_bad_count_errors(testCase)
            testCase.verifyError(@() macos.coating(testCase.Fold, ...
                'index', zeros(1,20), 'extinc', zeros(1,20), ...
                'thickness', zeros(1,20)), 'macos:coating:badCount');
        end

        % ---- ray_field structure + status mask -----------------------
        function test_ray_field_shape_and_status(testCase)
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
            macos.trace(testCase.Det);
            rf = macos.ray_field(testCase.Det);
            N = testCase.ModelSize;
            testCase.verifySize(rf.Ex, [N N]);
            testCase.verifyTrue(isequal(size(rf.status), [N N]));
            % At least some rays are OK (status 0) after a normal trace.
            testCase.verifyGreaterThan(nnz(rf.status == 0), 0);
            % OK rays carry a non-zero field for an x-polarized source.
            okmask = rf.status == 0;
            testCase.verifyGreaterThan(max(abs(rf.Ex(okmask))), 0);
        end

        % ---- physics: a coating changes the polarized throughput -----
        function test_coating_changes_polarized_intensity(testCase)
            % Baseline polarized PSF (no coating).
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
            I0 = macos.intensity(testCase.Det);

            % Add an absorbing metal coating on the fold and re-trace.
            macos.coating(testCase.Fold, 'index', 1.2, 'extinc', 7.0, ...
                          'thickness', 0.1);
            I1 = macos.intensity(testCase.Det);

            % The coating must change the focal-plane intensity (proof the
            % coating reaches the polarized trace, not just storage).
            rel = norm(I1(:) - I0(:)) / max(norm(I0(:)), eps);
            testCase.verifyGreaterThan(rel, 1e-6);
        end
    end
end
