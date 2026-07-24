classdef tBeam < matlab.unittest.TestCase
%TBEAM  macos.beam() -- source amplitude (apodization) shaping.
%   Exercises the BEAM engine command exposed via beam_set/beam_get:
%   set each beam type, verify the getter round-trips the profile, and
%   confirm a GAUSSIAN beam actually re-weights the traced intensity
%   relative to UNIFORM (i.e. the profile reaches the diffraction grid,
%   not just the state variables).

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
        Det       = 6        % FocalPlane
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
        function test_uniform_roundtrip(testCase)
            macos.beam('uniform');
            s = macos.beam();
            testCase.verifyEqual(s.type, 'uniform');
        end

        function test_gaussian_roundtrip(testCase)
            macos.beam('gaussian', 'waist', [12 9]);
            s = macos.beam();
            testCase.verifyEqual(s.type, 'gaussian');
            testCase.verifyEqual(s.waist, [12 9], 'AbsTol', 1e-9);
        end

        function test_gaussian_scalar_waist_broadcasts(testCase)
            macos.beam('gaussian', 'waist', 8);
            s = macos.beam();
            testCase.verifyEqual(s.waist, [8 8], 'AbsTol', 1e-9);
        end

        function test_cos_roundtrip(testCase)
            macos.beam('cos', 'radius', 10, 'power', 3);
            s = macos.beam();
            testCase.verifyEqual(s.type, 'cos');
            testCase.verifyEqual(s.waist(1), 10, 'AbsTol', 1e-9);  % rxBeam
            testCase.verifyEqual(s.power, 3, 'AbsTol', 1e-9);
        end

        function test_dipole_roundtrip(testCase)
            macos.beam('dipole');
            s = macos.beam();
            testCase.verifyEqual(s.type, 'dipole');
        end

        function test_missing_waist_errors(testCase)
            testCase.verifyError(@() macos.beam('gaussian'), ...
                'macos:beam:missingWaist');
        end

        function test_gaussian_reweights_intensity(testCase)
            % A GAUSSIAN beam must change the traced focal-plane
            % intensity relative to UNIFORM -- proof the profile reaches
            % the diffraction grid, not just src_mod state.  Use a waist
            % well inside the aperture so the apodization is pronounced.
            macos.beam('uniform');
            macos.trace(testCase.Det);
            I_uni = macos.intensity(testCase.Det);
            I_uni = I_uni / max(I_uni(:));

            ape = macos.get_src_size().aperture;
            macos.beam('gaussian', 'waist', 0.3 * ape);
            macos.trace(testCase.Det);
            I_g = macos.intensity(testCase.Det);
            I_g = I_g / max(I_g(:));

            rel = max(abs(I_g(:) - I_uni(:)));
            testCase.verifyGreaterThan(rel, 1e-3, ...
                'Gaussian beam did not change the intensity vs uniform.');
        end
    end
end
