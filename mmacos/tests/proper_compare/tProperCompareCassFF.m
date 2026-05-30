classdef tProperCompareCassFF < matlab.unittest.TestCase
%TPROPERCOMPARECASSFF  Port of pymacos test_cass_ff.py (Phase 1).
%   First PROPER-comparison slice in mmacos.  Runs the Cass-FF
%   geometry through both PROPER and mmacos and asserts agreement at
%   the focal plane.  Mirrors pymacos
%   tests/proper_compare/test_cass_ff.py:
%       test_proper_cass_ff_runs       PROPER-side sanity
%       test_macos_cass_ff_runs        mmacos-side sanity
%       test_compare_cass_ff_psf       compare PSFs (analytical PROPER aperture)
%       test_compare_cass_ff_psf_with_opd  compare with macos OPD pass-through
%
%   Requires MATLAB PROPER v3.3.1 on the path (run_mmacos_tests.sh
%   adds it automatically).  Requires Rx_Cass_FarField.in in the
%   pymacos Rx corpus.

    properties
        results_dir
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            % Sanity: prop_begin must be resolvable.
            assert(exist('prop_begin', 'file') == 2, ...
                ['PROPER MATLAB not on path; add ~/dev/proper_matlab/' ...
                 ' via addpath or run via run_mmacos_tests.sh.']);
            % Per-phase results dir for comparison PNGs (analog of
            % pymacos's results_dir_phase1).
            testCase.results_dir = fullfile(fileparts( ...
                mfilename('fullpath')), 'results', 'phase1');
        end
    end

    methods (Test)
        function test_proper_cass_ff_runs(testCase)
            % PROPER side produces a centered obstructed-aperture PSF.
            geom = geometries.cass_farfield();
            [intensity, sampling] = proper_run_cass_ff(geom);

            testCase.verifyEqual(ndims(intensity), 2);
            testCase.verifyEqual(size(intensity), [geom.proper_grid_n, geom.proper_grid_n]);
            testCase.verifyTrue(all(isfinite(intensity(:))));
            testCase.verifyGreaterThan(max(intensity(:)), 0);

            [~, idx] = max(intensity(:));
            [py, px] = ind2sub(size(intensity), idx);
            cy = geom.proper_grid_n/2 + 1;     % MATLAB 1-based center
            cx = cy;
            testCase.verifyLessThanOrEqual(abs(py - cy), 1);
            testCase.verifyLessThanOrEqual(abs(px - cx), 1);
            testCase.verifyEqual(sampling, geom.dx_focal_m, 'RelTol', 1e-3);
        end

        function test_macos_cass_ff_runs(testCase)
            % macos side produces a 512x512 focal-plane intensity array
            % with peak at array center, matching pymacos's reference
            % values (Peak=3.236e5, Sum=1.802e6 at model_size=512).
            geom = geometries.cass_farfield();
            [intensity, ~] = macos_run_cass_ff(geom);

            testCase.verifyEqual(ndims(intensity), 2);
            testCase.verifyEqual(size(intensity), [geom.macos_size, geom.macos_size]);
            testCase.verifyTrue(all(isfinite(intensity(:))));

            [~, idx] = max(intensity(:));
            [py, px] = ind2sub(size(intensity), idx);
            cy = geom.macos_size/2 + 1;
            cx = cy;
            testCase.verifyLessThanOrEqual(abs(py - cy), 1);
            testCase.verifyLessThanOrEqual(abs(px - cx), 1);
            testCase.verifyEqual(max(intensity(:)), 3.236e5, 'RelTol', 1e-3);
            testCase.verifyEqual(sum(intensity(:)), 1.802e6, 'RelTol', 1e-3);
        end

        function test_compare_cass_ff_psf(testCase)
            % Pixel-by-pixel comparison, analytical PROPER aperture vs
            % macos's full ray-traced Cass.  Looser tolerance (max_abs
            % < 0.1 on Strehl-normalised) — same bar pymacos uses.
            geom = geometries.cass_farfield();
            [proper_int, dx_p] = proper_run_cass_ff(geom);
            [macos_int,  dx_m] = macos_run_cass_ff(geom);

            testCase.verifyEqual(dx_p, dx_m, 'RelTol', 1e-3);
            testCase.verifyEqual(size(proper_int), size(macos_int));

            m = compare_and_record('cass_ff_nominal', ...
                macos_int, proper_int, dx_m, ...
                'results_dir', testCase.results_dir);
            fprintf('  cass_ff_nominal: max_abs=%.3e  rms=%.3e  max_aligned=%.3e\n', ...
                m.max_abs, m.rms_abs, m.max_abs_aligned);
            testCase.verifyLessThan(m.max_abs_aligned, 0.1, sprintf( ...
                'max|a-b| aligned = %.3e exceeds 0.1', m.max_abs_aligned));
        end

        function test_compare_cass_ff_psf_with_opd(testCase)
            % macos OPD passed through to PROPER as phase + amplitude
            % mask.  With sign-flip and amplitude-match, residual is
            % at FFT numerical precision; bar set at 1e-6.
            geom = geometries.cass_farfield();
            [macos_int, dx_m, opd] = macos_run_cass_ff(geom, ...
                'return_opd', true);
            [proper_int, dx_p] = proper_run_cass_ff(geom, ...
                'macos_opd', opd);

            testCase.verifyEqual(dx_p, dx_m, 'RelTol', 1e-3);

            m = compare_and_record('cass_ff_nominal_with_opd', ...
                macos_int, proper_int, dx_m, ...
                'results_dir', testCase.results_dir);
            fprintf('  cass_ff_with_opd: max_abs=%.3e  rms=%.3e  max_aligned=%.3e\n', ...
                m.max_abs, m.rms_abs, m.max_abs_aligned);
            testCase.verifyLessThan(m.max_abs, 1e-6, sprintf( ...
                'max|a-b| = %.3e exceeds 1e-6', m.max_abs));
        end
    end
end
