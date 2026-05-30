classdef tProperCompareCoroNFprop < matlab.unittest.TestCase
%TPROPERCOMPARECORONFPROP  Port of pymacos test_coro_nfprop.py (Phase 2).
%   3 tests covering the Rx_Coro Elt 2 → Elt 3 near-field plane-to-plane
%   propagation step (774 mm Fresnel).  Same wavefront fed to both
%   engines at Elt 2 (macos's diffraction-grid complex_field); compare
%   PROPER's propagated intensity to macos's INT at Elt 3.
%
%   model_size = 1024 — runs in the third per-size group of
%   run_mmacos_tests.sh's full-suite split.

    properties
        results_dir
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            assert(exist('prop_begin', 'file') == 2, ...
                ['PROPER MATLAB not on path; add ~/dev/proper_matlab/' ...
                 ' via addpath or run via run_mmacos_tests.sh.']);
            testCase.results_dir = fullfile(fileparts( ...
                mfilename('fullpath')), 'results', 'phase2');
        end
    end

    methods (Test)
        function test_coro_nfprop_macos_runs(testCase)
            % macos returns sensible intensity at Elt 3 and a
            % wavefront at Elt 2.  Reference values from pymacos's
            % diagnostic at nGridpts=511 / model_size=1024:
            %   Peak intensity ≈ 6.347e-6
            %   Sum  intensity ≈ 0.1962
            geom = geometries.coro_nfprop();
            [intensity_at_3, ~, wf2] = macos_run_coro_nfprop(geom);

            testCase.verifyEqual(size(intensity_at_3), ...
                [geom.macos_size, geom.macos_size]);
            testCase.verifyTrue(all(isfinite(intensity_at_3(:))));
            testCase.verifyGreaterThan(max(intensity_at_3(:)), 0);

            testCase.verifyEqual(size(wf2.amplitude), size(intensity_at_3));
            testCase.verifyTrue(all(isfinite(wf2.amplitude(:))));
            if ~isempty(wf2.opd)
                testCase.verifyTrue(all(isfinite(wf2.opd(:))));
            end

            testCase.verifyEqual(max(intensity_at_3(:)), 6.347e-6, ...
                'RelTol', 1e-3);
            testCase.verifyEqual(sum(intensity_at_3(:)), 0.1962, ...
                'RelTol', 1e-3);
        end

        function test_coro_nfprop_proper_runs(testCase)
            % PROPER ingests macos's Elt-2 wavefront and produces a
            % finite, non-degenerate intensity at the Elt-3 plane.
            geom = geometries.coro_nfprop();
            [~, dx_m, wf2] = macos_run_coro_nfprop(geom);
            [intensity_p, dxp] = proper_run_coro_nfprop(geom, wf2);

            testCase.verifyEqual(size(intensity_p), ...
                [geom.macos_size, geom.macos_size]);
            testCase.verifyTrue(all(isfinite(intensity_p(:))));
            testCase.verifyGreaterThan(max(intensity_p(:)), 0);
            testCase.verifyEqual(dxp, dx_m, 'RelTol', 1e-3);
        end

        function test_coro_nfprop_compare(testCase)
            % Pixel-level comparison.  NF prop is sum-norm (flux per
            % pixel), not Strehl-norm — peak-norm here would amplify a
            % tiny normalisation difference into a 9e-6 residual that
            % disappears under sum-norm.  Pymacos hits ~5e-12 RMS,
            % 2.5e-10 max; tolerance 1e-8 gives 4 decades of margin
            % for routine MKL / platform drift.
            geom = geometries.coro_nfprop();
            [intensity_m, dx_m, wf2] = macos_run_coro_nfprop(geom);
            [intensity_p, dx_p]      = proper_run_coro_nfprop(geom, wf2);

            testCase.verifyEqual(size(intensity_p), size(intensity_m));
            testCase.verifyEqual(dx_p, dx_m, 'RelTol', 1e-3);

            m = compare_and_record('coro_nfprop_elt2_to_elt3', ...
                intensity_m, intensity_p, dx_m, ...
                'norm_kind', 'sum', ...
                'crop_pixels', size(intensity_m, 1), ...
                'results_dir', testCase.results_dir);
            fprintf(['  coro_nfprop_elt2_to_elt3: max_abs=%.3e ' ...
                     ' rms=%.3e  Δcom=(%+d,%+d) px\n'], ...
                m.max_abs, m.rms_abs, m.dx_pix, m.dy_pix);
            testCase.verifyLessThan(m.max_abs, 1e-8, sprintf( ...
                'max |a-b| = %.3e (sum-norm) exceeds 1e-8', m.max_abs));
        end
    end
end
