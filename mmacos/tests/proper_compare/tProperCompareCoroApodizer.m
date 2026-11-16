classdef tProperCompareCoroApodizer < matlab.unittest.TestCase
%TPROPERCOMPARECOROAPODIZER  Port of pymacos test_coro_apodizer.py (Phase 6a).
%   NFPlane Elt 5 -> Elt 6 of Rx_Coro_noLyot.in with a Gaussian-edge
%   pupil apodiser applied identically to both engines:
%     - macos via macos.apodize(srf, mask)
%     - PROPER via prop_multiply(wfo, mask)
%   Same numpy/MATLAB array fed to both — bit-identical apodisation, no
%   parametric reconstruction drift.
%
%   Apodiser: soft Gaussian roll-off outside r0=18 mm, sigma=2 mm,
%   fully truncated at r1=26 mm.
%
%   Expected: agreement at or below Phase 3a's (3.7e-8 hard-edge);
%   the soft taper kills the high-frequency edge content that limited
%   that step.

    properties (Constant)
        R0_M      = 18.0e-3
        SIGMA_M   = 2.0e-3
        R1_M      = 26.0e-3
        Supersample = 16
    end

    properties
        results_dir
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            assert(exist('prop_begin', 'file') == 2, ...
                ['PROPER MATLAB not on path; add ~/dev/proper_matlab/' ...
                 ' or run via run_mmacos_tests.sh.']);
            testCase.results_dir = fullfile(fileparts( ...
                mfilename('fullpath')), 'results', 'phase6a');
        end
    end

    methods (Test)
        function test_coro_apodised_nfplane_elt5_to_elt6(testCase)
            geom = geometries.coro_nfprop( ...
                'rx_filename',  'Rx_Coro_noLyot.in', ...
                'src_elt',      5, ...
                'detector_elt', 6);

            % Probe-load to discover dx_at(src_elt), then build the
            % mask at the matching pitch.
            rx_path = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
                'pymacos', 'tests', 'Rx', geom.rx_filename);
            macos.init(geom.macos_size);
            macos.load_rx(rx_path);
            macos.intensity(geom.src_elt);
            dx_at_src = macos.dx_at(geom.src_elt);

            mask = apodizer.build_apodised_mask( ...
                geom.macos_size, dx_at_src, ...
                apodizer.circle(testCase.R1_M), ...
                apodizer.gaussian_edge_taper(testCase.R0_M, testCase.SIGMA_M), ...
                testCase.Supersample);

            testCase.verifyEqual(size(mask), ...
                [geom.macos_size, geom.macos_size]);
            testCase.verifyGreaterThanOrEqual(min(mask(:)), 0);
            testCase.verifyLessThanOrEqual(max(mask(:)), 1 + 1e-12);
            testCase.verifyGreaterThan(max(mask(:)), 0.99);

            [im, dx_m, wf] = macos_run_apodised_nf(geom, mask);
            [ip, dx_p]     = proper_run_apodised_nf(geom, wf);

            testCase.verifyEqual(size(im), size(ip));
            testCase.verifyEqual(dx_p, abs(dx_m), 'RelTol', 1e-3);

            m = compare_and_record('coro_apodised_nfplane_elt5_to_elt6', ...
                im, ip, abs(dx_m), ...
                'norm_kind', 'sum', ...
                'crop_pixels', size(im, 1), ...
                'results_dir', testCase.results_dir);
            fprintf(['  6a Apodised Elt 5->6: max=%.3e rms=%.3e ' ...
                     'Δcom=(%+d,%+d) px\n'], ...
                m.max_abs, m.rms_abs, m.dx_pix, m.dy_pix);

            % Expect agreement at or below Phase 3a's 3.7e-8 (hard-edge
            % clipping limit there); soft taper should be better.
            testCase.verifyLessThan(m.max_abs, 1e-7, sprintf( ...
                'max |a-b| = %.3e sum-norm > 1e-7', m.max_abs));
        end
    end
end
