classdef tProperCompareCoroDMPhase < matlab.unittest.TestCase
%TPROPERCOMPARECORODMPHASE  Port of pymacos test_coro_dm_phase.py (Phase 6b).
%   Pure-phase analog of Phase 6a's apodiser: imprint a known OPD at
%   the src element via macos.apodize_complex, do the same via
%   PROPER's prop_add_phase, propagate NF (Elt 5 → 6 of Rx_Coro_noLyot),
%   compare intensities.  Three diagnostic OPD shapes at λ/20 RMS:
%     - defocus (Z4, low-frequency)
%     - 5-cycle sinusoid (mid-frequency)
%     - filtered noise capped at 12 cycles per pupil radius
%
%   Pymacos hits ~3.7e-8 (Phase 3a hard-edge floor) or better.  Tolerance
%   1e-7 sum-norm RMS.

    properties (Constant)
        R_norm_m = 16e-3   % support disk radius
    end

    properties
        results_dir
    end

    properties (TestParameter)
        shape_name = struct( ...
            'defocus_Z4',         'defocus', ...
            'sinusoid_5cyc',      'sinusoid', ...
            'filtered_noise_k12', 'filtered_noise')
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            assert(exist('prop_begin', 'file') == 2, ...
                ['PROPER MATLAB not on path; add ~/dev/proper_matlab/' ...
                 ' or run via run_mmacos_tests.sh.']);
            testCase.results_dir = fullfile(fileparts( ...
                mfilename('fullpath')), 'results', 'phase6b');
        end
    end

    methods (Static, Access = private)
        function geom = build_geom()
            geom = geometries.coro_nfprop( ...
                'rx_filename',  'Rx_Coro_noLyot.in', ...
                'src_elt',      5, ...
                'detector_elt', 6);
        end
        function dx = probe_dx_at_src(geom)
            rx_path = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
                'pymacos', 'tests', 'Rx', geom.rx_filename);
            macos.init(geom.macos_size);
            macos.load_rx(rx_path);
            macos.intensity(geom.src_elt);
            dx = macos.dx_at(geom.src_elt);
        end
    end

    methods (Test)
        function test_coro_dm_phase_imprint(testCase, shape_name)
            geom      = tProperCompareCoroDMPhase.build_geom();
            dx_at_src = tProperCompareCoroDMPhase.probe_dx_at_src(geom);
            rms_target = geom.wavelength_m / 20.0;

            switch shape_name
                case 'defocus'
                    opd = opd_shapes('defocus', geom.macos_size, ...
                        dx_at_src, testCase.R_norm_m, rms_target);
                case 'sinusoid'
                    opd = opd_shapes('sinusoid', geom.macos_size, ...
                        dx_at_src, testCase.R_norm_m, rms_target, ...
                        'cycles', 5);
                case 'filtered_noise'
                    opd = opd_shapes('filtered_noise', geom.macos_size, ...
                        dx_at_src, testCase.R_norm_m, rms_target, ...
                        'max_cycles', 12, 'seed', 12345);
            end

            % Sanity: RMS within 1% of target.
            nz = opd(opd ~= 0);
            rms_actual = sqrt(mean(nz.^2));
            testCase.verifyEqual(rms_actual, rms_target, 'RelTol', 1e-2);

            [im, dx_m, wf] = macos_run_dm_phase(geom, opd);
            [ip, dx_p]     = proper_run_dm_phase(geom, wf);

            testCase.verifyEqual(size(im), size(ip));
            testCase.verifyEqual(dx_p, abs(dx_m), 'RelTol', 1e-3);

            tag = sprintf('coro_dm_phase_%s', shape_name);
            m = compare_and_record(tag, im, ip, abs(dx_m), ...
                'norm_kind', 'sum', ...
                'crop_pixels', size(im, 1), ...
                'results_dir', testCase.results_dir);
            fprintf(['  6b DM phase %-15s: max=%.3e rms=%.3e ' ...
                     'Δcom=(%+d,%+d) px\n'], ...
                shape_name, m.max_abs, m.rms_abs, m.dx_pix, m.dy_pix);
            testCase.verifyLessThan(m.rms_abs, 1e-7, sprintf( ...
                '%s: sum-norm RMS %.3e > 1e-7', shape_name, m.rms_abs));
        end
    end
end
