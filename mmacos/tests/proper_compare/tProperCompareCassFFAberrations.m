classdef tProperCompareCassFFAberrations < matlab.unittest.TestCase
%TPROPERCOMPARECASSFFABERRATIONS  Port of pymacos test_cass_ff_aberrations.py.
%   macos-vs-PROPER comparison under small Secondary-Mirror (Elt 3)
%   perturbations.  Each perturbation traced through macos, its OPD
%   captured at the exit pupil, then handed to PROPER's prop_add_phase
%   so both engines see the SAME aberrated wavefront.
%
%   6 parametrized cases — nominal + tiny Tx/Ty/Tz tweaks.  Tolerance
%   1e-6 on Strehl-normalised max|a-b|; observed ~1e-11 (same as the
%   no-perturb Phase 1 result).

    properties (Constant)
        SecondaryElt = 3
    end

    properties (TestParameter)
        perturbation = struct( ...
            'nominal',      [0, 0, 0,      0,      0,    0   ], ...
            'Tx_plus_1um',  [0, 0, 0,      1e-6,   0,    0   ], ...
            'Tx_minus_1um', [0, 0, 0,     -1e-6,   0,    0   ], ...
            'Ty_plus_1um',  [0, 0, 0,      0,      1e-6, 0   ], ...
            'Tz_plus_5um',  [0, 0, 0,      0,      0,    5e-6], ...
            'Tz_minus_5um', [0, 0, 0,      0,      0,   -5e-6])
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
                mfilename('fullpath')), 'results', 'phase1_aberrations');
        end
    end

    methods (Test)
        function test_compare_secondary_perturbation(testCase, perturbation)
            geom = geometries.cass_farfield();
            prb_vec = perturbation(:);
            if any(prb_vec)
                pert_arg = {testCase.SecondaryElt, prb_vec};
            else
                pert_arg = {};
            end

            [im, dx_m, opd] = macos_run_cass_ff(geom, ...
                'return_opd', true, ...
                'perturbation', pert_arg);
            [ip, dx_p] = proper_run_cass_ff(geom, 'macos_opd', opd);

            testCase.verifyEqual(dx_p, dx_m, 'RelTol', 1e-3);
            testCase.verifyEqual(size(im), size(ip));

            m = compare_and_record('cass_ff_perturb_SM', ...
                im, ip, dx_m, ...
                'results_dir', testCase.results_dir);
            fprintf(['  cass_ff_aberration: max=%.3e rms=%.3e ' ...
                     'Δcom=(%+d,%+d) px\n'], ...
                m.max_abs, m.rms_abs, m.dx_pix, m.dy_pix);
            testCase.verifyLessThan(m.max_abs, 1e-6, sprintf( ...
                'max |a-b| = %.3e Strehl-norm > 1e-6', m.max_abs));
        end
    end
end
