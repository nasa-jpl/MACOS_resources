classdef tCalib < matlab.unittest.TestCase
%TCALIB  Tests for macos.calib (Phase 1c: mmacos CALIB wrap).
%   Mirror of pymacos's tests/test_calib.py.

    properties (Constant)
        ModelSize = 256
        RxSrc = ['/home/dcr/dev/macos/ZGD_test_files/' ...
                 'opt_example.in']
    end

    properties
        m
        rx_path
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.m = macos.Session(testCase.ModelSize);
        end
    end

    methods (TestMethodSetup)
        function setupMethod(testCase)
            tmpDir = tempname;
            mkdir(tmpDir);
            testCase.rx_path = fullfile(tmpDir, 'opt_example.in');
            copyfile(testCase.RxSrc, testCase.rx_path);
            testCase.m.load_rx(testCase.rx_path);
        end
    end

    methods (Test)
        % -- Phase 1a coverage ---------------------------------------

        function test_calib_runs_baseline(testCase)
            testCase.m.stop(7, [0 0]);
            r = testCase.m.calib();
            testCase.verifyTrue(r.converged, ...
                sprintf('baseline calib should converge; rtn_flag=%d', ...
                        r.rtn_flag));
            testCase.verifyGreaterThanOrEqual(r.n_fov, 1);
            testCase.verifyGreaterThanOrEqual(r.n_wavelength, 1);
        end

        function test_calib_converges_after_perturbation(testCase)
            testCase.m.perturb(1, 'rotation', [1e-3; 0; 0], ...
                                'frame', 'global');
            testCase.m.stop(7, [0 0]);
            r = testCase.m.calib();
            testCase.verifyTrue(r.converged, ...
                sprintf('calib should converge after perturb; rtn_flag=%d', ...
                        r.rtn_flag));
        end

        function test_calib_returns_expected_schema(testCase)
            testCase.m.stop(7, [0 0]);
            r = testCase.m.calib();
            for f = {'converged', 'rtn_flag', 'n_fov', 'n_wavelength', ...
                     'old_wfe', 'new_wfe'}
                testCase.verifyTrue(isfield(r, f{1}));
            end
            testCase.verifyEqual(size(r.old_wfe), ...
                [r.n_fov r.n_wavelength]);
            testCase.verifyEqual(size(r.new_wfe), ...
                [r.n_fov r.n_wavelength]);
        end

        % -- Phase 1b setter coverage --------------------------------

        function test_calib_set_var_elt_via_name_list(testCase)
            testCase.m.stop(7, [0 0]);
            testCase.m.perturb(1, 'rotation', [1e-3; 0; 0], ...
                                'frame', 'global');
            testCase.m.calib_clear_var_elts();
            testCase.m.calib_set_var_elt(7, 'TIP', 'TILT');
            r = testCase.m.calib();
            testCase.verifyTrue(r.converged);
        end

        function test_calib_set_var_elt_via_positional_mask(testCase)
            testCase.m.stop(7, [0 0]);
            testCase.m.perturb(1, 'rotation', [1e-3; 0; 0], ...
                                'frame', 'global');
            testCase.m.calib_clear_var_elts();
            testCase.m.calib_set_var_elt(7, [1 1 0 0 0 0 0 0]);
            r = testCase.m.calib();
            testCase.verifyTrue(r.converged);
        end

        function test_calib_set_iter_and_tol(testCase)
            testCase.m.stop(7, [0 0]);
            testCase.m.calib_set_iter(1);
            testCase.m.calib_set_tol(1e-12);
            testCase.m.calib();
        end

        function test_calib_set_target_accepts_names(testCase)
            for nm = {'WFE', 'BEAM', 'SPOT', 'OPL'}
                testCase.m.calib_set_target(nm{1});
            end
            testCase.m.calib_set_target('WFE_ZMODE', [4 5 11]);
            testCase.m.calib_set_target('ZWF', [4]);
        end

        function test_calib_set_target_rejects_unknown_name(testCase)
            testCase.verifyError( ...
                @() testCase.m.calib_set_target('not_a_target'), ...
                'macos:calib_set_target:badName');
        end

        function test_calib_set_var_elt_rejects_unknown_dof_name(testCase)
            testCase.verifyError( ...
                @() testCase.m.calib_set_var_elt(7, 'TIP', 'not_a_dof'), ...
                'macos:calib_set_var_elt:badName');
        end
    end
end
