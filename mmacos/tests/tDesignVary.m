classdef tDesignVary < matlab.unittest.TestCase
%TDESIGNVARY  macos.design.System.vary declaration layer (Sprint 2A-i, slice 3).
%   vary() is pure spec bookkeeping (no engine traces), so this class is
%   fast.  It checks name-based DOF addressing (no 0-based leak), alias
%   expansion (tilt/decenter -> 2 vars), Zernike modes, roles, bounds,
%   and validation.

    properties (Constant)
        ModelSize    = 128
        RxName       = 'Rx_Cass_FarField.in'
        ExpectedNelt = 6
    end

    properties
        sys
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            rxp = rx_fixture_path(testCase.RxName);
            testCase.sys = macos.design.System.from_rx(rxp, ...
                'model_size', testCase.ModelSize);
        end
    end

    methods (TestMethodSetup)
        function resetVars(testCase)
            testCase.sys.clear_vars();
        end
    end

    methods (Test)
        function test_single_dof_by_name(testCase)
            s = testCase.sys;
            s.vary(2, 'Tz', 'bounds', [-2e-3 2e-3], 'unit', 'mm');
            v = s.design_vars();
            testCase.verifyEqual(numel(v), 1);
            testCase.verifyEqual(v.elt, 2);
            testCase.verifyEqual(v.family, 'rigid');
            testCase.verifyEqual(v.dof_name, 'Tz');     % name, not 0-based index
            testCase.verifyEqual(v.unit, 'mm');
            testCase.verifyEqual(v.role, 'design');
            % no 0-based numeric DOF field is exposed
            testCase.verifyFalse(isfield(v, 'dof'));
        end

        function test_despace_alias_is_Tz(testCase)
            s = testCase.sys;
            s.vary(3, 'despace', 'bounds', [-1 1]);
            v = s.design_vars();
            testCase.verifyEqual(numel(v), 1);
            testCase.verifyEqual(v.dof_name, 'Tz');
        end

        function test_tilt_expands_to_two(testCase)
            s = testCase.sys;
            s.vary(2, 'tilt', 'bounds', [-1e-3 1e-3], 'unit', 'mrad');
            v = s.design_vars();
            testCase.verifyEqual(numel(v), 2);
            testCase.verifyEqual({v.dof_name}, {'Rx','Ry'});
            testCase.verifyTrue(all(strcmp({v.family}, 'rigid')));
        end

        function test_decenter_expands_to_two(testCase)
            s = testCase.sys;
            s.vary(2, 'decenter', 'bounds', [-1 1]);
            v = s.design_vars();
            testCase.verifyEqual({v.dof_name}, {'Tx','Ty'});
        end

        function test_zern_mode(testCase)
            s = testCase.sys;
            s.vary(2, 'zern', 'mode', 5, 'bounds', [-1e-8 1e-8]);
            v = s.design_vars();
            testCase.verifyEqual(v.family, 'zern');
            testCase.verifyEqual(v.mode, 5);
        end

        function test_zern_requires_mode(testCase)
            s = testCase.sys;
            testCase.verifyError(@() s.vary(2, 'zern', 'bounds', [-1 1]), ...
                'macos:design:System:zernMode');
        end

        function test_compensator_role(testCase)
            s = testCase.sys;
            s.vary(2, 'Tz', 'role', 'compensator');
            v = s.design_vars();
            testCase.verifyEqual(v.role, 'compensator');
        end

        function test_accumulation_and_clear(testCase)
            s = testCase.sys;
            s.vary(2, 'Tz', 'bounds', [-1 1]);
            s.vary(3, 'Rx', 'bounds', [-1 1]);
            testCase.verifyEqual(numel(s.design_vars()), 2);
            s.clear_vars();
            testCase.verifyEmpty(s.design_vars());
        end

        function test_bad_element_errors(testCase)
            s = testCase.sys;
            testCase.verifyError( ...
                @() s.vary(testCase.ExpectedNelt + 1, 'Tz', 'bounds', [-1 1]), ...
                'macos:design:System:eltRange');
        end

        function test_bad_param_errors(testCase)
            s = testCase.sys;
            testCase.verifyError(@() s.vary(2, 'wobble', 'bounds', [-1 1]), ...
                'macos:design:System:param');
        end

        function test_bad_bounds_errors(testCase)
            s = testCase.sys;
            testCase.verifyError(@() s.vary(2, 'Tz', 'bounds', [2 -2]), ...
                'macos:design:System:bounds');
        end

        function test_surface_param_warns_and_records(testCase)
            s = testCase.sys;
            testCase.verifyWarning(@() s.vary(2, 'conic', 'bounds', [-1.2 -0.8]), ...
                'macos:design:System:noChannel');
            v = s.design_vars();
            testCase.verifyEqual(v.family, 'surface');
        end
    end
end
