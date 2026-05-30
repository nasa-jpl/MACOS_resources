classdef tMacosSession < matlab.unittest.TestCase
%TMACOSSESSION  Tests for the macos.Session classdef.
%   Verifies that every method delegates correctly to the package
%   function of the same name.  Most methods are 1-line passthroughs,
%   so we exercise each once via a smoke check.

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
    end

    properties
        rx_path
        m  % session instance
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.rx_path = rx_fixture_path(testCase.RxName);
        end
    end

    methods (TestMethodSetup)
        function setupMethod(testCase)
            testCase.m = macos.Session(testCase.ModelSize);
            testCase.m.load_rx(testCase.rx_path);
        end
    end

    methods (Test)
        function test_constructor_returns_handle(testCase)
            testCase.verifyClass(testCase.m, 'macos.Session');
        end

        function test_rx_path_recorded(testCase)
            testCase.verifyEqual(testCase.m.rx_path, testCase.rx_path);
        end

        function test_num_elt_matches_package(testCase)
            testCase.verifyEqual(testCase.m.num_elt(), macos.num_elt());
        end

        function test_has_rx_true(testCase)
            testCase.verifyTrue(testCase.m.has_rx());
        end

        function test_cbm_matches_package(testCase)
            testCase.verifyEqual(testCase.m.cbm(), macos.cbm());
        end

        function test_get_elt_vpt_matches_package(testCase)
            v_class = testCase.m.get_elt_vpt(2);
            v_pkg   = macos.get_elt_vpt(2);
            testCase.verifyEqual(v_class, v_pkg);
        end

        function test_trace_returns_struct(testCase)
            s = testCase.m.trace();
            testCase.verifyTrue(isstruct(s));
            testCase.verifyTrue(isfield(s, 'nRays'));
            testCase.verifyTrue(isfield(s, 'rmsWFE'));
        end

        function test_opd_matches_package(testCase)
            testCase.m.trace();
            W_class = testCase.m.opd();
            W_pkg   = macos.opd();
            testCase.verifyEqual(W_class, W_pkg);
        end

        function test_perturb_delegates(testCase)
            testCase.m.perturb(2, 'rotation', [1e-6;0;0], 'frame', 'global');
            testCase.m.modify();
            s = testCase.m.trace();
            testCase.verifyGreaterThan(s.nRays, 0);
        end

        function test_dx_at_with_unit(testCase)
            testCase.m.trace();
            dx_mm = testCase.m.dx_at(testCase.m.num_elt(), 'mm');
            dx_m  = testCase.m.dx_at(testCase.m.num_elt(), 'm');
            testCase.verifyEqual(dx_mm, dx_m * 1e3, 'AbsTol', 1e-18);
        end
    end
end
