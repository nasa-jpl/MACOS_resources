classdef tCrossSurface < matlab.unittest.TestCase
%TCROSSSURFACE  All three top-level surfaces (mmacos mex, +macos
%   package, macos.Session) share libsmacos.a state.  These tests
%   exercise that: mutate via one surface, observe via another.

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
    end

    properties
        rx_path
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.rx_path = rx_fixture_path(testCase.RxName);
            macos.init(testCase.ModelSize);
            macos.load_rx(testCase.rx_path);
        end
    end

    methods (Test)
        function test_all_three_surfaces_report_same_num_elt(testCase)
            n_raw   = mmacos('n_elt');
            n_pkg   = macos.num_elt();
            m       = macos.Session(testCase.ModelSize);
            m.load_rx(testCase.rx_path);
            n_class = m.num_elt();
            testCase.verifyEqual(n_raw, n_pkg);
            testCase.verifyEqual(n_pkg, n_class);
        end

        function test_pkg_set_class_get_src_sampling(testCase)
            macos.set_src_sampling(64);
            m = macos.Session(testCase.ModelSize);
            m.load_rx(testCase.rx_path);
            macos.set_src_sampling(96);
            % After load_rx the engine resets sampling to the Rx default.
            % Set again, then query via class.
            macos.set_src_sampling(96);
            testCase.verifyEqual(m.get_src_sampling(), 96);
        end

        function test_class_set_pkg_get_src_sampling(testCase)
            m = macos.Session(testCase.ModelSize);
            m.load_rx(testCase.rx_path);
            m.set_src_sampling(80);
            testCase.verifyEqual(macos.get_src_sampling(), 80);
        end

        function test_raw_set_pkg_get_src_wvl(testCase)
            macos.load_rx(testCase.rx_path);
            % Setter via raw mex; getter via package.
            mmacos('src_wvl', 1.234e-6, 1);
            wvl_pkg = macos.get_src_wvl();
            testCase.verifyEqual(wvl_pkg, 1.234e-6, 'AbsTol', 1e-18);
        end
    end
end
