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

        % --- new query/instrument method delegation ------------------
        function test_elt_conic_radius_delegate(testCase)
            testCase.verifyEqual(testCase.m.get_elt_kc(2), macos.get_elt_kc(2));
            testCase.verifyEqual(testCase.m.get_elt_kr(2), macos.get_elt_kr(2));
        end

        function test_src_size_delegate(testCase)
            s_class = testCase.m.get_src_size();
            s_pkg   = macos.get_src_size();
            testCase.verifyEqual(s_class.aperture, s_pkg.aperture);
            testCase.verifyEqual(testCase.m.is_point_source(), macos.is_point_source());
        end

        function test_surface_predicates_delegate(testCase)
            testCase.verifyEqual(testCase.m.elt_grid_any(), macos.elt_grid_any());
            testCase.verifyEqual(testCase.m.elt_zrn_any(),  macos.elt_zrn_any());
        end

        function test_stop_info_delegate(testCase)
            testCase.m.stop(2);
            s = testCase.m.get_stop_info();
            testCase.verifyEqual(s.elt, 2);
        end

        function test_every_stateful_package_function_has_a_delegator(testCase)
        % Coverage audit, kept honest by making it a gate rather than a
        % one-off sweep (Luis, 2026-08-19: Session was missing ~40 of the
        % package surface).  Rule: every +macos function that TOUCHES
        % ENGINE STATE gets a Session method.  The exceptions below are
        % the functions that do not -- pure array/file helpers and the
        % session-lifecycle entry points -- and a new one must be added
        % here CONSCIOUSLY, with a reason.
            NOT_SESSION_SCOPED = { ...
                ... % session lifecycle -- the constructor owns these
                'init', 'model_sizes', 'model_size_min', ...
                ... % pure array reshaping
                'm2v', 'v2m', 'pad', 'config_canvas', ...
                ... % pure file I/O, no engine involved
                'read_grid_file', 'write_grid_file', ...
                ... % pure math on arrays the caller already holds
                'zernike_grid_basis', 'first_airy_null', ...
                'lambda_over_D_pixels', 'radial_profile', ...
                'radial_contrast', 'dark_zone_metrics', 'mimg', ...
                ... % inner FD loops driven by the dw_d* supervisors
                'dwdx_for_current_source', 'dwdz_for_current_source'};

            pkg_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                               'src', '+macos');
            d = dir(fullfile(pkg_dir, '*.m'));
            fns = erase({d.name}, '.m');
            fns = setdiff(fns, [{'Session'}, NOT_SESSION_SCOPED]);

            have = methods('macos.Session');
            missing = setdiff(fns, have);
            testCase.verifyEmpty(missing, sprintf( ...
                ['these +macos functions touch engine state but have no ' ...
                 'macos.Session method: %s\n(add a delegator, or add the ' ...
                 'name to NOT_SESSION_SCOPED with a reason)'], ...
                strjoin(missing, ', ')));

            % and the exception list must not rot: every name on it must
            % still exist in the package.
            gone = setdiff(NOT_SESSION_SCOPED, erase({d.name}, '.m'));
            testCase.verifyEmpty(gone, sprintf( ...
                'NOT_SESSION_SCOPED names functions that no longer exist: %s', ...
                strjoin(gone, ', ')));
        end

        function test_new_delegators_reach_the_package(testCase)
        % Smoke the delegators added in the 2026-08-19 sweep that are
        % cheap to call against a loaded Rx.
            testCase.verifyEqual(testCase.m.met_geom(), macos.met_geom());
            testCase.verifyEqual(testCase.m.find_powered_elts(testCase.rx_path), ...
                                 macos.find_powered_elts(testCase.m, testCase.rx_path));
            testCase.m.load_rx(testCase.rx_path);
            testCase.m.trace();
            testCase.verifyEqual(testCase.m.draw_rays(), macos.draw_rays());
        end
    end
end
