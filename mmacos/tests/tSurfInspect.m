classdef tSurfInspect < matlab.unittest.TestCase
%TSURFINSPECT  Element surface-inspection veneer coverage.
%   Exercises the read/write veneers over the elt_srf_* api_mod family --
%   grid data, Zernike surface parameters, surface coordinate frames -- plus
%   the source/stop query veneers.  These replace direct Rx-file inspection
%   for programmatic access to loaded-prescription surface data.
%
%   e5hex1.in carries FreeForm grid surfaces (elts 1-7,9), one Zernike
%   surface (elt 8) and a collimated source, so a single fixture drives the
%   whole family.  Runs at ModelSize=256 (grid surfaces need mGridMat>=256).

    properties (Constant)
        ModelSize = 256
        RxName    = 'e5hex1.in'
    end

    properties
        rx_path
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.rx_path = rx_fixture_path(testCase.RxName);
            macos.init(testCase.ModelSize);
        end
    end

    methods (TestMethodSetup)
        function setupMethod(testCase)
            macos.load_rx(testCase.rx_path);
        end
    end

    methods (Static)
        function e = first_grid(testCase)
            g = macos.find_grid_elts();
            testCase.assumeNotEmpty(g);
            e = g(1);
        end
        function e = first_zern(testCase)
            e = [];
            for k = 1:macos.num_elt()
                if macos.get_elt_zrn_type(k) > 0, e = k; break; end
            end
            testCase.assumeNotEmpty(e);
        end
    end

    methods (Test)
        % --- predicates + maxes --------------------------------------
        function test_predicates(testCase)
            testCase.verifyTrue(macos.elt_grid_any());
            testCase.verifyTrue(macos.elt_zrn_any());
            testCase.verifyTrue(macos.elt_ff_any());
        end
        function test_maxes(testCase)
            testCase.verifyEqual(macos.grid_size_max(), testCase.ModelSize);
            testCase.verifyGreaterThan(macos.mon_zrn_max_modes(), 0);
        end

        % --- grid data -----------------------------------------------
        function test_grid_read(testCase)
            e = tSurfInspect.first_grid(testCase);
            g = macos.get_elt_grid(e);
            testCase.verifyGreaterThan(g.size, 0);
            testCase.verifyEqual(size(g.mat), [g.size, g.size]);
            testCase.verifyEqual(macos.get_elt_grid_size(e), g.size);
            testCase.verifyEqual(macos.get_elt_grid_spacing(e), g.dx, 'RelTol', 1e-12);
        end
        function test_grid_write_roundtrip(testCase)
            e = tSurfInspect.first_grid(testCase);
            g = macos.get_elt_grid(e);
            m = g.mat; m(2,3) = m(2,3) + 1e-6;      % perturb one node
            macos.set_elt_grid(e, g.dx, m);
            g2 = macos.get_elt_grid(e);
            testCase.verifyEqual(g2.mat, m, 'AbsTol', 1e-15);
            testCase.verifyEqual(g2.dx, g.dx, 'RelTol', 1e-12);
        end
        function test_grid_scale(testCase)
            e = tSurfInspect.first_grid(testCase);
            g = macos.get_elt_grid(e);
            m = g.mat; m(1,1) = 2e-6; macos.set_elt_grid(e, g.dx, m);
            macos.scale_elt_grid(e, 3.0);
            g2 = macos.get_elt_grid(e);
            testCase.verifyEqual(g2.mat(1,1), 6e-6, 'AbsTol', 1e-15);
        end
        function test_grid_spacing_set(testCase)
            e = tSurfInspect.first_grid(testCase);
            dx0 = macos.get_elt_grid_spacing(e);
            macos.set_elt_grid_spacing(e, dx0 * 1.5);
            testCase.verifyEqual(macos.get_elt_grid_spacing(e), dx0 * 1.5, 'RelTol', 1e-12);
        end

        % --- Zernike surface -----------------------------------------
        function test_zrn_read(testCase)
            e = tSurfInspect.first_zern(testCase);
            z = macos.get_elt_zrn(e);
            testCase.verifyGreaterThan(z.type, 0);
            testCase.verifyGreaterThan(z.norm_radius, 0);
            testCase.verifyEqual(macos.get_elt_zrn_type(e), z.type);
            testCase.verifyEqual(macos.get_elt_zrn_norm_radius(e), z.norm_radius, 'RelTol', 1e-12);
        end
        function test_zrn_norm_radius_roundtrip(testCase)
            e = tSurfInspect.first_zern(testCase);
            r0 = macos.get_elt_zrn_norm_radius(e);
            macos.set_elt_zrn_norm_radius(e, r0 * 1.25);
            testCase.verifyEqual(macos.get_elt_zrn_norm_radius(e), r0 * 1.25, 'RelTol', 1e-12);
        end
        function test_zrn_type_roundtrip(testCase)
            e = tSurfInspect.first_zern(testCase);
            t0 = macos.get_elt_zrn_type(e);
            newt = 1 + mod(t0, 3);             % a different valid type in 1..3
            macos.set_elt_zrn_type(e, newt);
            testCase.verifyEqual(macos.get_elt_zrn_type(e), newt);
        end
        function test_non_zern_type_is_neg(testCase)
            e = tSurfInspect.first_grid(testCase);   % grid elt is not Zernike
            testCase.verifyEqual(macos.get_elt_zrn_type(e), -1);
        end

        % --- surface coordinate frame --------------------------------
        function test_srf_csys_read(testCase)
            e = tSurfInspect.first_zern(testCase);
            c = macos.get_elt_srf_csys(e);
            testCase.verifyEqual(size(c.zMon), [3, 1]);
            testCase.verifyEqual(norm(c.zMon), 1.0, 'AbsTol', 1e-9);
        end

        % --- source queries ------------------------------------------
        function test_src_size_roundtrip(testCase)
            s = macos.get_src_size();
            testCase.verifyGreaterThan(s.aperture, 0);
            macos.set_src_size(s.aperture * 0.9, s.obscuration);
            s2 = macos.get_src_size();
            testCase.verifyEqual(s2.aperture, s.aperture * 0.9, 'RelTol', 1e-12);
        end
        function test_is_point_source(testCase)
            testCase.verifyClass(macos.is_point_source(), 'logical');
        end
        function test_src_csys_unit_z(testCase)
            c = macos.get_src_csys();
            testCase.verifyEqual(norm(c.zDir), 1.0, 'AbsTol', 1e-9);
        end

        % --- stop info -----------------------------------------------
        function test_stop_info_no_stop_raises(testCase)
            % e5hex1 has a global stop position but no element-anchored stop,
            % so get_stop_info must raise cleanly (not return garbage).  The
            % positive path (returns the stop element + offset) is covered in
            % tSpot on the Cassegrain fixture.
            testCase.verifyError(@() macos.get_stop_info(), ?MException);
        end

        % --- Session delegation smoke --------------------------------
        function test_session_delegates(testCase)
            m = macos.Session(testCase.ModelSize);
            m.load_rx(testCase.rx_path);
            testCase.verifyEqual(m.elt_grid_any(), macos.elt_grid_any());
            testCase.verifyEqual(m.grid_size_max(), macos.grid_size_max());
            e = tSurfInspect.first_zern(testCase);
            testCase.verifyEqual(m.get_elt_zrn_type(e), macos.get_elt_zrn_type(e));
        end
    end
end
