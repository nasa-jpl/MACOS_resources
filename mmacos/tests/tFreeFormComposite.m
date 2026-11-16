classdef tFreeFormComposite < matlab.unittest.TestCase
%TFREEFORMCOMPOSITE  Tests for macos.zrn_freeform composite get/set.
%   FFSegDemoAll.in carries all three FreeForm components on its
%   segments (FF Zernike, Mon Zernike, Grid).  Covers:
%     - Round-trip get returns active flags + coefficients
%     - Preserve semantics: setting one component preserves the others
%     - Explicit 'clear' wipes a component
%     - Setter with all three structs round-trips to the same state
%
%   Mirror of pymacos's zrn_freeform composite-accessor test.

    properties (Constant)
        ModelSize = 256
        RxName    = 'FFSegDemoAll.in'
    end

    properties
        rx_path
        m
        srf
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
            ff = testCase.m.find_freeform_elts();
            testCase.assertNotEmpty(ff, ...
                'fixture must have at least one FreeForm element');
            testCase.srf = ff(1);
        end
    end

    methods (Test)
        function test_getter_returns_struct_with_three_fields(testCase)
            out = testCase.m.zrn_freeform(testCase.srf);
            testCase.verifyTrue(isfield(out, 'ff'));
            testCase.verifyTrue(isfield(out, 'mon'));
            testCase.verifyTrue(isfield(out, 'grid'));
        end

        function test_active_components_have_csys(testCase)
            out = testCase.m.zrn_freeform(testCase.srf);
            % FFSegDemoAll segments have all three components active.
            testCase.assertNotEmpty(out.ff,   'FF should be active');
            testCase.assertNotEmpty(out.mon,  'Mon should be active');
            testCase.assertNotEmpty(out.grid, 'Grid should be active');

            for comp = [out.ff, out.mon]
                testCase.verifyEqual(size(comp.csys.pos), [3 1]);
                testCase.verifyEqual(size(comp.csys.x),   [3 1]);
                testCase.verifyEqual(size(comp.csys.y),   [3 1]);
                testCase.verifyEqual(size(comp.csys.z),   [3 1]);
                testCase.verifyGreaterThan(comp.norm_rad, 0);
            end
        end

        function test_preserve_unspecified_components(testCase)
            % Setting only 'mon' (passing the current mon back) must
            % leave FF and Grid intact.  This is the original bug:
            % the setter was wiping FF + Grid because the
            % LOGICAL == LOGICAL comparison in macos_api_mod was
            % producing GFM=[F,F,F] under ifx-via-mex.
            before = testCase.m.zrn_freeform(testCase.srf);

            testCase.m.zrn_freeform(testCase.srf, 'mon', before.mon);

            after = testCase.m.zrn_freeform(testCase.srf);
            testCase.assertNotEmpty(after.ff,   'FF was wiped!');
            testCase.assertNotEmpty(after.mon,  'Mon was wiped!');
            testCase.assertNotEmpty(after.grid, 'Grid was wiped!');

            testCase.verifyEqual(after.ff.zern_type, before.ff.zern_type);
            testCase.verifyEqual(after.ff.norm_rad,  before.ff.norm_rad);
            testCase.verifyEqual(after.ff.coefs,     before.ff.coefs);

            testCase.verifyEqual(after.grid.dx,  before.grid.dx);
            testCase.verifyEqual(after.grid.mat, before.grid.mat);
        end

        function test_explicit_clear_grid(testCase)
            % Sentinel 'clear' wipes only the named component.
            before = testCase.m.zrn_freeform(testCase.srf);
            testCase.assertNotEmpty(before.grid);

            testCase.m.zrn_freeform(testCase.srf, 'grid', 'clear');

            after = testCase.m.zrn_freeform(testCase.srf);
            testCase.verifyEmpty(after.grid, 'Grid should be cleared');
            testCase.assertNotEmpty(after.ff, 'FF should survive');
            testCase.assertNotEmpty(after.mon, 'Mon should survive');
            testCase.verifyEqual(after.ff.coefs, before.ff.coefs);
        end

        function test_explicit_clear_ff(testCase)
            before = testCase.m.zrn_freeform(testCase.srf);
            testCase.assertNotEmpty(before.ff);

            testCase.m.zrn_freeform(testCase.srf, 'ff', 'clear');

            after = testCase.m.zrn_freeform(testCase.srf);
            testCase.verifyEmpty(after.ff, 'FF should be cleared');
            testCase.assertNotEmpty(after.mon);
            testCase.assertNotEmpty(after.grid);
        end

        function test_setter_modifies_one_coefficient(testCase)
            % Round-trip: read FF, modify a single coef, write back,
            % re-read, verify the new value sticks and others didn't.
            before = testCase.m.zrn_freeform(testCase.srf);
            mod = before.ff;
            % Bump coef index 5 by a known delta.
            new_val = mod.coefs(5) + 1.234e-6;
            mod.coefs(5) = new_val;

            testCase.m.zrn_freeform(testCase.srf, 'ff', mod);

            after = testCase.m.zrn_freeform(testCase.srf);
            testCase.verifyEqual(after.ff.coefs(5), new_val, ...
                'AbsTol', 1e-15);
            % Other coefs unchanged
            other_idx = [1:4, 6:numel(mod.coefs)];
            testCase.verifyEqual(after.ff.coefs(other_idx), ...
                                 before.ff.coefs(other_idx), ...
                                 'AbsTol', 1e-15);
            % Mon + Grid still intact
            testCase.verifyEqual(after.mon.coefs, before.mon.coefs);
            testCase.verifyEqual(after.grid.dx,   before.grid.dx);
        end

        function test_session_method_delegates(testCase)
            % Session.zrn_freeform should behave identically to the
            % package function.
            out_class = testCase.m.zrn_freeform(testCase.srf);
            out_pkg   = macos.zrn_freeform(testCase.srf);
            testCase.verifyEqual(out_class.ff.coefs, out_pkg.ff.coefs);
            testCase.verifyEqual(out_class.grid.mat, out_pkg.grid.mat);
        end

        function test_elt_grid_add_adds_displacement(testCase)
            % elt_grid_add must ADD the matrix node-for-node to the live
            % grid (after = before + dz).  Use an ASYMMETRIC dz so a
            % transpose/orientation bug would be caught.
            before = testCase.m.zrn_freeform(testCase.srf).grid.mat;
            N = size(before, 1);
            % distinct value per (row,col): 1000*iy + ix, scaled small
            [IX, IY] = meshgrid(1:N, 1:N);     % IY=row index, IX=col index
            dz = 1e-7 * (1000*IY + IX);
            testCase.assertNotEqual(dz, dz.', ...
                'dz must be asymmetric to exercise orientation');

            testCase.m.elt_grid_add(testCase.srf, dz);

            after = testCase.m.zrn_freeform(testCase.srf).grid.mat;
            testCase.verifyEqual(after, before + dz, 'AbsTol', 1e-14, ...
                'grid not incremented node-for-node (check orientation)');
        end

        function test_elt_grid_add_size_mismatch_errors(testCase)
            % Wrong-size grid_dz must error, not silently corrupt.
            N = size(testCase.m.zrn_freeform(testCase.srf).grid.mat, 1);
            bad = zeros(N+1, N+1);
            testCase.verifyError(@() testCase.m.elt_grid_add(testCase.srf, bad), ...
                'macos:elt_grid_add:sizeMismatch');
        end

        function test_elt_grid_add_package_equals_session(testCase)
            % Package function and Session method must apply identically.
            N  = size(testCase.m.zrn_freeform(testCase.srf).grid.mat, 1);
            dz = 3.0e-8 * ones(N, N);
            base = testCase.m.zrn_freeform(testCase.srf).grid.mat;

            macos.elt_grid_add(testCase.srf, dz);
            v_pkg = testCase.m.zrn_freeform(testCase.srf).grid.mat;
            testCase.verifyEqual(v_pkg, base + dz, 'AbsTol', 1e-14);
        end
    end
end
