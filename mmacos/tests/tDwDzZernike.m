classdef tDwDzZernike < matlab.unittest.TestCase
%TDWDZZERNIKE  Regression tests for macos.dw_dz_zernike + multi-field.

    properties (Constant)
        ModelSize = 128
        RxName    = 'e5hex1.in'
        ZmodeStart = 4
        NZcoef     = 6   % Z4..Z6 -> 3 modes per element
        ExpectedFFCount = 8     % e5hex1 has 8 FreeForm elements (1-7,9)
        ExpectedZernCount = 1   % and 1 Zern-typed element (8)
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

    methods (Test)
        function test_eligibility_counts(testCase)
            macos.load_rx(testCase.rx_path);
            ff = macos.find_freeform_elts();
            ze = macos.find_zern_elts(testCase.rx_path);
            testCase.verifyEqual(numel(ff), testCase.ExpectedFFCount, ...
                'find_freeform_elts returned wrong count for e5hex1');
            testCase.verifyEqual(numel(ze), testCase.ExpectedZernCount, ...
                'find_zern_elts returned wrong count for e5hex1');
        end

        function test_single_field_shape(testCase)
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dz_zernike(m, testCase.rx_path, ...
                'zmode_start', testCase.ZmodeStart, ...
                'n_zcoef',    testCase.NZcoef, ...
                'delta', 1e-6);
            n_modes = testCase.NZcoef - testCase.ZmodeStart + 1;
            expected_chans = ...
                (testCase.ExpectedFFCount + testCase.ExpectedZernCount) ...
                * n_modes;
            testCase.verifyEqual(numel(out.channel_names), expected_chans, ...
                'Channel count does not match (Nelt-eligible) * (n_modes)');
            testCase.verifyEqual(size(out.dwdz, 2), expected_chans);
            testCase.verifyEqual(size(out.dwdz, 1), numel(out.w_nom_vec));
            testCase.verifyGreaterThan(max(abs(out.dwdz(:))), 0, ...
                '|dwdz| max is zero -- the FD sweep did not perturb anything');
        end

        function test_kind_major_channel_order(testCase)
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dz_zernike(m, testCase.rx_path, ...
                'zmode_start', testCase.ZmodeStart, ...
                'n_zcoef',    testCase.NZcoef);
            % First block: all MonZern (8 elts × 3 modes = 24 entries).
            % Second block: all Zern (1 elt × 3 modes = 3 entries).
            n_modes = testCase.NZcoef - testCase.ZmodeStart + 1;
            n_mz = testCase.ExpectedFFCount * n_modes;
            for k = 1:n_mz
                testCase.verifyTrue(contains(out.channel_names{k}, 'MonZern'), ...
                    sprintf('Channel %d (%s) should be MonZern', ...
                        k, out.channel_names{k}));
            end
            for k = n_mz+1:numel(out.channel_names)
                testCase.verifyTrue(contains(out.channel_names{k}, 'Zern') ...
                    && ~contains(out.channel_names{k}, 'MonZern'), ...
                    sprintf('Channel %d (%s) should be Zern (not MonZern)', ...
                        k, out.channel_names{k}));
            end
        end

        function test_multi_field_5fp_shapes(testCase)
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dz_zernike_multi(m, testCase.rx_path, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'zmode_start', testCase.ZmodeStart, ...
                'n_zcoef',    testCase.NZcoef);
            n_modes = testCase.NZcoef - testCase.ZmodeStart + 1;
            n_chans = ...
                (testCase.ExpectedFFCount + testCase.ExpectedZernCount) ...
                * n_modes;
            % 5 fields default.
            testCase.verifyEqual(numel(out.field_names), 5);
            testCase.verifyEqual(size(out.field_table, 1), 5);
            testCase.verifyEqual(size(out.field_table, 2), 4);
            % Per-field row counts × 5 = total non-zero pixels.
            n_rays_per_field = numel(out.per_field_dwdz{1}) ...
                / size(out.per_field_dwdz{1}, 2);
            testCase.verifyEqual(size(out.dwdxall, 1), ...
                5 * n_rays_per_field, ...
                'dwdxall row count should be 5 * per-field row count');
            testCase.verifyEqual(size(out.dwdxall, 2), n_chans);
            % Aliases match.
            testCase.verifyEqual(out.dwdxall, out.dwdzall, ...
                'dwdxall and dwdzall must alias the same matrix');
        end

        function test_multi_field_center_tile_bitwise(testCase)
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dz_zernike_multi(m, testCase.rx_path, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'zmode_start', testCase.ZmodeStart, ...
                'n_zcoef',    testCase.NZcoef);
            % Locate the center field entry.
            cidx = find(out.field_table(:,1) == 0 ...
                      & out.field_table(:,2) == 0, 1);
            testCase.verifyNotEmpty(cidx, ...
                'No (0,0) center field in the field_table');
            tr = out.field_table(cidx, 3);
            tc = out.field_table(cidx, 4);
            N  = size(out.per_field_dwdz{cidx}, 1) ... % nominal rows
                + 0;
            % Build a 2D mask: indxall positions that fall in the
            % center tile.
            indx = out.indxall;
            in_ctr = (indx.i > tr * 128) & (indx.i <= (tr+1)*128) ...
                   & (indx.j > tc * 128) & (indx.j <= (tc+1)*128);
            dwdzall_ctr = out.dwdxall(in_ctr, :);
            dwdz_C = out.per_field_dwdz{cidx};
            testCase.verifyEqual(size(dwdzall_ctr), size(dwdz_C), ...
                'Center-tile slice shape mismatch');
            testCase.verifyEqual(max(abs(dwdzall_ctr(:) - dwdz_C(:))), 0, ...
                'Center-tile rows of dwdxall must bitwise-match per_field_dwdz[center]');
        end
    end
end
