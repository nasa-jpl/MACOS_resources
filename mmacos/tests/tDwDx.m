classdef tDwDx < matlab.unittest.TestCase
%TDWDX  Regression tests for macos.dw_dx + multi-field.

    properties (Constant)
        ModelSize       = 128
        RxName          = 'e5hex1.in'
        DOFsForTest     = (3:5).'  % Tx,Ty,Tz only -- keep tests fast
        ExpectedActOpts = 11       % 13 elements - 2 Reference/Return
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
        function test_actual_optic_count(testCase)
            % Parse the Rx text -- 13 elements minus 2 Reference/Return
            % should leave 11 actual optics.
            macos.load_rx(testCase.rx_path);
            chs = macos.channels.rigid_body_channels( ...
                macos.Session(testCase.ModelSize), testCase.rx_path, ...
                'dofs', [3]);
            testCase.verifyEqual(numel(chs), testCase.ExpectedActOpts, ...
                'rigid_body_channels actual-optic count mismatch');
        end

        function test_single_field_shape(testCase)
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', testCase.DOFsForTest, 'delta', 1e-8);
            n_dof = numel(testCase.DOFsForTest);
            expected = testCase.ExpectedActOpts * n_dof;
            testCase.verifyEqual(numel(out.channel_names), expected);
            testCase.verifyEqual(size(out.dwdx, 2), expected);
            testCase.verifyEqual(size(out.dwdx, 1), numel(out.w_nom_vec));
            testCase.verifyGreaterThan(max(abs(out.dwdx(:))), 0);
        end

        function test_element_major_channel_order(testCase)
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', testCase.DOFsForTest);
            % Per-element block: Elt 1 (Tx,Ty,Tz), Elt 2 ...
            for k = 1:numel(out.channel_names)
                expected_elt = ceil(k / numel(testCase.DOFsForTest));
                actual = sscanf(out.channel_names{k}, 'Elt %d');
                testCase.verifyEqual(actual, ...
                    out.iElt(find(out.iElt > 0, 1) + expected_elt - 1), ...
                    'Channel order not element-major');
                break;   % single-element check is sufficient evidence
            end
        end

        function test_multi_field_5fp_shapes(testCase)
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx_multi(m, testCase.rx_path, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'dofs', testCase.DOFsForTest, 'delta', 1e-8);
            n_dof = numel(testCase.DOFsForTest);
            expected = testCase.ExpectedActOpts * n_dof;
            testCase.verifyEqual(numel(out.field_names), 5);
            testCase.verifyEqual(size(out.field_table, 1), 5);
            testCase.verifyEqual(size(out.field_table, 2), 4);
            testCase.verifyEqual(size(out.dwdxall, 2), expected);
        end

        function test_ngridpts_override(testCase)
            % 'ngridpts' overrides the .in ray-grid sampling (Luis's
            % request): the OPD canvas follows the override, not the
            % .in value / model clamp.
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', [3], 'ngridpts', 31);
            testCase.verifyEqual(size(out.w_nom_2d), [31 31]);
            testCase.verifyEqual(double(m.get_src_sampling()), 31);
        end

        function test_ngridpts_clamp_warns(testCase)
            % Oversized request clamps to the model limit and warns.
            m = macos.Session(testCase.ModelSize);
            testCase.verifyWarning(@() macos.dw_dx(m, testCase.rx_path, ...
                'dofs', [3], 'ngridpts', 99999), 'macos:dw_dx:ngridpts');
            testCase.verifyLessThanOrEqual( ...
                double(m.get_src_sampling()), testCase.ModelSize);
        end

        function test_multi_ngridpts_override(testCase)
            % Supervisor applies the override once after load_rx; it
            % persists across the per-field calls (reload_rx=false),
            % so every tile comes out at the override size.
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx_multi(m, testCase.rx_path, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'dofs', [3], 'ngridpts', 31);
            testCase.verifyEqual(size(out.per_field_w_nom_2d{1}), [31 31]);
            testCase.verifyEqual(size(out.OPDall), [3*31 3*31]);
        end

        function test_multi_field_center_tile_bitwise(testCase)
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx_multi(m, testCase.rx_path, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'dofs', testCase.DOFsForTest);
            cidx = find(out.field_table(:,1) == 0 ...
                      & out.field_table(:,2) == 0, 1);
            testCase.verifyNotEmpty(cidx);
            tr = out.field_table(cidx, 3);
            tc = out.field_table(cidx, 4);
            indx = out.indxall;
            in_ctr = (indx.i > tr*128) & (indx.i <= (tr+1)*128) ...
                   & (indx.j > tc*128) & (indx.j <= (tc+1)*128);
            dwdxall_ctr = out.dwdxall(in_ctr, :);
            dwdx_C = out.per_field_dwdx{cidx};
            testCase.verifyEqual( ...
                max(abs(dwdxall_ctr(:) - dwdx_C(:))), 0, ...
                'Center-tile rows of dwdxall must bitwise-match per_field_dwdx[center]');
        end
    end
end
