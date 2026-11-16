classdef tCodeVGrating < matlab.unittest.TestCase
%TCODEVGRATING  Port of pymacos's test_api_rx_grating.py.
%   17 tests covering the grating-element API.  Reference values
%   match pymacos's rx_data.Rx_Grating_001() since both share
%   libsmacos.a; numerical equality is expected at machine precision.
%
%   Layout: read tests in the first methods block, parametrized
%   read/write tests in the second.  matlab.unittest's TestParameter
%   gives one test instance per parameter value (analog of pytest's
%   @pytest.mark.parametrize).

    properties (Constant)
        ModelSize = 128
        AbsTol    = 1e-15
        RelTol    = 1e-15
    end

    properties (TestParameter)
        grating_order_     = struct('m4', -4, 'p4',  4)
        rule_width_        = struct('w005', 0.05, 'w100', 1.00)
        reflective_param   = struct('refl',  true, 'trans', false)
        direction_         = struct('x', [1 0 0], 'xyz', [1 1 1])
        compound_param_    = struct( ...
            'set1', struct('refl', true,  'order', -2, 'width', 5.0, 'dir', [1 0 0]), ...
            'set2', struct('refl', false, 'order',  3, 'width', 2.5, 'dir', [0 1 0]) ...
        )
    end

    properties
        data
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.data = rx_grating_001_data();
            macos.init(testCase.ModelSize);
        end
    end

    methods (TestMethodSetup)
        function setupMethod(testCase)
            macos.load_rx(rx_fixture_path(testCase.data.Rx));
            testCase.verifyEqual(macos.num_elt(), testCase.data.nElt);
        end
    end

    methods (Access = private)
        function [srf, ref_refl, ref_order, ref_width, ref_dir] = ...
                spec(testCase)
            % Unpack the single grating spec from data.gratings.
            srf       = testCase.data.gratings{1};
            ref_refl  = testCase.data.gratings{2};
            ref_order = testCase.data.gratings{3};
            ref_width = testCase.data.gratings{4};
            ref_dir   = testCase.data.gratings{5};
        end
    end

    % --- Read-only tests --------------------------------------------
    methods (Test)
        function test_grating_any(testCase)
            testCase.verifyTrue(macos.elt_grating_any());
        end

        function test_grating_fnd(testCase)
            [srf, ~, ~, ~, ~] = testCase.spec();
            s = macos.elt_grating_fnd();
            testCase.verifyEqual(s.srfs, srf);
        end

        function test_grating_params(testCase)
            [srf, ref_refl, ref_order, ref_width, ref_dir] = testCase.spec();
            p = macos.get_elt_grating_params(srf);
            testCase.verifyEqual(p.reflective, ref_refl);
            testCase.verifyEqual(p.diff_order, int32(ref_order));
            testCase.verifyEqual(p.rule_width, ref_width);
            testCase.verifyEqual(p.rule_dir, ref_dir, ...
                'AbsTol', testCase.AbsTol, 'RelTol', testCase.RelTol);
        end

        function test_grating_order(testCase)
            [srf, ~, ref_order, ~, ~] = testCase.spec();
            p = macos.get_elt_grating_params(srf);
            testCase.verifyEqual(p.diff_order, int32(ref_order));
            testCase.verifyEqual(macos.get_elt_grating_order(srf), ...
                int32(ref_order));
        end

        function test_grating_rulewidth(testCase)
            [srf, ~, ~, ref_width, ~] = testCase.spec();
            p = macos.get_elt_grating_params(srf);
            testCase.verifyEqual(p.rule_width, ref_width);
            testCase.verifyEqual(macos.get_elt_grating_rulewidth(srf), ref_width);
        end

        function test_grating_type(testCase)
            [srf, ref_refl, ~, ~, ~] = testCase.spec();
            p = macos.get_elt_grating_params(srf);
            testCase.verifyEqual(p.reflective, ref_refl);
            testCase.verifyEqual(macos.get_elt_grating_type(srf), ref_refl);
        end

        function test_grating_dir(testCase)
            [srf, ~, ~, ~, ref_dir] = testCase.spec();
            p = macos.get_elt_grating_params(srf);
            testCase.verifyEqual(p.rule_dir, ref_dir, ...
                'AbsTol', testCase.AbsTol, 'RelTol', testCase.RelTol);
            d = macos.get_elt_grating_dir(srf);
            testCase.verifyEqual(d, ref_dir, ...
                'AbsTol', testCase.AbsTol, 'RelTol', testCase.RelTol);
        end
    end

    % --- Parametrized read/write tests ------------------------------
    methods (Test)
        function test_grating_order_rw(testCase, grating_order_)
            [srf, ~, ref_order, ~, ~] = testCase.spec();
            testCase.verifyEqual(macos.get_elt_grating_order(srf), int32(ref_order));
            macos.set_elt_grating_order(srf, grating_order_);
            testCase.verifyEqual(macos.get_elt_grating_order(srf), int32(grating_order_));
            macos.set_elt_grating_order(srf, ref_order);
        end

        function test_grating_rulewidth_rw(testCase, rule_width_)
            [srf, ~, ~, ref_width, ~] = testCase.spec();
            testCase.verifyEqual(macos.get_elt_grating_rulewidth(srf), ref_width);
            macos.set_elt_grating_rulewidth(srf, rule_width_);
            testCase.verifyEqual(macos.get_elt_grating_rulewidth(srf), rule_width_);
            macos.set_elt_grating_rulewidth(srf, ref_width);
        end

        function test_grating_type_rw(testCase, reflective_param)
            [srf, ref_refl, ~, ~, ~] = testCase.spec();
            testCase.verifyEqual(macos.get_elt_grating_type(srf), ref_refl);
            macos.set_elt_grating_type(srf, reflective_param);
            testCase.verifyEqual(macos.get_elt_grating_type(srf), reflective_param);
            macos.set_elt_grating_type(srf, ref_refl);
        end

        function test_grating_dir_rw(testCase, direction_)
            [srf, ~, ~, ~, ref_dir] = testCase.spec();
            p = macos.get_elt_grating_params(srf);
            testCase.verifyEqual(p.rule_dir, ref_dir, ...
                'AbsTol', testCase.AbsTol, 'RelTol', testCase.RelTol);
            macos.set_elt_grating_dir(srf, direction_(:));
            expected = direction_(:) / norm(direction_);
            d = macos.get_elt_grating_dir(srf);
            testCase.verifyEqual(d, expected, ...
                'AbsTol', testCase.AbsTol, 'RelTol', testCase.RelTol);
            macos.set_elt_grating_dir(srf, ref_dir);
        end

        function test_grating_param_rw(testCase, compound_param_)
            [srf, ref_refl, ref_order, ref_width, ref_dir] = testCase.spec();
            % Read against reference.
            p = macos.get_elt_grating_params(srf);
            testCase.verifyEqual(p.reflective, ref_refl);
            testCase.verifyEqual(p.rule_width, ref_width);
            testCase.verifyEqual(p.diff_order, int32(ref_order));
            testCase.verifyEqual(p.rule_dir, ref_dir, ...
                'AbsTol', testCase.AbsTol, 'RelTol', testCase.RelTol);

            % Write composite struct.
            new_params.reflective = compound_param_.refl;
            new_params.rule_width = compound_param_.width;
            new_params.diff_order = compound_param_.order;
            new_params.rule_dir   = compound_param_.dir(:);
            macos.set_elt_grating_params(srf, new_params);

            % Read back.
            q = macos.get_elt_grating_params(srf);
            testCase.verifyEqual(q.reflective, compound_param_.refl);
            testCase.verifyEqual(q.rule_width, compound_param_.width);
            testCase.verifyEqual(q.diff_order, int32(compound_param_.order));
            expected_dir = compound_param_.dir(:) / norm(compound_param_.dir);
            testCase.verifyEqual(q.rule_dir, expected_dir, ...
                'AbsTol', testCase.AbsTol, 'RelTol', testCase.RelTol);

            % Restore.
            restore.reflective = ref_refl;
            restore.rule_width = ref_width;
            restore.diff_order = ref_order;
            restore.rule_dir   = ref_dir;
            macos.set_elt_grating_params(srf, restore);
        end
    end
end
