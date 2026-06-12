classdef tDesignSystem < matlab.unittest.TestCase
%TDESIGNSYSTEM  macos.design.System import core (Sprint 2A-i, slice 1).
%   Verifies from_rx engine-readback: the spec it builds must match what
%   the direct +macos getters return for the same loaded Rx (from_rx is
%   a faithful readback, not a reinterpretation).

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
        ExpectedNelt = 6
    end

    properties
        rx_path
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.rx_path = rx_fixture_path(testCase.RxName);
        end
    end

    methods (Test)
        function test_from_rx_returns_system(testCase)
            s = macos.design.System.from_rx(testCase.rx_path, ...
                'model_size', testCase.ModelSize);
            testCase.verifyClass(s, 'macos.design.System');
            testCase.verifyEqual(s.n_elt(), testCase.ExpectedNelt);
        end

        function test_spec_structure(testCase)
            s = macos.design.System.from_rx(testCase.rx_path, ...
                'model_size', testCase.ModelSize);
            sp = s.spec;
            testCase.verifyEqual(sp.source, 'import');
            testCase.verifyEqual(sp.rx_path, testCase.rx_path);
            testCase.verifyEqual(sp.model_size, testCase.ModelSize);
            for fn = {'units','src','n_elt','elt'}
                testCase.verifyTrue(isfield(sp, fn{1}), ...
                    sprintf('spec missing field %s', fn{1}));
            end
            testCase.verifyTrue(isfield(sp.units, 'cbm'));
            testCase.verifyGreaterThan(sp.units.cbm, 0);
            testCase.verifyEqual(numel(sp.elt), testCase.ExpectedNelt);
        end

        function test_readback_matches_getters(testCase)
            % from_rx must reproduce the direct getter values exactly —
            % it reads through the same engine surface, so this is bitwise.
            s = macos.design.System.from_rx(testCase.rx_path, ...
                'model_size', testCase.ModelSize);
            sp = s.spec;
            % The System left the Rx loaded; query getters directly.
            testCase.verifyEqual(sp.n_elt, macos.num_elt());
            testCase.verifyEqual(sp.src.wvl, macos.get_src_wvl());
            testCase.verifyEqual(sp.src.sampling, macos.get_src_sampling());
            for k = 1:sp.n_elt
                testCase.verifyEqual(sp.elt(k).vpt, macos.get_elt_vpt(k));
                testCase.verifyEqual(sp.elt(k).rpt, macos.get_elt_rpt(k));
                testCase.verifyEqual(sp.elt(k).psi, macos.get_elt_psi(k));
                testCase.verifyEqual(sp.elt(k).provenance, 'imported');
            end
        end

        function test_elt_vectors_are_column_3vecs(testCase)
            s = macos.design.System.from_rx(testCase.rx_path, ...
                'model_size', testCase.ModelSize);
            for k = 1:s.n_elt()
                testCase.verifyEqual(size(s.spec.elt(k).vpt), [3 1]);
                testCase.verifyEqual(size(s.spec.elt(k).psi), [3 1]);
            end
        end

        function test_describe_runs_clean(testCase)
            s = macos.design.System.from_rx(testCase.rx_path, ...
                'model_size', testCase.ModelSize);
            % describe() prints; just confirm it doesn't error.
            evalc('s.describe()');
        end
    end
end
