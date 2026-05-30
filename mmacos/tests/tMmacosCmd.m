classdef tMmacosCmd < matlab.unittest.TestCase
%TMMACOSCMD  Regression tests for the raw `mmacos('cmd', ...)` surface.
%   Exercises the mex layer directly (no +macos package, no Session
%   class).  These are the lowest-level tests; if they break, the
%   higher-level surfaces will too.

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
            mmacos('init', testCase.ModelSize);
        end
    end

    methods (TestMethodSetup)
        function setupMethod(testCase)
            % Fresh Rx every method so state doesn't leak between tests.
            % Strip the trailing .in (macos's OLD appends it).
            mmacos('load_rx', testCase.rx_path(1:end-3));
        end
    end

    methods (Test)
        function test_n_elt(testCase)
            testCase.verifyEqual(mmacos('n_elt'), testCase.ExpectedNelt);
        end

        function test_base_unit_to_metres_positive(testCase)
            cbm = mmacos('base_unit_to_metres');
            testCase.verifyGreaterThan(cbm, 0);
        end

        function test_modified_rx_idempotent(testCase)
            mmacos('modified_rx');
            mmacos('modified_rx');
            testCase.verifyEqual(mmacos('n_elt'), testCase.ExpectedNelt);
        end

        function test_trace_returns_nRays(testCase)
            [nRays, rmsWFE] = mmacos('trace_rays', testCase.ExpectedNelt);
            testCase.verifyGreaterThan(nRays, 0);
            testCase.verifyGreaterThanOrEqual(rmsWFE, 0);
        end

        function test_opd_shape(testCase)
            mmacos('trace_rays', testCase.ExpectedNelt);
            W = mmacos('opd');
            testCase.verifyTrue(ismatrix(W));
            testCase.verifyEqual(size(W, 1), size(W, 2));
        end

        function test_intensity_peak_positive(testCase)
            mmacos('trace_rays', testCase.ExpectedNelt);
            I = mmacos('intensity', testCase.ExpectedNelt, 1.0);
            testCase.verifyGreaterThan(max(I(:)), 0);
        end

        function test_complex_field_is_complex(testCase)
            mmacos('trace_rays', testCase.ExpectedNelt);
            cf = mmacos('complex_field', testCase.ExpectedNelt, 1.0);
            testCase.verifyFalse(isreal(cf));
        end

        function test_dx_at_positive_scalar(testCase)
            mmacos('trace_rays', testCase.ExpectedNelt);
            dx = mmacos('dx_at', testCase.ExpectedNelt);
            testCase.verifyClass(dx, 'double');
            testCase.verifyGreaterThan(dx, 0);
        end

        function test_get_src_sampling_positive(testCase)
            n = mmacos('get_src_sampling');
            testCase.verifyGreaterThan(n, 0);
        end

        function test_get_src_wvl_positive(testCase)
            wvl = mmacos('src_wvl', 0.0, 0);
            testCase.verifyGreaterThan(wvl, 0);
        end

        function test_elt_vpt_get_3vec(testCase)
            vpt = mmacos('elt_vpt', 2, zeros(3,1), 0, 1);
            testCase.verifyEqual(numel(vpt), 3);
        end

        function test_prb_elt_array_form(testCase)
            % Apply +1e-6 then -1e-6 around x; should run without error.
            mmacos('prb_elt', 2, [1e-6;0;0;0;0;0], 1);
            mmacos('prb_elt', 2, [-1e-6;0;0;0;0;0], 1);
            mmacos('modified_rx');
            [nRays, ~] = mmacos('trace_rays', testCase.ExpectedNelt);
            testCase.verifyGreaterThan(nRays, 0);
        end

        function test_perturb_elt_single_form(testCase)
            % Codegen-emitted single-element form.  Args: iElt, th(3),
            % del(3), useLocalCoord.
            mmacos('perturb_elt', 2, [1e-6;0;0], [0;0;0], 0);
            mmacos('perturb_elt', 2, [-1e-6;0;0], [0;0;0], 0);
            mmacos('modified_rx');
            [nRays, ~] = mmacos('trace_rays', testCase.ExpectedNelt);
            testCase.verifyGreaterThan(nRays, 0);
        end

        function test_elt_grp_max_all(testCase)
            % No element groups defined in Rx_Cass_FarField — expect 0.
            m = mmacos('elt_grp_max_all');
            testCase.verifyEqual(m, 0);
        end
    end
end
