classdef tDesignOptimize < matlab.unittest.TestCase
%TDESIGNOPTIMIZE  macos.design.System.evaluate / optimize (Sprint 2A-i, slice 4).
%   Closes 2A-i: the import analysis surface becomes an optimization
%   surface.  Controlled convergence on e5hex1 — a single-DOF despace on
%   an optical element: perturbing from nominal raises WFE, and optimize()
%   drives it back down.

    properties (Constant)
        ModelSize = 128
        RxName    = 'e5hex1.in'
        ELT       = 2            % an optical element
        OFFSET_MM = 0.10         % despace disturbance
        BOUNDS_MM = [-1 1]
    end

    properties
        sys
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            rxp = rx_fixture_path(testCase.RxName);
            testCase.sys = macos.design.System.from_rx(rxp, ...
                'model_size', testCase.ModelSize);
        end
    end

    methods (TestMethodSetup)
        function resetVars(testCase)
            testCase.sys.clear_vars();
        end
    end

    methods (Test)
        function test_perturbation_raises_wfe(testCase)
            s = testCase.sys;
            s.vary(testCase.ELT, 'despace', 'bounds', testCase.BOUNDS_MM, 'unit', 'mm');
            m_nom = s.evaluate(0).merit;
            m_off = s.evaluate(testCase.OFFSET_MM).merit;
            testCase.verifyGreaterThan(m_off, m_nom);     % despace hurts WFE
            % evaluate(0) reproduces nominal on repeat (restore is clean)
            testCase.verifyEqual(s.evaluate(0).merit, m_nom, 'RelTol', 1e-9);
        end

        function test_optimize_recovers_alignment(testCase)
            testCase.assumeTrue(~isempty(which('fmincon')), ...
                'Optimization Toolbox (fmincon) not available.');
            s = testCase.sys;
            s.vary(testCase.ELT, 'despace', 'bounds', testCase.BOUNDS_MM, 'unit', 'mm');
            m_nom = s.evaluate(0).merit;
            res = s.optimize('x0', testCase.OFFSET_MM, 'MaxIter', 25);
            % started offset, ended better than the start...
            testCase.verifyLessThan(res.merit_opt, res.merit0);
            % ...and recovered to roughly nominal (offset was the disturbance)
            testCase.verifyLessThanOrEqual(res.merit_opt, 1.5 * m_nom);
            % solution moved back toward nominal
            testCase.verifyLessThan(abs(res.x_opt), testCase.OFFSET_MM);
        end

        function test_evaluate_requires_vars(testCase)
            s = testCase.sys;
            testCase.verifyError(@() s.evaluate(0), ...
                'macos:design:System:noVars');
        end

        function test_evaluate_rejects_nonrigid(testCase)
            s = testCase.sys;
            s.vary(testCase.ELT, 'zern', 'mode', 5, 'bounds', [-1e-8 1e-8]);
            testCase.verifyError(@() s.evaluate(0), ...
                'macos:design:System:optimFamily');
        end
    end
end
