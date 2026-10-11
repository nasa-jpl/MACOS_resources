classdef tCalibStop < matlab.unittest.TestCase
%TCALIBSTOP  CALIB's LM stopping rule and its report (PLAN_CONSOLIDATION 3.3, 2026-10-11).
%   The old loop checked convergence only at itr = int(.21T/.5T/.71T) and only for
%   T > 4, judged the RMS WFE whatever the target, and exited on itr_done -- a
%   logical never set before the loop and STATIC under nls_optim_dvr's blanket
%   Save, so a T=4 CALIB exited on whatever the PREVIOUS CALIB had left (measured
%   on the CLI: the same deck 4 iterations fresh, 3 after a converged T=20 CALIB).
%   Now LM judges its own merit, rms = sqrt(chisq/rows) in the target's units:
%   'converged' = two consecutive accepted steps each moving it by <= OptTol
%   (absolute), 'floor' = ten consecutive rejected trials, 'cap' = OptMaxItrs.
%   macos.calib returns .n_iter / .merit / .stop_reason; macos.calib_stats reads
%   them (also after a failed run).
%   Fixture Rx_AsphCalib.in (SPOT target, h^4 spoiled) converges QUADRATICALLY to
%   the round-off floor by itr 3-4 (1.6e-25 -> 1.6e-33 -> 5.4e-43 -> 1.7e-43), so no
%   accepted step is ever flat: measured 'floor' at itr 14 with T=20, 'cap' with
%   T=10 (the floor would land at 14).  'converged' needs a non-quadratic tail (the
%   tDesignTelescope fourth_mirror / add_fold designs, itr 41) -- recorded in
%   macos cli_tests/records/calib_loop_old_vs_new.md, not pinned here.
%   Pre-fix mex: no calib_stats_get, every test red.
    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_AsphCalib.in'
    end
    methods (Access = private)
        function r = run_(tc, n_iter, tol)
            m = macos.Session(tc.ModelSize);
            m.load_rx(rx_fixture_path(tc.RxName));
            m.calib_set_iter(n_iter);
            if nargin > 2, m.calib_set_tol(tol); end
            r = m.calib();
        end
    end
    methods (Test)
        function test_the_count_does_not_depend_on_the_previous_calib(tc)
            r1 = tc.run_(4);
            r2 = tc.run_(20);
            r3 = tc.run_(4);
            tc.verifyEqual(r3.n_iter, r1.n_iter, sprintf(['the same T=4 solve took %d ' ...
                'iterations fresh and %d after a T=%d CALIB (pre-fix CLI: 4 then 3)'], ...
                r1.n_iter, r3.n_iter, 20));
            tc.verifyEqual(r3.merit, r1.merit, 'and ends at the same merit');
            tc.verifyLessThanOrEqual(r2.n_iter, 20);
        end
        function test_each_stop_reason_is_reached_and_reported(tc)
            r = tc.run_(20);
            tc.verifyEqual(r.stop_reason, 'floor', ['a quadratic solve never takes a ' ...
                'flat step: the ten-rejection floor stops it']);
            tc.verifyLessThan(r.n_iter, 20);
            tc.verifyGreaterThanOrEqual(r.n_iter, 11, 'the floor needs ten rejected trials');
            r = tc.run_(10);
            tc.verifyEqual(r.stop_reason, 'cap', 'T=10 ends before the floor (~14)');
            tc.verifyEqual(r.n_iter, 10);
            r = tc.run_(2);
            tc.verifyEqual(r.stop_reason, 'cap');
            tc.verifyEqual(r.n_iter, 2);
        end
        function test_calib_stats_matches_the_result(tc)
            r  = tc.run_(20);
            st = macos.calib_stats();
            tc.verifyEqual(st.n_iter, r.n_iter);
            tc.verifyEqual(st.merit, r.merit);
            tc.verifyEqual(st.stop_reason, r.stop_reason);
            tc.verifyGreaterThanOrEqual(r.merit, 0, 'the merit is a chisq');
        end
    end
end
