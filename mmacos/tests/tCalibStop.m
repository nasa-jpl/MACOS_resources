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
%   OptTol= (PLAN 3.2): the tDesignTelescope TMA conic solve ('fourth_mirror' leg,
%   2 fields, 60 iterations) stops 'converged' at the default tolerance; emitted
%   with 'tol' 1e-30 (Telescope.optimize writes OptTol=, the Rx keyword) no step can
%   be flat in absolute rms, so it must run on to the floor or the cap.  Pre-fix:
%   the keyword was not parsed (dropped silently) -- the two runs were identical.
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
            % a RANGE (Dave 2026-10-19): x86_64 reaches the ten-rejection floor at ~14, arm64's LM took 29
            tc.verifyTrue(any(strcmp(r.stop_reason, {'floor', 'cap'})), ['a quadratic solve never takes a ' ...
                'flat step: the ten-rejection floor (or the cap) stops it, got ' char(r.stop_reason)]);
            tc.verifyLessThanOrEqual(r.n_iter, 40);
            tc.verifyGreaterThanOrEqual(r.n_iter, 11, 'the floor needs ten rejected trials');
            r = tc.run_(10);
            tc.verifyEqual(r.stop_reason, 'cap', 'T=10 ends before the floor (~14)');
            tc.verifyEqual(r.n_iter, 10);
            r = tc.run_(2);
            tc.verifyEqual(r.stop_reason, 'cap');
            tc.verifyEqual(r.n_iter, 2);
        end
        function test_opt_tol_keyword_reaches_the_stop(tc)
            st = cell(1,2);  tol = {[], 1e-30};
            for k = 1:2
                t = macos.design.Telescope('family','TMA', 'aperture_diameter_m',1.0, ...
                    'model_size',128, 'grid_npts',21);
                t.add_mirror('M1','radius_m',8.0,'spacing_after_m',3.0);
                t.add_mirror('M2','radius_m',2.0,'spacing_after_m',4.5);
                t.add_mirror('M3','radius_m',4.0,'spacing_after','derive');
                t.build();
                args = {'fields_arcmin',[0.5 1.0], 'dofs',[0 0 0 0 0 0 0 1], 'max_iters',60};
                if ~isempty(tol{k}), args = [args {'tol', tol{k}}]; end %#ok<AGROW>
                t.optimize(args{:});
                st{k} = macos.calib_stats();
            end
            tc.verifyEqual(st{1}.stop_reason, 'converged', ...
                'default OptTol: the TMA conic solve takes two flat accepted steps');
            tc.verifyNotEqual(st{2}.stop_reason, 'converged', ...
                'OptTol= 1e-30: no step is flat in absolute rms, so it cannot converge');
            tc.verifyGreaterThan(st{2}.n_iter, st{1}.n_iter, 'and it runs longer');
            tc.verifyLessThanOrEqual(st{2}.merit, st{1}.merit, 'never to a worse merit');
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
