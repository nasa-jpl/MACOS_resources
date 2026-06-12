classdef tEndurance < matlab.unittest.TestCase
%TENDURANCE  Load/trace endurance — the design loop's exact profile.
%   Q5 (PLAN_DESIGN_LAYER §8 Sprint 0 / §9.1): loop load_rx -> trace
%   many times in one session and certify the foundation the design
%   layer's evaluate_ stacks on:
%     (1) rmsWFE is BIT-IDENTICAL every iteration (no state leak /
%         drift across repeated load_rx) — the hard correctness check;
%     (2) memory plateaus (no linear leak) — a generous guard against
%         a future regression, tolerant of one-time arena events.
%
%   On-demand only (~30 s): `./run_mmacos_tests.sh tEndurance`.  Not in
%   the fast suite or the `all` run.
%
%   Measured 2026-06-12 (model_size 128, Rx_Cass_FarField, 500 iters):
%   unique rmsWFE = 1 (max|diff| = 0); RSS warm-up through ~iter 175,
%   one ~14 MB arena event, then +0 kB/25-iter — flat steady state.

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
        N         = 300            % ~30 s at ~0.1 s/iter
        RSS_GUARD_KB = 100000      % 100 MB: catches a gross leak
    end                            % (240 kB/iter would be ~72 MB),
                                   % never flakes on the ~15 MB plateau

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
        function test_load_trace_endurance(testCase)
            N = testCase.N;
            w = zeros(N,1);

            % Optional RSS probe (Linux /proc only).
            pid = feature('getpid');
            statf = sprintf('/proc/%d/status', pid);
            haveRSS = isfile(statf);
            rssOf = @() str2double(regexp(fileread(statf), ...
                        'VmRSS:\s*(\d+)', 'tokens', 'once'));

            macos.load_rx(testCase.rx_path);   % prime
            macos.trace();
            rss0 = NaN; if haveRSS, rss0 = rssOf(); end

            for k = 1:N
                macos.load_rx(testCase.rx_path);
                s = macos.trace();
                w(k) = s.rmsWFE;
            end

            % (1) Bit-identical rmsWFE — the foundational correctness check.
            testCase.verifyEqual(numel(unique(w)), 1, ...
                sprintf(['rmsWFE drifted across %d load/trace iters ' ...
                         '(max|w-w(1)| = %.3e) — state leak in the ' ...
                         'design-loop profile.'], N, max(abs(w - w(1)))));

            % (2) Memory plateau — generous guard against a future leak.
            if haveRSS
                growth = rssOf() - rss0;
                testCase.verifyLessThan(growth, testCase.RSS_GUARD_KB, ...
                    sprintf(['RSS grew %d kB over %d iters (%.1f kB/iter) ' ...
                             '— possible load/trace leak; expected a ' ...
                             'bounded plateau.'], growth, N, growth/N));
            else
                testCase.assumeFail('No /proc RSS probe on this platform; skipped memory guard.');
            end
        end
    end
end
