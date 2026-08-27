classdef tCfCampaign < matlab.unittest.TestCase
%TCFCAMPAIGN  Pins for the e2e6m coronagraph-family campaign (CF0..S5).
%
%   The brief's S6 gate: the CF2 table and the S3b winner floor are
%   PINNED so regeneration drift is loud.  Two kinds of check:
%   records-integrity pins (the committed run-state .mats carry the
%   published numbers) and live engine gates (the winner's static
%   reproduces through the chain; the set_lambda memoization is
%   lambda-correct).  Registered in SUITE_CTB_512 (the campaign's
%   model size).

    properties
        rd   % the e2e6m_r2 template dir
    end

    methods (TestClassSetup)
        function setup(tc)
            here = fileparts(mfilename('fullpath'));
            root = fileparts(here);
            run(fullfile(root, 'mmacos_setup.m'));
            tc.rd = fullfile(root, 'templates', '80_end_to_end', 'e2e6m_r2');
            addpath(tc.rd);
            addpath(fullfile(root, 'templates', '30_instruments', 'bench_ctb'));
            tc.assumeTrue(isfolder(tc.rd), 'e2e6m_r2 not present');
        end
    end

    methods (Test)
        function test_cf2_table_pins(tc)
        % The published S2 table, from the committed run states.
            pins = { % key      static      relin
                'hard', 3.503e-06, 3.935e-07;
                'apl',  4.485e-07, 1.081e-07;
                'aplc', 1.074e-06, 7.519e-07;
                'blc',  1.636e-06, 1.257e-06;
                'v4',   1.625e-05, 5.377e-06;
                'v6',   1.844e-05, 3.925e-06};
            for k = 1:size(pins, 1)
                S = load(fullfile(tc.rd, sprintf('cf2_%s_run.mat', pins{k,1})));
                tc.verifyEqual(S.res.c_static, pins{k,2}, 'RelTol', 1e-3, ...
                    sprintf('%s static drifted', pins{k,1}));
                tc.verifyEqual(S.res.c_relin, pins{k,3}, 'RelTol', 1e-3, ...
                    sprintf('%s relin floor drifted', pins{k,1}));
                la = S.res.c_relin / S.res.la1_ach.floor;
                tc.verifyLessThan(la, 2.2, sprintf( ...
                    '%s no longer substrate-limited (la %.2fx)', pins{k,1}, la));
            end
        end

        function test_s3b_knee_is_the_baseline(tc)
        % The spacing sweep's verdict: the apl floor is minimal at 0.15 m.
            S = load(fullfile(tc.rd, 'cf3b_run.mat'));
            R = S.OUT.R(strcmp({S.OUT.R.leg}, 'apl'));
            [~, kn] = min([R.c_relin]);
            tc.verifyEqual(R(kn).d, 0.15, 'AbsTol', 1e-9, ...
                'the spacing knee moved off the baseline');
            tc.verifyEqual(R(kn).c_relin, 1.081e-07, 'RelTol', 1e-3);
        end

        function test_gate_operating_point_recorded(tc)
            txt = fileread(fullfile(tc.rd, 'cf_gate_proposal.txt'));
            tc.verifySubstring(txt, 'apodized Lyot');
            tc.verifySubstring(txt, '0.90');
            tc.verifySubstring(txt, '0.15 m');
        end

        function test_s5_toned_series_pins(tc)
        % The 0.3 nm drift series: open loop flat, closed hold at the
        % mechanization floor, estimator tracking.
            S = load(fullfile(tc.rd, 'r4_run.mat'));
            tc.verifyEqual(rms(S.OUT.X(:, end)), 2.955e-10, 'RelTol', 0.05, ...
                'the toned drift amplitude drifted');
            co = S.OUT.cor.con;   co = co(isfinite(co));
            un = S.OUT.unc.con;   un = un(isfinite(un));
            tc.verifyEqual(co(end), 2.464e-07, 'RelTol', 0.02, ...
                'the closed-loop hold moved');
            tc.verifyLessThan(abs(un(end)/un(1) - 1), 0.05, ...
                'the open loop is no longer flat at 0.3 nm drift');
        end

        function test_winner_static_reproduces_live(tc)
        % ENGINE: the gate operating point's static through the chain.
            C1 = load(fullfile(tc.rd, 'cf1_run.mat'));
            FC = struct();
            for k = 1:numel(C1.OUT.F), FC.(C1.OUT.F(k).key) = C1.OUT.F(k); end
            P = e2e6m_r2_params();
            ch = cf_chain('rx', fullfile(tc.rd, 'r1_seg_dm.in'), ...
                'model_size', P.dj.model, 'prolate_iter', P.co.prolate_iter, ...
                'circ_stop_frac', P.cf.circ_stop_frac, FC.apl.cfg{:});
            E = ch.run();
            dz = find(ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD));
            con = mean(abs(E(dz)).^2) / ch.peak_bare;
            tc.verifyEqual(con, 4.485e-07, 'RelTol', 0.01, ...
                'the winner static no longer reproduces through the chain');
        end

        function test_lambda_memo_is_correct_live(tc)
        % ENGINE: the S4 memoization gate (asserts internally).
            cf4_memo_gate();
            tc.verifyTrue(true);
        end
    end
end
