classdef tE2eCycles < matlab.unittest.TestCase
%TE2ECYCLES  e2e stage 11: the recompute-and-restart cycle driver (e2e_cycles) on the coupled model.
%   The driver re-measures the Jacobian about the loop's own state and
%   restarts the probed loop from it, cycle after cycle, resumably.  Here
%   on the committed coupled deck behind stage 9's annular stop with small
%   Jacobians (8 actuators per DM, dark zone 3-5 lambda/D, about four
%   minutes each on the coupled train), every file in a scratch directory.
%
%   Checks:
%     (a) two cycles: the second cycle's Jacobian is about the first
%         cycle's end state (to the bit), every cycle ends no brighter than
%         it starts, and the probe estimate at each fresh Jacobian is
%         within a pinned error of the field (the stage-10 control,
%         generalized: a Jacobian about the state estimates well); the
%         state file carries both cycles
%     (b) resume: the same call again measures nothing and returns the same
%         chain; raising the cap adds exactly one cycle
%     (c) the stop rules: a gain threshold no cycle can meet stops the chain
%         'converged' after cycle 0; a goal already met stops it 'goal';
%         both reuse the flat Jacobian of (a) (the reuse rule: same state)
%   Model 512 (the CTB's), the SUITE_CTB_512 group of run_mmacos_tests.sh.
%
%   PLAN_CONSOLIDATION 3a, the cycles (CCL, 2026-10-30).

    properties (Constant)
        ModelSize = 512
        LyotInner = 0.08
        PickRel   = 0.3
        JacOpts   = {'nact', 8, 'inner_lamD', 3, 'outer_lamD', 5}
        RelErrMaxFresh = 3e-2       % (a) the estimate at a fresh Jacobian: measured 1.444e-2 at cycle 0 with the
                                    %     8-actuator probes (tE2eSensing's 12-actuator ones: 1.2e-3 at 0.3x)
    end

    properties
        work, cfg
    end

    methods (TestClassSetup)
        function setup(tc)
            root = fullfile(fileparts(mfilename('fullpath')), '..');
            run(fullfile(root, 'mmacos_setup.m'));
            e2e   = fullfile(root, 'templates', '80_end_to_end', 'e2e');
            bench = fullfile(root, 'templates', '30_instruments', 'bench_ctb');
            addpath(bench);  addpath(e2e);
            compact = fullfile(e2e, 's9_coupled.in');
            tc.assumeTrue(isfile(compact), 'the e2e stage-9 deck is not committed here');
            tc.work = tempname;  mkdir(tc.work);
            tc.addTeardown(@() rmdir(tc.work, 's'));
            prev = cd(tc.work);  tc.addTeardown(@() cd(prev));
            rr = ctb_dm_rx('rx_in', compact, 'rx_out', fullfile(tc.work, 's9_coupled_dm.in'));
            names = regexp(fileread(compact), '(?m)^\s*EltName=\s*(\S+)', 'tokens');  names = [names{:}];
            ix = @(nm) find(strcmp(names, nm), 1);
            e = struct('DM1', ix('DM1'), 'DM2', ix('DM2'), 'Apodizer', ix('Apodizer'), 'FPM', ix('FPM'), ...
                       'Lyot', ix('Lyot'), 'ExitPupil', ix('ExitPupil'), 'FPA', ix('FPA'));
            chain = {'elt', e, 'r_lyot_inner_frac', tc.LyotInner};
            ch = ctb_chain('rx', rr.rx_out, 'model_size', tc.ModelSize, chain{:});
            tc.cfg = struct('exdir', tc.work, 'tag', 'tst', 'r', rr, 'e', e, 'ch', ch, 'chain_pick', {chain}, 'MODEL', tc.ModelSize, ...
                            'sensing', struct('probe_rel', tc.PickRel), 'a0', {{}}, 'niter', 3, 'alphas', logspace(-6, 0, 7), ...
                            'max_cycles', 1, 'stop_gain', 1.0, 'goal', 0, 'jac_opts', {tc.JacOpts}, 'say', @(varargin) fprintf('%s\n', sprintf(varargin{:})));
        end
    end

    methods (Test)
        function test_two_cycles_then_resume(tc)
            S = e2e_cycles(tc.cfg);
            tc.verifyEqual(numel(S.cycles), 2, 'cycles 0 and 1');
            tc.verifyEqual(S.stop_reason, 'cap');
            tc.verifyTrue(isfile(S.state_file), 'the state file is written');
            J1 = load(fullfile(tc.work, sprintf('ctb_dm_jacobian_N%d_e2e_tst_c1.mat', tc.ModelSize)), 'a0', 'dm');
            tc.verifyEqual(J1.a0, S.cycles(1).a_end, 'the second cycle''s Jacobian is about the first cycle''s end state');
            tc.verifyEqual(J1.dm(1).nact, 8, 'the test''s small Jacobian');
            tc.verifyTrue(all([S.cycles.c_end] <= [S.cycles.c_start]), 'every cycle ends no brighter than it starts');
            tc.verifyTrue(all(isfinite([S.cycles.est_rel0])), 'the estimate at every fresh Jacobian is reported');
            tc.verifyLessThan(max([S.cycles.est_rel0]), tc.RelErrMaxFresh, sprintf( ...
                'the estimate at a Jacobian about the state: %s', mat2str([S.cycles.est_rel0], 3)));
            tc.verifyEqual(S.a_final, S.cycles(end).a_end);
            % resume: nothing measured, the same chain back
            t1 = tic;  S2 = e2e_cycles(tc.cfg);  dt = toc(t1);
            tc.verifyEqual(numel(S2.cycles), 2);
            tc.verifyEqual(S2.c_final, S.c_final);
            tc.verifyLessThan(dt, 30, sprintf('a resume at the cap measures nothing (%.1f s)', dt));
            % raise the cap: exactly one more cycle, about the chain's end state
            c3 = tc.cfg;  c3.max_cycles = 2;
            S3 = e2e_cycles(c3);
            tc.verifyEqual(numel(S3.cycles), 3, 'one more cycle');
            tc.verifyEqual(S3.cycles(3).k, 2);
            tc.verifyFalse(S3.cycles(3).jac_reused, 'the new cycle measured its own Jacobian');
            J2 = load(fullfile(tc.work, sprintf('ctb_dm_jacobian_N%d_e2e_tst_c2.mat', tc.ModelSize)), 'a0');
            tc.verifyEqual(J2.a0, S.cycles(2).a_end, 'about the previous chain end');
            tc.verifyLessThanOrEqual(S3.c_final, S.c_final, 'the third cycle does not undo the second');
        end

        function test_stop_rules_reuse_the_flat_jacobian(tc)
            % the flat Jacobian of the first chain (tag e2e_tst_c0) is about the same (flat) state
            cg = tc.cfg;  cg.tag = 'tst_gain';  cg.stop_gain = 1e9;  cg.reuse = {'e2e_tst_c0'};
            tc.assumeTrue(isfile(fullfile(tc.work, sprintf('ctb_dm_jacobian_N%d_e2e_tst_c0.mat', tc.ModelSize))), 'the first chain''s flat Jacobian is needed');
            Sg = e2e_cycles(cg);
            tc.verifyEqual(numel(Sg.cycles), 1);
            tc.verifyEqual(Sg.stop_reason, 'converged', 'a gain no cycle can meet stops the chain after cycle 0');
            tc.verifyTrue(Sg.cycles(1).jac_reused, 'cycle 0 reused the flat Jacobian');
            tc.verifyEqual(Sg.cycles(1).jac_tag, 'e2e_tst_c0');
            cgo = tc.cfg;  cgo.tag = 'tst_goal';  cgo.goal = 1;  cgo.reuse = {'e2e_tst_c0'};
            So = e2e_cycles(cgo);
            tc.verifyEqual(numel(So.cycles), 1);
            tc.verifyEqual(So.stop_reason, 'goal', 'a goal already met stops the chain');
        end
    end
end
