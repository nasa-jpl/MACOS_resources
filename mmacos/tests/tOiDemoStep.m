classdef tOiDemoStep < matlab.unittest.TestCase
%TOIDEMOSTEP  Gates for the live adjacent-problem demo driver.
%
%   OI_DEMO_STEP takes a field-box full width from the audience, warm-
%   starts from the committed OI_WALK frontier, states the predicted
%   outcome BEFORE solving, and runs ONE full-freedom S5 continuation
%   step.  These gates pin the four things the demo beat rests on:
%
%     A  the pre-generated fallback and a fresh run AGREE -- the number
%        shown on stage is reproducible, not a lucky run;
%     B  the warm start is the committed step BELOW the ask, so an ask AT
%        a committed width is a real continuation step (the driver
%        working) and not a re-score of an already-solved design; and the
%        result lands in that committed row's neighbourhood;
%     C  an out-of-range ask REFUSES, before solving, with one accurate
%        sentence (the reserve path -- gated even though it is
%        deliberately not demonstrated live);
%     D  the verdict block's frontier-bracket line quotes the CORRECT
%        committed rows.
%
%   NOT REGISTERED IN ANY SUITE.  Two full demo-knob solves is ~30 min of
%   machine time (the same reason the five-stage offset_imager runs are
%   not in a suite).  Run it explicitly:
%
%     MACOS_HOME=/home/dcr/dev/macos/macos_f90 matlab -batch \
%       "run('<mmacos>/mmacos_setup.m'); \
%        r = runtests('<mmacos>/tests/tOiDemoStep.m'); disp(table(r)); exit(0)"
%
%   Registration in SUITE_FREEFORM (model size 256) is Dave's call at
%   review; the class is written so that it can be added unchanged.
%
%   PRE-GEN DEPENDENCY: gates A and D read the committed fallback bundle
%   demo_adjacent/oi_demo_12deg_run.mat, produced at the pinned knobs by
%   run_oi_demo.  If that bundle is regenerated at different knobs, the
%   knobs here must move with it (they are read FROM the bundle, so the
%   only thing to keep in step is this comment).
%
%   See also OI_DEMO_STEP, RUN_OI_DEMO, OI_WALK, tests/tOffsetImager.m.

    properties
        oidir           % the offset_imager template directory
        outdir          % scratch artifacts
        pregen          % the committed 12x12 deg fallback record
        O12             % a fresh 12 deg run at the pre-gen's own knobs
        O11             % an AT-a-committed-step (11 deg) ask
        rec             % the committed walk record
    end

    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(mfilename('fullpath'));
            run(fullfile(fileparts(h),'mmacos_setup.m'));
            tc.oidir = fullfile(fileparts(h),'templates','10_telescopes', ...
                                'offset_imager');
            addpath(tc.oidir);
            macos.init(256);

            W = load(fullfile(tc.oidir,'t5_walk','t5_walk_run.mat'));
            tc.rec = W.rec;

            f = fullfile(tc.oidir,'demo_adjacent','oi_demo_12deg_run.mat');
            tc.assumeTrue(exist(f,'file') == 2, ...
                sprintf(['the committed fallback bundle is missing (%s) -- ' ...
                         'regenerate it with run_oi_demo before gating'], f));
            S = load(f);
            tc.pregen = S.OUT;

            tc.outdir = tempname;  mkdir(tc.outdir);

            % Re-run at the PRE-GEN's own knobs, read from its own record --
            % never hard-coded here, so the gate follows a re-pin.  ALL
            % three solve knobs are carried: solve_sampling is as much a
            % result-determining knob as the other two (it moved the 12 deg
            % map 35.23 -> 33.55 nm), so leaving it to the wrapper default
            % would let a re-pin silently break reproducibility.
            kv = {'gn_iters',       tc.pregen.P.gn_iters, ...
                  'nsolve',         tc.pregen.P.nsolve, ...
                  'solve_sampling', tc.pregen.P.solve_sampling};
            tc.O12 = oi_demo_step(12, 'outdir',tc.outdir, 'tag','g12', kv{:});
            tc.O11 = oi_demo_step(11, 'outdir',tc.outdir, 'tag','g11', kv{:});
        end
    end

    methods (TestClassTeardown)
        function teardown(tc)
            if ~isempty(tc.outdir) && exist(tc.outdir,'dir')
                rmdir(tc.outdir,'s');
            end
        end
    end

    methods (Test)

        % ---- A: the stage number is reproducible ------------------------
        function test_fresh_run_reproduces_the_pregen_verdict(tc)
            p = tc.pregen;  o = tc.O12;
            tc.verifyEqual(o.verdict, p.verdict, ...
                'the fresh run must reach the same verdict as the fallback');
            tc.verifyEqual(o.warm.step, p.warm.step, ...
                'the fresh run must warm-start from the same committed step');
            % The solve is deterministic (damped GN, no randomness), so this
            % is an EQUALITY check with a numerical-noise tolerance, not a
            % "close enough" band.
            tc.verifyEqual(o.map.max_nm, p.map.max_nm, 'RelTol', 1e-6, ...
                'the dense-map maximum must reproduce');
            tc.verifyEqual(o.gates.clear_min_m, p.gates.clear_min_m, ...
                'RelTol', 1e-6, 'the clearance floor must reproduce');
            tc.verifyEqual(o.gates.exit_err_deg, p.gates.exit_err_deg, ...
                'AbsTol', 1e-6, 'the exit-direction error must reproduce');
        end

        function test_the_pregen_bundle_is_complete(tc)
            d = fullfile(tc.oidir,'demo_adjacent');
            for f = {'oi_demo_12deg_map.png', 'oi_demo_12deg_layout.png', ...
                     'oi_demo_12deg_fields.png', 'oi_demo_12deg_verdict.txt', ...
                     'oi_demo_12deg.in', 'REHEARSAL.md'}
                tc.verifyTrue(exist(fullfile(d,f{1}),'file') == 2, ...
                    sprintf('fallback asset missing: %s', f{1}));
            end
        end

        % ---- B: the warm-start rule -------------------------------------
        function test_an_ask_at_a_committed_step_starts_from_the_step_below(tc)
            % 11 deg IS committed (walk step 3).  The rule is STRICTLY
            % below, so the start must be step 2 (8 deg) -- otherwise the
            % "live" run would re-score a design already solved at that box.
            W = [tc.rec.width];
            tc.assertEqual(W(3), 11, 'the walk record moved; re-pin this gate');
            tc.verifyEqual(tc.O11.warm.step, 2, ...
                'an 11 deg ask must warm-start from the 8 deg step, not the 11');
            tc.verifyEqual(tc.O11.warm.width, 8, ...
                'the warm start width must be 8 deg');
        end

        function test_a_between_steps_ask_starts_from_the_step_below(tc)
            tc.verifyEqual(tc.O12.warm.step, 3, ...
                'a 12 deg ask must warm-start from the 11 deg step');
        end

        function test_the_committed_step_is_reproduced_from_below(tc)
            % Solving 11 deg from the 8 deg design must land in the
            % committed 11 deg row's neighbourhood.  Band, not equality:
            % the record ran 30 GN iterations on a 5x5 solve set and this
            % runs the DEMO knobs, so agreement is the claim, not identity.
            ref = tc.rec(3).map_max_nm;                 % 27.3 nm committed
            tc.verifyTrue(tc.O11.map.valid, ...
                'the 11 deg dense map must be valid (no lost fields)');
            tc.verifyLessThan(tc.O11.map.max_nm, 2.0*ref, ...
                sprintf(['the 11 deg re-solve (%.1f nm) must land in the ' ...
                         'committed row''s neighbourhood (%.1f nm)'], ...
                        tc.O11.map.max_nm, ref));
            tc.verifyGreaterThan(tc.O11.map.max_nm, 0.5*ref, ...
                'a result far BELOW the committed row would mean the map is not being scored over the asked box');
            tc.verifyTrue(tc.O11.gates.exit_pass, ...
                'the exit-direction gate must still pass at 11 deg');
        end

        % ---- C: the refusal path (reserve) ------------------------------
        function test_an_out_of_range_ask_refuses_before_solving(tc)
            t = tic;
            R = oi_demo_step(20, 'outdir', tc.outdir, 'quiet', true);
            dt = toc(t);
            tc.verifyTrue(R.refused, '20 deg must be refused');
            tc.verifyLessThan(dt, 60, ...
                'the refusal must come from the SCREEN -- before any solving');
            tc.verifyFalse(isfield(R,'map'), ...
                'a refused ask must not produce a map');
            % one accurate sentence: names the range, and says WHY
            tc.verifySubstring(R.why, '5-15 deg');
            tc.verifySubstring(R.why, 'no warm start');
            tc.verifyLessThan(numel(R.why), 500, ...
                'the refusal is one sentence, not a paragraph');
        end

        function test_an_untraceable_warm_start_refuses_via_the_F8_screen(tc)
            % The SECOND refusal path.  It CANNOT be reached by widening the
            % box -- measured 2026-08-28, the carried design keeps tracing
            % all the way out, degrading smoothly rather than losing rays:
            %   30 deg -> 1765 nm   60 -> 1.25e5   90 -> 2.59e5
            %   120 -> 4.55e6      150 -> 5.71e6   (never the 1e9 sentinel)
            % So a gate built from wide asks would exercise nothing and pass
            % vacuously (an earlier version of this test did exactly that).
            % Trigger the branch for real instead: hand the wrapper a warm
            % start that genuinely cannot trace, via the documented 'run_mat'
            % knob -- an M1 radius of 50 mm, where every ray misses.  Cheap
            % too: the screen answers before any solving.
            S = load(fullfile(tc.oidir,'t5_walk','t5_walk_run.mat'));
            rec = S.rec;
            rec(3).X.R(1) = 0.05;                 %#ok<STRNU> absurd M1 radius
            f = fullfile(tc.outdir, 'untraceable_walk.mat');
            OUT = S.OUT;  steps = S.steps;        %#ok<NASGU>
            save(f, 'OUT', 'rec', 'steps');

            t = tic;
            R = oi_demo_step(12, 'run_mat',f, 'outdir',tc.outdir, 'quiet',true);
            dt = toc(t);

            tc.verifyTrue(R.refused, ...
                'a warm start that cannot trace must be refused');
            tc.verifyGreaterThanOrEqual(R.screen_qmean_nm, 1e9, ...
                'the refusal must be the no-rays sentinel, not a range check');
            tc.verifySubstring(R.why, 'does not trace');
            tc.verifySubstring(R.why, 'F8');
            tc.verifyFalse(isfield(R,'map'), ...
                'a design scored untraceable must never produce a map');
            tc.verifyLessThan(dt, 300, ...
                'the F8 refusal must come from the SCREEN, before any solve');
        end

        function test_a_wide_ask_is_stopped_by_RANGE_not_traceability(tc)
            % The corollary of the measurement above, asserted so nobody
            % "fixes" the range screen thinking F8 would catch a wide ask.
            % It would not: at 60 deg the carried design still traces.
            R = oi_demo_step(60, 'outdir',tc.outdir, 'quiet',true);
            tc.verifyTrue(R.refused, '60 deg must be refused');
            tc.verifySubstring(R.why, 'validated continuation range');
            tc.verifyFalse(isfield(R,'screen_qmean_nm'), ...
                'the RANGE screen must answer before the traceability screen runs');
        end

        function test_the_range_is_the_walk_span(tc)
            % The refusal boundary must be the span the walk actually
            % traced -- not a number invented for the demo.
            W = [tc.rec.width];
            tc.verifyEqual([min(W) max(W)], [5 15], ...
                'the committed walk span moved; re-pin range_deg in oi_demo_step');
            lo = oi_demo_step(4.9,  'outdir',tc.outdir, 'quiet',true);
            hi = oi_demo_step(15.1, 'outdir',tc.outdir, 'quiet',true);
            tc.verifyTrue(lo.refused, 'just below the walk span must refuse');
            tc.verifyTrue(hi.refused, 'just above the walk span must refuse');
        end

        % ---- D: the frontier bracket ------------------------------------
        function test_the_verdict_quotes_the_correct_committed_rows(tc)
            p = tc.O12.predicted;
            tc.verifyEqual(p.lo.width, 11, 'lower bracket row for a 12 deg ask');
            tc.verifyEqual(p.hi.width, 13, 'upper bracket row for a 12 deg ask');
            tc.verifyEqual(p.lo.map_max_nm, tc.rec(3).map_max_nm, 'RelTol',1e-12);
            tc.verifyEqual(p.hi.map_max_nm, tc.rec(4).map_max_nm, 'RelTol',1e-12);
            % the interpolation is between them, and it is labelled as such
            tc.verifyGreaterThanOrEqual(p.wfe_nm, min(p.lo.map_max_nm, p.hi.map_max_nm));
            tc.verifyLessThanOrEqual(   p.wfe_nm, max(p.lo.map_max_nm, p.hi.map_max_nm));
            tc.verifySubstring(p.how, 'linear between');

            % and the written block carries them verbatim
            txt = fileread(tc.O12.files.verdict);
            tc.verifySubstring(txt, sprintf('%.1f nm', tc.rec(3).map_max_nm));
            tc.verifySubstring(txt, sprintf('%.1f nm', tc.rec(4).map_max_nm));
            tc.verifySubstring(txt, 'DENSE MAP MAX');
        end

        function test_an_ask_at_a_committed_step_brackets_onto_itself(tc)
            p = tc.O11.predicted;
            tc.verifyTrue(p.exact, ...
                'an ask AT a committed width brackets onto that row alone');
            tc.verifyEqual(p.wfe_nm, tc.rec(3).map_max_nm, 'RelTol',1e-12);
            tc.verifySubstring(p.how, 'committed row itself');
        end

        % ---- the metric contract is stated with every number ------------
        function test_the_verdict_block_states_its_metric(tc)
            txt = fileread(tc.O12.files.verdict);
            for s = {'strict RMS WFE', 'exit pupil', 'piston-only', ...
                     'map MAXIMUM', 'solve set'}
                tc.verifySubstring(txt, s{1});
            end
        end
    end
end
