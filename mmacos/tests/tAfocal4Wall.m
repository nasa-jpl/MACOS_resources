classdef tAfocal4Wall < matlab.unittest.TestCase
%TAFOCAL4WALL  Gates for the afocal4 union WALL (BRIEF_afocal4_wall).
%
%   The clearing stage promoted the union body-in-beam measurement to a
%   standing GATE.  This stage promotes it again, to a WALL inside
%   AFOCAL4_BUILD / CLEAR_BUILD -- because the clearing stage also measured
%   that a re-solve SPENDS whatever clearance a remedy wins (at -8 and -9 deg
%   it walked +23.3 and +42.3 mm of margin down to +2.3 and +0.7 mm) since
%   AFOCAL4_SCORE cannot see it.  A wall has three ways to be wrong and they
%   are what is gated here.
%
%     1  IT MUST BE ABLE TO REFUSE.  A wall that admits the committed 343 mm
%        deck -- the design that actually shipped, 79.9 mm inside its own
%        feed beam -- is decoration.  And it must refuse THROUGH THE
%        BUILDER, with the identifier the solver's catch clause turns into a
%        finite residual, not by returning a flag nobody reads.
%     2  IT MUST NOT BE A CAGE.  S4b's earned rule: a wall needs a compliant
%        seed.  Here that has a second, structural half -- CLEAR_BUILD tilts
%        the deck AFTER AFOCAL4_BUILD emits it, so a wall applied inside the
%        build would judge the UNTILTED train and reject every iterate
%        before the tilt could clear the beam.  The deferral is gated.
%     3  IT MUST BE ADDITIVE.  Default OFF, and with it off AFOCAL4_BUILD
%        must still re-emit the committed deck byte for byte -- otherwise
%        every S4 / S4b / S4c / clearing number in the record silently moved.
%
%   Plus the two properties the frontier rests on: the wall is a THRESHOLD
%   (raising union_min refuses a deck it used to admit), and the seeder
%   reports a failure to seed as a failure to SEED rather than as a design
%   verdict.
%
%   NOT asserted: which tilt to spend, what pupil price to accept, or where
%   the frontier's operating point lands.  Those are study results.
%
%   Size 256 (SUITE_FREEFORM), ~2 min.

    properties
        P
        here
        clr
        wall
        deck        % the committed 343 mm design
        cleared     % the delivered -10 deg cleared design
        D0
    end

    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(mfilename('fullpath'));
            run(fullfile(fileparts(h),'mmacos_setup.m'));
            tc.here = fullfile(fileparts(h),'challenges','afocal4');
            tc.clr  = fullfile(tc.here,'clearing');
            tc.wall = fullfile(tc.here,'wall');
            addpath(tc.here);   addpath(tc.clr);   addpath(tc.wall);
            addpath(fullfile(tc.here,'packaging'));
            macos.init(256);
            tc.P       = afocal4_params();
            tc.deck    = fullfile(tc.here,'afocal4_b2long_343mm.in');
            tc.cleared = fullfile(tc.clr,'afocal4_clear_343mm.in');
            tc.D0      = wall_recover(tc.P, tc.deck, 'verify',false);
        end
    end

    methods (Test)

        function test_the_wall_is_off_by_default_and_additive(tc)
        %  Additivity, and it is the load-bearing one: with P as the study
        %  ships it, AFOCAL4_BUILD traces no ray for the wall and re-emits
        %  the committed deck byte for byte.  If this fails, every committed
        %  artifact in the afocal4 record has stopped reproducing.
            tc.verifyFalse(logical(tc.P.pack.union_enforce), ...
                'the union wall must default OFF');
            tc.verifyEqual(tc.P.pack.union_min, 0, ...
                'the default floor is the gate''s own pass condition');
            f = [tempname '.in'];
            c = onCleanup(@() tAfocal4Wall.rm_({f})); %#ok<NASGU>
            o = afocal4_build(tc.P, tc.D0, f, 'verify',false);
            tc.verifyEqual(fileread(f), fileread(tc.deck), ...
                'the committed deck no longer rebuilds byte for byte');
            tc.verifyTrue(isempty(o.union) || ~o.union.enforced, ...
                'the wall reported a verdict while switched off');
        end

        function test_the_wall_refuses_the_deck_that_shipped(tc)
        %  Non-vacuity, half one, THROUGH THE BUILDER.  The identifier
        %  matters: CLEAR_SOLVE's catch clause turns a build error into the
        %  large finite residual that makes this a wall rather than a crash,
        %  and it catches by exception, so an error raised under some other
        %  identifier would still work -- but a wall that returned quietly
        %  would not.
            Q = tc.P;   Q.pack.union_enforce = true;   Q.pack.union_min = 0;
            f = [tempname '.in'];
            c = onCleanup(@() tAfocal4Wall.rm_({f})); %#ok<NASGU>
            threw = false;   id = '';
            try
                afocal4_build(Q, tc.D0, f, 'verify',false);
            catch ME
                threw = true;   id = ME.identifier;
            end
            tc.verifyTrue(threw, ...
                'the wall admits the design that is 79.9 mm inside its own beam');
            tc.verifyEqual(id, 'macos:design:afocal4_build:union');
        end

        function test_the_wall_admits_the_cleared_design(tc)
        %  Non-vacuity, half two.  A wall nothing can pass is a cage, and
        %  the design that PASSES has to be the one the clearing stage
        %  delivered -- measured on the deck itself, not on a rebuild.
            W = afocal4_union_wall(tc.P, tc.cleared, 'throw',false, ...
                                   'bare',true, 'quiet',true);
            tc.verifyGreaterThan(W.floor_m, 0.030, ...
                'the delivered cleared deck should hold ~+38 mm');
            tc.verifyGreaterThan(W.bare_m, W.floor_m, ...
                'bare lit glass must read BETTER than the declared body');
        end

        function test_the_wall_is_deferred_past_the_tilt(tc)
        %  The cage trap, structurally.  CLEAR_BUILD emits the UNTILTED
        %  train first and swings it afterwards, so a wall applied inside
        %  AFOCAL4_BUILD would judge the design the tilt exists to get away
        %  from and reject every iterate.  Both halves are asserted on the
        %  SAME P: the build alone refuses, the build-plus-tilt does not.
            Q = tc.P;   Q.pack.union_enforce = true;   Q.pack.union_min = 0;
            Dt = tc.D0;   Dt.tilt_deg = -10;
            f = [tempname '.in'];
            c = onCleanup(@() tAfocal4Wall.rm_({f})); %#ok<NASGU>
            refused = false;
            try
                afocal4_build(Q, Dt, f, 'verify',false);   % ignores tilt_deg
            catch
                refused = true;
            end
            tc.verifyTrue(refused, ...
                'AFOCAL4_BUILD is supposed to refuse the untilted train');
            o = clear_build(Q, Dt, f, 'verify',false);
            tc.verifyGreaterThan(o.union.floor_m, 0, ...
                'CLEAR_BUILD must apply the wall AFTER the swing, or it is a cage');
        end

        function test_the_wall_is_a_threshold_not_a_boolean(tc)
        %  union_min has to mean something: a floor above the cleared deck's
        %  own must refuse it.  Without this the frontier's two arms (0 and
        %  +15 mm) would be the same run twice.
            W = afocal4_union_wall(tc.P, tc.cleared, 'throw',false, 'quiet',true);
            Q = tc.P;   Q.pack.union_enforce = true;
            Q.pack.union_min = W.floor_m + 0.010;
            threw = false;
            try
                afocal4_union_wall(Q, tc.cleared, 'quiet',true);
            catch ME
                threw = strcmp(ME.identifier,'macos:design:afocal4_build:union');
            end
            tc.verifyTrue(threw, 'raising union_min did not refuse anything');
            Q.pack.union_min = W.floor_m - 0.010;
            V = afocal4_union_wall(Q, tc.cleared, 'quiet',true);
            tc.verifyTrue(V.ok, 'lowering union_min did not admit it again');
        end

        function test_the_solve_sampling_bias_is_small_and_optimistic(tc)
        %  The wall is evaluated INSIDE the solver at solve sampling (ngrid
        %  21) and the frontier quotes the gate at reporting sampling (41).
        %  More rays make a bigger union hull, so the wall's number is the
        %  OPTIMISTIC one.  That is fine only while the difference is small
        %  next to the seeder's margin -- which is the property gated, not
        %  the particular millimetre.
            Dt = tc.D0;   Dt.tilt_deg = -10;
            a = [tempname '.in'];   b = [tempname '.in'];
            c = onCleanup(@() tAfocal4Wall.rm_({a,b})); %#ok<NASGU>
            Ds = Dt;   Ds.ngrid = tc.P.solve.ngrid;
            clear_build(tc.P, Ds, a, 'verify',false);
            clear_build(tc.P, Dt, b, 'verify',false);
            Ks = afocal4_union(a, 'fields',tc.P.Fsolve, 'quiet',true);
            Kr = afocal4_union(b, 'fields',tc.P.Fsolve, 'quiet',true);
            bias = Ks.floor_m - Kr.floor_m;
            tc.verifyGreaterThan(bias, 0, ...
                'coarser sampling is supposed to read a SMALLER body');
            tc.verifyLessThan(bias, 0.010, ...
                ['the sampling bias has outgrown the seeder''s 10 mm ' ...
                 'margin -- the wall no longer holds what the gate reports']);
        end

        function test_the_seeder_finds_a_seed_where_one_exists(tc)
        %  A wall needs a compliant seed or it is a cage (S4b).  At -9 deg
        %  the swung parent already clears a +15 mm floor, so the seeder must
        %  return it having moved NOTHING -- which is also what keeps the
        %  frontier comparable with the delivered row.
            Q = tc.P;   Q.pack.union_enforce = true;   Q.pack.union_min = 0.015;
            [D, info] = wall_seed(Q, tc.D0, -9, 'quiet',true);
            tc.verifyTrue(info.ok, 'no seed at a tilt whose parent already clears');
            tc.verifyEqual(info.source, 'tilt alone');
            tc.verifyEqual(D.fm_standoff, tc.D0.fm_standoff, 'AbsTol',0);
            tc.verifyEqual(D.R2, tc.D0.R2, 'AbsTol',0);
            tc.verifyEqual(D.tilt_deg, -9);
            tc.verifyGreaterThanOrEqual(info.floor_m, info.need_m);
        end

        function test_a_failure_to_seed_is_reported_as_one(tc)
        %  The S4b lesson in its reporting half: dropped at a non-compliant
        %  point the solver hands back its seed, and that reads as "this
        %  operating point has no design" when it is a seeding failure.  So
        %  an unreachable floor must come back as INFO.ok false with the
        %  best floor it saw -- never as an exception, and never as a design
        %  verdict.  Half a metre of clearance is unreachable by
        %  construction.
            Q = tc.P;   Q.pack.union_enforce = true;   Q.pack.union_min = 0.500;
            [~, info] = wall_seed(Q, tc.D0, -9, 'quiet',true, 'max_gate',3, ...
                                  'fallback','none');
            tc.verifyFalse(info.ok, 'a 500 mm floor is not supposed to be reachable');
            tc.verifyTrue(isfinite(info.best_floor_m), ...
                'a failed seed must still report how close it got');
            tc.verifyGreaterThan(info.n_gated, 0);
        end
    end

    methods (Static, Access = private)
        function rm_(fs)
            for i = 1:numel(fs)
                if exist(fs{i},'file'), delete(fs{i}); end
            end
        end
    end
end
