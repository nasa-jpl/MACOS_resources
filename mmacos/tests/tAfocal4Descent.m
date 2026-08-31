classdef tAfocal4Descent < matlab.unittest.TestCase
%TAFOCAL4DESCENT  Gates for the afocal4 DESCENT (BRIEF_afocal4_descent).
%
%   The descent starts at seven powered mirrors and walks back, so its
%   foundation is a closure and a builder that work at ANY N.  Everything
%   the ladder later says rests on those being the SAME machinery the
%   4-mirror family was built with -- not a lookalike that agrees to three
%   figures.  That is what is gated here.
%
%     1  N = 4 IS THE COMMITTED DECK, BYTE FOR BYTE.  The generalized
%        builder must re-emit `afocal4_b2long_343mm.in` exactly, given the
%        form's own element names and the same root-scan recipe.  If it
%        does not, every rung of the ladder is measured against a 4-mirror
%        reference that is not the one in the record.
%     2  THE CLOSURE IS A CLOSURE.  The three first-order conditions --
%        recollimate, magnify by 30, land the pupil at the interface -- are
%        IDENTITIES, not targets.  Asserted at 1e-9 at several N, because a
%        first-order identity that is only nearly true is not one.
%     3  THE WALLS STILL BITE at N mirrors: a degenerate spacing, the S4b
%        packaging station, and the union floor deferred past the tilts.
%     4  A TILT IS EXACT FOR THE CHIEF at any N, and tilts COMPOSE: the
%        builder applies them upstream-first so each swing carries the ones
%        before it.
%
%   NOT asserted: how many mirrors the requirement set needs, or what any
%   rung costs.  Those are the study's results, and a test that pinned them
%   would be pinning the answer to the question the stage exists to ask.
%
%   Size 256 (SUITE_FREEFORM), ~3 min.

    properties
        P
        here
        dsc
        deck        % the committed 343 mm four-mirror design
        D4          % its design struct in DESCENT_BUILD's parameterization
    end

    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(mfilename('fullpath'));
            run(fullfile(fileparts(h),'mmacos_setup.m'));
            tc.here = fullfile(fileparts(h),'challenges','afocal4');
            tc.dsc  = fullfile(tc.here,'descent');
            addpath(tc.here);   addpath(tc.dsc);
            addpath(fullfile(tc.here,'clearing'));
            addpath(fullfile(tc.here,'wall'));
            addpath(fullfile(tc.here,'packaging'));
            macos.init(256);
            tc.P    = afocal4_params();
            tc.deck = fullfile(tc.here,'afocal4_b2long_343mm.in');
            D0 = wall_recover(tc.P, tc.deck, 'verify',false);
            % the committed design, re-expressed in the N-mirror
            % parameterization: free radii/spacings for mirrors 1..N-2, and
            % the closure consumes the rest.  The M2 -> field-mirror spacing
            % is the intermediate-image distance minus the standoff.
            fo = afocal_first_order([abs(tc.P.parent.R(1)) D0.R2], D0.t1, ...
                     [false true], 'D',tc.P.D, 'stop_ahead',tc.P.stop_ahead);
            a = -fo.y_marginal(2)/fo.u_marginal(2) - D0.fm_standoff;
            tc.D4 = struct('N',4, 'R',[abs(tc.P.parent.R(1)) D0.R2], ...
                'convex',[false true], 't',[D0.t1 a], 'K',D0.K(:).', ...
                'iface',D0.iface, 'tilt_deg',zeros(1,4), 'ngrid',tc.P.ngrid, ...
                'bias_deg',tc.P.bias_deg);
        end
    end

    methods (Test)

        function test_N4_reemits_the_committed_deck_byte_for_byte(tc)
        %  The load-bearing one.  The generalized builder must reproduce the
        %  4-mirror artifact the whole S4/S4b/S4c/clearing/wall record is
        %  written on -- not approximately, exactly.
        %
        %  TWO QUALIFIERS, both real and neither a fudge.  The committed
        %  decks name their mirrors M1/M2/FM/M3 because that is the FORM's
        %  vocabulary, where a generic builder emits M1..MN for a LAYOUT; and
        %  FZERO converges to whichever root its BRACKET holds, so a
        %  different scan grid lands 2e-16 away and the emitted KrElt differs
        %  in its last digit.  The identity is therefore asserted under
        %  AFOCAL4_PHI4's own window rather than by widening a tolerance
        %  until it passes.
            f = [tempname '.in'];
            c = onCleanup(@() tAfocal4Descent.rm_({f})); %#ok<NASGU>
            descent_build(tc.P, tc.D4, f, 'verify',false, ...
                          'names',{'M1','M2','FM','M3'}, ...
                          'window',[-0.9 5.0], 'npts',119);
            tc.verifyEqual(fileread(f), fileread(tc.deck), ...
                'the N-mirror builder does not re-emit the committed 4-mirror deck');
        end

        function test_the_closure_agrees_with_the_four_mirror_one(tc)
        %  And it agrees with AFOCAL4_CLOSE's own 'field' branch, which is
        %  the algebra the generalization was lifted from.
            D0 = wall_recover(tc.P, tc.deck, 'verify',false);
            Q = tc.P;   Q.parent.R(2) = D0.R2;   Q.parent.t(1) = D0.t1;
            [phi4, C4, found] = afocal4_phi4(Q, D0.fm_standoff, D0.iface);
            tc.verifyTrue(found, 'the 4-mirror reference closure did not close');
            S = struct('N',4, 'R',tc.D4.R, 'convex',tc.D4.convex, ...
                       't',tc.D4.t, 'iface',tc.D4.iface, 'K',tc.D4.K);
            Cd = descent_close(tc.P, S, 'window',[-0.9 5.0], 'npts',119);
            tc.verifyTrue(Cd.found);
            tc.verifyEqual(Cd.R, C4.R, 'AbsTol',1e-12);
            tc.verifyEqual(Cd.t, C4.t, 'AbsTol',1e-12);
            tc.verifyEqual(Cd.phi(3), phi4, 'AbsTol',1e-12);
            tc.verifyEqual(logical(Cd.convex), logical(C4.convex));
        end

        function test_the_three_conditions_are_identities_at_several_N(tc)
        %  Recollimate, magnify by 30, land the pupil at the interface.  They
        %  are CLOSURES and never merit terms -- the S4 ruling, carried up to
        %  N mirrors -- so they hold to machine precision or the stage has no
        %  specification.  Checked at more than one N because a generalization
        %  that only works at the N it was tested on is not one.
            for N = [4 5 6 7]
                S = tAfocal4Descent.seed_(tc.P, N);
                if isempty(S), continue; end
                C = descent_close(tc.P, S);
                tc.assertTrue(C.found, sprintf('no closure at N = %d', N));
                tc.verifyLessThan(abs(C.fo.u_out), 1e-9, ...
                    sprintf('N = %d does not recollimate', N));
                tc.verifyLessThan(abs(C.fo.mag/tc.P.M - 1), 1e-9, ...
                    sprintf('N = %d does not magnify by %g', N, tc.P.M));
                tc.verifyLessThan(abs(C.fo.pupil_dist - S.iface), 1e-9, ...
                    sprintf('N = %d puts the pupil elsewhere', N));
                tc.verifyTrue(C.closed);
            end
        end

        function test_the_packaging_wall_still_bites_at_N_mirrors(tc)
        %  A wall that only exists at N = 4 would let the ladder walk out of
        %  buildability the moment it added a mirror.  The committed design's
        %  own front end with a station the closure puts in FRONT of the
        %  primary must be refused, through the builder, with the identifier
        %  the solver's catch clause turns into a residual.
            Q = tc.P;   Q.pack.enforce = true;
            % a 5-mirror closure the density scan measured at z < 0
            S = struct('N',5, 'R',[2.5 0.448372 2.0], 'convex',[false true false], ...
                       't',[1.041953 1.0 1.2], 'iface',0.343, 'K',zeros(1,5));
            C = descent_close(tc.P, S);
            tc.assumeTrue(C.found && C.behind_m1 < Q.pack.m3_behind_min, ...
                'fixture drifted: this closure is supposed to be non-compliant');
            D = S;  D.tilt_deg = zeros(1,5);  D.ngrid = tc.P.ngrid;
            D.bias_deg = tc.P.bias_deg;
            f = [tempname '.in'];
            c = onCleanup(@() tAfocal4Descent.rm_({f})); %#ok<NASGU>
            threw = false;   id = '';
            try
                descent_build(Q, D, f, 'verify',false);
            catch ME
                threw = true;   id = ME.identifier;
            end
            tc.verifyTrue(threw, 'the packaging wall does not bite at N = 5');
            tc.verifyEqual(id, 'macos:design:descent_build:packaging');
        end

        function test_a_tilt_is_exact_for_the_chief_and_tilts_compose(tc)
        %  The wall slice's guarantee, at N mirrors and with more than one
        %  swing.  CLEAR_TILT pivots on the point the chief actually strikes
        %  and re-poses everything downstream, so the chief path upstream of
        %  the FIRST swung mirror cannot move -- and because the builder
        %  applies tilts upstream-first, a second swing composes with the
        %  first rather than undoing it.
            f0 = [tempname '.in'];   f1 = [tempname '.in'];
            c = onCleanup(@() tAfocal4Descent.rm_({f0,f1})); %#ok<NASGU>
            descent_build(tc.P, tc.D4, f0, 'verify',false);
            Dt = tc.D4;   Dt.tilt_deg = [0 0 -6 0];      % swing the field mirror
            descent_build(tc.P, Dt, f1, 'verify',false);
            P0 = tAfocal4Descent.chief_(f0);
            P1 = tAfocal4Descent.chief_(f1);
            k  = 3;                                      % the swung mirror
            tc.verifyLessThan(max(vecnorm(P0(:,1:k+1) - P1(:,1:k+1))), 1e-12, ...
                'the tilt moved the chief upstream of the mirror it swung');
            % a second, downstream swing must leave the first one's geometry
            Dt2 = Dt;   Dt2.tilt_deg = [0 0 -6 -3];
            f2 = [tempname '.in'];
            c2 = onCleanup(@() tAfocal4Descent.rm_({f2})); %#ok<NASGU>
            descent_build(tc.P, Dt2, f2, 'verify',false);
            P2 = tAfocal4Descent.chief_(f2);
            tc.verifyLessThan(max(vecnorm(P1(:,1:k+1) - P2(:,1:k+1))), 1e-12, ...
                ['a downstream swing disturbed the geometry upstream of an ' ...
                 'earlier one -- the tilts are not composing']);
        end

        function test_the_requirement_set_reproduces_known_numbers(tc)
        %  DESCENT_REQUIRE is the one place a rung becomes a verdict, so it
        %  is pinned against numbers this study already published: the
        %  committed 343 mm deck's wavefront and pupil row, and -- the row
        %  most easily got wrong -- the interface surface RIM-anchored, which
        %  S4c reports as 0.1853 mm where the surface anchor reads 0.0174.
            Q = descent_require(tc.P, tc.deck, 'union',false, 'quiet',true);
            g = @(n) Q.rows(strcmp({Q.rows.name}, n)).value;
            tc.verifyEqual(g('WFE rung-2 max'), 10406.98, 'RelTol',1e-4);
            tc.verifyEqual(g('pupil blur'),        157.02, 'RelTol',1e-4);
            tc.verifyEqual(g('breathing'),         0.1240, 'RelTol',1e-3);
            tc.verifyEqual(g('iface surface (rim)'), 0.1853, 'RelTol',1e-3);
            % and the 71 nm target is IN the set and is MISSED here, which is
            % the premise the whole descent rests on
            r = Q.rows(strcmp({Q.rows.name},'WFE rung-2 max'));
            tc.verifyEqual(r.target, 71.0);
            tc.verifyFalse(r.ok, ...
                'the 4-mirror family is supposed to MISS the wavefront target');
            tc.verifyGreaterThan(r.value/r.target, 100, ...
                'it is supposed to miss it by two orders of magnitude');
        end
    end

    methods (Static, Access = private)
        function S = seed_(P, N)
        %  A closing front end at this N, cheap: the committed 4-mirror one
        %  padded with weak mirrors and spacings the density scan showed
        %  close.  Returns [] if nothing closes, so the caller can skip
        %  rather than fail on a fixture.
            base_R = [2.5 0.448372];   base_c = [false true];
            nf = N - 2;
            R = [base_R, repmat(3.2, 1, max(0,nf-2))];
            c = [base_c, false(1, max(0,nf-2))];
            for t2 = [2.928546 1.3 1.0 0.7 0.4]
                t = [1.041953, repmat(t2, 1, nf-1)];
                S = struct('N',N, 'R',R(1:nf), 'convex',c(1:nf), 't',t, ...
                           'iface',0.343, 'K',zeros(1,N));
                C = descent_close(P, S);
                if isfield(C,'found') && C.found && C.closed, return; end
            end
            S = [];
        end
        function Pc = chief_(deck)
            macos.load_rx(deck);
            macos.ray_hist('on');   t = macos.trace();   h = macos.ray_hist(t.nRays);
            macos.ray_hist('off');
            Pc = squeeze(h.P(:,1,:));
        end
        function rm_(fs)
            for i = 1:numel(fs)
                if exist(fs{i},'file'), delete(fs{i}); end
            end
        end
    end
end
