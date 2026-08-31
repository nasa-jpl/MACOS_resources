classdef tAfocal4Clear < matlab.unittest.TestCase
%TAFOCAL4CLEAR  Gates for the afocal4 CLEARING stage (BRIEF_afocal4_clear).
%
%   The clearing stage answers one question -- does a BODY stand in a BEAM --
%   and then spends an extraction tilt to make the answer no.  Three things
%   would silently corrupt every number it produced, and they are what is
%   gated here.  Nothing about the delivered DESIGN is asserted: which tilt
%   to spend and what pupil quality to accept are study results, not tests.
%
%     1  THE GATE MUST BE ABLE TO FAIL.  AFOCAL4_UNION exists because the
%        packaging gate could not see a part standing in a beam, and the
%        committed 343 mm family-2 deck -- the design that actually shipped
%        -- is 79.9 mm inside its own feed beam.  A gate that passes that
%        deck is measuring something else.  Both halves are asserted: it
%        fails there, and it passes on a deck built to clear it.
%     2  THE TILT MUST BE EXACT FOR THE CHIEF.  CLEAR_TILT swings a mirror
%        about the point the chief strikes it and re-poses the train by the
%        rotation that carries the old outgoing chief onto the new one.  If
%        the pivot drifts off the surface, or the downstream map is not the
%        one the new chief needs, the study is measuring a mis-aligned
%        telescope and calling the result an aberration price.
%     3  ZERO TILT MUST BE A NO-OP, BYTE-FOR-BYTE.  CLEAR_BUILD wraps
%        AFOCAL4_BUILD, which emits the committed evidence for the whole S4
%        trade.  At tilt 0 the two must produce identical files, or the
%        clearing stage is quietly comparing against a design nobody
%        published.
%
%   Plus the law itself: on a coaxial deck the field-INDEPENDENT offset
%   between the two bundles must be zero, because that is the statement the
%   ratio law rests on -- and it is the quantity a tilt then supplies.
%
%   Size 256 (SUITE_FREEFORM), ~90 s.

    properties
        P
        here
        clr
        deck
    end

    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(mfilename('fullpath'));
            run(fullfile(fileparts(h),'mmacos_setup.m'));
            tc.here = fullfile(fileparts(h),'challenges','afocal4');
            tc.clr  = fullfile(tc.here,'clearing');
            % PACKAGING too: CLEAR_FOLD builds its decks with PACK_FOLD, and
            % without it every station raises an undefined-function error
            % that reads as "a flat cannot clear it" -- the answer the test
            % is looking for.  CLEAR_FOLD now refuses an all-failed sweep
            % for the same reason; this is the other half of that fix.
            addpath(tc.here);   addpath(tc.clr);
            addpath(fullfile(tc.here,'packaging'));
            macos.init(256);
            tc.P = afocal4_params();
            tc.deck = fullfile(tc.here,'afocal4_b2long_343mm.in');
        end
    end

    methods (Test)

        function test_the_gate_fails_on_the_deck_that_shipped(tc)
        %  Non-vacuity, half one.  The committed 343 mm family-2 deck is the
        %  design the S4b/S4c trade delivered, and its feed beam runs through
        %  the collimator's glass.  Measured over HIS 3x3 field box: -79.9 mm
        %  with the declared body model, -55.4 mm against bare lit glass.  A
        %  gate that cannot report that is not a gate.
            K = afocal4_union(tc.deck, 'fields',tc.P.Fsolve, 'quiet',true);
            tc.verifyFalse(K.ok, 'the union gate passes the deck it must fail');
            tc.verifyLessThan(K.floor_m, -0.05, ...
                'the pierce is supposed to be tens of millimetres deep');
            tc.verifyEqual(K.pair(K.worst).obst, K.nElt-1, ...
                'the collimator is supposed to be the part in the beam');
            % ... and it is the DESIGN's, not the body model's: bare lit
            % glass, no 1.15 and no edge allowance, is still pierced.
            B = afocal4_union(tc.deck, 'fields',tc.P.Fsolve, 'body_k',1.0, ...
                              'body_pad',0.0, 'init',false, 'quiet',true);
            tc.verifyFalse(B.ok, ...
                'bare lit glass clears -- the interference would be the model''s');
            tc.verifyGreaterThan(B.floor_m, K.floor_m, ...
                'the allowance is supposed to make the floor WORSE, not better');
        end

        function test_one_field_would_have_passed_it(tc)
        %  Why it shipped, as a test rather than a story.  The gate must be
        %  run over the field BOX: on the deck's own single field the same
        %  measurement is far more forgiving, and that is exactly the check
        %  AFOCAL4_PACK was making.
            box = afocal4_union(tc.deck, 'fields',tc.P.Fsolve, 'quiet',true);
            one = afocal4_union(tc.deck, 'quiet',true);
            tc.verifyGreaterThan(one.floor_m, box.floor_m + 0.02, ...
                ['the single-field answer is supposed to be much more ' ...
                 'optimistic than the union one']);
            tc.verifyLessThan(box.foot_r(box.nElt-1)/one.foot_r(one.nElt-1), 10, ...
                'sanity: the union footprint should be a few x the per-field one');
            tc.verifyGreaterThan(box.foot_r(box.nElt-1)/one.foot_r(one.nElt-1), 2, ...
                'the union footprint is supposed to DOMINATE the per-field one');
        end

        function test_the_field_walk_law_on_a_coaxial_deck(tc)
        %  The law rests on one fact: on a coaxial train the separation
        %  between two bundles is proportional to field angle and has NO
        %  field-independent part.  Measured as the intercept of the
        %  walk+offset fit, that part must be small next to the walk itself
        %  -- and the ratio must be the one that fails.
            nE = macos.num_elt();
            L = clear_law(tc.deck, 'fields',tc.P.Fsolve, 'leg',2, ...
                          'elt',nE-1, 'M',tc.P.M, 'quiet',true);
            tc.verifyLessThan(abs(L.offset_m), 0.010, ...
                'a coaxial deck is not supposed to carry a field-independent offset');
            tc.verifyLessThan(L.ratio, L.need, ...
                'the walk ratio is supposed to be short of what the box demands');
            tc.verifyLessThan(L.fit_resid_m, 5e-3, ...
                'the walk is supposed to be linear in the field angle');
            % The collimator's own walk is PINNED by the interface spec at
            % M * iface.  The pin is PARAXIAL and the measurement is a real
            % traced footprint centroid, so the two agree to ~8 % on this
            % deck -- that gap is the design's own pupil aberration, which
            % is the quantity the fourth mirror exists to control.  The
            % gate is that the pin is the right ORDER, not that the design
            % is aberration-free.
            tc.verifyLessThan(abs(L.M_iface_err), 0.15, ...
                'c_body should be M * iface to better than 15 %');
        end

        function test_zero_tilt_is_a_byte_for_byte_no_op(tc)
        %  CLEAR_BUILD must not perturb the committed evidence.  Two claims:
        %  the design struct recovered from the committed deck rebuilds it
        %  exactly, and CLEAR_BUILD at zero tilt equals AFOCAL4_BUILD.
            D = tAfocal4Clear.recover_(tc.P, tc.deck);
            a = [tempname '.in'];   b = [tempname '.in'];
            c = onCleanup(@() tAfocal4Clear.rm_({a,b})); %#ok<NASGU>
            afocal4_build(tc.P, D, a, 'verify',false);
            clear_build(tc.P, D, b, 'verify',false);
            tc.verifyEqual(fileread(a), fileread(tc.deck), ...
                'the recovered design does not rebuild the committed deck');
            tc.verifyEqual(fileread(a), fileread(b), ...
                'clear_build at zero tilt differs from afocal4_build');
        end

        function test_the_tilt_keeps_the_chief_ray_exactly(tc)
        %  The pivot is the point the chief actually strikes, so two things
        %  must hold to machine precision, both read from a RE-TRACE of the
        %  written deck rather than from the transform that made it:
        %    * nothing upstream of the swung mirror moves, INCLUDING the hit
        %      point on the mirror itself -- which is the statement that the
        %      pivot is still on the surface;
        %    * the beam turns by exactly 2*alpha, which is only true when the
        %      tilt axis is perpendicular to the plane of incidence, a
        %      property of this coaxial biased design and not of the code.
        %  NOT asserted: that the unsigned incidence angle moves by alpha.
        %  It does not in general -- the field mirror is worked at a few
        %  degrees, so a tilt bigger than that carries the SIGNED angle
        %  through zero and |AOI| folds back.  Asserting it once cost a
        %  perfectly correct tilt a failing test.
            a = 6;                                  % deg
            f = [tempname '.in'];
            c = onCleanup(@() tAfocal4Clear.rm_({f})); %#ok<NASGU>
            o = clear_tilt(tc.deck, struct('elt','FM','alpha',deg2rad(a), ...
                                           'axis',[1 0 0]), f);
            P0 = tAfocal4Clear.chief_(tc.deck);
            P1 = tAfocal4Clear.chief_(f);
            k  = o.elt;                             % polyline index k+1 = elt k
            tc.verifyLessThan(max(vecnorm(P0(:,1:k+1) - P1(:,1:k+1))), 1e-12, ...
                'the tilt moved the chief ray upstream of the mirror it swung');
            tc.verifyLessThan(norm(P1(:,k+1) - o.Q(:)), 1e-12, ...
                'the pivot is not on the surface: the chief no longer lands there');
            tc.verifyLessThan(abs(o.turn_deg - 2*a), 1e-9, ...
                'a mirror tilted by alpha is supposed to turn the beam by 2 alpha');
        end

        function test_the_tilt_supplies_a_field_independent_offset(tc)
        %  The mechanism, isolated.  A tilt must NOT improve the walk ratio
        %  (that is the part the law forbids moving) and must supply a large
        %  field-independent offset instead -- which is what takes the gate
        %  from failing to passing.
            f = [tempname '.in'];
            c = onCleanup(@() tAfocal4Clear.rm_({f})); %#ok<NASGU>
            clear_tilt(tc.deck, struct('elt','FM','alpha',deg2rad(-10), ...
                                       'axis',[1 0 0]), f);
            nE = macos.num_elt();
            L0 = clear_law(tc.deck, 'fields',tc.P.Fsolve, 'leg',2, 'elt',nE-1, ...
                           'quiet',true);
            L1 = clear_law(f, 'fields',tc.P.Fsolve, 'leg',2, 'elt',nE-1, ...
                           'quiet',true);
            tc.verifyLessThan(L1.ratio, L0.need, ...
                'the tilt is not supposed to fix the walk ratio');
            tc.verifyGreaterThan(L1.offset_m, 0.10, ...
                'the tilt is supposed to buy a field-independent offset');
            % non-vacuity, half two: the gate now passes
            K = afocal4_union(f, 'fields',tc.P.Fsolve, 'quiet',true);
            tc.verifyTrue(K.ok, 'the tilted deck is still a body in a beam');
            tc.verifyEqual(K.nLost, 0, 'the tilt lost rays');
        end

        function test_a_flat_fold_cannot_clear_it(tc)
        %  Leverage 1, as a gate rather than a claim: over stations either
        %  side of the critical one, in both turn directions, the best floor
        %  a single flat reaches is the parent's own.  If someone later makes
        %  a fold appear to fix this, that is a bug in the fold, not a design.
            R = clear_fold(tc.P, tc.deck, 'fields',tc.P.Fsolve, ...
                           'dist',[0.40 0.75 0.90], 'to',{[1 0 0]}, 'quiet',true);
            K = afocal4_union(tc.deck, 'fields',tc.P.Fsolve, 'quiet',true);
            tc.verifyLessThan(max([R.pt.floor_body]), 0, ...
                'a single flat fold appears to clear the feed beam');
            tc.verifyLessThan(max([R.pt.floor_body]) - K.floor_m, 1e-6, ...
                ['a fold is an isometry: before the critical station the ' ...
                 'floor must equal the parent''s']);
            tc.verifyGreaterThan(R.crit_frac, 0);
            tc.verifyLessThan(R.crit_frac, 1);
        end

        function test_pack_gate_carries_the_union_clause_and_stays_additive(tc)
        %  The promotion.  AFOCAL4_PACK must now refuse the committed deck on
        %  the union clause, and 'union',false must reproduce the old verdict
        %  -- otherwise the change is not additive and every S4b/S4c gate
        %  result silently moved.
            f = [tempname '.in'];
            c = onCleanup(@() tAfocal4Clear.rm_({f})); %#ok<NASGU>
            copyfile(tc.deck, f);
            Kon  = afocal4_pack(tc.P, f, 'quiet',true);
            Koff = afocal4_pack(tc.P, f, 'union',false, 'quiet',true);
            tc.verifyFalse(Kon.ok_union, 'the union clause passes the pierced deck');
            tc.verifyFalse(Kon.ok, 'the gate as a whole passes the pierced deck');
            tc.verifyTrue(Koff.ok_union, ...
                'a clause that was not run must not report a verdict');
            % the pre-existing sub-flags are untouched by the addition
            tc.verifyEqual(Kon.ok_station, Koff.ok_station);
            tc.verifyEqual(Kon.ok_fold,    Koff.ok_fold);
            tc.verifyEqual(Kon.fold_pick.gap, Koff.fold_pick.gap, 'AbsTol',1e-12);
        end
    end

    methods (Static, Access = private)
        function D = recover_(P, deck)
        %  The design struct behind a committed afocal4 deck.  Spacings come
        %  from zElt and NOT from the vertices: the builder poses the
        %  interface plane on the traced chief, so the last mirror's vertex
        %  is 16 mm further from the interface vertex than the standoff is.
            txt = fileread(deck);
            Kc = tAfocal4Clear.g1_(txt,'KcElt');
            Kr = tAfocal4Clear.g1_(txt,'KrElt');
            zE = tAfocal4Clear.g1_(txt,'zElt');
            nM = numel(Kc) - 1;
            D = struct('form','field', 'K',Kc(1:nM).', 'bias_deg',P.bias_deg, ...
                       'ngrid',P.ngrid, 'rb',zeros(numel(P.rb_elts),2), ...
                       'tilt_deg',0, 'R2',abs(Kr(2)), 't1',zE(1), 'iface',zE(nM));
            fo = afocal_first_order([abs(Kr(1)) abs(Kr(2))], D.t1, [false true], ...
                                    'D',P.D, 'stop_ahead',P.stop_ahead);
            D.fm_standoff = -fo.y_marginal(2)/fo.u_marginal(2) - zE(2);
        end
        function P = chief_(deck)
            macos.load_rx(deck);
            macos.ray_hist('on');   t = macos.trace();   h = macos.ray_hist(t.nRays);
            macos.ray_hist('off');
            P = squeeze(h.P(:,1,:));
        end
        function v = g1_(txt, key)
            t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens');
            v = zeros(1, numel(t));
            for i = 1:numel(t), v(i) = sscanf(strrep(t{i}{1},'D','E'), '%f', 1); end
        end
        function rm_(fs)
            for i = 1:numel(fs)
                if exist(fs{i},'file'), delete(fs{i}); end
            end
        end
    end
end
