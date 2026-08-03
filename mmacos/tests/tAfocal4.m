classdef tAfocal4 < matlab.unittest.TestCase
%TAFOCAL4  Gates for the afocal4 S4 joint-solve machinery.
%
%   The S4 solve nests an OUTER optimiser (conics, the field-mirror standoff,
%   the front end, rigid bodies) inside an EXACT first-order closure.  What
%   has to be gated is therefore not "did the solve converge" -- that is a
%   study result, not a test -- but the three things a silent break would
%   corrupt every result through:
%
%     1  THE IDENTITIES.  M = 30.000, the afocal condition and the pupil
%        station at P.iface are properties of every design the solver sees,
%        re-derived at each iterate rather than penalised in the merit.  If
%        the closure stops closing, the whole ladder is measuring a family of
%        telescopes that are not the telescope.
%     2  ONE POSING PATH.  The interface plane is put on the traced exit
%        chief by AFOCAL4_BUILD for perturbed and unperturbed decks alike; a
%        rigid-bodied deck is no longer a Telescope, so it cannot use
%        Telescope/align_exit_reference.  The two must agree EXACTLY where
%        both apply, or the rung-4 pose is a different convention from rungs
%        1-3 and the ladder is not a ladder.
%     3  THE MERIT'S SHAPE.  Log-domain residuals with a floor: a term at its
%        target contributes log(1/floor), a term twice inside contributes
%        zero, and nothing is negative.  A sign or floor slip would silently
%        re-weight the study.
%
%   Plus a FAST solve smoke -- two iterations on a coarse grid -- which
%   checks the outer loop is wired to the inner one at all.  THE FULL SOLVE
%   IS NOT IN ANY SUITE: a ladder rung is ~10 minutes of machine time and a
%   test suite is not where that belongs (afocal4_ladder is).
%
%   Model size 256 group: ./run_mmacos_tests.sh freeform
%
%   See also AFOCAL4_BUILD, AFOCAL4_SCORE, AFOCAL4_SOLVE, TDESIGNAFOCAL.

    properties
        P
        here
    end

    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(mfilename('fullpath'));
            run(fullfile(fileparts(h),'mmacos_setup.m'));
            tc.here = fullfile(fileparts(h),'design','examples','afocal4');
            addpath(tc.here);
            macos.init(256);
            tc.P = afocal4_params();
        end
    end

    methods (Test)

        function test_closure_holds_the_first_order_identities(tc)
        %  The three conditions, at the flagged operating point and at two
        %  others, on axis and at the bias.  They are IDENTITIES: the
        %  tolerances are numerical, not engineering.
            for iface = [0.09 0.14 0.30]
                for bias = [0 tc.P.bias_deg]
                    D = afocal4_seed(tc.P, 'bias_deg',bias, 'iface',iface);
                    f = [tempname '.in'];
                    c = onCleanup(@() tc.rm_(f)); %#ok<NASGU>
                    b = afocal4_build(tc.P, D, f);
                    tc.verifyLessThan(abs(b.C.closure.u_out), 1e-9, ...
                        sprintf('afocal condition at iface %.2f', iface));
                    tc.verifyLessThan(abs(b.C.closure.mag_err), 1e-9, ...
                        sprintf('magnification at iface %.2f', iface));
                    tc.verifyLessThan(abs(b.iface_pred - iface), 1e-9, ...
                        sprintf('pupil station at iface %.2f', iface));
                    % ... and the TRACE agrees with the paraxial model it was
                    % closed from.  A closure that only closes on paper is
                    % the S3 builder bug all over again -- which read 19.4x
                    % against a paraxial 30.0, i.e. 35% out.  The tolerance
                    % here is 6% and NOT tighter on purpose: these are UNSOLVED
                    % seeds (K_FM = 0) carrying microns of spherical
                    % aberration, and a full-aperture marginal ray in a design
                    % with microns of spherical does not land where the
                    % paraxial one does -- 3.6% at the tightest interface
                    % standoff tried here.  That is the design, not the
                    % builder.  The SHARP gate on the closure is the 1e-9
                    % paraxial identity above; this one exists to catch a
                    % mis-emitted curvature sense, and 6% catches that with
                    % six times the margin.
                    tc.verifyLessThan(abs(b.traced.mag/tc.P.M - 1), 0.06, ...
                        'traced magnification vs the 30x it was closed for');
                end
            end
        end

        function test_the_standoff_does_not_move_the_magnification(tc)
        %  The whole reason the field mirror is the chosen form: it sits
        %  where the marginal ray is small, so its power cannot buy M.  At a
        %  fixed interface the closure re-derives phi4 as the standoff moves,
        %  and M must not notice.
            m = [];
            for s = [0.05 0.20 0.40]
                D = afocal4_seed(tc.P, 'fm_standoff', s);
                f = [tempname '.in'];
                c = onCleanup(@() tc.rm_(f)); %#ok<NASGU>
                b = afocal4_build(tc.P, D, f, 'verify',false);
                m(end+1) = b.C.fo.mag; %#ok<AGROW>
            end
            tc.verifyLessThan(max(abs(m/tc.P.M - 1)), 1e-9, ...
                'magnification moved with the field-mirror standoff');
        end

        function test_interface_pose_matches_the_builder_exactly(tc)
        %  ONE POSING PATH (gate 2).  AFOCAL4_BUILD's place_coldstop_ and
        %  Telescope/align_exit_reference must land on the same plane for an
        %  unperturbed deck, or rung 4 uses a different convention from the
        %  rungs it is compared against.
            D = afocal4_seed(tc.P);
            f = [tempname '.in'];
            c = onCleanup(@() tc.rm_(f)); %#ok<NASGU>
            b = afocal4_build(tc.P, D, f, 'verify',false);

            t = macos.design.Telescope('family','tma', ...
                'aperture_diameter_m',tc.P.D, 'wavelength_m',tc.P.lambda, ...
                'grid_npts',tc.P.ngrid, 'model_size',tc.P.model_size);
            tt = [b.C.t, D.iface];
            for k = 1:numel(b.R)
                t.add_mirror(b.names{k}, 'radius_m',b.R(k), ...
                    'spacing_after_m',tt(k), 'convex',logical(b.C.convex(k)), ...
                    'conic',b.conic(k));
            end
            t.add_exit_reference('ColdStop','dist_m',D.iface);
            t.set_field_bias(tc.P.bias_deg*60);
            res = t.align_exit_reference();

            tc.verifyLessThan(norm(res.vpt(:) - b.coldstop.Vpt(:)), 1e-12, ...
                'interface station differs from Telescope/align_exit_reference');
            tc.verifyLessThan(norm(res.psi(:) - b.coldstop.psi(:)), 1e-12, ...
                'interface normal differs from Telescope/align_exit_reference');
        end

        function test_rigid_body_lands_in_the_deck_and_still_traces(tc)
        %  A decenter then a tilt about the DECENTERED vertex (CODE V's
        %  YDE-then-ADE order, which is what "his tilt/dec" means).  Read the
        %  emitted text, because it is a text-level edit.
            dy = 1.5e-3;   al = 2.0e-3;
            D = afocal4_seed(tc.P);
            f0 = [tempname '.in'];   f1 = [tempname '.in'];
            c0 = onCleanup(@() tc.rm_(f0)); %#ok<NASGU>
            c1 = onCleanup(@() tc.rm_(f1)); %#ok<NASGU>
            b0 = afocal4_build(tc.P, D, f0, 'verify',false);
            D.rb(1,:) = [dy al];                       % element P.rb_elts(1)
            b1 = afocal4_build(tc.P, D, f1);

            k  = tc.P.rb_elts(1);
            V0 = tc.grab_(fileread(f0),'VptElt');   P0 = tc.grab_(fileread(f0),'psiElt');
            V1 = tc.grab_(fileread(f1),'VptElt');   P1 = tc.grab_(fileread(f1),'psiElt');
            tc.verifyLessThan(abs((V1(2,k)-V0(2,k)) - dy), 1e-12, 'decenter');
            Rx = [1 0 0; 0 cos(al) -sin(al); 0 sin(al) cos(al)];
            tc.verifyLessThan(norm(P1(:,k) - Rx*P0(:,k)), 1e-12, 'tilt');
            % untouched elements stay untouched
            other = setdiff(1:size(V0,2)-1, k);
            tc.verifyLessThan(max(max(abs(V1(:,other)-V0(:,other)))), 1e-12, ...
                'a rigid body on one element moved another');
            % and the perturbed deck is still a telescope
            tc.verifyLessThan(abs(b1.traced.mag/tc.P.M - 1), 0.05, ...
                'perturbed deck no longer magnifies 30x');
            tc.verifyGreaterThan(b1.traced.nrays, 100, 'perturbed deck lost its rays');
            tc.verifyGreaterThan(norm(b1.coldstop.Vpt - b0.coldstop.Vpt), 0, ...
                'the interface plane did not follow the perturbed beam');
        end

        function test_score_terms_and_the_log_merit(tc)
        %  Gate 3: the merit's shape.  Residuals are non-negative; the WFE
        %  block is one entry per solve field; and the WFE-only diagnostic
        %  mode returns the SAME wavefront number as the full score (it must
        %  differ only in what it declines to measure).
            D = afocal4_seed(tc.P);
            f = [tempname '.in'];
            c = onCleanup(@() tc.rm_(f)); %#ok<NASGU>
            afocal4_build(tc.P, D, f, 'verify',false);

            S = afocal4_score(tc.P, f, 'nodes',9);
            tc.verifyTrue(S.ok, 'score failed on a buildable deck');
            tc.verifyEqual(numel(S.resid), size(tc.P.Fsolve,1) + 5, ...
                'residual length is not (one per field) + five pupil terms');
            tc.verifyGreaterThanOrEqual(min(S.resid), 0, ...
                'a log residual went negative -- the floor is not a floor');
            tc.verifyEqual(S.merit, sum(S.resid.^2), 'AbsTol', 1e-12);
            tc.verifyGreaterThan(S.worst, 0);

            W = afocal4_score(tc.P, f, 'pupil',false);
            tc.verifyEqual(W.wfe_max_nm, S.wfe_max_nm, 'RelTol', 1e-12, ...
                'the WFE-only mode changed the wavefront number');
            tc.verifyFalse(W.pupil_scored);
            tc.verifyEqual(numel(W.resid), size(tc.P.Fsolve,1));

            % a term twice inside its target earns nothing more
            tc.verifyEqual(max(0, log(0.4/tc.P.merit_floor)), 0, ...
                'AbsTol', 0, 'the merit floor is not clamping at zero');
        end

        function test_solve_smoke_is_wired_to_the_closure(tc)
        %  FAST: two iterations, coarse grid, conics only.  Checks the outer
        %  loop reaches the inner one and gives back a design that still
        %  closes -- not that it converged.
            Q = tc.P;
            Q.solve.ngrid = 11;   Q.solve.nodes = 7;   Q.solve.nodes_score = 7;
            Q.ngrid = 11;         Q.grid_n = 0;
            D = afocal4_seed(Q, 'bias_deg', 0);
            f = [tempname '.in'];
            c = onCleanup(@() tc.rm_(f)); %#ok<NASGU>
            R = afocal4_solve(Q, D, 'dofs',{'conic'}, 'deck',f, ...
                              'max_iter',2, 'quiet',true);
            tc.verifyGreaterThan(R.nfev, 1, 'the objective was never evaluated');
            tc.verifyTrue(R.S.ok, 'the converged design does not score');
            b = afocal4_build(Q, R.D, [tempname '.in'], 'verify',false);
            tc.verifyLessThan(abs(b.iface_pred - Q.iface), 1e-9, ...
                'the solved design no longer holds the pupil station');
            tc.verifyLessThan(abs(b.C.closure.mag_err), 1e-9, ...
                'the solved design no longer magnifies 30x');
        end
    end

    methods (Static, Access = private)
        function M = grab_(txt, key)
            t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens');
            M = zeros(3, numel(t));
            for i = 1:numel(t)
                M(:,i) = sscanf(strrep(t{i}{1},'D','E'), '%f', 3);
            end
        end
        function rm_(f)
            if exist(f,'file'), delete(f); end
        end
    end
end
