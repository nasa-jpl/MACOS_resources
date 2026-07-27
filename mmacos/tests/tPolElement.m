classdef tPolElement < matlab.unittest.TestCase
%TPOLELEMENT  Polarizer + waveplate elements (PLAN_POLARIZATION Phase 3).
%   Gates the two new engine elements -- TrPolarizer (EltID 15, finished
%   from a name-table-only stub) and WavePlate (EltID 18, new) -- plus the
%   PolElt trace routine, the PolAxis=/Retardance= Rx keywords, the SAVE
%   round-trip, and the polelt_set/polelt_get/jmat_elt_get API.
%
%   EVERY PHYSICS GATE IS A CLOSED-FORM JONES IDENTITY WRITTEN FROM THE
%   TEXTBOOK, NOT FROM THE ENGINE. That is the standing lesson from the
%   r_p sign defect (REVIEW_POL_SP_SIGN_2026-07-27.md): the Fresnel fold
%   gate could not catch a sign error because its "analytic" reference was
%   transcribed from the engine's own expression. The predictions below are
%   derived in the comments from the element Jones diag(1, exp(-i*delta))
%   and the geometry, and every one of them is a number the engine has to
%   land on rather than a ratio it can satisfy trivially.
%
%   CIRCULAR INPUT STATES ARE LOAD-BEARING, for the same reason they are in
%   tPolContrast and tVecChain: a linear-only suite is blind to conjugation
%   and retardance-sign errors.  test_qwp_linear_to_circular pins the SIGNED
%   circular Stokes parameter, which flips if the retardance convention
%   flips; test_qwp_circular_to_linear runs a circular state in.
%
%   SCOPE. The fixture is at strict normal incidence. That is not
%   laziness: off normal, an ideal polarizer carries an unresolved O(sin^2
%   AOI) convention question (declared PASS axis projected, versus declared
%   BLOCK axis projected and complemented), which is written up in
%   REVIEW_POL_ELEMENTS_2026-07-27.md and referred to the Fable lane rather
%   than guessed.  test_offnormal_convention_magnitude MEASURES that
%   ambiguity so the packet carries a number, and is explicitly a scope
%   report, not a correctness claim.

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_PolElt.in'
        RxRefName = 'Rx_PolElt_Ref.in'
        Pol1      = 2    % TrPolarizer -- input polarizer
        WP1       = 3    % WavePlate
        WP2       = 4    % WavePlate (defaults to zero retardance = identity)
        Anal      = 5    % TrPolarizer -- analyzer
        Det       = 7    % FocalPlane
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            macos.init(testCase.ModelSize);
        end
    end

    methods (TestMethodSetup)
        function loadRx(testCase)
            macos.load_rx(rx_fixture_path(testCase.RxName));
        end
    end

    methods (Access = private)
        function [Ex, Ey, Ez, m] = fieldAt(~, srf)
            % Per-ray transverse field at SRF over the unvignetted rays.
            rf = macos.ray_field(srf);
            m  = (rf.status == 0);
            Ex = rf.Ex(m);  Ey = rf.Ey(m);  Ez = rf.Ez(m);
        end

        function S = stokes(testCase, srf)
            % Stokes parameters summed over the pupil. The conjugation
            % order is written out explicitly rather than with MATLAB's ',
            % which conjugates its LEFT operand -- the exact slip that
            % built conj(C) in pol_contrast_floor and passed every linear
            % gate vacuously.
            [Ex, Ey, ~] = testCase.fieldAt(srf);
            S.S0 = sum(abs(Ex).^2 + abs(Ey).^2);
            S.S1 = sum(abs(Ex).^2 - abs(Ey).^2);
            S.S2 = sum(2*real(Ex .* conj(Ey)));
            S.S3 = sum(2*imag(Ex .* conj(Ey)));
        end

        function configure(testCase, thPol, thWP1, rWP1, thWP2, rWP2, thAn)
            % Axis angles in RADIANS from +x, in the z-normal plane.
            ax = @(t) [cos(t) sin(t) 0];
            macos.polarizer(testCase.Pol1, 'axis', ax(thPol));
            macos.waveplate(testCase.WP1, 'axis', ax(thWP1), 'retardance', rWP1);
            macos.waveplate(testCase.WP2, 'axis', ax(thWP2), 'retardance', rWP2);
            macos.polarizer(testCase.Anal, 'axis', ax(thAn));
        end

        function polOn(~, ex, ey)
            macos.polarization('on', 'Ex', [real(ex) imag(ex)], ...
                                     'Ey', [real(ey) imag(ey)]);
        end
    end

    methods (Test)

        % =============================================================
        % A. Plumbing: keywords, API round-trip, SAVE, guards
        % =============================================================

        function test_rx_keywords_parse(testCase)
            % The Rx PolAxis= / Retardance= keywords reach elt_mod. Before
            % this phase both element types were name-table-only stubs, so
            % a prescription naming them loaded and then did nothing.
            p = macos.polarizer(testCase.Pol1);
            testCase.verifyEqual(p.elt_type, 15);
            testCase.verifyEqual(p.axis, [1;0;0], 'AbsTol', 1e-14);

            w = macos.waveplate(testCase.WP1);
            testCase.verifyEqual(w.elt_type, 18);
            testCase.verifyEqual(w.axis, [1;0;0], 'AbsTol', 1e-14);
            % Retardance= 0.25 waves at the file's Wavelen
            testCase.verifyEqual(w.retardance, 0.25, 'AbsTol', 1e-12);

            w2 = macos.waveplate(testCase.WP2);
            testCase.verifyEqual(w2.retardance, 0, 'AbsTol', 1e-14);
        end

        function test_polelt_roundtrip(testCase)
            % polelt_get o polelt_set == identity at fixed wavelength.
            a = [0.3 -0.8 0.5];
            macos.waveplate(testCase.WP1, 'axis', a, 'retardance', 0.375);
            w = macos.waveplate(testCase.WP1);
            testCase.verifyEqual(w.axis(:).', a, 'AbsTol', 1e-14);
            testCase.verifyEqual(w.retardance, 0.375, 'AbsTol', 1e-14);

            % the axis is stored as given -- NOT silently unitized, so a
            % caller can read back exactly what it wrote
            testCase.verifyGreaterThan(abs(norm(a) - 1), 1e-3);
        end

        function test_partial_update_preserves_other_field(testCase)
            macos.waveplate(testCase.WP1, 'axis', [0 1 0], 'retardance', 0.4);
            macos.waveplate(testCase.WP1, 'retardance', 0.1);   % axis untouched
            w = macos.waveplate(testCase.WP1);
            testCase.verifyEqual(w.axis, [0;1;0], 'AbsTol', 1e-14);
            testCase.verifyEqual(w.retardance, 0.1, 'AbsTol', 1e-14);

            macos.waveplate(testCase.WP1, 'axis', [1 0 0]);     % ret untouched
            w = macos.waveplate(testCase.WP1);
            testCase.verifyEqual(w.retardance, 0.1, 'AbsTol', 1e-14);
        end

        function test_save_roundtrip(testCase)
            % PolAxis= / Retardance= survive SAVE and reload. The writer
            % inverts the load-time Wavelen scaling, so the WAVES value is
            % what round-trips -- the same inversion the Coating= writer does.
            macos.polarizer(testCase.Pol1, 'axis', [0 1 0]);
            macos.waveplate(testCase.WP1, 'axis', [1 1 0], 'retardance', 0.3125);

            f = [tempname '.in'];
            c = onCleanup(@() delete_if_present(f));
            macos.save_rx(f);
            macos.load_rx(f);

            p = macos.polarizer(testCase.Pol1);
            testCase.verifyEqual(p.elt_type, 15);
            testCase.verifyEqual(p.axis, [0;1;0], 'AbsTol', 1e-12);

            w = macos.waveplate(testCase.WP1);
            testCase.verifyEqual(w.elt_type, 18);
            % NOTE the deliberate asymmetry: the API stores the axis AS
            % GIVEN (so a caller reads back what it wrote), while the Rx
            % parser UNITIZES on load, matching psiElt=. So a non-unit
            % axis comes back normalized after a save/reload round trip.
            % That is a direction either way -- PolElt normalizes per ray.
            testCase.verifyEqual(w.axis, [1;1;0]/sqrt(2), 'AbsTol', 1e-12);
            testCase.verifyEqual(w.retardance, 0.3125, 'AbsTol', 1e-10);
        end

        function test_wrong_element_type_refused(testCase)
            % Element 1 is Obscuring. Setting polarizer parameters on it
            % must FAIL loudly rather than write into a surface that will
            % never read them back.
            testCase.verifyError(@() macos.polarizer(1, 'axis', [1 0 0]), ...
                                 ?MException);
        end

        function test_zero_axis_refused(testCase)
            testCase.verifyError( ...
                @() macos.polarizer(testCase.Pol1, 'axis', [0 0 0]), ...
                'macos:polarizer:zeroAxis');
        end

        function test_setter_dirties_the_trace(testCase)
            % polelt_set calls modified_rx. Without that, a configuration
            % change followed by a trace would harvest the PREVIOUS state's
            % RayE -- the exact stale-state bug pol_set had (macos bef4598).
            testCase.polOn(1, 0);
            testCase.configure(0, 0, 0, 0, 0, 0);      % analyzer along x
            macos.trace(testCase.Anal);
            [Ex1, ~, ~] = testCase.fieldAt(testCase.Anal);

            macos.polarizer(testCase.Anal, 'axis', [0 1 0]);   % now crossed
            macos.trace(testCase.Anal);
            [Ex2, Ey2, ~] = testCase.fieldAt(testCase.Anal);

            testCase.verifyGreaterThan(max(abs(Ex1)), 1e-6);
            testCase.verifyLessThan(max(abs(Ex2) + abs(Ey2)), 1e-30);
        end

        % =============================================================
        % B. The unpolarized path is untouched
        % =============================================================

        function test_unpolarized_bit_identical_to_reference_twin(testCase)
            % PolElt's geometry IS RefSrf's, so with polarization OFF the
            % polarizing elements must be indistinguishable from plain
            % Reference surfaces. Compared against the geometric twin
            % fixture, BIT-identical (verifyEqual with no tolerance).
            macos.polarization('off');
            macos.load_rx(rx_fixture_path(testCase.RxName));
            tA = macos.trace(testCase.Det);
            opdA = macos.opd();
            intA = macos.intensity(testCase.Det);
            stA  = macos.get_ray_status(tA.nRays);

            macos.load_rx(rx_fixture_path(testCase.RxRefName));
            tB = macos.trace(testCase.Det);
            opdB = macos.opd();
            intB = macos.intensity(testCase.Det);
            stB  = macos.get_ray_status(tB.nRays);

            testCase.verifyEqual(tA.nRays, tB.nRays);
            testCase.verifyEqual(stA.status, stB.status);
            testCase.verifyEqual(stA.fail_elt, stB.fail_elt);
            testCase.verifyEqual(opdA, opdB);        % bit-identical
            testCase.verifyEqual(intA, intB);        % bit-identical
        end

        % =============================================================
        % C. Polarizer physics
        % =============================================================

        function test_malus_law(testCase)
            % Input polarizer along x, analyzer at theta. The projection
            % onto the analyzer axis is cos(theta), so the transmitted
            % INTENSITY is I0*cos^2(theta) -- Malus. Textbook; nothing
            % here is read off the engine.
            testCase.polOn(1, 0);
            th = linspace(0, pi, 13);
            I  = zeros(size(th));
            for k = 1:numel(th)
                testCase.configure(0, 0, 0, 0, 0, th(k));
                macos.trace(testCase.Anal);
                [Ex, Ey, Ez] = testCase.fieldAt(testCase.Anal);
                I(k) = sum(abs(Ex).^2 + abs(Ey).^2 + abs(Ez).^2);
            end
            pred = I(1) * cos(th).^2;
            testCase.verifyEqual(I, pred, 'AbsTol', 1e-12*I(1));

            % NON-VACUITY: the curve must actually sweep. An element that
            % simply passed everything (the pre-Phase-3 behaviour of these
            % EltIDs, which were name-table-only stubs) gives a FLAT
            % I(theta) and would satisfy none of this.
            testCase.verifyGreaterThan(std(I)/mean(I), 0.5);
            testCase.verifyLessThan(min(I)/max(I), 1e-25);
        end

        function test_crossed_polarizer_extinction(testCase)
            % Ideal crossed polarizers extinguish EXACTLY -- not to a
            % tolerance -- when the two axes are exactly orthogonal as
            % floating-point vectors. The projection onto the analyzer
            % axis of a field lying entirely along the orthogonal axis is
            % identically zero, so anything nonzero would mean the
            % transverse basis is not orthonormal.
            testCase.polOn(1, 0);
            testCase.configure(0, 0, 0, 0, 0, 0);
            macos.polarizer(testCase.Pol1, 'axis', [1 0 0]);
            macos.polarizer(testCase.Anal, 'axis', [0 1 0]);
            macos.trace(testCase.Anal);
            [Ex, Ey, Ez] = testCase.fieldAt(testCase.Anal);
            testCase.verifyEqual(max(abs(Ex)), 0);
            testCase.verifyEqual(max(abs(Ey)), 0);
            testCase.verifyEqual(max(abs(Ez)), 0);

            % Rotating the analyzer to pi/2 instead gives cos(pi/2) =
            % 6.1e-17 rather than 0, so the residual INTENSITY is the
            % square of that -- ~1e-33 of the incident power, and bounded
            % by it. Stated rather than tolerated: the extinction is
            % limited by the caller's axis, not by the element.
            testCase.configure(0, 0, 0, 0, 0, pi/2);
            macos.trace(testCase.Anal);
            [Ex, Ey, ~] = testCase.fieldAt(testCase.Anal);
            resid = sum(abs(Ex).^2 + abs(Ey).^2);
            testCase.verifyLessThan(resid, 10*cos(pi/2)^2);
        end

        function test_polarizer_is_idempotent(testCase)
            % Two polarizers on a common axis == one. P^2 = P.
            testCase.polOn(1/sqrt(2), 1/sqrt(2));
            testCase.configure(0.3, 0, 0, 0, 0, 0.3);   % both at 0.3 rad
            macos.trace(testCase.Anal);
            [Ex2, Ey2, ~] = testCase.fieldAt(testCase.Anal);

            % same rig with the analyzer neutralised by aligning it with
            % the light already passed
            testCase.configure(0.3, 0, 0, 0, 0, 0.3);
            macos.trace(testCase.Pol1);
            [Ex1, Ey1, ~] = testCase.fieldAt(testCase.Pol1);

            testCase.verifyEqual(abs(Ex2), abs(Ex1), 'AbsTol', 1e-15);
            testCase.verifyEqual(abs(Ey2), abs(Ey1), 'AbsTol', 1e-15);
        end

        function test_polarizer_jones_is_projector(testCase)
            testCase.polOn(1, 0);
            testCase.configure(0, 0, 0, 0, 0, 0);
            macos.trace(testCase.Anal);
            J = macos.elt_jones(testCase.Anal);
            testCase.verifyEqual(J, [1 0; 0 0], 'AbsTol', 1e-15);
            testCase.verifyEqual(J*J, J, 'AbsTol', 1e-15);
        end

        % =============================================================
        % D. Waveplate physics
        % =============================================================

        function test_qwp_linear_to_circular(testCase)
            % Linear input along x, quarter-wave plate with its FAST axis
            % at 45 degrees. In the (fast,slow) basis the input is
            % (1,-1)/sqrt(2); the plate applies diag(1, exp(-i*pi/2)) =
            % diag(1,-i), giving ((1-i)/2, (1+i)/2) in (x,y). Hence
            %   S1 = S2 = 0,  S3/S0 = -1,
            % all four numbers written from the Jones algebra above.
            %
            % THE SIGN OF S3 IS THE POINT. It is what distinguishes the
            % engine's exp(+i*omega*t) / fast-axis-leads convention from
            % its opposite; a suite that only checked |S3| would accept
            % either. With Jb = exp(+i*delta) this gate reads +1.
            testCase.polOn(1, 0);
            testCase.configure(0, pi/4, 0.25, 0, 0, 0);
            macos.trace(testCase.WP1);
            S = testCase.stokes(testCase.WP1);

            testCase.verifyEqual(S.S1/S.S0, 0, 'AbsTol', 1e-14);
            testCase.verifyEqual(S.S2/S.S0, 0, 'AbsTol', 1e-14);
            testCase.verifyEqual(S.S3/S.S0, -1, 'AbsTol', 1e-14);

            % NON-VACUITY, in-suite: the same rig with zero retardance
            % leaves the state linear, so S3 collapses and S1/S2 take over.
            testCase.configure(0, pi/4, 0, 0, 0, 0);
            macos.trace(testCase.WP1);
            S0r = testCase.stokes(testCase.WP1);
            testCase.verifyEqual(S0r.S3/S0r.S0, 0, 'AbsTol', 1e-14);
            testCase.verifyEqual(S0r.S1/S0r.S0, 1, 'AbsTol', 1e-14);
        end

        function test_qwp_circular_to_linear(testCase)
            % The return trip. A CIRCULAR state cannot be delivered to a
            % plate through the fixture's input polarizer -- an ideal
            % polarizer always projects -- so the circular state is MADE by
            % the first plate and consumed by the second:
            %   linear -> QWP(45) -> circular -> QWP(45) -> linear.
            % The intermediate state is checked as well, otherwise the
            % round trip would prove nothing about what was in between.
            testCase.polOn(1, 0);
            testCase.configure(0, pi/4, 0.25, pi/4, 0.25, 0);
            macos.trace(testCase.WP2);
            S = testCase.stokes(testCase.WP2);
            DoLP = hypot(S.S1, S.S2)/S.S0;
            DoCP = abs(S.S3)/S.S0;
            testCase.verifyEqual(DoLP, 1, 'AbsTol', 1e-14);
            testCase.verifyEqual(DoCP, 0, 'AbsTol', 1e-14);

            % and after ONE plate it really was circular -- otherwise the
            % round trip proves nothing about the intermediate state
            macos.trace(testCase.WP1);
            Smid = testCase.stokes(testCase.WP1);
            testCase.verifyEqual(abs(Smid.S3)/Smid.S0, 1, 'AbsTol', 1e-14);
        end

        function test_hwp_rotates_by_2theta(testCase)
            % A half-wave plate with its fast axis at theta applies
            % diag(1,-1) in the (fast,slow) basis, which is a REFLECTION of
            % the transverse plane about the fast axis. Linear input at 0
            % therefore emerges at 2*theta:  E = (cos 2theta, sin 2theta).
            testCase.polOn(1, 0);
            th  = linspace(0, pi/2, 9);
            ang = zeros(size(th));
            % Orientation is taken from the Stokes pair as
            % 0.5*atan2(S2,S1), not from atan2(Ey,Ex): the field carries a
            % common complex propagation phase exp(-i*2*pi*L/lambda) from
            % the hop between elements, which a real-part ratio would fold
            % into the answer. S1/S2 are quadratic in the field, so the
            % common phase cancels identically.
            for k = 1:numel(th)
                testCase.configure(0, th(k), 0.5, 0, 0, 0);
                macos.trace(testCase.WP1);
                S = testCase.stokes(testCase.WP1);
                ang(k) = 0.5*atan2(S.S2, S.S1);
            end
            testCase.verifyEqual(unwrap(2*ang)/2, unwrap(2*(2*th))/2, ...
                                 'AbsTol', 1e-12);

            p = polyfit(th, unwrap(2*ang)/2, 1);
            testCase.verifyEqual(p(1), 2, 'AbsTol', 1e-10);   % the 2-theta law

            % NON-VACUITY: zero retardance means rotating the plate does
            % nothing at all -- slope 0, not 2.
            for k = 1:numel(th)
                testCase.configure(0, th(k), 0, 0, 0, 0);
                macos.trace(testCase.WP1);
                S = testCase.stokes(testCase.WP1);
                ang(k) = 0.5*atan2(S.S2, S.S1);
            end
            p0 = polyfit(th, unwrap(2*ang)/2, 1);
            testCase.verifyLessThan(abs(p0(1)), 1e-10);
        end

        function test_two_qwp_equal_one_hwp(testCase)
            % Composition: diag(1,e^-i(pi/2))^2 = diag(1,e^-i(pi)). Two
            % quarter-wave plates on a COMMON axis must reproduce a single
            % half-wave plate, field for field. This is the gate that the
            % elements CASCADE correctly -- a single-element test cannot
            % see an error in how RayE is carried between them.
            testCase.polOn(1, 0);
            thf = 0.4;

            testCase.configure(0, thf, 0.25, thf, 0.25, 0);
            macos.trace(testCase.WP2);
            [ExA, EyA, ~] = testCase.fieldAt(testCase.WP2);

            testCase.configure(0, thf, 0.5, thf, 0, 0);
            macos.trace(testCase.WP2);
            [ExB, EyB, ~] = testCase.fieldAt(testCase.WP2);

            testCase.verifyEqual(ExA, ExB, 'AbsTol', 1e-15);
            testCase.verifyEqual(EyA, EyB, 'AbsTol', 1e-15);

            % NON-VACUITY: the pair is not trivially equal to anything --
            % a single QWP at the same axis differs grossly.
            testCase.configure(0, thf, 0.25, thf, 0, 0);
            macos.trace(testCase.WP2);
            [ExC, ~, ~] = testCase.fieldAt(testCase.WP2);
            testCase.verifyGreaterThan(max(abs(ExA - ExC))/max(abs(ExA)), 0.1);
        end

        function test_waveplate_is_unitary(testCase)
            % A lossless retarder conserves power for EVERY input state.
            %
            % Getting genuinely different states onto a plate takes care in
            % this fixture: the input polarizer projects, so simply varying
            % the SOURCE state would deliver the same linear state to WP1
            % every time and the loop would be vacuous. Instead the
            % relative angle is varied at WP1 (linear input, generic
            % relative orientation), and a CIRCULAR input is delivered to
            % WP2 by making WP1 a quarter-wave plate at 45 degrees.
            testCase.polOn(1, 0);

            % --- linear input, three relative orientations, at WP1 ---
            for th = [0.0, 0.37, pi/4]
                testCase.configure(0, th, 0.31, 0, 0, 0);
                macos.trace(testCase.Pol1);
                [a, b, c] = testCase.fieldAt(testCase.Pol1);
                P0 = sum(abs(a).^2 + abs(b).^2 + abs(c).^2);
                macos.trace(testCase.WP1);
                [a, b, c] = testCase.fieldAt(testCase.WP1);
                P1 = sum(abs(a).^2 + abs(b).^2 + abs(c).^2);
                testCase.verifyEqual(P1/P0, 1, 'AbsTol', 1e-14);
                testCase.verifyGreaterThan(P0, 0);

                J = macos.elt_jones(testCase.WP1);
                testCase.verifyEqual(J'*J, eye(2), 'AbsTol', 1e-15);
            end

            % --- CIRCULAR input at WP2, built by a QWP at WP1 ---
            testCase.configure(0, pi/4, 0.25, 0.19, 0.42, 0);
            macos.trace(testCase.WP1);
            Smid = testCase.stokes(testCase.WP1);
            testCase.verifyEqual(abs(Smid.S3)/Smid.S0, 1, 'AbsTol', 1e-14);
            [a, b, c] = testCase.fieldAt(testCase.WP1);
            P0 = sum(abs(a).^2 + abs(b).^2 + abs(c).^2);
            macos.trace(testCase.WP2);
            [a, b, c] = testCase.fieldAt(testCase.WP2);
            P1 = sum(abs(a).^2 + abs(b).^2 + abs(c).^2);
            testCase.verifyEqual(P1/P0, 1, 'AbsTol', 1e-14);

            % NON-VACUITY: the same check on the POLARIZER must FAIL the
            % unitarity condition -- otherwise the test would also pass on
            % an engine that had stopped applying anything at all.
            Jp = macos.elt_jones(testCase.Pol1);
            testCase.verifyGreaterThan(norm(Jp'*Jp - eye(2)), 0.5);
        end

        function test_zero_retardance_is_identity(testCase)
            % A plate with no retardance must be EXACTLY the identity, not
            % approximately: exp(-i*0) is 1 in floating point, so the
            % reassembly must return the field bit-for-bit. This is what
            % makes "leave the second plate alone" safe in every other gate.
            % Compared in MAGNITUDE: the hop between elements applies the
            % engine's common propagation phase exp(-i*2*pi*L/lambda) to
            % all three components, which is geometry, not the element.
            testCase.polOn(1/sqrt(2), 1i/sqrt(2));
            testCase.configure(0.6, 0.2, 0, 0, 0, 0.6);
            macos.trace(testCase.Pol1);
            [Ex0, Ey0, ~] = testCase.fieldAt(testCase.Pol1);
            macos.trace(testCase.WP1);
            [Ex1, Ey1, ~] = testCase.fieldAt(testCase.WP1);
            testCase.verifyEqual(abs(Ex1), abs(Ex0), 'AbsTol', 1e-17);
            testCase.verifyEqual(abs(Ey1), abs(Ey0), 'AbsTol', 1e-17);
            % and the relative phase between components is untouched
            testCase.verifyEqual(angle(sum(Ey1 .* conj(Ex1))), ...
                                 angle(sum(Ey0 .* conj(Ex0))), ...
                                 'AbsTol', 1e-14);
        end

        % =============================================================
        % E. Conventions and scope
        % =============================================================

        function test_retardance_is_chromatic(testCase)
            % The stored quantity is the PHYSICAL retardance
            % (n_slow-n_fast)*d, so a plate set to R waves at lambda is
            % R*lambda/lambda2 waves at lambda2 -- a fixed piece of glass.
            % Same convention Coating= layer thickness already carries.
            lam0 = macos.get_src_wvl();
            macos.waveplate(testCase.WP1, 'axis', [1 0 0], 'retardance', 0.25);
            macos.set_src_wvl(2*lam0);
            w = macos.waveplate(testCase.WP1);
            testCase.verifyEqual(w.retardance, 0.125, 'RelTol', 1e-12);
            macos.set_src_wvl(lam0);
            w = macos.waveplate(testCase.WP1);
            testCase.verifyEqual(w.retardance, 0.25, 'RelTol', 1e-12);
        end

        function test_retardance_sign_matches_engine_convention(testCase)
            % Direct read of the applied phase. Input at 45 degrees to a
            % plate whose fast axis is x: the y (slow) component must LAG
            % the x (fast) component by delta = 2*pi*R. Under the engine's
            % exp(-i*2*pi*L*N/lambda) propagator that is a factor
            % exp(-i*delta), so arg(Ey/Ex) = -delta.
            R = 0.3;
            testCase.polOn(1/sqrt(2), 1/sqrt(2));
            testCase.configure(pi/4, 0, R, 0, 0, 0);
            macos.trace(testCase.WP1);
            [Ex, Ey, ~] = testCase.fieldAt(testCase.WP1);
            ph = angle(sum(Ey .* conj(Ex)));
            testCase.verifyEqual(ph, -2*pi*R, 'AbsTol', 1e-12);

            J = macos.elt_jones(testCase.WP1);
            testCase.verifyEqual(angle(J(2,2)), -2*pi*R, 'AbsTol', 1e-14);
            testCase.verifyEqual(J(1,1), complex(1,0), 'AbsTol', 1e-15);
        end

        function test_axis_parallel_to_ray_extinguishes(testCase)
            % An axis along the propagation direction leaves the element
            % with no transverse axis. That is a prescription error, and
            % the engine extinguishes rather than silently picking some
            % arbitrary basis (which is what the s/p fallback in Reflector
            % does, and which would be undetectable here).
            testCase.polOn(1, 0);
            testCase.configure(0, 0, 0, 0, 0, 0);
            macos.polarizer(testCase.Anal, 'axis', [0 0 1]);
            macos.trace(testCase.Anal);
            [Ex, Ey, Ez] = testCase.fieldAt(testCase.Anal);
            testCase.verifyEqual(max(abs(Ex) + abs(Ey) + abs(Ez)), 0);
        end

        function test_grid_carries_the_polarizing_train(testCase)
            % SCOPE CLAIM, measured. Every polarizing element in the
            % fixture precedes the single physical-optics leg, so the
            % Tranche-1 seed carries the whole train and the DETECTOR plane
            % must obey Malus too -- not just the rays. On a train where a
            % polarizing element sat AFTER the first leg this would fail,
            % which is exactly the Tranche-2 gap tPolContrastCoro pins.
            testCase.polOn(1, 0);
            th = [0 pi/6 pi/3 pi/2];
            I  = zeros(size(th));
            for k = 1:numel(th)
                testCase.configure(0, 0, 0, 0, 0, th(k));
                macos.trace(testCase.Det);
                I(k) = sum(macos.intensity(testCase.Det), 'all');
            end
            testCase.verifyEqual(I, I(1)*cos(th).^2, 'RelTol', 1e-10, ...
                                 'AbsTol', 1e-12*I(1));
        end

        function test_offnormal_convention_magnitude(testCase)
            % SCOPE REPORT, NOT A CORRECTNESS CLAIM.
            %
            % Off normal incidence an ideal polarizer is ambiguous at
            % O(sin^2 AOI): declaring the PASS axis and projecting it (what
            % PolElt does) is not the same as declaring the BLOCK axis,
            % projecting THAT, and transmitting the complement, because
            % orthographic projection does not preserve orthogonality.
            %
            % This test computes both constructions for the fixture's ray
            % bundle at a deliberately large tilt and records the
            % difference, so the packet carries a number rather than a
            % worry. It asserts only (a) that at normal incidence the
            % ambiguity is identically zero -- which is why every physics
            % gate above is safe -- and (b) that the off-normal difference
            % has the O(sin^2) size the argument predicts, so nobody later
            % mistakes it for a bug or for a bigger effect than it is.
            % ambiguity(aoi, azimuth) -- angle between the two candidate
            % pass axes, in radians
            amb = @(aoi, az) local_axis_ambiguity(aoi, az);

            % (a) NORMAL INCIDENCE: identically zero at every azimuth.
            %     This is why every physics gate in this class is safe.
            for az = [0 pi/8 pi/4 pi/3]
                testCase.verifyEqual(amb(0, az), 0, 'AbsTol', 1e-15);
            end

            % (b) The ambiguity also vanishes off normal when the declared
            %     axis lies IN the plane of incidence or PERPENDICULAR to
            %     it -- both projections stay orthogonal there. Worth
            %     knowing: the obvious test tilt (axis along x, tilt in
            %     x-z) is exactly this degenerate case and would report a
            %     reassuring zero that means nothing.
            testCase.verifyEqual(amb(deg2rad(20), 0),    0, 'AbsTol', 1e-15);
            testCase.verifyEqual(amb(deg2rad(20), pi/2), 0, 'AbsTol', 1e-15);

            % (c) It is largest at 45 degrees of azimuth, where the closed
            %     form is  acos( 2*cos(aoi) / (1 + cos(aoi)^2) ). Pinned
            %     so the packet quotes a measured number rather than a
            %     worry: 3.56 deg of axis ambiguity at 20 deg AOI.
            for aoi = deg2rad([5 10 20 30])
                pred = acos(2*cos(aoi)/(1 + cos(aoi)^2));
                testCase.verifyEqual(amb(aoi, pi/4), pred, 'AbsTol', 1e-12);
                testCase.verifyLessThanOrEqual(amb(aoi, pi/4), sin(aoi)^2);
            end
            testCase.verifyEqual(rad2deg(amb(deg2rad(20), pi/4)), 3.562, ...
                                 'AbsTol', 0.005);
        end
    end
end

function delete_if_present(f)
if exist(f, 'file'), delete(f); end
end

function dth = local_axis_ambiguity(aoi, az)
%LOCAL_AXIS_AMBIGUITY  Angle between the two candidate polarizer pass axes.
%   Ray tilted by AOI in the x-z plane; the element's declared in-plane
%   axis pair is (pass, block) rotated by AZ about z. Construction 1 (what
%   PolElt does) projects the PASS axis into the transverse plane.
%   Construction 2 projects the BLOCK axis and takes the complement. They
%   agree only where projection happens to preserve orthogonality.
r    = [sin(aoi) 0 cos(aoi)];
pass = [cos(az)  sin(az) 0];
blok = [-sin(az) cos(az) 0];
proj = @(v) (v - dot(v,r)*r) / norm(v - dot(v,r)*r);
a1 = proj(pass);
a2 = cross(r, proj(blok));  a2 = a2 / norm(a2);
dth = real(acos(min(1, abs(dot(a1, a2)))));
end
