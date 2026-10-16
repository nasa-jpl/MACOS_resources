classdef tPolRadiometric < matlab.unittest.TestCase
%TPOLRADIOMETRIC  Coated-Refractor transmission radiometry (PLAN_POLARIZATION).
%   Gates the landing of REVIEW_POL_RADIOMETRIC_2026-07-28.md: the coated
%   Refractor branch in elemsub.F composed plain Fresnel FIELD coefficients
%   through the Airy recursion, while the UNCOATED branch folds in the
%   radiometric factor sqrt(n2*cos_t/(n1*cos_i)) so that |TP|^2 IS the POWER
%   transmittance.  The two branches therefore disagreed on a surface that is
%   physically the SAME surface, and a coated lens under-transmitted.
%
%   THE CONVENTION IS THE INCUMBENT ONE, NOT A NEW ONE.  Section A gates the
%   UNCOATED branch against the textbook power transmittance first, so what
%   the coated branch is being brought to is established from Macleod rather
%   than asserted.  The fix is ONE factor applied ONCE after the recursion:
%
%       TP *= sqrt(Re(n_sub)*cos_sub/(na*cos_inc))     (TS likewise)
%
%   NEVER per interface: the radiometric conversion in the multilayer theorem
%   is boundary-to-boundary and the interior-layer factors cancel identically
%   (elemsub.F carries the argument, including why the fact that a plain
%   chain's per-interface factors happen to telescope is NOT a licence).
%
%   Two tests here pin what the landing did NOT move, which is most of what
%   makes it safe: section D's polarization-state test (a common real scalar
%   cancels in t_p/t_s, so diattenuation and retardance are untouched and the
%   existing Phase-2/2b coated results stand) and section F's wavelength sweep
%   (a real scalar cannot create, shift, or flatten the quarter-wave
%   interference structure the recursion already had).
%
%   EVERY ANALYTIC IS WRITTEN FRESH FROM THE TEXTBOOK.  multilayer_T below is
%   the Abeles characteristic-matrix formulation (Macleod, "Thin-Film Optical
%   Filters", ch. 2; equivalently Born & Wolf 1.6.4), typed from the tilted-
%   admittance definitions and NOT transcribed from elemsub.F.  That is the
%   standing lesson from the r_p sign defect
%   (REVIEW_POL_SP_SIGN_2026-07-27.md), where a gate built its "analytic"
%   reference out of the engine's own expression and was therefore circular
%   in exactly the sign it should have caught.
%
%   THE PRE-FIX NUMBERS, so the gates are provably non-vacuous.  Measured on
%   these two fixtures against the engine as it stood at pol-core a3417ce:
%
%     index-matched layer, coated/uncoated amplitude
%       normal incidence            0.8164965809  = 1/sqrt(1.5)
%       45 deg, p                   0.7311104457  = 1/sqrt(1.5*cos_t/cos_i)
%       45 deg, s                   0.7311104457  (same -- the factor is scalar)
%     detector-plane intensity      0.6666666667  = 1/1.5
%
%   All four are 1.0 after the fix.  Rebuilt against that pre-fix engine this
%   class scores 6 pass / 7 fail; the six that pass are the ones that SHOULD
%   (section A's two uncoated-convention tests, the fixture's s/p purity, the
%   both-faces-coated telescoping identity, the polarization-state invariance,
%   and pol-off inertness), and each of those says so in its own comment.
%
%   Note that the normal-incidence number alone gates only the INDEX half of
%   the factor: both cosines are 1 there, so a normal-only suite is blind to
%   the cos_sub/cos_inc half.  That is why Rx_Refract45.in exists.
%
%   Fixtures: Rx_Refract.in (normal incidence, air/glass/air parallel plate,
%   one physical-optics leg so the detector plane is reached) and
%   Rx_Refract45.in (single interface at 45 deg AOI, ray level).  Both put
%   a passive Reference immediately upstream of the refractor: every ratio
%   below is formed against the MEASURED incident field, never an assumed
%   source amplitude -- the IFO slice-1 finding-2 rule.  See the fixture
%   headers for the rest of the reasoning.

    properties (Constant)
        ModelSize = 128

        % --- normal-incidence plate ---
        RxName  = 'Rx_Refract.in'
        Pre     = 2      % Reference: the measured incident field
        Face1   = 3      % Refractor, air -> glass (element under test)
        Face2   = 4      % Refractor, glass -> air
        Det     = 6      % FocalPlane (behind the one NFPlane->Geometric leg)

        % --- 45 deg single interface ---
        Rx45    = 'Rx_Refract45.in'
        Pre45   = 2
        Face45  = 3

        nGlass  = 1.5    % substrate index, both fixtures
        nAir    = 1.0
        lambda0 = 1.0e-6 % Wavelen in both fixtures (BaseUnits = WaveUnits = m)

        % MgF2 at ~1 um.  Quarter-wave optical thickness at lambda0:
        % d = lambda0/(4*n) -- the classic single-layer AR coating, and the
        % one stack whose spectral shape is unmistakable.
        nMgF2   = 1.38
        dMgF2   = 1.0e-6/(4*1.38)
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            macos.init(testCase.ModelSize);
        end
    end

    methods (Access = private)

        function polOn(~, ex, ey)
            macos.polarization('on', 'Ex', [real(ex) imag(ex)], ...
                                     'Ey', [real(ey) imag(ey)]);
        end

        function [amp, geo] = ampRatio(~, iPre, iFace)
            % Per-ray |E_face| / |E_pre| over the rays good at BOTH stations,
            % plus the geometry needed for the analytic, read back out of the
            % engine (ray directions + surface normal) rather than assumed
            % from the fixture's declared numbers.
            %
            % macos.ray_field returns the CURRENT RayE -- iElt selects only
            % the surface normal -- so each station needs its own trace.
            macos.trace(iPre);   a = macos.ray_field(iPre);
            macos.trace(iFace);  b = macos.ray_field(iFace);
            m  = (a.status == 0) & (b.status == 0);
            Aa = sqrt(abs(a.Ex(m)).^2 + abs(a.Ey(m)).^2 + abs(a.Ez(m)).^2);
            Ab = sqrt(abs(b.Ex(m)).^2 + abs(b.Ey(m)).^2 + abs(b.Ez(m)).^2);
            amp = Ab ./ Aa;

            k  = [a.kx(m) a.ky(m) a.kz(m)];        % incident ray directions
            r  = [b.kx(m) b.ky(m) b.kz(m)];        % refracted ray directions
            nn = [b.nx(m) b.ny(m) b.nz(m)];        % element normal
            geo.cos_inc = abs(sum(k .* nn, 2));
            geo.cos_sub = abs(sum(r .* nn, 2));
            geo.ihat = k;  geo.rhat = r;  geo.nhat = nn;
            geo.Ein  = [a.Ex(m) a.Ey(m) a.Ez(m)];
            geo.Eout = [b.Ex(m) b.Ey(m) b.Ez(m)];
        end

        function t2 = powerT(testCase, iPre, iFace)
            % Engine power transmittance |t|^2, with a uniformity check: a
            % collimated beam on a flat surface must give ONE number over the
            % whole pupil, so a mean that hides scatter would be a bug in
            % itself.
            [amp, ~] = testCase.ampRatio(iPre, iFace);
            testCase.verifyLessThan(max(amp) - min(amp), 1e-12, ...
                'transmitted amplitude is not uniform over the pupil');
            t2 = median(amp).^2;
        end

        function setMgF2(testCase, iElt)
            macos.coating(iElt, 'index', testCase.nMgF2, 'extinc', 0, ...
                                'thickness', testCase.dMgF2);
        end
    end

    methods (Test)

        % =================================================================
        % A. The incumbent convention, established from the textbook.
        %    Everything after this compares the coated branch to the
        %    uncoated one; this is what says the uncoated one is right.
        % =================================================================

        function test_uncoated_transmission_is_the_power_transmittance(testCase)
            % |TP|^2 and |TS|^2 out of the UNCOATED branch are the textbook
            % POWER transmittances -- not the bare Fresnel field coefficients.
            % At normal incidence T = 4*n1*n2/(n1+n2)^2 = 0.96 for 1 -> 1.5,
            % so the amplitude is sqrt(0.96) = 0.97980, whereas the field
            % coefficient t = 2n1/(n1+n2) = 0.8.  The gap between those two
            % numbers IS the convention, and it is what the coated branch was
            % missing.
            macos.load_rx(rx_fixture_path(testCase.RxName));
            testCase.polOn(1, 0);
            T = testCase.powerT(testCase.Pre, testCase.Face1);
            Ta = multilayer_T(testCase.nAir, [], [], testCase.nGlass, ...
                              testCase.lambda0, 0, 's');
            testCase.verifyEqual(T, Ta, 'RelTol', 1e-12);
            testCase.verifyEqual(Ta, 0.96, 'AbsTol', 1e-14);   % sanity on the analytic
        end

        function test_uncoated_transmission_oblique_s_and_p(testCase)
            % Same claim at 45 deg, where s and p separate: T_s = 0.89999,
            % T_p = 0.99153 for a bare 1 -> 1.5 interface.  A gate that ran
            % only at normal incidence could not tell the two apart.
            th = pi/4;
            for pol = {'p', 's'}
                macos.load_rx(rx_fixture_path(testCase.Rx45));
                if strcmp(pol{1}, 'p'), testCase.polOn(1, 0);   % x^ = p_i^
                else,                   testCase.polOn(0, 1);   % y^ = s^
                end
                T  = testCase.powerT(testCase.Pre45, testCase.Face45);
                Ta = multilayer_T(testCase.nAir, [], [], testCase.nGlass, ...
                                  testCase.lambda0, th, pol{1});
                testCase.verifyEqual(T, Ta, 'RelTol', 1e-12, ...
                    sprintf('uncoated %s at 45 deg', pol{1}));
            end
            % non-vacuity: the two polarizations really are different here
            Ts = multilayer_T(testCase.nAir, [], [], testCase.nGlass, ...
                              testCase.lambda0, th, 's');
            Tp = multilayer_T(testCase.nAir, [], [], testCase.nGlass, ...
                              testCase.lambda0, th, 'p');
            testCase.verifyGreaterThan(abs(Tp - Ts), 0.08);   % actually 0.0835
        end

        function test_fixture_45_is_pure_p_and_pure_s(testCase)
            % The 45 deg fixture claims an x-polarized source is PURE p and a
            % y-polarized source PURE s.  Check it from the engine's own ray
            % directions and surface normal, so no gate below rests on the
            % fixture's declared psiElt.
            macos.load_rx(rx_fixture_path(testCase.Rx45));
            testCase.polOn(1, 0);
            [~, g] = testCase.ampRatio(testCase.Pre45, testCase.Face45);
            [sh, pih, ~] = spFrames(g);
            Ein  = g.Ein;
            fs = abs(sum(Ein .* sh,  2));
            fp = abs(sum(Ein .* pih, 2));
            testCase.verifyLessThan(max(fs ./ fp), 1e-12, ...
                'x-polarized input is not pure p on this fixture');

            testCase.polOn(0, 1);
            [~, g] = testCase.ampRatio(testCase.Pre45, testCase.Face45);
            [sh, pih, ~] = spFrames(g);
            fs = abs(sum(g.Ein .* sh,  2));
            fp = abs(sum(g.Ein .* pih, 2));
            testCase.verifyLessThan(max(fp ./ fs), 1e-12, ...
                'y-polarized input is not pure s on this fixture');
        end

        % =================================================================
        % B. Gate 1 -- an index-matched layer IS a bare interface.
        %    The self-contained statement of the bug: a coating whose index
        %    equals the substrate's is optically nothing at all, so the
        %    coated branch must reproduce the uncoated branch exactly.
        % =================================================================

        function test_index_matched_layer_equals_bare_interface_normal(testCase)
            % PRE-FIX: 0.8164965809 = 1/sqrt(1.5).
            macos.load_rx(rx_fixture_path(testCase.RxName));
            testCase.polOn(1, 0);
            Tu = testCase.powerT(testCase.Pre, testCase.Face1);
            macos.coating(testCase.Face1, 'index', testCase.nGlass, ...
                          'extinc', 0, 'thickness', 1.0e-7);
            Tc = testCase.powerT(testCase.Pre, testCase.Face1);
            testCase.verifyEqual(sqrt(Tc/Tu), 1.0, 'AbsTol', 1e-13);
        end

        function test_index_matched_layer_equals_bare_interface_oblique(testCase)
            % The cosine half of the factor.  PRE-FIX both polarizations read
            % 0.7311104457 = 1/sqrt(1.5*cos(28.1255deg)/cos(45deg)) -- the
            % same number for s and p, which is itself the signature of a
            % missing COMMON scalar rather than a Fresnel error.
            for pol = {'p', 's'}
                macos.load_rx(rx_fixture_path(testCase.Rx45));
                if strcmp(pol{1}, 'p'), testCase.polOn(1, 0);
                else,                   testCase.polOn(0, 1);
                end
                Tu = testCase.powerT(testCase.Pre45, testCase.Face45);
                macos.coating(testCase.Face45, 'index', testCase.nGlass, ...
                              'extinc', 0, 'thickness', 1.0e-7);
                Tc = testCase.powerT(testCase.Pre45, testCase.Face45);
                testCase.verifyEqual(sqrt(Tc/Tu), 1.0, 'AbsTol', 1e-13, ...
                    sprintf('index-matched %s at 45 deg', pol{1}));
            end
        end

        function test_index_matched_layer_at_the_detector_plane(testCase)
            % BOTH DISPATCH CHAINS.  propsub's CPROPAGATE re-traces the seed
            % rays through its own EltID chain, so a transmittance can be
            % right at ray level and wrong in the image -- the Phase-3
            % polarizer finding.  PRE-FIX this ratio was 0.6666666667 = 1/1.5
            % in INTENSITY, i.e. the grid under-reported flux by a third.
            macos.load_rx(rx_fixture_path(testCase.RxName));
            testCase.polOn(1, 0);
            macos.trace(testCase.Det);
            Iu = sum(macos.intensity(testCase.Det), 'all');
            macos.coating(testCase.Face1, 'index', testCase.nGlass, ...
                          'extinc', 0, 'thickness', 1.0e-7);
            macos.trace(testCase.Det);
            Ic = sum(macos.intensity(testCase.Det), 'all');
            testCase.verifyEqual(Ic/Iu, 1.0, 'AbsTol', 1e-12);
        end

        % =================================================================
        % C. Gate 2 -- a real coating against the textbook multilayer T.
        % =================================================================

        function test_mgf2_quarterwave_normal_incidence(testCase)
            % Single-layer MgF2 AR on glass, quarter-wave at the fixture
            % wavelength.  T rises from the bare-interface 0.96 to 0.9859;
            % the gate is against the characteristic-matrix T, and the
            % improvement over bare glass is asserted so the test cannot be
            % satisfied by a stack that silently did nothing.
            macos.load_rx(rx_fixture_path(testCase.RxName));
            testCase.polOn(1, 0);
            testCase.setMgF2(testCase.Face1);
            T  = testCase.powerT(testCase.Pre, testCase.Face1);
            Ta = multilayer_T(testCase.nAir, testCase.nMgF2, testCase.dMgF2, ...
                              testCase.nGlass, testCase.lambda0, 0, 's');
            testCase.verifyEqual(T, Ta, 'RelTol', 1e-12);
            testCase.verifyGreaterThan(T, 0.98);      % AR coating does its job
        end

        function test_mgf2_quarterwave_45deg_s_and_p(testCase)
            % The oblique multilayer T, s and p separately.  This is the test
            % that exercises the full factor: n_sub AND cos_sub/cos_inc, on
            % top of a genuine interference stack.
            th = pi/4;
            for pol = {'p', 's'}
                macos.load_rx(rx_fixture_path(testCase.Rx45));
                if strcmp(pol{1}, 'p'), testCase.polOn(1, 0);
                else,                   testCase.polOn(0, 1);
                end
                testCase.setMgF2(testCase.Face45);
                T  = testCase.powerT(testCase.Pre45, testCase.Face45);
                Ta = multilayer_T(testCase.nAir, testCase.nMgF2, ...
                                  testCase.dMgF2, testCase.nGlass, ...
                                  testCase.lambda0, th, pol{1});
                testCase.verifyEqual(T, Ta, 'RelTol', 1e-12, ...
                    sprintf('MgF2 %s at 45 deg', pol{1}));
            end
        end

        % =================================================================
        % D. Gate 3 -- air-to-air closure, and the "common scalar" claim.
        % =================================================================

        function test_air_to_air_power_closure_mixed_plate(testCase)
            % THE DISCRIMINATING CLOSURE.  Coat only the FRONT face and leave
            % the back one bare, then measure air-to-air: the total must be
            % T1*T2, the product of the two textbook single-face
            % transmittances (the engine models the faces independently -- no
            % plate etalon -- which is what T1*T2 assumes).
            %
            % Why one face and not two: with BOTH faces coated the two
            % radiometric factors are sqrt(n_g*c_g/(1*c_i)) and
            % sqrt(1*c_o/(n_g*c_g)), whose product is sqrt(c_o/c_i) = 1, so a
            % both-coated plate is air-to-air INVARIANT under this landing and
            % cannot see the defect at all (verified: it passes against the
            % pre-fix engine).  Mixing a coated face with an uncoated one
            % breaks the cancellation -- the coated face's missing factor no
            % longer has a partner -- and this test fails pre-fix by 1/sqrt(1.5)
            % in amplitude.  The invariance itself is gated separately below.
            macos.load_rx(rx_fixture_path(testCase.RxName));
            testCase.polOn(1, 0);
            testCase.setMgF2(testCase.Face1);
            Ttot = testCase.powerT(testCase.Pre, testCase.Face2);

            T1 = multilayer_T(testCase.nAir, testCase.nMgF2, testCase.dMgF2, ...
                              testCase.nGlass, testCase.lambda0, 0, 's');
            T2 = multilayer_T(testCase.nGlass, [], [], testCase.nAir, ...
                              testCase.lambda0, 0, 's');
            testCase.verifyEqual(Ttot, T1*T2, 'RelTol', 1e-12);
        end

        function test_air_to_air_factors_telescope(testCase)
            % The self-consistency that makes the power-amplitude convention
            % the right one to keep (decision grounds #3), stated as a test
            % rather than an argument: for a parallel plate the two
            % radiometric factors multiply to sqrt(cos_out/cos_in) = 1, so the
            % air-to-air AMPLITUDE through a fully coated plate equals the
            % bare product of the two composed Fresnel FIELD coefficients,
            % with no radiometric factor left over anywhere.
            %
            % NOT A DEFECT DETECTOR -- it holds identically before and after
            % the landing, precisely because of that cancellation.  It is here
            % to pin that the factors compose correctly, and it is labelled so
            % nobody later mistakes its green for coverage of the fix.
            macos.load_rx(rx_fixture_path(testCase.RxName));
            testCase.polOn(1, 0);
            testCase.setMgF2(testCase.Face1);
            testCase.setMgF2(testCase.Face2);
            [amp, ~] = testCase.ampRatio(testCase.Pre, testCase.Face2);

            t1 = multilayer_t(testCase.nAir, testCase.nMgF2, testCase.dMgF2, ...
                              testCase.nGlass, testCase.lambda0, 0, 's');
            t2 = multilayer_t(testCase.nGlass, testCase.nMgF2, testCase.dMgF2, ...
                              testCase.nAir, testCase.lambda0, 0, 's');
            testCase.verifyEqual(median(amp), abs(t1*t2), 'RelTol', 1e-12);

            % and the closure it implies, against the textbook powers
            T1 = multilayer_T(testCase.nAir, testCase.nMgF2, testCase.dMgF2, ...
                              testCase.nGlass, testCase.lambda0, 0, 's');
            T2 = multilayer_T(testCase.nGlass, testCase.nMgF2, testCase.dMgF2, ...
                              testCase.nAir, testCase.lambda0, 0, 's');
            testCase.verifyEqual(median(amp).^2, T1*T2, 'RelTol', 1e-12);
        end

        function test_scalar_factor_leaves_the_polarization_state_alone(testCase)
            % WHAT THE LANDING DID NOT CHANGE.  The fix is a COMMON REAL
            % scalar on TP and TS, so it cancels identically in the ratio
            % t_p/t_s: the transmitted polarization STATE -- and with it
            % every diattenuation and retardance quantity a Jones-pupil
            % analysis reads off a coated refractor -- is untouched.  Only
            % the absolute transmitted amplitude moved.  That is the claim
            % that lets the existing Phase-2/2b coated results stand, so it
            % is gated rather than asserted.
            %
            % Reference value is the ratio of the two composed Fresnel FIELD
            % coefficients, computed with NO radiometric factor at all --
            % which is precisely the point: the engine must land on the
            % factor-free ratio.
            macos.load_rx(rx_fixture_path(testCase.Rx45));
            testCase.polOn(1/sqrt(2), 1/sqrt(2));   % 45 deg linear: p and s equally
            testCase.setMgF2(testCase.Face45);
            [~, g] = testCase.ampRatio(testCase.Pre45, testCase.Face45);
            [sh, pih, prh] = spFrames(g);

            Epi = sum(g.Ein  .* pih, 2);   Esi = sum(g.Ein  .* sh, 2);
            Epr = sum(g.Eout .* prh, 2);   Esr = sum(g.Eout .* sh, 2);
            rat = abs((Epr ./ Epi) ./ (Esr ./ Esi));

            t_p = multilayer_t(testCase.nAir, testCase.nMgF2, testCase.dMgF2, ...
                               testCase.nGlass, testCase.lambda0, pi/4, 'p');
            t_s = multilayer_t(testCase.nAir, testCase.nMgF2, testCase.dMgF2, ...
                               testCase.nGlass, testCase.lambda0, pi/4, 's');
            testCase.verifyEqual(median(rat), abs(t_p/t_s), 'RelTol', 1e-11);
            % Non-vacuous: p and s are genuinely different through this
            % stack at 45 deg (|t_p/t_s| = 1.0213), so the assertion above
            % is a 2%-away number matched to 1e-11, not "the ratio is 1".
            testCase.verifyGreaterThan(abs(abs(t_p/t_s) - 1), 0.02);
        end

        % =================================================================
        % E. Gate 4 -- the factor lives inside ifPol.
        % =================================================================

        function test_pol_off_is_untouched_by_the_coating(testCase)
            % The whole coated branch, radiometric factor included, is gated
            % on ifPol.  With polarization OFF a coating must be inert: the
            % OPD and the detector intensity have to be BIT-identical between
            % a coated and an uncoated run.  verifyEqual with no tolerance is
            % deliberate -- this is an identity, not an agreement.
            macos.load_rx(rx_fixture_path(testCase.RxName));
            macos.polarization('off');
            macos.trace(testCase.Det);
            Iu = macos.intensity(testCase.Det);
            Wu = macos.opd();

            testCase.setMgF2(testCase.Face1);
            testCase.setMgF2(testCase.Face2);
            macos.trace(testCase.Det);
            Ic = macos.intensity(testCase.Det);
            Wc = macos.opd();

            testCase.verifyEqual(Ic, Iu);
            testCase.verifyEqual(Wc, Wu);
        end

        % =================================================================
        % F. Gate 5 -- a wavelength sweep: the scalar factor cannot have
        %    created, moved, or flattened the interference structure.
        % =================================================================

        function test_quarterwave_structure_survives_the_scalar_factor(testCase)
            % Sweep lambda across the quarter-wave design point with the
            % PHYSICAL stack fixed (coat_set stores physical thickness, so
            % the sweep is genuinely chromatic).  Three claims:
            %   1. T(lambda) tracks the characteristic-matrix T at every
            %      point -- the factor did not distort the spectrum;
            %   2. the ratio T_engine/T_textbook is CONSTANT to round-off,
            %      i.e. whatever the factor is, it is wavelength-independent
            %      as a real scalar must be;
            %   3. the structure is real -- T peaks at the design wavelength
            %      and falls away by a stated margin, so claims 1 and 2
            %      cannot be satisfied by a flat curve.
            lam = [0.6 0.8 1.0 1.2 1.5] * 1e-6;
            macos.load_rx(rx_fixture_path(testCase.RxName));
            testCase.polOn(1, 0);
            testCase.setMgF2(testCase.Face1);

            T = zeros(size(lam));  Ta = zeros(size(lam));
            for i = 1:numel(lam)
                macos.set_src_wvl(lam(i));
                T(i)  = testCase.powerT(testCase.Pre, testCase.Face1);
                Ta(i) = multilayer_T(testCase.nAir, testCase.nMgF2, ...
                                     testCase.dMgF2, testCase.nGlass, ...
                                     lam(i), 0, 's');
            end
            macos.set_src_wvl(testCase.lambda0);

            testCase.verifyEqual(T, Ta, 'RelTol', 1e-11);
            r = T ./ Ta;
            testCase.verifyLessThan(max(r) - min(r), 1e-11, ...
                'engine/analytic ratio drifts with wavelength');

            % structure: the design point is the maximum, and the sweep has
            % real contrast (bare glass would be a flat 0.96).
            [~, imax] = max(Ta);
            testCase.verifyEqual(lam(imax), testCase.lambda0, 'AbsTol', 1e-15);
            testCase.verifyGreaterThan(max(Ta) - min(Ta), 0.01);
        end

    end
end

% =====================================================================
% Textbook analytics -- Abeles characteristic matrix.
% Macleod, "Thin-Film Optical Filters", ch. 2 (equivalently Born & Wolf
% 1.6.4).  Written from the tilted-admittance definitions, NOT from
% elemsub.F: an analytic transcribed from the engine is circular in
% exactly the quantity it is supposed to check
% (REVIEW_POL_SP_SIGN_2026-07-27.md).
%
%   tilted admittance   eta = N*cos(theta)   (s)   or   N/cos(theta)  (p)
%   layer phase         delta = 2*pi*N*d*cos(theta)/lambda
%   layer matrix        [cos d, i sin d/eta; i eta sin d, cos d]
%   stack               [B;C] = prod(M_j, OUTERMOST first) * [1; eta_sub]
%   power transmittance T = 4*eta_0*Re(eta_sub)/|eta_0*B + C|^2
%
% With zero layers the product is the identity and T collapses to the bare
% Fresnel interface transmittance, so the same routine serves section A.
% =====================================================================
function T = multilayer_T(n0, nL, dL, nsub, lambda, theta0, pol)
[B, C, eta0, etas] = charmat(n0, nL, dL, nsub, lambda, theta0, pol);
T = 4*eta0*real(etas) / abs(eta0*B + C)^2;
end

function t = multilayer_t(n0, nL, dL, nsub, lambda, theta0, pol)
% The composed FIELD amplitude coefficient of the same stack, in the
% ORDINARY Fresnel sense: t = |E_sub| / |E_inc|, each measured along its own
% p^ (or s^).  No radiometric factor -- used where the point is that the
% factor cancels.
%
% A REAL SUBTLETY, worth stating because it bit this file once.  Macleod's
% 2*eta_0/(eta_0*B + C) is NOT that coefficient for p-polarization.  B and C
% are the normalized TANGENTIAL fields, and for p the tangential component
% is E*cos(theta), so Macleod's t is
%       t_tangential = t_Fresnel * cos(theta_sub)/cos(theta_inc),
% which at 45 deg into n=1.5 is a factor 1.2472 -- large, and exactly the
% size of a plausible radiometric-factor error, so it must not be waved off.
% (T is unaffected either way, which is why every transmittance gate above
% passed while this one did not.)  Convert back for p; s needs nothing since
% the s field is already tangential.
[B, C, eta0, ~] = charmat(n0, nL, dL, nsub, lambda, theta0, pol);
t = 2*eta0 / (eta0*B + C);
if strcmp(pol, 'p')
    sin0 = n0*sin(theta0);
    t = t * sqrt(1 - (sin0/n0)^2) / sqrt(1 - (sin0/nsub)^2);
end
end

function [B, C, eta0, etas] = charmat(n0, nL, dL, nsub, lambda, theta0, pol)
nL = nL(:).';  dL = dL(:).';
sin0 = n0*sin(theta0);                       % Snell invariant
eta  = @(n) admittance(n, sin0, pol);
eta0 = eta(n0);  etas = eta(nsub);
M = eye(2);
for j = 1:numel(nL)                          % OUTERMOST layer first
    cj  = sqrt(1 - (sin0/nL(j))^2);
    dlt = 2*pi*nL(j)*dL(j)*cj/lambda;
    ej  = eta(nL(j));
    M   = M * [cos(dlt), 1i*sin(dlt)/ej; 1i*ej*sin(dlt), cos(dlt)];
end
BC = M * [1; etas];
B  = BC(1);  C = BC(2);
end

function e = admittance(n, sin0, pol)
c = sqrt(1 - (sin0/n)^2);
if strcmp(pol, 's'), e = n*c; else, e = n/c; end
end

% ---------------------------------------------------------------------
function [sh, pih, prh] = spFrames(g)
% s^ / incident p^ / reflected-or-refracted p^ per ray, built from the
% ENGINE's ray directions and surface normal (g.ihat, g.rhat, g.nhat) --
% the same construction elemsub.F uses, but assembled here from geometry
% the test read back, so no fixture number is taken on trust.
sh  = cross(g.ihat, g.nhat, 2);
sh  = sh ./ vecnorm(sh, 2, 2);
pih = cross(sh, g.ihat, 2);
prh = cross(sh, g.rhat, 2);
end
