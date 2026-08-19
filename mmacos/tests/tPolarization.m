classdef tPolarization < matlab.unittest.TestCase
%TPOLARIZATION  Phase-1 polarization exposure (PLAN_POLARIZATION).
%   Exercises the engine polarization physics newly exposed through the
%   bindings: macos.polarization (pol_set/pol_get), macos.vector_diffraction
%   (vecdif_set), macos.coating (coat_set/coat_get, Model A), and
%   macos.ray_field (rayfield_get).  These are the state/round-trip/geometry
%   gates; the Jones-pupil physics lands in Phase 2.

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
        Det       = 6        % FocalPlane
        Fold      = 3        % a reflector to coat
        Stop      = 1        % SecMirObs (Obscuring) -- incident ray dirs
        Mirror1   = 2        % Primary: the ODD-mirror (single-reflection) state
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

    methods (Test)
        % ---- polarization on/off + source state round-trip -----------
        function test_pol_on_off_roundtrip(testCase)
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
            s = macos.polarization();
            testCase.verifyTrue(s.on);
            testCase.verifyEqual(s.Ex, complex(1,0), 'AbsTol', 1e-12);
            testCase.verifyEqual(s.Ey, complex(0,0), 'AbsTol', 1e-12);

            macos.polarization('off');
            s = macos.polarization();
            testCase.verifyFalse(s.on);
        end

        function test_pol_circular_state(testCase)
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 1]);
            s = macos.polarization();
            testCase.verifyEqual(s.Ey, complex(0,1), 'AbsTol', 1e-12);
        end

        function test_pol_enables_vector(testCase)
            % ModelSize 128 has mWF=3, so POLARIZATION enables vector diff.
            macos.polarization('on');
            s = macos.polarization();
            testCase.verifyTrue(s.vector);
        end

        % ---- vector / scalar toggle + ordering guard -----------------
        function test_vecdif_toggle(testCase)
            macos.polarization('on');
            macos.vector_diffraction(false);
            s = macos.polarization();
            testCase.verifyFalse(s.vector);
            macos.vector_diffraction(true);
            s = macos.polarization();
            testCase.verifyTrue(s.vector);
        end

        function test_vector_requires_polarization(testCase)
            macos.polarization('off');
            % VECTOR with polarization off must error (not silently revert).
            % mexErrMsgTxt throws without a stable identifier, so just
            % assert that an error is raised.
            threw = false;
            try
                macos.vector_diffraction(true);
            catch
                threw = true;
            end
            testCase.verifyTrue(threw, ...
                'vector_diffraction(true) should error when polarization is off');
        end

        % ---- coating set/get round-trip (Model A) --------------------
        function test_coat_roundtrip_identity(testCase)
            n = [1.38 2.30];
            k = [0.0  0.10];
            t = [1.0e-7 5.0e-8];   % physical thickness (BaseUnits = m)
            macos.coating(testCase.Fold, 'index', n, 'extinc', k, ...
                          'thickness', t);
            s = macos.coating(testCase.Fold);
            testCase.verifyEqual(s.n_layer, 2);
            testCase.verifyEqual(s.index,     n, 'AbsTol', 1e-12);
            testCase.verifyEqual(s.extinc,    k, 'AbsTol', 1e-12);
            testCase.verifyEqual(s.thickness, t, 'RelTol', 1e-9);
        end

        function test_coat_query_uncoated(testCase)
            s = macos.coating(testCase.Det);   % focal plane, no coating
            testCase.verifyEqual(s.n_layer, 0);
        end

        function test_coat_bad_count_errors(testCase)
            testCase.verifyError(@() macos.coating(testCase.Fold, ...
                'index', zeros(1,20), 'extinc', zeros(1,20), ...
                'thickness', zeros(1,20)), 'macos:coating:badCount');
        end

        % ---- ray_field structure + status mask -----------------------
        function test_ray_field_shape_and_status(testCase)
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
            macos.trace(testCase.Det);
            rf = macos.ray_field(testCase.Det);
            N = testCase.ModelSize;
            testCase.verifySize(rf.Ex, [N N]);
            testCase.verifyTrue(isequal(size(rf.status), [N N]));
            % At least some rays are OK (status 0) after a normal trace.
            testCase.verifyGreaterThan(nnz(rf.status == 0), 0);
            % OK rays carry a non-zero field for an x-polarized source.
            okmask = rf.status == 0;
            testCase.verifyGreaterThan(max(abs(rf.Ex(okmask))), 0);
        end

        % ---- ODD-MIRROR gate (r_p sign, 2026-07-27) -------------------
        %   Every pre-existing polarization gate was structurally blind to
        %   the reflected-p-hat / Fresnel-r_p sign conflict fixed in macos
        %   cb29ea5: the defect reflects the transverse field about the
        %   local p-hat instead of negating it, which is an INVOLUTION
        %   (cancels exactly across a mirror PAIR -- and Rx_Cass_FarField
        %   has exactly two mirrors) and is UNITARY (invisible to the
        %   unitarity gate).  These two tests read the field after ONE
        %   mirror, which is where it shows.
        %   See macos/REVIEW_POL_SP_SIGN_2026-07-27.md and the reproducer
        %   mmacos/tools/pol_sp_sign_probe/.

        function test_odd_mirror_crosspol_pec_analytic(testCase)
            % A perfect conductor (the Rx's IndRef=1/Extinc=1e22 idiom)
            % has r_s = -1, r_p = +1 in the engine's ray-following p-hat
            % basis, so ONE reflection of an x-polarized collimated beam
            % is fixed by geometry alone.  Writing k_i = zhat, AOI = a,
            % pupil azimuth = phi (measured about the axis, from +x):
            %     s-hat  = (sin phi, -cos phi, 0)
            %     p-hat_i = s x k_i,   p-hat_r = s x k_r
            %     E_out = (E.p_i) p_r - (E.s) s
            % which gives, EXACTLY (no small-angle expansion),
            %     Ex/E0 = -(cos 2a cos^2 phi + sin^2 phi) == -den
            %     Ey/Ex = -sin(2 phi) sin^2(a) / den
            %     Ez/Ex = -sin(2a) cos(phi)   / den
            % with den = 1 - 2 sin^2(a) cos^2(phi).  This is textbook
            % Born & Wolf + geometry -- NOT transcribed from the engine's
            % own expression, so unlike the pre-2026-07-27 fold gate it is
            % not circular in the sign it checks.
            %
            % BOTH a and phi are taken from the engine's RAY DIRECTIONS
            % (a from the deflection between the stop and the mirror; phi
            % from the outgoing transverse direction), so the test makes
            % no assumption about the pixel-grid-to-pupil mapping.  Note
            % phi from the outgoing (inward-pointing) direction is the
            % pupil azimuth + pi, which flips cos(phi) -- hence the +
            % sign used for Ez below.
            %
            % NON-VACUITY (measured 2026-07-27 by rebuilding the engine
            % with the sign flipped back, model 128, same fixture): the
            % relative error against this closed form goes to median
            % 1.14e+02 / max 1.61e+05, i.e. 13 orders over the 1e-11
            % asserted below, because the pre-fix engine returns
            % |Ey/Ex| ~ 1.0 where the closed form predicts 1.8e-3 to
            % 3.4e-2.  The retardance assertion fails too (3.9e-10).
            [aoi, phi, rEy, rEz, ok] = testCase.oneMirrorField();

            den   = 1 - 2*sin(aoi).^2 .* cos(phi).^2;
            predY = -sin(aoi).^2 .* sin(2*phi) ./ den;
            predZ =  sin(2*aoi)  .* cos(phi)   ./ den;

            % Restrict to rays where the predicted component is not near
            % its own zero (the ratio is ill-conditioned there), and skip
            % the near-axis rays where both sides vanish.
            sy = ok & abs(sin(2*phi)) > 0.2 & aoi > deg2rad(1);
            sz = ok & abs(cos(phi))   > 0.2 & aoi > deg2rad(1);
            testCase.verifyGreaterThan(nnz(sy), 2000);

            testCase.verifyLessThan( ...
                max(abs((rEy(sy) - predY(sy)) ./ predY(sy))), 1e-11, ...
                'single-mirror cross-pol vs the PEC closed form');
            testCase.verifyLessThan( ...
                max(abs((rEz(sz) - predZ(sz)) ./ predZ(sz))), 1e-11, ...
                'single-mirror longitudinal component vs the PEC closed form');

            % The field must stay strictly real for a perfect conductor
            % (r_s, r_p real): no spurious retardance.
            testCase.verifyLessThan(max(abs(imag(rEy(ok)))), 1e-14, ...
                'perfect conductor must introduce no retardance');
        end

        function test_odd_mirror_crosspol_rho2_law(testCase)
            % The fixture-free half of the same claim, and the one that
            % needs no reference value at all: cross-polarization from an
            % isotropic rotationally symmetric mirror is slope-driven, so
            % it must VANISH ON AXIS and grow as rho^2, staying bounded by
            % O(sin^2 AOI).  Pre-fix it was FLAT in radius at ~1.0
            % (measured 1.014/1.016/1.010/1.005 at rho = 32/64/96/128 px,
            % model 256; 0.988/1.038/1.029/1.035 at model 128, log-log
            % slope 0.033 against the 1.7 asserted below) -- which is the
            % tell that needs no fixture and no reference value.
            [aoi, ~, rEy, ~, ok] = testCase.oneMirrorField();
            ratio = abs(rEy);

            % (a) the O(sin^2 beta) physical bound, beta = local AOI.
            testCase.verifyLessThan(max(ratio(ok)), ...
                1.05 * max(sin(aoi(ok)).^2), ...
                'cross-pol must stay within the O(sin^2 AOI) bound');

            % (b) cross-polarized power fraction after ONE mirror (|Ex|
            %     is uniform to 6% here, so the mean squared ratio is the
            %     probe's Py/Px to the same accuracy).  Pre-fix that
            %     number was 1.0163 -- a 50/50 mixture at <11 deg AOI.
            %     Post-fix: 2.09e-4.
            Py_Px = mean(ratio(ok).^2);
            testCase.verifyLessThan(Py_Px, 1e-3, ...
                'one-mirror cross-polarized power fraction');

            % (c) the rho^2 law, binned on the pixel grid exactly as the
            %     probe's radial table does (rho enters only as a binning
            %     coordinate, so a transposed grid cannot fake it).
            N = testCase.ModelSize;  c = (N + 1)/2;
            [jj, ii] = meshgrid(1:N, 1:N);
            rho  = hypot(ii - c, jj - c);
            rmax = max(rho(ok));
            frac = [0.25 0.5 0.75 1.0];
            med  = nan(size(frac));
            for t = 1:numel(frac)
                sel = ok & abs(rho - frac(t)*rmax) < 2;
                testCase.verifyGreaterThan(nnz(sel), 100);
                med(t) = median(ratio(sel));
            end
            testCase.verifyTrue(all(diff(med) > 0), ...
                'cross-pol must grow monotonically with pupil radius');
            slope = polyfit(log(frac(:)), log(med(:)), 1);
            testCase.verifyGreaterThan(slope(1), 1.7, ...
                'radial power law must be ~rho^2 (pre-fix it was flat)');
            testCase.verifyLessThan(slope(1), 2.3, ...
                'radial power law must be ~rho^2');
        end

        % ---- physics: a coating changes the polarized throughput -----
        function test_coating_changes_polarized_intensity(testCase)
            % Baseline polarized PSF (no coating).
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
            I0 = macos.intensity(testCase.Det);

            % Add an absorbing metal coating on the fold and re-trace.
            macos.coating(testCase.Fold, 'index', 1.2, 'extinc', 7.0, ...
                          'thickness', 0.1);
            I1 = macos.intensity(testCase.Det);

            % The coating must change the focal-plane intensity (proof the
            % coating reaches the polarized trace, not just storage).
            rel = norm(I1(:) - I0(:)) / max(norm(I0(:)), eps);
            testCase.verifyGreaterThan(rel, 1e-6);
        end
    end

    methods (Access = private)
        function [aoi, phi, rEy, rEz, ok] = oneMirrorField(testCase)
        %ONEMIRRORFIELD  Ray-side state after exactly ONE reflection.
        %   Traces to the stop (incident directions) and then to the
        %   Primary (post-reflection field + directions).  Two traces are
        %   required: RayE/RayDir are the CURRENT trace state, not a
        %   per-element history, so ray_field(e) must follow trace(e).
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
            macos.trace(testCase.Stop);
            r0 = macos.ray_field(testCase.Stop);
            macos.trace(testCase.Mirror1);
            r1 = macos.ray_field(testCase.Mirror1);

            ok = (r0.status == 0) & (r1.status == 0);

            % AOI from the deflection: the angle between the incident and
            % reflected directions is pi - 2*AOI for a mirror.
            kdot = r0.kx.*r1.kx + r0.ky.*r1.ky + r0.kz.*r1.kz;
            aoi  = (pi - acos(min(max(kdot, -1), 1))) / 2;
            % Azimuth from the outgoing transverse direction (the ray
            % turns toward the axis, so this is the pupil azimuth + pi).
            phi  = atan2(r1.ky, r1.kx);

            rEy = r1.Ey ./ r1.Ex;
            rEz = r1.Ez ./ r1.Ex;
        end
    end
end
