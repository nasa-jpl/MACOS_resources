classdef tPolExternal < matlab.unittest.TestCase
%TPOLEXTERNAL  External anchor for the thin-film polarization machinery.
%
%   Closes the worklist item Fable opened in REVIEW_POL_2C_2026-07-27.md:
%   every coated-mirror number the polarization work had reported was
%   MODEL-RELATIVE, gated only against our own analytics.  tJonesPupil's
%   Fresnel gate covers an optically THICK SINGLE LAYER (a bare interface);
%   tPolRadiometric covers the Abeles matrix in TRANSMISSION.  Neither
%   exercises a DIELECTRIC-ON-METAL stack, which is what a protected mirror
%   is and what the "151x" MgF2/Al claim rests on.
%
%   THE ANCHOR.  Reproduce a published protected-aluminum configuration in
%   the engine -- their indices, their film thickness, their wavelengths,
%   their incidence angles -- and compare curve-on-curve against their own
%   model.  Source, inputs and provenance: tools/pol_external_anchor/vh_data.m.
%
%     G. van Harten, F. Snik & C. U. Keller, PASP 121, 377 (2009),
%     doi:10.1086/599043; arXiv:0903.2740v1.
%
%   Driving the engine with THEIR inputs is the whole point: aluminum index
%   tables genuinely disagree with one another (the paper says so, and FITS
%   k rather than adopting a table), and a disagreement traceable to index
%   tables is NOT a machinery error.  This construction cannot confuse the
%   two.
%
%   TOLERANCE comes from the publication's own error bars: their Sec. 2
%   states an absolute accuracy of ~1% of Mueller element [1,1], i.e.
%   +-0.01 per normalized Mueller element.  The engine is required to sit
%   FAR inside that -- it reproduces their model to ~1e-14, because both
%   implement the same thin-film theory, so the meaningful assertion is
%   round-off agreement with a stated headroom to the measurement bar.
%
%   The analytic (vh_thinfilm) is written from Macleod ch. 2 and the
%   paper's stated equations, NEVER transcribed from elemsub.F -- an
%   "analytic" copied out of the engine is circular in exactly the
%   coefficient it should check, which is how the 2022 r_p sign defect
%   survived four years of gates.

    properties (Constant)
        ModelSize = 128
        NGrid     = 41
        % angles inside the publication's measured range (6-70 deg)
        AOI       = [6 20 45 70]
    end

    properties
        work
        d
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            here = fileparts(mfilename('fullpath'));
            addpath(fullfile(here, '..', 'tools', 'pol_external_anchor'));
            macos.init(testCase.ModelSize);
            testCase.d    = vh_data();
            testCase.work = [tempname '_vhgate'];
            mkdir(testCase.work);
        end
    end

    methods (TestClassTeardown)
        function teardownClass(testCase)
            if ~isempty(testCase.work) && isfolder(testCase.work)
                rmdir(testCase.work, 's');
            end
        end
    end

    methods
        function L = vhStack(testCase, iL)
            % the publication's physical stack, thicknesses in mm (the
            % Bench rig's BaseUnits)
            dd = testCase.d;
            L = [complex(dd.nf(iL), 0),               dd.d_oxide_nm * 1e-6; ...
                 complex(dd.n_al(iL), -dd.k_al(iL)),  dd.d_al_nm    * 1e-6];
        end
    end

    methods (Test)

        % ---- the p-hat convention, MEASURED not assumed ------------------
        function test_phat_convention_bridge_is_zero(testCase)
            % The Bench emits perfect-conductor mirrors, for which the
            % reflection is polarization-neutral: r_s/r_p is +1 in a FIXED
            % transverse frame and -1 in a ray-following one.  Which the
            % engine reports here IS the frame our measured ratio lives in,
            % and therefore the bridge that must (or must not) be applied
            % before comparing retardance with the publication.
            %
            % Measured 2026-07-28: exactly +1 (imaginary part exactly 0) at
            % every angle -> bridge = 0.  An earlier version of this
            % harness ASSUMED a pi bridge from the ray-following doctrine
            % and was wrong by exactly 180 deg everywhere; that is why this
            % is a measurement.
            for th = [2 10 45 80]
                m = vh_measure(testCase.work, 632.8e-6, th, [], testCase.NGrid);
                r = median(real(m.rho)) + 1i*median(imag(m.rho));
                testCase.verifyEqual(real(r), 1, 'AbsTol', 1e-12, ...
                    sprintf('PEC r_s/r_p real part at %g deg', th));
                testCase.verifyEqual(imag(r), 0, 'AbsTol', 1e-12, ...
                    sprintf('PEC r_s/r_p imag part at %g deg', th));
            end
        end

        % ---- the frame-free estimator validates itself -------------------
        function test_ratio_estimator_self_consistency(testCase)
            % r_s/r_p = M11/M22 = -M12/M21 with the unknown input-frame
            % rotation cancelling identically.  The two estimates agreeing
            % at round-off is what licenses the construction; it is a
            % built-in validity guard, not an assumption.
            L = testCase.vhStack(3);            % 600 nm
            for th = testCase.AOI
                m = vh_measure(testCase.work, 600e-6, th, L, testCase.NGrid);
                testCase.verifyLessThan(m.consistency, 1e-10, ...
                    sprintf('M11/M22 vs -M12/M21 at %g deg', th));
            end
        end

        % ---- the rig actually presents the AOI it is asked for -----------
        function test_fold_rig_aoi_is_what_was_requested(testCase)
            % Mirror DEVIATION is 180 - 2*AOI, not 2*AOI.  Getting that
            % backwards sweeps the COMPLEMENT of the intended angles and is
            % self-cancelling at exactly 45 deg -- the one angle the
            % pre-existing Fresnel gate runs at, so nothing else in the
            % suite would catch it.  This cost the anchor a full cycle.
            L = testCase.vhStack(3);
            for th = testCase.AOI
                m = vh_measure(testCase.work, 600e-6, th, L, testCase.NGrid);
                act = rad2deg(acos(median(m.cthi)));
                testCase.verifyEqual(act, th, 'AbsTol', 0.5, ...
                    sprintf('requested %g deg', th));
            end
        end

        % ---- (a) THE MACHINERY CHECK -------------------------------------
        function test_vanharten_machinery(testCase)
            % Engine vs publication, at the publication's own inputs, over
            % all four of their wavelengths and four angles inside their
            % measured range.  Both sides implement the same thin-film
            % theory, so agreement must be at round-off; the publication's
            % +-0.01 is the bar this has to sit far inside.
            dd = testCase.d;
            worstD = 0;  worstR = 0;
            for iL = 1:numel(dd.lambda_nm)
                lam_mm = dd.lambda_nm(iL) * 1e-6;
                L = testCase.vhStack(iL);
                for th = testCase.AOI
                    m = vh_measure(testCase.work, lam_mm, th, L, testCase.NGrid);
                    [rp, rs] = vh_thinfilm(L, complex(1.52,0), m.cthi, lam_mm);
                    Rp = abs(rp).^2;  Rs = abs(rs).^2;
                    Da = (Rs - Rp)./(Rs + Rp);
                    dla = angle(rp) - angle(rs);

                    rho = m.rho;
                    De  = (abs(rho).^2 - 1)./(abs(rho).^2 + 1);
                    dle = -angle(rho);      % bridge = 0, pinned above

                    % retardance deviation in NORMALIZED MUELLER units: the
                    % lower 2x2 block carries 2 sqrt(Rp Rs)/(Rp+Rs)*{cos,sin}
                    amp = 2*sqrt(Rp.*Rs)./(Rp + Rs);
                    worstD = max(worstD, max(abs(De - Da)));
                    worstR = max(worstR, max(abs(amp .* wrapPi(dle - dla))));
                end
            end
            % round-off agreement
            testCase.verifyLessThan(worstD, 1e-10, ...
                'diattenuation vs published model');
            testCase.verifyLessThan(worstR, 1e-10, ...
                'retardance vs published model');
            % and, stated explicitly, far inside the publication's own bar
            testCase.verifyLessThan(worstD, dd.mueller_accuracy);
            testCase.verifyLessThan(worstR, dd.mueller_accuracy);
        end

        % ---- non-vacuity 1: the oxide layer MATTERS ----------------------
        function test_omitting_the_oxide_exceeds_published_accuracy(testCase)
            % The publication's central claim is that a ~4 nm oxide must be
            % modelled to describe the mirror's polarization properties.
            % If the gate above could not tell a bare mirror from an
            % oxidized one, it would be measuring nothing.  Compare the
            % engine (WITH the 4.12 nm oxide) against an analytic WITHOUT
            % it: the disagreement must EXCEED their +-0.01, which
            % independently reproduces the paper's own conclusion.
            dd = testCase.d;
            worst = 0;
            for iL = 1:numel(dd.lambda_nm)
                lam_mm = dd.lambda_nm(iL) * 1e-6;
                L = testCase.vhStack(iL);
                bare = L(2,:);                       % drop the oxide row
                for th = testCase.AOI
                    m = vh_measure(testCase.work, lam_mm, th, L, testCase.NGrid);
                    [rp, rs] = vh_thinfilm(bare, complex(1.52,0), m.cthi, lam_mm);
                    Rp = abs(rp).^2;  Rs = abs(rs).^2;
                    dla = angle(rp) - angle(rs);
                    amp = 2*sqrt(Rp.*Rs)./(Rp + Rs);
                    dle = -angle(m.rho);
                    worst = max(worst, max(abs(amp .* wrapPi(dle - dla))));
                end
            end
            testCase.verifyGreaterThan(worst, dd.mueller_accuracy, ...
                'a 4 nm oxide must be distinguishable at their accuracy');
        end

        % ---- non-vacuity 2: the REJECTED thickness is distinguishable ----
        function test_rejected_50nm_oxide_is_excluded(testCase)
            % The paper's Sec. 1 notes that an earlier single-wavelength
            % study deduced ~50 nm of oxide -- an order of magnitude more
            % than tunneling-limited growth allows -- and their Sec. 4
            % attributes that to a SIGN error on the imaginary index.  Our
            % engine must be able to tell 4.12 nm from 50 nm at their
            % accuracy, or the anchor would not constrain the film at all.
            dd = testCase.d;
            iL = 3;                                   % 600 nm
            lam_mm = dd.lambda_nm(iL) * 1e-6;
            L = testCase.vhStack(iL);
            wrong = L;  wrong(1,2) = 50e-6;           % 50 nm oxide
            worst = 0;
            for th = testCase.AOI
                m = vh_measure(testCase.work, lam_mm, th, L, testCase.NGrid);
                [rp, rs] = vh_thinfilm(wrong, complex(1.52,0), m.cthi, lam_mm);
                Rp = abs(rp).^2;  Rs = abs(rs).^2;
                Da = (Rs - Rp)./(Rs + Rp);
                De = (abs(m.rho).^2 - 1)./(abs(m.rho).^2 + 1);
                worst = max(worst, max(abs(De - Da)));
            end
            testCase.verifyGreaterThan(worst, dd.mueller_accuracy, ...
                '4.12 nm vs 50 nm oxide must be separable');
        end

        % ---- the admittance assignment is pinned -------------------------
        function test_admittance_assignment_sign(testCase)
            % The paper's Eqs (5)-(6) print in an order PDF extraction
            % scrambles, so which of eta = N cos(theta) / N / cos(theta)
            % belongs to s was ambiguous.  It is settled by the SIGN of the
            % [1,2] Mueller element: for a metal R_s > R_p, so [1,2] = D is
            % POSITIVE, and their Fig. 1a plots it on a 0.00-0.15 axis.
            % Pinned here so the choice cannot silently rot -- and the
            % magnitude is checked against that axis too.
            dd = testCase.d;
            for iL = 1:numel(dd.lambda_nm)
                lam_mm = dd.lambda_nm(iL) * 1e-6;
                L = testCase.vhStack(iL);
                m = vh_measure(testCase.work, lam_mm, 70, L, testCase.NGrid);
                [rp, rs] = vh_thinfilm(L, complex(1.52,0), m.cthi, lam_mm);
                D = median((abs(rs).^2 - abs(rp).^2)./(abs(rs).^2 + abs(rp).^2));
                testCase.verifyGreaterThan(D, 0, ...
                    'metal diattenuation must be positive (R_s > R_p)');
                testCase.verifyLessThan(D, 0.15, ...
                    'within the published Fig. 1a [1,2] axis');
            end
        end

        % ---- (b) the context finding, pinned -----------------------------
        function test_overcoat_trade_reverses_with_optical_thickness(testCase)
            % CONTEXT, not a machinery gate -- but it pins the finding that
            % closed this worklist item.
            %
            % The scalar that drives cross-polarization in an on-axis train
            % is eps = (r_s - r_p)/(r_s + r_p).  Whether a dielectric
            % overcoat RAISES or LOWERS it is set by the film's OPTICAL
            % thickness AT THE OPERATING WAVELENGTH, and the sign reverses
            % across that -- a fact the phrase "a quarter wave" hides.
            %
            % The SAME 110 nm MgF2 film, on the same aluminium, at the same
            % few-degree incidence:
            %   at 632.8 nm  110 nm = 0.96 quarter-waves -> REDUCES eps
            %   at 1000  nm  110 nm = 0.61 quarter-waves -> RAISES  eps
            % Phase 2c applied this film on Rx_Cass_FarField, which runs at
            % Wavelen = 1.0E-06 m, and described it as "a quarter wave";
            % at 1 um a true quarter-wave is 181 nm and would have SUPPRESSED
            % the floor instead.  See macos/REVIEW_POL_EXTERNAL_2026-07-28.md
            % and polval section 8.
            bare = [complex(1.45,-7.54), 2.0e-4];
            mgf2 = [complex(1.38,0), 1.1e-4; complex(1.45,-7.54), 2.0e-4];
            aoi  = 6;

            function e = driver(lam_mm, L)
                m = vh_measure(testCase.work, lam_mm, aoi, L, testCase.NGrid);
                e = median(abs((m.rho - 1)./(m.rho + 1)));
            end

            % 632.8 nm: near a quarter wave -> the overcoat HELPS
            r633 = driver(632.8e-6, mgf2) / driver(632.8e-6, bare);
            testCase.verifyLessThan(r633, 0.6, ...
                'at 632.8 nm the 110 nm film must REDUCE the driver');

            % 1 um (the 2c fixture's wavelength): 0.61 QW -> the overcoat HURTS
            r1000 = driver(1000e-6, mgf2) / driver(1000e-6, bare);
            testCase.verifyGreaterThan(r1000, 2.0, ...
                'at 1 um the same film must RAISE the driver');

            % the reversal itself, stated as one number
            testCase.verifyGreaterThan(r1000/r633, 5, ...
                'the overcoat trade must reverse between the two wavelengths');

            % and a TRUE quarter wave at 1 um suppresses it hard
            qw   = 1000/(4*1.38) * 1e-6;      % 181.2 nm, in mm
            mgqw = [complex(1.38,0), qw; complex(1.45,-7.54), 2.0e-4];
            rqw  = driver(1000e-6, mgqw) / driver(1000e-6, bare);
            testCase.verifyLessThan(rqw, 0.2, ...
                'a true quarter-wave overcoat must SUPPRESS the driver');
        end
    end
end

function a = wrapPi(a)
    a = mod(a + pi, 2*pi) - pi;
end
