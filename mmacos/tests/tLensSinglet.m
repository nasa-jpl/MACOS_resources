classdef tLensSinglet < matlab.unittest.TestCase
%TLENSSINGLET  The singlet template (templates/35_lenses/singlet) and the lens deck tools.
%
%   Legs
%     iodc link    macos.design.lens_emit + lens_eval score the IODC 2010 gate
%                  pair (the symmetric N-BK7 pair, stop mid-gap) at 0 and 10 deg
%                  to the SAME RMS and distortion as iodc2010_emit + iodc2010_eval,
%                  the evaluator tIodc2010 gates against an independent skew-ray
%                  tracer (2e-10 wave): the generalized tools inherit that gate
%     pins         the template's defaults (N-BK7, f = 100 mm, f/5, t = 4 mm,
%                  587.56 nm, object at infinity): Coddington's X =
%                  2(n^2-1)/(n+2) = 0.7397; CALIB on both radii from the
%                  equi-convex start lands within 0.002 of it (measured 0.7393);
%                  the best form's RMS at the paraxial focus is 1.2941 waves
%                  (pinned to 1e-3 relative), below the convex-plano lens's
%                  1.4006; the paraxial-to-best-focus ratio is the third-order
%                  value 4 to 2 % (measured 3.96)
%     conjugate    the object at 300 mm: the textbook X (position factor from the
%                  thin-lens conjugates) and CALIB's agree within 0.005 (measured
%                  0.2466 / 0.2489)
%
%   Model 128 -> SUITE_FAST.  Runtime ~15 s.

    properties
        tmp
    end
    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(h, 'templates', '35_lenses', 'singlet'));
            tc.tmp = tempname;  mkdir(tc.tmp);
            macos.init(128);
        end
    end
    methods (TestClassTeardown)
        function teardown(tc)
            if exist(tc.tmp, 'dir'), rmdir(tc.tmp, 's'); end
        end
    end

    methods (Test)
        function test_lens_tools_match_the_gated_iodc_evaluator(tc)
            here = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'challenges', 'iodc2010');
            addpath(here);  c = onCleanup(@() rmpath(here));
            p = iodc2010_params();  p.nGridpts = 33;
            n = iodc2010_glass(p.glass, p.lambda_nm);
            mk = @(d) iodc2010_lens('R', 80, 'sgn', [1 0 0 -1], 't', [6 d 6 NaN], ...
                'glass', {'N-BK7', 'air', 'N-BK7', 'air'}, 'stop', [6 + d/2, 7], 'name', 'gate_pair');
            f1 = 80 / (n - 1);  d0 = 2 * f1 - f1^2 / p.efl_mm;
            d  = fzero(@(dd) abs(iodc2010_paraxial(mk(dd), p.lambda_nm).efl) - p.efl_mm, d0);
            Li = mk(d);  Li.semidia = repmat(p.semidia_default, 1, Li.N);
            Pi = iodc2010_paraxial(Li, p.lambda_nm);
            Ei = iodc2010_emit(Li, Pi, p, fullfile(tc.tmp, 'iodc.in'));
            L  = macos.design.Lens.from_iodc(Li);
            P  = macos.design.lens_pupils(L, p.lambda_nm, 'stop', Li.stop);
            tc.verifyEqual(P.D, Pi.D, 'RelTol', 1e-13);
            tc.verifyEqual(P.z_xp, Pi.z_xp, 'RelTol', 1e-12);
            E  = macos.design.lens_emit(L, P, fullfile(tc.tmp, 'lens.in'), 'nGridpts', p.nGridpts, ...
                'standoff', p.src_standoff_mm);
            for th = [0 10]
                macos.load_rx(Ei.path);  Fi = iodc2010_eval(Ei, Pi, p, th);
                macos.load_rx(E.path);   F  = macos.design.lens_eval(E, P, th);
                tc.verifyEqual(F.rms_waves, Fi.rms_waves, 'RelTol', 1e-10, sprintf('RMS at %g deg', th));
                tc.verifyEqual(F.nLost, Fi.nLost);
                if th > 0, tc.verifyEqual(F.dist_pct, Fi.dist_pct, 'AbsTol', 1e-10); end
            end
            tc.verifyGreaterThan(F.rms_waves, 1, 'the 10-deg field is aberrated enough to tell decks apart');
        end

        function test_template_pins_at_infinity(tc)
            out = singlet('NSCAN', 0, 'OUTDIR', fullfile(tc.tmp, 'inf'));
            n = out.n;
            tc.verifyEqual(out.X_textbook, 2 * (n^2 - 1) / (n + 2), 'RelTol', 1e-14);
            tc.verifyEqual(out.X_textbook, 0.7397, 'AbsTol', 1e-4);
            tc.verifyEqual(out.X_calib, out.X_textbook, 'AbsTol', 2e-3, 'CALIB finds the textbook bending');
            tc.verifyEqual(out.rms_textbook, 1.2941, 'RelTol', 1e-3, 'best-form RMS at f/5 (paraxial focus)');
            tc.verifyLessThan(out.rms_textbook, out.rms_convex_plano);
            tc.verifyLessThan(out.rms_convex_plano, out.rms_plano_convex);
            tc.verifyLessThanOrEqual(out.rms_calib, out.rms_textbook * (1 + 1e-4));
            tc.verifyEqual(out.rms_textbook / out.rms_best_textbook, 4, 'RelTol', 0.02, ...
                'third-order SA: balancing focus cuts the RMS fourfold');
            tc.verifyTrue(isfile(fullfile(tc.tmp, 'inf', 'singlet_report.txt')));
        end

        function test_template_finite_conjugate(tc)
            out = singlet('NSCAN', 0, 'OUTDIR', fullfile(tc.tmp, 'fin'), 'OBJECT_MM', 300);
            tc.verifyEqual(out.X_calib, out.X_textbook, 'AbsTol', 5e-3);
            tc.verifyLessThan(out.X_textbook, 0.7397 - 0.3, 'a near object moves the best form toward equi-convex');
        end
    end
end
