classdef tLensDoublet < matlab.unittest.TestCase
%TLENSDOUBLET  The doublet template (templates/35_lenses/doublet) and macos.design.lens_catalog.
%
%   Legs
%     catalog      both shipped Newport parts (PAC097AR.14, PAC095AR.14): the
%                  paraxial EFL within 0.5 % and BFL within 0.5 mm of the
%                  vendor's quoted numbers at 589 nm (measured 250.32 / 239.88
%                  against 250 / 240.1 for PAC095); the engine's EFL on a
%                  collimated bench equals the paraxial solve to 1e-9; their
%                  flints N-SF5 / N-F2 come from the engine table and hit
%                  Schott's published n_d to 5e-6
%     glass rows   a table GLASS line the engine table lacks is carried as a
%                  Sellmeier row: the Lens holds it, the deck writes it as
%                  GlassCoef= under its own name, the engine's EFL equals the
%                  paraxial solve to 1e-9; must-fail: the same part without the
%                  glass line is refused as an unknown glass
%     template     the defaults (N-BK7 / F2, f = 100 mm, f/5, t = [6 2.5]):
%                  the paraxial achromat brings F and C to one focus (1e-10 mm);
%                  the ENGINE's focal shifts at F and C, one trace per
%                  wavelength, equal the y-nu ones to 1e-6 mm; its secondary
%                  spectrum is the textbook thin-lens f(P1-P2)/(V1-V2) to 3 %
%                  (measured -0.04895 against -0.04993 mm); CALIB at F, d, C
%                  holds the worst RMS across the band at 0.0759 waves (pinned
%                  to 2 %), a third of the paraxial achromat's 0.2672; CALIB at
%                  d alone nulls d (0.0056) and leaves C at 0.127 (the reason
%                  the template solves at three wavelengths)
%
%   Model 128 -> SUITE_FAST.  Runtime ~20 s.

    properties
        tmp
    end
    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(h, 'templates', '35_lenses', 'doublet'));
            tc.tmp = tempname;  mkdir(tc.tmp);
            macos.init(128);
        end
    end
    methods (TestClassTeardown)
        function teardown(tc)
            if exist(tc.tmp, 'dir'), rmdir(tc.tmp, 's'); end
        end
    end

    methods
        function [efl, O] = engine_efl(tc, L, lam, tag)
            b = macos.design.Bench(tag, 'dir', [0; 0; 1], 'zsource', 1e22, 'aperture', 0.4 * L.D, ...
                'wavelen', lam * 1e-6, 'ngridpts', 21);
            O = b.add_optic(L, 50);  b.add_detector(L.paraxial(lam).bfl, 'FP');
            f = fullfile(tc.tmp, [tag '.in']);  b.emit(f);  macos.load_rx(f);
            hs = [1e-4 2e-4];  E = zeros(1, 2);  ax = [0; 0; 1];
            for i = 1:2
                r = macos.srt(hs(i), 0, O.i_last);
                p1 = r.pos(:, O.i_first);  hv = p1 - ax * (ax.' * p1);  e1 = hv / norm(hv);
                d = r.dir(:, O.i_last);
                E(i) = -norm(hv) / ((e1.' * d) / (ax.' * d));
            end
            efl = (4 * E(1) - E(2)) / 3;
        end
    end

    methods (Test)
        function test_catalog_parts_meet_the_vendor_numbers(tc)
            parts = macos.design.lens_catalog();
            tc.verifyEqual(sort(parts), sort({'PAC095AR.14', 'PAC097AR.14'}));
            for k = 1:numel(parts)
                [L, info] = macos.design.lens_catalog(parts{k});
                P = L.paraxial(info.lambda_nm);
                tc.verifyEqual(P.efl, info.efl, 'RelTol', 5e-3, [parts{k} ' EFL vs vendor']);
                tc.verifyEqual(P.bfl, info.bfl, 'AbsTol', 0.5, [parts{k} ' BFL vs vendor']);
                efl = tc.engine_efl(L, info.lambda_nm, ['cat' num2str(k)]);
                tc.verifyEqual(efl, P.efl, 'RelTol', 1e-9, [parts{k} ' engine EFL vs y-nu']);
            end
            tc.verifyEqual(macos.design.glass('N-SF5', 587.5618), 1.67271, 'AbsTol', 5e-6);
            tc.verifyEqual(macos.design.glass('N-F2', 587.5618), 1.62005, 'AbsTol', 5e-6);
        end

        function test_table_glass_rows_reach_the_deck(tc)
            row = '1.39757037 0.159201403 1.2686543 0.00995906143 0.0546931752 119.248346';
            prt = 'part | TP1 | test | 140.890;-101.720;-454.867 | 14.5;4.0 | N-BK7;TESTGLASS | 76.2 | 250 | 240.1 | 589 | tLensDoublet';
            f1 = fullfile(tc.tmp, 'cat_ok.txt');  f0 = fullfile(tc.tmp, 'cat_bad.txt');
            writelines({['glass | TESTGLASS | ' row ' | 1.62005 | N-F2 coefficients under another name'], prt}, f1);
            writelines({prt}, f0);
            [L, info] = macos.design.lens_catalog('TP1', 'file', f1);
            tc.verifyTrue(isstruct(L.glass{2}));
            tc.verifyEqual(numel(info.glass_rows), 1);
            [efl, ~] = tc.engine_efl(L, 589, 'rowdeck');
            txt = fileread(fullfile(tc.tmp, 'rowdeck.in'));
            tc.verifyNotEmpty(strfind(txt, 'GlassElt=  TESTGLASS'));
            tc.verifyNotEmpty(strfind(txt, 'GlassCoef=  1.39757037  0.159201403'));
            tc.verifyEqual(efl, L.paraxial(589).efl, 'RelTol', 1e-9, 'the engine traces the row it was given');
            Lb = macos.design.lens_catalog('TP1', 'file', f0);
            tc.verifyError(@() Lb.paraxial(589), 'macos:design:glass:unknown');
        end

        function test_template_pins(tc)
            out = doublet('VENDOR_PART', '', 'OUTDIR', fullfile(tc.tmp, 'd'));
            z0 = out.bfl_paraxial_achromat;  zE = out.bfl_engine_achromat;
            tc.verifyLessThan(abs(z0(1) - z0(3)), 1e-10, 'paraxial achromat: F and C at one focus');
            tc.verifyEqual(zE - zE(2), z0 - z0(2), 'AbsTol', 1e-6, 'engine focal shift (3 traces) = y-nu');
            tc.verifyEqual(out.secondary_engine, -out.secondary_textbook, 'RelTol', 0.03, 'secondary spectrum vs textbook');
            tc.verifyEqual(max(out.rms_calib), 0.0759, 'RelTol', 0.02);
            tc.verifyLessThan(max(out.rms_calib), max(out.rms_paraxial) / 3);
            tc.verifyLessThan(out.rms_calib_d(2), 0.01, 'd alone nulls d');
            tc.verifyGreaterThan(out.rms_calib_d(3), 2 * out.rms_calib(3), '... and lets C go');
        end
    end
end
