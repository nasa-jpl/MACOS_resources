classdef tSmallCamera < matlab.unittest.TestCase
%TSMALLCAMERA  The small dedicated imager template (templates/36_cameras/small_camera).
%
%   The found deck is the macos manual's ImageDemo.in beside this checkout;
%   the class is skipped when that checkout is absent.
%
%   Legs
%     found_ab        the strip removes the primary's 36 Zernike terms and puts
%                     three elements back on the axis; the builder rebuilds the
%                     stripped deck from its own numbers and the two trace alike
%                     (|diff| < 1e-8 wave at 0, half and full field; measured
%                     1.7e-10)
%     first_order     against the camera's design values (the template's
%                     defaults: 3.5 m, f/10, 13 um pixels, 2048 of them): f within
%                     0.5 %, IFOV 3.7 urad within 0.05, the field 0.44 deg within
%                     0.01; the deck's conics are the Ritchey-Chretien's of its own
%                     m and beta (K1 within 0.005, K2 within 0.03; the classical
%                     K2 is 0.11 away)
%     refocus         the secondary's despace moves the paraxial focus by
%                     (m^2 + 1) M_relay^2 times the despace (the two-mirror
%                     longitudinal magnification through the relay, written from
%                     the textbook, 2 %); the refocus is -6.08 um (2 %); every
%                     field is then under Marechal's 0.0714 wave at 700 nm
%     glasses_band    the fixed indices match BaF8 (< 1e-5) and SF19 (< 3e-4); in
%                     those glasses the found design stays under 0.0714 wave over
%                     500-900 nm and the whole field; its axial color fits inside
%                     the depth of focus
%     bare_rc         without the relay the RC's edge field is over 10x the
%                     camera's at 700 nm; the relay brings the distortion (> 1 %
%                     against < 0.05 % bare)
%     spec_design     the camera from its spec hits f = 3500 mm (1e-9) at f/10
%                     with the detector 0.127 mm behind the filter, loses no ray
%                     beyond the shadow, clears the hole, and beats both its seed
%                     and the found design in glass; merit pinned 0.0447 (5 %)
%     sampling        Q = lambda N / p from the result (0.538 at 700 nm)
%
%   Model 128 -> SUITE_FAST.  Runtime ~25 s.

    properties
        tmp
        out
    end
    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(fileparts(mfilename('fullpath')));
            deck = fullfile(fileparts(fileparts(h)), 'macos', 'docs', 'macos-manual', 'examples', 'ImageDemo.in');
            tc.assumeTrue(isfile(deck), 'the macos checkout with ImageDemo.in is not beside MACOS_resources');
            addpath(fullfile(h, 'templates', '36_cameras', 'small_camera'));
            tc.tmp = tempname;  mkdir(tc.tmp);
            macos.init(128);
            tc.out = small_camera('OUTDIR', tc.tmp, 'DECK', deck);
        end
    end
    methods (TestClassTeardown)
        function teardown(tc)
            if ~isempty(tc.tmp) && exist(tc.tmp, 'dir'), rmdir(tc.tmp, 's'); end
        end
    end

    methods (Test)
        function test_found_ab(tc)
            o = tc.out;
            tc.verifyEqual(o.info.n_zern, 36, 'Zernike terms stripped');
            tc.verifyEqual(o.info.zern_elt, 'PRI_MIR');
            tc.verifyEqual(sort(o.info.recentred), sort({'SEC_MIR', 'EXIT_RET', 'IMG_PL'}));
            tc.verifyLessThan(max(abs(o.ab(1, :) - o.ab(2, :))), 1e-8, 'builder vs the stripped deck');
            tc.verifyGreaterThan(min(o.ab(:)), 0.1, 'the A/B is on the unrefocused deck (not a vacuous zero)');
        end

        function test_first_order_against_the_design(tc)
            o = tc.out;  C = o.C1;
            tc.verifyEqual(C.par.yP_unit, 3500, 'RelTol', 5e-3, 'focal length 3.5 m');
            tc.verifyEqual(C.par.fno, 10, 'AbsTol', 0.05, 'f/10');
            tc.verifyEqual(13e-3 / C.par.yP_unit * 1e6, 3.7, 'AbsTol', 0.05, '3.7 urad per pixel');
            tc.verifyEqual(2 * o.half0, 0.44, 'AbsTol', 0.01, 'the 2048-pixel field');
            M = o.found.M;  f1 = M.R1 / 2;  m = o.m;  b = o.beta;
            cs = ((m + 1) / (m - 1))^2;                                 % Schroeder Table 6.1
            K1rc = -1 - 2 * (1 + b) / (m^2 * (m - b));
            K2rc = -cs - 2 * m * (m + 1) / ((m - b) * (m - 1)^3);
            tc.verifyEqual(M.sep, f1 * (m - b) / (m + 1), 'RelTol', 1e-10, 'the (m, beta) solve');
            tc.verifyEqual(M.K1, K1rc, 'AbsTol', 0.005, 'primary: the RC conic');
            tc.verifyEqual(M.K2, K2rc, 'AbsTol', 0.03, 'secondary: the RC conic');
            tc.verifyGreaterThan(abs(M.K2 + cs), 0.1, 'and not the classical Cassegrain''s');
        end

        function test_refocus(tc)
            o = tc.out;
            dsep = o.refocus_um * 1e-3;
            dz = o.C1.par.z_paraxial - o.C0.par.z_paraxial;
            lng = (o.m^2 + 1) * o.C0.par.telescope.relay_mag^2;
            tc.verifyEqual(dz / -dsep, lng, 'RelTol', 0.02, 'focus per despace = (m^2+1) M_relay^2');
            tc.verifyEqual(o.refocus_um, -6.08, 'RelTol', 0.02);
            tc.verifyLessThan(o.scan_refocused.rms, 0.0714, 'diffraction-limited at 700 nm over the field');
            tc.verifyTrue(o.C1.par.hole.clears, 'the hole clears');
        end

        function test_glasses_band(tc)
            o = tc.out;
            tc.verifyEqual(o.glasses(1:2), {'BaF8', 'SF19'});
            tc.verifyEqual(macos.design.glass('BaF8', 700), 1.61783, 'AbsTol', 1e-5);
            tc.verifyEqual(macos.design.glass('SF19', 700), 1.657863, 'AbsTol', 3e-4);
            tc.verifyLessThan(o.scan_glass.rms, 0.0714, 'the found design over 500-900 nm');
            tc.verifyLessThan(max(o.zpar_band) - min(o.zpar_band), 2 * 700e-6 * o.C3.par.fno^2, 'axial color inside the depth of focus');
        end

        function test_bare_rc(tc)
            o = tc.out;
            tc.verifyGreaterThan(o.scan_bare.rms(end), 10 * o.scan_refocused.rms(end), 'the corrector''s job: the field');
            tc.verifyLessThan(abs(o.scan_bare.dist(end)), 0.05);
            tc.verifyGreaterThan(o.scan_refocused.dist(end), 1, 'the relay brings the distortion');
        end

        function test_spec_design(tc)
            o = tc.out;  C = o.C5;
            tc.verifyEqual(C.par.efl, 3500, 'RelTol', 1e-9);
            tc.verifyEqual(C.par.fno, 10, 'RelTol', 1e-9);
            gap = C.par.z_img - (C.surf(end).z);
            tc.verifyEqual(gap, 0.127, 'AbsTol', 1e-4, 'the detector 0.127 mm behind the filter');
            tc.verifyEqual(o.scan_design.lost, o.scan_design.lost(1, 1) * ones(size(o.scan_design.lost)), 'only the shadow');
            tc.verifyTrue(C.par.hole.clears);
            tc.verifyLessThan(o.merit, o.merit_seed, 'beats its seed');
            tc.verifyLessThan(max(o.scan_design.rms(:)), max(o.scan_glass.rms(:)), 'beats the found design in glass');
            tc.verifyEqual(o.merit, 0.0447, 'RelTol', 0.05);
        end

        function test_sampling(tc)
            o = tc.out;
            tc.verifyEqual(o.Q, o.lams * 1e-6 * o.C5.par.fno / 13e-3, 'RelTol', 1e-12);
            tc.verifyEqual(o.Q(2), 0.538, 'AbsTol', 1e-3);
        end
    end
end
