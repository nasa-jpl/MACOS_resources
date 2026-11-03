classdef tCassCamera < matlab.unittest.TestCase
%TCASSCAMERA  macos.design.cass_camera + camera_emit (the small-camera builder).
%
%   Legs
%     bare_stigmatic   a classical Cassegrain (paraboloid + the confocal
%                      hyperboloid) and a Ritchey-Chretien from the closed form,
%                      no relay, at their paraxial focus: the classical form is
%                      stigmatic on axis (< 1e-6 wave; the RC's third-order conics
%                      leave < 1e-3), its coma at 0.005 deg is Schroeder's
%                      paraboloid coma to 0.5 % (measured 0.15 %), and the RC,
%                      aplanatic, has under a tenth of it
%     paraxial         the y-nu trace through the folded train against two
%                      independent numbers: Telescope's closed-form focal length
%                      m f1 (1e-9) and the ENGINE's near-axis chief height per
%                      tan(theta) with a refractive relay in place (1e-5)
%     despace          moving the secondary moves the camera's paraxial focus by
%                      (m^2 + 1) M_relay^2 times the despace: the secondary's
%                      longitudinal magnification through the relay, written from
%                      the textbook (m = f_telescope / f1); 0.1 %
%     calib_block      the CALIB keywords land on the asked elements (DOF 8 on a
%                      conic mirror, DOF 6 on the secondary, DOF 7 on a relay
%                      radius, ApStop on the primary); a radius and a position on
%                      one relay surface is refused (the engine fails on it)
%     glass_rows       a catalog glass writes GlassElt / Sellmeier / GlassCoef
%     hole_warns       a shadow smaller than the returning beam warns
%                      (macos:design:cass_camera:hole), it does not refuse
%
%   Model 128 -> SUITE_FAST.  Runtime ~6 s.

    properties
        tmp
    end
    methods (TestClassSetup)
        function setup(tc)
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
        function test_bare_stigmatic(tc)
            % D 350, f/9, the shadow 0.27 of the radius; 0.005 deg is deep in the
            % coma-linear regime (the classical form's RMS doubles from 0.005 to
            % 0.01 deg, the RC's quadruples)
            D = 350;  Fn = 9;  ep = 0.27;  th = 0.005;  lam = 700e-6;
            rms = zeros(2, 2);  fam = {'cassegrain', 'rc'};
            for i = 1:2
                C = macos.design.cass_camera('family', fam{i}, 'D', D, 'primary_fnum', 1.334, ...
                    'telescope_fnum', Fn, 'bfd', 47, 'obscuration', ep);
                E = macos.design.camera_emit(C, fullfile(tc.tmp, [fam{i} '.in']), 'nGridpts', 129);
                macos.load_rx(E.path);
                for j = 1:2
                    F = macos.design.lens_eval(E, E.P, (j - 1) * th);
                    tc.verifyLessThan(F.nLost, 0.1 * F.nLaunched, sprintf('%s: only the shadow loses rays', fam{i}));
                    rms(i, j) = F.rms_waves;
                end
            end
            tc.verifyLessThan(rms(1, 1), 1e-6, 'classical: paraboloid + confocal hyperboloid is stigmatic on axis');
            tc.verifyLessThan(rms(2, 1), 1e-3, 'RC: the third-order conics leave only fifth-order spherical');
            % the classical form's coma is the paraboloid's (Schroeder, Astronomical
            % Optics, 2nd ed., Table 6.3): angular sagittal coma theta/(16 F^2), so
            % W131 = theta D / (32 F^2); RMS of W131 rho^3 cos(phi) over the annulus
            % = W131 sqrt((1 - ep^8) / (8 (1 - ep^2))) (chief-referenced, tilt kept)
            w131 = deg2rad(th) * D / (32 * Fn^2) / lam;
            tc.verifyEqual(rms(1, 2), w131 * sqrt((1 - ep^8) / (8 * (1 - ep^2))), 'RelTol', 5e-3, ...
                'classical Cassegrain coma against the textbook');
            tc.verifyLessThan(rms(2, 2), 0.1 * rms(1, 2), 'the RC is aplanatic: no linear coma');
        end

        function test_paraxial_against_closed_form_and_engine(tc)
            C = macos.design.cass_camera('family', 'rc', 'D', 350, 'primary_fnum', 1.334, ...
                'telescope_fnum', 9, 'bfd', 47);
            d = macos.design.Telescope('family', 'rc', 'aperture_diameter_mm', 350, 'system_fnum', 9, ...
                'primary_fnum', 1.334, 'BFD_mm', 47).spec.derived;
            tc.verifyEqual(C.par.telescope.efl, d.f * 1e3, 'RelTol', 1e-9, 'm f1');
            tc.verifyEqual(C.par.telescope.z_focus, d.bfd * 1e3, 'RelTol', 1e-9, 'BFD');
            % with a relay (small_camera.in's corrector, fixed indices): the engine's chief height
            Lc = macos.design.Lens('R', [-42.41546 31.5595 553.32376], 't', [1.50114 3.9497], ...
                'glass', {1.61783, 1.657863}, 'name', 'COR');
            M = struct('R1', 933.7886403, 'K1', -1.009724001, 'R2', 156.59354, 'K2', -1.932283999, 'sep', 400.25416);
            C = macos.design.cass_camera('mirrors', M, 'relay', {Lc, 39.88054}, 'obscuration', 0.27);
            E = macos.design.camera_emit(C, fullfile(tc.tmp, 'relay.in'), 'nGridpts', 17);
            macos.load_rx(E.path);
            th = 0.002;
            F = macos.design.lens_eval(E, E.P, th);
            tc.verifyEqual(F.yR / tand(th), C.par.yP_unit, 'RelTol', 1e-5, 'chief height per tan(theta)');
            tc.verifyGreaterThan(C.par.telescope.relay_mag, 1, 'a negative corrector lengthens the focal length');
        end

        function test_despace(tc)
            Lc = macos.design.Lens('R', [-42.41546 31.5595 553.32376], 't', [1.50114 3.9497], ...
                'glass', {1.61783, 1.657863}, 'name', 'COR');
            M = struct('R1', 933.7886403, 'K1', -1.009724001, 'R2', 156.59354, 'K2', -1.932283999, 'sep', 400.25416);
            mk = @(sep) macos.design.cass_camera('mirrors', setfield(M, 'sep', sep), 'relay', {Lc, 39.88054}); %#ok<SFLD>
            C = mk(M.sep);  h = 1e-4;
            dz = (mk(M.sep - h).par.z_paraxial - mk(M.sep + h).par.z_paraxial) / (2 * h);
            m = abs(C.par.telescope.efl) / (M.R1 / 2);
            tc.verifyEqual(dz, (m^2 + 1) * C.par.telescope.relay_mag^2, 'RelTol', 1e-3, ...
                'focus per despace = (m^2 + 1) M_relay^2');
        end

        function test_calib_block(tc)
            Lc = macos.design.Lens('R', [-42.4 31.6 553], 't', [1.5 3.95], 'glass', {'SK10', 'SF19'}, 'name', 'COR');
            C = macos.design.cass_camera('family', 'rc', 'D', 350, 'primary_fnum', 1.334, ...
                'telescope_fnum', 9, 'bfd', 47, 'relay', {Lc, 40}, 'obscuration', 0.27);
            p = fullfile(tc.tmp, 'cal.in');
            E = macos.design.camera_emit(C, p, 'calib', struct('fields_deg', [0 0.2], 'var_K', 2, ...
                'var_sep', true, 'var_R', [1 3], 'var_fp', true));
            blk = elements_(p);
            tc.verifyTrue(contains(blk{E.iM1}, 'ApStop=') && ~contains(blk{E.iM1}, 'VarElt='), 'primary: stop, no DOF');
            tc.verifyTrue(contains(blk{E.iM2}, 'VarElt=  0 0 0 0 0 1 0 1'), 'secondary: position + conic');
            tc.verifyTrue(contains(blk{E.iSurf(1)}, 'VarElt=  0 0 0 0 0 0 1 0'), 'relay 1: radius');
            tc.verifyFalse(contains(blk{E.iSurf(2)}, 'VarElt='), 'relay 2: fixed');
            tc.verifyTrue(contains(blk{E.iSurf(3)}, 'VarElt=  0 0 0 0 0 0 1 0'), 'relay 3: radius');
            tc.verifyTrue(contains(blk{E.iFP}, 'VarElt='), 'image plane free');
            tc.verifyError(@() macos.design.camera_emit(C, p, 'calib', struct('var_R', 2, 'var_pos', 2)), ...
                ?MException);
            macos.load_rx(p);                       % the deck loads
        end

        function test_glass_rows(tc)
            Lc = macos.design.Lens('R', [-42.4 31.6 553], 't', [1.5 3.95], 'glass', {'SK10', 'SF19'}, 'name', 'COR');
            C = macos.design.cass_camera('family', 'rc', 'D', 350, 'primary_fnum', 1.334, ...
                'telescope_fnum', 9, 'bfd', 47, 'relay', {Lc, 40});
            p = fullfile(tc.tmp, 'glass.in');
            E = macos.design.camera_emit(C, p);
            blk = elements_(p);
            [~, S] = macos.design.glass('SK10', 700);
            tc.verifyTrue(contains(blk{E.iSurf(1)}, 'GlassElt=  SK10') && contains(blk{E.iSurf(1)}, 'GlassType=  Sellmeier'));
            tc.verifyTrue(contains(blk{E.iSurf(1)}, sprintf('%.12G', S.B(1))), 'the table row is written');
            tc.verifyFalse(contains(blk{E.iSurf(3)}, 'GlassElt='), 'air after the last surface');
        end

        function test_hole_warns(tc)
            Lc = macos.design.Lens('R', [-42.4 31.6 553], 't', [1.5 3.95], 'glass', {1.6, 1.65}, 'name', 'COR');
            f = @() macos.design.cass_camera('family', 'rc', 'D', 350, 'primary_fnum', 1.334, ...
                'telescope_fnum', 9, 'bfd', 47, 'relay', {Lc, 40}, 'obscuration', 0.005, 'field_deg', 0.2);
            C = tc.verifyWarning(f, 'macos:design:cass_camera:hole');
            tc.verifyFalse(C.par.hole.clears);
        end
    end
end

function blk = elements_(p)
% the deck's element blocks, indexed by iElt
t = fileread(p);
parts = regexp(t, '(?m)^\s*iElt=\s*\d+', 'split');
blk = parts(2:end);
blk{end} = regexprep(blk{end}, 'nOutCord=.*', '');
end
