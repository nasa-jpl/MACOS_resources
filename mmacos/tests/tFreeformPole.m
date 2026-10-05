classdef tFreeformPole < matlab.unittest.TestCase
%TFREEFORMPOLE  A Zernike freeform on an off-axis SECTION is evaluated about
%   the section's POLE (RptElt), not the parent vertex.
%
%   TO's finding (dyson5 addendum 44, 2026-10-05): Telescope's Zernike
%   emitter wrote `pMon = Vpt` -- the PARENT vertex -- while xMon/yMon/zMon
%   were already the section-pole frame.  On the -4 deg / 190 mm eccentric
%   TMA section M1's lit patch is centred 190 mm from that pMon, so with
%   lMon = the ~50 mm footprint radius the modes were evaluated around
%   rho ~ 3.8, far outside the unit disc -- the ill-conditioned regime
%   set_freeform's own note warns about, from the ORIGIN rather than lMon.
%   Every set_freeform / optimize_freeform solve on a section (CCMac's
%   step-5 route B included) ran its modes about the parent vertex.  The
%   engine evaluates Zernikes about pMon: TO measured a 1 um Z5 on M1 moving
%   the chief 0.194 mm at the FP vertex-centred vs 0.0045 mm pole-centred.
%
%   Fix: the emitter uses the same `pole` the RptElt line resolves (pMon ==
%   RptElt); coaxial elements have no pole and emit pMon == VptElt exactly
%   as before (byte-identical twin).  The must-fail leg is TO's engine
%   check: a 1 um defocus mode on the section's M1 must move the chief by
%   less than 0.03 mm (pre-fix 0.19 mm).
    properties (Constant)
        ModelSize = 256
    end
    methods (Access = private)
        function tel = tma_(tc, D, section)
            [R, t] = macos.design.tma_layout(D, 1.0, 1.7462, 'secondary_mag', 3.5, 'int_focus_m', -0.125*D, 'telecentric', true);
            tel = macos.design.Telescope('family','TMA','aperture_diameter_m',D,'wavelength_m',633e-9,'model_size',tc.ModelSize);
            tel.add_mirror('M1','radius_m',R(1),'spacing_after_m',t(1));
            tel.add_mirror('M2','radius_m',R(2),'spacing_after_m',t(2),'convex',true);
            tel.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
            tel.add_focal_plane('FP');  tel.build();
            if section
                tel.set_field_bias(-3*60);  tel.set_offaxis('none', 'dist', 0.18);  tel.build();
            end
        end
        function v = key_(~, txt, key, ielt)
            % the 3-vector after `key=` inside element ielt's block
            blocks = regexp(txt, '(?<!\w)iElt=\s*(\d+)', 'split');   % psiElt= contains iElt=
            b = blocks{ielt + 1};
            m = regexp(b, [key '=\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)'], 'tokens', 'once');
            v = cellfun(@str2double, m);
        end
        function p = chief_(~, tel)
            tel.build();  s = macos.trace(numel(tel.spec.elt));  ri = macos.get_ray_info(s.nRays);  p = ri.pos(:, 1);
        end
    end
    methods (Test)
        function test_section_pmon_is_the_pole(tc)
            macos.init(tc.ModelSize);
            tel = tc.tma_(0.330/1.8, true);
            tel.set_freeform(1, [4 6], [0 0]);  tel.build();
            rx = tel.save([tempname '.in']);  txt = fileread(rx);  delete(rx);
            pmon = tc.key_(txt, 'pMon', 1);  rpt = tc.key_(txt, 'RptElt', 1);  vpt = tc.key_(txt, 'VptElt', 1);
            tc.verifyGreaterThan(norm(rpt - vpt), 0.1, 'the fixture is a section: pole 180 mm from the vertex');
            tc.verifyEqual(pmon, rpt, 'AbsTol', 1e-12, sprintf('pMon must be the pole (got %s, pole %s)', mat2str(pmon, 4), mat2str(rpt, 4)));
        end
        function test_coaxial_pmon_is_the_vertex(tc)
            macos.init(tc.ModelSize);
            tel = tc.tma_(0.330/1.8, false);
            tel.set_freeform(1, [4 6], [0 0]);  tel.build();
            rx = tel.save([tempname '.in']);  txt = fileread(rx);  delete(rx);
            pmon = tc.key_(txt, 'pMon', 1);  vpt = tc.key_(txt, 'VptElt', 1);  rpt = tc.key_(txt, 'RptElt', 1);
            tc.verifyEqual(rpt, vpt, 'AbsTol', 0, 'coaxial: no pole');
            tc.verifyEqual(pmon, vpt, 'AbsTol', 0, 'coaxial: pMon == VptElt, byte-identical to before');
        end
        function test_defocus_mode_acts_about_the_lit_patch(tc)
            % TO's engine check: 1 um of Z5 on the section's M1 moves the
            % chief 0.194 mm when the modes sit at the vertex, ~0.005 mm at
            % the pole (the mode is then a true defocus over the lit patch)
            macos.init(tc.ModelSize);
            tel = tc.tma_(0.330/1.8, true);
            tel.set_freeform(1, 5, 0, 'lmon', 0.05);
            p0 = tc.chief_(tel);
            tel.set_freeform(1, 5, 1e-6, 'lmon', 0.05);
            p1 = tc.chief_(tel);
            d = norm(p1 - p0);
            tc.verifyLessThan(d, 3e-5, sprintf('chief moved %.4f mm for 1 um Z5 on M1 (vertex-centred: 0.194 mm)', d*1e3));
            tc.verifyGreaterThan(d, 1e-7, 'the mode must do something (not vacuous)');
        end
    end
end
