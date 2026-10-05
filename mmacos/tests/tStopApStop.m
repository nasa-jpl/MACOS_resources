classdef tStopApStop < matlab.unittest.TestCase
%TSTOPAPSTOP  Telescope.add_pupil / optimize(OptFEX) set the stop at the DECK's ApStop, not a bare stop(1).
%   dyson5 addendum 42 (2026-10-05): on an eccentric section (set_offaxis) VptElt(1) is the PARENT vertex, off the
%   beam; add_pupil's bare macos.stop(1) before its FEX call aimed the chief there, so FEX placed the exit-pupil
%   sphere for the parent-axis beam (-4/190 1.5k section: StopPos (0,0,0), EP at (0,-6.7 mm,0.425 m), r 0.571 m,
%   probe axes 0.333 vs 0.809 m), and optimize's OptFEX branch re-issued it at every CALIB evaluation.  Now both set
%   the stop at the deck's ApStop in object space (Telescope.stop_at_apstop_).
%   Fixture: the telescope's telecentric Korsch parent (tma_layout 'telecentric', F/1.7133), the -4 deg / 180 mm
%   eccentric section; coaxial twin = the same parent unsectioned.
%   (1) MUST-PASS: after add_pupil the EP vertex lies on the section's exit chief -- to 1 mm (FEX traces an off-axis
%       PROBE field, 1 arcmin, so its chief is 35 um off the nominal one: measured) against ~0.2 m for the defect.
%   (2) MUST-FAIL (the defect, reproduced by hand): the old path -- bare macos.stop(1) then FEX -- puts the EP off
%       that chief by > 0.1 m.
%   (3) optimize with the EP merit leaves the chief on the section at M1 (no parent-vertex re-aim).  On THIS
%       near-telecentric section CALIB's OptFEX then aborts (FEX finds a far, astigmatic pupil -- T/S split
%       0.47-0.77 m -- and the EP sphere loses every ray): that is a separate finding (addendum 42 checkpoint 1),
%       so the test asserts the STOP, with the CALIB outcome caught.
%   (4) coaxial twin (tAsphHook's NON-telecentric TMA parent, ApStop = M1's vertex): the new add_pupil's EP equals
%       the old bare-stop path's to 1e-9 m.  (The telecentric parent cannot serve: its pupil is at infinity and FEX
%       is ill-conditioned there -- old vs new differ by 0.29 mm on round-off.)
    properties (Constant)
        ModelSize = 256
    end
    methods (Access = private)
        function tel = parent_(~)
            D = 0.330/1.8;
            [R, t] = macos.design.tma_layout(D, 1.0, 1.7133, 'secondary_mag', 3.5, 'int_focus_m', -0.125*D, 'telecentric', true);
            tel = macos.design.Telescope('family','TMA','aperture_diameter_m',D,'wavelength_m',633e-9,'model_size',256);
            tel.add_mirror('M1','radius_m',R(1),'spacing_after_m',t(1));
            tel.add_mirror('M2','radius_m',R(2),'spacing_after_m',t(2),'convex',true);
            tel.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
            tel.add_focal_plane('FP');  tel.build();
        end
        function tel = section_(tc)
            tel = tc.parent_();  tel.set_field_bias(-4*60);  tel.set_offaxis('none','dist',0.18);  tel.build();
        end
        function [p, d] = exit_chief_(~, nE)
            % the chief at the detector (last element) and its direction there, on the loaded deck
            s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);  p = ri.pos(:, 1);  d = ri.dir(:, 1)/norm(ri.dir(:, 1));
        end
        function v = old_ep_(~, nE, field_rad)
            % the pre-fix add_pupil EP, by hand: off-axis probe field, BARE stop(1), FEX, field restored
            cur = macos.get_src_fov();  d0 = cur.src_dir(:)/norm(cur.src_dir);
            px = [1;0;0];  px = px - dot(px, d0)*d0;  px = px/norm(px);
            macos.set_src_fov('src_dir', d0*cos(field_rad) + px*sin(field_rad));
            macos.stop(1);  macos.trace(nE);  f = macos.fex(1);  v = f.vpt(:);
            macos.set_src_fov('src_dir', cur.src_dir);
        end
    end
    methods (TestClassSetup)
        function setupClass(tc), macos.init(tc.ModelSize); end
    end
    methods (Test)
        function test_section_ep_lies_on_the_sections_exit_chief(tc)
            tel = tc.section_();  nE0 = numel(tel.spec.elt);
            [pc, dc] = tc.exit_chief_(nE0);                       % the section's exit chief at the detector
            tel.add_pupil(nE0);
            ep = tel.spec.pupil.ep_vpt(:);  w = ep - pc;  off = norm(w - dc*(dc'*w));
            tc.verifyLessThan(off, 1e-3, sprintf('EP %s must lie on the section''s exit chief: off by %.3e m', mat2str(ep', 4), off));
            tc.verifyGreaterThan(tel.spec.pupil.ep_radius, 0.3, sprintf('EP radius %.3f m (the section''s pupil is ~1 m out)', tel.spec.pupil.ep_radius));
        end
        function test_old_bare_stop_path_puts_the_ep_off_the_section(tc)
            tel = tc.section_();  nE0 = numel(tel.spec.elt);
            [pc, dc] = tc.exit_chief_(nE0);
            tel.add_pupil(nE0);  nE = macos.num_elt();
            ep_old = tc.old_ep_(nE, 2.908882e-4);  w = ep_old - pc;  off = norm(w - dc*(dc'*w));
            tc.verifyGreaterThan(off, 0.1, sprintf('the bare-stop(1) EP must be OFF the section''s chief (measured the defect at ~0.2 m); got %.3e m', off));
        end
        function test_optimize_with_the_ep_merit_keeps_the_chief_on_the_section(tc)
            tel = tc.section_();  nE0 = numel(tel.spec.elt);
            s = macos.trace(1);  ri = macos.get_ray_info(s.nRays);  y0 = ri.pos(2, 1);
            tel.add_pupil(nE0);
            fx = [-1.17 1.17]*pi/180;
            try
                tel.optimize('fields', [fx.' [0; 0]], 'dofs', [0 0 0 0 0 0 0 1], 'max_iters', 2);
            catch e
                tc.verifyTrue(contains(e.message, 'calib_run failed'), ['only CALIB''s own abort is expected here: ' e.message]);
            end
            s = macos.trace(1);  ri = macos.get_ray_info(s.nRays);  y1 = ri.pos(2, 1);
            tc.verifyGreaterThan(y0, 0.15, sprintf('the seed chief hits M1 on the section (y %.4f m)', y0));
            tc.verifyLessThan(abs(y1 - y0), 1e-3, sprintf('after the EP-merit solve the chief must still hit M1 on the section: y %.4f -> %.4f m', y0, y1));
        end
        function test_coaxial_twin_is_unchanged(tc)
            D = 0.330/1.8;
            [R, t] = macos.design.tma_layout(D, 1.0, 1.8, 'secondary_mag', 3.5, 'int_focus_m', -0.125*D, 'm3_behind_m', 0.6*D);
            tel = macos.design.Telescope('family','TMA','aperture_diameter_m',D,'wavelength_m',633e-9,'model_size',256);
            tel.add_mirror('M1','radius_m',R(1),'spacing_after_m',t(1));
            tel.add_mirror('M2','radius_m',R(2),'spacing_after_m',t(2),'convex',true);
            tel.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
            tel.add_focal_plane('FP');  tel.build();  nE0 = numel(tel.spec.elt);
            tel.add_pupil(nE0);  ep_new = tel.spec.pupil.ep_vpt(:);  nE = macos.num_elt();
            ep_old = tc.old_ep_(nE, 2.908882e-4);
            tc.verifyLessThan(norm(ep_new - ep_old), 1e-9, sprintf('coaxial: ApStop = M1''s vertex, EP must not move (%.3e m)', norm(ep_new - ep_old)));
        end
    end
end
