classdef tTmaTelecentric < matlab.unittest.TestCase
%TTMATELECENTRIC  tma_layout's 'telecentric' option (2026-10-04, dyson5 round 4):
%   M3 placed so the exit pupil is at infinity, with the stop at M2.
%
%   Pinned: (1) with 'stop','M2' and 'telecentric' the paraxial chief's exit
%   slope is zero and the identity t2 = R3/2 (M2 at M3's front focus) holds
%   to round-off, for feeds m2 2.5..8 at f/1.8; M3 sits in FRONT of the
%   intermediate focus (a virtual intermediate image -- the Cook/EMIT regime
%   the ordinary order check forbids).  (2) With the stop at M1 -- the stop
%   the Telescope emits -- the layout is telecentric IN THE ENGINE: the chief
%   directions at the FP for +-0.02 deg fields are parallel to 1e-7 rad with
%   every ray alive (measured 9.9e-9), and the traced EFL is 330 mm; the
%   default (non-telecentric) layout reports its finite exit pupil (53 mm
%   after M3 on the dyson5 parent -- the 22 mm-from-image pupil TO measured
%   on the section).  Non-vacuity: the default layout's chiefs differ by
%   > 1e-3 rad for the same fields.
    methods (Test)
        function test_stop_at_m2_gives_a_telecentric_korsch(tc)
            D = 0.330/1.8;
            for m2 = [2.5 3.5 5 8]
                [R, t, i] = macos.design.tma_layout(D, 1.0, 1.8, 'secondary_mag', m2, ...
                    'int_focus_m', -0.125*D, 'telecentric', true, 'stop', 'M2');
                tc.verifyLessThan(abs(i.chief_exit_slope), 1e-9, sprintf('m2 %.1f: chief exit slope %.3g', m2, i.chief_exit_slope));
                tc.verifyLessThan(abs(t(2) - R(3)/2), 1e-9, 'M2 must sit at M3''s front focus (t2 = R3/2)');
                tc.verifyTrue(isinf(i.exit_pupil_from_m3) || abs(i.exit_pupil_from_m3) > 1e6);
                tc.verifyLessThan(i.m3_z, i.int_focus_z, 'the telecentric Korsch has M3 in FRONT of the intermediate focus (virtual image)');
                tc.verifyTrue(all(R > 0) && all(t > 0));
            end
        end
        function test_stop_at_m1_telecentric_in_the_engine_and_default_reports_the_pupil(tc)
            D = 0.330/1.8;
            [~, ~, i0] = macos.design.tma_layout(D, 1.0, 1.8, 'secondary_mag', 3.5, 'int_focus_m', -0.125*D, 'm3_behind_m', 0.6*D);
            tc.verifyFalse(i0.telecentric);
            tc.verifyLessThan(abs(i0.chief_exit_slope + 21.91), 0.05, 'the default parent''s chief exit slope is pinned (-21.9 per unit field)');
            tc.verifyLessThan(abs(i0.exit_pupil_from_m3 - 0.0533), 5e-4, 'its exit pupil sits 53 mm after M3');
            [R1, t1, i1] = macos.design.tma_layout(D, 1.0, 1.8, 'secondary_mag', 3.5, 'int_focus_m', -0.125*D, 'telecentric', true);
            tc.verifyLessThan(abs(i1.chief_exit_slope), 1e-9);
            macos.init(128);
            d0 = tc.chiefs_(R1, t1, D);  tc.verifyLessThan(d0.dang, 1e-7, sprintf('telecentric: chief directions must be parallel (%.3e rad)', d0.dang));
            tc.verifyLessThan(abs(d0.efl - 0.330), 2e-3);  tc.verifyTrue(d0.all_ok);
            [R0, t0] = macos.design.tma_layout(D, 1.0, 1.8, 'secondary_mag', 3.5, 'int_focus_m', -0.125*D, 'm3_behind_m', 0.6*D);
            d1 = tc.chiefs_(R0, t0, D);  tc.verifyGreaterThan(d1.dang, 1e-3, 'the default layout is not telecentric (control)');
        end
    end
    methods (Access = private)
        function d = chiefs_(~, R, t, D)
            tel = macos.design.Telescope('family','TMA','aperture_diameter_m',D,'wavelength_m',633e-9,'model_size',128);
            tel.add_mirror('M1','radius_m',R(1),'spacing_after_m',t(1));
            tel.add_mirror('M2','radius_m',R(2),'spacing_after_m',t(2),'convex',true);
            tel.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
            tel.add_focal_plane('FP');  tel.build();
            nE = numel(tel.spec.elt);  th = 0.02*pi/180;  dd = zeros(3,2);  pp = zeros(3,2);  ok = true;
            for j = 1:2
                tel.trace_at_field([(2*j-3)*th 0]);  s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);
                dd(:,j) = ri.dir(:,1);  pp(:,j) = ri.pos(:,1);  ok = ok && all(ri.ok_trace & ri.ok_pass);
            end
            tel.trace_at_field([]);
            d = struct('dang', norm(cross(dd(:,1), dd(:,2))), 'efl', abs(pp(1,2)-pp(1,1))/(2*th), 'all_ok', ok);
        end
    end
end
