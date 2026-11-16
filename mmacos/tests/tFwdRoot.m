classdef tFwdRoot < matlab.unittest.TestCase
%TFWDROOT  A forward ray lands on the VERTEX sheet of a two-sheet conic.
%
%   dyson5 stage B (TO, addendum 38, 2026-10-04): `Telescope.optimize` on the
%   3k eccentric TMA section at -3 deg / 180 mm reported mm-scale per-field
%   WFE (8.5 / 6.8 / 3.9 mm) with every ray passing, while the trace's
%   best-focus spot (4-sigma outlier cut) read 400 um.  CALIB was faithful:
%   the plain OPD at the FP IS 7.8 mm rms on that deck, because the engine
%   put 40 of 1185 rays on the FAR sheet of M3 (K = -15.8, Kr = -0.1617 m,
%   sheets 2a = 21.9 mm apart): dz = +22.0 mm, 44 mm of extra path, a 10 mm
%   spot tail, all "passing".  The forward (.NOT.ifLNsrf) root pick in
%   ConSrf and its siblings chose between the quadratic's two roots by
%   |L^2 - mpr| proximity, mpr = |pin - pv|^2 -- the ray's distance to the
%   VERTEX, lateral height included, with no sense of which SHEET the hit
%   is on (the same metric the 2026-10-03 LNsrfRoot fix removed from the
%   ifLNsrf branch, where it had been left as "forward trains never reach
%   this").
%
%   FwdRoot (surfsub.F): the real surface is the sheet the vertex is on;
%   the normal's axial component Kr + (1+Kc) z has the sign of Kr there and
%   the opposite sign on the other sheet.  A forward root on the vertex
%   sheet is taken, first crossing on a tie; with none, the legacy pick is
%   kept to the bit.  "First forward crossing" alone is WRONG and was
%   measured so on this very deck: M2's far sheet (a bowl opening toward
%   M1, 37 mm in front of the mirror) is crossed by 125 rays BEFORE the
%   real surface -- min-positive put them there and lost 350 rays at M3.
%
%   Non-vacuity (pre-fix engine, same fixture, measured on the CLI and in
%   MATLAB): rms OPD at the FP 7.843e-3 m, spot tail 10.3 mm, 40 rays
%   22 mm off M3's near sheet, CALIB initial WFE 8.47e-3 m at the centre
%   field.  Post-fix: 5.79e-4 m, 2a residual 0 to 1e-6, flips counted.
    properties (Constant)
        ModelSize = 256
        RxName    = 'Rx_TwoSheetTMA.in'
        % M3 as the fixture declares it (KrElt, KcElt, VptElt z)
        M3Kr = -1.6171897517165229E-01
        M3Kc = -1.5769581197615901E+01
        M3Vz = -1.0412762581781022E-01
    end
    methods (Access = private)
        function [ri, tr] = trace_(tc, m, ie)
            m.load_rx(rx_fixture_path(tc.RxName));
            tr = m.trace(ie);  ri = macos.get_ray_info(tr.nRays);
        end
    end
    methods (Test)
        function test_every_ray_sits_on_m3_vertex_sheet(tc)
            % the near-sheet sag residual at every ray's M3 hit (was 22 mm
            % = 2a for 40 rays)
            m = macos.Session(tc.ModelSize);
            ri = tc.trace_(m, 3);
            ok = ri.ok_trace(:) & ri.ok_pass(:);
            p = ri.pos(:, ok);  h = hypot(p(1,:), p(2,:));  dz = p(3,:) - tc.M3Vz;
            R = tc.M3Kr;  K = tc.M3Kc;
            sag = h.^2 ./ (R*(1 + sqrt(1 - (1+K)*h.^2/R^2)));
            res = abs(dz - sag);
            tc.verifyEqual(nnz(ok), numel(ok), 'every ray passes this section');
            tc.verifyLessThan(max(res), 1e-6, ...
                sprintf('max |dz - sag| on M3 = %.3e m (pre-fix: 2.2e-2 m = 2a on 40 rays)', max(res)));
            n = mmacos('fwd_root_flips_get');
            tc.verifyGreaterThan(n, 0, 'the fixture must exercise the flip (else the gate is vacuous)');
            tc.verifyLessThan(n, numel(ok)/10, sprintf('%d flips: only the far-sheet rays move', n));
        end
        function test_fp_opd_is_sub_mm_and_the_spot_has_no_tail(tc)
            m = macos.Session(tc.ModelSize);
            ri = tc.trace_(m, 4);
            W = macos.opd();  w = W(W ~= 0);
            tc.verifyLessThan(std(w), 1e-3, sprintf('rms OPD at the FP %.3e m (pre-fix 7.843e-3)', std(w)));
            P = ri.pos;  ch = ri.dir(:, 1);
            a = P - ch*(ch'*P);  a = a - median(a, 2);  r = vecnorm(a);
            tc.verifyLessThan(max(r), 3e-3, sprintf('spot tail %.2f mm (pre-fix 10.3 mm)', max(r)*1e3));
        end
        function test_elements_before_m3_are_bit_identical(tc)
            % M1 and M2 never flip: M2's far sheet is crossed first by 125
            % rays, and the sheet rule leaves them on the mirror
            m = macos.Session(tc.ModelSize);
            tc.trace_(m, 2);
            tc.verifyEqual(mmacos('fwd_root_flips_get'), 0, 'no flip through M2');
        end
        function test_must_not_change_twin(tc)
            % tTraceRestart's Schwarzschild (EP Reference ahead of a convex
            % hyperboloid, the ifLNsrf case): zero flips = bit-identical by
            % construction, and its 1e-5 m spot still holds
            m = macos.Session(tc.ModelSize);
            m.load_rx(rx_fixture_path('Rx_SchwarzschildEP.in'));  nE = m.num_elt();
            m.stop(1);  m.set_src_fov('src_pos', [0; 0; -0.711826], 'src_dir', [0; 0; 1], 'zSrc', 1e22);  m.modify();
            tr = m.trace(nE);
            tc.verifyEqual(mmacos('fwd_root_flips_get'), 0, 'the Schwarzschild EP deck does not move');
            ri = macos.get_ray_info(tr.nRays);  ok = ri.ok_trace(:) & ri.ok_pass(:);  Q = ri.pos(:, ok);
            tc.verifyLessThan(sqrt(mean(sum((Q - mean(Q, 2)).^2, 1))), 2e-5, 'tTraceRestart''s spot still holds');
        end
        function test_calib_sees_every_field(tc)
            % the user-facing symptom: Telescope.optimize's per-field WFE on
            % the stage-B section (CALIB initial WFE 8.47e-3 m pre-fix)
            f = 330e-3;  D = f/1.8;  lam = 633e-9;
            macos.init(tc.ModelSize);
            [R, t] = macos.design.tma_layout(D, 1.0, 1.7462, 'secondary_mag', 3.5, 'int_focus_m', -0.125*D, 'telecentric', true);
            tel = macos.design.Telescope('family','TMA','aperture_diameter_m',D,'wavelength_m',lam,'model_size',tc.ModelSize);
            tel.add_mirror('M1','radius_m',R(1),'spacing_after_m',t(1));
            tel.add_mirror('M2','radius_m',R(2),'spacing_after_m',t(2),'convex',true);
            tel.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
            tel.add_focal_plane('FP');  tel.build();
            tel.set_field_bias(-3*60);  tel.set_offaxis('none', 'dist', 0.18);  tel.build();
            hs = 3000*(18e-6/f)/2;  fx = linspace(-hs, hs, 7);  fx = fx(fx ~= 0);
            rc = tel.optimize('fields', [fx(:) zeros(numel(fx), 1)], 'dofs', [0 0 0 0 0 0 0 1], 'max_iters', 1);
            tc.verifyTrue(all(rc.wfe_before < 1e-3), ...
                sprintf('CALIB per-field WFE (m): %s', mat2str(rc.wfe_before, 3)));
            tc.verifyTrue(all(rc.wfe_before > 1e-5), 'the section is not perfect either (a real evaluation)');
        end
    end
end
