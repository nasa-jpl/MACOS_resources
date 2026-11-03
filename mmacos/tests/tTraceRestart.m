classdef tTraceRestart < matlab.unittest.TestCase
%TTRACERESTART  A trace in ONE call equals the same trace element by element,
%   and both are right, on a deck whose entrance-pupil Reference sits just
%   ahead of a convex hyperboloid.
%
%   TO's dyson5 beat 5c finding (2026-10-03, repro_trace_onecall.m): on the
%   R2c Schwarzschild, macos.trace(5) gave a 93 mm rms spot while
%   trace(1)..trace(5) gave 10.6 um and matched the exact chain per ray.  Two
%   engine defects (surfsub.F / tracesub.F, fixed the same day):
%     * the ifLNsrf root pick (a surface right after a Reference / Return,
%       where a NEGATIVE L is allowed for the FEX exit-pupil leg) chose the
%       conic root by |L^2 - mpr| proximity, mpr = |pin - pv|^2 -- the ray's
%       distance to the VERTEX, lateral height included.  5.3 mm behind M1
%       and 90 mm off axis, mpr ~ h^2 made the root BEHIND the ray (-0.107 m)
%       "closer" than the real hit (+0.011 m).  LNsrfRoot now picks the root
%       whose hit point is nearest the element's reference point (RptElt: the
%       vertex, or an off-axis section's pole), which also makes the FEX
%       sphere's +-R choice strict instead of a tie (FEX radii bit-identical
%       on e5hex1 / Cass / jwst / SegDemo3, measured) and keeps a 90-deg OAP
%       on the side its pole is on (tBench);
%     * a trace RESTARTED at element k (what the OPD command does when a
%       trace exists) never reset PrevNonSeg between rays, so every ray after
%       the first saw the previous ray's last element as "previous" and ran
%       with ifLNsrf false -- the accident that made the stepwise trace right.
%   Non-vacuity: the pre-fix engine gives 9.33e-2 m for the one-call spot on
%   this fixture (TO's number, reproduced on the CLI: RMS OPD 8.87e-2 m).
%   A third restart defect, the paired-Return parity (2026-10-09, PLAN_CONSOLIDATION
%   item 2), is gated by test_restart_at_the_first_return_of_a_pair.
    properties (Constant)
        ModelSize = 256
        RxName    = 'Rx_SchwarzschildEP.in'
        SrcPos    = [0; 0; -0.711826]
    end
    methods (Access = private)
        function [s, Q] = spot_(~, m, mode)
            m.load_rx(rx_fixture_path('Rx_SchwarzschildEP.in'));  nE = m.num_elt();
            m.stop(1);  m.set_src_fov('src_pos', [0; 0; -0.711826], 'src_dir', [0; 0; 1], 'zSrc', 1e22);  m.modify();
            if strcmp(mode, 'one')
                tr = m.trace(nE);
            else
                for ie = 1:nE, tr = m.trace(ie); end
            end
            ri = macos.get_ray_info(tr.nRays);
            ok = ri.ok_trace(:) & ri.ok_pass(:);  Q = ri.pos(:, ok);
            s = sqrt(mean(sum((Q - mean(Q, 2)).^2, 1)));
        end
    end
    methods (Test)
        function test_one_call_equals_stepwise_and_both_are_small(tc)
            m = macos.Session(tc.ModelSize);
            [s1, Q1] = tc.spot_(m, 'one');
            [s2, Q2] = tc.spot_(m, 'step');
            tc.verifyLessThan(s1, 2e-5, sprintf('one-call rms spot must be the 1e-5 m class, not 9e-2 (got %.3e m)', s1));
            tc.verifyEqual(size(Q1), size(Q2), 'both traces must keep the same rays');
            tc.verifyLessThan(max(abs(Q1(:) - Q2(:))), 1e-12, ...
                sprintf('one-call and stepwise ray positions must agree: max |dQ| = %.3e m', max(abs(Q1(:) - Q2(:)))));
        end
        function test_every_ray_hits_m1_ahead_of_the_pupil(tc)
            % The defect put M1's hit behind the pupil plane: every ray's
            % position at element 2 must lie downstream (z > the pupil's z).
            m = macos.Session(tc.ModelSize);
            m.load_rx(rx_fixture_path(tc.RxName));
            m.stop(1);  m.set_src_fov('src_pos', tc.SrcPos, 'src_dir', [0; 0; 1], 'zSrc', 1e22);  m.modify();
            tr = m.trace(2);
            ri = macos.get_ray_info(tr.nRays);
            ok = ri.ok_trace(:) & ri.ok_pass(:);
            zEP = -0.305486443582606;
            tc.verifyTrue(all(ri.pos(3, ok) > zEP), sprintf('%d rays hit M1 behind the pupil plane', nnz(ri.pos(3, ok) <= zEP)));
        end
        function test_reference_types_after_a_pupil_take_the_near_root(tc)
            % PLAN_CONSOLIDATION item 4 (2026-10-09).  The ifLNsrf root pick (a surface right after a
            % Reference / Return, where a NEGATIVE L is allowed) moved to LNsrfRoot in the five base-conic
            % routines on 2026-10-03, but RefSrf / ObsSrf / PolElt (elemsub.F) and IntSrf (didesub.F) kept
            % the |L^2 - mpr| vertex-distance metric.  The Schwarzschild's M1 (5.3 mm behind the pupil
            % Reference, rays to 90 mm off axis) turned into each of those element types -- a Reference, an
            % Obscuring element, a TrPolarizer -- must be hit DOWNSTREAM of the pupil plane by every ray,
            % as the Reflector already is.  Pre-fix the metric picked the sheet behind the ray for the
            % off-axis rays.
            src = string(fileread(rx_fixture_path(tc.RxName)));
            i2 = strfind(src, "             iElt=  2");  i3 = strfind(src, "             iElt=  3");
            tc.assertNumElements(i2, 1);  tc.assertNumElements(i3, 1);
            head = extractBefore(src, i2);  blk = extractBetween(src, i2, i3 - 1);  tail = extractAfter(src, i3 - 1);
            old = ["          Element=  Reflector" + newline + "          Surface=  Aspheric", ...
                   "         AsphCoef=  6.025679492284151E+00  -3.579726675055740E+01  0.000000000000000E+00  0.000000000000000E+00  " + newline, ...
                   "           IndRef=  1.000000E+00" + newline + "           Extinc=  1.000000E+22"];
            for k = 1:numel(old), tc.assertEqual(count(blk, old(k)), 1, sprintf('element-2 anchor %d', k)); end
            types = {'Reference', '', 'Obscuring', '', 'TrPolarizer', "           PolAxis=  1  0  0" + newline};
            zEP = -0.305486443582606;
            wd = tempname;  mkdir(wd);  cln = onCleanup(@() rmdir(wd, 's'));
            m = macos.Session(tc.ModelSize);
            for t = 1:2:numel(types)
                b2 = replace(blk, old(1), "          Element=  " + types{t} + newline + "          Surface=  Conic");
                b2 = replace(b2, old(2), "");
                b2 = replace(b2, old(3), "           IndRef=  1.000000E+00" + newline + "           Extinc=  0.000000E+00" + newline + types{t+1});
                s2 = head + b2 + tail;
                p = fullfile(wd, [types{t} '.in']);  fid = fopen(p, 'w');  fprintf(fid, '%s', s2);  fclose(fid);
                m.load_rx(p);  m.stop(1);  m.set_src_fov('src_pos', tc.SrcPos, 'src_dir', [0; 0; 1], 'zSrc', 1e22);  m.modify();
                tr = m.trace(2);  ri = macos.get_ray_info(tr.nRays);
                ok = ri.ok_trace(:);
                tc.assertGreaterThan(nnz(ok), 100, sprintf('%s: rays reach element 2', types{t}));
                tc.verifyTrue(all(ri.pos(3, ok) > zEP), sprintf('%s: %d of %d rays hit element 2 behind the pupil plane', types{t}, nnz(ri.pos(3, ok) <= zEP), nnz(ok)));
            end
        end
        function test_restart_at_the_first_return_of_a_pair(tc)
            % PLAN_CONSOLIDATION item 2 (2026-10-09).  CTRACE's paired-Return parity (ifReturn: each
            % Return toggles it, a FocalPlane clears it; while set, the leg's path is SUBTRACTED) was
            % reset to .FALSE. on a RESTARTED trace, so a restart AT the first Return of a pair added the
            % back-traced exit-pupil leg instead -- geometry bit-identical, OPD wrong.  Measured pre-fix:
            % jwst zoom trace(26) then trace(27) 3.81e-5 m rms vs 7.01e-6 fresh (max |dOPD| 2.3e-4 m);
            % e5hex1 restart 11 -> 12 / 13 max |dOPD| 2.2e-4 m.  The (25, 27) pair is the must-pass
            % control: a restart that is not at a Return was always right.
            zoom = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'templates', '50_sensitivities', 'zoom_5x5', 'jwst_ote_designc.in');
            cases = {zoom, 26, 27; zoom, 25, 27; rx_fixture_path('e5hex1.in'), 11, 12; rx_fixture_path('e5hex1.in'), 11, 13};
            m = macos.Session(tc.ModelSize);
            for c = 1:size(cases, 1)
                [rx, k, j] = cases{c, :};
                m.load_rx(rx);  t0 = m.trace(j);  o0 = macos.opd();
                m.load_rx(rx);  m.trace(k);  t1 = m.trace(j);  o1 = macos.opd();
                v = isfinite(o0) & o0 ~= 0;
                [~, nm] = fileparts(rx);
                tc.verifyEqual(t1.rmsWFE, t0.rmsWFE, 'RelTol', 1e-12, sprintf('%s: trace(%d) then trace(%d) rms == the fresh trace(%d)', nm, k, j, j));
                tc.verifyLessThan(max(abs(o1(v) - o0(v))), 1e-15, sprintf('%s: restart %d -> %d OPD map == fresh', nm, k, j));
            end
        end
    end
end
