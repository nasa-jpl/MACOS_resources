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
    end
end
