classdef tCheckClippingSource < matlab.unittest.TestCase
%TCHECKCLIPPINGSOURCE  check_clipping's incoming beam starts where the beam is.
%   PLAN_CONSOLIDATION 1.2 (ffff), 2026-10-29.  check_clipping builds 3-D beam
%   segments from the two DRAW meridian fans; each point's off-plane
%   coordinate is the beam center at the crossed element.  The SOURCE point
%   (elt 0) had no center and got 0, so on an off-axis section the x-fan's
%   incoming segment ran from the AXIS at the source to the beam at M1 -- a
%   slanted phantom that crossed M2's plane 0.42 m from M2's center.  On
%   tma_offaxis at 4.36 m decenter it read M2's clearance as 0.091 m (the
%   real incoming-beam edge is 0.331 m away), so set_offaxis('all') pushed
%   the decenter to 6.40 m and the design lost diffraction-limited
%   correction.  Bisected to resources c4f7780 (2026-07-05), whose shorter
%   source standoff (16.5 -> 8.8 m) steepened the phantom past the margin;
%   the 0 itself dates from bc21082 (2026-06-21).
%
%   Checks:
%     (a) M2's reported clearance equals the clearance against the real
%         collimated incoming beam, from plane geometry: the lowest incoming
%         ray's height minus M2's footprint center, projected into M2's
%         tilted plane, minus the footprint radius (the old code: 0.091 m
%         against ~0.33 m)
%     (b) set_offaxis('all') on the same layout lands at the clearance the
%         geometry predicts, not at the phantom's
%   Measured 2026-10-29 on this seed layout (K from the build, no optimize):
%   fixed -- M2 clearance at 4.36 m 0.3496 m (geometry 0.3481), set_offaxis
%   ('all') 4.339 m with M2 at 0.3306 m against the 0.3303 m margin; old code
%   -- 0.0954 m and 6.367 m, both checks fail.
%
%   See also: Telescope.check_clipping, Telescope.set_offaxis, tma_offaxis.

    properties (Constant)
        D     = 6.605
        R     = [15.879722 1.778913 3.016227]   % tma_offaxis (j18mono) |radii|
        TBET  = [7.169041556 7.965313479]
        DEC   = 4.36                            % m, the June design's decenter
    end

    methods (TestClassSetup)
        function setup(tc)
            here = fileparts(mfilename('fullpath'));
            addpath(fullfile(fileparts(here), 'src'));
        end
    end

    methods (Test)
        function test_m2_clearance_is_against_the_real_incoming_beam(tc)
            t = tc.layout();
            t.set_offaxis('none', 'dist', tc.DEC);
            t.build('', 'init', false);
            rep = t.check_clipping('noload', true, 'quiet', true);
            k   = find(strcmp({rep.name}, 'M2'));
            nE  = numel(t.spec.elt);
            b   = macos.draw_rays('YZ', 0, nE);           % y-fan: V = y, U = z
            ok  = b.nper(:).' >= 1;
            y_edge = min(b.V(1, ok));                     % lowest incoming ray
            m2  = (b.elt == k);
            cy  = mean(b.V(m2));                          % M2 footprint center (y)
            p   = t.spec.elt(k).psi(:);  p = p / norm(p);
            dy  = y_edge - cy;
            expect = dy * sqrt(1 + (p(2)/p(3))^2) - rep(k).foot_r;
            tc.verifyGreaterThan(y_edge, cy, 'the incoming beam must pass M2 on the +y side');
            tc.verifyEqual(rep(k).clearance, expect, 'AbsTol', 2e-3, sprintf( ...
                ['M2 clearance %.4f m, geometry says %.4f m: the incoming segment ' ...
                 'must start at the beam, not on the axis'], rep(k).clearance, expect));
        end

        function test_set_offaxis_lands_at_the_geometric_clearance(tc)
            t = tc.layout();
            d = t.set_offaxis('all');
            margin = 0.05 * tc.D;
            t.build('', 'init', false);
            rep = t.check_clipping('noload', true, 'quiet', true);
            c   = rep(strcmp({rep.name}, 'M2')).clearance;
            % the bisection returns the cleared side within 1e-4*D
            tc.verifyGreaterThanOrEqual(c, margin - 1e-6);
            tc.verifyLessThan(c, margin + 2e-3, sprintf( ...
                'M2 clears by %.4f m at %.3f m: the solve overshot the %.4f m margin', c, d, margin));
            tc.verifyLessThan(d, 5.0, sprintf( ...
                'decenter %.3f m: the phantom incoming segment is back (it gave 6.40 m)', d));
        end
    end

    methods
        function t = layout(tc)
            t = macos.design.Telescope('family','TMA','aperture_diameter_m',tc.D, ...
                    'model_size',128,'wavelength_m',2.3e-6,'grid_npts',41);
            t.add_mirror('M1','radius_m',tc.R(1),'spacing_after_m',tc.TBET(1));
            t.add_mirror('M2','radius_m',tc.R(2),'spacing_after_m',tc.TBET(2),'convex',true);
            t.add_mirror('M3','radius_m',tc.R(3),'spacing_after','derive');
            t.add_focal_plane('FP');
            t.build();
        end
    end
end
