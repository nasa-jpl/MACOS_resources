function W = pie_wedge_geom(a0, dth, rc, w, g, off, has_outer, has_center_hex)
%PIE_WEDGE_GEOM  Physical pie-wedge edge geometry with UNIFORM gaps.
%   W = macos.design.pie_wedge_geom(A0, DTH, RC, W_, G, OFF, HAS_OUTER,
%   HAS_CENTER_HEX) is the single source of truth for the pie-tiling
%   wedge boundary shared by macos.design.seg_boundary (tiling
%   overlay), macos.design.seg_apertures (emitted PolyApVec), and
%   macos.view_rx (segment tiles).
%
%   The wedge side edges are lines PARALLEL to the sector boundary
%   rays, offset toward the wedge interior by go = g/2 - off -- the
%   inter-segment gap is a uniform-width slot, NOT an angular gap that
%   converges at the tiling center (Dave 2026-07-18).  Ring 1 facing
%   the center hexagon takes a straight CHORD inner edge perpendicular
%   to the wedge bisector at d = (w+g)/2 - off (hexagon flat apothem
%   (w-g)/2 + gap g = uniform); deeper rings take an inner arc.  The
%   resulting ring-1 wedge (outer arc + two parallel side edges +
%   chord) is CONVEX -- a disc intersected with three half-planes --
%   so it is directly emittable as a convex PolyApVec with NO
%   obscuration.
%
%   Inputs: A0 wedge bisector azimuth, DTH ring angular pitch, RC ring
%   center radius, W_ radial band width, G gap, OFF outward clearance
%   (pad; 0 = physical edge), HAS_OUTER = a ring exists outside this
%   one (outer edge carries g/2), HAS_CENTER_HEX = ring 1 abutting the
%   center hexagon (chord inner edge).
%
%   W fields:
%     go        side-edge perpendicular offset (g/2 - off)
%     ro        outer arc radius
%     th1, th2  outer-arc azimuth span (side-line/arc intersections)
%     er, et    bisector radial/tangential unit vectors
%     d, A, B   ring 1 only: chord distance + chord-x-side vertices
%               (A at the low-azimuth side, B at the high side)
%     ri, ti1, ti2  deeper rings: inner arc radius + azimuth span
%     X         intersection point of the two offset side lines (the
%               convex apex for deeper-ring sector apertures)
%
%   All quantities are 2-D in the tiling plane about the tiling center.
%
%   See also: macos.design.seg_boundary, macos.design.seg_apertures,
%             macos.view_rx.
arguments
    a0  (1,1) double
    dth (1,1) double {mustBePositive}
    rc  (1,1) double {mustBePositive}
    w   (1,1) double {mustBePositive}
    g   (1,1) double
    off (1,1) double
    has_outer      (1,1) logical
    has_center_hex (1,1) logical
end
go = g/2 - off;
ro = rc + w/2 + off;
if has_outer, ro = ro - g/2; end
b1 = a0 - dth/2;  b2 = a0 + dth/2;
er = [cos(a0); sin(a0)];  et = [-sin(a0); cos(a0)];
W = struct('go', go, 'ro', ro, 'er', er, 'et', et, ...
           'th1', b1 + asin(min(max(go/ro, -1), 1)), ...
           'th2', b2 - asin(min(max(go/ro, -1), 1)), ...
           'd', [], 'A', [], 'B', [], 'ri', [], 'ti1', [], 'ti2', [], ...
           'X', (go/sin(dth/2)) * er);
if has_center_hex
    % chord inner edge: p . er = d, intersected with the two offset
    % side lines p = t*e_side + go*n_side (n_side = interior normal)
    d  = (w + g)/2 - off;
    t1 = (d - go*sin(dth/2)) / cos(dth/2);
    W.d = d;
    W.A = t1*[cos(b1); sin(b1)] + go*[-sin(b1); cos(b1)];
    W.B = t1*[cos(b2); sin(b2)] + go*[ sin(b2); -cos(b2)];
else
    ri = max(rc - w/2 + g/2 - off, 1e-9);
    W.ri  = ri;
    W.ti1 = b1 + asin(min(max(go/ri, -1), 1));
    W.ti2 = b2 - asin(min(max(go/ri, -1), 1));
end
end
