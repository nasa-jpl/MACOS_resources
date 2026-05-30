function [inside, outside] = chk_polygon_pts(pts, bounds)
%CHK_POLYGON_PTS  Test whether all points lie inside / outside a polygon.
%   pts:    P×2 [x y]
%   bounds: N×3 [A B C] from POLY_LINES (one per edge)
%   inside  = true iff every pts is strictly inside the polygon
%   outside = true iff every pts is strictly outside the polygon
%
%   Polygon interior is defined by the side of each edge that the
%   reference point (0,0) sits on; same convention as pymacos's
%   tests/test_masks.py:chk_polygon_pts.
arguments
    pts    (:,2) double
    bounds (:,3) double
end
dx = 0;
dy = 0;
n_edges = size(bounds, 1);
states_all = false(size(pts, 1), 1);
for i = 1:n_edges
    a = bounds(i, 1);
    b = bounds(i, 2);
    c = bounds(i, 3);
    pts_ = pts(:, 1) * a + pts(:, 2) * b + c;
    pt_ref_ge = (dx * a + dy * b + c) >= 0;
    pt_ref_le = (dx * a + dy * b + c) <= 0;
    state = (pt_ref_ge & (pts_ < 0)) | (pt_ref_le & (pts_ > 0));
    if i == 1
        states_all = state;
    else
        states_all = states_all | state;
    end
end
inside  = all(~states_all);
outside = all(states_all);
end
