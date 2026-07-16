function B = seg_boundary(seg, off)
%SEG_BOUNDARY  True per-segment boundary polylines for a segmented primary.
%   B = macos.design.seg_boundary(SEG, OFF) reconstructs each segment's
%   ACTUAL boundary in the tiling plane from the segmentation itself
%   (SEG = macos.design.segment_rx output), grown outward by OFF (same
%   units; e.g. a launcher edge clearance).  Supports both engine
%   tilings (manual 4.x, Figures 22/23):
%
%     Hex  regular hex tiles, apothem = width/2, ONE global clocking for
%          the whole tiling (flat normals = neighbor-center directions;
%          via macos.design.hex_tile)
%     Pie  annular wedges on rings (radial band = center radius +/-
%          width/2, angular span = ring pitch minus the gap arc) plus a
%          central disc when a segment sits on the axis; offset grows
%          radii by OFF and opens the angular edges by OFF/r
%
%   Returns:
%     B.poly{s}   3 x M closed boundary polyline (global, first point
%                 repeated at the end)
%     B.u,B.v,B.n,B.c0   tiling-plane basis + origin (center-seg triad)
%     B.sample(s, n, phase)   n points equally spaced by ARC LENGTH
%                 along B.poly{s}, starting at fraction PHASE (0..1) of
%                 the perimeter -- grid-agnostic launcher placement
%     B.kind      'hex' | 'pie'
%
%   PLANNED (Dave 2026-07-16): an 'rxpoly' source for segmentations
%   defined as .in-file polygonal apertures (ApType=Polygonal ApVec /
%   PolyApVec vertices on each segment element) -- the boundary then
%   comes from the Rx aperture definition itself, not the tiling
%   reconstruction, so launchers land on the edges the Rx declares.
%   Dispatch on aperture presence before the GridType branch when built.
%
%   See also: macos.design.hex_tile, macos.design.add_met,
%             macos.design.met_view.
arguments
    seg (1,1) struct
    off (1,1) double = 0
end
kind = 'hex';
if isfield(seg, 'grid'), kind = lower(char(seg.grid)); end
fr = seg.frames;
n  = numel(fr);

switch kind
    case 'hex'
        T = macos.design.hex_tile(seg, off);
        poly = cell(1, n);
        for s = 1:n
            P = T.corners{s};
            poly{s} = P(:, [1:end 1]);
        end
        B = struct('poly', {poly}, 'u', T.u, 'v', T.v, 'n', T.n, ...
                   'c0', T.c0, 'kind', 'hex');

    case 'pie'
        % tiling basis + in-plane centers
        u = fr(1).xhat;  vN = fr(1).zhat;  v = cross(vN, u);
        c0 = mean([fr.rpt], 2);
        C2 = [u.'; v.'] * ([fr.rpt] - c0);
        rc = vecnorm(C2);
        w  = seg.width;   g = 0;
        if isfield(seg, 'gap') && isfinite(seg.gap), g = seg.gap; end
        poly = cell(1, n);
        % ring membership by center radius
        isctr = rc < 1e-6 * max(rc);
        rings = uniquetol(rc(~isctr), 1e-6, 'DataScale', max(rc));
        for s = 1:n
            if isctr(s)
                % central disc: out to the first ring's inner edge - gap
                r0 = w/2 + off;
                if ~isempty(rings), r0 = min(r0, rings(1) - w/2 - g + off); end
                th = linspace(0, 2*pi, 73);
                P2 = C2(:, s) + r0*[cos(th); sin(th)];
            else
                ri = max(rc(s) - w/2 - off, 1e-9);
                ro = rc(s) + w/2 + off;
                % angular pitch of this ring
                m  = abs(rc - rc(s)) < 1e-6 * max(rc);
                nring = nnz(m);
                dth = 2*pi / nring;
                a0  = atan2(C2(2,s), C2(1,s));
                ha  = dth/2 - (g/2)/rc(s) + off/rc(s);   % half-span + offset
                tho = linspace(a0 - ha, a0 + ha, 25);    % outer arc
                thi = fliplr(tho);                       % inner arc (back)
                P2  = [ro*[cos(tho); sin(tho)], ri*[cos(thi); sin(thi)]];
                P2  = [P2, P2(:,1)]; %#ok<AGROW>
            end
            poly{s} = c0 + u*P2(1,:) + v*P2(2,:);
        end
        B = struct('poly', {poly}, 'u', u, 'v', v, 'n', vN, ...
                   'c0', c0, 'kind', 'pie');

    otherwise
        error('macos:design:seg_boundary:grid', ...
              'no boundary model for GridType %s (hex | pie)', kind);
end
B.sample = @(s, np, phase) arcsample_(B.poly{s}, np, phase);
end

% ---------------------------------------------------------------------------
function P = arcsample_(poly, np, phase)
%ARCSAMPLE_  np points equally spaced by arc length along a closed polyline.
d = [0, cumsum(vecnorm(diff(poly, 1, 2)))];
L = d(end);
t = mod(phase*L + (0:np-1)*L/np, L);
P = zeros(3, np);
for q = 1:np
    j = find(d <= t(q), 1, 'last');
    j = min(j, size(poly, 2) - 1);
    f = (t(q) - d(j)) / max(d(j+1) - d(j), eps);
    P(:, q) = poly(:, j) + f*(poly(:, j+1) - poly(:, j));
end
end
