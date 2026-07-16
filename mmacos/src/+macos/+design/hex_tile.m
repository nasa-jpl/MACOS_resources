function T = hex_tile(seg, off)
%HEX_TILE  Boundary-true hex tile geometry for a segmented primary.
%   T = macos.design.hex_tile(SEG) reconstructs each segment's ACTUAL hex
%   boundary from the tiling itself (SEG = macos.design.segment_rx output):
%
%     - apothem a = width/2 (manual: width = flat-to-flat, gap = the
%       inter-segment spacing, neighbor center distance = width + gap);
%     - orientation is GLOBAL: a hex tiling has ONE clocking for every
%       tile, and adjacent hexes share a flat, so the flat normals are
%       the neighbor-center directions (taken as the consensus over all
%       nearest-neighbor pairs -- NOT each segment's own face-frame xhat,
%       which SegMirMaker clocks per segment).
%
%   T = macos.design.hex_tile(SEG, OFF) grows every boundary outward by
%   OFF (same units; e.g. a launcher edge-clearance).
%
%   Returns:
%     T.corners {nseg} of 3 x 6   global hex corners per segment
%     T.apothem                   width/2 + OFF
%     T.flat_ang                  tiling flat-normal angle (rad, in T.u/T.v)
%     T.u, T.v, T.n               tiling-plane basis (center-segment triad)
%     T.c0                        tiling-plane origin (mean segment center)
%     T.boundary(phi, s)          boundary distance from segment s's center
%                                 at tiling-plane angle phi (vectorized)
%
%   See also: macos.design.segment_rx, macos.design.met_view,
%             macos.design.add_met.
arguments
    seg (1,1) struct
    off (1,1) double = 0
end
fr = seg.frames;
n  = numel(fr);
if isfield(seg, 'width') && isfinite(seg.width)
    a = seg.width/2 + off;
else                                      % geometry-only seg structs
    a = median([fr.lmon]) + off;
end

% tiling-plane basis from the center segment (all tiles are coplanar to
% the tiling; per-segment zhat differences are the parent-surface tilt)
u = fr(1).xhat;  vN = fr(1).zhat;  v = cross(vN, u);
c0 = mean([fr.rpt], 2);
prj = @(P) [u.'; v.'] * (P - c0);
C2 = prj([fr.rpt]);                       % 2 x n in-plane centers

% consensus flat-normal angle from every nearest-neighbor direction
% (mod 60 deg -- all six flat normals are equivalent)
if n >= 2
    angs = [];
    D = squeeze(vecnorm(reshape(C2, 2, 1, n) - reshape(C2, 2, n, 1)));
    D(1:n+1:end) = inf;
    dmin = min(D(:));
    [ii, jj] = find(D < 1.05*dmin);
    for q = 1:numel(ii)
        d = C2(:, jj(q)) - C2(:, ii(q));
        angs(end+1) = mod(atan2(d(2), d(1)), pi/3); %#ok<AGROW>
    end
    % circular mean on the 60-deg torus
    flat_ang = angle(mean(exp(1i*6*angs))) / 6;
else
    flat_ang = 0;
end

% hexagon: flats at flat_ang + k*60deg, corners between the flats
phic = flat_ang + pi/6 + (0:5)*pi/3;
r    = a / cos(pi/6);
corners = cell(1, n);
for s = 1:n
    c = C2(:, s);
    P2 = c + r*[cos(phic); sin(phic)];
    corners{s} = c0 + u*P2(1,:) + v*P2(2,:);
end

T = struct('corners', {corners}, 'apothem', a, 'flat_ang', flat_ang, ...
           'u', u, 'v', v, 'n', vN, 'c0', c0);
T.boundary = @(phi, s) a ./ cos(mod(phi - flat_ang + pi/6, pi/3) - pi/6);
end
