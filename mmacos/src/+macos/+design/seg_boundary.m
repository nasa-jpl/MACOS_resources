function B = seg_boundary(seg, off, opts)
%SEG_BOUNDARY  True per-segment boundary polylines for a segmented primary.
%   B = macos.design.seg_boundary(SEG, OFF) reconstructs each segment's
%   ACTUAL boundary in the tiling plane (SEG = macos.design.segment_rx
%   output), grown outward by OFF (same units; e.g. a launcher edge
%   clearance).  Three sources:
%
%     rxpoly  (AUTO when every segment block of seg.in carries a
%             PolyApVec polygonal aperture, e.g. from segment_rx's
%             emit_apertures) -- the boundary is the Rx-DECLARED
%             aperture polygon itself, minus its PolyObsVec obscuration
%             when one is present (a pie wedge's convex sector minus the
%             inner sector = the physical annular band), so launchers
%             land on the edges the prescription defines.  This is the
%             general case and covers imported segmented prescriptions.
%     Hex     regular hex tiles, apothem = width/2, ONE global clocking
%             for the whole tiling (flat normals = neighbor-center
%             directions; via macos.design.hex_tile)
%     Pie     center segment = a HEXAGON (the (X,L,R) hex-coordinate
%             tiling's central cell, verified against the traced ray
%             footprint -- NOT a disc), apothem (width-gap)/2; ring
%             wedges = radial band with the gap at INTERNAL shared
%             edges only (inner edge -gap/2 always; outer edge -gap/2
%             only when another ring sits outside; the tiling rim
%             carries no gap), angular span = ring pitch minus the gap
%             arc
%
%   B = seg_boundary(SEG, OFF, source="tiling"|"rxpoly"|"auto") forces
%   the source ("auto" default: rxpoly when declared, else the tiling).
%
%   Returns:
%     B.poly{s}   3 x M closed boundary polyline (global, first point
%                 repeated at the end)
%     B.u,B.v,B.n,B.c0   tiling-plane basis + origin (center-seg triad)
%     B.sample(s, n, phase)   n points equally spaced by ARC LENGTH
%                 along B.poly{s}, starting at fraction PHASE (0..1) of
%                 the perimeter -- grid-agnostic launcher placement
%     B.kind      'hex' | 'pie' | 'rxpoly'
%
%   See also: macos.design.hex_tile, macos.design.seg_apertures,
%             macos.design.add_met, macos.design.met_view.
arguments
    seg (1,1) struct
    off (1,1) double = 0
    opts.source (1,1) string {mustBeMember(opts.source, ...
        ["auto" "tiling" "rxpoly"])} = "auto"
end
fr = seg.frames;
n  = numel(fr);

% tiling-plane basis from the center segment (all tiles are coplanar to
% the tiling; per-segment zhat differences are the parent-surface tilt)
u = fr(1).xhat;  vN = fr(1).zhat;  v = cross(vN, u);
c0 = mean([fr.rpt], 2);

% --- rxpoly: the Rx-declared aperture polygons themselves ---------------
V = {};
if opts.source ~= "tiling" && isfield(seg, 'in') && isfile(seg.in)
    V = rxpolys_(seg);
end
if opts.source == "rxpoly" && isempty(V)
    error('macos:design:seg_boundary:rxpoly', ...
        ['source="rxpoly" but not every segment block of %s carries ' ...
         'a PolyApVec polygonal aperture'], seg.in);
end
if ~isempty(V)
    poly = cell(1, n);
    for s = 1:n
        P2 = [u.'; v.'] * (V{s}.ap - c0);
        shp = polyshape(P2(1,:), P2(2,:), 'Simplify', true);
        if ~isempty(V{s}.ob)
            O2 = [u.'; v.'] * (V{s}.ob - c0);
            shp = subtract(shp, polyshape(O2(1,:), O2(2,:), 'Simplify', true));
        end
        if off ~= 0, shp = polybuffer(shp, off); end
        [px, py] = boundary(shp, 1);
        P2 = [px.'; py.'];
        if any(vecnorm(P2(:,1) - P2(:,end)) > 1e-9)
            P2 = [P2, P2(:,1)]; %#ok<AGROW>
        end
        poly{s} = c0 + u*P2(1,:) + v*P2(2,:);
    end
    B = struct('poly', {poly}, 'u', u, 'v', v, 'n', vN, ...
               'c0', c0, 'kind', 'rxpoly');
    B.sample = @(s, np, phase) arcsample_(B.poly{s}, np, phase);
    return
end

% --- tiling reconstruction ----------------------------------------------
kind = 'hex';
if isfield(seg, 'grid'), kind = lower(char(seg.grid)); end
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
                % central cell of the hex-coordinate tiling: a hexagon
                % with flats facing the ring-1 wedge centers
                r1 = ~isctr & abs(rc - rings(1)) < 1e-6*max(rc);
                az = atan2(C2(2,r1), C2(1,r1));
                flat_ang = angle(mean(exp(1i*6*az))) / 6;
                a0h = (w - g)/2 + off;
                phic = flat_ang + pi/6 + (0:5)*pi/3;
                P2 = C2(:,s) + (a0h/cos(pi/6))*[cos(phic); sin(phic)];
                P2 = [P2, P2(:,1)]; %#ok<AGROW>
            else
                % gap at internal shared edges only; rim carries none
                ri = max(rc(s) - w/2 + g/2 - off, 1e-9);
                ro = rc(s) + w/2 + off;
                if any(rings > rc(s) + 1e-6*max(rc)), ro = ro - g/2; end
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
              'no boundary model for GridType %s (hex | pie | rxpoly)', kind);
end
B.sample = @(s, np, phase) arcsample_(B.poly{s}, np, phase);
end

% ---------------------------------------------------------------------------
function V = rxpolys_(seg)
%RXPOLYS_  Read per-segment PolyApVec (+PolyObsVec) polygons from seg.in.
%   Returns {} unless EVERY segment block carries a 3-D-vertex PolyApVec.
V = {};
lines = readlines(seg.in);
tl = strtrim(lines);
starts = find(startsWith(tl, "iElt="));
if isempty(starts) || max(seg.seg_elts) > numel(starts), return; end
out = cell(1, seg.nseg);
for s = 1:seg.nseg
    k = seg.seg_elts(s);
    b0 = starts(k);
    b1 = numel(lines); if k < numel(starts), b1 = starts(k+1) - 1; end
    ap = readpoly_(lines, tl, b0, b1, "PolyApVec=");
    if isempty(ap), return; end            % not fully declared -> tiling
    out{s} = struct('ap', ap, 'ob', readpoly_(lines, tl, b0, b1, "PolyObsVec="));
end
V = out;
end

function P = readpoly_(lines, tl, b0, b1, key)
P = [];
ip = find(startsWith(tl(b0:b1), key), 1) + b0 - 1;
if isempty(ip), return; end
nv = sscanf(regexprep(char(lines(ip)), '.*=', ''), '%d');
if isempty(nv) || nv < 3 || ip + nv > numel(lines), P = []; return; end
P = zeros(3, nv);
for q = 1:nv
    r = sscanf(char(lines(ip+q)), '%f');
    if numel(r) < 3, P = []; return; end   % 2-D form: not ours to read
    P(:, q) = r(1:3);
end
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
