function C = tls_clearance(G, M, P, opts)
%TLS_CLEARANCE  Does every beam leg clear every telescope body it does not traverse?
%
%   C = TLS_CLEARANCE(G, M, P) scores the ray paths TLS_MEASURE recorded
%   (every strip field; the rim of the bundle plus the chief) against the
%   bodies of the section G.  The rule is the dyson5 record's
%   (spectrometer_clearance): a mirror body is its lit footprint over ALL
%   strip fields grown by the mount margin (P.mount_m, default 5 mm), as an
%   ellipse in the section frame about the footprint centre, lifted onto the
%   mirror (the local quadric x^2/2R_s + y^2/2R_t toward the centre of
%   curvature -- the section's own first-order surface; the higher-order
%   sag is sub-mm over these footprints); the slit is a mask plate
%   (slit length + 10 mm) x 4 mm in the slit plane.  Each LEG (input ->
%   M1 -> M2 -> M3 -> slit) is scored against every body it neither starts
%   nor ends on: the clearance is the smallest distance from a ray segment
%   of the leg to the body's sampled surface, NEGATIVE when a segment
%   pierces the body (minus the depth inside its outline).
%
%   C.table rows (worst first): leg, body, clear_mm.  C.min_mm, C.pass.
%   Options: 'sample_m' (4e-3), 'nrim' (rays kept per field, 160).
arguments
    G struct
    M struct
    P struct
    opts.sample_m (1,1) double = 4e-3
    opts.nrim (1,1) double = 160
end
mount = 5e-3;  if isfield(P, 'mount_m'), mount = P.mount_m; end
nM = numel(G.m);  stn = [{'input'}, {G.m.name}, {'Slit'}];
% ---- bodies
B = struct('name', {}, 'station', {}, 'pts', {}, 'c', {}, 'n', {}, 'ex', {}, 'ey', {}, 'a', {}, 'b', {});
for k = 1:nM
    m = G.m(k);  f = M.foot(k);  Fr = m.frame;
    a = f.x_m/2 + mount;  b = f.y_m/2 + mount;
    c0 = [f.cx_m; f.cy_m];
    [u, v] = meshgrid(-a:opts.sample_m:a, -b:opts.sample_m:b);
    in = (u/a).^2 + (v/b).^2 <= 1;  u = u(in) + c0(1);  v = v(in) + c0(2);
    w = u.^2/(2*abs(m.Rs)) + v.^2/(2*abs(m.Rt));
    pts = m.pole + Fr(:, 1)*u.' + Fr(:, 2)*v.' + Fr(:, 3)*w.';
    B(end+1) = struct('name', m.name, 'station', k + 1, 'pts', pts, 'c', m.pole + Fr(:, 1:2)*c0, ...
                      'n', Fr(:, 3), 'ex', Fr(:, 1), 'ey', Fr(:, 2), 'a', a, 'b', b);   %#ok<AGROW>
end
fz = -G.slit.normal(:);  fx = [1; 0; 0];  fy = cross(fz, fx);
a = P.slit_m/2 + 5e-3;  b = 2e-3;
[u, v] = meshgrid(-a:opts.sample_m/2:a, -b:opts.sample_m/4:b);
B(end+1) = struct('name', 'SlitMask', 'station', nM + 2, 'pts', G.slit.point(:) + fx*u(:).' + fy*v(:).', ...
                  'c', G.slit.point(:), 'n', fz, 'ex', fx, 'ey', fy, 'a', a, 'b', b);
% ---- legs: station j -> j+1 (1 = input, 2..nM+1 = mirrors, nM+2 = slit)
rows = {};
for j = 1:nM + 1
    A = [];  Bq = [];
    for q = 1:numel(M.paths)
        pq = M.paths{q};  nr = size(pq, 2);
        r = vecnorm(pq(:, :, 2) - mean(pq(:, :, 2), 2));   % rim at M1
        [~, o] = sort(r, 'descend');  sel = unique([1, o(1:min(opts.nrim, nr))]);
        A = [A, pq(:, sel, j)];  Bq = [Bq, pq(:, sel, j + 1)];   %#ok<AGROW>
    end
    for k = 1:numel(B)
        if B(k).station == j || B(k).station == j + 1, continue; end
        cl = seg_body_(A, Bq, B(k));
        rows(end+1, :) = {sprintf('%s -> %s', stn{j}, stn{j+1}), B(k).name, cl*1e3};   %#ok<AGROW>
    end
end
[~, o] = sort(cell2mat(rows(:, 3)));  rows = rows(o, :);
C = struct('table', {rows}, 'min_mm', rows{1, 3}, 'pass', rows{1, 3} > 0, 'mount_m', mount);
end

function cl = seg_body_(A, Bq, bd)
% smallest distance from the segments A->B to the body samples; negative = pierces
D = Bq - A;  L2 = sum(D.^2, 1);
% piercing: the segment crosses the body's plane inside its ellipse
den = bd.n.'*D;  t = (bd.n.'*(bd.c - A))./den;
hit = abs(den) > 1e-15 & t > 0 & t < 1;
pen = 0;
if any(hit)
    Q = A(:, hit) + D(:, hit).*t(hit);
    e = sqrt(((bd.ex.'*(Q - bd.c))/bd.a).^2 + ((bd.ey.'*(Q - bd.c))/bd.b).^2);
    if any(e < 1), pen = -(1 - min(e))*min(bd.a, bd.b); end
end
if pen < 0, cl = pen; return, end
cl = inf;  S = bd.pts;  ns = size(S, 2);  chunk = max(1, floor(4e6/ns));
for i0 = 1:chunk:size(A, 2)
    ii = i0:min(size(A, 2), i0 + chunk - 1);
    a = A(:, ii);  d = D(:, ii);  l2 = L2(ii);
    % t* per (sample, segment)
    tt = ((S.'*d) - sum(a.*d, 1))./max(l2, 1e-30);  tt = min(max(tt, 0), 1);
    px = a(1, :) + d(1, :).*tt;  py = a(2, :) + d(2, :).*tt;  pz = a(3, :) + d(3, :).*tt;
    dd = (px - S(1, :).').^2 + (py - S(2, :).').^2 + (pz - S(3, :).').^2;
    cl = min(cl, sqrt(min(dd(:))));
end
end
