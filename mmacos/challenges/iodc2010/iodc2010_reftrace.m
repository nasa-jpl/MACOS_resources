function T = iodc2010_reftrace(L, P, p0, d, sph)
%IODC2010_REFTRACE  Independent exact ray trace of a green lens (the gate's analytic).
%   T = iodc2010_reftrace(L, P, p0, d, sph) traces the rays that START at the
%   object-space points p0 (3xN, all travelling along the unit vector d, a
%   collimated field) through the prescription L (n at P.lambda_nm from
%   iodc2010_glass, the stop plane as a passive plane in its own medium, the
%   image plane at P.z_img), and evaluates each ray's optical path to the
%   reference sphere sph (.C centre = the chief ray's image point, .R radius,
%   .V vertex: the exit-pupil sphere MACOS's FEX placed).  Nothing here calls
%   the engine; it is written from the textbook so the engine's wavefront is
%   argued against it, not transcribed from it.
%
%   Returns
%     T.opl   1xN optical path from the start points to the sphere (mm; the
%             start plane is perpendicular to d, so all rays share one object
%             wavefront)
%     T.W     1xN opl - opl(chief), chief = ray 1 (mm)
%     T.Q     3xN the point on the sphere
%     T.Pimg  3xN the point on the image plane;  T.Pstop 3xN on the stop plane
%     T.P1    3xN the first-surface crossing (to compare with the engine's)
%     T.dimg  3xN the final direction
%     T.lost  1xN true for a surface miss or total internal reflection
%
%   Recipe (Welford, Aberrations of Optical Systems, Sec. 4.2; Spencer & Murty
%   1962): with the vertex as origin and curvature c,
%       F = c (x^2+y^2+z^2) - 2 z,   G = d_z - c (x d_x + y d_y + z d_z),
%       s = F / (G + sqrt(G^2 - c F))        (the vertex-side crossing),
%       N = (-c x, -c y, 1 - c z)            (unit normal, +z at the vertex),
%   Snell in vector form  n' d' = n d + (n' cos I' - n cos I) N  with
%       cos I = d.N,  cos I' = sqrt(1 - (n/n')^2 (1 - cos^2 I)).
%   The sphere crossing after the image plane is the ray line's root nearer
%   the pupil vertex V (the ray's continuation when the pupil is virtual).
arguments
    L struct
    P struct
    p0 (3,:) double
    d (3,1) double
    sph struct
end
d = d / norm(d);
N = size(p0, 2);

% ---- surfaces in z order: lens surfaces, the stop plane, the image plane
S = struct('z', {}, 'c', {}, 'n2', {}, 'tag', {});
for k = 1:L.N
    S(end+1) = struct('z', L.z(k), 'c', P.c(k), 'n2', P.n_after(k), 'tag', 'surf'); %#ok<AGROW>
end
S(end+1) = struct('z', L.stop(1), 'c', 0, 'n2', NaN, 'tag', 'stop');   % n2 = the medium it sits in
S(end+1) = struct('z', P.z_img,   'c', 0, 'n2', 1,   'tag', 'img');
key = [S.z] + 1e-9 * strcmp({S.tag}, 'surf');
[~, ord] = sort(key);  S = S(ord);

opl  = zeros(1, N);  lost = false(1, N);
Q    = zeros(3, N);  Pimg = zeros(3, N);  Pstop = zeros(3, N);  P1 = zeros(3, N);
dimg = zeros(3, N);
for i = 1:N
    pt = p0(:, i);  dr = d;  n = 1;  L_i = 0;  ok = true;
    for j = 1:numel(S)
        c = S(j).c;
        x = pt(1);  y = pt(2);  z = pt(3) - S(j).z;
        Fv = c * (x^2 + y^2 + z^2) - 2 * z;
        Gv = dr(3) - c * (x * dr(1) + y * dr(2) + z * dr(3));
        disc = Gv^2 - c * Fv;
        if disc < 0 || (Gv + sqrt(disc)) == 0, ok = false; break; end
        s  = Fv / (Gv + sqrt(disc));
        pt = pt + s * dr;  L_i = L_i + n * s;
        x = pt(1);  y = pt(2);  z = pt(3) - S(j).z;
        switch S(j).tag
            case 'surf'
                if j == find(strcmp({S.tag}, 'surf'), 1), P1(:, i) = pt; end
                n2 = S(j).n2;
                if n2 ~= n
                    Nv   = [-c * x; -c * y; 1 - c * z];  Nv = Nv / norm(Nv);
                    cosI = dr.' * Nv;
                    kr   = n / n2;
                    sin2 = kr^2 * (1 - cosI^2);
                    if sin2 > 1, ok = false; break; end           % TIR
                    cosT = sqrt(1 - sin2);
                    dr   = kr * dr + (cosT - kr * cosI) * Nv;  dr = dr / norm(dr);
                    n    = n2;
                end
            case 'stop'
                Pstop(:, i) = pt;
            case 'img'
                Pimg(:, i) = pt;  dimg(:, i) = dr;
        end
    end
    if ~ok, lost(i) = true; continue; end
    % the reference sphere: |pt + s dr - C|^2 = R^2, root nearer the vertex V
    w  = pt - sph.C(:);
    b  = dr.' * w;
    cc = w.' * w - sph.R^2;
    rad = b^2 - cc;
    if rad < 0, lost(i) = true; continue; end
    s1 = -b + sqrt(rad);  s2 = -b - sqrt(rad);
    q1 = pt + s1 * dr;  q2 = pt + s2 * dr;
    if norm(q1 - sph.V(:)) <= norm(q2 - sph.V(:)), s = s1; Q(:, i) = q1; else, s = s2; Q(:, i) = q2; end
    opl(i) = L_i + n * s;
end
T = struct('opl', opl, 'W', opl - opl(1), 'Q', Q, 'Pimg', Pimg, 'Pstop', Pstop, ...
    'P1', P1, 'dimg', dimg, 'lost', lost);
end
