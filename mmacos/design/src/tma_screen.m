function S = tma_screen(phi, t, bfd, D, off_deg, opts)
%TMA_SCREEN  First-order (paraxial, engine-free) clearance screen of a coaxial three-mirror train.
%   S = tma_screen(PHI, T, BFD, D, OFF_DEG) evaluates the NINE leg x
%   obstacle clearances of OI_CLEAR at first order, in milliseconds, for a
%   coaxial three-mirror imager with the stop at M2's vertex:
%     PHI   [phi1 phi2 phi3] unfolded mirror powers 2/R (concave +), 1/m
%     T     [t1 t2] vertex spacings M1->M2, M2->M3 along the beam, m (> 0)
%     BFD   M3 -> image along the beam, m (> 0; the paraxial focus for an
%           imaging design)
%     D     entrance-pupil diameter (= the collimated beam), m
%     OFF_DEG  the along-track (y) field offset of the box centre, deg
%   The geometry is OI_CLEAR's: three field bundles at XAN = 0 (box centre
%   and the two YAN extremes, OPTS.by_deg each side), legs in->M1, M1->M2,
%   M2->M3, M3->FP, per-element glass = one disk per field, centred on the
%   footprint centre with OPTS.margin x its radius, in the element's plane
%   (normal z: the FP is taken normal to the axis), the stop not an
%   obstacle.  OI_CLEAR's measure exactly: a leg that crosses an obstacle
%   plane INSIDE a disk returns minus its in-plane distance to the disk
%   edge; otherwise the minimum 3-D distance leg -> disk.  At first order
%   the footprint centre is the chief height and its radius the axial
%   marginal height, so meridional rays (x = 0) carry every extreme: the
%   section is exact to first order, sag and higher-order terms ignored.
%   Folded frame = the template's: beam enters +z, M1 at OPTS.z_m1, M2 =
%   stop at z_m1 - t1, M3 at M2 + t2, FP at M3 - BFD; y is the offset axis.
%
%   Returns S: .d (9x1, m, OI_CLEAR order), .pairs, .dmin, .worst (pair
%   name), .y_c (4x3 chief heights M1 M2 M3 FP x fields, m), .y_m (1x4
%   marginal heights), .z (1x4 element z), .diam (1x3 mirror footprint
%   diameters incl. the cross-track field OPTS.xhalf_deg: [x y] extents,
%   max taken), .diam_xy (3x2), .len (z extent of the optics, m), .hgt (y
%   extent of mirrors + FP, m), .rmax (1x3, the footprint's farthest reach
%   from each vertex at the box corners, m) and .rho_R = rmax/|R| (a
%   beam reaching rho/|R| ~ 1 walks off the sphere -- the paraxial screen
%   does not model that; the engine misses there).
%
%   See also OI_CLEAR, TELESCOPE_SEED.
    arguments
        phi (1,3) double
        t (1,2) double
        bfd (1,1) double
        D (1,1) double
        off_deg (1,1) double
        opts.by_deg (1,1) double = 0.15
        opts.margin (1,1) double = 1.15
        opts.xhalf_deg (1,1) double = 0
        opts.z_m1 (1,1) double = 0.2
        opts.nrho (1,1) double = 41
    end
    pairs = {'in->M1 x M2','in->M1 x M3','in->M1 x FP', 'M1->M2 x M3','M1->M2 x FP', ...
             'M2->M3 x M1','M2->M3 x FP', 'M3->FP x M1','M3->FP x M2'};
    yans = off_deg + [0 -opts.by_deg opts.by_deg];
    z = [opts.z_m1, opts.z_m1 - t(1), opts.z_m1 - t(1) + t(2), opts.z_m1 - t(1) + t(2) - bfd];
    span = 1.2*(t(1) + t(2));
    rho = linspace(-1, 1, opts.nrho);
    % paraxial chief (through M2's vertex) + axial marginal, unfolded
    ym = trace_(phi, t, bfd, 0, D/2, 0);                 % axial marginal: y at M1 = D/2, u_in = 0
    yc = zeros(4, 3);  Y = cell(1, 3);
    for q = 1:3
        u0 = tand(yans(q));
        y1 = -t(1)*u0/(1 - t(1)*phi(1));               % hits M2 at y = 0
        yc(:, q) = trace_(phi, t, bfd, u0, y1, 0)';
        % every meridional ray: chief + rho x marginal (linear)
        Y{q} = yc(:, q) + ym(:)*rho;                    % 4 x nrho heights at M1 M2 M3 FP
        Y{q} = [Y{q}(1, :) - span*u0; Y{q}];            % prepend the incoming start (5 x nrho)
    end
    zz = [z(1) - span, z];                              % station z: start, M1, M2, M3, FP
    r = opts.margin*abs(ym);                            % disk radii M1 M2 M3 FP
    obst = {[2 3 4], [3 4], [1 4], [1 2]};
    d = zeros(9, 1);  j = 0;
    for L = 1:4
        for o = obst{L}
            j = j + 1;  dm = inf;
            for q = 1:3
                A = [Y{q}(L, :); repmat(zz(L), 1, opts.nrho)];
                B = [Y{q}(L+1, :); repmat(zz(L+1), 1, opts.nrho)];
                for pq = 1:3
                    dm = min(dm, seg_chord_(A, B, yc(o, pq), z(o), r(o)));
                end
            end
            d(j) = dm;
        end
    end
    [dmin, iw] = min(d);
    % sizes: footprint extent in y (3 fields) and x (cross-track +-xhalf)
    xc = abs(trace_(phi, t, bfd, tand(opts.xhalf_deg), -t(1)*tand(opts.xhalf_deg)/(1 - t(1)*phi(1)), 0));
    dxy = zeros(3, 2);
    for k = 1:3
        dxy(k, 1) = 2*xc(k) + 2*abs(ym(k));
        dxy(k, 2) = max(yc(k, :)) - min(yc(k, :)) + 2*abs(ym(k));
    end
    % how far the footprint reaches from each vertex at the box CORNERS
    % (cross-track +-xhalf x along-track extremes), against |R|: a
    % paraxial screen cannot see a beam that walks off a strongly curved
    % mirror (rho/|R| -> 1 is the hemisphere; the engine misses there)
    rmax = sqrt(xc(1:3).^2 + max(abs(yc(1:3, :)), [], 2)'.^2) + abs(ym(1:3));
    yall = [yc(1:3, :) + abs(ym(1:3))'; yc(1:3, :) - abs(ym(1:3))'; yc(4, :)];
    S = struct('d', d, 'pairs', {pairs}, 'dmin', dmin, 'worst', pairs{iw}, 'y_c', yc, 'y_m', ym, 'z', z, ...
               'diam_xy', dxy, 'diam', max(dxy, [], 2)', 'len', max(z) - min(z), 'hgt', max(yall(:)) - min(yall(:)), ...
               'rmax', rmax, 'rho_R', rmax.*abs(phi)/2);
end

function y = trace_(phi, t, bfd, u, y1, ~)
%TRACE_  Paraxial heights at M1, M2, M3, FP for a ray with height y1 at M1 and slope u before it.
    y = zeros(1, 4);  y(1) = y1;
    u = u - y1*phi(1);  y(2) = y1 + t(1)*u;
    u = u - y(2)*phi(2);  y(3) = y(2) + t(2)*u;
    u = u - y(3)*phi(3);  y(4) = y(3) + bfd*u;
end

function dm = seg_chord_(A, B, c, zo, r)
%SEG_CHORD_  OI_CLEAR's signed measure for meridional segments A->B (rows y, z) vs the disk (centre y = c, plane z = zo, radius r).
    hA = A(2, :) - zo;  hB = B(2, :) - zo;
    cx = hA.*hB < 0;
    dm = inf;
    if any(cx)
        s = hA(cx)./(hA(cx) - hB(cx));
        yq = A(1, cx) + s.*(B(1, cx) - A(1, cx));
        sd = abs(yq - c) - r;
        dm = min(sd);
        if dm < 0, return; end
    end
    % proximity: min distance of each segment to the disk's chord [c-r, c+r] at zo (exact 2-D segment-segment)
    for i = 1:size(A, 2)
        dm = min(dm, segseg_(A(:, i), B(:, i), [c - r; zo], [c + r; zo]));
    end
end

function d = segseg_(p1, p2, q1, q2)
    d = min([pseg_(p1, q1, q2), pseg_(p2, q1, q2), pseg_(q1, p1, p2), pseg_(q2, p1, p2)]);
end

function d = pseg_(p, a, b)
    ab = b - a;  tt = max(0, min(1, dot(p - a, ab)/max(dot(ab, ab), eps)));
    d = norm(p - (a + tt*ab));
end
