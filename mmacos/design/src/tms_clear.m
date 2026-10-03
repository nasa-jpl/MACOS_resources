function C = tms_clear(G, opts)
%TMS_CLEAR  Beam-leg vs body clearances of a two-mirror TMS chain (tms_geom), exact rays, signed.
%   C = tms_clear(G) traces, through the TMS chain G (tms_geom, placed or
%   not -- distances are frame-free), the box centre and its corners (the
%   cross-track ends x the along-track extremes about the bias), a ring of
%   marginal rays and the chief per field, and measures each leg against the
%   bodies it does not traverse:
%       in  (launch -> M1)  vs  M2, IMG
%       12  (M1 -> M2)      vs  IMG
%       2I  (M2 -> image)   vs  M1
%   Bodies (OI_CLEAR's model): a disc per mirror in its own plane (normal
%   psi), centred on the footprint centroid over every traced field, radius
%   opts.margin x the farthest footprint point; the image body is the image
%   footprint's disc grown by opts.img_pad (the slit / FPA package scale).
%   SIGNED: a leg crossing a disc INSIDE it returns minus its in-plane depth;
%   otherwise the minimum 3-D distance (sampled along the leg).
%   Options: 'by' (along-track half-width, rad; 0.15 deg), 'nring' (24),
%   'margin' (1.15), 'img_pad' (10 mm), 'lam' (1 um), 'standoff' (the in-leg
%   start: 1 m ahead of M1 along -d).
%   Returns C: .d (1x4, m: in-M2, in-IMG, 12-IMG, 2I-M1), .pairs, .min, .worst,
%   .bodies (C, n, r per body), .lost (rays that did not trace).
%
%   See also TMS_GEOM, OI_CLEAR, SPECTROMETER_CLEARANCE.
    arguments
        G struct
        opts.by (1,1) double = 0.15*pi/180
        opts.nring (1,1) double = 24
        opts.margin (1,1) double = 1.15
        opts.img_pad (1,1) double = 10e-3
        opts.lam (1,1) double = 1e-6
        opts.standoff (1,1) double = 1.0
    end
    pairs = {'in x M2', 'in x IMG', 'M1->M2 x IMG', 'M2->img x M1'};
    fov = G.src.fov;  D = G.src.D;  lam = opts.lam;
    ths = [0, -fov/2, fov/2];  dbs = [0, -opts.by, opts.by];
    % the stations: launch point, M1, M2, image (chain surfaces 2, 3, end)
    iM1 = find(strcmp({G.surf.name}, 'TelM1'));  iM2 = find(strcmp({G.surf.name}, 'TelM2'));  nS = numel(G.surf);
    L = struct('A', {}, 'B', {});  P1 = [];  P2 = [];  PI = [];  lost = 0;
    legs = cell(1, 3);
    for th = ths
        for db = dbs
            d = G.field_dir_raw(th, db);  d = d/norm(d);
            [p0, ok] = G.aim_pt(d, lam);  if ~ok, lost = lost + 1; continue, end
            u = cross(d, [1; 0; 0]);  if norm(u) < 1e-9, u = cross(d, [0; 1; 0]); end
            u = u/norm(u);  v = cross(d, u);
            starts = [p0, p0 + (D/2)*(cos(2*pi*(0:opts.nring-1)/opts.nring).*u + sin(2*pi*(0:opts.nring-1)/opts.nring).*v)];
            for k = 1:size(starts, 2)
                [pp, ~, okr] = G.trace(starts(:, k), d, lam);
                if ~okr, lost = lost + 1; continue, end
                a1 = pp(:, iM1);  a2 = pp(:, iM2);  ai = pp(:, nS);
                legs{1}(:, end+1, :) = cat(3, a1 - opts.standoff*d, a1);   %#ok<AGROW> in-leg (ahead of M1)
                legs{2}(:, end+1, :) = cat(3, a1, a2);                     %#ok<AGROW>
                legs{3}(:, end+1, :) = cat(3, a2, ai);                     %#ok<AGROW>
                P1(:, end+1) = a1;  P2(:, end+1) = a2;  PI(:, end+1) = ai; %#ok<AGROW>
            end
        end
    end
    B = struct('C', {}, 'n', {}, 'r', {});
    B(1) = disc_(P1, G.surf(iM1).psi, opts.margin, 0);
    B(2) = disc_(P2, G.surf(iM2).psi, opts.margin, 0);
    B(3) = disc_(PI, G.surf(nS).psi, 1, opts.img_pad);
    % pairs: in vs M2, in vs IMG, 12 vs IMG, 2I vs M1
    spec = [1 2; 1 3; 2 3; 3 1];
    dd = inf(1, 4);
    for j = 1:4
        Lg = legs{spec(j, 1)};  b = B(spec(j, 2));
        for k = 1:size(Lg, 2)
            dd(j) = min(dd(j), seg_disc_(Lg(:, k, 1), Lg(:, k, 2), b));
        end
    end
    [mn, iw] = min(dd);
    C = struct('d', dd, 'pairs', {pairs}, 'min', mn, 'worst', pairs{iw}, 'bodies', B, 'lost', lost);
end

function b = disc_(Pts, n, margin, pad)
    n = n(:)/norm(n);  c = mean(Pts, 2);
    rr = vecnorm((Pts - c) - n*(n'*(Pts - c)), 2, 1);
    b = struct('C', c, 'n', n, 'r', margin*max(rr) + pad);
end

function dm = seg_disc_(A, Bp, b)
%SEG_DISC_  Signed: a crossing inside the disc -> minus its depth; else the sampled minimum 3-D distance.
    hA = b.n'*(A - b.C);  hB = b.n'*(Bp - b.C);
    if hA*hB < 0
        q = A + (Bp - A)*(hA/(hA - hB));
        rad = norm((q - b.C) - b.n*(b.n'*(q - b.C)));
        if rad < b.r, dm = rad - b.r; return, end
    end
    dm = inf;
    for s = linspace(0, 1, 101)
        q = A + s*(Bp - A);  h = b.n'*(q - b.C);
        rad = norm((q - b.C) - b.n*h);
        dm = min(dm, hypot(max(rad - b.r, 0), h));
    end
end
