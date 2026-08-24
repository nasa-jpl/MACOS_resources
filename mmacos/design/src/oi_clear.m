function [dmin, d] = oi_clear(X, G, P, offset_deg)
%OI_CLEAR  Beam-leg vs element clearances of the box-centre field.
%
%   [DMIN, D] = OI_CLEAR(X, G, P, OFFSET) traces THREE field bundles
%   (box centre + the two YAN extremes -- the y-z blockers) through
%   every element and measures, for each of the NINE ordered
%   leg/obstacle pairs, the minimum distance from any field's leg
%   segments to the obstacle's GLASS: the union of all three fields'
%   patches (per field: a disk, footprint centre + 1.15x radius -- the
%   model of record -- or, with P.clear_footprint = 'hull', the
%   1.15-scaled convex hull of the footprint, tighter on elongated
%   off-axis patches; the hardware must carry every field either way).
%   The FP counts as an obstacle
%   for legs that do not end on it (an image plane sitting in the
%   incoming corridor is exactly as unbuildable as a mirror there).
%   Centre-field-only measurement is NOT enough: the rodgers3 S4
%   blockage that motivated this (M3->FP beam through M2) is carried by
%   the EDGE fields while the centre field clears by 48 mm.
%
%     legs:   in->M1, M1->M2, M2->M3, M3->FP
%     pairs:  in->M1 x {M2,M3,FP};  M1->M2 x {M3,FP};
%             M2->M3 x {M1,FP};     M3->FP x {M1,M2}
%
%   D is the fixed-length 9-vector (m, order above); DMIN = min(D).
%   SIGNED: a leg that PIERCES an obstacle returns MINUS the deepest
%   penetration (crossing point to patch boundary), so the solve hinge
%   max(0, dreq - d) keeps a gradient while blocked -- a zero-at-pierce
%   measure is flat under small pokes and the solver never moves (the
%   t4-wide lesson, 2026-08-20).  Positive = true clearance.
%   The stop (a Reference aperture, not glass) is not an obstacle.
%
%   This is the SOLVE-side model backing the S4/S5 clearance hinge rows
%   (per-element patches about their own centres -- much tighter than
%   OI_GATES' report-side union disks).  Failure to trace returns
%   DMIN = 0 (a candidate that cannot trace has no clearance).
%
%   PROMOTED to design/src (Dave, 2026-08-20) from
%   templates/10_telescopes/offset_imager -- generic to any three-mirror
%   train scored by OI_SCORE, and shared by its solve hinge rows and
%   report gates.  Scope: the element map (M1/M2/M3/FP = 1/3/4/5 with a
%   Reference stop at 2) and the nine ordered pairs are the three-mirror
%   layout's; an N-mirror generalization parameterizes ie/obst.  NOTE:
%   builds the deck via OI_DECK, which lives with the template -- callers
%   outside the template must have that directory on the path (its users
%   addpath it).
%
%   See also OI_SOLVE, OI_GATES, OI_SCORE, OFFSET_IMAGER.

    d = zeros(9,1);  dmin = 0;
    tmp = [tempname '.in'];
    cu  = onCleanup(@() delete_if_(tmp));

    D = X;
    D.EPD_m = P.EPD_m;  D.WL_m = P.lambda_m;
    D.sampling = P.sampling;  D.name = P.name;
    if isfield(P,'solve_sampling') && ~isempty(P.solve_sampling)
        D.sampling = P.solve_sampling;
    end
    txt = oi_deck(D);
    by = P.box_deg(2)/2;
    yans = offset_deg + [0, -by, +by];

    ie = [1 3 4 5];                       % M1 M2 M3 FP element ids
    % Full-grid positions + per-element ok MASKS (ray identity kept):
    % a leg pairs the SAME ray's crossings at its two ends, so its mask
    % is the AND of both elements' -- per-element masking alone breaks
    % (different counts) the moment rays are lost mid-train, which is
    % exactly when clearance is most in question.
    SF = cell(3,4);  OKm = cell(3,4);  S = cell(3,4);  cds = cell(1,3);
    for q = 1:3
        cdir = tancomp_(0, yans(q));  cds{q} = cdir;
        emit_src_(txt, tmp, seed_pos_(G, cdir), cdir);
        macos.load_rx(tmp);
        if ~macos.has_rx(), return; end
        macos.stop(2, [0 0]);
        % one traced pass records every station (engine RayPosHist)
        macos.ray_hist('on');
        tr = macos.trace(macos.num_elt());
        h = macos.ray_hist(tr.nRays);
        macos.ray_hist('off');
        for k = 1:4
            ok = h.ok(:, ie(k)+1);  ok(1) = false;
            if nnz(ok) < 5, return; end
            SF{q,k} = h.P(:, :, ie(k)+1);
            OKm{q,k} = ok(:).';
            S{q,k} = SF{q,k}(:, ok);      % per-element footprint (patches)
        end
    end

    % glass per element: one patch PER FIELD.  Model per P.clear_footprint:
    %   'disk' (record): footprint centre + 1.15x max radius
    %   'hull':          1.15-scaled convex hull of the footprint --
    %                    tighter where the off-axis patch is elongated
    %                    (a disk circumscribes it and over-forbids)
    mdl = 'disk';
    if isfield(P,'clear_footprint') && ~isempty(P.clear_footprint)
        mdl = P.clear_footprint;
    end
    nrm = { [0;0;1], [0;0;1], [0;0;1], G.fpa.psi(:)/norm(G.fpa.psi) };
    pat = cell(1,4);
    for k = 1:4
        pk = struct('C',{},'n',{},'r',{},'xa',{},'ya',{},'V2',{});
        for q = 1:3
            C = mean(S{q,k}, 2);
            r = 1.15 * max(vecnorm(S{q,k} - C, 2, 1));
            n = nrm{k};
            [~, i0] = min(abs(n));  xa = zeros(3,1);  xa(i0) = 1;
            xa = xa - (n.'*xa)*n;  xa = xa/norm(xa);  ya = cross(n, xa);
            V2 = [];
            if strcmp(mdl, 'hull')
                U2 = [xa.'; ya.'] * (S{q,k} - C);
                try
                    kk = convhull(U2(1,:).', U2(2,:).');
                    H2 = U2(:, kk(1:end-1));
                    c2 = mean(H2, 2);
                    V2 = 1.15*(H2 - c2) + c2;    % same margin as the disk
                catch                            % degenerate footprint
                    V2 = [];                     % falls back to the disk
                end
            end
            pk(q) = struct('C',C, 'n',n, 'r',r, 'xa',xa, 'ya',ya, 'V2',V2);
        end
        pat{k} = pk;
    end

    % nine pairs; each = min over (leg field) x (obstacle field disks).
    % Proximity sampling resolution: a quarter of the tightest clearance
    % requirement, so sampled minima are accurate at the gate knee.
    ds = 0.25 * min([P.clear_m(:); 0.020]);
    span = 1.2 * max(1e-3, abs(X.spacings(1)) + abs(X.spacings(3)));
    obst = { [2 3 4], [3 4], [1 4], [1 2] };
    j = 0;
    for L = 1:4
        for o = obst{L}
            j = j + 1;
            dm = inf;
            for q = 1:3
                if L == 1
                    m = OKm{q,1};
                    B = SF{q,1}(:,m);  A = B - span*cds{q};
                else
                    m = OKm{q,L-1} & OKm{q,L};   % same ray at BOTH ends
                    if nnz(m) == 0, continue; end
                    A = SF{q,L-1}(:,m);  B = SF{q,L}(:,m);
                end
                for pq = 1:3
                    dm = min(dm, seg_patch_min_(A, B, pat{o}(pq), ds));
                end
            end
            d(j) = dm;
        end
    end
    dmin = min(d);
end

% =========================================================================
function dm = seg_patch_min_(A, B, dk, ds)
%SEG_PATCH_MIN_  Min distance of leg segments A->B to one field patch:
%   the convex-hull polygon when dk.V2 is set, else the disk of record.
%
%   TWO-PART measure (the t4-wide lesson, 2026-08-20): the original 25
%   fixed samples spaced ~60 mm on a 1.4 m leg reported 7 mm "clearance"
%   for legs that PIERCE the glass between samples -- the S4/S5 hinge
%   rows were blind to real blockage and the report gate asserted a
%   buildability the layout figure plainly contradicted.
%     (1) EXACT PIERCING: every segment that crosses the patch plane is
%         tested AT its crossing point -- a leg through the glass is 0
%         at any sampling;
%     (2) proximity between the plane crossings is sampled at spacing
%         <= DS (a quarter of the tightest clearance requirement).
    use_hull = ~isempty(dk.V2) && size(dk.V2, 2) >= 3;

    % ---- (1) exact piercing test at the plane crossings -------------------
    % SIGNED at the crossings: a crossing INSIDE the patch returns MINUS
    % its in-plane distance to the patch boundary (penetration depth).
    % Zero-at-pierce is right for a gate but FLAT under small pokes --
    % the re-run of a fully blocked train converged without moving
    % because every hinge row was maxed out with no slope.  The signed
    % depth restores the gradient; oi_solve's deficit max(0, dreq - d)
    % consumes it unchanged.
    hA = dk.n.'*(A - dk.C);
    hB = dk.n.'*(B - dk.C);
    cx = find(hA.*hB < 0);
    if ~isempty(cx)
        sstar = hA(cx) ./ (hA(cx) - hB(cx));
        Q = A(:,cx) + (B(:,cx) - A(:,cx)) .* sstar;
        v = Q - dk.C;
        if use_hull
            [db, ins] = poly_bdist_([dk.xa.'; dk.ya.'] * v, dk.V2);
            sd = db;  sd(ins) = -db(ins);
        else
            sd = vecnorm(v - dk.n*(dk.n.'*v), 2, 1) - dk.r;
        end
        dm = min(sd);
        if dm < 0, return; end     % pierced: deepest penetration, signed
    else
        dm = inf;
    end

    % ---- (2) length-scaled proximity sampling -----------------------------
    ns = min(4001, max(51, ceil(max(vecnorm(B - A, 2, 1)) / ds)));
    for s = linspace(0, 1, ns)
        Q = A + s*(B - A);
        v = Q - dk.C;
        h = dk.n.'*v;
        if use_hull
            [db, ins] = poly_bdist_([dk.xa.'; dk.ya.'] * v, dk.V2);
            ro = db;  ro(ins) = 0;
        else
            Pp = v - dk.n*h;
            rr = vecnorm(Pp, 2, 1);
            ro = max(rr - dk.r, 0);
        end
        dm = min(dm, min(hypot(ro, abs(h))));
    end
end

function [db, inside] = poly_bdist_(U2, V2)
%POLY_BDIST_  Distance of points U2 (2,N) to the BOUNDARY of convex
%   polygon V2 (2,M, either winding) + inside flags.  Boundary distance
%   is valid on both sides: outside it is the clearance, inside the
%   penetration depth.
    N = size(U2, 2);  M = size(V2, 2);
    db = inf(1, N);
    inside = true(1, N);
    Vn = V2(:, [2:M 1]);
    sgn = sign(sum(V2(1,:).*Vn(2,:) - Vn(1,:).*V2(2,:)));   % winding
    if sgn == 0, sgn = 1; end
    for e = 1:M
        a = V2(:, e);  b = V2(:, mod(e, M) + 1);
        ab = b - a;
        ap = U2 - a;
        cr = ab(1)*ap(2,:) - ab(2)*ap(1,:);
        inside = inside & (sgn*cr >= -eps);
        t = max(0, min(1, (ab.'*ap) / max(ab.'*ab, eps)));
        E = a + ab.*t;
        db = min(db, vecnorm(U2 - E, 2, 1));
    end
end

function p = seed_pos_(G, cdir)
    cdR = [cdir(1); cdir(2); -cdir(3)];
    tq  = (G.z_m1 - G.stopC(3))/cdir(3);
    q   = G.stopC - tq*cdR;
    p   = q - (standoff_(G)/cdir(3))*cdir;
end

function s = standoff_(S)
%STANDOFF_  OI_STANDOFF off whatever aperture the struct carries; the
%   legacy 0.75 m when it carries none (a hand-built geometry struct).
    if isfield(S,'EPD_m') && ~isempty(S.EPD_m)
        s = oi_standoff(S.EPD_m);
    else
        s = 0.75;
    end
end

function emit_src_(txt0, tmp, p0, cdir)
    v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));
    s = regexprep(txt0, '(ChfRayDir=\s*)[^\n]*', ['$1' v3(cdir)]);
    s = regexprep(s,    '(ChfRayPos=\s*)[^\n]*', ['$1' v3(p0)]);
    fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
end

function d = tancomp_(xan_deg, yan_deg)
    d = [tand(xan_deg); tand(yan_deg); 1];
    d = d/norm(d);
end

function delete_if_(p), if exist(p,'file'), delete(p); end, end
