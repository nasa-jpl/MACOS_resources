function K = pack_clear(deck, opts)
%PACK_CLEAR  Signed clearance floors of a loaded prescription, engine truth.
%
%   K = PACK_CLEAR(DECK) traces the deck and measures, for every ordered
%   (leg, obstacle) pair the layout admits:
%
%     LEG vs OPTIC BODY.  Minimum distance from the leg's traced ray
%     segments to the obstacle's BODY -- the BODY_K-scaled convex hull of
%     the measured footprint, grown by BODY_PAD, in the element's own plane.
%     The footprint is measured, the body model is stated: a mirror is
%     bigger than the light it catches, and a study that scores against the
%     LIT patch reports clearances no hardware will have.
%
%     A HULL AND NOT A DISK, and on this design the difference is the whole
%     answer.  Over a field box the collimator's footprint is nine
%     overlapping patches walking ~70 mm across it, so a centred disk of the
%     union's max radius fills in the middle -- exactly where this train's
%     feed beam passes -- and reports a 107 mm interference that is the
%     model's, not the design's.  (OI_CLEAR carries the same choice, for the
%     same reason.)
%
%     LEG vs LEG.  Minimum distance between the two bundles' RAY SEGMENTS,
%     over the rim rays (plus a coarse interior sample and the chief).
%     READ THIS ONE AS 'IS THERE ROOM FOR A FOLD', NOT AS AN INTERFERENCE:
%     light passes through light, and on a wide-field system the beams of
%     DIFFERENT FIELDS genuinely cross -- so a zero here says no flat can be
%     inserted between those two legs anywhere along them, not that the
%     design is blocked.  The leg-vs-BODY column is the interference one.
%
%   BOTH MEASURES ARE SAMPLING-FREE, and that is the point.  A fold is an
%   isometry, so every clearance must read the SAME before and after
%   folding -- and that is the only independent check this study has that
%   the fold really is one.  A station-sampled model cannot deliver it:
%   folding cuts one long leg into several short ones, so a fixed station
%   COUNT re-samples the same geometry at a different resolution, and even a
%   fixed station SPACING lands its stations at a different phase.  Measured
%   on the first two drafts: 10.6 mm vs 5.8 mm, then 8.4 mm vs 4.3 mm, on a
%   pair that an isometry cannot have moved.  Here the leg-body distance is
%   an exact ternary search (distance to a convex disk is convex along a
%   straight segment) and the leg-leg distance is the closed-form
%   segment-to-segment distance, so the two decks agree to round-off.
%
%   SIGNED for the body case, as in OI_CLEAR: a leg that pierces a body
%   returns MINUS the deepest penetration measured from the body rim, so a
%   blocked layout is ordered by how badly it is blocked instead of
%   collapsing to zero.  Rim rays cannot quantify a leg-leg
%   interpenetration, so that case reports the (small, positive) rim-to-rim
%   distance and is flagged by the floor rather than by a sign.
%
%   THROUGH-HOLES ARE A REQUIREMENT, NOT A COLLISION.  A two-mirror front
%   end sends its beam back through the primary, so 'leg pierces M1' is the
%   statement that M1 needs a hole -- true of the UNFOLDED design too, and
%   nothing to do with packaging.  Elements named in 'hole' report a hole
%   RADIUS requirement instead of a pierce.
%
%   THE INCOMING BEAM is checked separately: downstream of the point where
%   the train re-crosses the primary's plane, nothing -- body or beam -- may
%   sit in the cylinder of radius R_IN in front of it.  The front end is
%   excluded by construction, since that IS the incoming beam's business.
%   The absence of this clause is what retracted the S4 trade.
%
%   MEASURED OVER THE WHOLE FIELD BOX, not the deck's own field.  A
%   clearance is a hardware statement and the hardware has to carry every
%   field: on this design the box corners walk the beam to 80 mm off the
%   fold axis where the centre field shows 44 mm, so a centre-field
%   clearance table is optimistic by nearly a factor of two exactly where
%   the folds are tightest.  Every bundle below is the UNION over the field
%   set, and every footprint is the union footprint -- which is also the
%   right model for a body, since one mirror has to catch all of them.
%
%   Name-value:
%     'fields'    K x 2 field offsets, rad (default: the deck's own field
%                 only).  Pass the box corners for a hardware answer.
%     'body_k'/'body_pad'  body model (1.15, 0.015 m)
%     'hole'      element indices allowed a through-hole (default 1)
%     'r_in'      incoming-beam radius, m (default: element 1 body radius)
%     'rim_frac'  rays at >= this fraction of the stop radius are 'rim'
%                 (0.95);  'n_int' interior rays sampled beside them (64)
%     'init'/'quiet'
%
%   Returns K with .pair .hole .in_beam .floor_m .worst .foot_* .body_r and
%   .ap (per element: the beam's offset from the vertex and the clear
%   aperture the part must actually carry).
%
%   See also PACK_LEGS, PACK_FOLD, PACK_ROUTE, AFOCAL4_PACK, OI_CLEAR.

    arguments
        deck (1,:) char
        opts.body_k   (1,1) double = 1.15
        opts.body_pad (1,1) double = 0.015
        opts.hole     (1,:) double = 1
        opts.r_in     (1,1) double = -1
        opts.fields   (:,2) double = []
        opts.rim_frac (1,1) double = 0.95
        opts.n_int    (1,1) double = 64
        opts.n_rim    (1,1) double = 288
        opts.fold_elts (1,:) double = []
        opts.init     (1,1) logical = true
        opts.quiet    (1,1) logical = false
    end

    if opts.init, macos.load_rx(deck); end
    nE = macos.num_elt();
    [h, nR, nLost] = field_union_(deck, opts.fields);
    off = size(h.P,3) - nE;

    % ---- per-element footprint, body and required aperture ---------------
    C = zeros(3,nE);  Rf = zeros(1,nE);  Rb = zeros(1,nE);
    psi = zeros(3,nE);  V = zeros(3,nE);
    body = cell(1,nE);
    ap  = struct('elt',{},'off_m',{},'r_req_m',{});
    for k = 1:nE
        j = k + off;  m = h.ok(:,j);
        Q = squeeze(h.P(:,m,j));
        C(:,k)   = mean(Q,2);
        Rf(k)    = max(vecnorm(Q - C(:,k)));
        Rb(k)    = opts.body_k*Rf(k) + opts.body_pad;
        psi(:,k) = macos.get_elt_psi(k);
        V(:,k)   = macos.get_elt_vpt(k);
        body{k}  = hull_(Q, C(:,k), psi(:,k), opts.body_k);
        % what the PART has to be: the beam sits where the field bias puts
        % it, not on the vertex, so the clear aperture is measured about the
        % VERTEX (which is where a part is mounted) and not about the beam.
        d    = C(:,k) - V(:,k);
        doff = d - psi(:,k)*(psi(:,k).'*d);
        ap(end+1) = struct('elt',k, 'off_m',norm(doff), ...
                   'r_req_m', norm(doff) + Rf(k) + opts.body_pad); %#ok<AGROW>
    end
    r_in = opts.r_in;   if r_in < 0, r_in = Rb(1); end
    K = struct('deck',deck, 'nElt',nE, 'foot_c',C, 'foot_r',Rf, ...
               'body_r',Rb, 'psi',psi, 'vpt',V, 'ap',ap, ...
               'body_k',opts.body_k, 'body_pad',opts.body_pad, ...
               'hole_elts',opts.hole, 'r_in',r_in, 'body',[body{:}], ...
               'nRays',nR, 'nLost',nLost, 'nFields',max(1,size(opts.fields,1)));

    % ---- which rays carry the bundle boundary ---------------------------
    % The closest points of two disjoint convex bodies lie on their
    % boundaries, and a ray bundle's boundary is swept by the pupil RIM --
    % so rim rays are the ones that decide a clearance.  A coarse interior
    % sample rides along as a guard against a bundle whose boundary is not
    % the rim's image (a caustic, a vignetted patch).
    Q1  = squeeze(h.P(:,:,1+off));
    rr  = vecnorm(Q1 - mean(Q1(:,h.ok(:,1+off)),2));
    rim = find(rr >= opts.rim_frac*max(rr));
    nAll = size(h.P,2);
    step = max(1, floor(nAll/max(1,opts.n_int)));
    if numel(rim) > opts.n_rim
        rim = rim(round(linspace(1, numel(rim), opts.n_rim)));
    end
    use = unique([1, rim(:).', 1:step:nAll]);
    K.n_probe = numel(use);

    pair = struct('kind',{},'leg',{},'obst',{},'name',{},'d_m',{},'pierced',{});
    hole = struct('elt',{},'r_req_m',{},'leg',{});

    seg = cell(1,nE-1);   segp = cell(1,nE-1);
    for k = 1:nE-1
        ja = k+off;  jb = k+1+off;
        m  = h.ok(:,ja) & h.ok(:,jb);
        A  = squeeze(h.P(:,m,ja));  B = squeeze(h.P(:,m,jb));
        if size(A,1) ~= 3, A = A(:);  B = B(:); end
        seg{k} = struct('A',A, 'B',B);
        mu = m(use);                                   % probe rays that survive
        ii = cumsum(m);  idx = ii(use(mu));
        segp{k} = struct('A',A(:,idx), 'B',B(:,idx));
    end

    % ---- leg vs optic body -------------------------------------------------
    for k = 1:nE-1
        A = seg{k}.A;  B = seg{k}.B;
        Ap = segp{k}.A;  Bp = segp{k}.B;
        for e = 1:nE
            if e == k || e == k+1, continue; end
            [d, pierced, rmax] = seg_patch_(A, B, Ap, Bp, body{e}, opts.body_pad);
            if pierced && any(e == opts.hole)
                hole(end+1) = struct('elt',e, 'r_req_m',rmax + opts.body_pad, ...
                                     'leg',k); %#ok<AGROW>
                continue;
            end
            pair(end+1) = struct('kind','leg-body', 'leg',k, 'obst',e, ...
                'name',sprintf('leg %d-%d vs body %d', k, k+1, e), ...
                'd_m',d, 'pierced',pierced); %#ok<AGROW>
        end
    end

    % ---- leg vs leg ---------------------------------------------------------
    for k = 1:nE-1
        for j = k+2:nE-1
            d = bundle_gap_(segp{k}, segp{j});
            pair(end+1) = struct('kind','leg-leg', 'leg',k, 'obst',j, ...
                'name',sprintf('leg %d-%d vs leg %d-%d', k, k+1, j, j+1), ...
                'd_m',d, 'pierced',false); %#ok<AGROW>
        end
    end

    % ---- the incoming beam ---------------------------------------------------
    Pc = squeeze(h.P(:,1,:));       % ray 1 of the first field = its chief
    kx = 0;
    for k = 1:nE-1
        if Pc(3,k+off) < 0 && Pc(3,k+1+off) > 0, kx = k; end
    end
    bad = {};  worst = Inf;
    for e = kx+1:nE
        if C(3,e) < 0 && hypot(C(1,e),C(2,e)) - Rb(e) < r_in
            bad{end+1} = sprintf('body %d', e); %#ok<AGROW>
            worst = min(worst, hypot(C(1,e),C(2,e)) - Rb(e) - r_in);
        end
    end
    for k = kx+1:nE-1
        A = segp{k}.A;  B = segp{k}.B;
        ts = linspace(0,1,201);
        for i = 1:numel(ts)
            X = A + ts(i)*(B - A);
            m = X(3,:) < 0;
            if any(m)
                g = min(hypot(X(1,m),X(2,m))) - r_in;
                if g < 0
                    bad{end+1} = sprintf('leg %d-%d', k, k+1); %#ok<AGROW>
                    worst = min(worst, g);   break;
                end
            end
        end
    end
    K.k_recross = kx;
    K.in_beam = struct('ok', isempty(bad), 'who',{unique(bad)}, ...
                       'margin_m', tern_(isempty(bad), NaN, worst));

    K.pair = pair;  K.hole = hole;
    [K.floor_m, K.worst] = min([pair.d_m]);
    % Split the floor into the geometry that was already there and the
    % geometry the folds introduced.  A fold is an isometry, so the
    % PRE-EXISTING floor must read the same on the folded deck as on its
    % parent -- that is the study's independent check that the fold really
    % is null.  A new flat legitimately adds new pairs, and lumping them in
    % would make an honest isometry look like it moved something.
    %
    % CLASSIFY BY THE PARENT'S OWN PAIR SET, not by "does a fold appear in
    % it".  A fold SPLITS one leg into pieces, and a piece of a pre-existing
    % leg is still that leg: the pierce this design already had (its feed
    % beam against the collimator body) reads as 'leg F4-FM vs body M3'
    % after folding, and calling that fold-induced because F4 is a fold
    % would hide the one number the null is supposed to preserve.  So a pair
    % is PRE-EXISTING when its obstacle is not a fold body and, for a
    % leg-leg pair, the two legs come from ORIGINAL legs that were not
    % adjacent -- which is exactly the parent's pair set, mapped.
    K.fold_elts = opts.fold_elts;
    isfold = false(1,nE);   isfold(opts.fold_elts) = true;
    % ORIG(k) = how many non-fold elements sit at or before k, which is both
    % the original index of element k and the ORIGINAL LEG that folded leg k
    % is a piece of.
    orig = (1:nE) - cumsum(isfold);
    isf  = false(1,numel(pair));
    for i = 1:numel(pair)
        p = pair(i);
        if strcmp(p.kind,'leg-body')
            isf(i) = isfold(p.obst);
        else
            % the parent tests only NON-ADJACENT legs; a pair whose two
            % pieces come from the same or neighbouring original legs is
            % newly visible because the fold split them apart.
            isf(i) = orig(p.obst) <= orig(p.leg) + 1;
        end
    end
    K.is_fold_pair = isf;
    K.orig = orig;
    if any(~isf), [K.floor_pre_m,  K.worst_pre]  = min([pair(~isf).d_m]);
    else,          K.floor_pre_m = NaN;  K.worst_pre = []; end
    if any(isf),  [K.floor_fold_m, K.worst_fold] = min([pair(isf).d_m]);
    else,          K.floor_fold_m = NaN; K.worst_fold = []; end
    % And the same split by KIND, because the two kinds are different
    % questions.  A leg-BODY floor is an interference: hardware in the beam.
    % A leg-LEG floor only says whether a flat could be put between them --
    % two beams crossing in free space is ordinary (a Cassegrain's own beam
    % crosses its primary's plane), so a zero there is not a defect.
    isb = strcmp({pair.kind},'leg-body');
    K.floor_pre_body_m  = flr_(pair, ~isf &  isb);
    K.floor_fold_body_m = flr_(pair,  isf &  isb);
    K.floor_pre_leg_m   = flr_(pair, ~isf & ~isb);
    K.floor_fold_leg_m  = flr_(pair,  isf & ~isb);
    K.worst_body = wname_(pair, isb);
    if ~opts.quiet, report_(K); end
end

% =====================================================================
function [u, v] = perp_(a)
%PERP_  An orthonormal pair spanning the plane normal to A.
    a = a(:)/norm(a);
    [~, i] = min(abs(a));   u = zeros(3,1);  u(i) = 1;
    u = u - (u.'*a)*a;  u = u/norm(u);
    v = cross(a, u);
end

function H = hull_(Q, c, n, kscale)
%HULL_  The element's BODY: the convex hull of its measured footprint in its
%   own plane, scaled by KSCALE about the footprint centroid.  Returned as
%   the plane (c,n), an orthonormal in-plane basis, and the hull polygon in
%   that basis -- everything SEG_PATCH_ needs.
    n = n(:)/norm(n);
    [u, v] = perp_(n);
    w = [u.'*(Q - c); v.'*(Q - c)];
    if size(w,2) >= 3
        try
            i = convhull(w(1,:).', w(2,:).');
            poly = w(:,i);
        catch
            poly = w;                       % degenerate (collinear) footprint
        end
    else
        poly = w;
    end
    % Cap the vertex count.  A hull over ten thousand rays can carry
    % hundreds of nearly-collinear vertices, and the distance query is
    % O(edges x points); 64 is far finer than the 15 mm body pad.  Keep
    % the ORIGINAL vertices (a subsample of a convex polygon is inscribed,
    % i.e. slightly SMALLER than the true body) and then re-inflate to the
    % full hull radius so the body is never under-reported.
    nmax = 64;
    if size(poly,2) > nmax
        r0 = max(vecnorm(poly));
        poly = poly(:, round(linspace(1, size(poly,2), nmax)));
        poly = poly * (r0/max(vecnorm(poly)));
    end
    poly = kscale*poly;                     % about the centroid, which is 0 here
    H = struct('c',c, 'n',n, 'u',u, 'v',v, 'poly',poly, ...
               'r', max(vecnorm(poly)));
end

function [d, pierced, rmax] = seg_patch_(A, B, Ap, Bp, H, pad)
%SEG_PATCH_  Signed distance from a bundle of segments to a body patch.
%   PIERCE first, exactly: a segment crossing the patch plane INSIDE the
%   grown hull is a hit, and the answer is minus the deepest penetration
%   from the hull boundary.  RMAX is the farthest crossing from the patch
%   centroid (a through-hole radius).  Otherwise a ternary search in t --
%   the distance to a convex set is convex along a straight segment, so the
%   search is exact rather than a sampling.  The PIERCE test runs on every
%   ray (it is one vectorised plane crossing); the search runs on the probe
%   subset (Ap,Bp), which carries the bundle boundary and is where the
%   minimum lives.
    n = H.n;
    den = n.'*(B - A);
    num = n.'*(H.c - A);
    tt  = num./den;
    hit = abs(den) > 1e-14 & tt > 0 & tt < 1;
    pierced = false;  rmax = NaN;
    if any(hit)
        X  = A(:,hit) + tt(hit).*(B(:,hit) - A(:,hit));
        dp = poly_dist_(H, X) - pad;        % negative = inside the body
        if any(dp <= 0)
            pierced = true;
            d = min(dp);
            rmax = max(vecnorm(X - H.c));
            return;
        end
    end
    lo = zeros(1,size(Ap,2));   hi = ones(1,size(Ap,2));
    for it = 1:40
        m1 = lo + (hi-lo)/3;   m2 = hi - (hi-lo)/3;
        f1 = pt_patch_(Ap + m1.*(Bp-Ap), H, pad);
        f2 = pt_patch_(Ap + m2.*(Bp-Ap), H, pad);
        sw = f1 < f2;
        hi(sw)  = m2(sw);
        lo(~sw) = m1(~sw);
    end
    d = min(pt_patch_(Ap + 0.5*(lo+hi).*(Bp-Ap), H, pad));
end

function d = pt_patch_(X, H, pad)
    ax = H.n.'*(X - H.c);
    Xp = X - H.n*ax;
    dp = max(0, poly_dist_(H, Xp + H.n*(H.n.'*H.c)) - pad);
    d  = hypot(ax, dp);
end

function d = poly_dist_(H, X)
%POLY_DIST_  In-plane signed distance from points X to the hull polygon:
%   positive outside, negative inside (the depth past the boundary).
    w = [H.u.'*(X - H.c); H.v.'*(X - H.c)];
    P = H.poly;
    if size(P,2) < 2
        d = vecnorm(w - P(:,1));   return;
    end
    d = inf(1, size(w,2));
    for i = 1:size(P,2)-1
        a = P(:,i);  b = P(:,i+1);
        ab = b - a;   L2 = ab.'*ab;
        if L2 < eps, continue; end
        t  = max(0, min(1, (ab.'*(w - a))/L2));
        dx = w(1,:) - (a(1) + ab(1)*t);
        dy = w(2,:) - (a(2) + ab(2)*t);
        d  = min(d, hypot(dx, dy));
    end
    if size(P,2) >= 3
        in = inpolygon(w(1,:), w(2,:), P(1,:), P(2,:));
        d(in) = -d(in);
    end
end

function [h, nR, nLost] = field_union_(deck, F)
%FIELD_UNION_  Ray history over a FIELD SET, stacked along the ray index.
%   Each field is traced on its own re-pointed copy of the deck (chief
%   direction + yGrid, pivoting about the stop -- the AFOCAL_LADDER_DECK
%   recipe, so 'the field box' means the same thing here as in the scoring)
%   and the histories are concatenated.  Apertures are NOT stripped: a ray
%   the design itself vignettes is not a ray any hardware has to clear.
    if isempty(F)
        nE = macos.num_elt();
        macos.ray_hist('on');   t = macos.trace();   h = macos.ray_hist(t.nRays);
        macos.ray_hist('off');
        nR = t.nRays;   nLost = t.nRays - sum(h.ok(:,end));
        return;
    end
    txt = fileread(deck);
    cd0 = grab3_(txt,'ChfRayDir');   cp0 = grab3_(txt,'ChfRayPos');
    apst = grab3_(txt,'ApStop');
    stand = dot(apst - cp0, cd0);
    bx0 = asin(cd0(1));   by0 = asin(cd0(2));
    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>
    P = [];  OK = [];  nR = 0;  nLost = 0;
    for i = 1:size(F,1)
        bx = bx0 + F(i,1);   by = by0 + F(i,2);
        cdir = [sin(bx); sin(by); sqrt(max(0,1-sin(bx)^2-sin(by)^2))];
        cpos = apst - stand*cdir;
        s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3_(cdir)]);
        s = regexprep(s,   '(ChfRayPos=\s*)[^\n]*', ['$1' v3_(cpos)]);
        s = regexprep(s,   '(yGrid=\s*)[^\n]*', ['$1' v3_([0;cos(by);-sin(by)])]);
        fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
        macos.load_rx(tmp);
        macos.ray_hist('on');   t = macos.trace();   hi = macos.ray_hist(t.nRays);
        macos.ray_hist('off');
        P  = cat(2, P, hi.P);   OK = cat(1, OK, hi.ok);
        nR = nR + t.nRays;      nLost = nLost + t.nRays - sum(hi.ok(:,end));
    end
    % LOGICAL, explicitly.  Seeding the accumulator with [] makes cat return
    % a DOUBLE array of ones and zeros, and h.P(:,mask,j) then indexes BY
    % VALUE -- N copies of ray 1, silently, with the right size and a
    % plausible-looking centroid.  Cost: one full run of the study whose
    % every clearance was the chief ray's.
    h = struct('P',P, 'ok',logical(OK));
    macos.load_rx(deck);
end

function v = grab3_(txt, key)
    t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens', 'once');
    v = sscanf(strrep(t{1},'D','E'), '%f', 3);
end
function s = v3_(v),  s = sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));  end
function del_(p),  if exist(p,'file'), delete(p); end,  end

function [d, pierced, rmax] = seg_disk_(A, B, c, n, r)
%SEG_DISK_  Signed distance from a bundle of segments to a disk (c,n,r).
%   PIERCE first, exactly: a segment crossing the disk plane inside r is a
%   hit, and the answer is minus the deepest penetration from the disk rim.
%   RMAX = the farthest crossing from the disk centre (a through-hole
%   radius).  Otherwise a ternary search in t -- distance to a convex set is
%   a convex function of the point, and the segment is affine in t, so the
%   distance is convex in t and the search is exact, not a sampling.
    n   = n(:)/norm(n);
    den = n.'*(B - A);
    num = n.'*(c - A);
    tt  = num./den;
    hit = abs(den) > 1e-14 & tt > 0 & tt < 1;
    pierced = false;  rmax = NaN;
    if any(hit)
        X   = A(:,hit) + tt(hit).*(B(:,hit) - A(:,hit));
        rho = vecnorm(X - c);
        if any(rho <= r)
            pierced = true;
            d    = -max(r - rho(rho <= r));
            rmax = max(rho);
            return;
        end
    end
    lo = zeros(1,size(A,2));   hi = ones(1,size(A,2));
    for it = 1:80
        m1 = lo + (hi-lo)/3;   m2 = hi - (hi-lo)/3;
        f1 = pt_disk_(A + m1.*(B-A), c, n, r);
        f2 = pt_disk_(A + m2.*(B-A), c, n, r);
        s  = f1 < f2;
        hi(s)  = m2(s);
        lo(~s) = m1(~s);
    end
    d = min(pt_disk_(A + 0.5*(lo+hi).*(B-A), c, n, r));
end

function d = pt_disk_(X, c, n, r)
    w  = X - c;
    ax = n.'*w;
    rh = vecnorm(w - n*ax);
    d  = abs(ax);
    o  = rh > r;
    d(o) = hypot(ax(o), rh(o) - r);
end

function d = bundle_gap_(S1, S2)
%BUNDLE_GAP_  Minimum distance between two bundles of segments, closed form
%   (the standard clamped segment-segment distance), every pair.
    d = Inf;
    for i = 1:size(S1.A,2)
        d = min(d, min(seg_seg_(S1.A(:,i), S1.B(:,i), S2.A, S2.B)));
    end
end

function d = seg_seg_(p1, q1, P2, Q2)
%SEG_SEG_  Distance from one segment (p1,q1) to each of N segments (P2,Q2).
    d1 = q1 - p1;                       % 3x1
    d2 = Q2 - P2;                       % 3xN
    r  = p1 - P2;                       % 3xN
    a  = d1.'*d1;                       % scalar
    e  = sum(d2.^2, 1);                 % 1xN
    f  = sum(d2.*r, 1);                 % 1xN
    b  = d1.'*d2;                       % 1xN
    c  = d1.'*r;                        % 1xN
    den = a*e - b.^2;
    s = zeros(1,numel(e));   tt = zeros(1,numel(e));
    nd = den > 1e-30;
    s(nd)  = min(1, max(0, (b(nd).*f(nd) - c(nd).*e(nd))./den(nd)));
    s(~nd) = 0;
    tt = (b.*s + f)./max(e, 1e-30);
    lo = tt < 0;   hi = tt > 1;
    tt(lo) = 0;    tt(hi) = 1;
    s(lo)  = min(1, max(0, -c(lo)/a));
    s(hi)  = min(1, max(0, (b(hi) - c(hi))/a));
    W = (p1 + d1.*s) - (P2 + d2.*tt);
    d = vecnorm(W);
end

function report_(K)
    fprintf('\n  CLEARANCES  %s\n', K.deck);
    fprintf(['    body %.2f x footprint + %.0f mm; %d field(s), %d rays ' ...
             '(%d probes), %d lost\n'], K.body_k, K.body_pad*1e3, ...
            K.nFields, K.nRays, K.n_probe, K.nLost);
    fprintf('    %-4s %10s %10s %10s\n','elt','foot r mm','hull r mm','off vtx mm');
    for k = 1:K.nElt
        fprintf('    %-4d %10.1f %10.1f %10.1f\n', k, K.foot_r(k)*1e3, ...
                K.body(k).r*1e3, K.ap(k).off_m*1e3);
    end
    [~, ord] = sort([K.pair.d_m]);
    show = ord(1:min(10,numel(ord)));
    fprintf('    tightest pairs:\n');
    for i = show
        p = K.pair(i);
        fprintf('      %-26s %9.2f mm  %s\n', p.name, p.d_m*1e3, ...
                tern_(p.pierced, '<-- PIERCED', ''));
    end
    for i = 1:numel(K.hole)
        fprintf('    through-hole: leg %d-%d crosses elt %d -> hole radius >= %.1f mm\n', ...
                K.hole(i).leg, K.hole(i).leg+1, K.hole(i).elt, K.hole(i).r_req_m*1e3);
    end
    if K.in_beam.ok
        fprintf('    incoming beam (r %.0f mm, z<0, after leg %d): clear\n', ...
                K.r_in*1e3, K.k_recross);
    else
        fprintf('    incoming beam: OCCUPIED by %s (%.0f mm inside)\n', ...
                strjoin(K.in_beam.who,', '), -K.in_beam.margin_m*1e3);
    end
    fprintf('    FLOOR %.2f mm  (%s)\n', K.floor_m*1e3, K.pair(K.worst).name);
    fprintf(['      body floors: pre-existing %8.2f mm, fold-induced %8.2f mm ' ...
             '(worst body pair: %s)\n'], K.floor_pre_body_m*1e3, ...
            K.floor_fold_body_m*1e3, K.worst_body);
    fprintf(['      leg-leg floors (room for a further flat, not an ' ...
             'interference): pre %8.2f, new %8.2f mm\n'], ...
            K.floor_pre_leg_m*1e3, K.floor_fold_leg_m*1e3);
end

function d = flr_(pair, m)
    if any(m), d = min([pair(m).d_m]); else, d = NaN; end
end

function n = wname_(pair, m)
    n = '';
    if ~any(m), return; end
    q = pair(m);   [~,i] = min([q.d_m]);   n = q(i).name;
end

function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
