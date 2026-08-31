function K = afocal4_union(deck, opts)
%AFOCAL4_UNION  Does a BODY stand in a BEAM?  The union body-in-beam gate.
%
%   K = AFOCAL4_UNION(DECK, 'fields', F) traces a committed prescription
%   over the FIELD SET F and measures, for every (leg, optic) pair the
%   layout admits, the signed distance from that leg's traced ray segments
%   to that optic's BODY -- the union of its footprint over the whole field
%   set, hulled, scaled by BODY_K and grown by BODY_PAD.  Negative means the
%   beam goes through the glass.
%
%   WHY THIS GATE EXISTS.  AFOCAL4_PACK checks the LAST mirror's exit leg
%   for beam-to-beam daylight, at the box centre, and asks whether a fold
%   FITS.  It never asks whether a PART is standing in a BEAM.  On the
%   committed 343 mm family-2 deck the answer is that one is: the
%   M2 -> field-mirror feed beam runs through the collimator's own glass by
%   -55.4 mm bare and -79.9 mm with the body model, and the whole S4b/S4c
%   trade shipped without anyone measuring it (BRIEF_r2_packaging delivery
%   log; BRIEF_afocal4_clear).  A margin is a NUMBER, not a BODY: a gate
%   that reports daylight without sizing the part it is making room for can
%   pass a design no one can build.
%
%   THREE MODEL CHOICES, ALL LOAD-BEARING, ALL FROM THE PACKAGING STAGE:
%
%   1  THE UNION, OVER THE FIELD BOX -- not the deck's own field.  One
%      mirror has to carry every field, so its body is the union of its
%      footprints.  On this design the collimator's footprint is 17.0 mm
%      per field and 87.0 mm over the box, and the interference is entirely
%      in the difference: per field there is 10.8 mm of daylight all round.
%      A centre-field table would have passed it.
%
%   2  A HULL, NOT A DISK.  The union is nine overlapping patches walking
%      ~70 mm across the part; a centred disk of the union's max radius
%      fills in the middle -- exactly where this train's feed beam passes --
%      and invents a 107 mm interference that is the model's, not the
%      design's.  (OI_CLEAR and PACK_CLEAR carry the same choice.)
%
%   3  THE ALLOWANCE IS DECLARED, NOT ASSUMED.  BODY_K x footprint +
%      BODY_PAD, defaulting to 1.15 and 15 mm, and both are printed with
%      every answer.  Run it at BODY_K = 1, BODY_PAD = 0 for BARE LIT GLASS:
%      an interference that survives that one is the design's and not the
%      body model's.
%
%   SAMPLING-FREE, so a fold cannot change it.  A fold is an isometry, so
%   every pre-existing clearance must read the same before and after
%   folding, and that is this gate's only independent check that a fold
%   really is one.  A station-sampled model cannot deliver it (folding cuts
%   one long leg into several short ones and re-samples the same geometry at
%   a different phase -- measured at 10.6 vs 5.8 mm on a pair an isometry
%   cannot have moved).  Here a pierce is an exact plane crossing and a
%   clearance is an exact ternary search, distance to a convex set being
%   convex along a straight segment.
%
%   THROUGH-HOLES ARE A REQUIREMENT, NOT A COLLISION.  A two-mirror front
%   end sends its beam back through the primary; 'leg pierces M1' is the
%   statement that M1 needs a hole, true of every deck in this family.
%   Elements listed in 'hole' report a hole RADIUS instead of a pierce.
%
%   LEG-VERSUS-LEG IS DELIBERATELY NOT HERE.  Light passes through light,
%   and on a wide-field system different fields' beams genuinely cross, so a
%   leg-leg zero is not an interference -- it says only whether a further
%   flat could be inserted.  PACK_CLEAR reports it; this gate does not,
%   because this gate's verdict has to mean one thing.
%
%   Name-value:
%     'fields'    K x 2 field offsets, rad, RELATIVE to the deck's own
%                 chief.  Required in spirit: with none, the deck's single
%                 field is used and the answer is the optimistic one.
%     'body_k'    body = this x footprint (1.15)
%     'body_pad'  ... grown by this, m (0.015)
%     'hole'      element indices allowed a through-hole (default 1, M1)
%     'floor'     the pass threshold, m (0 -- a body may touch a beam but
%                 not enter it)
%     'rim_frac'  rays at >= this fraction of the stop radius are 'rim'
%                 (0.95);  'n_rim' / 'n_int' how many of each are probed
%     'init'      load the deck (true);  'quiet' (false)
%
%   Returns K with .ok .floor_m .worst .why, .pair (.leg .obst .name .d_m
%   .pierced), .hole, .foot_r .foot_c .body_r .body .ap (per element: the
%   beam's offset from the vertex and the clear aperture the part must
%   carry), .nRays .nLost .nFields.
%
%   See also AFOCAL4_PACK, PACK_CLEAR, CLEAR_SCAN, CLEAR_GATE_NONVACUITY.

    arguments
        deck (1,:) char
        opts.fields   (:,2) double = []
        opts.body_k   (1,1) double = 1.15
        opts.body_pad (1,1) double = 0.015
        opts.hole     (1,:) double = 1
        opts.floor    (1,1) double = 0
        opts.rim_frac (1,1) double = 0.95
        opts.n_rim    (1,1) double = 288
        opts.n_int    (1,1) double = 64
        opts.init     (1,1) logical = true
        opts.quiet    (1,1) logical = false
    end

    if opts.init, macos.load_rx(deck); end
    nE = macos.num_elt();
    [h, nR, nLost] = field_union_(deck, opts.fields);
    off = size(h.P,3) - nE;

    % ---- per-element union footprint, body and required aperture ---------
    C = zeros(3,nE);  Rf = zeros(1,nE);  Rb = zeros(1,nE);
    psi = zeros(3,nE);  V = zeros(3,nE);   body = cell(1,nE);
    ap = struct('elt',{},'off_m',{},'r_req_m',{});
    for k = 1:nE
        j = k + off;   m = h.ok(:,j);
        if ~any(m)
            error('macos:design:afocal4_union:dark', ...
                  'element %d catches no ray on any field -- nothing to gate.', k);
        end
        Q = squeeze(h.P(:,m,j));   if size(Q,1) ~= 3, Q = Q(:); end
        C(:,k)   = mean(Q,2);
        Rf(k)    = max(vecnorm(Q - C(:,k)));
        Rb(k)    = opts.body_k*Rf(k) + opts.body_pad;
        psi(:,k) = macos.get_elt_psi(k);
        V(:,k)   = macos.get_elt_vpt(k);
        body{k}  = hull_(Q, C(:,k), psi(:,k), opts.body_k);
        % what the PART has to be: the beam sits where the field bias puts
        % it, not on the vertex, and a part is mounted on its vertex.
        d    = C(:,k) - V(:,k);
        doff = d - psi(:,k)*(psi(:,k).'*d);
        ap(end+1) = struct('elt',k, 'off_m',norm(doff), ...
                   'r_req_m', norm(doff) + Rf(k) + opts.body_pad); %#ok<AGROW>
    end

    K = struct('deck',deck, 'nElt',nE, 'foot_c',C, 'foot_r',Rf, 'body_r',Rb, ...
               'psi',psi, 'vpt',V, 'ap',ap, 'body',[body{:}], ...
               'body_k',opts.body_k, 'body_pad',opts.body_pad, ...
               'hole_elts',opts.hole, 'floor_req_m',opts.floor, ...
               'nRays',nR, 'nLost',nLost, 'nFields',max(1,size(opts.fields,1)));

    % ---- which rays carry the bundle boundary ---------------------------
    % The closest points of two disjoint convex bodies lie on their
    % boundaries, and a bundle's boundary is swept by the pupil RIM -- so
    % rim rays decide a clearance.  A coarse interior sample rides along as
    % a guard against a bundle whose boundary is not the rim's image.
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

    seg = cell(1,nE-1);  segp = cell(1,nE-1);
    for k = 1:nE-1
        ja = k+off;  jb = k+1+off;
        m  = h.ok(:,ja) & h.ok(:,jb);
        A  = squeeze(h.P(:,m,ja));   B = squeeze(h.P(:,m,jb));
        if size(A,1) ~= 3, A = A(:);  B = B(:); end
        seg{k}  = struct('A',A, 'B',B);
        mu = m(use);   ii = cumsum(m);   idx = ii(use(mu));
        segp{k} = struct('A',A(:,idx), 'B',B(:,idx));
    end

    pair = struct('leg',{},'obst',{},'name',{},'d_m',{},'pierced',{});
    hole = struct('elt',{},'r_req_m',{},'leg',{});
    for k = 1:nE-1
        A = seg{k}.A;  B = seg{k}.B;  Ap = segp{k}.A;  Bp = segp{k}.B;
        for e = 1:nE
            if e == k || e == k+1, continue; end
            [d, pierced, rmax] = seg_patch_(A, B, Ap, Bp, body{e}, opts.body_pad);
            if pierced && any(e == opts.hole)
                hole(end+1) = struct('elt',e, 'r_req_m',rmax + opts.body_pad, ...
                                     'leg',k); %#ok<AGROW>
                continue;
            end
            pair(end+1) = struct('leg',k, 'obst',e, ...
                'name',sprintf('leg %d-%d vs body %d', k, k+1, e), ...
                'd_m',d, 'pierced',pierced); %#ok<AGROW>
        end
    end

    K.pair = pair;   K.hole = hole;
    [K.floor_m, K.worst] = min([pair.d_m]);
    K.worst_name = pair(K.worst).name;
    K.ok  = K.floor_m >= opts.floor;
    K.why = '';
    if ~K.ok
        K.why = sprintf('%s: %.2f mm (need >= %.1f)', K.worst_name, ...
                        K.floor_m*1e3, opts.floor*1e3);
    end
    if ~opts.quiet, report_(K); end
end

% =====================================================================
function report_(K)
    fprintf('\n  UNION BODY-IN-BEAM  %s\n', K.deck);
    fprintf(['    body = %.2f x union footprint + %.0f mm; %d field(s), ' ...
             '%d rays (%d probes), %d lost\n'], K.body_k, K.body_pad*1e3, ...
            K.nFields, K.nRays, K.n_probe, K.nLost);
    fprintf('    %-4s %11s %10s %11s %11s\n', 'elt','union r mm','body r mm', ...
            'off vtx mm','clear ap mm');
    for k = 1:K.nElt
        fprintf('    %-4d %11.1f %10.1f %11.1f %11.1f\n', k, K.foot_r(k)*1e3, ...
                K.body(k).r*1e3, K.ap(k).off_m*1e3, K.ap(k).r_req_m*1e3);
    end
    [~, ord] = sort([K.pair.d_m]);
    fprintf('    tightest pairs:\n');
    for i = ord(1:min(6,numel(ord)))
        p = K.pair(i);
        fprintf('      %-26s %9.2f mm  %s\n', p.name, p.d_m*1e3, ...
                tern_(p.pierced,'<-- PIERCED',''));
    end
    for i = 1:numel(K.hole)
        fprintf(['    through-hole: leg %d-%d crosses elt %d -> hole radius ' ...
                 '>= %.1f mm\n'], K.hole(i).leg, K.hole(i).leg+1, ...
                K.hole(i).elt, K.hole(i).r_req_m*1e3);
    end
    if K.ok
        fprintf('    => CLEAR: floor %+.2f mm (%s)\n', K.floor_m*1e3, K.worst_name);
    else
        fprintf('    => BODY IN BEAM: %s\n', K.why);
    end
end

% ---------------------------------------------------------------------
function [h, nR, nLost] = field_union_(deck, F)
%FIELD_UNION_  Ray history over a FIELD SET, stacked along the ray index.
%   Each field is traced on its own re-pointed copy of the deck (chief
%   direction + yGrid, pivoting about the stop -- the AFOCAL_LADDER_DECK
%   recipe, so 'the field box' means the same thing here as in the scoring)
%   and the histories are concatenated.  Apertures are NOT stripped: a ray
%   the design itself vignettes is not a ray any hardware has to clear.
    if isempty(F)
        macos.ray_hist('on');   t = macos.trace();   h = macos.ray_hist(t.nRays);
        macos.ray_hist('off');
        nR = t.nRays;   nLost = t.nRays - sum(h.ok(:,end));
        return;
    end
    txt  = fileread(deck);
    cd0  = grab3_(txt,'ChfRayDir');   cp0 = grab3_(txt,'ChfRayPos');
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
    % plausible-looking centroid.  Cost, once already: one full run of the
    % packaging study whose every clearance was the chief ray's.
    h = struct('P',P, 'ok',logical(OK));
    macos.load_rx(deck);
end

function [u, v] = perp_(a)
    a = a(:)/norm(a);
    [~, i] = min(abs(a));   u = zeros(3,1);  u(i) = 1;
    u = u - (u.'*a)*a;  u = u/norm(u);
    v = cross(a, u);
end

function H = hull_(Q, c, n, kscale)
%HULL_  The element's BODY: the convex hull of its union footprint in its
%   own plane, scaled by KSCALE about the footprint centroid.
    n = n(:)/norm(n);
    [u, v] = perp_(n);
    w = [u.'*(Q - c); v.'*(Q - c)];
    if size(w,2) >= 3
        try
            i = convhull(w(1,:).', w(2,:).');
            poly = w(:,i);
        catch
            poly = w;                        % degenerate (collinear) footprint
        end
    else
        poly = w;
    end
    % Cap the vertex count: a hull over ten thousand rays carries hundreds
    % of nearly-collinear vertices and the distance query is O(edges x
    % points).  Keep ORIGINAL vertices (a subsample of a convex polygon is
    % inscribed, i.e. slightly SMALLER than the true body) and re-inflate to
    % the full hull radius so the body is never under-reported.
    nmax = 64;
    if size(poly,2) > nmax
        r0 = max(vecnorm(poly));
        poly = poly(:, round(linspace(1, size(poly,2), nmax)));
        poly = poly * (r0/max(vecnorm(poly)));
    end
    poly = kscale*poly;
    H = struct('c',c, 'n',n, 'u',u, 'v',v, 'poly',poly, 'r',max(vecnorm(poly)));
end

function [d, pierced, rmax] = seg_patch_(A, B, Ap, Bp, H, pad)
%SEG_PATCH_  Signed distance from a bundle of segments to a body patch.
%   PIERCE first, exactly: a segment crossing the patch plane INSIDE the
%   grown hull is a hit and the answer is minus the deepest penetration
%   from the hull boundary; RMAX is the farthest crossing from the centroid
%   (a through-hole radius).  Otherwise a ternary search in t -- the
%   distance to a convex set is convex along a straight segment, so the
%   search is exact rather than a sampling.  The pierce test runs on EVERY
%   ray (one vectorised plane crossing); the search runs on the probe
%   subset, which carries the bundle boundary and is where the minimum is.
    n   = H.n;
    den = n.'*(B - A);
    num = n.'*(H.c - A);
    tt  = num./den;
    hit = abs(den) > 1e-14 & tt > 0 & tt < 1;
    pierced = false;   rmax = NaN;
    if any(hit)
        X  = A(:,hit) + tt(hit).*(B(:,hit) - A(:,hit));
        dp = poly_dist_(H, X) - pad;
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
    Pg = H.poly;
    if size(Pg,2) < 2
        d = vecnorm(w - Pg(:,1));   return;
    end
    d = inf(1, size(w,2));
    for i = 1:size(Pg,2)-1
        a = Pg(:,i);  b = Pg(:,i+1);
        ab = b - a;   L2 = ab.'*ab;
        if L2 < eps, continue; end
        t  = max(0, min(1, (ab.'*(w - a))/L2));
        dx = w(1,:) - (a(1) + ab(1)*t);
        dy = w(2,:) - (a(2) + ab(2)*t);
        d  = min(d, hypot(dx, dy));
    end
    if size(Pg,2) >= 3
        in = inpolygon(w(1,:), w(2,:), Pg(1,:), Pg(2,:));
        d(in) = -d(in);
    end
end

function v = grab3_(txt, key)
    t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens', 'once');
    v = sscanf(strrep(t{1},'D','E'), '%f', 3);
end
function s = v3_(v),  s = sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));  end
function del_(p),  if exist(p,'file'), delete(p); end,  end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
