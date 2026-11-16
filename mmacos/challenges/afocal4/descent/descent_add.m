function [S2, info] = descent_add(P, S, k, opts)
%DESCENT_ADD  Insert one powered mirror into a rung -- the ASCENT move.
%
%   [S2, info] = DESCENT_ADD(P, S, K) splits the spacing after free element K
%   and puts a new powered mirror in the gap, returning the spec one rung UP.
%   It is DESCENT_REMOVE's inverse, and it exists because of a measurement:
%
%   CLOSING AN ARBITRARY N-MIRROR TRAIN COLD DOES NOT PRODUCE A GOOD DESIGN.
%   Four seven-mirror attempts from compliant-but-arbitrary closures landed
%   at merit 54..707 with 12 um of wavefront -- WORSE than the four-mirror
%   family's own 10.4 um, with more than twice the freedom.  Freeing the
%   radii bought 5.7 %.  The limit was never the DOF set; it was that a
%   cold seven-mirror closure with spherical conics is a bad design and 24
%   DOFs of local optimisation cannot rescue it.
%
%   So the top of the ladder is BUILT UP from the committed four-mirror
%   design -- which already reaches 10.4 um and satisfies its own walls --
%   one mirror at a time, each step warm-started from a design that works.
%   That is the rodgers3 continuation walk the brief names as this stage's
%   method precedent ("solve the easy end, step toward the hard one
%   warm-started, every rung finished and gated"), applied in the direction
%   that makes the top rung reachable.  The ladder then walks back DOWN
%   through rungs that were built UP, which is not circular: the ascent
%   supplies a starting design, and every rung of the descent is re-solved
%   and re-judged on its own.
%
%   THE INSERTED MIRROR STARTS NEAR-FLAT, and that is a warm start rather
%   than a no-op.  A flat still REFLECTS, so inserting one flips the
%   packaging parity of everything downstream (README section 3) -- the
%   stations move even though the paraxial marginal and chief do not.  The
%   closure re-solves the last two powers and the last spacing around that,
%   so the child is close to its parent in POWER but not in LAYOUT, and the
%   packaging wall has to be re-checked rather than inherited.
%
%   AND THE INSERTION NEEDS A COMPLIANT SEED, EXACTLY LIKE A COLD START.
%   The first ascent attempt inserted a near-flat mirror at the midpoint of a
%   spacing and the child landed 479 mm in FRONT of the primary, with no root
%   for the interface condition at all.  Nothing was wrong with the arithmetic:
%   splitting t = 2.9285 into 1.464 + 1.464 re-folds the alternating sum, and
%   z goes 0, -1.042, +0.422, -1.042, b-1.042 -- the parity law charging for
%   the extra reflection.  So WHERE in the spacing the mirror goes is not a
%   detail, it is the compliance knob, and 'search' scans it (and the
%   insertion point) for closures that clear the station.  S4b's rule again:
%   a wall needs a compliant seed or it is a cage.
%
%   Name-value:
%     'search' scan insertion points and splits for a COMPLIANT closure and
%              return the best (false).  With it on, K is a starting hint and
%              'splits' / 'ks' bound the scan.
%     'splits' split fractions to scan (0.1 : 0.05 : 0.9)
%     'ks'     insertion points to scan ([] = every free element)
%     'phi0'   the new mirror's starting power, /m (0 = flat).  A flat gives
%              the closest warm start; a small nonzero value can be handed in
%              when the caller wants the solve to begin off the degenerate
%              R = Inf point.
%     'split'  where in the spacing the new mirror goes, 0..1 (0.5)
%     'convex' the new mirror's sense (false)
%
%   Returns S2 and INFO with .k .n_elements .parity_flips .split_from .why.
%
%   See also DESCENT_REMOVE, DESCENT_CLOSE, DESCENT_SOLVE.

    arguments
        P (1,1) struct %#ok<INUSA>
        S (1,1) struct
        k (1,1) double
        opts.phi0   (1,1) double = 0
        opts.split  (1,1) double = 0.5
        opts.convex (1,1) logical = false
        opts.search (1,1) logical = false
        opts.splits (1,:) double = 0.1:0.05:0.9
        opts.ks     (1,:) double = []
        opts.margin (1,1) double = 0.03
    end
    if opts.search
        [S2, info] = search_(P, S, opts);
        return;
    end
    Kel = S.N;   nfree = Kel - 2;
    if k < 1 || k > nfree
        error('macos:design:descent_add:index', ...
              ['insert after free element 1..%d (asked %d).  The last two ' ...
               'elements'' powers are consumed by the closure; a mirror ' ...
               'inserted among them would be re-solved away.'], nfree, k);
    end
    if opts.split <= 0 || opts.split >= 1
        error('macos:design:descent_add:split','split must be in (0,1).');
    end

    Rnew = flat_R_(opts.phi0);
    S2 = S;
    S2.N      = Kel + 1;
    S2.R      = [S.R(1:k),      Rnew,        S.R(k+1:end)];
    S2.convex = [S.convex(1:k), opts.convex, S.convex(k+1:end)];
    S2.K      = [S.K(1:k),      0,           S.K(k+1:end)];
    t = S.t;
    tsplit = t(k);
    S2.t = [t(1:k-1), tsplit*opts.split, tsplit*(1-opts.split), t(k+1:end)];

    info = struct('k',k, 'n_elements',Kel+1, 'parity_flips',true, ...
        'split_from',tsplit, 'phi0',opts.phi0, ...
        'why',['a new element means a new reflection, so every station after ' ...
               'it re-folds and the packaging parity FLIPS -- the child must ' ...
               'be re-checked against the wall, never assumed to inherit it']);
end

function [S2, info] = search_(P, S, opts)
%SEARCH_  The compliant insertion: scan where the mirror goes and how the
%   spacing splits, keep the closures that clear the packaging station, and
%   take the one whose powers are CLOSEST to the parent's -- the warmest
%   start, which is the whole point of ascending rather than closing cold.
    Kel = S.N;   nfree = Kel - 2;
    ks = opts.ks;   if isempty(ks), ks = 1:nfree; end
    need = P.pack.m3_behind_min + opts.margin;
    yard = abs(P.parent.t(1));
    zmax = 3*yard;
    Cp = descent_close(P, S);
    phip = [];   if isfield(Cp,'found') && Cp.found, phip = Cp.phi; end
    best = [];   ntry = 0;   nclose = 0;   ncomp = 0;
    for k = ks
        for sp = opts.splits
            ntry = ntry + 1;
            o2 = opts;  o2.search = false;  o2.split = sp;
            [Sc, ic] = descent_add(P, S, k, 'phi0',opts.phi0, 'split',sp, ...
                                   'convex',opts.convex); %#ok<ASGLU>
            C = descent_close(P, Sc);
            if ~isfield(C,'found') || ~C.found || ~C.closed, continue; end
            nclose = nclose + 1;
            if C.behind_m1 < need || C.behind_m1 > zmax, continue; end
            ncomp = ncomp + 1;
            % warmth: how far the child's powers sit from the parent's, with
            % the inserted element skipped
            d = Inf;
            if ~isempty(phip)
                q = C.phi;  q(k+1) = [];
                d = norm(q - phip);
            end
            if isempty(best) || d < best.d
                best = struct('S',Sc, 'C',C, 'd',d, 'k',k, 'split',sp);
            end
        end
    end
    if isempty(best)
        S2 = [];
        info = struct('ok',false, 'n_tried',ntry, 'n_closed',nclose, ...
            'n_compliant',0, 'why',sprintf(['no insertion over %d points x ' ...
            '%d splits closes AND clears the station -- a LADDER DATUM'], ...
            numel(ks), numel(opts.splits)));
        return;
    end
    S2 = best.S;
    info = struct('ok',true, 'k',best.k, 'split',best.split, ...
        'n_elements',best.C.n_elements, 'parity_flips',true, ...
        'n_tried',ntry, 'n_closed',nclose, 'n_compliant',ncomp, ...
        'warmth',best.d, 'behind_m1',best.C.behind_m1, ...
        'why',['inserted where the alternating sum still lands the last ' ...
               'mirror behind the primary, and closest in power to the parent']);
end

function R = flat_R_(phi)
%FLAT_R_  A radius for a requested power.  phi = 0 is a FLAT, given a large
%   finite radius rather than Inf so the emitter and the engine never see one.
    if abs(phi) < 1e-12, R = 1e12; else, R = abs(2/phi); end
end
