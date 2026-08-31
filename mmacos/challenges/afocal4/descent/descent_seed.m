function [S, info] = descent_seed(P, N, opts)
%DESCENT_SEED  A BUILDABLE N-mirror front end to start a rung from.
%
%   [S, info] = DESCENT_SEED(P, N) searches front ends for an N-mirror
%   closure that satisfies the S4b packaging station (the last powered mirror
%   at least P.pack.m3_behind_min BEHIND the primary) and returns it as a
%   DESCENT_BUILD design struct.
%
%   WHY THIS EXISTS, AND IT IS MEASURED RATHER THAN ASSUMED.  A wall needs a
%   compliant seed or it is a cage (S4b rule 10).  At N = 4 the binding wall
%   was the packaging station; the descent's own first scan says it gets
%   WORSE with N.  Twelve N = 5 front ends built on the committed 4-mirror
%   front end: eight closed with residuals at 2e-16, and SIX OF THOSE EIGHT
%   put the last powered mirror in FRONT of the primary (z = -0.071 to
%   -0.860 m).  Only one cleared the 500 mm minimum.  That is S4b's finding
%   -- one extra mirror flips the parity of the back end -- reappearing at
%   N = 5, and it says the top of the ladder cannot be seeded by scanning
%   arbitrary front ends.
%
%   THE SEARCH IS CHEAP BECAUSE THE CLOSURE IS ALGEBRA.  DESCENT_CLOSE traces
%   no rays, so a front end can be closed and checked for a few hundred
%   microseconds and the search can afford to be wide.  Only the survivors
%   are ever built.  That is the same division AFOCAL4_PACK_SEED uses (cheap
%   algebra filters, expensive gate decides) and the same one WALL_SEED
%   needed when its own predictor turned out to be 5-6x optimistic.
%
%   PREFERENCES, in order -- the study's own, carried up:
%     1  HIS FRONT END where it reaches: M1 and M2 are Rodgers' and are the
%        thing this study is entitled to change last.
%     2  THE WEAKEST added mirrors: least sum of squared power, so the
%        power concentrates where it is needed and the near-flat mirrors
%        identify themselves.  That is the descent brief's power-economy
%        preference used where a preference belongs -- as a TIE-BREAKER
%        among compliant closures, never as a target.
%     3  MARGIN over the station bound, not the bound itself: a seed sitting
%        exactly on a wall has half its finite-difference stencil outside it.
%
%   Name-value:
%     'margin'   m of clearance over P.pack.m3_behind_min (0.03)
%     'R'        radii to try for each added mirror (default 0.8:0.4:3.2 m)
%                The PARENT's own spacings are always added to the 't' grid,
%                so a coarse grid can never report "no compliant closure" for
%                a design that is sitting in the repository complying.
%     'convex'   convexity patterns to try ('auto' = both per added mirror)
%     't'        spacings to try (default 0.4:0.2:1.4 m)
%     'iface'    interface standoff (default P.iface_dist)
%     'ncand'    most closures to keep and rank (20000)
%     'zmax'     upper bound on the last powered mirror's station, m
%                (default 3x the M1-M2 spacing).  The packaging wall bounds
%                that station from BELOW only, and nothing else in the
%                closure cares how long the train is -- so the first N = 7
%                seed this function produced put the last mirror **10.96 m**
%                behind the primary and was, by every check in the study, a
%                compliant design.  It is not a telescope anyone would build.
%                The yardstick is the study's own: the packaging record
%                measures depth as a MULTIPLE OF THE M1-M2 SPACING (committed
%                1.81x, cleared 1.24x), so the bound is stated in the same
%                unit rather than as a round number of metres.
%     'quiet'    (true)
%
%   Returns S (ready for DESCENT_BUILD) and INFO with .ok .n_closed
%   .n_compliant .behind_m1 .phi .sum_phi2 .used_his_front_end .seconds.
%
%   See also DESCENT_CLOSE, DESCENT_BUILD, AFOCAL4_PACK_SEED, WALL_SEED.

    arguments
        P (1,1) struct
        N (1,1) double
        opts.margin (1,1) double = 0.03
        opts.R      (1,:) double = 0.8:0.4:3.2
        opts.t      (1,:) double = 0.4:0.2:1.4
        opts.iface  (1,1) double = NaN
        opts.ncand  (1,1) double = 20000
        opts.zmax   (1,1) double = NaN
        opts.quiet  (1,1) logical = true
    end
    t0 = tic;
    iface = opts.iface;   if isnan(iface), iface = P.iface_dist; end
    need  = P.pack.m3_behind_min + opts.margin;
    yard  = abs(P.parent.t(1));                 % the M1-M2 spacing
    zmax  = opts.zmax;   if isnan(zmax), zmax = 3*yard; end
    nfree = N - 2;                       % free mirrors: 1..N-2

    % His M1/M2 are held wherever the search can reach; only the mirrors
    % BEYOND them are searched.  At N = 4 that is nothing to search and the
    % seed is his front end plus a spacing, which is the committed design.
    R0 = abs(P.parent.R(1:min(2,nfree)));
    c0 = P.parent.convex(1:min(2,nfree));
    nadd = max(0, nfree - 2);

    info = struct('ok',false, 'n_closed',0, 'n_compliant',0, 'behind_m1',NaN, ...
                  'phi',[], 'sum_phi2',NaN, 'used_his_front_end',true, ...
                  'capped',false, 'seconds',0);
    S = struct();

    % THE PARENT'S OWN SPACINGS ARE ALWAYS CANDIDATES.  A grid is a grid: at
    % N = 4 a coarse one simply does not contain t = [1.0420, 2.9285] and the
    % scan reports "no compliant closure" for the very design that shipped
    % and complies at +1323 mm.  That would be a statement about the grid
    % masquerading as a statement about the topology -- so his front-end
    % spacings, and the committed design's, are injected explicitly.
    gt = unique([opts.t, P.parent.t, abs(diff(P.parent.z))], 'stable');
    gR = opts.R;
    best = [];   nclosed = 0;   ncomp = 0;   capped = false;
    % grids over the ADDED mirrors' radii/convexity and every free spacing
    Rsets = combos_(gR, nadd);
    Csets = combos_([0 1], nadd);
    tsets = combos_(gt, nfree);
    for it = 1:size(tsets,1)
        if capped, break; end
        tt = tsets(it,:);
        for ir = 1:size(Rsets,1)
            if capped, break; end
            for ic = 1:size(Csets,1)
                Rf = [R0, Rsets(ir,:)];
                cf = [logical(c0), logical(Csets(ic,:))];
                if numel(Rf) ~= nfree, continue; end
                Sc = struct('N',N, 'R',Rf, 'convex',cf, 't',tt, ...
                            'iface',iface, 'K',zeros(1,N));
                C = descent_close(P, Sc);
                if ~isfield(C,'found') || ~C.found || ~C.closed, continue; end
                nclosed = nclosed + 1;
                if C.behind_m1 < need || C.behind_m1 > zmax, continue; end
                ncomp = ncomp + 1;
                sp = sum(C.phi.^2);
                if isempty(best) || sp < best.sp
                    best = struct('S',Sc, 'C',C, 'sp',sp);
                end
                if nclosed >= opts.ncand, capped = true; break; end
            end
        end
    end
    info.capped = capped;
    info.n_closed = nclosed;   info.n_compliant = ncomp;   info.seconds = toc(t0);
    if isempty(best)
        if ~opts.quiet
            fprintf(['  descent_seed N=%d: NO COMPLIANT CLOSURE over %d closed ' ...
                     'front ends (needed the last mirror in [%+.0f, %+.0f] mm ' ...
                     'behind M1)\n'], N, nclosed, need*1e3, zmax*1e3);
        end
        return;
    end
    S = best.S;   S.K = zeros(1,N);
    info.ok = true;   info.behind_m1 = best.C.behind_m1;
    info.phi = best.C.phi;   info.sum_phi2 = best.sp;
    if ~opts.quiet
        fprintf(['  descent_seed N=%d: %d closed, %d compliant (station in ' ...
                 '[%.0f, %.0f] mm = [%.2f, %.2f] x the M1-M2 spacing), best ' ...
                 'sum(phi^2) %.3f,\n    last mirror %+.0f mm behind M1 ' ...
                 '(%.2fx), %.1f s\n'], N, nclosed, ncomp, need*1e3, zmax*1e3, ...
                need/yard, zmax/yard, best.sp, best.C.behind_m1*1e3, ...
                best.C.behind_m1/yard, info.seconds);
        fprintf('    R [%s] m, t [%s] m, phi [%s] /m\n', ...
            strjoin(arrayfun(@(x)sprintf('%.3f',x),S.R,'UniformOutput',false),' '), ...
            strjoin(arrayfun(@(x)sprintf('%.3f',x),S.t,'UniformOutput',false),' '), ...
            strjoin(arrayfun(@(x)sprintf('%+.3f',x),best.C.phi,'UniformOutput',false),' '));
    end
end

function M = combos_(v, n)
%COMBOS_  All n-tuples from v, as rows.  n = 0 gives one empty row, which is
%   what makes the N = 4 case (nothing to add) fall out of the same loop.
    if n <= 0, M = zeros(1,0);   return; end
    g = cell(1,n);   [g{:}] = ndgrid(v);
    M = reshape(cat(n+1, g{:}), [], n);
end
