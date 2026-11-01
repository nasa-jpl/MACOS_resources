function [Ls, info] = iodc2010_relax(L, p, opts)
%IODC2010_RELAX  Relax and snap: the on-axis construction for the pupil (Dave's ruling, 2026-10-31).
%   [Ls, info] = iodc2010_relax(L, p, 'D_target', D) takes a green lens L
%   (its sign pattern is the start, its spacings the seed) and returns a
%   green lens Ls of the same element count whose spacings AND surface
%   shapes were re-solved for the smallest on-axis RMS wavefront error at the
%   entrance pupil D, with the shapes first RELAXED to continuous curvatures
%   and then SNAPPED back to the problem's {-R, plano, +R}.
%
%   Why.  With one radius magnitude the spacings are the only continuous
%   variables a green lens has, and no enumerated sign pattern starts near a
%   balanced f/1 solution: measured 2026-10-31 (README "The on-axis
%   construction"), no seed family passes much above 6 mm on axis at EFL 100
%   and no solve over the spacings climbs from there (CALIB, a CALIB pupil
%   climb, a bounded least squares outside the engine).  The entrants' route
%   (the paper, Sec. 4) was a global optimizer with the curvatures free and a
%   constraint function c_j (c - c_j)(c + c_j), weighted low so the
%   curvatures roam and high so they snap.  This is that route around the
%   engine's trace, with one change for a LOCAL solver: the cubic has no
%   gradient near |c_j| = 0.58 c (the first scratch run stalled there at
%   every weight), so the pull is linear, toward each curvature's NEAREST
%   grid value, and re-aimed at every stage.
%
%   The solve (lsqnonlin, the Optimization Toolbox):
%     variables   kappa_j (every surface's curvature in units of 1/R0, the
%                 common curvature being kappa = +-1) and log tau_k (every
%                 gap before the image in units of R0; the image at the
%                 paraxial focus); bounds: |kappa| <= opts.kappa_max, the
%                 thickness floors of iodc2010_solve (glass 0.03 R, air
%                 0.01 R) and ceilings (1.9 R glass, 20 R air);
%     residual    the engine's per-ray OPD on axis at D (iodc2010_eval on
%                 the lens scaled to |EFL| = p.efl_mm at every evaluation --
%                 the problem is scale-free except for the EFL -- on
%                 opts.score_grid rays across, the chief-less map, in waves),
%               + sqrt(mu) sqrt(n) (kappa_j - snap(kappa_j))  the pull,
%               + w_P sqrt(n) (the Petzval sum in units of 1/R0)   so the
%                 snapped pattern is balanced (every entry but one was),
%               + a soft wall on the edge thickness of every gap at the
%                 marginal ray's height (glass and air both positive);
%                 a lens that loses rays or has no real image returns the
%                 passing rays' OPD plus a lost-fraction row, not a cliff;
%     stages      mu = 0 (free, opts.iters_free iterations); then the SNAP.
%                 The grid is +-kc and plano with kc the COMMON MAGNITUDE
%                 the free solution implies (the median |kappa| of its
%                 curved surfaces; kappa's units are arbitrary because the
%                 lens is re-scaled to the EFL at every evaluation, so a
%                 free solution at +-0.5 is a green lens of twice the
%                 radius -- the first product run snapped to 1 and moved
%                 every surface by half its curvature; info.kappa_common).
%                 opts.snap = 'choice' (default): the free curvature nearest
%                 its grid value is fixed there and the remaining variables
%                 re-solved (opts.iters), surface by surface until every
%                 shape is on the grid -- the last re-solve is the
%                 spacing-only one; a curvature farther than opts.ambiguous
%                 x kc from the grid tries BOTH neighbouring values (plano
%                 and the sign, or the two signs) and keeps the better
%                 re-solve (the first stage run, 2026-10-31: the nearest-value
%                 rule went plano-heavy, six elements to two curved surfaces,
%                 and the last fixes lost the solution).  With
%                 opts.petzval_balance the count balance is HARD: a candidate
%                 value is refused when the surfaces still free could not
%                 restore equal converging and diverging counts, so the last
%                 fixes are forced to the balance (a soft weight of 1 did not
%                 hold it).  'sequential' is the nearest value only.
%                 'pull': opts.mu in turn (opts.iters each), the pull
%                 re-aimed at every stage, then one snap and a spacing
%                 re-solve.  MEASURED on the first 6-element seed at D 20
%                 (2026-10-31 evening): the free stage reaches 0.041 wave;
%                 'pull' stalls 0.18 off the grid at every weight and its
%                 snap lands on a lens whose sphere is smaller than its beam
%                 (320 waves, every ray lost) -- hence the beam-in-sphere
%                 wall below and the sequential default.
%               + a wall keeping every surface's beam inside its sphere
%                 (|c_j| h_j <= 0.9: a ray outside the sphere is a surface
%                 miss, which no residual can grade);
%   D is held by re-sizing the stop at every evaluation; the seed's stop
%   keeps its place as a fraction of the lens's length.  The clear apertures
%   are open during the solve (the metric sizes them afterwards).
%
%   Returns Ls (a green lens: R, sgn, t, the stop sized for D_target, Rk
%   cleared) and info: .stages (table: mu, rms_waves at D, max_pull = the
%   largest |kappa - snap|, petzval, R, oal, flag, iterations, evals, s),
%   .rms_snapped (the re-solved snapped lens's on-axis RMS at D, waves),
%   .rms_free (after the free stage), .kappa_free, .sgn, .capacity (the
%   snapped lens's iodc2010_capacity, mm) and .capacity_seed.
arguments
    L struct
    p struct
    opts.D_target (1,1) double
    opts.mu (1,:) double = [0.01 0.1 1 10 100]
    opts.iters_free (1,1) double = 60
    opts.iters (1,1) double = 25
    opts.kappa_max (1,1) double = 1.5
    opts.petzval_weight (1,1) double = 1
    opts.floor_glass_R (1,1) double = 0.03
    opts.floor_air_R (1,1) double = 0.01
    opts.snap (1,:) char = 'choice'       % 'choice': fix one curvature at a time, nearest the grid first, an AMBIGUOUS one by trying its two neighbouring grid values and keeping the better re-solve; 'sequential': the nearest value only; 'pull': the mu stages, then one snap
    opts.ambiguous (1,1) double = 0.15    % a curvature farther than this (x kc) from its nearest grid value is ambiguous (2 re-solves)
    opts.petzval_balance (1,1) logical = true   % HARD: the snapped pattern has equal converging and diverging surface counts (a candidate that cannot be balanced by the surfaces still free is refused)
    opts.score_grid (1,1) double = 21
    opts.n_res (1,1) double = 300
    opts.deck (1,:) char = ''
    opts.capacity (1,1) logical = true
    opts.quiet (1,1) logical = true
end
say = @(varargin) iif_(~opts.quiet, varargin{:});
assert(~isempty(which('lsqnonlin')), 'iodc2010_relax: lsqnonlin (Optimization Toolbox) is needed');
if isempty(opts.deck), opts.deck = fullfile(p.outdir, [L.name '_relax.in']); end
pq = p;  pq.nGridpts = opts.score_grid;  pq.size_apertures = false;
N = L.N;  D = opts.D_target;  nr = opts.n_res;
isglass = ~strcmpi(L.glass(1:N-1), 'air');
R0 = 100;
c0 = iodc2010_curv(L);  kap = c0 * L.R;                    % the seed's shapes: +-1 / 0
tau = L.t(1:N-1) / L.R;
x = [kap, log(tau)];
lb = [-opts.kappa_max * ones(1, N), log(opts.floor_air_R * ones(1, N-1))];  lb(N + find(isglass)) = log(opts.floor_glass_R);
ub = [ opts.kappa_max * ones(1, N), log(20 * ones(1, N-1))];                ub(N + find(isglass)) = log(1.9);
x = min(max(x, lb), ub);
ctx = struct('L', L, 'D', D, 'pq', pq, 'deck', opts.deck, 'nr', nr, 'R0', R0, 'wP', opts.petzval_weight, 'N', N);
lsq = @(it) optimoptions('lsqnonlin', 'Display', 'off', 'MaxIterations', it, 'FunctionTolerance', 1e-10, ...
    'StepTolerance', 1e-9, 'FiniteDifferenceStepSize', 1e-4, 'MaxFunctionEvaluations', 200 * (2 * N));
if opts.capacity, [info.capacity_seed, ci] = iodc2010_capacity(L, p, 'deck', opts.deck); else, info.capacity_seed = NaN; ci = struct('rms', NaN); end
say('relax %s: N %d, D %g mm; seed capacity %.2f (rms %.4f)\n', L.name, N / 2, D, info.capacity_seed, ci.rms);
rows = {};  t_all = tic;
% ---- the free stage
fun = @(x) resid_(x, ctx, 0, snap_(x(1:N)));
[x, ~, ~, flag, out] = lsqnonlin(fun, x, lb, ub, lsq(opts.iters_free));
rows{end+1} = row_(x, fun, ctx, 'free', 0, flag, out, toc(t_all));
say('   %s\n', rows{end}.line);
info.rms_free = rows{end}.rms_waves;  info.kappa_free = x(1:N);
kc = common_(x(1:N));  info.kappa_common = kc;
say('   common curvature magnitude from the free solution: %.3f (R0 / kc = %.1f mm before the scale)\n', kc, R0 / kc);
ctx.kc = kc;
switch opts.snap
    case 'pull'
        for mu = opts.mu
            snap = snap_(x(1:N), kc);
            fun = @(x) resid_(x, ctx, mu, snap);
            [x, ~, ~, flag, out] = lsqnonlin(fun, x, lb, ub, lsq(opts.iters));
            rows{end+1} = row_(x, fun, ctx, sprintf('pull mu %g', mu), mu, flag, out, toc(t_all)); %#ok<AGROW>
            say('   %s\n', rows{end}.line);
        end
        fixed = true(1, N);  ks = snap_(x(1:N), kc);  xs = x;  xs(1:N) = ks;
        lbs = lb;  ubs = ub;  lbs(1:N) = ks;  ubs(1:N) = ks;
        fun = @(x) resid_(x, ctx, 0, ks);
        [xs, ~, ~, flag, out] = lsqnonlin(fun, xs, lbs, ubs, lsq(opts.iters));
        rows{end+1} = row_(xs, fun, ctx, 'snap + spacings', Inf, flag, out, toc(t_all));
        say('   %s\n', rows{end}.line);
    case {'sequential', 'choice'}
        fixed = false(1, N);  xs = x;  lbs = lb;  ubs = ub;
        cd = convdiv_(L);                                     % per surface: [conv, div] contribution of +1 and -1
        for s = 1:N
            d = abs(xs(1:N) - snap_(xs(1:N), kc));  d(fixed) = Inf;
            [dj, j] = min(d);
            k0 = round(snap_(xs(j), kc) / kc);
            cands = k0;
            if strcmp(opts.snap, 'choice') && dj > opts.ambiguous * kc
                other = sign(xs(j));  if k0 ~= 0, other = 0; end      % the other neighbour: the sign when plano is nearest, plano otherwise
                if other == 0 && k0 == 0, other = sign(xs(j)); end
                cands = [k0, other];
            end
            if opts.petzval_balance                                   % refuse what the free surfaces cannot balance
                ok = false(size(cands));
                for q = 1:numel(cands)
                    kk = round(xs(1:N) / kc);  kk(~fixed) = 0;  kk(j) = cands(q);
                    imb = balance_(kk, fixed | ((1:N) == j), cd);
                    ok(q) = abs(imb) <= N - nnz(fixed) - 1;
                end
                if ~any(ok)                                          % every neighbour refused: the balancing value
                    kk = round(xs(1:N) / kc);  kk(~fixed) = 0;
                    for q = [-1 0 1]
                        kk(j) = q;  if abs(balance_(kk, fixed | ((1:N) == j), cd)) <= N - nnz(fixed) - 1, cands = q;  ok = true;  break; end
                    end
                else
                    cands = cands(ok);
                end
            end
            best = [];  tried = '';
            for q = 1:numel(cands)
                kj = cands(q) * kc;
                xq = xs;  xq(j) = kj;  lq = lbs;  uq = ubs;  lq(j) = kj;  uq(j) = kj;
                fun = @(x) resid_(x, ctx, 0, snap_(x(1:N), kc));
                [xq, resn, ~, flag, out] = lsqnonlin(fun, xq, lq, uq, lsq(opts.iters));
                tried = [tried sprintf(' %+d:%.4f', cands(q), sqrt(resn / nr))]; %#ok<AGROW>
                if isempty(best) || resn < best.resn, best = struct('x', xq, 'lb', lq, 'ub', uq, 'k', cands(q), 'resn', resn, 'flag', flag, 'out', out, 'fun', fun); end
            end
            xs = best.x;  lbs = best.lb;  ubs = best.ub;  fixed(j) = true;
            tag = sprintf('fix S%d = %+d (%d of %d)', j, best.k, s, N);
            if numel(cands) > 1, tag = [tag ' <' strtrim(tried) '>']; end %#ok<AGROW>
            rows{end+1} = row_(xs, best.fun, ctx, tag, Inf, best.flag, best.out, toc(t_all)); %#ok<AGROW>
            say('   %s\n', rows{end}.line);
        end
    otherwise
        error('iodc2010_relax: snap must be ''sequential'' or ''pull''');
end
ks = round(xs(1:N) / kc);
Ls = lens_(xs, ctx);  Ls.Rk = [];  Ls.sgn = ks;  Ls.R = Ls.R / kc;        % a green lens again (its radius R0 / kc, scaled)
P = iodc2010_paraxial(Ls, p.lambda_nm);  Ls.stop(2) = D / P.D;  Ls.semidia = [];
Ls.name = [L.name '_relax_D' strrep(sprintf('%g', D), '.', 'p')];
info.stages = struct2table([rows{:}], 'AsArray', true);
info.stages.line = [];
info.rms_snapped = rows{end}.rms_waves;  info.sgn = ks;
say('   snap %s -> rms %.4f waves at D %g (R %.1f, oal %.1f)\n', mat2str(ks), info.rms_snapped, D, Ls.R, Ls.z(end));
if opts.capacity
    [info.capacity, cs] = iodc2010_capacity(Ls, p, 'deck', opts.deck);
    info.capacity_reason = cs.reason;
    say('   capacity %.2f mm (rms %.4f; above: %s) against the seed''s %.2f\n', info.capacity, cs.rms, cs.reason, info.capacity_seed);
else
    info.capacity = NaN;  info.capacity_reason = '';
end
info.deck = opts.deck;  info.s = toc(t_all);
end

% -------------------------------------------------------------------------
function row = row_(x, fun, ctx, stage, mu, flag, out, s)
r = fun(x);  Lr = lens_(x, ctx);  N = ctx.N;
if isfield(ctx, 'kc'), kc = ctx.kc; else, kc = common_(x(1:N)); end
row = struct('stage', stage, 'mu', mu, 'rms_waves', sqrt(mean(r(1:ctx.nr).^2)), 'max_pull', max(abs(x(1:N) - snap_(x(1:N), kc))) / kc, ...
    'petzval', petzval_(x(1:N), Lr), 'R', Lr.R, 'oal', Lr.z(end), 'flag', flag, 'iterations', out.iterations, ...
    'evals', out.funcCount, 's', s);
row.line = sprintf('%-22s rms %8.4f waves  off-grid %.3f  Petzval %+.3f  R %7.1f  oal %8.1f  (flag %d, %d it)  %.0f s', ...
    stage, row.rms_waves, row.max_pull, row.petzval, Lr.R, Lr.z(end), flag, out.iterations, s);
end

function cd = convdiv_(L)
% per surface, the [converging, diverging] count a +1 / -1 shape contributes: air->glass +R converges,
% glass->air -R converges (iodc2010_forms' rule); cd(k, :) = [count for +1, count for -1] as +1 conv / -1 div
cd = zeros(L.N, 2);
n_after = zeros(1, L.N);  for k = 1:L.N, n_after(k) = iodc2010_glass(L.glass{k}, 532); end
nb = [1, n_after(1:end-1)];
for k = 1:L.N
    if n_after(k) > nb(k), cd(k, :) = [+1, -1]; else, cd(k, :) = [-1, +1]; end   % +1 shape: conv (+1) entering glass, div (-1) leaving
end
end

function imb = balance_(kk, fixed, cd)
% converging minus diverging surfaces among the FIXED shapes kk (+1 / 0 / -1)
imb = 0;
for k = find(fixed)
    if kk(k) == 1, imb = imb + cd(k, 1); elseif kk(k) == -1, imb = imb + cd(k, 2); end
end
end

function k = snap_(kap, kc)
% the nearest green shape: +-kc (the common curvature magnitude) or plano below kc / 2
if nargin < 2, kc = 1; end
k = kc * sign(kap);  k(abs(kap) < kc / 2) = 0;
end

function kc = common_(kap)
% the common curvature magnitude the free solution implies: the median |kappa| of its clearly
% curved surfaces (the units of kappa are arbitrary -- the lens is re-scaled to the EFL at every
% evaluation -- so a free solution whose curvatures sit at +-0.5 IS a green lens of twice the
% radius; snapping to kc = 1 would move every surface by half its curvature)
a = abs(kap);  a = a(a > 0.3 * max(a));
kc = median(a);
if ~isfinite(kc) || kc <= 0, kc = 1; end
end

function Ls = lens_(x, ctx)
% the relaxed lens for the variables: Rk = R0 / kappa (Inf = plano), gaps exp(x) R0, scaled to EFL 100,
% the stop at the seed's fraction of the length
L = ctx.L;  N = ctx.N;  R0 = ctx.R0;
kap = x(1:N);  Rk = R0 ./ kap;  Rk(abs(kap) < 1e-9) = Inf;
Ls = L;  Ls.R = R0;  Ls.Rk = Rk;  Ls.t = [exp(x(N+1:end)) * R0, NaN];  Ls.z = [0, cumsum(Ls.t(1:N-1))];
Ls.stop = [L.stop(1) / L.z(end) * Ls.z(end), 1];
try
    P = iodc2010_paraxial(Ls, ctx.pq.lambda_nm);
catch
    Ls.R = NaN;  return                                      % afocal / degenerate: the residual grades it
end
if ~isfinite(P.efl) || P.efl == 0, Ls.R = NaN;  return; end
sc = ctx.pq.efl_mm / abs(P.efl);
Ls.R = R0 * sc;  Ls.Rk = Rk * sc;  Ls.t = Ls.t * sc;  Ls.z = Ls.z * sc;  Ls.stop(1) = Ls.stop(1) * sc;
end

function pz = petzval_(kap, Ls)
% the Petzval sum in units of 1/R0: sum c_j (1/n_j' - 1/n_j) with the curvatures in units of the common one
n_after = zeros(1, Ls.N);  for k = 1:Ls.N, n_after(k) = iodc2010_glass(Ls.glass{k}, 532); end
nb = [1, n_after(1:end-1)];
pz = sum(kap .* (1 ./ n_after - 1 ./ nb));
end

function r = resid_(x, ctx, mu, snap)
N = ctx.N;  nr = ctx.nr;  pq = ctx.pq;  D = ctx.D;
r = zeros(1, nr + 2 * N + 1);
Ls = lens_(x, ctx);
if ~isfinite(Ls.R), r(1:nr) = 2e2;  return; end
try
    P = iodc2010_paraxial(Ls, pq.lambda_nm);
catch
    r(1:nr) = 2e2;  return
end
if ~isfinite(P.efl) || P.efl == 0 || P.z_img <= Ls.z(end) || ~isfinite(P.D) || P.D <= 0
    r(1:nr) = 1e2 * (1 + max(0, (Ls.z(end) - P.z_img)) / Ls.R);      % no real image: graded, not a cliff
    return
end
Ls.stop(2) = D / P.D;  Ls.semidia = inf(1, N);
P = iodc2010_paraxial(Ls, pq.lambda_nm);
% the beam inside every sphere (a ray outside it is a surface miss, which nothing can grade)
h = abs(P.h_marg) * D / 2;  c = iodc2010_curv(Ls);
over = max(0, abs(c) .* h - 0.9);
if any(over > 0), r(1:nr) = 1e2 * (1 + 10 * max(over));  return; end
E = iodc2010_emit(Ls, P, pq, ctx.deck);
macos.load_rx(E.path);
F = iodc2010_eval(E, P, pq, 0);
W = macos.opd();  M = macos.opd_mask();
macos.modify();
v = W(M & isfinite(W) & abs(W) < 1e30);
if numel(v) < 7 || ~isfinite(F.rms_waves)
    r(1:nr) = 1e2 * (1 + F.nLost / max(F.nLaunched, 1));  return
end
v = (v - mean(v)) / E.lambda_mm;  v = v(:).';
m = min(nr, numel(v));  r(1:m) = v(1:m);
r(nr + 1) = 1e2 * sqrt(nr) * F.nLost / F.nLaunched;                        % rays lost: a graded wall
kap = x(1:N);
r(nr+2:nr+1+N) = sqrt(mu) * sqrt(nr) * (kap - snap);                        % the pull toward the grid
r(nr+2+N) = ctx.wP * sqrt(nr) * petzval_(kap, Ls);                          % Petzval balance
% edge thickness at the marginal height, glass and air: a soft wall
sag = c .* h.^2 ./ (1 + sqrt(max(0, 1 - c.^2 .* h.^2)));
edge = Ls.t(1:N-1) - sag(1:N-1) + sag(2:N);
r(nr+3+N:nr+1+2*N) = 10 * sqrt(nr) * max(0, -edge) / Ls.R;
end

function iif_(c, varargin)
if c, fprintf(varargin{:}); end
end
