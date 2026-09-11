function L = dmg_loop(ins, opt)
%DMG_LOOP  Closed-loop hold of a DM surface through a gauge -- the shared loop.
%   L = dmg_loop(ins, opt) drives a DM against a specified drift with a
%   proportional loop closed through an instrument, and scores the HOLD
%   ERROR: the rms surface error over the lit actuators, cycle by cycle.
%   One implementation for every gauge (ZWFS, T-G IFO, a synthetic
%   instrument for the unit test), so the comparison between them is the
%   instrument and nothing else (BRIEF_loop_metric.md, Dave 2026-09-11).
%
%   THE LOOP.  The set point is the working surface A0 (command space,
%   nact x nact).  Its frames are captured ONCE as the reference.  Per
%   cycle k: the disturbance advances (drift), the surface s = cmd + dist
%   is measured (frames captured noiseless, photon noise injected for
%   opt.nph photons per state), the reading's DIFFERENTIAL to the
%   reference is fitted to actuator changes a_hat, and cmd <- cmd -
%   g*a_hat.  The residual r_k = s - A0 is recorded at measurement time
%   (the error that exists during the cycle).  Differencing against the
%   set-point frames is the same protocol as every differential row of
%   the campaign; it is also what a running differential against the
%   PREVIOUS cycle's frames integrates to (the sum telescopes), with one
%   difference: a noisy reference is then a FIXED bias the loop converges
%   to.  opt.ref selects a noiseless (calibration-grade) or single-shot
%   reference so that term can be measured.
%
%   Dynamics (G the reading's gain, e_k the single-shot estimate noise,
%   d_k the drift increment):  r_{k+1} = (1 - g G) r_k - g G e_k + d_{k+1}
%     noise only:   rms_ss = sigma_n * sqrt(g G / (2 - g G))
%     random walk:  rms_ss^2 = (sigma_d^2 + g^2 G^2 sigma_n^2) / (g G (2 - g G))
%     ramp r/cycle: lag = r / (g G)  (the residual settles at the ramp's
%                   shape times that), plus the noise term
%   L.theory carries these for the run's own sigma_n and drift; the gates
%   in tDmgLoop pin the loop code to them on a synthetic instrument.
%
%   ins -- the instrument (function handles; any frame type):
%     ins.measure(cmd)         frames of the DM at command cmd, NOISELESS
%     ins.noisy(F, nph, seed)  the frames with photon noise for nph photons
%                              per state (the reading's frames share it);
%                              nph = Inf returns F unchanged
%     ins.diff(F1, F0)         the reading's differential map, F1 minus F0
%     ins.est(map)             actuator-space estimate (nact x nact) of a map
%     ins.lit                  logical nact x nact: the actuators scored
%   opt -- the loop (all optional; defaults in brackets):
%     .A0     set point, nact x nact                      [zeros]
%     .g      loop gain                                   [0.5]
%     .K      cycles                                      [60]
%     .nph    photons per state per cycle (Inf = noiseless) [Inf]
%     .seed   seeds the drift stream (opt.seed) and the per-cycle noise
%             seeds (a hash of opt.seed and the cycle, disjoint between
%             run seeds): the drift realization is the same at every
%             photon level and on every instrument                    [1]
%     .drift  struct: .kind 'none' | 'walk' | 'thermal' | 'step'
%             'walk'    .sigma  per-actuator Gaussian increment per cycle
%             'thermal' .rate   per cycle, rms over lit, of .shape (a map;
%                               [] = defocus + astigmatism over the lit disc)
%             'step'    .amp    rms over lit of one disturbance at cycle
%                               .at [1]; .shape a map ([] = seeded random
%                               over lit)                       [none]
%     .ref    'noiseless' | 'noisy' reference frames            ['noiseless']
%     .nss    cycles at the end averaged as the steady state     [floor(K/2)]
%     .track_noise  also estimate the noiseless frames each cycle to
%             measure the single-shot noise sigma_n in-run (one more
%             est call per cycle)                                [true if nph finite]
%     .rmax   residual rms above which the loop is declared DIVERGED and
%             stopped (the remaining cycles NaN; L.diverged true)   [Inf]
%   L -- the record (surface units are the instrument's; ZWFS = mm):
%     .rms      1 x K residual rms over lit at each cycle
%     .ss       steady-state rms: root mean square of .rms over the last nss cycles
%     .bias     rms over lit of the MEAN residual over the last nss cycles
%               (noise averages down as 1/sqrt(nss); a bias stays)
%     .bias_map that mean residual (nact x nact)
%     .sig_n    single-shot estimate noise, rms over lit and cycles (NaN
%               when not tracked); .sig_n_unlit the same off the lit set
%     .rho, .tau, .k_1e  per-cycle contraction fitted on the decay (log-
%               linear over the cycles above 3x the steady state), the time
%               constant -1/log(rho) in cycles, and the first cycle at
%               which rms < rms(1)/e (NaN if never)
%     .spec     steady-state residual power spectrum, radial in cycles per
%               aperture: .f (bin centres), .rms (rms in each bin), .band
%               = rms in [<4, 4-12, >12] cycles/aperture
%     .theory   .ss_noise, .ss_walk, .lag_ramp as above from .sig_n and
%               the drift (NaN where not applicable)
%     .diverged true when the residual exceeded opt.rmax (scores then use
%               the cycles that ran); .k_end the last cycle run
%     .r_final  the last residual map; .cmd the last command; .drift_rms
%               per-cycle rms over lit of the drift increments; .nstates
%               states measured (K + 1 reference); .opt the options used
%
%   Cost: K + 1 instrument states (the reference plus one per cycle), each
%   one measure + one noisy + one diff + one est (+ one est when tracking).

% ---- options ------------------------------------------------------------
lit = logical(ins.lit);  nact = size(lit, 1);
o = struct('A0', zeros(nact), 'g', 0.5, 'K', 60, 'nph', Inf, 'seed', 1, ...
           'drift', struct('kind', 'none'), 'ref', 'noiseless', 'nss', [], 'track_noise', [], 'rmax', Inf);
if nargin > 1
    fn = fieldnames(opt);
    for i = 1:numel(fn), o.(fn{i}) = opt.(fn{i}); end
end
if isempty(o.nss), o.nss = floor(o.K/2); end
if isempty(o.track_noise), o.track_noise = isfinite(o.nph); end
o.nss = max(1, min(o.nss, o.K));
dr = o.drift;  if ~isfield(dr, 'kind'), dr.kind = 'none'; end
sd = RandStream('mt19937ar', 'Seed', o.seed);              % the drift stream
rmsl = @(m) sqrt(mean(m(lit).^2));

% ---- the drift generator -------------------------------------------------
switch dr.kind
    case 'none'
        gen = @(k) zeros(nact);
    case 'walk'
        gen = @(k) walk_(sd, dr.sigma, lit, nact);
    case 'thermal'
        if ~isfield(dr, 'shape') || isempty(dr.shape), dr.shape = lowshape_(lit, nact); end
        shp = dr.shape / rmsl(dr.shape);
        gen = @(k) dr.rate * shp;
    case 'step'
        if ~isfield(dr, 'at'), dr.at = 1; end
        if ~isfield(dr, 'shape') || isempty(dr.shape)
            dr.shape = randn(sd, nact) .* lit;               % drawn on the full grid, masked:
        end                                                  % the same pattern on any instrument
        shp = dr.shape / rmsl(dr.shape);
        gen = @(k) (k == dr.at) * dr.amp * shp;
    otherwise
        error('dmg_loop: drift.kind must be none | walk | thermal | step');
end
o.drift = dr;

% ---- the reference (set point) --------------------------------------------
Fref = ins.measure(o.A0);
nseed = @(k) double(mod(uint64(o.seed)*100003 + 7919 + uint64(k), 2^32));   % per-cycle noise seed
if strcmp(o.ref, 'noisy'), Fref = ins.noisy(Fref, o.nph, nseed(0)); end
nstates = 1;

% ---- the loop -----------------------------------------------------------
cmd = o.A0;  dist = zeros(nact);
rms = nan(1, o.K);  drms = nan(1, o.K);  en = nan(1, o.K);  enu = nan(1, o.K);
R = zeros(nact, nact, o.nss);  nR = 0;                       % residual maps of the tail
diverged = false;  k_end = o.K;
for k = 1:o.K
    d = gen(k);  dist = dist + d;  drms(k) = rmsl(d);
    s = cmd + dist;  r = s - o.A0;  rms(k) = rmsl(r);
    if rms(k) > o.rmax, diverged = true;  k_end = k;  break; end
    if k > o.K - o.nss, nR = nR + 1;  R(:,:,nR) = r; end
    F = ins.measure(s);  nstates = nstates + 1;
    Fn = ins.noisy(F, o.nph, nseed(k));
    a = ins.est(ins.diff(Fn, Fref));
    if o.track_noise
        a0 = ins.est(ins.diff(F, Fref));  e = a - a0;
        en(k) = rmsl(e);  un = ~lit;  enu(k) = sqrt(mean(e(un).^2));
    end
    cmd = cmd - o.g * a;
end

% ---- scores ---------------------------------------------------------------
L = struct('rms', rms, 'drift_rms', drms, 'nstates', nstates, 'opt', o, 'diverged', diverged, 'k_end', k_end);
if diverged
    tail = max(1, k_end-o.nss+1):k_end;  R = R(:,:,1:max(nR,1));   % what ran
else
    tail = o.K-o.nss+1:o.K;
end
L.ss = sqrt(mean(rms(tail).^2));
L.bias_map = mean(R, 3);  L.bias = rmsl(L.bias_map);
ok = ~isnan(en);  L.sig_n = sqrt(mean(en(ok).^2));  L.sig_n_unlit = sqrt(mean(enu(ok).^2));
L.r_final = r;  L.cmd = cmd;
if strcmp(dr.kind, 'step') && ~diverged, [L.rho, L.tau, L.k_1e] = decay_(rms(dr.at:end), L.ss);
else, L.rho = NaN;  L.tau = NaN;  L.k_1e = NaN; end          % a transient is fitted on a step run only
L.spec = spec_(L.bias_map, R, lit, nact);
gG = o.g;                                                     % G folded into rho when measured
if ~isnan(L.rho), gG = 1 - L.rho; end
L.theory = struct('gG', gG, 'ss_noise', L.sig_n*sqrt(gG/(2-gG)), 'ss_walk', NaN, 'lag_ramp', NaN);
switch dr.kind
    case 'walk',    L.theory.ss_walk  = sqrt((dr.sigma^2 + gG^2*L.sig_n^2) / (gG*(2-gG)));
    case 'thermal', L.theory.lag_ramp = dr.rate / gG;
end
end

% =====================================================================
function d = walk_(sd, sigma, lit, nact)
% drawn on the full grid and masked, so two instruments with slightly
% different lit sets see the SAME realization where both are lit
d = sigma * randn(sd, nact) .* lit;
end

function s = lowshape_(lit, nact)
% defocus + astigmatism over the lit disc, unit-free (normalized by the caller)
[c, r] = meshgrid(1:nact, 1:nact);
[rl, cl] = find(lit);  c0 = mean(cl);  r0 = mean(rl);  rad = max(hypot(cl-c0, rl-r0));
x = (c - c0)/rad;  y = (r - r0)/rad;
s = (2*(x.^2 + y.^2) - 1) + 0.5*(x.^2 - y.^2);
s(~lit) = 0;
end

function [rho, tau, k1e] = decay_(rms, ss)
% per-cycle contraction fitted log-linearly over the decay: the cycles
% before the residual first reaches 3x the steady state (or 1e-6 of the
% start, whichever comes first); needs >= 3 cycles of decay
rho = NaN;  tau = NaN;  k1e = NaN;
K = numel(rms);  if K < 3 || ~(rms(1) > 0), return; end
flo = max(3*ss, 1e-6*rms(1));
kd = find(rms <= flo, 1);  if isempty(kd), kd = K; end
kd = kd - 1;
if kd >= 3
    p = polyfit(1:kd, log(rms(1:kd)), 1);  rho = exp(p(1));  tau = -1/log(rho);
end
i = find(rms < rms(1)/exp(1), 1);  if ~isempty(i), k1e = i; end
end

function S = spec_(bias_map, R, lit, nact)
% radial power spectrum of the steady-state residual (mean over the tail's
% cycles of the per-cycle spectrum), in cycles per aperture; the lit disc
% is the aperture, maps are zero outside it
[rl, cl] = find(lit);  D = max(max(cl)-min(cl), max(rl)-min(rl)) + 1;   % aperture, actuators
n = size(R, 3);  P = zeros(nact);
for i = 1:n
    m = R(:,:,i);  m(~lit) = 0;  P = P + abs(fft2(m)).^2;
end
P = P / n / (nact^2 * nlit_(lit));                            % sum over f = mean square per lit actuator
fx = (0:nact-1)/nact;  fx(fx > 0.5) = fx(fx > 0.5) - 1;       % cycles per actuator
[FX, FY] = meshgrid(fx, fx);  f = hypot(FX, FY) * D;          % cycles per aperture
edges = [0 2 4 8 12 16 24 32 48 max(f(:))+1];
nb = numel(edges) - 1;  fr = nan(1, nb);  fc = nan(1, nb);
for b = 1:nb
    sel = f >= edges(b) & f < edges(b+1);
    if any(sel(:)), fr(b) = sqrt(sum(P(sel)));  fc(b) = mean(f(sel)); end
end
band = [sqrt(sum(P(f < 4))), sqrt(sum(P(f >= 4 & f < 12))), sqrt(sum(P(f >= 12)))];
S = struct('f', fc, 'rms', fr, 'edges', edges, 'band', band, 'D', D, 'bias_rms', sqrt(mean(bias_map(lit).^2)));
end

function n = nlit_(lit)
n = nnz(lit);
end
