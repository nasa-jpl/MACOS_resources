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
%   opt.nph photons per measurement), the reading's DIFFERENTIAL to the
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
%     ins.noisy(F, nph, seed [, cam])
%                              the frames with photon noise for nph photons
%                              per measurement (one DM shape measured once;
%                              the reading's frames share it);
%                              nph = Inf returns F unchanged.  When the loop
%                              runs a CAMERA drift (opt.cam) it passes cam =
%                              struct('o', 'd'): the detector's additive
%                              offset at the start of this measurement's
%                              scan and its increment over the scan, in
%                              ELECTRONS per pixel (ins.npix x ins.npix);
%                              the instrument adds o + (j-1)/(nf-1) d to its
%                              j-th frame (nf frames), converted to frame
%                              units by its own photon scale
%     ins.npix                 (with opt.cam) the camera's side, pixels
%     ins.diff(F1, F0)         the reading's differential map, F1 minus F0
%     ins.est(map)             actuator-space estimate (nact x nact) of a map
%     ins.lit                  logical nact x nact: the actuators scored
%     ins.measure(cmd, aux)    (only when opt.intra or opt.ref_walk is on)
%                              the same capture, with the measurement's OWN
%                              drift: aux.dstep is a DM map that develops
%                              ACROSS the scan -- frame j of nf sees cmd +
%                              (j-1)/(nf-1) aux.dstep, so a simultaneous or
%                              single-frame reading sees NOTHING of it and a
%                              temporally stepped one does; aux.ref_phase is
%                              a phase (rad) added to the reference arm of a
%                              NON-COMMON-PATH reading for this measurement
%                              (the P/SRI's reference walk), zero for a
%                              common-path one
%     ins.recal(cmd)           (only when opt.recal_every > 0) re-measure the
%                              response matrix on the surface the loop is
%                              holding NOW, through the instrument's own
%                              calibration at DM command cmd; returns
%                              struct('est', <new estimator handle>,
%                              'nstates', <states it cost> [optional])
%   opt -- the loop (all optional; defaults in brackets):
%     .A0     set point, nact x nact                      [zeros]
%     .g      loop gain                                   [0.5]
%     .K      cycles                                      [60]
%     .nph    photons per measurement, one per cycle (Inf = noiseless) [Inf]
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
%     .start_rms  the loop STARTS from a surface of this rms (mm) instead of
%             from the set point -- the DM's initial figure, the capture
%             problem (Dave 2026-09-13): the starting surface is
%             start_rms * unit(.start_shape), so the residual at cycle 1 is
%             |start_rms - rms(A0)| when the shape is the set point's own.
%             [] or 0 = start at the set point
%     .start_shape  the starting surface's shape ([] = the set point A0
%             itself, rescaled -- "the same random field as the base,
%             scaled"; a seeded random field over lit when A0 is zero)
%     .recal_every  cycles between re-measurements of the response matrix ON
%             the loop's current surface, through ins.recal (0 = never: the
%             matrix measured once, at the start, is used throughout).  A
%             recalibration costs instrument states, counted in .nstates
%     .intra  fraction of the NEXT cycle's drift increment that develops
%             WITHIN one measurement's scan (the DM / thermal analogue of
%             cam.intra; 0 = the DM is still while a scan is taken).  What a
%             temporally stepped reading cannot remove, and what a
%             simultaneous one is for                                    [0]
%     .ref_walk  rms (rad per cycle) of a random walk of the reference
%             arm's phase relative to the test arm -- the NON-COMMON-PATH
%             term of a P/SRI, which no common-path reading has.  Passed to
%             the instrument as aux.ref_phase                            [0]
%     .ref_seed  its own stream                              [opt.seed + 2]
%     .reach  rms levels (mm) whose first cycle is reported in .k_reach --
%             the descent question ("how many cycles to 10 nm, to 3 pm")
%     .ref    'noiseless' | 'noisy' reference frames            ['noiseless']
%     .nss    cycles at the end averaged as the steady state     [floor(K/2)]
%     .track_noise  also estimate the noiseless frames each cycle to
%             measure the single-shot noise sigma_n in-run (one more
%             est call per cycle)                                [true if nph finite]
%     .rmax   residual rms above which the loop is declared DIVERGED and
%             stopped (the remaining cycles NaN; L.diverged true)   [Inf]
%     .cam    CAMERA 1/f drift (Dube et al. 2024: Roman LOWFS's error budget
%             is dominated by internal camera drift, ~1 electron per pixel
%             over 12 h, which temporal PSI high-pass filters because its
%             weights sum to zero within a scan): struct
%               .walk   electrons per pixel per CYCLE, the rms increment of
%                       a per-pixel random-walk offset (the 1/f model at
%                       the loop's time scale)                       [0 = off]
%               .intra  fraction of the next increment that develops WITHIN
%                       a measurement's scan (frame to frame; 0 = the
%                       offset is constant within a scan, so a zero-sum
%                       reading is exactly immune; 1 = the whole step
%                       develops across the frames)                  [0]
%               .seed   its own stream                        [opt.seed + 1]
%             The offset walks on the full ins.npix^2 camera; the reference
%             frames carry the offset at cycle 0 when .ref is 'noisy', none
%             when 'noiseless' (a calibration-grade reference), so a
%             reading that is NOT immune sees o_k - o_0, a random walk it
%             imprints on the DM
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
%     .k_reach  first cycle at or below each opt.reach level (NaN if never);
%               .surf_rms the rms of the STARTING surface (= rms(A0 + the
%               start offset)); .n_recal, .k_recal the recalibrations and
%               their cycles; .ref_phase the reference-arm phase per cycle
%     .diverged true when the residual exceeded opt.rmax (scores then use
%               the cycles that ran); .k_end the last cycle run
%     .r_final  the last residual map; .cmd the last command; .drift_rms
%               per-cycle rms over lit of the drift increments; .cam_rms
%               per-cycle rms of the camera offset (electrons; NaN when
%               off); .nstates
%               states measured (K + 1 reference); .opt the options used
%
%   Cost: K + 1 instrument states (the reference plus one per cycle), each
%   one measure + one noisy + one diff + one est (+ one est when tracking),
%   plus whatever each recalibration costs.  With opt.intra a stepped
%   reading's measurement traces its frames separately -- the instrument's
%   cost, not the loop's.

% ---- options ------------------------------------------------------------
lit = logical(ins.lit);  nact = size(lit, 1);
o = struct('A0', zeros(nact), 'g', 0.5, 'K', 60, 'nph', Inf, 'seed', 1, ...
           'drift', struct('kind', 'none'), 'ref', 'noiseless', 'nss', [], 'track_noise', [], 'rmax', Inf, ...
           'cam', struct('walk', 0, 'intra', 0, 'seed', []), ...
           'start_rms', [], 'start_shape', [], 'recal_every', 0, ...
           'intra', 0, 'ref_walk', 0, 'ref_seed', [], 'reach', []);
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
% the increments are drawn ONCE, in cycle order, for cycles 1..K+1: the extra
% one is the increment that develops across cycle K's scan under opt.intra.
% Drawing them ahead does not change the realization (same stream, same
% order), so every record taken before the intra knob existed reproduces.
D = zeros(nact, nact, o.K + 1);
for k = 1:o.K+1, D(:,:,k) = gen(k); end
cm = o.cam;
if ~isfield(cm, 'walk') || isempty(cm.walk), cm.walk = 0; end
if ~isfield(cm, 'intra') || isempty(cm.intra), cm.intra = 0; end
if ~isfield(cm, 'seed') || isempty(cm.seed), cm.seed = o.seed + 1; end
o.cam = cm;
camon = cm.walk > 0;
if camon
    assert(isfield(ins, 'npix') && ~isempty(ins.npix), 'dmg_loop: opt.cam needs ins.npix (the camera side, pixels)');
    sc = RandStream('mt19937ar', 'Seed', cm.seed);           % the camera stream
    np = ins.npix;  ocam = zeros(np);                         % the offset at cycle 0
    camstep = @() cm.walk * randn(sc, np);
    dnext = camstep();                                        % the increment that develops over cycle 1
    camarg = @(oo, dd) {struct('o', oo, 'd', cm.intra*dd)};
else
    camarg = @(oo, dd) {};
    ocam = [];  dnext = [];
end

% ---- the reference arm's own phase walk (non-common path) -------------------
if isempty(o.ref_seed), o.ref_seed = o.seed + 2; end
psi = zeros(1, o.K);
if o.ref_walk > 0
    sr = RandStream('mt19937ar', 'Seed', o.ref_seed);
    psi = cumsum(o.ref_walk * randn(sr, 1, o.K));      % a random walk from the reference capture
end
aux_on = (o.intra > 0) || (o.ref_walk > 0);
if aux_on
    nm_ = nargin(ins.measure);            % 2 = (cmd, aux); negative = varargin
    assert(nm_ == 2 || nm_ < 0, ...
        'dmg_loop: opt.intra / opt.ref_walk need an instrument whose measure takes (cmd, aux)');
end

% ---- the starting surface (the DM's initial figure) -------------------------
% dist carries the surface's departure from the command, so a start offset is
% simply the disturbance the loop opens with: s(1) = start_rms * unit(shape).
d0 = zeros(nact);
if ~isempty(o.start_rms) && o.start_rms > 1e-2       % 10 um: beyond any DM stroke -- a bare-nanometre slip
    warning('dmg:loop:startUnits', ['dmg_loop: start_rms = %g is read in mm, like every rms knob ' ...
        '(base_rms 30e-6 = 30 nm, steps 1e-6 = 1 nm, walk_sigma 2e-9 = 2 pm): that is a %g mm starting ' ...
        'surface.  A %g nm start is %g.  (TO 2026-09-15: two queued sequences carried the bare number.)'], ...
        o.start_rms, o.start_rms, o.start_rms, o.start_rms*1e-6);
end
if ~isempty(o.start_rms) && o.start_rms > 0
    shp = o.start_shape;
    if isempty(shp), shp = o.A0; end
    if rmsl(shp) == 0, shp = randn(sd, nact) .* lit; end    % a flat set point: a seeded field
    d0 = o.start_rms * (shp / rmsl(shp)) - o.A0;
end

% ---- the reference (set point) --------------------------------------------
Fref = ins.measure(o.A0);
nseed = @(k) double(mod(uint64(o.seed)*100003 + 7919 + uint64(k), 2^32));   % per-cycle noise seed
if strcmp(o.ref, 'noisy'), ca = camarg(ocam, dnext);  Fref = ins.noisy(Fref, o.nph, nseed(0), ca{:}); end
nstates = 1;

% ---- the loop -----------------------------------------------------------
cmd = o.A0;  dist = d0;
rms = nan(1, o.K);  drms = nan(1, o.K);  en = nan(1, o.K);  enu = nan(1, o.K);  crms = nan(1, o.K);
R = zeros(nact, nact, o.nss);  nR = 0;                       % residual maps of the tail
diverged = false;  k_end = o.K;
est = ins.est;  k_recal = [];                                % the estimator in force
for k = 1:o.K
    d = D(:,:,k);  dist = dist + d;  drms(k) = rmsl(d);
    s = cmd + dist;  r = s - o.A0;  rms(k) = rmsl(r);
    if rms(k) > o.rmax, diverged = true;  k_end = k;  break; end
    if k > o.K - o.nss, nR = nR + 1;  R(:,:,nR) = r; end
    if aux_on
        F = ins.measure(s, struct('dstep', o.intra * D(:,:,k+1), 'ref_phase', psi(k)));
    else
        F = ins.measure(s);
    end
    nstates = nstates + 1;
    if camon, ocam = ocam + dnext;  dnext = camstep();  crms(k) = sqrt(mean(ocam(:).^2)); end    % the offset at this cycle's scan
    ca = camarg(ocam, dnext);  Fn = ins.noisy(F, o.nph, nseed(k), ca{:});
    a = est(ins.diff(Fn, Fref));
    if o.track_noise
        a0 = est(ins.diff(F, Fref));  e = a - a0;
        en(k) = rmsl(e);  un = ~lit;  enu(k) = sqrt(mean(e(un).^2));
    end
    cmd = cmd - o.g * a;
    if o.recal_every > 0 && mod(k, o.recal_every) == 0 && k < o.K
        % re-measure the response matrix ON the surface the loop holds now
        rc = ins.recal(cmd);
        est = rc.est;  k_recal(end+1) = k; %#ok<AGROW>
        if isfield(rc, 'nstates') && ~isempty(rc.nstates), nstates = nstates + rc.nstates; end
    end
end

% ---- scores ---------------------------------------------------------------
L = struct('rms', rms, 'drift_rms', drms, 'cam_rms', crms, 'nstates', nstates, 'opt', o, 'diverged', diverged, 'k_end', k_end);
if diverged
    tail = max(1, k_end-o.nss+1):k_end;  R = R(:,:,1:max(nR,1));   % what ran
else
    tail = o.K-o.nss+1:o.K;
end
L.ss = sqrt(mean(rms(tail).^2));
L.bias_map = mean(R, 3);  L.bias = rmsl(L.bias_map);
ok = ~isnan(en);  L.sig_n = sqrt(mean(en(ok).^2));  L.sig_n_unlit = sqrt(mean(enu(ok).^2));
L.r_final = r;  L.cmd = cmd;
L.surf_rms = rmsl(o.A0 + d0);                                 % the STARTING surface
L.n_recal = numel(k_recal);  L.k_recal = k_recal;  L.ref_phase = psi;
L.k_reach = nan(1, numel(o.reach));                           % cycles to each level
for q = 1:numel(o.reach)
    i = find(rms <= o.reach(q), 1);  if ~isempty(i), L.k_reach(q) = i; end
end
descent = ~isempty(o.start_rms) && o.start_rms > 0;
if (strcmp(dr.kind, 'step') || descent) && ~diverged
    k0 = 1;  if strcmp(dr.kind, 'step'), k0 = dr.at; end      % a descent starts at cycle 1
    [L.rho, L.tau, L.k_1e] = decay_(rms(k0:end), L.ss);
else, L.rho = NaN;  L.tau = NaN;  L.k_1e = NaN; end          % a transient is fitted on a step / descent run only
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
