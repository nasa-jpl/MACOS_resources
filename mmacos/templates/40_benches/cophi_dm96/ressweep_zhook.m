function R = ressweep_zhook(S)
%RESSWEEP_ZHOOK  The resolution sweep's rows for the ZERNIKE-family gauges (stepped
%   ZWFS S, vector ZWFS V, pinhole P) at ONE sampling, on zwfs_run's battery matrix
%   (P.battery.deck_hook = 'ressweep_zhook'; the IFO / COPHI twin is ressweep_hook,
%   whose header carries the definitions -- the same five quantities here).
%
%   Differences from the interferometers, all from the record (zwfs_run's camera
%   form, camS_oap): V's pair in ONE exposure of the polarization camera (each image
%   on a quarter of the pixels, k per orientation); S's four frames and P's K frames
%   SEQUENTIAL on the mono sensor; each IMAGE's brightest pixel at fill x well (the
%   ZWFS core sets the exposure); the solvers' intensity calibrations (the flat's
%   |E0|^2, |b|^2, b2cal) on the camera's pixels (dmg_zwfs_gauge / dmg_pdi_gauge
%   under P.camres).  Estimates one-step (the record's) and converged (dmg_converge
%   through the measured J).  The loop's cycle per reading holds ~3 pm at the
%   record's sampling (M readings: 5.2 pm per cycle from camS_oap's pm per reading).
say = S.say;  P = S.P;  ZW = S.ZW;  RD = S.RD;  KC = S.KC;
ic = S.ic;  lit = S.lit;  nact = S.nact;  msk = S.msk;  N = size(msk, 1);
cfg = dmg_camres('get');  c = 1;  if ~isempty(cfg), c = cfg.c; end
ppa = S.px_per_act_model;  s_eff = ppa / c;
RS = P.ressweep;
dev = zeros(nact);  dev(ic(1), ic(2)) = S.dev;
un = lit;  un(ic(1), ic(2)) = false;  ilit = find(lit);
say('\n---- ressweep (ZWFS family %s): %.3f model px per actuator / %.4f model px per camera px = %.3f CAMERA px per actuator ----\n', ...
    strjoin(RD, ' '), ppa, c, s_eff);

% ---- the two states, traced ONCE, unbinned -------------------------------
dmg_camres('suspend', true);  clsus = onCleanup(@() dmg_camres('suspend', false));
t0 = tic;  F0u = S.framesf(S.A0);  F1u = S.framesf(S.A0 + dev);
dmg_camres('suspend', false);  clear clsus
say('  traced the set point and the single (%.0f s)\n', toc(t0));
cf0 = cfg;  if isempty(cf0), cf0 = struct('c', 1, 'offset', [0 0], 'center', []); end
[F0, C0, T0] = binF_(F0u, cf0, ZW);  [F1, C1] = binF_(F1u, cf0, ZW);
fwd = cell(1, 6);  estc = cell(1, 6);
for kc = unique(KC)
    Jk = S.J{kc};
    fwd{kc} = @(a) reshape(Jk*a(ilit), N, N) .* msk;
    estc{kc} = @(h) dmg_converge(S.est{kc}, fwd{kc}, h, lit, 1e-12, 50);
end
R = struct('c', c, 'ppa_model', ppa, 's', s_eff, 'ncell', [T0.ncy T0.ncx], 'dmod', S.dmod, ...
    'readings', {RD}, 'res', S.res, 'nlit', S.nlit, 'nstates', S.nstates);

% ---- the battery rows again, converged (the record's are one-step) --------
say('  battery rows, CONVERGED (dmg_converge through the measured J; the one-step rows are the battery''s above):\n');
rows = S.ROWS;                                   % the battery's own rows (name, base, change)
conv = nan(size(rows, 1), numel(RD), 3);
for r = 1:size(rows, 1)
    % frames on the camera (the factories bin at capture under P.camres); the
    % working-surface base is the set point already traced, the single its change
    assert(isequal(rows{r, 2}, S.A0), 'ressweep_zhook: row %d''s base is not the set point', r);
    Fb = F0;
    if isequal(rows{r, 3}, dev), Fa = F1; else, Fa = S.framesf(rows{r, 2} + rows{r, 3}); end
    for k = 1:numel(RD)
        [a, inf_] = estc{KC(k)}(S.hdiff(Fa, Fb, k));
        [g, e, fl] = score_(a, rows{r, 3}, lit);
        conv(r, k, :) = [g e fl];
        say('    %-17s %-2s %8.4f %8.1f %8s   (%d steps)\n', rows{r, 1}, RD{k}, g, e, f0_(fl), inf_.iters);
    end
end
R.rows_conv = conv;  R.rows_conv_names = rows(:, 1);

% ---- (4) registration ------------------------------------------------------
sh = RS.reg_pitch * ppa;
offs = [0 0; sh 0; -sh 0; 0 sh; 0 -sh];
say('  REGISTRATION (camera shifted %g pitch = %.3f model px after calibration): per reading gain one-step / converged, largest neighbour / dev, floor pm, wrong\n', RS.reg_pitch, sh);
nb = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
reg = nan(size(offs, 1), numel(RD), 6);
for q = 1:size(offs, 1)
    cq = cf0;  cq.offset = offs(q, :);
    if q == 1, G0 = F0;  G1 = F1; else, G0 = binF_(F0u, cq, ZW);  G1 = binF_(F1u, cq, ZW); end
    for k = 1:numel(RD)
        h = S.hdiff(G1, G0, k);  a = S.est{KC(k)}(h);  ac = estc{KC(k)}(h);
        [g, nbm, fl, wr] = regscore_(a, ic, nb, S.dev, un, nact);  gc = ac(ic(1), ic(2))/S.dev;
        reg(q, k, :) = [g gc nbm fl wr 0];
        say('    %+5.2f %+5.2f  %-2s %8.4f %8.4f %8.4f %8.2f %3d\n', offs(q, :), RD{k}, g, gc, nbm, fl, wr);
    end
end
R.reg = reg;  R.offs = offs;

% ---- (2) noise --------------------------------------------------------------
cams = RS.cams;  nr = RS.nreal;  pf = 0.95;
a_cl = cell(1, numel(RD));
for k = 1:numel(RD), a_cl{k} = S.est{KC(k)}(S.hdiff(F1, F0, k)); end
say('  NOISE (%d draws; record cameras binned to this sampling; one reading = differential / sqrt 2):\n', nr);
say('  %-3s %-8s %-7s %6s %8s %9s %9s %9s %9s %9s %8s\n', 'rd', 'camera', 'binning', 'k', 'phys/px', 'read e-', 'pm/read', 'hold-out', 'reads/s', 's per pm', 'Gb/s');
NZ = [];
for k = 1:numel(RD)
    cam = cams.(RD{k});
    kk = pf*cam.npix/(1 + (cam.pol ~= 0)) / S.dmod;
    for bm = {'digital', 'onchip'}
        rd = cam.read * iff_(strcmp(bm{1}, 'digital'), kk*c, 1);
        ce = struct('well', cam.well*kk^2, 'read', rd, 'fill', cam.fill, 'c', c);
        pk = zeros(nr, 1);  fl = zeros(nr, 1);
        for r = 1:nr
            G0 = noisyF_(C0, F0u, T0, ce, RD{k}, 1, 2e4 + 100*k + 2*r, ZW);
            G1 = noisyF_(C1, F1u, T0, ce, RD{k}, 1, 2e4 + 100*k + 2*r + 1, ZW);
            a = S.est{KC(k)}(S.hdiff(G1, G0, k)) - a_cl{k};  a = a - mean(a(lit));
            pk(r) = a(ic(1), ic(2));  fl(r) = 1.4826*median(abs(a(un) - median(a(un))));
        end
        flr = median(fl)/sqrt(2)*1e9;  sgh = std(pk)/sqrt(2)*1e9;
        rate = cam.rate / nfr_(RD{k}, F0u);
        npx = (cam.npix/(kk*c))^2;  gbs = npx*16*cam.rate/1e9;
        say('  %-3s %-8s %-7s %6.2f %8.2f %9.3g %9.3g %9.3g %9.3g %9.3g %8.3g\n', RD{k}, cam.tag, bm{1}, kk, (kk*c)^2, rd, flr, sgh, rate, max(flr^2, 1)/rate, gbs);
        NZ = [NZ; struct('rd', RD{k}, 'cam', cam.tag, 'binning', bm{1}, 'k', kk, 'phys_per_px', (kk*c)^2, 'read_e', rd, ...
            'pm_per_reading', flr, 'holdout_pm', sgh, 'readings_per_s', rate, 's_per_pm', max(flr^2, 1)/rate, ...
            'npix_out', npx, 'gbps', gbs)]; %#ok<AGROW>
    end
end
R.noise = NZ;
R.pupil_px = pi/4*(S.dmod/c)^2;
say('  data: %.0f camera px across the pupil, %.3g per reading over the pupil (V and P solve each one, %d iterations of the reference wave)\n', ...
    S.dmod/c, R.pupil_px, ZW.NITER + 1);

% ---- (3) the loop, linear instrument (digital binning) ----------------------
LP = [];
if RS.loop
    for k = 1:numel(RD)
        cam = cams.(RD{k});  M = RS.M.(RD{k});
        kk = pf*cam.npix/(1 + (cam.pol ~= 0)) / S.dmod;
        ce = struct('well', cam.well*kk^2, 'read', cam.read*kk*c, 'fill', cam.fill, 'c', c);
        cache = containers.Map('KeyType', 'double', 'ValueType', 'any');
        nmap = @(seed) nmap_(cache, seed, C0, F0u, T0, ce, RD{k}, M, ZW, S, F0, k);
        ins = struct('measure', @(cmd) fwd{KC(k)}(cmd - S.A0), 'noisy', @(Fm, nph, seed) Fm + nmap(seed), ...
            'diff', @(Fa, Fb) Fa - Fb, 'est', S.est{KC(k)}, 'lit', lit);
        cyc = M * nfr_(RD{k}, F0u) / cam.rate;
        for dk = {'none', 'thermal'}
            o = struct('A0', S.A0, 'g', P.loop.g, 'K', P.loop.K, 'nph', 1, 'seed', P.loop.seed, 'rmax', 1e-3);
            if strcmp(dk{1}, 'thermal'), o.drift = struct('kind', 'thermal', 'rate', P.loop.thermal_rate, 'shape', []);
            else, o.drift = struct('kind', 'none'); end
            t1 = tic;  L = dmg_loop(ins, o);
            th = L.theory.ss_noise;
            if strcmp(dk{1}, 'thermal') && ~isnan(L.theory.lag_ramp), th = hypot(L.theory.lag_ramp, L.theory.ss_noise); end
            say('  LOOP %-2s %-8s M %5d (cycle %.3g s) %-8s ss %.3g pm (piston-free %.3g), sig_n %.3g pm, theory %.3g pm, gG %.3f%s  [%.0f s]\n', ...
                RD{k}, cam.tag, M, cyc, dk{1}, 1e9*L.ss, 1e9*L.ss_pf, 1e9*L.sig_n, 1e9*th, L.theory.gG, iff_(L.diverged, ' DIVERGED', ''), toc(t1));
            LP = [LP; struct('rd', RD{k}, 'cam', cam.tag, 'M', M, 'cycle_s', cyc, 'drift', dk{1}, 'ss_pm', 1e9*L.ss, 'ss_pf_pm', 1e9*L.ss_pf, ...
                'sig_n_pm', 1e9*L.sig_n, 'theory_pm', 1e9*th, 'gG', L.theory.gG, 'diverged', L.diverged)]; %#ok<AGROW>
        end
    end
end
R.loop = LP;
end

% =====================================================================
function [Fb, Cc, T] = binF_(F, cf, ZW)
% bin every captured image of a frames_ struct onto the camera's pixels (cf; [] =
% the frames are already as captured); the stepped retrieval X re-made from the
% binned frames (reconS under the active, unsuspended camera: b2cal binned too)
Fb = F;  Cc = struct();  T = struct('id', true, 'ncy', NaN, 'ncx', NaN);
if isempty(cf), return; end
id = cf.c == 1 && all(cf.offset == 0);
for f = {'Ia', 'Ip', 'Im', 'Fr', 'FP', 'FF'}
    x = F.(f{1});
    if isempty(x), continue; end
    if id
        Cc.(f{1}) = x;  T = struct('id', true, 'ncy', size(x, 1), 'ncx', size(x, 2));
    else
        [B, T] = dmg_camres('bin', x, cf);  T.id = false;
        Cc.(f{1}) = B;  Fb.(f{1}) = dmg_camres('rep', B, T);
    end
end
if ~isempty(F.Fr) && ~id, Fb.X = ZW.reconS(Fb.Fr); end
end

function n = nfr_(rd, F)
switch rd
    case 'S', n = 4;
    case 'V', n = 1;                         % the pair in one polarization exposure
    case 'P', n = size(F.FP, 3);
    otherwise, n = 1;
end
end

function G = noisyF_(Cc, Fu, T, ce, rd, M, seed, ZW)
% the reading's images on the camera: each IMAGE's exposure from its unbinned
% brightest model pixel (fill x well x k^2), shot + read per CAMERA pixel, M
% readings averaged (well x M, read x sqrt M), replicated back
rs = RandStream('mt19937ar', 'Seed', seed);
G = struct('Ia', [], 'Ip', [], 'Im', [], 'Fr', [], 'X', [], 'FP', [], 'FF', []);
switch rd
    case 'S', fl = {'Fr'};
    case 'V', fl = {'Ip', 'Im'};
    case 'P', fl = {'FP'};
    otherwise, fl = {'Ia'};
end
for f = fl
    B = Cc.(f{1});  U = Fu.(f{1});  Bn = B;
    for j = 1:size(B, 3)
        Uj = U(:, :, j);  sc = ce.fill*ce.well / max(Uj(:));        % e- per intensity unit per MODEL pixel
        a = sc * ce.c^2 * M;  r = ce.read * sqrt(M);
        E = B(:, :, j) * a;
        Bn(:, :, j) = (E + sqrt(max(E, 0)) .* randn(rs, size(E)) + r*randn(rs, size(E))) / a;
    end
    if T.id, G.(f{1}) = Bn; else, G.(f{1}) = dmg_camres('rep', Bn, T); end
end
if ~isempty(G.Fr), G.X = ZW.reconS(G.Fr); end
end

function h = nmap_(cache, seed, C0, F0u, T, ce, rd, M, ZW, S, F0, k)
if isKey(cache, seed), h = cache(seed); return; end
h = S.hdiff(noisyF_(C0, F0u, T, ce, rd, M, seed, ZW), F0, k);
cache(seed) = h; %#ok<NASGU>
end

function [g, e, fl] = score_(a, Ad, lit)
g = Ad(lit) \ a(lit);  e = sqrt(mean((a(lit) - Ad(lit)).^2))*1e9;
pk = (Ad ~= 0) & lit;  un = lit & ~pk;
if nnz(pk) < nnz(lit)/4, fl = std(a(un))*1e9; else, fl = NaN; end
end

function [g, nbm, fl, wr] = regscore_(a, ic, nb, d, un, nact)
g = a(ic(1), ic(2)) / d;  v = zeros(size(nb, 1), 1);
for j = 1:size(nb, 1)
    p = ic + nb(j, :);
    if all(p >= 1 & p <= nact), v(j) = abs(a(p(1), p(2))); end
end
nbm = max(v) / d;  fl = std(a(un))*1e9;
[mx, im] = max(abs(a(:)));  wr = mx > 0 && im ~= sub2ind([nact nact], ic(1), ic(2));
end

function s = f0_(x), if isnan(x), s = '-'; else, s = sprintf('%.0f', x); end, end
function s = iff_(c, a, b), if c, s = a; else, s = b; end, end
