function R = ressweep_hook(S)
%RESSWEEP_HOOK  The resolution sweep's rows at ONE sampling, on the deck stage's own
%   matrix (tg96_run P.battery.deck_hook = 'ressweep_hook'; BRIEF_to_resolution_sweep).
%
%   tg96_run's deck stage has already run placement, the 64-state matrix and the
%   battery rows THROUGH the camera's pixels (P.camres -> dmg_camres in every
%   frame).  This hook adds, on that matrix (S.est30 / S.est30c / S.fwd30):
%
%   (2) NOISE -- per camera in P.ressweep.cams: the two working-surface states
%       (A0, A0 + the 10 nm single at the hold-out) are traced ONCE, unbinned
%       (dmg_camres suspended), then binned onto the camera's pixels here.  The
%       camera: the record's sensor with the pupil image as recorded (k physical
%       pixels per model pixel, as the record's camera rows), BINNED to the sweep's
%       sampling -- a camera pixel holds (k c)^2 sensor pixels, so the photons per
%       reading do not change with the sampling (every sensor pixel is used; the
%       exposure is set by the brightest MODEL pixel at fill x well x k^2, the
%       record's rule).  Read noise: DIGITAL binning (the CMOS case, the record's
%       convention) adds read x (k c) per camera pixel -- the same total read
%       variance per actuator at every sampling; ON-CHIP (charge) binning reads
%       once per camera pixel (read x 1), the row that says what binning restores.
%       Metric = the record's: piston-free robust floor over the unpoked lit set
%       (1.48 MAD), median over draws, one reading = differential / sqrt 2;
%       seconds per pm = floor^2 / readings per second.
%   (3) LOOP -- the shared dmg_loop on a LINEAR instrument: truth = this
%       sampling's measured matrix (S.fwd30), the reading's noise = the camera's
%       on the binned set-point frames (a cycle averages M readings: well x M, read
%       x sqrt M), estimator = S.est30.  Noise-only and the record's thermal ramp
%       (5 pm/cycle), g 0.5, K 60, seed 77.  (The engine-traced record loops sat
%       on the theory lines within 8 %; this instrument sits on them by
%       construction and carries the sampling's noise propagation.)
%   (4) REGISTRATION -- the camera grid shifted +-0.1 actuator pitch (x and y)
%       against the image AFTER calibration, read through the unshifted matrix:
%       gain at the hold-out, the largest neighbour's reading, the noise-free
%       floor, and whether the largest reading is the wrong actuator.
%   (5) DATA -- camera pixels per frame and Gb/s at 16 bit and the full frame
%       rate; camera pixels per reading over the pupil (the per-pixel solve's
%       unit).
say = S.say;  P = S.P;  ic = S.ic;  lit = S.litmask;  nact = S.nact;
cfg = dmg_camres('get');  c = 1;  if ~isempty(cfg), c = cfg.c; end
ppa = S.px_per_act_model;  s_eff = ppa / c;
RS = P.ressweep;
dev = zeros(nact);  dev(ic(1), ic(2)) = P.battery.dev_single;
un = lit;  un(ic(1), ic(2)) = false;

say('\n---- ressweep: %.3f model px per actuator / %.4f model px per camera px = %.3f CAMERA px per actuator ----\n', ppa, c, s_eff);

% ---- the two states, traced ONCE, unbinned -------------------------------
dmg_camres('suspend', true);  clsus = onCleanup(@() dmg_camres('suspend', false));
t0 = tic;
F0u = S.framesf(S.A0);  F1u = S.framesf(S.A0 + dev);
dmg_camres('suspend', false);  clear clsus
say('  traced the set point and the single (%.0f s); frames %d planes\n', toc(t0), size(F0u, 3));
binf = @(F, cf) bin_(F, cf);
cf0 = cfg;  if isempty(cf0), cf0 = struct('c', 1, 'offset', [0 0], 'center', []); end
[F0, T0, C0] = binf(F0u, cf0);  [F1, ~, C1] = binf(F1u, cf0);   % replicated frames + the camera cells
h_cl = S.hdiff(F1, F0);  a_cl = S.est30(h_cl);
R = struct('c', c, 'ppa_model', ppa, 's', s_eff, 'ncell', [T0.ncy T0.ncx], 'dmod', S.dmod, ...
    'rows', {S.rows}, 'rows_conv', {S.rows_conv}, 'nlit', S.nlit, 'nstates', S.nstates);
R.single_clean = struct('gain', a_cl(ic(1), ic(2)) / P.battery.dev_single);

% ---- (4) registration: the camera shifted +-0.1 pitch after calibration ----
sh = RS.reg_pitch * ppa;              % model px
offs = [0 0; sh 0; -sh 0; 0 sh; 0 -sh];
say('  REGISTRATION (the camera grid shifted against the image AFTER calibration; %g pitch = %.3f model px = %.3f camera px):\n', ...
    RS.reg_pitch, sh, sh/c);
say('  %-12s %8s %8s %10s %10s %10s %6s\n', 'offset px', 'gain', 'g conv', 'nbr max/d', 'flr pm', 'flr conv', 'wrong');
nb = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
reg = [];
for q = 1:size(offs, 1)
    cq = cf0;  cq.offset = offs(q, :);
    if q == 1, G0 = F0;  G1 = F1; else, G0 = binf(F0u, cq);  G1 = binf(F1u, cq); end
    h = S.hdiff(G1, G0);  a = S.est30(h);  ac = S.est30c(h);
    [g, nbm, fl, wr] = regscore_(a, ic, nb, P.battery.dev_single, un, nact);
    [gc, ~, flc] = regscore_(ac, ic, nb, P.battery.dev_single, un, nact);
    say('  %+5.2f %+5.2f %8.4f %8.4f %10.4f %10.2f %10.2f %6d\n', offs(q, :), g, gc, nbm, fl, flc, wr);
    reg = [reg; offs(q, :), g, gc, nbm, fl, flc, wr]; %#ok<AGROW>
end
R.reg = reg;  R.reg_cols = {'dx', 'dy', 'gain', 'gain_conv', 'nbr_max_over_dev', 'floor_pm', 'floor_conv_pm', 'wrong'};

% ---- (2) noise, per camera, both binning forms ----------------------------
cams = RS.cams;  nr = RS.nreal;  pf = 0.95;
say('  NOISE (%d draws; the record camera binned to this sampling; one reading = differential / sqrt 2):\n', nr);
say('  %-12s %-7s %6s %8s %9s %9s %9s %9s %10s %10s %9s\n', 'camera', 'binning', 'k', 'phys/px', 'e-/px', 'read e-', 'pm/read', 'hold-out', 'reads/s', 's per pm', 'Gb/s');
NZ = [];
for ci = 1:numel(cams)
    cam = cams(ci);
    k = pf*cam.npix/(1 + (cam.pol ~= 0)) / S.dmod;            % sensor px (per orientation) per model px
    sc = scale_(F0u, cam);                                    % e- per (model-px intensity unit) for ONE sensor pixel's exposure x k^2
    sc = sc * k^2;
    for bm = {'digital', 'onchip'}
        rd = cam.read * iff_(strcmp(bm{1}, 'digital'), k*c, 1);
        pk = zeros(nr, 1);  fl = zeros(nr, 1);
        for r = 1:nr
            G0 = noisy_(C0, T0, sc, c, rd, cam, 1, 1e4*ci + 2*r);
            G1 = noisy_(C1, T0, sc, c, rd, cam, 1, 1e4*ci + 2*r + 1);
            a = S.est30(S.hdiff(G1, G0)) - a_cl;
            a = a - mean(a(lit));
            pk(r) = a(ic(1), ic(2));  fl(r) = 1.4826*median(abs(a(un) - median(a(un))));
        end
        flr = median(fl)/sqrt(2)*1e9;  sgh = std(pk)/sqrt(2)*1e9;
        rate = cam.rate / cam.fpr;
        epx = cam.fill*cam.well*(k*c)^2;                     % e- per CAMERA pixel per frame at the brightest model pixel's level
        npx = (cam.npix/(k*c))^2;                            % output pixels per frame, all orientations (square sensor of side npix)
        gbs = npx * 16 * cam.rate / 1e9;
        say('  %-12s %-7s %6.2f %8.2f %9.3g %9.3g %9.3g %9.3g %10.3g %10.3g %9.3g\n', cam.tag, bm{1}, k, (k*c)^2, epx, rd, flr, sgh, rate, max(flr^2, 1)/rate, gbs);
        NZ = [NZ; struct('cam', cam.tag, 'binning', bm{1}, 'k', k, 'phys_per_px', (k*c)^2, 'read_e', rd, ...
            'pm_per_reading', flr, 'holdout_pm', sgh, 'readings_per_s', rate, 's_per_pm', max(flr^2, 1)/rate, ...
            'npix_out', npx, 'gbps', gbs)]; %#ok<AGROW>
    end
end
R.noise = NZ;
R.pupil_px = pi/4*(S.dmod/c)^2;                               % camera pixels per reading over the pupil
say('  data: %.0f camera px across the pupil, %.3g per reading over the pupil (the per-pixel solve''s unit)\n', S.dmod/c, R.pupil_px);

% ---- (3) the loop, linear instrument, each camera at its cycle (digital binning) ----
LP = [];
if RS.loop
    A0 = S.A0;
    for ci = 1:numel(cams)
        cam = cams(ci);  M = cam.M;
        k = pf*cam.npix/(1 + (cam.pol ~= 0)) / S.dmod;
        sc = scale_(F0u, cam) * k^2;  rd = cam.read * k*c;
        nmap = @(seed) S.hdiff(noisy_(C0, T0, sc, c, rd, cam, M, seed), F0);
        ins = struct('measure', @(cmd) S.fwd30(cmd - A0), ...
            'noisy', @(Fm, nph, seed) Fm + nmap(seed), ...
            'diff', @(Fa, Fb) Fa - Fb, 'est', S.est30, 'lit', lit);
        for dk = {'none', 'thermal'}
            o = struct('A0', A0, 'g', P.loop.g, 'K', P.loop.K, 'nph', 1, 'seed', P.loop.seed, 'rmax', P.loop.rmax);
            if strcmp(dk{1}, 'thermal'), o.drift = struct('kind', 'thermal', 'rate', P.loop.thermal_rate, 'shape', []);
            else, o.drift = struct('kind', 'none'); end
            t1 = tic;  L = dmg_loop(ins, o);
            th = L.theory.ss_noise;
            if strcmp(dk{1}, 'thermal') && isfield(L.theory, 'lag_ramp') && ~isnan(L.theory.lag_ramp)
                th = hypot(L.theory.lag_ramp, L.theory.ss_noise);   % the ramp's lag with the noise in quadrature
            end
            say('  LOOP %-12s M %4d (cycle %.3g s) %-8s ss %.3g pm (piston-free %.3g), sig_n %.3g pm (pf %.3g), theory %.3g pm, gG %.3f%s  [%.0f s]\n', ...
                cam.tag, M, M*cam.fpr/cam.rate, dk{1}, 1e9*L.ss, 1e9*L.ss_pf, 1e9*L.sig_n, 1e9*L.sig_n_pf, 1e9*th, L.theory.gG, ...
                iff_(L.diverged, ' DIVERGED', ''), toc(t1));
            LP = [LP; struct('cam', cam.tag, 'M', M, 'cycle_s', M*cam.fpr/cam.rate, 'drift', dk{1}, ...
                'ss_pm', 1e9*L.ss, 'ss_pf_pm', 1e9*L.ss_pf, 'sig_n_pm', 1e9*L.sig_n, 'sig_n_pf_pm', 1e9*L.sig_n_pf, ...
                'theory_pm', 1e9*th, 'gG', L.theory.gG, 'diverged', L.diverged)]; %#ok<AGROW>
        end
    end
end
R.loop = LP;
end

% =====================================================================
function [B, T, Bc] = bin_(F, cf)
% the camera's pixels Bc and the same replicated back onto the model grid B;
% identity at c 1 with no offset
if cf.c == 1 && all(cf.offset == 0)
    B = F;  Bc = F;  T = struct('ncy', size(F, 1), 'ncx', size(F, 2), 'id', true);  return
end
[Bc, T] = dmg_camres('bin', F, cf);  T.id = false;
B = dmg_camres('rep', Bc, T);
end

function sc = scale_(Fu, cam)
% e- per intensity unit for ONE sensor pixel: the brightest model pixel's frame mean
% (I/Q: its dc) at fill x well -- the record's exposure rule, on the UNBINNED frames
if cam.iq, m = Fu(:, :, 1); else, m = mean(Fu, 3); end
sc = cam.fill * cam.well / max(m(:));
end

function G = noisy_(Bc, T, sc, c, rd, cam, M, seed)
% shot + read noise per CAMERA pixel on the camera cells Bc, replicated back onto the
% model grid; M readings averaged (well x M, read x sqrt M).  sc = e- per intensity
% unit per MODEL pixel (k^2 sensor pixels); a camera pixel spans c^2 model px.
rs = RandStream('mt19937ar', 'Seed', seed);
a = sc * c^2 * M;  r = rd * sqrt(M);
E = Bc * a;
if cam.iq
    s2 = sqrt(2*(max(E(:, :, 1), 0) + r^2));
    E(:, :, 1) = E(:, :, 1) + sqrt(max(E(:, :, 1), 0)) .* randn(rs, size(s2));
    E(:, :, 2) = E(:, :, 2) + s2 .* randn(rs, size(s2));
    E(:, :, 3) = E(:, :, 3) + s2 .* randn(rs, size(s2));
else
    E = E + sqrt(max(E, 0)) .* randn(rs, size(E)) + r * randn(rs, size(E));
end
Bn = E / a;
if T.id, G = Bn; else, G = dmg_camres('rep', Bn, T); end
end

function [g, nbm, fl, wr] = regscore_(a, ic, nb, d, un, nact)
g = a(ic(1), ic(2)) / d;
v = zeros(size(nb, 1), 1);
for j = 1:size(nb, 1)
    p = ic + nb(j, :);
    if all(p >= 1 & p <= nact), v(j) = abs(a(p(1), p(2))); end
end
nbm = max(v) / d;
fl = std(a(un))*1e9;
[mx, im] = max(abs(a(:)));  wr = mx > 0 && im ~= sub2ind([nact nact], ic(1), ic(2));
end

function s = iff_(c, a, b), if c, s = a; else, s = b; end, end
