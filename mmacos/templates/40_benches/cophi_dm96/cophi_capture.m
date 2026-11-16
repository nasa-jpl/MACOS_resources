function CP = cophi_capture(P, G, say)
%COPHI_CAPTURE  The capture stage (campaign brief sec. 9.2; CC 2026-10-12):
%   capture range of the H reading from COLD, on the bench's own fields.
%
%   (a) TEMPORAL unwrapping between readings: a reading is unambiguous while
%       every pixel's surface moves < lambda/4 per reading; a sequence of readings
%       on a drifting surface is unwrapped per pixel and compared with the engine,
%       at 0.9 and 1.1 x that step -- the failure drift rate is lambda/4 x the
%       reading rate (nm/s, at the PEAK pixel), at each camera's cadence.
%   (b) TWO COLOURS, frequency-multiplexed: lambda_1 = P.LAM at a 7.5 Hz beat (4
%       frames per beat) and lambda_2 = lambda_1 + dlam at 3.75 Hz (8 frames per
%       beat), the SAME 8 frames at 30 fps carrying both (the two harmonics are
%       orthogonal over 8 frames, so each colour is read by dmg_cophi_recon at its
%       own bucket centres); lambda_2's fields are a RE-TRACE of both decks at
%       lambda_2.  Synthetic wavelength Lambda = l1 l2/|dl|: the coarse surface
%       wrap(phi_1 - phi_2) Lambda/4pi (range +-Lambda/4) fixes the fine fringe
%       order of h_1.  On the battery's capture ladder (30..480 nm rms), single
%       colour vs two colours, against the engine's own OPD (unwrapped, a ray
%       path), noise-free and with each camera's photon + read noise.
%   The two-colour arithmetic (range, amplification, the fine-noise requirement)
%   is printed beside, for 632.8 nm and the 1.3 um row (analytic: the C-RED 2 is a
%   placeholder).  Returns CP (rows + the numbers).
lam1 = P.LAM;  dl = P.capture.dlam;  lam2 = lam1 + dl;
Lam = lam1*lam2/abs(dl);
say('\n---- capture: from cold (lambda_1 %.4g nm, lambda_2 %.4g nm, Lambda = %.4g um) ----\n', lam1*1e6, lam2*1e6, Lam*1e3);
% ---- the two colours' reading contexts -----------------------------------
C1 = cophi_ctx(P, G, @(varargin) []);
P2 = P;  P2.LAM = lam2;  P2.cophi.lam_trace = lam2;
C2 = cophi_ctx(P2, G, @(varargin) []);
say('  colour 2 re-traced at %.4g nm: registration null %.3g nm, contrast %.4f (colour 1: %.3g nm, %.4f)\n', ...
    lam2*1e6, C2.reg.null.rms_p*lam2*1e6/(4*pi), C2.reg.g_med, C1.reg.null.rms_p*lam1*1e6/(4*pi), C1.reg.g_med);
msk = C1.mskT;  N = size(msk, 1);
cfg = P.dm(1);  nact = cfg.nact;
lat = ((1:nact) - (nact+1)/2)*cfg.pitch;  [ax, ay] = meshgrid(lat);
lit = hypot(ax, ay) <= P.capture.lit_r;            % the control set (the aperture-cap rule)
dmap = @(A) dm_influence_map(P.grid.N_G, P.grid.DX_G, 'nact', nact, 'pitch', cfg.pitch, 'act', A);
surf = @(rms_) surf_(nact, lit, P.battery.seed_base, rms_);
% ---- truth: the engine's OPD per pixel (unwrapped), and its sign vs the reading
iq = C1.iq_;
if iq
    say('  LOCK-IN (I/Q) form: the two colours on ALTERNATE I/Q samples (one sample each), no buckets, no multiplexing\n');
end
frames2_ = @(Ca, Cb, M) frames2x_(Ca, Cb, M, iq);
read2_   = @(F) read2x_(F, iq);
W0 = C1.opdf(zeros(P.grid.N_G));
F10 = frames2_(C1, C2, zeros(P.grid.N_G));
[p10, p20] = read2_(F10);
A30 = surf(30e-6);  F1 = frames2_(C1, C2, dmap(A30));  [p1, ~] = read2_(F1);
dW = C1.opdf(dmap(A30)) - W0;
h1 = angle(exp(1i*(p1 - p10))) * lam1/(4*pi);
sg = sign(sum(h1(msk) .* dW(msk)));  say('  truth: the engine''s OPD change; reading h = %+d x dOPD/2 (sign measured on the 30 nm rung)\n', sg);
truth = @(M) sg*(C1.opdf(M) - W0)/2;
% ---- (b) the ladder, single vs two colours ---------------------------------
lad = P.capture.ladder;  rows = {};
cams = [{'noise-free'}, P.capture.cams];
say('\n  ladder (surface rms over lit; pixel errors over the test pupil, piston removed):\n');
say('  %-11s %8s | %-28s | %-34s\n', 'camera', 'surface', 'single colour: rms err, wrong', 'two colours: rms err, wrong orders');
for j = 1:numel(lad)
    A = surf(lad(j));  M = dmap(A);
    Fm = frames2_(C1, C2, M);  ht = truth(M);
    for c = 1:numel(cams)
        Fn = Fm;  Fn0 = F10;
        if c > 1
            cam = P.cophi.cameras.(cams{c});  k = 0.95*cam.npix/(2*C1.rT);
            ce = struct('well', cam.well*k^2, 'read', cam.read*k, 'fill', P.cophi.fill);
            if iq
                ce.iq = true;  cn = @(F, sd) cat(3, dmg_cophi_camera(F(:,:,1:3), ce, sd), dmg_cophi_camera(F(:,:,4:6), ce, sd + 7));
                Fn = cn(Fm, 100*j + c);  Fn0 = cn(F10, 100*j + c + 50);
            else
                Fn = dmg_cophi_camera(Fm, ce, 100*j + c);  Fn0 = dmg_cophi_camera(F10, ce, 100*j + c + 50);
            end
        end
        [q1, q2] = read2_(Fn);  [q10, q20] = read2_(Fn0);
        d1 = angle(exp(1i*(q1 - q10)));  d2 = angle(exp(1i*(q2 - q20)));
        hs = d1 * lam1/(4*pi);                                  % single colour (wrapped)
        hc = angle(exp(1i*(d1 - d2))) * Lam/(4*pi);             % coarse, range +-Lambda/4
        m  = round((hc - hs)/(lam1/2));  hh = hs + m*lam1/2;    % fine order fixed by the coarse
        [es, ws] = err_(hs, ht, msk, lam1);  [e2, w2] = err_(hh, ht, msk, lam1);
        say('  %-11s %6.0f nm | %9.3g nm, %8.4f %%      | %9.3g nm, %10.4f %%\n', cams{c}, lad(j)*1e6, es*1e6, 100*ws, e2*1e6, 100*w2);
        rows(end+1, :) = {cams{c}, lad(j), es, ws, e2, w2}; %#ok<AGROW>
    end
end
% ---- crosstalk: each colour read from the multiplexed frames vs alone --------
if ~iq
Fa = frames_one_(C1, dmap(surf(lad(1))), 4, 2);  Fb = frames_one_(C2, dmap(surf(lad(1))), 8, 1);
[pa, ~] = read2_(Fa + Fb);  pA = dmg_cophi_recon(Fa, al_(4, 2));
say('  multiplexing crosstalk (colour 1 read from both colours'' frames vs from its own alone): %.3g rad rms\n', ...
    sqrt(mean(angle(exp(1i*(pa(msk) - pA(msk)))).^2)));
end
% ---- (a) temporal unwrapping --------------------------------------------------
sh = surf(30e-6);  Ms = dmap(sh);  hsh = truth(Ms);  pk = max(abs(hsh(msk) - mean(hsh(msk))));
say('\n  temporal unwrapping: a drifting surface (the 30 nm shape, %.3g nm peak), %d readings, the step per reading as a fraction of lambda/4 at the PEAK pixel:\n', pk*1e6, P.capture.nseq);
trow = {};
for fr = P.capture.temporal_frac
    step = fr * (lam1/4) / pk;                          % scale so the peak pixel moves fr x lambda/4 per reading
    acc = zeros(N);  pprev = p10;  ok = true;
    for t = 1:P.capture.nseq
        Ft = frames2_(C1, C2, dmap(sh * step * t / 1e-0));  [pt, ~] = read2_(Ft);
        acc = acc + angle(exp(1i*(pt - pprev)));  pprev = pt;
    end
    hte = truth(dmap(sh * step * P.capture.nseq));
    [e, w] = err_(acc*lam1/(4*pi), hte, msk, lam1);  ok = w == 0;
    say('    %.2f x lambda/4 per reading: final rms err %.3g nm, wrong pixels %.4f %% -> %s\n', fr, e*1e6, 100*w, iff_(ok, 'UNWRAPS', 'FAILS'));
    trow(end+1, :) = {fr, e, w}; %#ok<AGROW>
end
rates = {};
for c = P.capture.cams
    cam = P.cophi.cameras.(c{1});  rr = cam.rate/C1.nfr_;
    rates(end+1, :) = {c{1}, rr, lam1/4*rr}; %#ok<AGROW>
    say('    %-11s %6.2f readings/s -> temporal unwrapping holds to a drift of %.3g um/s at the peak pixel (%.3g um/s of rms for this shape)\n', ...
        c{1}, rr, lam1/4*rr*1e3, lam1/4*rr*1e3 * std(hsh(msk))/pk);
end
% ---- the arithmetic (sec. 9.2), 632.8 nm and the 1.3 um row ------------------
say('\n  two-colour arithmetic (range +-Lambda/4; amplification Lambda/lambda; the fine reading must hold sqrt2 sigma x Lambda/lambda < lambda/8):\n');
ar = {};
for L1 = [lam1, P.capture.lam_ir]
    for d = P.capture.dlam_table
        Lm = L1*(L1 + d)/d;  amp = Lm/L1;  req = (L1/8)/(sqrt(2)*amp);
        say('    lambda %.4g nm, dlam %5.1f nm: Lambda %8.3g um, range +-%8.3g um, amplification %7.1f, fine sigma < %7.3g nm\n', ...
            L1*1e6, d*1e6, Lm*1e3, Lm/4*1e3, amp, req*1e6);
        ar(end+1, :) = {L1, d, Lm, Lm/4, amp, req}; %#ok<AGROW>
    end
end
say(['  the bench''s per-pixel phase noise is ~75 pm of surface per reading (baseline; the noise stage), so even\n' ...
     '  dlam 1 nm (amplification ~630, requirement ~0.07 nm) resolves the fringe in ONE multiplexed reading; a 200 nm rms\n' ...
     '  surface from cold (peaks ~0.8-1 um) needs range +-1 um, i.e. Lambda >= 4 um -- ONE rung of dlam <= ~100 nm at 633\n']);
CP = struct('rows', {rows}, 'temporal', {trow}, 'rates', {rates}, 'arith', {ar}, 'Lam', Lam, 'lam2', lam2, 'sign', sg);
end

% ---------------------------------------------------------------------
function F = frames_one_(C, M, Nb, reps)
% one colour's frames over 8 frames: Nb buckets per beat, repeated reps times
    [Fb, ~] = dmg_cophi_frames(C.fieldf(M), C.Elo, struct('N', Nb));
    F = repmat(Fb, 1, 1, reps);
end
function F = frames2x_(C1, C2, M, iq)
% the two colours: on the same 8 frames (colour 1 at 4 frames per beat, two beats;
% colour 2 at 8 per beat; intensities add -- the colours do not interfere), or, for
% the lock-in (iq), ONE I/Q sample of each colour on alternate samples, stacked
    if iq
        [A, ~] = dmg_cophi_frames(C1.fieldf(M), C1.Elo, C1.opt_);
        [B, ~] = dmg_cophi_frames(C2.fieldf(M), C2.Elo, C2.opt_);
        F = cat(3, A, B);
    else
        F = frames_one_(C1, M, 4, 2) + frames_one_(C2, M, 8, 1);
    end
end
function [p1, p2] = read2x_(F, iq)
    if iq
        p1 = dmg_cophi_recon(F(:,:,1:3), 'iq');  p2 = dmg_cophi_recon(F(:,:,4:6), 'iq');
    else
        p1 = dmg_cophi_recon(F, al_(4, 2));  p2 = dmg_cophi_recon(F, al_(8, 1));
    end
end
function a = al_(Nb, reps)
% the bucket centres of a colour with Nb buckets per beat over reps beats
    D = 2*pi/Nb;  a = (0:Nb*reps-1)*D + D/2;
end
function [e, w] = err_(h, ht, msk, lam)
% rms error (piston removed) and the fraction of pixels a fringe off (> lambda/4)
    d = h(msk) - ht(msk);  d = d - median(d);
    e = sqrt(mean(d.^2));  w = mean(abs(d) > lam/4);
end
function A = surf_(nact, lit, seed, rms_)
    rng(seed);  A = zeros(nact);  A(lit) = rms_*randn(nnz(lit), 1);
end
function s = iff_(c, a, b), if c, s = a; else, s = b; end, end
