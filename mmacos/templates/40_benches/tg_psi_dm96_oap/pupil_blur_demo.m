function out = pupil_blur_demo(varargin)
%PUPIL_BLUR_DEMO  When is pupil-imaging blur a concern for a DM gauge?
%   A simple, engine-free companion to tg96_pupilsim (Fang Shi's concern).
%   tg96_pupilsim answers the same question rigorously -- real engine rays,
%   zone-by-zone PSFs, a Fourier cross-check -- but it is complex.  This
%   demo strips the question to its physics so it can be checked by eye:
%
%     1. build a KNOWN DM surface from a known command set (the same
%        Gaussian influence-function model as dm_influence_map),
%     2. BLUR it with a pupil-imaging kernel (a Gaussian PSF of 1/e radius
%        sigma, swept) + add read noise -- this is what the camera sees,
%     3. RECONSTRUCT the actuator commands from the blurred map,
%     4. plot the actuator-command error vs blur width.
%
%   Two reconstructions are run at every blur width:
%     * NAIVE      -- deconvolves the influence stencil only (blur ignored).
%     * CALIBRATED -- deconvolves the EFFECTIVE stencil (influence (*) blur),
%       i.e. the blur is folded into the forward model, as a calibration does.
%
%   The estimator is the gauges' kernel form (dmg_stencil + the Tikhonov
%   lattice deconvolution of dmg_act_fit, lambda relative to the stencil
%   peak), solved for the LIT actuators only, with the map sampled exactly at
%   the actuator sites (the sites sit on the map grid: dx = pitch/4, N odd).
%   The legacy all-unknowns solve (dmg_act_fit as it stands: every one of the
%   96 x 96 commands free while the map is masked to the lit set) is kept as
%   the negative control -- its free ring outside the lit set is what made the
%   old 'blur-free floor' (8.1 % / 3.6 %, CC review 2026-10-09).
%
%   The lesson the curves make visible: blur attenuates the DM's HIGH
%   spatial frequencies (an MTF roll-off).  A calibrated reconstruction
%   undoes that -- until the MTF at the DM's highest spatial frequency (the
%   actuator Nyquist, where the +/- checkerboard lives) falls to the
%   reconstruction's regularization/noise floor.  Below it, blur costs almost
%   nothing; the naive reconstruction, by contrast, degrades far earlier.
%
%   Error metric: rms over the lit actuators of (recovered - true) commands,
%   lit-mean (piston) removed from both, divided by the rms of the true
%   commands (piston removed).
%
%   Name/value (defaults match the 96x96 tg96 rig):
%     'nact'   96     actuators across
%     'pitch'  1.0    actuator pitch (mm)
%     'infl_w' 0.85   influence 1/e radius, in PITCH units (dm_influence_map's)
%     'sub'    4      map pixels per pitch (dx = pitch/sub; the sites are nodes)
%     'work_nm' 30    command RMS of the working surface (random pattern)
%     'poke_nm' 50    command amplitude of the checkerboard (Nyquist stress)
%     'noise_pm' 20   read noise added to the blurred map, pm RMS per pixel
%     'lam'    0.05   kernel Tikhonov weight (dmg_act_fit default)
%     'lam_sweep' [0.05 1e-2 1e-3 1e-4]   the sigma = 0 lambda x noise sweep
%     'hw'     6      stencil half-width, actuators (tg96/zwfs value)
%     'sig_pitch' [0 0.05 0.1:0.1:1.5]  blur 1/e radii to sweep, PITCH units
%     'estimator' 'matrix'  the headline calibrated read: 'matrix' (the record's: the
%                     measured response matrix, columns = the blurred unit influence at
%                     each lit site, est_matrix_tg's solve) or 'kernel' (the bench's
%                     calib_mode 'kernel').  Both are computed and plotted either way.
%     'lam_m'  1e-3   matrix Tikhonov weight, of the median column energy (P.battery.matrix_lam)
%     'lam_m_sweep' [1e-2 1e-3 1e-4]   the matrix's sigma = 0 sweep
%     'seed'   7      RNG seed (random command pattern + noise)
%     'figures' true  write the PNGs
%     'outdir' runs/pupil_blur_demo   where the report + PNGs are written
%
%   Writes <outdir>/pupil_blur_demo_{report.txt, curve.png, maps.png}.
%   Run (from this directory):  >> pupil_blur_demo
%   NOTE: a demo, not a bench run -- no engine, no mex; it only needs the
%   dm_gauge_lib helpers on the path (../dm_gauge_lib, added automatically).
%   It is the plain-physics companion to tg96_pupilsim in this same folder.

o = struct('nact',96,'pitch',1.0,'infl_w',0.85,'sub',4, ...
           'work_nm',30,'poke_nm',50,'noise_pm',20,'lam',0.05,'lam_sweep',[0.05 1e-2 1e-3 1e-4],'hw',6, ...
           'sig_pitch',[0 0.05 0.1:0.1:1.5],'seed',7,'figures',true,'outdir','', ...
           'estimator','matrix','lam_m',1e-3,'lam_m_sweep',[1e-2 1e-3 1e-4]);
assert(any(strcmp(o.estimator,{'matrix','kernel'})), 'pupil_blur_demo:estimator', '''estimator'' is ''matrix'' or ''kernel''');
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
if isempty(o.outdir), o.outdir = fullfile(here,'runs','pupil_blur_demo'); end
if isempty(which('dmg_act_fit')) && exist(fullfile(here,'..','dm_gauge_lib'),'dir')
    addpath(fullfile(here,'..','dm_gauge_lib'));
end
assert(~isempty(which('dmg_act_fit')) && ~isempty(which('dmg_stencil')), ...
    'dm_gauge_lib not found; pass its path or run from the macos tree.');
if ~exist(o.outdir,'dir'), mkdir(o.outdir); end
rep = fopen(fullfile(o.outdir,'pupil_blur_demo_report.txt'),'w');
clean = onCleanup(@() fclose(rep));
say = @(varargin) deal_(fprintf(varargin{:}), fprintf(rep,varargin{:}));

% ---- geometry: the actuator sites are NODES of the map grid ----
pitch = o.pitch;  nact = o.nact;  dx = pitch/o.sub;
w_mm = o.infl_w*pitch;                       % influence 1/e radius, mm
xa = ((1:nact)-(nact+1)/2)*pitch;            % actuator centres, mm (half-integers for even nact)
R_ap = (nact/2)*pitch;                       % DM aperture radius
halfN = ceil((R_ap + 8*pitch)/dx);           % margin for the blur kernels
if mod(nact,2) == 0 && mod(o.sub,2) ~= 0, error('pupil_blur_demo:sub', '''sub'' must be even for an even nact (sites at half-pitch must be nodes)'); end
xg = (-halfN:halfN)*dx;  N = numel(xg);  [X,Y] = meshgrid(xg,xg);
[axg,ayg] = meshgrid(xa);
ic = round((xa - xg(1))/dx) + 1;             % grid index of each site
assert(max(abs(xg(ic) - xa)) < 1e-9*pitch, 'the actuator sites are not grid nodes');
R_ill = 0.74*R_ap;                           % source cone fills ~74% (the bench's illuminated-fill doctrine)
lit = hypot(axg,ayg) < 0.85*R_ill;           % lit actuators, dmg_lit's 0.85 R_ill rule (2852 here)
msk = hypot(X,Y) < R_ill;                    % the illuminated map (the detector support)
f_nyq = 1/(2*pitch);                         % actuator Nyquist, cyc/mm (the checkerboard)
mtf = @(sig) exp(-pi^2*sig.^2*f_nyq^2);      % MTF of exp(-r^2/sig^2) (1/e radius sig) at the Nyquist frequency

say('=== pupil_blur_demo: when is pupil-imaging blur a concern? ===\n');
say('DM %dx%d at %.2f mm pitch; influence 1/e %.2f mm; map grid %d x %.3f mm (the actuator sites are grid nodes); %d lit actuators (0.85 x 0.74 R_ap)\n', ...
    nact,nact,pitch,w_mm,N,dx,nnz(lit));
say('actuator Nyquist = %.3f cyc/mm; kernel lambda = %.3g (of the stencil peak); read noise = %g pm rms per map pixel (%d pixels per actuator)\n', ...
    f_nyq,o.lam,o.noise_pm,o.sub^2);
say('error = rms(recovered - true) over the lit actuators, piston removed, / rms(true commands)\n\n');

% ---- the unit influence (a poke at the node (0,0)) and the true stencil ----
Kinf = infl_kernel_(w_mm, dx);
Mu = zeros(N);  Mu((N+1)/2,(N+1)/2) = 1;  Mu = conv2(Mu, Kinf, 'same');
stn_true = dmg_stencil(Mu, xg, 0, 0, pitch, o.hw);

% ---- the two known command sets ----
rng(o.seed);
pat.checker = o.poke_nm*1e-6 * (-1).^((1:nact).'+(1:nact));   % Nyquist stress (hardest the DM makes)
pat.random  = o.work_nm*1e-6 * randn(nact);                   % the record's 30 nm working surface
pat.random(~lit) = 0;  pat.checker(~lit) = 0;
names = {'checker','random'};
for p = 1:2, surf.(names{p}) = build_surf_(pat.(names{p}), ic, N, Kinf); end
noise = @(s) (o.noise_pm*1e-9)*randn(N);

% ================= the floor (sigma = 0): lambda x noise, lit-only vs all unknowns =================
say('---- the blur-free floor (sigma = 0): kernel estimator, lambda x read noise ----\n');
say('%-8s %8s | %-23s | %-23s\n', 'lambda', 'noise', 'checker: lit / all unk.', 'random: lit / all unk.');
fl = struct();
for il = 1:numel(o.lam_sweep)
    for nz = [o.noise_pm 0]
        e = zeros(2,2);
        for p = 1:2
            rng(o.seed + 1000);  m = surf.(names{p}) + (nz*1e-9)*randn(N);
            a1 = act_fit_lit_(m, ic, stn_true, lit, o.lam_sweep(il));
            evalc('a2 = dmg_act_fit(m, xg, axg, ayg, stn_true, lit, o.lam_sweep(il));');   % (its pcg prints)
            e(p,:) = [relerr_(a1, pat.(names{p}), lit), relerr_(a2, pat.(names{p}), lit)];
        end
        say('%-8.0e %5g pm | %9.4f%% / %8.3f%% | %9.4f%% / %8.3f%%\n', o.lam_sweep(il), nz, 100*e(1,1), 100*e(1,2), 100*e(2,1), 100*e(2,2));
        if o.lam_sweep(il) == o.lam && nz == o.noise_pm, fl.lit = e(:,1); fl.all = e(:,2); end
    end
end
say('=> lit-only, sites on the grid: the floor is the regularization bias (the checker''s Nyquist transfer against lambda) plus the noise -- it moves with both;\n');
say('   all unknowns (dmg_act_fit as it stands, the legacy control): the free ring outside the lit set holds the floor at percent level whatever lambda and noise.\n\n');

% ================= the record's estimator: the measured response matrix =================
% columns = the (blurred) unit influence at each lit site, cut from the illuminated
% map; lambda_m of the median column energy; est_matrix_tg's solve (the piston
% rank-one term included).  Its NAIVE form is the unblurred matrix; its CALIBRATED
% form is the matrix measured through the blur -- which is all a calibration is.
ilit = find(lit);  Am = nnz(msk);  pix = zeros(N);  pix(msk) = 1:Am;
Mx0 = matrix_(Mu, ic, ilit, pix, Am, o.lam_m);                 % the naive matrix (no blur)
say('---- the blur-free floor (sigma = 0): the matrix estimator (the record''s), lambda_m x read noise ----\n');
say('%-8s %8s | %-12s | %-12s\n', 'lambda_m', 'noise', 'checker', 'random');
for lm = o.lam_m_sweep
    Mxl = Mx0;  if lm ~= o.lam_m, Mxl = matrix_(Mu, ic, ilit, pix, Am, lm); end
    for nz = [o.noise_pm 0]
        e = zeros(1,2);
        for p = 1:2
            rng(o.seed + 1000);  m = surf.(names{p}) + (nz*1e-9)*randn(N);
            e(p) = relerr_(est_matrix_(m, Mxl, lit), pat.(names{p}), lit);
        end
        say('%-8.0e %5g pm | %10.4f%% | %10.4f%%\n', lm, nz, 100*e(1), 100*e(2));
        if lm == o.lam_m && nz == o.noise_pm, fl.mat = e(:); end
    end
end
say('=> the matrix reads every illuminated pixel (%d per actuator): its NOISE part is the deck''s pm class (1.4 / 2.3 pm on a 10 nm change);\n', o.sub^2);
say('   its floor at the record''s lambda_m is the REGULARIZATION BIAS at the actuator Nyquist (noise-free identical to the digits) -- the same roll-off the\n');
say('   bench reports in its own Stage D (runs/lensuw2: the (96,96) diagonal-Nyquist mode at gain 0.966); the full +/-50 nm checkerboard is all of that mode\n\n');

% ================= the blur sweep =================
nS = numel(o.sig_pitch);
blank = struct('naive',nan(1,nS),'cal',nan(1,nS),'mnaive',nan(1,nS),'mcal',nan(1,nS));
err = struct('checker',blank,'random',blank);
mtfN = mtf(o.sig_pitch*pitch);
say('---- the blur sweep: kernel (lambda %.3g of the stencil peak) and matrix (lambda_m %.0e of the column energy), lit unknowns ----\n', o.lam, o.lam_m);
say('%-8s %7s %7s | %-35s | %-35s\n','blur','sig/pit','MTF@Nyq','checker: kernel naive/cal, MATRIX naive/cal','random: kernel naive/cal, MATRIX naive/cal');
for is = 1:nS
    sig = o.sig_pitch(is)*pitch;
    Mub = blur_(Mu, sig, dx);
    stn_eff = dmg_stencil(Mub, xg, 0, 0, pitch, o.hw);          % the CALIBRATED (effective) stencil
    Mxc = matrix_(Mub, ic, ilit, pix, Am, o.lam_m);             % the matrix measured THROUGH the blur
    for p = 1:2
        nm = names{p};
        rng(o.seed + is);  m = blur_(surf.(nm), sig, dx) + noise();      % what the camera sees
        err.(nm).naive(is)  = relerr_(act_fit_lit_(m, ic, stn_true, lit, o.lam), pat.(nm), lit);
        err.(nm).cal(is)    = relerr_(act_fit_lit_(m, ic, stn_eff,  lit, o.lam), pat.(nm), lit);
        err.(nm).mnaive(is) = relerr_(est_matrix_(m, Mx0, lit), pat.(nm), lit);
        err.(nm).mcal(is)   = relerr_(est_matrix_(m, Mxc, lit), pat.(nm), lit);
    end
    say('%6.3fmm %7.3f %7.4f | %7.3f%% %7.3f%% %8.4f%% %8.4f%% | %7.3f%% %7.3f%% %8.4f%% %8.4f%%\n', sig, o.sig_pitch(is), mtfN(is), ...
        100*err.checker.naive(is), 100*err.checker.cal(is), 100*err.checker.mnaive(is), 100*err.checker.mcal(is), ...
        100*err.random.naive(is),  100*err.random.cal(is),  100*err.random.mnaive(is),  100*err.random.mcal(is));
end
hc = 'mcal';  hn = 'mnaive';  hlab = 'matrix (the record''s estimator, every deck number)';
if strcmp(o.estimator,'kernel'), hc = 'cal';  hn = 'naive';  hlab = 'kernel (the bench''s calib_mode ''kernel'', S3 flavor)'; end

% ---- the lesson ----
alt = (-1).^((-o.hw:o.hw).'+(-o.hw:o.hw));  Hn = abs(sum(stn_true.*alt,'all'))/max(abs(stn_true(:)));   % stencil transfer at Nyquist / peak
sig_lam = sqrt(max(0,-log(o.lam/Hn))/(pi^2*f_nyq^2));         % kernel: where the blurred Nyquist transfer falls to lambda
spmm = o.sig_pitch*pitch;
thr = @(f) max(2*f(1), 0.01);
say('\n---- the lesson (headline estimator: %s) ----\n', hlab);
say('blur-free floor (sigma = 0, %g pm): matrix checker %.4f%%, random %.4f%%; kernel (lambda %.3g) %.3f%% / %.3f%%; all-unknowns kernel control %.2f%% / %.2f%%\n', ...
    o.noise_pm, 100*fl.mat(1), 100*fl.mat(2), o.lam, 100*fl.lit(1), 100*fl.lit(2), 100*fl.all(1), 100*fl.all(2));
say('the checker error first exceeds max(2 x its floor, 1%%): %s calibrated at sigma = %.2f pitch, naive at sigma = %.2f pitch\n', ...
    o.estimator, cross_(spmm, err.checker.(hc), thr(err.checker.(hc)))/pitch, cross_(spmm, err.checker.(hn), thr(err.checker.(hc)))/pitch);
say('kernel knee: the stencil''s Nyquist transfer (%.3f of its peak) x the blur MTF falls to lambda at sigma = %.2f pitch\n', Hn, sig_lam/pitch);
say('in the matrix form "calibrated" is automatic: the blurred columns ARE the matrix measured through the camera\n');

% ================= figures =================
if o.figures
fc = figure('Visible','off','Position',[100 100 1500 520]);
sp = o.sig_pitch;  cc = {[.85 .33 .1], [0 .45 .74]};  ttl = {'checkerboard (actuator Nyquist), \pm50 nm', 'random 30 nm working surface'};
for p = 1:2
    nm = names{p};  e = err.(nm);
    subplot(1,3,p); hold on; grid on;
    plot(sp, 100*e.naive, ':o', 'Color', cc{p}, 'DisplayName', 'kernel, naive');
    plot(sp, 100*e.cal,   ':s', 'Color', cc{p}, 'DisplayName', 'kernel, calibrated');
    plot(sp, 100*e.mnaive,'-o', 'Color', 'k',   'DisplayName', 'matrix, naive');
    plot(sp, 100*e.mcal,  '-s', 'Color', 'k', 'MarkerFaceColor', 'k', 'DisplayName', 'matrix, calibrated (the record''s)');
    xlabel('blur 1/e radius / actuator pitch'); ylabel('actuator-command error, % of command rms');
    title(ttl{p}); legend('Location','northwest'); ylim([0 100]);
end
subplot(1,3,3); hold on; grid on;
plot(sp, mtfN,'k-','DisplayName','blur MTF at actuator Nyquist');
plot(sp, err.checker.mcal,'-s','Color',cc{1},'DisplayName','checker, matrix calibrated (frac)');
plot(sp, err.checker.cal,':s','Color',cc{1},'DisplayName','checker, kernel calibrated (frac)');
xline(sig_lam/pitch,'k--','kernel knee','LabelOrientation','horizontal','HandleVisibility','off');
xlabel('blur 1/e radius / actuator pitch'); ylabel('fraction');
title('Nyquist MTF and the calibrated error'); legend('Location','east'); ylim([0 1]);
sgtitle('pupil\_blur\_demo: pupil-imaging blur vs DM-gauge reconstruction (engine-free)');
print(fc, fullfile(o.outdir,'pupil_blur_demo_curve.png'),'-dpng','-r110');

is0 = find(o.sig_pitch >= 0.5,1);  if isempty(is0), is0 = nS; end
sig0 = o.sig_pitch(is0)*pitch;  Mub0 = blur_(Mu,sig0,dx);
rng(o.seed + is0);  m0 = blur_(surf.checker,sig0,dx) + noise();
aN = act_fit_lit_(m0,ic,stn_true,lit,o.lam);  aC = est_matrix_(m0, matrix_(Mub0, ic, ilit, pix, Am, o.lam_m), lit);
fm = figure('Visible','off','Position',[100 100 1500 480]);
subplot(1,4,1); imagesc(xg,xg,surf.checker*1e6); axis image off; colorbar; title('true surface, nm');
subplot(1,4,2); imagesc(xg,xg,m0*1e6); axis image off; colorbar; title(sprintf('blurred (\\sigma=%.1f pitch), nm',o.sig_pitch(is0)));
subplot(1,4,3); imagesc(xa,xa,(aN-pat.checker)*1e9.*lit); axis image off; colorbar; title(sprintf('kernel naive cmd error, pm (%.1f%%)',100*err.checker.naive(is0)));
subplot(1,4,4); imagesc(xa,xa,(aC-pat.checker)*1e9.*lit); axis image off; colorbar; title(sprintf('matrix calibrated cmd error, pm (%.2f%%)',100*err.checker.mcal(is0)));
sgtitle(sprintf('pupil\\_blur\\_demo: the checkerboard (actuator Nyquist) at \\sigma = %.1f pitch',o.sig_pitch(is0)));
print(fm, fullfile(o.outdir,'pupil_blur_demo_maps.png'),'-dpng','-r110');
end

out = struct('o',o,'sig_pitch',o.sig_pitch,'mtf_nyq',mtfN,'err',err,'floor',fl,'headline',struct('cal',hc,'naive',hn), ...
             'sig_lambda_mm',sig_lam,'lit',lit,'f_nyq',f_nyq);
say('\nwrote pupil_blur_demo_report.txt%s in %s\n', iff_(o.figures, ', _curve.png, _maps.png', ''), o.outdir);
end

% ---------------------------------------------------------------------------
function K = infl_kernel_(w, dx)
% the Gaussian influence function exp(-r^2/w^2) on the grid (peak 1), out to 5 w
hw = ceil(5*w/dx);  t = (-hw:hw)*dx;  [kx,ky] = meshgrid(t,t);
K = exp(-(kx.^2+ky.^2)/w^2);
end
function h = build_surf_(C, ic, N, Kinf)
% the DM surface = sum of Gaussian influence functions: the commands as deltas
% at the site nodes, convolved with the influence (exact: the sites are nodes)
D = zeros(N);  D(ic, ic) = C;                 % meshgrid convention: row = y, col = x
h = conv2(D, Kinf, 'same');
end
function b = blur_(h, sig, dx)
% the Gaussian blur exp(-r^2/sig^2)/(pi sig^2) (1/e radius sig) applied EXACTLY as its
% transfer function exp(-pi^2 sig^2 f^2) on the FFT grid -- any sig, including the
% sub-pixel ones a pixel kernel cannot represent (the built leg's 0.016 pitch).  The
% surfaces sit >= 8 pitch inside the grid edge, so the periodic wrap is immaterial.
if sig == 0, b = h; return; end
N = size(h,1);  f = ifftshift(((0:N-1) - floor(N/2))/(N*dx));  [FX,FY] = meshgrid(f,f);
b = real(ifft2(fft2(h).*exp(-pi^2*sig^2*(FX.^2+FY.^2))));
end
function a = act_fit_lit_(hd, ic, stn, lit, lam)
% dmg_act_fit's Tikhonov lattice deconvolution, two changes: the map is read
% EXACTLY at the site nodes (no interpolation), and the unknowns are the LIT
% actuators only (the mask on both sides of the operator; an unlit command
% sees only the regularizer and stays 0).
m = hd(ic, ic);  m(~lit) = 0;
Wf = double(lit);
l2 = (lam*max(abs(stn(:))))^2;
Cop = @(x) reshape(Wf.*conv2(Wf.*reshape(x, size(lit)), stn, 'same'), [], 1);
Ct  = @(x) reshape(Wf.*conv2(Wf.*reshape(x, size(lit)), rot90(stn,2), 'same'), [], 1);
Aop = @(x) Ct(Cop(x)) + l2*x;
[x, ~] = pcg(Aop, Ct(m(:)), 1e-12, 400);
a = reshape(x, size(lit));
end
function Mx = matrix_(Mub, ic, ilit, pix, Am, lam)
% the response matrix J (illuminated map pixels x lit actuators): column k = the
% unit influence (blurred) Mub, centred on the node (0,0), shifted to site k (exact:
% the sites are nodes), cut from the illuminated map; then est_matrix_tg's normal
% equations: JtJ - v v'/Am (mean-referenced), + lam*median(diag) (matrix_reg 'median').
N = size(Mub,1);  c0 = (N+1)/2;
[rr, cc] = find(abs(Mub) > 1e-10*max(abs(Mub(:))));
r = max(abs([rr; cc] - c0));  w = Mub(c0-r:c0+r, c0-r:c0+r);
[dc, dr] = meshgrid(-r:r, -r:r);
nl = numel(ilit);  nw = numel(w);
I = zeros(nw*nl,1);  Jc = I;  V = I;
[ia, ja] = ind2sub([numel(ic) numel(ic)], ilit);
for k = 1:nl
    rows = ic(ia(k)) + dr(:);  cols = ic(ja(k)) + dc(:);
    q = (k-1)*nw + (1:nw);
    I(q) = pix(sub2ind([N N], rows, cols));  Jc(q) = k;  V(q) = w(:);
end
keep = I > 0;
J = sparse(I(keep), Jc(keep), V(keep), Am, nl);
v = full(sum(J,1)).';
JtJ = full(J.'*J) - (v*v.')/Am;
d = diag(JtJ);
Mx = struct('J', J, 'v', v, 'Am', Am, 'Rf', chol(JtJ + lam*median(d(d>0))*eye(nl)), 'ilit', ilit, 'pix', pix);
end
function a = est_matrix_(h, Mx, lit)
% est_matrix_tg verbatim: b = J'h - v (1'h)/Am over the mask, x = Rf \ (Rf' \ b)
hm = h(Mx.pix > 0);  hm = hm(:);  [~, ord] = sort(Mx.pix(Mx.pix > 0));  hm = hm(ord);
b = Mx.J.'*hm - Mx.v*(sum(hm)/Mx.Am);
x = Mx.Rf \ (Mx.Rf.' \ b);
a = zeros(size(lit));  a(Mx.ilit) = x;
end
function e = relerr_(a_hat, a_true, lit)
d = a_hat(lit) - a_true(lit);  d = d - mean(d);
t = a_true(lit) - mean(a_true(lit));
e = sqrt(mean(d.^2)) / sqrt(mean(t.^2));
end
function s = cross_(x, y, thr)
i = find(y>thr,1);  if isempty(i), s = NaN; else, s = x(i); end
end
function s = iff_(c, a, b), if c, s = a; else, s = b; end, end
function deal_(~,~), end
