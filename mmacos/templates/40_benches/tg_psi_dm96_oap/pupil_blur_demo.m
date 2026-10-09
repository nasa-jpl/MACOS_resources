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
           'sig_pitch',[0 0.05 0.1:0.1:1.5],'seed',7,'figures',true,'outdir','');
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

% ================= the blur sweep =================
nS = numel(o.sig_pitch);
err = struct('checker',struct('naive',nan(1,nS),'cal',nan(1,nS)), ...
             'random', struct('naive',nan(1,nS),'cal',nan(1,nS)));
mtfN = mtf(o.sig_pitch*pitch);
say('---- the blur sweep (kernel estimator, lambda %.3g, lit unknowns) ----\n', o.lam);
say('%-8s %8s %8s | %-24s | %-24s\n','blur','sig/pit','MTF@Nyq','checker err (naive/cal)','random err (naive/cal)');
for is = 1:nS
    sig = o.sig_pitch(is)*pitch;
    stn_eff = dmg_stencil(blur_(Mu, sig, dx), xg, 0, 0, pitch, o.hw);   % the CALIBRATED (effective) stencil
    for p = 1:2
        nm = names{p};
        rng(o.seed + is);  m = blur_(surf.(nm), sig, dx) + noise();      % what the camera sees
        err.(nm).naive(is) = relerr_(act_fit_lit_(m, ic, stn_true, lit, o.lam), pat.(nm), lit);
        err.(nm).cal(is)   = relerr_(act_fit_lit_(m, ic, stn_eff,  lit, o.lam), pat.(nm), lit);
    end
    say('%6.3fmm %8.3f %8.4f | %8.3f%% / %8.3f%% | %8.3f%% / %8.3f%%\n', sig, o.sig_pitch(is), mtfN(is), ...
        100*err.checker.naive(is), 100*err.checker.cal(is), 100*err.random.naive(is), 100*err.random.cal(is));
end

% ---- the lesson ----
alt = (-1).^((-o.hw:o.hw).'+(-o.hw:o.hw));  Hn = abs(sum(stn_true.*alt,'all'))/max(abs(stn_true(:)));   % stencil transfer at Nyquist / peak
sig_lam = sqrt(max(0,-log(o.lam/Hn))/(pi^2*f_nyq^2));         % where the blurred Nyquist transfer falls to lambda
spmm = o.sig_pitch*pitch;
fl_c = err.checker.cal(1);
sig2_cal_c = cross_(spmm, err.checker.cal,   max(2*fl_c, 0.01));
sig2_nai_c = cross_(spmm, err.checker.naive, max(2*fl_c, 0.01));
say('\n---- the lesson ----\n');
say('blur-free floor (sigma = 0, lambda %.3g, %g pm): checker %.3f%%, random %.3f%% (all-unknowns control: %.2f%% / %.2f%%)\n', ...
    o.lam, o.noise_pm, 100*fl.lit(1), 100*fl.lit(2), 100*fl.all(1), 100*fl.all(2));
say('the checker error passes max(2 x floor, 1%%): calibrated at sigma = %.2f pitch, naive at sigma = %.2f pitch\n', sig2_cal_c/pitch, sig2_nai_c/pitch);
say('the structural knee: the stencil''s Nyquist transfer (%.3f of its peak) x the blur MTF falls to lambda at sigma = %.2f pitch -- beyond it the checkerboard is lost even to calibration\n', Hn, sig_lam/pitch);

% ================= figures =================
if o.figures
fc = figure('Visible','off','Position',[100 100 1500 520]);
sp = o.sig_pitch;
subplot(1,3,1); hold on; grid on;
plot(sp, 100*err.checker.naive,'-o','Color',[.85 .33 .1],'DisplayName','checker, naive');
plot(sp, 100*err.checker.cal,  '-s','Color',[.85 .33 .1],'MarkerFaceColor',[.85 .33 .1],'DisplayName','checker, calibrated');
plot(sp, 100*err.random.naive, '-o','Color',[0 .45 .74],'DisplayName','random 30 nm, naive');
plot(sp, 100*err.random.cal,   '-s','Color',[0 .45 .74],'MarkerFaceColor',[0 .45 .74],'DisplayName','random 30 nm, calibrated');
xline(sig_lam/pitch,'k--','Nyquist transfer = \lambda','LabelOrientation','horizontal','HandleVisibility','off');
xlabel('blur 1/e radius / actuator pitch'); ylabel('actuator-command error, % of command rms');
title('Reconstruction error vs pupil blur'); legend('Location','northwest'); ylim([0 100]);
subplot(1,3,2); hold on; grid on;
ff = linspace(0,1.2*f_nyq,200);
for sgp = [0.3 0.6 1.0]
    plot(ff/f_nyq, exp(-pi^2*(sgp*pitch)^2*ff.^2),'DisplayName',sprintf('\\sigma = %.1f pitch',sgp));
end
xline(1,'k--','actuator Nyquist','LabelOrientation','horizontal','HandleVisibility','off');
xlabel('spatial frequency / actuator Nyquist'); ylabel('blur MTF');
title('Why: blur rolls off the high DM frequencies'); legend('Location','northeast');
subplot(1,3,3); hold on; grid on;
plot(sp, mtfN,'k-','DisplayName','blur MTF at actuator Nyquist');
plot(sp, err.checker.cal,'-s','Color',[.85 .33 .1],'DisplayName','checker calibrated error (frac)');
xlabel('blur 1/e radius / actuator pitch'); ylabel('fraction');
title('Nyquist MTF and the calibrated error'); legend('Location','east'); ylim([0 1]);
sgtitle('pupil\_blur\_demo: pupil-imaging blur vs DM-gauge reconstruction (engine-free)');
print(fc, fullfile(o.outdir,'pupil_blur_demo_curve.png'),'-dpng','-r110');

is0 = find(o.sig_pitch >= 0.5,1);  if isempty(is0), is0 = nS; end
sig0 = o.sig_pitch(is0)*pitch;
stn0 = dmg_stencil(blur_(Mu,sig0,dx),xg,0,0,pitch,o.hw);
rng(o.seed + is0);  m0 = blur_(surf.checker,sig0,dx) + noise();
aN = act_fit_lit_(m0,ic,stn_true,lit,o.lam);  aC = act_fit_lit_(m0,ic,stn0,lit,o.lam);
fm = figure('Visible','off','Position',[100 100 1500 480]);
subplot(1,4,1); imagesc(xg,xg,surf.checker*1e6); axis image off; colorbar; title('true surface, nm');
subplot(1,4,2); imagesc(xg,xg,m0*1e6); axis image off; colorbar; title(sprintf('blurred (\\sigma=%.1f pitch), nm',o.sig_pitch(is0)));
subplot(1,4,3); imagesc(xa,xa,(aN-pat.checker)*1e9.*lit); axis image off; colorbar; title(sprintf('naive cmd error, pm (%.1f%%)',100*err.checker.naive(is0)));
subplot(1,4,4); imagesc(xa,xa,(aC-pat.checker)*1e9.*lit); axis image off; colorbar; title(sprintf('calibrated cmd error, pm (%.1f%%)',100*err.checker.cal(is0)));
sgtitle(sprintf('pupil\\_blur\\_demo: the checkerboard (actuator Nyquist) at \\sigma = %.1f pitch',o.sig_pitch(is0)));
print(fm, fullfile(o.outdir,'pupil_blur_demo_maps.png'),'-dpng','-r110');
end

out = struct('o',o,'sig_pitch',o.sig_pitch,'mtf_nyq',mtfN,'err',err,'floor',fl, ...
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
