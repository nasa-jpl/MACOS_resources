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
%     3. RECONSTRUCT the actuator commands from the blurred map with the
%        gauges' OWN reconstruction (dmg_stencil + dmg_act_fit, the
%        Tikhonov lattice deconvolution -- not a toy),
%     4. plot the actuator-command error vs blur width.
%
%   Two reconstructions are run at every blur width:
%     * NAIVE   -- deconvolves the influence stencil only (blur ignored).
%     * CALIBRATED -- deconvolves the EFFECTIVE stencil (influence (*) blur),
%       i.e. the blur is folded into the forward model, as a calibration does.
%
%   The lesson the curves make visible: blur attenuates the DM's HIGH
%   spatial frequencies (an MTF roll-off).  A calibrated reconstruction
%   undoes that -- until the MTF at the DM's highest spatial frequency (the
%   actuator Nyquist, where the +/- checkerboard lives) falls to the
%   reconstruction's regularization/noise floor (~lambda).  THAT crossover
%   is "when blur is a concern".  Below it, blur costs almost nothing; the
%   naive reconstruction, by contrast, degrades far earlier.
%
%   Name/value (defaults match the 96x96 tg96 rig):
%     'nact'   96     actuators across
%     'pitch'  1.0    actuator pitch (mm)
%     'infl_w' 0.85   influence 1/e radius, in PITCH units (dm_influence_map's)
%     'N'      384    WF grid (= tg96 N_G)
%     'dx'     0.28   WF grid pitch, mm (= tg96 DX_G)
%     'work_nm' 30    command RMS of the working surface (random pattern)
%     'poke_nm' 50    command amplitude of the checkerboard (Nyquist stress)
%     'noise_pm' 20   read noise added to the blurred map, pm RMS
%     'lam'    0.05   Tikhonov weight (dmg_act_fit default)
%     'hw'     6      stencil half-width, actuators (tg96/zwfs value)
%     'sig_pitch' [0 0.1:0.1:1.5]  blur 1/e radii to sweep, in PITCH units
%                                  (0 = the blur-free floor)
%     'seed'   7      RNG seed (random command pattern + noise)
%     'outdir' runs/pupil_blur_demo   where the report + PNGs are written
%
%   Writes <outdir>/pupil_blur_demo_{report.txt, curve.png, maps.png}.
%   Run (from this directory):  >> pupil_blur_demo
%   NOTE: a demo, not a bench run -- no engine, no mex; it only needs the
%   dm_gauge_lib helpers on the path (../dm_gauge_lib, added automatically).
%   It is the plain-physics companion to tg96_pupilsim in this same folder.

o = struct('nact',96,'pitch',1.0,'infl_w',0.85,'N',384,'dx',0.28, ...
           'work_nm',30,'poke_nm',50,'noise_pm',20,'lam',0.05,'hw',6, ...
           'sig_pitch',[0 0.1:0.1:1.5],'seed',7,'outdir','');
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
if isempty(o.outdir), o.outdir = fullfile(here,'runs','pupil_blur_demo'); end
% the two gauge-library helpers that carry the real reconstruction live next door
if isempty(which('dmg_act_fit')) && exist(fullfile(here,'..','dm_gauge_lib'),'dir')
    addpath(fullfile(here,'..','dm_gauge_lib'));
end
assert(~isempty(which('dmg_act_fit')) && ~isempty(which('dmg_stencil')), ...
    'dm_gauge_lib not found; pass its path or run from the macos tree.');
if ~exist(o.outdir,'dir'), mkdir(o.outdir); end
rep = fopen(fullfile(o.outdir,'pupil_blur_demo_report.txt'),'w');
clean = onCleanup(@() fclose(rep));
say = @(varargin) deal_(fprintf(varargin{:}), fprintf(rep,varargin{:}));

N = o.N;  dx = o.dx;  pitch = o.pitch;  nact = o.nact;
w_mm = o.infl_w*pitch;                       % influence 1/e radius, mm
xg = ((1:N)-(N+1)/2)*dx;  [X,Y] = meshgrid(xg,xg);
xa = ((1:nact)-(nact+1)/2)*pitch;            % actuator centres, mm
[axg,ayg] = meshgrid(xa);
R_ap = (nact/2)*pitch;                       % DM aperture radius
R_ill = 0.74*R_ap;                           % source cone fills ~74% (the bench's illuminated-fill doctrine)
lit = hypot(axg,ayg) < 0.85*R_ill;           % lit actuators, per dmg_lit's 0.85 R_ill rule
litf = hypot(X,Y) < 0.85*R_ill;              % the same region on the fine grid

f_nyq = 1/(2*pitch);                         % actuator Nyquist, cyc/mm (the checkerboard)
mtf = @(sig) exp(-2*pi^2*sig.^2*f_nyq^2);    % Gaussian-blur MTF at the Nyquist frequency

say('=== pupil_blur_demo: when is pupil-imaging blur a concern? ===\n');
say('DM %dx%d at %.2f mm pitch; influence 1/e %.2f mm; WF grid %d x %.3f mm; %d lit actuators\n', ...
    nact,nact,pitch,w_mm,N,dx,nnz(lit));
say('actuator Nyquist = %.3f cyc/mm; Tikhonov lambda = %.3f; read noise = %g pm rms\n', f_nyq,o.lam,o.noise_pm);
say('reconstruction: dmg_stencil + dmg_act_fit (the gauges'' own Tikhonov lattice deconvolution)\n\n');

% ---- the unit influence (a single poke) and the two stencils ----
infl = @(u,v) exp(-((X-u).^2 + (Y-v).^2)/w_mm^2);
Mu = infl(0,0);                              % unit poke at the DM centre, on the fine grid
stn_true = dmg_stencil(Mu, xg, 0, 0, pitch, o.hw);   % the TRUE influence stencil (blur ignored)

% ---- the two known command sets ----
rng(o.seed);
pat.checker = o.poke_nm*1e-6 * (-1).^((1:nact).'+(1:nact));   % Nyquist stress (hardest the DM makes)
pat.random  = o.work_nm*1e-6 * randn(nact);                   % the record's 30 nm working surface
pat.random(~lit) = 0;  pat.checker(~lit) = 0;
names = {'checker','random'};
surf.checker = build_surf_(pat.checker, xa, X, Y, w_mm, R_ap);
surf.random  = build_surf_(pat.random,  xa, X, Y, w_mm, R_ap);

nS = numel(o.sig_pitch);
err = struct('checker',struct('naive',nan(1,nS),'cal',nan(1,nS)), ...
             'random', struct('naive',nan(1,nS),'cal',nan(1,nS)));
mtfN = mtf(o.sig_pitch*pitch);

say('%-8s %8s %8s | %-18s | %-18s\n','blur','sig/pit','MTF@Nyq','checker err (naive/cal)','random err (naive/cal)');
for is = 1:nS
    sig = o.sig_pitch(is)*pitch;                 % blur 1/e radius, mm
    K = blur_kernel_(sig, dx);                   % normalized Gaussian PSF (sums to 1)
    Mu_blur = conv2(Mu, K, 'same');              % the blurred unit influence
    stn_eff = dmg_stencil(Mu_blur, xg, 0, 0, pitch, o.hw);   % the CALIBRATED (effective) stencil
    for p = 1:2
        nm = names{p};
        m = conv2(surf.(nm), K, 'same');                     % what the camera sees
        m = m + (o.noise_pm*1e-9)*randn(N);                  % + read noise (mm)
        a_naive = dmg_act_fit(m, xg, axg, ayg, stn_true, lit, o.lam);
        a_cal   = dmg_act_fit(m, xg, axg, ayg, stn_eff,  lit, o.lam);
        err.(nm).naive(is) = relerr_(a_naive, pat.(nm), lit);
        err.(nm).cal(is)   = relerr_(a_cal,   pat.(nm), lit);
    end
    say('%6.2fmm %8.3f %8.4f | %7.2f%% / %6.2f%% | %7.2f%% / %6.2f%%\n', sig, o.sig_pitch(is), mtfN(is), ...
        100*err.checker.naive(is), 100*err.checker.cal(is), 100*err.random.naive(is), 100*err.random.cal(is));
end

% ---- the lesson: the blur-free floor, and where blur doubles it ----
sig_lam  = sqrt(-log(o.lam)/(2*pi^2*f_nyq^2));                % the sigma where MTF(f_nyq) = lambda
fl_chk   = err.checker.cal(1);  fl_rnd = err.random.cal(1);   % sig = 0: the intrinsic (blur-free) floor
spmm = o.sig_pitch*pitch;
sig2_cal_c = cross_(spmm, err.checker.cal,   2*fl_chk);       % calibrated checker reaches 2x its floor
sig2_nai_c = cross_(spmm, err.checker.naive, 2*fl_chk);       % naive reaches 2x the same floor
say('\n---- the lesson ----\n');
say('blur-free floor (sigma=0, intrinsic Nyquist regularization at lambda=%.3f): checker %.1f%%, random %.1f%%\n', o.lam, 100*fl_chk, 100*fl_rnd);
say('a CALIBRATED read holds near that floor until the blur doubles it: checker 2x floor at sigma = %.2f pitch; the NAIVE read doubles the floor by sigma = %.2f pitch\n', ...
    sig2_cal_c/pitch, sig2_nai_c/pitch);
say('the structural knee is MTF(actuator Nyquist) = lambda at sigma = %.2f pitch -- beyond it the checkerboard is lost even to calibration\n', sig_lam/pitch);
say('=> blur is a concern only when its MTF at the DM''s highest spatial frequency approaches the reconstruction''s regularization/noise floor; below that a calibrated read removes it.\n');
say('   the real tg96 leg (tg96_pupilsim) broadens a single poke by <~1%% (width/true ~1.0) -- i.e. sigma ~ 0, the far-left of these curves.\n');

% ================= figures =================
fc = figure('Visible','off','Position',[100 100 1500 520]);
sp = o.sig_pitch;
subplot(1,3,1); hold on; grid on;
plot(sp, 100*err.checker.naive,'-o','Color',[.85 .33 .1],'DisplayName','checker, naive');
plot(sp, 100*err.checker.cal,  '-s','Color',[.85 .33 .1],'MarkerFaceColor',[.85 .33 .1],'DisplayName','checker, calibrated');
plot(sp, 100*err.random.naive, '-o','Color',[0 .45 .74],'DisplayName','random 30 nm, naive');
plot(sp, 100*err.random.cal,   '-s','Color',[0 .45 .74],'MarkerFaceColor',[0 .45 .74],'DisplayName','random 30 nm, calibrated');
xline(sig_lam/pitch,'k--','MTF@Nyq = \lambda','LabelOrientation','horizontal','HandleVisibility','off');
xlabel('blur 1/e radius / actuator pitch'); ylabel('actuator-command error, % of command rms');
title('Reconstruction error vs pupil blur'); legend('Location','northwest'); ylim([0 100]);
subplot(1,3,2); hold on; grid on;
ff = linspace(0,1.2*f_nyq,200);
for sgp = [0.3 0.6 1.0]
    plot(ff/f_nyq, exp(-2*pi^2*(sgp*pitch)^2*ff.^2),'DisplayName',sprintf('\\sigma = %.1f pitch',sgp));
end
xline(1,'k--','actuator Nyquist','LabelOrientation','horizontal','HandleVisibility','off');
yline(o.lam,'r:','\lambda floor','HandleVisibility','off');
xlabel('spatial frequency / actuator Nyquist'); ylabel('blur MTF');
title('Why: blur rolls off the high DM frequencies'); legend('Location','northeast');
subplot(1,3,3); hold on; grid on;
plot(sp, mtfN,'k-','DisplayName','MTF at actuator Nyquist');
plot(sp, err.checker.cal,'-s','Color',[.85 .33 .1],'DisplayName','checker calibrated error (frac)');
yline(o.lam,'r:','\lambda','HandleVisibility','off');
xlabel('blur 1/e radius / actuator pitch'); ylabel('fraction');
title('Nyquist MTF tracks the calibrated error floor'); legend('Location','northeast'); ylim([0 1]);
sgtitle('pupil\_blur\_demo: pupil-imaging blur vs DM-gauge reconstruction (engine-free)');
print(fc, fullfile(o.outdir,'pupil_blur_demo_curve.png'),'-dpng','-r110');

% example maps at a representative blur where the checker is clearly hurt (naive) but calibration still holds
is0 = find(o.sig_pitch >= 0.5,1);  if isempty(is0), is0 = nS; end
sig0 = o.sig_pitch(is0)*pitch;  K0 = blur_kernel_(sig0,dx);
Mu0 = conv2(Mu,K0,'same');  stn0 = dmg_stencil(Mu0,xg,0,0,pitch,o.hw);
m0 = conv2(surf.checker,K0,'same') + (o.noise_pm*1e-9)*randn(N);
aN = dmg_act_fit(m0,xg,axg,ayg,stn_true,lit,o.lam);
aC = dmg_act_fit(m0,xg,axg,ayg,stn0,lit,o.lam);
fm = figure('Visible','off','Position',[100 100 1500 480]);
subplot(1,4,1); imagesc(xg,xg,surf.checker*1e6); axis image off; colorbar; title('true surface, nm');
subplot(1,4,2); imagesc(xg,xg,m0*1e6); axis image off; colorbar; title(sprintf('blurred (\\sigma=%.1f pitch), nm',o.sig_pitch(is0)));
subplot(1,4,3); imagesc(xa,xa,(aN-pat.checker)*1e9.*lit); axis image off; colorbar; title(sprintf('naive cmd error, pm (%.1f%%)',100*err.checker.naive(is0)));
subplot(1,4,4); imagesc(xa,xa,(aC-pat.checker)*1e9.*lit); axis image off; colorbar; title(sprintf('calibrated cmd error, pm (%.1f%%)',100*err.checker.cal(is0)));
sgtitle(sprintf('pupil\\_blur\\_demo: the checkerboard (actuator Nyquist) at \\sigma = %.1f pitch',o.sig_pitch(is0)));
print(fm, fullfile(o.outdir,'pupil_blur_demo_maps.png'),'-dpng','-r110');

out = struct('o',o,'sig_pitch',o.sig_pitch,'mtf_nyq',mtfN,'err',err, ...
             'sig_lambda_mm',sig_lam,'lit',lit,'f_nyq',f_nyq);
say('\nwrote pupil_blur_demo_{report.txt, curve.png, maps.png} in %s\n', o.outdir);
end

% ---------------------------------------------------------------------------
function h = build_surf_(C, xa, X, Y, w_mm, R_ap)
% the DM surface = sum of Gaussian influence functions (dm_influence_map's model, meshgrid)
h = zeros(size(X));
for ia = 1:numel(xa), for ja = 1:numel(xa)
    if C(ia,ja)==0 || hypot(xa(ia),xa(ja))>R_ap, continue; end
    h = h + C(ia,ja)*exp(-((X-xa(ja)).^2 + (Y-xa(ia)).^2)/w_mm^2);  % meshgrid: row=y (ia), col=x (ja)
end, end
end
function K = blur_kernel_(sig, dx)
% normalized Gaussian PSF (1/e radius sig), truncated at 4 sigma; sig~0 -> delta
if sig < dx/4, K = 1; return; end
hw = max(1, ceil(4*sig/dx));  t = (-hw:hw)*dx;  [kx,ky] = meshgrid(t,t);
K = exp(-(kx.^2+ky.^2)/sig^2);  K = K/sum(K(:));
end
function e = relerr_(a_hat, a_true, lit)
e = sqrt(mean((a_hat(lit)-a_true(lit)).^2)) / sqrt(mean(a_true(lit).^2));
end
function s = cross_(x, y, thr)
i = find(y>thr,1);  if isempty(i), s = NaN; else, s = x(i); end
end
function deal_(~,~), end
