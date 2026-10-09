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
%     'legs'   {name, pupilsim report}  the built legs to place on the axis: each
%                     record's stage-2 (as built) Nyquist gain line (sinusoid 0.5 cyc/mm,
%                     min over u and v) -> the Gaussian 1/e radius with that MTF
%     'deck_share' [0.0013 0.0029]  the deck's raw-map pupil-imaging share (lens, mirror)
%     'deck_band_mm' [0.05 0.14]    the deck's stated blur width (shaded on the axis)
%     'kernel' 'both' 'gauss' (the swept Gaussian blur + the built legs), 'box' (the detector
%                     cell average: one reading per cell, cell size swept -- COPHI's
%                     photodiode array) or 'both'
%     'cells'  [0.5 1 1.5 2 3]  box cell sizes, PITCH units
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
           'estimator','matrix','lam_m',1e-3,'lam_m_sweep',[1e-2 1e-3 1e-4], ...
           'legs',{{'lens','runs/pupilsim_redo_lens/pupilsim_redo_lens_report.txt'; ...
                    'mirror','runs/pupilsim_redo_oap/pupilsim_redo_oap_report.txt'}}, ...
           'deck_share',[0.0013 0.0029], 'deck_band_mm',[0.05 0.14], ...
           'kernel','both', 'cells',[0.5 1 1.5 2 3]);
assert(any(strcmp(o.kernel,{'both','gauss','box'})), 'pupil_blur_demo:kernel', '''kernel'' is ''gauss'', ''box'' or ''both''');
doG = ~strcmp(o.kernel,'box');  doB = ~strcmp(o.kernel,'gauss') && ~isempty(o.cells);
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
say('error = rms(recovered - true) over the lit actuators, piston removed, / rms(true commands)\n');
say('APPROXIMATIONS (say them to a skeptic): the blur is applied to the PHASE map -- the small-phase limit (the 30 nm surface is ~0.1 wave of\n');
say('  wavefront); the camera blurs INTENSITY, and the four-step phase of blurred fringes equals the blurred phase only near null, which holds here.\n');
say('  The built leg is not a Gaussian: it is a phase gain cos(phi) with amplitude cross-talk sin(phi) plus a distortion; a Gaussian is matched to it\n');
say('  at the actuator-Nyquist MTF only.  The lit set is the demo''s %d actuators (0.85 x 0.74 R_ap); the benches light 5072 (lens) / 6948 (mirror)\n', nnz(lit));
say('  from the traced cone, and the redo pupilsim decks fill the DM (the DM is the stop) -- the error is a per-actuator rms, so the count sets the\n');
say('  edge-to-interior ratio, not the scale.\n\n');

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

if doG
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
% ================= the built leg on the axis (the pupilsim records) =================
say('\n---- the built leg on the axis (tg96_pupilsim redo records, stage 2 as built) ----\n');
nL = size(o.legs,1);  leg = struct('name',{},'gain',{},'sig_mm',{},'sig_lo',{},'sig_hi',{},'shift_mm',{});
for L = 1:nL
    f = o.legs{L,2};  if ~isfile(f), f = fullfile(here, f); end
    [g, shf] = nyq_gain_(f);
    sg = @(g) sqrt(max(0,-log(g))/(pi^2*f_nyq^2));                 % 1/e radius of the Gaussian with that Nyquist MTF
    leg(L) = struct('name',o.legs{L,1},'gain',g,'sig_mm',sg(g),'sig_lo',sg(min(1,g+5e-5)),'sig_hi',sg(g-5e-5),'shift_mm',shf);
    say('%-6s Nyquist gain (min over u, v) %.4f -> Gaussian 1/e radius %.4f pitch (%.4f..%.4f for the 4-digit print; std form %.4f) [%s]\n', ...
        leg(L).name, g, leg(L).sig_mm/pitch, leg(L).sig_lo/pitch, leg(L).sig_hi/pitch, leg(L).sig_mm/pitch/sqrt(2), o.legs{L,2});
end
pts = [[leg.sig_mm], o.deck_band_mm, sqrt(-log(0.98)/(pi^2*f_nyq^2))];
lab = [{leg.name}, {sprintf('deck %.2f mm', o.deck_band_mm(1)), sprintf('deck %.2f mm', o.deck_band_mm(2)), 'deck MTF 0.98'}];
say('%-14s %8s | %-37s | %-37s | %s\n', 'point', 'sig/pit', 'checker: kernel n/c, MATRIX n/c', 'random: kernel n/c, MATRIX n/c', 'random: blur cost naive (cmd) / map');
cost = zeros(size(pts));  mapc = cost;  legerr = struct();
x0 = struct();  for p = 1:2, x0.(names{p}) = est_matrix_(surf.(names{p}), Mx0, lit); end     % noise-free sigma = 0 read
for ip = 1:numel(pts)
    sig = pts(ip);  Mub = blur_(Mu, sig, dx);
    stn_eff = dmg_stencil(Mub, xg, 0, 0, pitch, o.hw);  Mxc = matrix_(Mub, ic, ilit, pix, Am, o.lam_m);
    e = zeros(2,4);
    for p = 1:2
        nm = names{p};  rng(o.seed + 500 + ip);  hb = blur_(surf.(nm), sig, dx);  m = hb + noise();
        e(p,:) = [relerr_(act_fit_lit_(m, ic, stn_true, lit, o.lam), pat.(nm), lit), relerr_(act_fit_lit_(m, ic, stn_eff, lit, o.lam), pat.(nm), lit), ...
                  relerr_(est_matrix_(m, Mx0, lit), pat.(nm), lit), relerr_(est_matrix_(m, Mxc, lit), pat.(nm), lit)];
        if p == 2
            % what the blur ADDS when it is ignored: the naive read of the blurred map against the read of the unblurred one (noise-free)
            cost(ip) = relerr_(est_matrix_(hb, Mx0, lit), x0.(nm), lit) * rmsp_(x0.(nm), lit)/rmsp_(pat.(nm), lit);
            mapc(ip) = maperr_(hb, surf.(nm), X, Y, 0.85*R_ill);        % tg96_pupilsim's metric: map - true, lit pupil, piston+tilt removed
        end
    end
    legerr.(sprintf('p%d',ip)) = e;
    say('%-14s %8.4f | %7.3f%% %7.3f%% %8.4f%% %8.4f%% | %7.3f%% %7.3f%% %8.4f%% %8.4f%% | %.4f%% / %.4f%%\n', lab{ip}, sig/pitch, 100*e(1,:), 100*e(2,:), 100*cost(ip), 100*mapc(ip));
end
say('THE CROSS-CHECK (Fang''s question): the blur the built leg would cost, ignored, on the random 30 nm surface, vs the deck''s raw-map pupil-imaging share:\n');
for L = 1:nL
    r = mapc(L)/o.deck_share(L);
    say('  %-6s Gaussian at the leg''s Nyquist MTF: %.4f%% (map) / %.4f%% (naive command read)  vs the deck''s %.2f%%  -> ratio %.2g%s\n', leg(L).name, ...
        100*mapc(L), 100*cost(L), 100*o.deck_share(L), r, iff_(r < 1/3 || r > 3, '  ** MORE THAN 3x: a FINDING, not a rounding', ''));
end
% the Gaussian that WOULD give the deck's share (the map metric scales as sigma^2 at these widths: from the deck-0.05 mm row)
s_eq = o.deck_band_mm(1)*sqrt(o.deck_share/mapc(nL+1));  leg_shift_cost = zeros(1,nL);
say('  a Gaussian would need a 1/e radius of %.3f mm (lens) / %.3f mm (mirror) to cost the deck''s shares -- inside the deck''s stated 0.05-0.14 mm, i.e. the deck''s\n', s_eq);
say('  width and the records'' Nyquist gain line describe DIFFERENT things: a gain 0.9994-0.9999 at Nyquist is a 1/e radius of 0.006-0.016 pitch, not 0.05.\n');
say('  What a gain line cannot show is error OUT of phase with the surface -- a registration residual.  The records'' recovered single pokes are shifted\n');
for L = 1:nL
    sh = shift_(surf.random, leg(L).shift_mm, dx);  leg_shift_cost(L) = maperr_(sh, surf.random, X, Y, 0.85*R_ill);
    say('  %-6s by up to %.4f mm; the random surface shifted by that much costs %.4f%% (map metric) = %.2g of the deck''s share\n', leg(L).name, leg(L).shift_mm, ...
        100*maperr_(sh, surf.random, X, Y, 0.85*R_ill), maperr_(sh, surf.random, X, Y, 0.85*R_ill)/o.deck_share(L));
end
say('  read: the leg''s Nyquist MTF alone accounts for %.0f%% (lens) / %.0f%% (mirror) of the deck''s share; the rest is what the Gaussian does not model.\n', ...
    100*mapc(1)/o.deck_share(1), 100*mapc(2)/o.deck_share(2));
say('  The records'' error by band also puts the mirror''s largest share in the LOWEST band (0-0.06 cyc/mm: 0.50%% of the surface there), which no blur produces.\n');
say('  The demo''s answer to "is blur a concern": no -- at the built leg''s MTF the blur costs %.3f%% / %.3f%% even uncalibrated; the deck''s share is an upper\n', 100*mapc(1), 100*mapc(2));
say('  bound on the pupil imaging that carries more than blur.\n');
end   % doG (the Gaussian sweep and the built legs)
if ~doG, nS = 0;  err = struct();  mtfN = [];  leg = struct([]);  pts = [];  lab = {};  legerr = struct();  cost = [];  mapc = [];  leg_shift_cost = [];  nL = 0; end
% ================= the BOX kernel: a detector cell's average (COPHI's photodiode array) =================
% a coarse element averages the phase over its square cell and gives ONE reading per
% cell: the cell average + sampling on the cell grid, with exact pixel-area weights
% (cell edges on the actuator boundaries).  CALIBRATED: the response matrix of the
% cell readings (every column cell-averaged) -- what a calibration through the array
% measures.  NAIVE: each reading taken as the surface at the cell centre (the
% influences point-sampled there).  Read noise o.noise_pm per ELEMENT.  Matrix
% estimator only: the kernel form needs a map at the actuator sites, which a cell
% coarser than the pitch does not give.
boxr = struct('cell',o.cells,'nread',nan(size(o.cells)),'per_act',nan(size(o.cells)), ...
              'checker',struct('naive',nan(size(o.cells)),'cal',nan(size(o.cells))), ...
              'random', struct('naive',nan(size(o.cells)),'cal',nan(size(o.cells))));
if doB
    say('\n---- the BOX kernel: one reading per detector cell (the cell average), matrix estimator, lambda_m %.0e, %g pm per element ----\n', o.lam_m, o.noise_pm);
    say('%-10s %8s %9s | %-24s | %-24s\n', 'cell/pitch', 'readings', 'per lit', 'checker: naive / cal', 'random: naive / cal');
    Jfull = cols_(Mu, ic, ilit, N);                              % the unit influences, full grid (N^2 x lit)
    for ib = 1:numel(o.cells)
        c = o.cells(ib)*pitch;
        e0 = -c*ceil((R_ap + 2*pitch)/c);  edges = e0:c:-e0;       % cell edges on the actuator boundaries (0 is an edge)
        O = overlap_(edges, xg, dx);                              % cells x grid, the 1-D area weights / c
        ctr = edges(1:end-1) + c/2;  [CX, CY] = meshgrid(ctr);
        inc = hypot(CX, CY) + c/sqrt(2) < R_ill;                  % cells wholly inside the illuminated pupil
        B = kron(O, O);  B = B(inc(:), :);                        % vec(O H O') = kron(O,O) vec(H): cell readings
        Jc = B*Jfull;                                             % CALIBRATED: the matrix through the cells
        Jn = infl_at_(CX(inc), CY(inc), axg(ilit), ayg(ilit), w_mm);   % NAIVE: the influences at the cell centres
        Fc = fact_(Jc, o.lam_m);  Fn = fact_(Jn, o.lam_m);
        boxr.nread(ib) = nnz(inc);  boxr.per_act(ib) = nnz(inc)/numel(ilit);
        for p = 1:2
            nm = names{p};  rng(o.seed + 700 + ib);
            y = B*surf.(nm)(:) + (o.noise_pm*1e-9)*randn(nnz(inc),1);
            boxr.(nm).naive(ib) = relerr_(solve_(y, Fn, ilit, lit), pat.(nm), lit);
            boxr.(nm).cal(ib)   = relerr_(solve_(y, Fc, ilit, lit), pat.(nm), lit);
        end
        say('%-10.2f %8d %9.3f | %8.3f%% / %8.3f%% | %8.3f%% / %8.3f%%\n', o.cells(ib), boxr.nread(ib), boxr.per_act(ib), ...
            100*boxr.checker.naive(ib), 100*boxr.checker.cal(ib), 100*boxr.random.naive(ib), 100*boxr.random.cal(ib));
    end
    say('THE LINE (COPHI''s resolution half), calibrated:\n');
    for ib = 1:numel(o.cells)
        say('  a %.1f-pitch cell (%.2f readings per lit actuator) %s the checkerboard (%.1f%% error) and costs %.2f%% on the 30 nm surface\n', o.cells(ib), ...
            boxr.per_act(ib), iff_(boxr.checker.cal(ib) > 0.5, 'LOSES', 'keeps'), 100*boxr.checker.cal(ib), 100*boxr.random.cal(ib));
    end
    say('  (fewer readings than lit actuators -- cells of a pitch or more -- leaves the solve underdetermined: what it returns is the regularized\n');
    say('  minimum-norm read; the camera''s %d pixels per actuator are the other end of the trade)\n', o.sub^2);
end

hc = 'mcal';  hn = 'mnaive';  hlab = 'matrix (the record''s estimator, every deck number)';
if strcmp(o.estimator,'kernel'), hc = 'cal';  hn = 'naive';  hlab = 'kernel (the bench''s calib_mode ''kernel'', S3 flavor)'; end

if doG
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
else
    sig_lam = NaN;
end

% ================= figures =================
if o.figures
fc = figure('Visible','off','Position',[100 100 1800 1000]);  cc = {[.85 .33 .1], [0 .45 .74]};
if doG
sp = o.sig_pitch;  ttl = {'checkerboard (actuator Nyquist), \pm50 nm', 'random 30 nm working surface'};
for p = 1:2
    nm = names{p};  e = err.(nm);
    subplot(2,3,p); hold on; grid on;
    plot(sp, 100*e.naive, ':o', 'Color', cc{p}, 'DisplayName', 'kernel, naive');
    plot(sp, 100*e.cal,   ':s', 'Color', cc{p}, 'DisplayName', 'kernel, calibrated');
    plot(sp, 100*e.mnaive,'-o', 'Color', 'k',   'DisplayName', 'matrix, naive');
    plot(sp, 100*e.mcal,  '-s', 'Color', 'k', 'MarkerFaceColor', 'k', 'DisplayName', 'matrix, calibrated (the record''s)');
    patch([o.deck_band_mm fliplr(o.deck_band_mm)]/pitch, [0 0 100 100], [.6 .6 .6], 'FaceAlpha', .15, 'EdgeColor', 'none', 'DisplayName', 'the deck''s stated blur width');
    xlabel('blur 1/e radius / actuator pitch'); ylabel('actuator-command error, % of command rms');
    title(ttl{p}); legend('Location', iff_(p == 1, 'east', 'northwest')); ylim([0 100]);
end
subplot(2,3,4); hold on; grid on;
plot(sp, mtfN,'k-','DisplayName','blur MTF at actuator Nyquist');
plot(sp, err.checker.mcal,'-s','Color',cc{1},'DisplayName','checker, matrix calibrated (frac)');
plot(sp, err.checker.cal,':s','Color',cc{1},'DisplayName','checker, kernel calibrated (frac)');
xline(sig_lam/pitch,'k--','kernel knee','LabelOrientation','horizontal','HandleVisibility','off');
xlabel('blur 1/e radius / actuator pitch'); ylabel('fraction');
title('Nyquist MTF and the calibrated error'); legend('Location','east'); ylim([0 1]);
% the cross-check, log-log: what the blur costs IGNORED (tg96_pupilsim's map metric) vs sigma, the legs, the deck's shares
subplot(2,3,[5 6]); hold on; grid on;  set(gca,'XScale','log','YScale','log');
sz = logspace(log10(0.003), log10(0.3), 25)*pitch;  mz = arrayfun(@(q) maperr_(blur_(surf.random, q, dx), surf.random, X, Y, 0.85*R_ill), sz);
plot(sz/pitch, 100*mz, 'k-', 'DisplayName', 'Gaussian blur, ignored (random 30 nm, map metric)');
lc = {[0 .45 .74], [.85 .33 .1]};
for L = 1:nL
    plot(leg(L).sig_mm/pitch, 100*mapc(L), 'o', 'Color', lc{L}, 'MarkerFaceColor', lc{L}, 'MarkerSize', 8, ...
        'DisplayName', sprintf('%s leg: Nyquist gain %.4f -> %.3f pitch', leg(L).name, leg(L).gain, leg(L).sig_mm/pitch));
    yline(100*o.deck_share(L), '--', 'Color', lc{L}, 'DisplayName', sprintf('deck: %s pupil-imaging share %.2f%%', leg(L).name, 100*o.deck_share(L)));
    plot(leg(L).sig_mm/pitch, 100*leg_shift_cost(L), 'd', 'Color', lc{L}, 'MarkerSize', 8, ...
        'DisplayName', sprintf('%s: a %.4f mm registration shift (the record''s poke shift)', leg(L).name, leg(L).shift_mm));
end
patch([o.deck_band_mm fliplr(o.deck_band_mm)]/pitch, [1e-4 1e-4 10 10], [.6 .6 .6], 'FaceAlpha', .15, 'EdgeColor', 'none', 'DisplayName', 'the deck''s stated blur width');
xlim([0.003 0.3]);  ylim([1e-4 10]);
xlabel('blur 1/e radius / actuator pitch'); ylabel('error, % of the surface (piston + tilt removed)');
title('The cross-check: the built legs vs the deck''s pupil-imaging share'); legend('Location','northwest','FontSize',8);
end
if doB
    subplot(2,3,3); hold on; grid on;
    plot(o.cells, 100*boxr.checker.naive, ':o', 'Color', cc{1}, 'DisplayName', 'checker, naive (cell = point sample)');
    plot(o.cells, 100*boxr.checker.cal,   '-s', 'Color', cc{1}, 'MarkerFaceColor', cc{1}, 'DisplayName', 'checker, calibrated (cell-averaged matrix)');
    plot(o.cells, 100*boxr.random.naive,  ':o', 'Color', cc{2}, 'DisplayName', 'random 30 nm, naive');
    plot(o.cells, 100*boxr.random.cal,    '-s', 'Color', cc{2}, 'MarkerFaceColor', cc{2}, 'DisplayName', 'random 30 nm, calibrated');
    xline(1, 'k--', 'one reading per actuator', 'LabelOrientation', 'horizontal', 'HandleVisibility', 'off');
    xlabel('detector cell / actuator pitch'); ylabel('actuator-command error, % of command rms');
    title('BOX: one reading per detector cell (COPHI''s array)'); legend('Location','southeast','FontSize',8); ylim([0 105]);
end
sgtitle({'pupil\_blur\_demo: pupil-imaging blur vs DM-gauge reconstruction (engine-free)', ...
    'blur applied to the PHASE map (the small-phase limit); the built legs are a cos\phi gain + sin\phi cross-talk,', ...
    sprintf('matched to a Gaussian at the actuator-Nyquist MTF only; %d lit actuators (0.85 x 0.74 R_{ap})', nnz(lit))}, 'FontSize', 11);
print(fc, fullfile(o.outdir,'pupil_blur_demo_curve.png'),'-dpng','-r110');

if doG
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
end

out = struct('box',boxr,'leg',leg,'leg_pts',pts,'leg_lab',{lab},'leg_err',legerr,'leg_cost',cost,'leg_map',mapc,'o',o,'sig_pitch',o.sig_pitch,'mtf_nyq',mtfN,'err',err,'floor',fl,'headline',struct('cal',hc,'naive',hn), ...
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
function J = cols_(Mub, ic, ilit, N)
% the unit-influence columns on the FULL grid (N^2 x lit): Mub shifted to each site
c0 = (N+1)/2;  [rr, cc] = find(abs(Mub) > 1e-10*max(abs(Mub(:))));
r = max(abs([rr; cc] - c0));  w = Mub(c0-r:c0+r, c0-r:c0+r);  [dc, dr] = meshgrid(-r:r, -r:r);
nl = numel(ilit);  nw = numel(w);  I = zeros(nw*nl,1);  Jc = I;  V = I;
[ia, ja] = ind2sub([numel(ic) numel(ic)], ilit);
for k = 1:nl
    q = (k-1)*nw + (1:nw);  I(q) = sub2ind([N N], ic(ia(k)) + dr(:), ic(ja(k)) + dc(:));  Jc(q) = k;  V(q) = w(:);
end
J = sparse(I, Jc, V, N*N, nl);
end
function F = fact_(J, lam)
% est_matrix_tg's normal equations for a reading vector: mean-referenced, lam*median(diag)
J = sparse(J);  Am = size(J,1);  v = full(sum(J,1)).';
JtJ = full(J.'*J) - (v*v.')/Am;  d = diag(JtJ);
F = struct('J', J, 'v', v, 'Am', Am, 'Rf', chol(JtJ + lam*median(d(d>0))*eye(size(J,2))));
end
function a = solve_(y, F, ilit, lit)
b = F.J.'*y - F.v*(sum(y)/F.Am);  x = F.Rf \ (F.Rf.' \ b);
a = zeros(size(lit));  a(ilit) = x;
end
function O = overlap_(edges, xg, dx)
% 1-D area weights: O(k,j) = |[edges(k), edges(k+1)] ^ [xg(j)-dx/2, xg(j)+dx/2]| / cell width
lo = max(edges(1:end-1).', xg - dx/2);  hi = min(edges(2:end).', xg + dx/2);
O = sparse(max(0, hi - lo) ./ (edges(2:end) - edges(1:end-1)).');
end
function J = infl_at_(xc, yc, ax, ay, w)
% the Gaussian influences point-sampled at (xc, yc): readings x lit, cut at 5 w
[I, K] = find(sparse(abs(xc(:) - ax(:).') < 5*w & abs(yc(:) - ay(:).') < 5*w));
V = exp(-((xc(I) - ax(K)).^2 + (yc(I) - ay(K)).^2)/w^2);
J = sparse(I, K, V, numel(xc), numel(ax));
end
function a = est_matrix_(h, Mx, lit)
% est_matrix_tg verbatim: b = J'h - v (1'h)/Am over the mask, x = Rf \ (Rf' \ b)
hm = h(Mx.pix > 0);  hm = hm(:);  [~, ord] = sort(Mx.pix(Mx.pix > 0));  hm = hm(ord);
b = Mx.J.'*hm - Mx.v*(sum(hm)/Mx.Am);
x = Mx.Rf \ (Mx.Rf.' \ b);
a = zeros(size(lit));  a(Mx.ilit) = x;
end
function [g, shf] = nyq_gain_(f)
% the record's stage-2 AS-BUILT Nyquist gain: the first two 'sinusoid f 0.5000' lines (u, v), min
t = fileread(f);  tk = regexp(t, 'sinusoid f 0\.5000 cyc/mm[^\n]*?min ([0-9.]+) in the lit pupil', 'tokens');
assert(numel(tk) >= 2, 'pupil_blur_demo:leg', 'no Nyquist gain lines in %s', f);
g = min(str2double(tk{1}{1}), str2double(tk{2}{1}));
% the largest recovered-poke centroid shift of the FIRST (as-built) stage-2 block
b = regexp(t, '---- stage 2, as built.*?working surface', 'match', 'once');
sk = regexp(b, 'shift ([0-9.]+) mm', 'tokens');  shf = max(str2double([sk{:}]));
end
function b = shift_(h, d, dx)
% the map translated by d (mm) along x and y equally (|shift| = d), exactly, by a Fourier phase ramp
N = size(h,1);  f = ifftshift(((0:N-1) - floor(N/2))/(N*dx));  [FX,FY] = meshgrid(f,f);
b = real(ifft2(fft2(h).*exp(-2i*pi*(FX+FY)*d/sqrt(2))));
end
function r = rmsp_(a, lit), x = a(lit) - mean(a(lit));  r = sqrt(mean(x.^2)); end
function e = maperr_(hb, h, X, Y, R)
% tg96_pupilsim's working-surface metric: (map - true) in the lit pupil, piston and tilt removed, / the surface rms there
in = hypot(X,Y) < R;  A = [ones(nnz(in),1) X(in) Y(in)];
d = hb(in) - h(in);  d = d - A*(A\d);  t = h(in) - mean(h(in));
e = sqrt(mean(d.^2))/sqrt(mean(t.^2));
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
