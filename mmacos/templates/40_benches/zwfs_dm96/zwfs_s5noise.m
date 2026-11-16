function out = zwfs_s5noise()
%ZWFS_S5NOISE  Stage 5: photon-noise pricing of the 1 pm target (ZWFS).
%   The noiseless campaign (S4) established the systematic floors; this
%   stage prices the NOISE contribution.  Scenario = the head-to-head
%   row: single-act 10 nm differential on the rng(7) 30 nm working
%   state, 96x96 DM, actuator-space scoring through the calibrated
%   estimator.
%
%   Method: the optical fields do not depend on noise, so the noiseless
%   frames are captured ONCE per DM state and shot noise is injected
%   numerically (relative sigma = 1/sqrt(n_photons_pixel), Gaussian
%   approximation) between capture and reconstruction -- zero re-trace
%   cost per realization.  Calibration (den / I_flat / b2cal /
%   registration / kernel / modal gains) is treated as NOISELESS (the
%   long-exposure assumption; a real budget adds a calibration term).
%
%   Photon axis: N_STATE = detected photons per DM STATE, split equally
%   across the frames that a reading needs -- linear 1 frame/state,
%   stepped 4 frames/state (the IFO twin uses 4).  A differential
%   measurement spends 2 x N_STATE regardless of modality, so the axis
%   prices the modalities at equal light and equal time.
%
%   Output per (reading, N_STATE): sigma_noise of the poked-site
%   estimate over NREAL realizations (pm), the unpoked noise floor, and
%   the 1/sqrt(N) fit with the extrapolated N_STATE at sigma = 1 pm.
%   Run:  cd <this dir>;  matlab -batch "zwfs_s5noise"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
addpath(fullfile(exdir, '..', 'dm_gauge_lib'));
t_all = tic;
rep = fopen('zwfs_s5noise_report.txt', 'w');

s = 96/56;  LAM = 6.328e-4;
MODEL = 1024;  NGRID = 193;  N_G = 384;  DX_G = 0.28;
AOI = 7;  D_BS_TO = 700;  R_BEAM = s*30;
T_FL_F = 42.5325;  T_FL_Kc = -2.58764;  T_DMF = 39.7694;  T_TRIM = -1.2473;
MASK_TRIM = -5.582;
N_FS = 1.45702;  ETCH_MM = 346.2e-6;
PHI_M = 2*pi*(N_FS-1)*ETCH_MM/LAM;
DIA_LAMD = 2.0;  S_CONV = -1;
PHIS = [pi/2, pi, 3*pi/2];
BETA = 0.1;
NACT = 96;  PITCH = 1.0;
NSTATES = 10.^(6:2:14);                        % photons per DM state
NREAL = 8;

dmg_say(rep, '=== ZWFS S5: photon-noise pricing of the 1 pm target ===\n');
dmg_say(rep, 'scenario: single act (60,40) 10 nm differential on the 30 nm base, 96x96\n');
dmg_say(rep, 'axis: photons per DM STATE (linear splits over 1 frame, stepped over 4)\n');

macos.init(MODEL);
macos.write_grid_file('zwfs_flat.txt', zeros(N_G));
G = macos.design.twyman_green('polarizing',false, 'ngridpts',NGRID, ...
    'BS_AOI',AOI, ...
    'F1',s*500, 'F2',s*250, 'D_LENS',s*60, 'R_BAFFLE',s*12.5, 'D_SB',s*250, ...
    'BS_T',s*1.5, 'D_L1_BS',s*150, 'D_BS_TO',D_BS_TO, 'D_BS_CMP',s*100, ...
    'R_TO_AP',s*30, 'L1_Kr',s*236.866, 'L1_Kc',-0.5829, ...
    'L2_Kr',-s*124.076, 'L2_Kc',-0.5826, ...
    'to_grid_file','zwfs_flat.txt', 'to_grid_n',N_G, 'to_grid_dx',DX_G, ...
    'tail_arch','fieldlens', 'mask_prop','nf', 'MASK_TRIM',MASK_TRIM, ...
    'FL_F',T_FL_F, 'FL_Kc',T_FL_Kc, ...
    'FL_D',s*12, 'D_MASK_FL',T_DMF, 'DET_TRIM',T_TRIM);
G.bt.emit('zwfs_test.in');
iTO = G.T.iTO;  iMASK = G.T.iMASK;  iDET = G.T.iDET;
macos.load_rx('zwfs_test.in');

ZW = dmg_zwfs_gauge(iTO, iMASK, iDET, struct('LAM',LAM, 'F2',s*250, ...
    'R_BEAM',R_BEAM, 'DIA_LAMD',DIA_LAMD, 'PHI_M',PHI_M, 'PHIS',PHIS, ...
    'S_CONV',S_CONV));
N_WF = ZW.N_WF;  msk = ZW.msk;
[mag, dxd_mm] = dmg_frame(iTO, iDET);
xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
[gxd, gyd] = meshgrid(xg, xg);
PARb = [1 2 1 1];  sgn = +1;
R0 = struct('P',PARb, 'tax',0, 'tay',0, 'bx',0, 'by',0, ...
    'dxd_mm',dxd_mm, 'mag',mag, 'msk',msk, 'N_WF',N_WF, ...
    'gxd',gxd, 'gyd',gyd);

dmap = @(act) dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', act);
ic = NACT/2;  POKE = 20e-6;
Aa = zeros(NACT);  Aa(ic,ic) = 1;
Ma = dmap(POKE*Aa);
hA = ZW.measL(Ma);
R = R0;
[R.bx, R.by, R.tax, R.tay] = dmg_anchor(hA, Ma, msk, N_WF, xg);
hAd = sgn*dmg_samp(hA, R);  hAd(isnan(hAd)) = 0;
stn = dmg_stencil(hAd, xg, R.tax, R.tay, PITCH, 6) / POKE;
[axg, ayg] = meshgrid(((1:NACT)-(NACT+1)/2)*PITCH);
lit = dmg_lit(msk, dxd_mm, mag, axg, ayg);
est = @(h) dmg_act_fit(sgn*dmg_samp(h, R), xg, axg, ayg, stn, lit);
S3 = load('zwfs_s3.mat');
PQ = [1 0;2 0;4 0;8 0;16 0;24 0;32 0;48 0;64 0;80 0;8 8;24 24];
is1d = PQ(:,2) == 0;
corr = @(a) dmg_modal_corr(a, 'separable', PQ(is1d,1), ...
                           S3.out.n96.gk(is1d), BETA, NACT);

% ---- capture the noiseless frames (once per state) -----------------
rng(7);
Ab30 = zeros(NACT);  Ab30(lit) = 30e-6*randn(nnz(lit),1);
Asng = zeros(NACT);  Asng(60,40) = 10e-6;
M0 = dmap(Ab30);  M1 = dmap(Ab30 + Asng);
IL0 = ZW.frameL(M0);   IL1 = ZW.frameL(M1);          % linear: 1 frame/state
FS0 = ZW.framesS(M0);  FS1 = ZW.framesS(M1);         % stepped: 4 frames/state
dmg_say(rep, 'frames captured (%.1f min); Monte-Carlo is trace-free\n', toc(t_all)/60);

% ---- noise sweep ----------------------------------------------------
noisy = @(I, nph) I .* (1 + randn(size(I)) ./ ...
                        sqrt(max(I / sum(I(:)) * nph, 1)));
res = struct('reading',{},'n_state',{},'sig_pk_pm',{},'flr_pm',{},'g_mean',{});
dmg_say(rep, '%10s | %10s %10s %8s | %10s %10s %8s\n', 'N/state', ...
    'sig_lin', 'flr_lin', 'g_lin', 'sig_stp', 'flr_stp', 'g_stp');
for n = NSTATES
    pk_l = zeros(NREAL,1);  fl_l = zeros(NREAL,1);
    pk_s = zeros(NREAL,1);  fl_s = zeros(NREAL,1);
    for r = 1:NREAL
        rng(1000 + r + round(log10(n))*100);
        % linear: whole state budget in the one frame
        aL = corr(est(ZW.reconL(noisy(IL1, n)) - ZW.reconL(noisy(IL0, n))));
        % stepped: budget split across the 4 frames
        F0n = FS0;  F1n = FS1;
        for k = 1:4
            F0n(:,:,k) = noisy(FS0(:,:,k), n/4);
            F1n(:,:,k) = noisy(FS1(:,:,k), n/4);
        end
        aS = corr(est(ZW.stepdiff(ZW.reconS(F1n), ZW.reconS(F0n))));
        pk_l(r) = aL(60,40);  pk_s(r) = aS(60,40);
        un = lit;  un(60,40) = false;
        fl_l(r) = std(aL(un));  fl_s(r) = std(aS(un));
    end
    sl = std(pk_l)*1e9;  ss = std(pk_s)*1e9;
    dmg_say(rep, '%10.1e | %8.1f pm %8.1f pm %8.3f | %8.1f pm %8.1f pm %8.3f\n', ...
        n, sl, mean(fl_l)*1e9, mean(pk_l)/10e-6, ...
           ss, mean(fl_s)*1e9, mean(pk_s)/10e-6);
    res(end+1) = struct('reading','both', 'n_state',n, ...
        'sig_pk_pm',[sl ss], 'flr_pm',[mean(fl_l) mean(fl_s)]*1e9, ...
        'g_mean',[mean(pk_l) mean(pk_s)]/10e-6); %#ok<AGROW>
end
% 1/sqrt(N) pricing from the shot-noise-dominated points (sig > 5x the
% noiseless jitter): fit sig = c/sqrt(N), quote N at 1 pm.
sig_l = arrayfun(@(q) q.sig_pk_pm(1), res);
sig_s = arrayfun(@(q) q.sig_pk_pm(2), res);
nn = [res.n_state];
for pair = {{'linear', sig_l}, {'stepped', sig_s}}
    nmr = pair{1}{1};  sg = pair{1}{2};
    use = sg > 0.5;                             % above numeric jitter, pm
    if nnz(use) >= 2
        c = exp(mean(log(sg(use)) + 0.5*log(nn(use))));
        dmg_say(rep, '%s: sigma ~ %.3g/sqrt(N) pm  ->  N(1 pm) ~ %.2e photons/state\n', ...
            nmr, c, c^2);
    end
end
dmg_say(rep, 'S5 noise complete in %.1f min\n', toc(t_all)/60);
fclose(rep);
save('zwfs_s5noise.mat', 'res');
fprintf('wrote zwfs_s5noise_report.txt + zwfs_s5noise.mat\n');
out = res;
end
