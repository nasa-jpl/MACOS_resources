function out = tg96_s5noise()
%TG96_S5NOISE  Stage 5, IFO twin of zwfs_dm96/zwfs_s5noise.m: photon-
%   noise pricing of the 1 pm target.  Same scenario (single-act 10 nm
%   differential on the rng(7) 30 nm base, 96x96), same axis (photons
%   per DM STATE, the four-step splits the budget across its 4 frames),
%   same method (noiseless frames captured once; shot noise injected
%   numerically; calibration incl. p_null treated as noiseless).
%   Run:  cd <this dir>;  matlab -batch "tg96_s5noise"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
addpath(fullfile(exdir, '..', 'dm_gauge_lib'));
t_all = tic;
rep = fopen('tg96_s5noise_report.txt', 'w');

s = 96/56;  LAM = 6.328e-4;  QWP = 0.25;  THETAS = [0 45 90 135];
MODEL = 1024;  NGRID = 385;  N_G = 384;  DX_G = 0.28;
AOI = 7;  D_BS_TO = 700;
BETA = 0.1;  NACT = 96;  PITCH = 1.0;
NSTATES = 10.^(6:2:14);
NREAL = 8;

macos.init(MODEL);
tl = load('tg96_tail.mat');
T_FL_F = tl.out.FL_F;  T_FL_Kc = tl.out.FL_Kc;
T_DMF  = tl.out.D_MASK_FL;  T_TRIM = tl.out.DET_TRIM;
macos.write_grid_file('tg96_flat.txt', zeros(N_G));
G = macos.design.twyman_green('polarizing',true, 'ngridpts',NGRID, ...
    'BS_AOI',AOI, ...
    'F1',s*500, 'F2',s*250, 'D_LENS',s*60, 'R_BAFFLE',s*12.5, 'D_SB',s*250, ...
    'BS_T',s*1.5, 'D_L1_BS',s*150, 'D_BS_TO',D_BS_TO, 'D_BS_CMP',s*100, ...
    'R_TO_AP',s*30, 'L1_Kr',s*236.866, 'L1_Kc',-0.5829, ...
    'L2_Kr',-s*124.076, 'L2_Kc',-0.5826, ...
    'to_grid_file','tg96_flat.txt', 'to_grid_n',N_G, 'to_grid_dx',DX_G, ...
    'qwp_ret',QWP, 'pol_in_deg',45, 'qwp_test_deg',0, 'qwp_ref_deg',45, ...
    'out_qwp_deg',0, 'analyzer_deg',0, ...
    'tail_arch','fieldlens', 'FL_F',T_FL_F, 'FL_Kc',T_FL_Kc, ...
    'FL_D',s*12, 'D_MASK_FL',T_DMF, 'DET_TRIM',T_TRIM);
G.bt.emit('tg96_test.in');  G.br.emit('tg96_ref.in');
AT = dmg_arm_desc('tg96_test.in', G.bt, G.T, 0);
AR = dmg_arm_desc('tg96_ref.in',  G.br, G.R, 45);

dmg_say(rep, '=== IFO S5: photon-noise pricing of the 1 pm target ===\n');
dmg_say(rep, 'scenario: single act (60,40) 10 nm differential on the 30 nm base, 96x96\n');
dmg_say(rep, 'axis: photons per DM STATE, split across the 4 analyzer frames\n');

IFO = dmg_ifo_gauge(AT, AR, QWP, THETAS, LAM);
msk = IFO.msk;

macos.load_rx(AT.rx);
[mag, dxd_mm] = dmg_frame(AT.iTO, AT.iDET);
N_WF = size(IFO.I0,1);
xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
[gxd, gyd] = meshgrid(xg, xg);
PARb = [1 2 1 1];  sgn = -1;
R0 = struct('P',PARb, 'tax',0, 'tay',0, 'bx',0, 'by',0, ...
    'dxd_mm',dxd_mm, 'mag',mag, 'msk',msk, 'N_WF',N_WF, ...
    'gxd',gxd, 'gyd',gyd);

dmap = @(act) dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', act);
ic = NACT/2;
Aa = zeros(NACT);  Aa(ic,ic) = 1;
Ma = dmap(150e-6*Aa);
hA = IFO.meas(Ma);
R = R0;
[R.bx, R.by, R.tax, R.tay] = dmg_anchor(hA, Ma, msk, N_WF, xg);
stn = dmg_stencil(dmap(Aa), xg, R.tax, R.tay, PITCH, 6);
[axg, ayg] = meshgrid(((1:NACT)-(NACT+1)/2)*PITCH);
lit = dmg_lit(msk, dxd_mm, mag, axg, ayg);
est = @(h) dmg_act_fit(sgn*dmg_samp(h, R), xg, axg, ayg, stn, lit);
S3 = load('tg96_s3.mat');
corr = @(a) dmg_modal_corr(a, 'radial', S3.out.n96.fk, S3.out.n96.gk, ...
                           BETA, NACT);
wsafe = @(h1, h0) angle(exp(1i*(h1 - h0)*4*pi/LAM))*LAM/(4*pi);

% ---- capture the noiseless frames (once per state) -----------------
rng(7);
Ab30 = zeros(NACT);  Ab30(lit) = 30e-6*randn(nnz(lit),1);
Asng = zeros(NACT);  Asng(60,40) = 10e-6;
F0 = IFO.frames(dmap(Ab30));                   % 4 analyzer frames/state
F1 = IFO.frames(dmap(Ab30 + Asng));
dmg_say(rep, 'frames captured (%.1f min); Monte-Carlo is trace-free\n', toc(t_all)/60);

% ---- noise sweep ----------------------------------------------------
noisy = @(I, nph) I .* (1 + randn(size(I)) ./ ...
                        sqrt(max(I / sum(I(:)) * nph, 1)));
res = struct('n_state',{},'sig_pk_pm',{},'flr_pm',{},'g_mean',{});
dmg_say(rep, '%10s | %10s %10s %8s\n', 'N/state', 'sig_pk', 'floor', 'gain');
for n = NSTATES
    pk = zeros(NREAL,1);  fl = zeros(NREAL,1);
    for r = 1:NREAL
        rng(1000 + r + round(log10(n))*100);
        F0n = F0;  F1n = F1;
        for k = 1:4
            F0n(:,:,k) = noisy(F0(:,:,k), n/4);
            F1n(:,:,k) = noisy(F1(:,:,k), n/4);
        end
        aD = corr(est(wsafe(IFO.recon(F1n), IFO.recon(F0n))));
        pk(r) = aD(60,40);
        un = lit;  un(60,40) = false;
        fl(r) = std(aD(un));
    end
    sg = std(pk)*1e9;
    dmg_say(rep, '%10.1e | %8.1f pm %8.1f pm %8.3f\n', ...
        n, sg, mean(fl)*1e9, mean(pk)/10e-6);
    res(end+1) = struct('n_state',n, 'sig_pk_pm',sg, ...
        'flr_pm',mean(fl)*1e9, 'g_mean',mean(pk)/10e-6); %#ok<AGROW>
end
sig = [res.sig_pk_pm];  nn = [res.n_state];
use = sig > 0.5;
if nnz(use) >= 2
    c = exp(mean(log(sig(use)) + 0.5*log(nn(use))));
    dmg_say(rep, 'fourstep: sigma ~ %.3g/sqrt(N) pm  ->  N(1 pm) ~ %.2e photons/state\n', ...
        c, c^2);
end
dmg_say(rep, 'S5 noise complete in %.1f min\n', toc(t_all)/60);
fclose(rep);
save('tg96_s5noise.mat', 'res');
fprintf('wrote tg96_s5noise_report.txt + tg96_s5noise.mat\n');
out = res;
end
