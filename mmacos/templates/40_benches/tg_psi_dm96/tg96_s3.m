function out = tg96_s3()
%TG96_S3  Stage 3, IFO twin of zwfs_dm96/zwfs_s3.m: the modal-calibrated
%   battery for the interferometer.  Per DM size (96x96 at 1 mm pitch;
%   48x48 at 2 mm -- DST-class DM, same bench): the 12-mode lattice
%   transfer THROUGH the actuator-space estimator (true influence
%   kernel), the Wiener modal correction (radial -- the IFO transfer is
%   near-isotropic; beta 0.1), and the battery rows in pm: null,
%   piston, held-out poke, held-out random 10 nm (raw vs corrected).
%   RETROFITTED onto ../dm_gauge_lib (2026-09-05; one copy of the
%   scoring machinery before S4).  Re-run reproduces the 10cf593 report.
%   Run:  cd <this dir>;  matlab -batch "tg96_s3"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
addpath(fullfile(exdir, '..', 'dm_gauge_lib'));
t_all = tic;
rep = fopen('tg96_s3_report.txt', 'w');

s = 96/56;  LAM = 6.328e-4;  QWP = 0.25;  THETAS = [0 45 90 135];
MODEL = 1024;  NGRID = 385;  N_G = 384;  DX_G = 0.28;
AOI = 7;  D_BS_TO = 700;
BETA = 0.1;

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

dmg_say(rep, '=== IFO S3: modal-calibrated battery ===\n');

IFO = dmg_ifo_gauge(AT, AR, QWP, THETAS, LAM);
msk = IFO.msk;  measI = IFO.meas;

macos.load_rx(AT.rx);
[mag, dxd_mm] = dmg_frame(AT.iTO, AT.iDET);
N_WF = size(IFO.I0,1);
xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
[gxd, gyd] = meshgrid(xg, xg);
PARb = [1 2 1 1];  sgn = -1;                  % eprime registration
R0 = struct('P',PARb, 'tax',0, 'tay',0, 'bx',0, 'by',0, ...
    'dxd_mm',dxd_mm, 'mag',mag, 'msk',msk, 'N_WF',N_WF, ...
    'gxd',gxd, 'gyd',gyd);

CFG = struct('nact', {96, 48}, 'pitch', {1.0, 2.0}, ...
    'pq', {[1 1;2 2;4 4;8 8;16 16;24 24;32 32;48 48;64 64;80 80;96 96;48 0], ...
           [1 1;2 2;4 4;8 8;12 12;16 16;24 24;32 32;40 40;48 48;24 0]});

out = struct();
for icfg = 1:2
    NACT = CFG(icfg).nact;  PITCH = CFG(icfg).pitch;  PQ = CFG(icfg).pq;
    dmg_say(rep, '\n---- DM %dx%d, pitch %.1f mm ----\n', NACT, NACT, PITCH);
    dmap = @(act) dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', act);

    ic = NACT/2;  POKE = 20e-6;
    Aa = zeros(NACT);  Aa(ic,ic) = 1;
    Ma = dmap(150e-6*Aa);                      % strong anchor for translation
    hA = measI(Ma);
    R = R0;
    [R.bx, R.by, R.tax, R.tay] = dmg_anchor(hA, Ma, msk, N_WF, xg);
    Mu = dmap(Aa);                             % unit influence kernel
    HW = 6;
    stn = dmg_stencil(Mu, xg, R.tax, R.tay, PITCH, HW);
    [axg, ayg] = meshgrid(((1:NACT)-(NACT+1)/2)*PITCH);
    [lit, ~] = dmg_lit(msk, dxd_mm, mag, axg, ayg);
    est = @(h) dmg_act_fit(sgn*dmg_samp(h, R), xg, axg, ayg, stn, lit);

    AMPM = 10e-6;
    [ii, jj] = meshgrid((0.5:NACT)/NACT);
    nm_modes = size(PQ,1);
    gk = zeros(nm_modes,1);  fk = zeros(nm_modes,1);
    dmg_say(rep, 'modal transfer (through the actuator-space estimator):\n');
    for k = 1:nm_modes
        p = PQ(k,1);  q = PQ(k,2);
        Ak = cos(pi*p*ii).*cos(pi*q*jj);
        aK = est(measI(dmap(AMPM*Ak)));
        gk(k) = (AMPM*Ak(lit)) \ aK(lit);
        fk(k) = hypot(p, q)/2;
        dmg_say(rep, '  mode(%2d,%2d)  %5.1f cyc/ap  gain %7.4f\n', p, q, fk(k), gk(k));
    end
    corr = @(a) dmg_modal_corr(a, 'radial', fk, gk, BETA, NACT);

    a_null = est(measI(dmap(zeros(NACT))));
    dmg_say(rep, 'null: %.2f pm rms (estimator on the flat state)\n', std(a_null(lit))*1e9);
    a_pist = est(measI(dmap(20e-6*ones(NACT))));
    dmg_say(rep, 'piston 20 nm: mean gain %.4f\n', mean(a_pist(lit))/20e-6);

    ih = round(NACT*0.625);  jh = round(NACT*0.417);
    Ah = zeros(NACT);  Ah(ih,jh) = 1;
    aH = est(measI(dmap(POKE*Ah)));
    aHc = corr(aH);
    dmg_say(rep, 'held-out poke (%d,%d) 20 nm: raw gain %.4f, modal-corrected %.4f, err %.1f pm\n', ...
         ih, jh, aH(ih,jh)/POKE, aHc(ih,jh)/POKE, ...
         sqrt(mean((aHc(lit) - POKE*Ah(lit)).^2))*1e9);

    rng(23);
    Ar = zeros(NACT);  Ar(lit) = 10e-6*randn(nnz(lit),1);
    aR = est(measI(dmap(Ar)));
    aRc = corr(aR);
    gr_raw = Ar(lit)\aR(lit);   er_raw = sqrt(mean((aR(lit)-Ar(lit)).^2))*1e9;
    gr_cor = Ar(lit)\aRc(lit);  er_cor = sqrt(mean((aRc(lit)-Ar(lit)).^2))*1e9;
    dmg_say(rep, 'held-out random 10 nm: raw gain %.4f / %.0f pm; modal-corrected %.4f / %.0f pm\n', ...
         gr_raw, er_raw, gr_cor, er_cor);

    out.(sprintf('n%d', NACT)) = struct('gk',gk, 'fk',fk, ...
        'er_raw',er_raw, 'er_cor',er_cor);
end
dmg_say(rep, 'S3 complete in %.1f min\n', toc(t_all)/60);
fclose(rep);
save('tg96_s3.mat', 'out');
fprintf('wrote tg96_s3_report.txt + tg96_s3.mat\n');
end
