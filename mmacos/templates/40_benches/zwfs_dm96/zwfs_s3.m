function out = zwfs_s3()
%ZWFS_S3  Stage 3: the MODAL-calibrated battery (Dave "Go!" x2).
%   For each DM size (96x96 at 1 mm pitch; 48x48 at 2 mm -- the
%   DST-class DM on the same 96 mm bench):
%     1. registration (two pokes) + measured response kernel (S2);
%     2. the 12-mode lattice-cosine transfer THROUGH the actuator-space
%        estimator = the modal calibration;
%     3. Wiener modal correction (separable transfer, beta = 0.1)
%        applied on the actuator lattice by FFT;
%     4. battery rows in pm: null, piston 20 nm, held-out single poke,
%        held-out random 10 nm (raw vs modal-corrected);
%     5. THE RESCUE: grid pokes on the 30 nm base -- re-measured with
%        the PHASE-STEPPED retrieval (S2b) + modal correction.
%   Frozen-linear reconstructor for the battery (small differentials);
%   stepped retrieval where range/base-crosstalk demands it -- the
%   complementary-reconstructor doctrine from S2b.
%   RETROFITTED onto ../dm_gauge_lib (2026-09-05; one copy of the
%   scoring machinery before S4).  Re-run reproduces the 10cf593 report.
%   Run:  cd <this dir>;  matlab -batch "zwfs_s3"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
addpath(fullfile(exdir, '..', 'dm_gauge_lib'));
t_all = tic;
rep = fopen('zwfs_s3_report.txt', 'w');

s = 96/56;  LAM = 6.328e-4;
MODEL = 1024;  NGRID = 193;  N_G = 384;  DX_G = 0.28;
AOI = 7;  D_BS_TO = 700;  R_BEAM = s*30;
T_FL_F = 42.5325;  T_FL_Kc = -2.58764;  T_DMF = 39.7694;  T_TRIM = -1.2473;
MASK_TRIM = -5.582;
N_FS = 1.45702;  ETCH_MM = 346.2e-6;
PHI_M = 2*pi*(N_FS-1)*ETCH_MM/LAM;
DIA_LAMD = 2.0;  S_CONV = -1;
PHIS = [pi/2, pi, 3*pi/2];
BETA = 0.1;                                    % Wiener parameter

dmg_say(rep, '=== ZWFS S3: modal-calibrated battery (frozen-linear + stepped rescue) ===\n');

% ---- bench (one build; DM size changes only the command lattice) ---
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

% flat references + masks + ray frame (dm_gauge_lib)
ZW = dmg_zwfs_gauge(iTO, iMASK, iDET, struct('LAM',LAM, 'F2',s*250, ...
    'R_BEAM',R_BEAM, 'DIA_LAMD',DIA_LAMD, 'PHI_M',PHI_M, 'PHIS',PHIS, ...
    'S_CONV',S_CONV));
N_WF = ZW.N_WF;  msk = ZW.msk;
[mag, dxd_mm] = dmg_frame(iTO, iDET);
xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
[gxd, gyd] = meshgrid(xg, xg);
PARb = [1 2 1 1];  sgn = +1;                  % S2 registration, this deck
R0 = struct('P',PARb, 'tax',0, 'tay',0, 'bx',0, 'by',0, ...
    'dxd_mm',dxd_mm, 'mag',mag, 'msk',msk, 'N_WF',N_WF, ...
    'gxd',gxd, 'gyd',gyd);

% (p,0) probes measure the 1-D transfer (the kernel is SEPARABLE --
% the radial assumption measured anisotropy on the first run); (p,p)
% rows validate separability.  p = NACT is degenerate: a zero command.
CFG = struct('nact', {96, 48}, 'pitch', {1.0, 2.0}, ...
    'pq', {[1 0;2 0;4 0;8 0;16 0;24 0;32 0;48 0;64 0;80 0;8 8;24 24], ...
           [1 0;2 0;4 0;8 0;12 0;16 0;24 0;32 0;40 0;8 8;16 16]});

out = struct();
for icfg = 1:2
    NACT = CFG(icfg).nact;  PITCH = CFG(icfg).pitch;  PQ = CFG(icfg).pq;
    dmg_say(rep, '\n---- DM %dx%d, pitch %.1f mm ----\n', NACT, NACT, PITCH);
    dmap = @(act) dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', act);
    measL = ZW.measL;

    % kernel from a center poke (fresh per lattice)
    ic = NACT/2;  POKE = 20e-6;
    Aa = zeros(NACT);  Aa(ic,ic) = 1;
    Ma = dmap(POKE*Aa);
    hA = measL(Ma);
    R = R0;
    [R.bx, R.by, R.tax, R.tay] = dmg_anchor(hA, Ma, msk, N_WF, xg);
    hAd = sgn*dmg_samp(hA, R);
    hAd(isnan(hAd)) = 0;
    HW = 6;
    stn = dmg_stencil(hAd, xg, R.tax, R.tay, PITCH, HW) / POKE;
    [axg, ayg] = meshgrid(((1:NACT)-(NACT+1)/2)*PITCH);
    [lit, ~] = dmg_lit(msk, dxd_mm, mag, axg, ayg);
    est = @(h) dmg_act_fit(sgn*dmg_samp(h, R), xg, axg, ayg, stn, lit);

    % ---- modal transfer through the estimator ----------------------
    AMPM = 10e-6;
    [ii, jj] = meshgrid((0.5:NACT)/NACT);
    nm_modes = size(PQ,1);
    gk = zeros(nm_modes,1);  fk = zeros(nm_modes,1);
    dmg_say(rep, 'modal transfer (through the actuator-space estimator):\n');
    for k = 1:nm_modes
        p = PQ(k,1);  q = PQ(k,2);
        Ak = cos(pi*p*ii).*cos(pi*q*jj);
        aK = est(measL(dmap(AMPM*Ak)));
        gk(k) = (AMPM*Ak(lit)) \ aK(lit);
        fk(k) = hypot(p, q)/2;
        dmg_say(rep, '  mode(%2d,%2d)  %5.1f cyc/ap  gain %7.4f\n', p, q, fk(k), gk(k));
    end
    % separable Wiener from the (p,0) rows; diagonals validate
    is1d = PQ(:,2) == 0;
    pk1 = PQ(is1d,1);  gk1 = gk(is1d);
    for k = find(~is1d).'
        ppp = PQ(k,1);
        g1p = interp1([0; pk1], [gk1(1); gk1], ppp, 'linear');
        dmg_say(rep, '  separability check (%d,%d): measured %.4f, g1(%d)^2 = %.4f\n', ...
             ppp, ppp, gk(k), ppp, g1p^2);
    end
    corr = @(a) dmg_modal_corr(a, 'separable', pk1, gk1, BETA, NACT);

    % ---- battery rows ----------------------------------------------
    a_null = est(measL(dmap(zeros(NACT))));
    dmg_say(rep, 'null: %.2f pm rms (estimator on the flat state)\n', std(a_null(lit))*1e9);
    a_pist = est(measL(dmap(20e-6*ones(NACT))));
    dmg_say(rep, 'piston 20 nm: mean gain %.4f (piston is INVISIBLE to a ZWFS -- expected 0; the IFO reads 0.98)\n', mean(a_pist(lit))/20e-6);

    ih = round(NACT*0.625);  jh = round(NACT*0.417);
    Ah = zeros(NACT);  Ah(ih,jh) = 1;
    aH = est(measL(dmap(POKE*Ah)));
    aHc = corr(aH);
    dmg_say(rep, 'held-out poke (%d,%d) 20 nm: raw gain %.4f, modal-corrected %.4f, err %.1f pm\n', ...
         ih, jh, aH(ih,jh)/POKE, aHc(ih,jh)/POKE, ...
         sqrt(mean((aHc(lit) - POKE*Ah(lit)).^2))*1e9);

    rng(23);
    Ar = zeros(NACT);  Ar(lit) = 10e-6*randn(nnz(lit),1);
    aR = est(measL(dmap(Ar)));
    aRc = corr(aR);
    gr_raw = Ar(lit)\aR(lit);   er_raw = sqrt(mean((aR(lit)-Ar(lit)).^2))*1e9;
    gr_cor = Ar(lit)\aRc(lit);  er_cor = sqrt(mean((aRc(lit)-Ar(lit)).^2))*1e9;
    dmg_say(rep, 'held-out random 10 nm: raw gain %.4f / %.0f pm; modal-corrected %.4f / %.0f pm\n', ...
         gr_raw, er_raw, gr_cor, er_cor);

    % ---- THE RESCUE: grid pokes on the 30 nm base, STEPPED ----------
    Pg = zeros(NACT);  Pg(8:8:NACT, 8:8:NACT) = 1;  Pg = Pg .* lit;
    rng(7);
    Ab30 = zeros(NACT);  Ab30(lit) = 30e-6*randn(nnz(lit),1);
    AMPG = 1e-6;                                % 1 nm grid pokes
    X0 = ZW.steppedX(dmap(Ab30));
    X1 = ZW.steppedX(dmap(Ab30 + AMPG*Pg));
    hstep = ZW.stepdiff(X1, X0);
    aS = dmg_act_fit(sgn*dmg_samp(hstep, R), xg, axg, ayg, stn, lit);
    aSc = corr(aS);
    pk = Pg > 0;  un = lit & ~pk;
    snr_st = mean(aSc(pk)) / max(std(aSc(un)), eps);
    % frozen-linear comparison on the same scenario
    hlin1 = measL(dmap(Ab30 + AMPG*Pg));
    hlin0 = measL(dmap(Ab30));
    aL = dmg_act_fit(sgn*dmg_samp(hlin1-hlin0, R), xg, axg, ayg, stn, lit);
    aLc = corr(aL);
    snr_ln = mean(aLc(pk)) / max(std(aLc(un)), eps);
    dmg_say(rep, 'RESCUE grid(%d)-on-30nm-base at 1 nm: stepped SNR %.2f (linear %.2f; sens stage 1.64)\n', ...
         nnz(pk), snr_st, snr_ln);
    dmg_say(rep, '  stepped: gain %.4f, floor %.3g pm\n', mean(aS(pk))/AMPG, std(aSc(un))*1e9);

    out.(sprintf('n%d', NACT)) = struct('gk',gk, 'fk',fk, 'snr_st',snr_st, ...
        'snr_ln',snr_ln, 'er_raw',er_raw, 'er_cor',er_cor);
end
dmg_say(rep, 'S3 complete in %.1f min\n', toc(t_all)/60);
fclose(rep);
save('zwfs_s3.mat', 'out');
fprintf('wrote zwfs_s3_report.txt + zwfs_s3.mat\n');
end
