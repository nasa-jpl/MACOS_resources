function out = zwfs_s4()
%ZWFS_S4  Stage 4: the differential head-to-head rows + break scale.
%   (Dave: "S4: the four differential rows per instrument per DM size
%   through the CALIBRATED estimators ... working-state BREAK SCALE
%   (grow base until linear folds; stepped should hold), all pm.")
%   Per DM size (96x96 @ 1 mm; 48x48 @ 2 mm):
%     ROWS: bases {flat, random 30 nm rms} x devs {single act 10 nm,
%       random 10 nm rms}; differential (measure base, base+dev,
%       difference), actuator-space fit (measured kernel), separable
%       Wiener modal correction (from zwfs_s3.mat) -- BOTH readings
%       reported: frozen-reference LINEAR (the small-differential
%       workhorse) and PHASE-STEPPED (the range/base-crosstalk owner);
%       the complementary-reconstructor doctrine (S2b).
%     BREAK SCALE: base amplitude ladder 30..480 nm rms (same rng(7)
%       field, scaled), single-act 10 nm differential per rung; the
%       linear reading is expected to FOLD as the base phase leaves the
%       linear regime, the stepped reading to HOLD (+-pi differential).
%   Scoring: gain at poked site(s), resid over lit actuators, floor =
%   rms of unpoked lit actuators, all pm.  Machinery: ../dm_gauge_lib.
%   Run:  cd <this dir>;  matlab -batch "zwfs_s4"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
addpath(fullfile(exdir, '..', 'dm_gauge_lib'));
t_all = tic;
rep = fopen('zwfs_s4_report.txt', 'w');

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
LADDER = [30 60 120 240 480]*1e-6;             % base rms, mm

dmg_say(rep, '=== ZWFS S4: differential head-to-head rows + break scale (pm) ===\n');

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
PARb = [1 2 1 1];  sgn = +1;                  % S2 registration, this deck
R0 = struct('P',PARb, 'tax',0, 'tay',0, 'bx',0, 'by',0, ...
    'dxd_mm',dxd_mm, 'mag',mag, 'msk',msk, 'N_WF',N_WF, ...
    'gxd',gxd, 'gyd',gyd);

% modal calibration measured by zwfs_s3 (separable, (p,0) rows)
S3 = load('zwfs_s3.mat');
PQ96 = [1 0;2 0;4 0;8 0;16 0;24 0;32 0;48 0;64 0;80 0;8 8;24 24];
PQ48 = [1 0;2 0;4 0;8 0;12 0;16 0;24 0;32 0;40 0;8 8;16 16];
CFG = struct('nact', {96, 48}, 'pitch', {1.0, 2.0}, ...
    'pq', {PQ96, PQ48}, 's3', {S3.out.n96, S3.out.n48}, ...
    'hold', {[60 40], [30 20]});

out = struct();
for icfg = 1:2
    NACT = CFG(icfg).nact;  PITCH = CFG(icfg).pitch;
    PQ = CFG(icfg).pq;  hij = CFG(icfg).hold;
    dmg_say(rep, '\n---- DM %dx%d, pitch %.1f mm ----\n', NACT, NACT, PITCH);
    dmap = @(act) dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', act);

    % registration anchor + measured kernel (the S2/S3 recipe)
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
    is1d = PQ(:,2) == 0;
    corr = @(a) dmg_modal_corr(a, 'separable', PQ(is1d,1), ...
                               CFG(icfg).s3.gk(is1d), BETA, NACT);

    % ---- the four rows ---------------------------------------------
    rng(7);
    Ab30 = zeros(NACT);  Ab30(lit) = 30e-6*randn(nnz(lit),1);
    Asng = zeros(NACT);  Asng(hij(1),hij(2)) = 10e-6;
    rng(23);
    Arnd = zeros(NACT);  Arnd(lit) = 10e-6*randn(nnz(lit),1);
    bases = {zeros(NACT), 'flat'; Ab30, 'rand30nm'};
    devs  = {Asng, 'single10nm'; Arnd, 'rand10nm'};
    dmg_say(rep, '%-9s %-11s | %7s %9s | %7s %9s\n', 'base', 'dev', ...
        'g_lin', 'res_lin', 'g_stp', 'res_stp');
    rows = struct('base',{},'dev',{},'g_lin',{},'e_lin',{}, ...
                  'g_stp',{},'e_stp',{});
    for ib = 1:2
        h0 = ZW.measL(dmap(bases{ib,1}));
        X0 = ZW.steppedX(dmap(bases{ib,1}));
        for id = 1:2
            Ad = devs{id,1};
            M1 = dmap(bases{ib,1} + Ad);
            dl = ZW.measL(M1) - h0;
            aL = corr(est(dl));
            X1 = ZW.steppedX(M1);
            aS = corr(est(ZW.stepdiff(X1, X0)));
            gl = Ad(lit) \ aL(lit);
            el = sqrt(mean((aL(lit) - Ad(lit)).^2))*1e9;
            gs = Ad(lit) \ aS(lit);
            es = sqrt(mean((aS(lit) - Ad(lit)).^2))*1e9;
            dmg_say(rep, '%-9s %-11s | %7.4f %7.0f pm | %7.4f %7.0f pm\n', ...
                bases{ib,2}, devs{id,2}, gl, el, gs, es);
            rows(end+1) = struct('base',bases{ib,2}, 'dev',devs{id,2}, ...
                'g_lin',gl, 'e_lin',el, 'g_stp',gs, 'e_stp',es); %#ok<AGROW>
        end
    end

    % ---- break scale: grow the working state -----------------------
    dmg_say(rep, 'break scale (single act 10 nm differential on a growing base):\n');
    dmg_say(rep, '%9s | %7s %9s %7s | %7s %9s %7s\n', 'base_rms', ...
        'g_lin', 'flr_lin', 'SNRl', 'g_stp', 'flr_stp', 'SNRs');
    rng(7);
    Bfield = zeros(NACT);  Bfield(lit) = randn(nnz(lit),1);
    Bfield = Bfield / std(Bfield(lit));        % unit-rms base field
    pkm = Asng > 0;  un = lit & ~pkm;
    lad = struct('amp',{},'g_lin',{},'flr_lin',{},'snr_lin',{}, ...
                 'g_stp',{},'flr_stp',{},'snr_stp',{});
    for amp = LADDER
        Ab = amp * Bfield;
        h0 = ZW.measL(dmap(Ab));
        X0 = ZW.steppedX(dmap(Ab));
        M1 = dmap(Ab + Asng);
        aL = corr(est(ZW.measL(M1) - h0));
        aS = corr(est(ZW.stepdiff(ZW.steppedX(M1), X0)));
        gl = aL(hij(1),hij(2))/10e-6;  fl = std(aL(un))*1e9;
        gs = aS(hij(1),hij(2))/10e-6;  fs = std(aS(un))*1e9;
        snl = abs(gl)*1e4/max(fl, eps);        % 10 nm = 1e4 pm
        sns = abs(gs)*1e4/max(fs, eps);
        dmg_say(rep, '%6.0f nm | %7.4f %7.0f pm %7.1f | %7.4f %7.0f pm %7.1f\n', ...
            amp*1e6, gl, fl, snl, gs, fs, sns);
        lad(end+1) = struct('amp',amp, 'g_lin',gl, 'flr_lin',fl, ...
            'snr_lin',snl, 'g_stp',gs, 'flr_stp',fs, 'snr_stp',sns); %#ok<AGROW>
    end
    out.(sprintf('n%d', NACT)) = struct('rows',rows, 'ladder',lad);
end
dmg_say(rep, 'S4 complete in %.1f min\n', toc(t_all)/60);
fclose(rep);
save('zwfs_s4.mat', 'out');
fprintf('wrote zwfs_s4_report.txt + zwfs_s4.mat\n');
end
