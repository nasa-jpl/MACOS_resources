function out = tg96_s4()
%TG96_S4  Stage 4, IFO twin of zwfs_dm96/zwfs_s4.m: the differential
%   head-to-head rows + break scale, through the CALIBRATED estimator
%   (true influence kernel + radial Wiener modal correction from
%   tg96_s3.mat).  Per DM size:
%     ROWS: bases {flat, random 30 nm} x devs {single act 10 nm,
%       random 10 nm rms}; wrap-safe four-step differential (the E'
%       protocol), actuator fit, modal correction; gain + resid, pm.
%     BREAK SCALE: base ladder 30..480 nm rms, single-act 10 nm
%       differential per rung -- where does the four-step differential
%       fold?  (The PSI reading wraps at +-lambda/4 per pixel of
%       DIFFERENTIAL; the base cancels in the difference, so the fold
%       is set by the base's effect on the frames, not the wrap.)
%   Machinery: ../dm_gauge_lib.  Run: cd <this dir>; matlab -batch "tg96_s4"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
addpath(fullfile(exdir, '..', 'dm_gauge_lib'));
t_all = tic;
rep = fopen('tg96_s4_report.txt', 'w');

s = 96/56;  LAM = 6.328e-4;  QWP = 0.25;  THETAS = [0 45 90 135];
MODEL = 1024;  NGRID = 385;  N_G = 384;  DX_G = 0.28;
AOI = 7;  D_BS_TO = 700;
BETA = 0.1;
LADDER = [30 60 120 240 480]*1e-6;

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

dmg_say(rep, '=== IFO S4: differential head-to-head rows + break scale (pm) ===\n');

IFO = dmg_ifo_gauge(AT, AR, QWP, THETAS, LAM);
msk = IFO.msk;  measI = IFO.meas;
wsafe = @(h1, h0) angle(exp(1i*(h1 - h0)*4*pi/LAM))*LAM/(4*pi);

macos.load_rx(AT.rx);
[mag, dxd_mm] = dmg_frame(AT.iTO, AT.iDET);
N_WF = size(IFO.I0,1);
xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
[gxd, gyd] = meshgrid(xg, xg);
PARb = [1 2 1 1];  sgn = -1;                  % eprime registration
R0 = struct('P',PARb, 'tax',0, 'tay',0, 'bx',0, 'by',0, ...
    'dxd_mm',dxd_mm, 'mag',mag, 'msk',msk, 'N_WF',N_WF, ...
    'gxd',gxd, 'gyd',gyd);

S3 = load('tg96_s3.mat');
CFG = struct('nact', {96, 48}, 'pitch', {1.0, 2.0}, ...
    's3', {S3.out.n96, S3.out.n48}, 'hold', {[60 40], [30 20]});

out = struct();
for icfg = 1:2
    NACT = CFG(icfg).nact;  PITCH = CFG(icfg).pitch;  hij = CFG(icfg).hold;
    dmg_say(rep, '\n---- DM %dx%d, pitch %.1f mm ----\n', NACT, NACT, PITCH);
    dmap = @(act) dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', act);

    ic = NACT/2;
    Aa = zeros(NACT);  Aa(ic,ic) = 1;
    Ma = dmap(150e-6*Aa);                      % strong anchor
    hA = measI(Ma);
    R = R0;
    [R.bx, R.by, R.tax, R.tay] = dmg_anchor(hA, Ma, msk, N_WF, xg);
    stn = dmg_stencil(dmap(Aa), xg, R.tax, R.tay, PITCH, 6);   % true kernel
    [axg, ayg] = meshgrid(((1:NACT)-(NACT+1)/2)*PITCH);
    lit = dmg_lit(msk, dxd_mm, mag, axg, ayg);
    est = @(h) dmg_act_fit(sgn*dmg_samp(h, R), xg, axg, ayg, stn, lit);
    corr = @(a) dmg_modal_corr(a, 'radial', CFG(icfg).s3.fk, ...
                               CFG(icfg).s3.gk, BETA, NACT);

    % ---- the four rows ---------------------------------------------
    rng(7);
    Ab30 = zeros(NACT);  Ab30(lit) = 30e-6*randn(nnz(lit),1);
    Asng = zeros(NACT);  Asng(hij(1),hij(2)) = 10e-6;
    rng(23);
    Arnd = zeros(NACT);  Arnd(lit) = 10e-6*randn(nnz(lit),1);
    bases = {zeros(NACT), 'flat'; Ab30, 'rand30nm'};
    devs  = {Asng, 'single10nm'; Arnd, 'rand10nm'};
    dmg_say(rep, '%-9s %-11s | %7s %9s\n', 'base', 'dev', 'gain', 'resid');
    rows = struct('base',{},'dev',{},'g',{},'e_pm',{});
    for ib = 1:2
        h0 = measI(dmap(bases{ib,1}));
        for id = 1:2
            Ad = devs{id,1};
            h1 = measI(dmap(bases{ib,1} + Ad));
            aD = corr(est(wsafe(h1, h0)));
            g = Ad(lit) \ aD(lit);
            e = sqrt(mean((aD(lit) - Ad(lit)).^2))*1e9;
            dmg_say(rep, '%-9s %-11s | %7.4f %7.0f pm\n', ...
                bases{ib,2}, devs{id,2}, g, e);
            rows(end+1) = struct('base',bases{ib,2}, 'dev',devs{id,2}, ...
                'g',g, 'e_pm',e); %#ok<AGROW>
        end
    end

    % ---- break scale ------------------------------------------------
    dmg_say(rep, 'break scale (single act 10 nm differential on a growing base):\n');
    dmg_say(rep, '%9s | %7s %9s %7s\n', 'base_rms', 'gain', 'floor', 'SNR');
    rng(7);
    Bfield = zeros(NACT);  Bfield(lit) = randn(nnz(lit),1);
    Bfield = Bfield / std(Bfield(lit));
    pkm = Asng > 0;  un = lit & ~pkm; %#ok<NASGU>
    lad = struct('amp',{},'g',{},'flr',{},'snr',{});
    for amp = LADDER
        Ab = amp * Bfield;
        h0 = measI(dmap(Ab));
        h1 = measI(dmap(Ab + Asng));
        aD = corr(est(wsafe(h1, h0)));
        g = aD(hij(1),hij(2))/10e-6;
        fl = std(aD(un))*1e9;
        sn = abs(g)*1e4/max(fl, eps);
        dmg_say(rep, '%6.0f nm | %7.4f %7.0f pm %7.1f\n', amp*1e6, g, fl, sn);
        lad(end+1) = struct('amp',amp, 'g',g, 'flr',fl, 'snr',sn); %#ok<AGROW>
    end
    out.(sprintf('n%d', NACT)) = struct('rows',rows, 'ladder',lad);
end
dmg_say(rep, 'S4 complete in %.1f min\n', toc(t_all)/60);
fclose(rep);
save('tg96_s4.mat', 'out');
fprintf('wrote tg96_s4_report.txt + tg96_s4.mat\n');
end
