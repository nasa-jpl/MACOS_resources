function out = zwfs_s6color()
%ZWFS_S6COLOR  Stage 6: MULTI-COLOR combination (Dave 2026-09-08: "try
%   running both systems at multiple colors, maybe the combination will
%   help with some poor SNR regions").
%   The sensor is measured at K wavelengths with ONE physical mask (the
%   346.2 nm fused-silica etch, 2.0 lam/D at 633 nm; Malitson index), so
%   the dimple phase, its size in lam/D, the reference wave and hence the
%   response KERNEL are all chromatic; each color gets its OWN calibration
%   (flat references, den, b2cal, anchor, measured kernel, modal
%   transfer).  The poor-SNR regions on record (S3/S4, 96x96): the
%   OSCILLATORY fine-scale transfer ((64,0) -0.52 at 32 cyc/ap), the
%   grid-on-30nm-base rescue row (stepped SNR 2.32), dense random
%   (12-42 nm).  Combination = multi-channel Wiener on the actuator
%   lattice (../dm_gauge_lib/dmg_color_comb): a_hat(f) = sum_k G_k A_k /
%   (sum_k G_k^2 + beta^2) -- where one color's transfer crosses zero
%   another's need not -- scored against every single color, the plain
%   mean, and the best pair.  Noiseless model (the S5 verdict: photons
%   are not the blocker); K colors cost K x the frames -- stated, not
%   priced.  Only the deck's header Wavelen= changes per color; the QWP-
%   free ZWFS train is dispersionless (IndRef fixed), so the chromatic
%   content is diffraction + the mask + the DM phase scaling 4 pi h/lam.
%   Run:  cd <this dir>;  matlab -batch "zwfs_s6color; exit(0)"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
addpath(fullfile(exdir, '..', 'dm_gauge_lib'));
t_all = tic;
rep = fopen('zwfs_s6color_report.txt', 'w');

s = 96/56;  LAM0 = 6.328e-4;
MODEL = 1024;  NGRID = 193;  N_G = 384;  DX_G = 0.28;
AOI = 7;  D_BS_TO = 700;  R_BEAM = s*30;  F2 = s*250;
T_FL_F = 42.5325;  T_FL_Kc = -2.58764;  T_DMF = 39.7694;  T_TRIM = -1.2473;
MASK_TRIM = -5.582;
ETCH_MM = 346.2e-6;
DIA_LAMD0 = 2.0;  S_CONV = -1;
PHIS0 = [pi/2, pi, 3*pi/2];                  % depth ladder: phases AT 633 nm
BETA = 0.1;
NACT = 96;  PITCH = 1.0;
LAMS_NM = [632.8 480 532 700 780];           % 632.8 (the record) FIRST: lit + tie-in
K = numel(LAMS_NM);
PQ = [1 0;2 0;4 0;8 0;16 0;24 0;32 0;40 0;48 0;56 0;64 0;72 0;80 0;8 8;24 24];
is1d = PQ(:,2) == 0;  pk1 = PQ(is1d,1);
hij = [60 40];
% fused silica, Malitson 1965 (n = 1.45702 at 632.8 nm)
n_fs = @(um) sqrt(1 + 0.6961663*um^2/(um^2-0.0684043^2) ...
                    + 0.4079426*um^2/(um^2-0.1162414^2) ...
                    + 0.8974794*um^2/(um^2-9.896161^2));
n0 = n_fs(LAM0*1e3);

dmg_say(rep, '=== ZWFS S6 COLOR: multi-wavelength combination (96x96, actuator space, pm) ===\n');
dmg_say(rep, 'colors: %s nm; ONE physical mask (2.0 lam/D at 633 = %.4e mm, 346.2 nm FS etch, Malitson n);\n', ...
    num2str(LAMS_NM), DIA_LAMD0*LAM0*F2/(2*R_BEAM));
dmg_say(rep, 'model %d, NGRID %d; per-color calibration; lit + bases from 633; beta %.2f; combiner DC form unit\n', ...
    MODEL, NGRID, BETA);

% ---- bench (one build); per-color decks differ ONLY in Wavelen= ----
macos.init(MODEL);
macos.write_grid_file('zwfs_flat.txt', zeros(N_G));
G = macos.design.twyman_green('polarizing',false, 'ngridpts',NGRID, ...
    'BS_AOI',AOI, ...
    'F1',s*500, 'F2',F2, 'D_LENS',s*60, 'R_BAFFLE',s*12.5, 'D_SB',s*250, ...
    'BS_T',s*1.5, 'D_L1_BS',s*150, 'D_BS_TO',D_BS_TO, 'D_BS_CMP',s*100, ...
    'R_TO_AP',s*30, 'L1_Kr',s*236.866, 'L1_Kc',-0.5829, ...
    'L2_Kr',-s*124.076, 'L2_Kc',-0.5826, ...
    'to_grid_file','zwfs_flat.txt', 'to_grid_n',N_G, 'to_grid_dx',DX_G, ...
    'tail_arch','fieldlens', 'mask_prop','nf', 'MASK_TRIM',MASK_TRIM, ...
    'FL_F',T_FL_F, 'FL_Kc',T_FL_Kc, ...
    'FL_D',s*12, 'D_MASK_FL',T_DMF, 'DET_TRIM',T_TRIM);
G.bt.emit('zwfs_test.in');
iTO = G.T.iTO;  iMASK = G.T.iMASK;  iDET = G.T.iDET;
decks = cell(1, K);
for k = 1:K, decks{k} = color_deck_('zwfs_test.in', LAMS_NM(k)); end

xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
[gxd, gyd] = meshgrid(xg, xg);
[axg, ayg] = meshgrid(((1:NACT)-(NACT+1)/2)*PITCH);
dmap = @(act) dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', act);
PARb = [1 2 1 1];  sgn = +1;                 % S2 registration, this deck
POKE = 20e-6;  AMPM = 10e-6;  AMPG = 1e-6;
ic = NACT/2;  Aa = zeros(NACT);  Aa(ic,ic) = 1;  Ma = dmap(POKE*Aa);
Ah = zeros(NACT);  Ah(hij(1),hij(2)) = 1;
[ii, jj] = meshgrid((0.5:NACT)/NACT);
nm_modes = size(PQ,1);
gk = zeros(nm_modes, K);
NR = 5;
RAWL = cell(NR, K);  RAWS = cell(NR, K);
col = struct('nm',{},'phi',{},'absc',{},'dia_lamd',{},'dimple_px',{}, ...
             'nmsk',{},'bx',{},'by',{},'kpk',{},'tmin',{});

for k = 1:K
    tk = tic;
    LAM = LAMS_NM(k)*1e-6;  nk = n_fs(LAMS_NM(k)*1e-3);
    PHI_M = 2*pi*(nk-1)*ETCH_MM/LAM;
    PHIS  = PHIS0 * (nk-1)/(n0-1) * (LAM0/LAM);   % fixed glass, three depths
    DIA_LAMD = DIA_LAMD0 * LAM0/LAM;              % fixed PHYSICAL dimple
    macos.load_rx(decks{k});
    wl = macos.get_src_wvl();
    assert(abs(wl - LAM) < 1e-9*LAM, 'deck %s did not take Wavelen (%g)', decks{k}, wl);
    ZW = dmg_zwfs_gauge(iTO, iMASK, iDET, struct('LAM',LAM, 'F2',F2, ...
        'R_BEAM',R_BEAM, 'DIA_LAMD',DIA_LAMD, 'PHI_M',PHI_M, 'PHIS',PHIS, ...
        'S_CONV',S_CONV));
    N_WF = ZW.N_WF;  msk = ZW.msk;
    dimple_px = ZW.dia_mm*1e-3 / abs(macos.dx_at(iMASK));
    if k == 1
        [mag, dxd_mm] = dmg_frame(iTO, iDET);
        lit = dmg_lit(msk, dxd_mm, mag, axg, ayg);
        rng(7);   Ab30 = zeros(NACT);  Ab30(lit) = 30e-6*randn(nnz(lit),1);
        rng(23);  Arnd = zeros(NACT);  Arnd(lit) = 10e-6*randn(nnz(lit),1);
        Pg = zeros(NACT);  Pg(8:8:NACT, 8:8:NACT) = 1;  Pg = Pg .* lit;
        ROWS = {'flat/hold20',      zeros(NACT), POKE*Ah; ...
                'flat/rand10',      zeros(NACT), Arnd; ...
                'rand30/single10',  Ab30,        10e-6*Ah; ...
                'rand30/grid47@1nm', Ab30,       AMPG*Pg; ...
                'rand30/rand10',    Ab30,        Arnd};
        dmg_say(rep, 'lit actuators %d; hold-out (%d,%d); grid sites %d\n', ...
            nnz(lit), hij(1), hij(2), nnz(Pg));
        dmg_say(rep, '\n%5s | %7s %6s %8s %9s %7s | %8s %8s %9s\n', 'nm', 'phi_m', '|c|', ...
            'dia_l/D', 'dimplePx', 'msk_px', 'anchor_x', 'anchor_y', 'kern_pk');
    end
    R = struct('P',PARb, 'tax',0, 'tay',0, 'bx',0, 'by',0, ...
        'dxd_mm',dxd_mm, 'mag',mag, 'msk',msk, 'N_WF',N_WF, ...
        'gxd',gxd, 'gyd',gyd);
    hA = ZW.measL(Ma);
    [R.bx, R.by, R.tax, R.tay] = dmg_anchor(hA, Ma, msk, N_WF, xg);
    hAd = sgn*dmg_samp(hA, R);  hAd(isnan(hAd)) = 0;
    stn = dmg_stencil(hAd, xg, R.tax, R.tay, PITCH, 6) / POKE;
    est = @(h) dmg_act_fit(sgn*dmg_samp(h, R), xg, axg, ayg, stn, lit);
    dmg_say(rep, '%5g | %7.4f %6.3f %8.3f %9.2f %7d | %8.2f %8.2f %9.4f\n', ...
        LAMS_NM(k), PHI_M, abs(exp(1i*PHI_M)-1), DIA_LAMD, dimple_px, nnz(msk), ...
        R.bx, R.by, max(stn(:)));
    % modal transfer through this color's estimator
    for m = 1:nm_modes
        p = PQ(m,1);  q = PQ(m,2);
        Ak = cos(pi*p*ii).*cos(pi*q*jj);
        aK = est(ZW.measL(dmap(AMPM*Ak)));
        gk(m,k) = (AMPM*Ak(lit)) \ aK(lit);
    end
    % rows: raw actuator maps, both readings, differential
    for r = 1:NR
        base = ROWS{r,2};  dev = ROWS{r,3};
        if r == 1 || ~isequal(base, ROWS{r-1,2})
            h0 = ZW.measL(dmap(base));  X0 = ZW.steppedX(dmap(base));
        end
        M1 = dmap(base + dev);
        RAWL{r,k} = est(ZW.measL(M1) - h0);
        RAWS{r,k} = est(ZW.stepdiff(ZW.steppedX(M1), X0));
    end
    col(end+1) = struct('nm',LAMS_NM(k), 'phi',PHI_M, 'absc',abs(exp(1i*PHI_M)-1), ...
        'dia_lamd',DIA_LAMD, 'dimple_px',dimple_px, 'nmsk',nnz(msk), ...
        'bx',R.bx, 'by',R.by, 'kpk',max(stn(:)), 'tmin',toc(tk)/60); %#ok<AGROW>
    fprintf('color %g nm done in %.1f min\n', LAMS_NM(k), toc(tk)/60);
end

% ---- modal transfer table + the combination's transfer ---------------
gk1 = gk(is1d,:);
gcell = @(ks) num2cell(gk1(:,ks), 1);
G1 = gk1.^2 ./ (gk1.^2 + BETA^2);                     % single-color Geff per (p,0) row
Gc = sum(gk1.^2, 2) ./ (sum(gk1.^2, 2) + BETA^2);     % K-color combination's Geff
dmg_say(rep, '\nmodal transfer (p,0) rows through each color''s estimator; Gcomb = the K-color combination''s transfer on that row\n');
dmg_say(rep, '%9s %7s |', 'mode', 'cyc/ap');
for k = 1:K, dmg_say(rep, ' %7g', LAMS_NM(k)); end
dmg_say(rep, ' | %7s %7s\n', 'G632.8', 'Gcomb');
for m = find(is1d).'
    p = PQ(m,1);  i1 = find(pk1 == p, 1);
    dmg_say(rep, '  (%2d, 0) %7.1f |', p, p/2);
    for k = 1:K, dmg_say(rep, ' %7.4f', gk(m,k)); end
    dmg_say(rep, ' | %7.4f %7.4f\n', G1(i1,1), Gc(i1));
end
for m = find(~is1d).'
    dmg_say(rep, '  (%2d,%2d) %7.1f |', PQ(m,1), PQ(m,2), hypot(PQ(m,1),PQ(m,2))/2);
    for k = 1:K, dmg_say(rep, ' %7.4f', gk(m,k)); end
    dmg_say(rep, ' | separability rows\n');
end
dmg_say(rep, 'worst region over the (p,0) rows -- min |g|:');
for k = 1:K, dmg_say(rep, ' %g:%.3f', LAMS_NM(k), min(abs(gk1(:,k)))); end
dmg_say(rep, '\n  min Geff (transfer after Wiener, beta %.2f):', BETA);
for k = 1:K, dmg_say(rep, ' %g:%.3f', LAMS_NM(k), min(G1(:,k))); end
dmg_say(rep, ' | K-comb %.3f\n', min(Gc));

% ---- rows: per color, mean, K-comb, best pair; both readings -----------
dmg_say(rep, '\nrows (differential, actuator space).  g = gain, e = rms error over lit (pm),\n');
dmg_say(rep, 'floor = rms of unpoked lit actuators (pm), SNR = mean(poked)/floor (poke-type rows).\n');
dmg_say(rep, 'columns: each single color (joint Wiener, that color''s transfer) | mean of the singles | K-color comb | best pair\n');
res = struct('row',{},'reading',{},'g',{},'e',{},'fl',{},'snr',{},'best_pair',{});
pairs = nchoosek(1:K, 2);
for rd = {'L', 'S'}
    rdn = rd{1};
    if strcmp(rdn, 'L'), RAW = RAWL; else, RAW = RAWS; end
    dmg_say(rep, '\n-- reading %s (%s) --\n', rdn, ifelse_(strcmp(rdn,'L'), 'frozen-reference LINEAR', 'phase-STEPPED'));
    for r = 1:NR
        Ad = ROWS{r,3};
        A = RAW(r, :);
        gs = zeros(1,K);  es = gs;  fls = gs;  sns = gs;
        singles = cell(1,K);
        for k = 1:K
            singles{k} = dmg_color_comb(A(k), 'separable', pk1, gcell(k), BETA, NACT);
            [gs(k), es(k), fls(k), sns(k)] = score_(singles{k}, Ad, lit);
        end
        amean = singles{1};  for k = 2:K, amean = amean + singles{k}; end
        amean = amean / K;
        [gm, em, flm, snm] = score_(amean, Ad, lit);
        acomb = dmg_color_comb(A, 'separable', pk1, gcell(1:K), BETA, NACT);
        [gc, ec, flc, snc] = score_(acomb, Ad, lit);
        % best pair by rms error (SNR for poke-type rows)
        bp = [NaN NaN];  bscore = -Inf;  bg = NaN;  be = NaN;  bfl = NaN;  bsn = NaN;
        for ip = 1:size(pairs,1)
            ks = pairs(ip,:);
            ap = dmg_color_comb(A(ks), 'separable', pk1, gcell(ks), BETA, NACT);
            [gp, ep, flp, snp] = score_(ap, Ad, lit);
            sc = ifelse_(isnan(snp), -ep, abs(snp));
            if sc > bscore, bscore = sc;  bp = LAMS_NM(ks);  bg = gp;  be = ep;  bfl = flp;  bsn = snp; end
        end
        dmg_say(rep, '%-18s g   :', ROWS{r,1});
        for k = 1:K, dmg_say(rep, ' %8.4f', gs(k)); end
        dmg_say(rep, ' | %8.4f %8.4f | %8.4f (%g+%g)\n', gm, gc, bg, bp(1), bp(2));
        dmg_say(rep, '%-18s e pm:', '');
        for k = 1:K, dmg_say(rep, ' %8.0f', es(k)); end
        dmg_say(rep, ' | %8.0f %8.0f | %8.0f\n', em, ec, be);
        if ~isnan(sns(1))
            dmg_say(rep, '%-18s flr :', '');
            for k = 1:K, dmg_say(rep, ' %8.0f', fls(k)); end
            dmg_say(rep, ' | %8.0f %8.0f | %8.0f\n', flm, flc, bfl);
            dmg_say(rep, '%-18s SNR :', '');
            for k = 1:K, dmg_say(rep, ' %8.2f', sns(k)); end
            dmg_say(rep, ' | %8.2f %8.2f | %8.2f\n', snm, snc, bsn);
        end
        res(end+1) = struct('row',ROWS{r,1}, 'reading',rdn, 'g',[gs gm gc bg], ...
            'e',[es em ec be], 'fl',[fls flm flc bfl], 'snr',[sns snm snc bsn], ...
            'best_pair',bp); %#ok<AGROW>
    end
end
% record tie-in: the S3/S4 per-axis Wiener at 633 on the flat hold-out (L)
a633 = dmg_modal_corr(RAWL{1,1}, 'separable', pk1, gk1(:,1), BETA, NACT);
[g_rec, e_rec] = score_(a633, ROWS{1,3}, lit);
dmg_say(rep, '\nrecord tie-in (632.8, flat/hold20, L, dmg_modal_corr per-axis form): g %.4f, e %.0f pm  [S3: 1.1814 / 212 pm on 12 modes]\n', g_rec, e_rec);
% DC-normalization variant of the combination (g1(0) = 1), grid row, S
[~, ~, ~, sn_unit] = score_(dmg_color_comb(RAWS(4,:), 'separable', pk1, gcell(1:K), BETA, NACT, 'unit'), ROWS{4,3}, lit);
[~, ~, ~, sn_rec]  = score_(dmg_color_comb(RAWS(4,:), 'separable', pk1, gcell(1:K), BETA, NACT, 'record'), ROWS{4,3}, lit);
dmg_say(rep, 'DC-normalization check (grid row, S, K-comb): record g(0) SNR %.2f vs g(0)=1 SNR %.2f\n', sn_rec, sn_unit);
dmg_say(rep, 'S6 color complete in %.1f min\n', toc(t_all)/60);
fclose(rep);
out = struct('lams_nm',LAMS_NM, 'PQ',PQ, 'gk',gk, 'col',col, 'rows',{ROWS(:,1)}, ...
    'RAWL',{RAWL}, 'RAWS',{RAWS}, 'lit',lit, 'res',res, 'beta',BETA);
save('zwfs_s6color.mat', 'out');
fprintf('wrote zwfs_s6color_report.txt + zwfs_s6color.mat\n');
end

% ---- helpers ----------------------------------------------------------
function fn = color_deck_(src, nm)
% copy of an emitted deck with ONLY the header Wavelen= line changed
txt = fileread(src);
txt = regexprep(txt, 'Wavelen=\s*\S+', sprintf('Wavelen=  %.9E', nm*1e-6), 'once');
[p, b, e] = fileparts(src);
fn = fullfile(p, sprintf('%s_%gnm%s', b, nm, e));
fid = fopen(fn, 'w');  fwrite(fid, txt);  fclose(fid);
end

function [g, e, fl, snr] = score_(a, Ad, lit)
g = Ad(lit) \ a(lit);
e = sqrt(mean((a(lit) - Ad(lit)).^2))*1e9;
pk = (Ad ~= 0) & lit;  un = lit & ~pk;
if nnz(pk) < nnz(lit)/4
    fl = std(a(un))*1e9;  snr = mean(a(pk)) / max(std(a(un)), eps);
else
    fl = NaN;  snr = NaN;
end
end

function y = ifelse_(c, a, b)
if c, y = a; else, y = b; end
end
