function out = tg96_s6color()
%TG96_S6COLOR  Stage 6, IFO twin of zwfs_dm96/zwfs_s6color.m: MULTI-COLOR
%   combination (Dave 2026-09-08: "try running both systems at multiple
%   colors, maybe the combination will help with some poor SNR regions").
%   The gauge is measured at K wavelengths; each color gets its OWN
%   calibration (analyzer bases, null reference, anchor, modal transfer).
%   What is chromatic here: the tail's diffraction MTF (the transfer
%   roll-off 1.02 -> 0.58 by 57 cyc/ap that owns the dense-random
%   deficit, gain 0.77-0.83) and the height scaling 4 pi h / lam.  What
%   is NOT: the lenses (IndRef fixed, no dispersion), the polarizers
%   (ideal), and the QWPs -- macos.waveplate stores retardance at the
%   CURRENT wavelength, so each color's deck carries an ideal ACHROMATIC
%   quarter-wave (a chromatic-QWP trade is a separate stage).  Only the
%   deck header Wavelen= changes per color.  True (achromatic) influence
%   kernel for the estimator, as S3/S4.  Combination = multi-channel
%   Wiener (../dm_gauge_lib/dmg_color_comb, 'radial'), scored against
%   every single color, the plain mean, and the best pair.  Noiseless.
%   Run:  cd <this dir>;  matlab -batch "tg96_s6color; exit(0)"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
addpath(fullfile(exdir, '..', 'dm_gauge_lib'));
t_all = tic;
rep = fopen('tg96_s6color_report.txt', 'w');

s = 96/56;  LAM0 = 6.328e-4;  QWP = 0.25;  THETAS = [0 45 90 135];
MODEL = 1024;  NGRID = 385;  N_G = 384;  DX_G = 0.28;
AOI = 7;  D_BS_TO = 700;
BETA = 0.1;  NACT = 96;  PITCH = 1.0;
LAMS_NM = [632.8 480 532 700 780];           % 632.8 (the record) FIRST: lit + tie-in
K = numel(LAMS_NM);
PQ = [1 1;2 2;4 4;8 8;16 16;24 24;32 32;48 48;64 64;80 80;96 96;48 0];
hij = [60 40];

dmg_say(rep, '=== IFO S6 COLOR: multi-wavelength combination (96x96, actuator space, pm) ===\n');
dmg_say(rep, 'colors: %s nm; ideal achromatic QWPs (lam/4 at each color), dispersionless lenses;\n', num2str(LAMS_NM));
dmg_say(rep, 'model %d, NGRID %d; per-color calibration (bases, null, anchor, modal transfer); true kernel; lit + bases from 633; beta %.2f\n', ...
    MODEL, NGRID, BETA);

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
dT = cell(1,K);  dR = cell(1,K);
for k = 1:K
    dT{k} = color_deck_('tg96_test.in', LAMS_NM(k));
    dR{k} = color_deck_('tg96_ref.in',  LAMS_NM(k));
end

xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
[gxd, gyd] = meshgrid(xg, xg);
[axg, ayg] = meshgrid(((1:NACT)-(NACT+1)/2)*PITCH);
dmap = @(act) dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', act);
PARb = [1 2 1 1];  sgn = -1;                 % eprime registration, this deck
POKE = 20e-6;  AMPM = 10e-6;  AMPG = 1e-6;
ic = NACT/2;  Aa = zeros(NACT);  Aa(ic,ic) = 1;
Ma = dmap(150e-6*Aa);                        % strong anchor for translation
Mu = dmap(Aa);                               % unit influence kernel (achromatic)
Ah = zeros(NACT);  Ah(hij(1),hij(2)) = 1;
[ii, jj] = meshgrid((0.5:NACT)/NACT);
nm_modes = size(PQ,1);
gk = zeros(nm_modes, K);  fk = hypot(PQ(:,1), PQ(:,2))/2;
NR = 5;
RAW = cell(NR, K);
col = struct('nm',{},'nmsk',{},'bx',{},'by',{},'null_pm',{},'tmin',{});

for k = 1:K
    tk = tic;
    LAM = LAMS_NM(k)*1e-6;
    AT = dmg_arm_desc(dT{k}, G.bt, G.T, 0);
    AR = dmg_arm_desc(dR{k}, G.br, G.R, 45);
    IFO = dmg_ifo_gauge(AT, AR, QWP, THETAS, LAM);
    msk = IFO.msk;  measI = IFO.meas;
    wsafe = @(h1, h0) angle(exp(1i*(h1 - h0)*4*pi/LAM))*LAM/(4*pi);
    macos.load_rx(AT.rx);
    wl = macos.get_src_wvl();
    assert(abs(wl - LAM) < 1e-9*LAM, 'deck %s did not take Wavelen (%g)', dT{k}, wl);
    N_WF = size(IFO.I0,1);
    if k == 1
        [mag, dxd_mm] = dmg_frame(AT.iTO, AT.iDET);
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
        dmg_say(rep, '\n%5s | %7s %8s %8s %9s\n', 'nm', 'msk_px', 'anchor_x', 'anchor_y', 'null_pm');
    end
    R = struct('P',PARb, 'tax',0, 'tay',0, 'bx',0, 'by',0, ...
        'dxd_mm',dxd_mm, 'mag',mag, 'msk',msk, 'N_WF',N_WF, ...
        'gxd',gxd, 'gyd',gyd);
    hA = measI(Ma);
    [R.bx, R.by, R.tax, R.tay] = dmg_anchor(hA, Ma, msk, N_WF, xg);
    stn = dmg_stencil(Mu, xg, R.tax, R.tay, PITCH, 6);
    est = @(h) dmg_act_fit(sgn*dmg_samp(h, R), xg, axg, ayg, stn, lit);
    % static null at this color (the tail is tuned at 633; common-mode):
    % the raw null-phase map, as tg96 run 10 quotes it.  (The 2026-09-08
    % run printed 0.0 here -- it measured meas(flat), which is referenced
    % to p_null and is zero by construction; fixed after that run.)
    hnull = IFO.p_null * LAM/(4*pi);
    null_pm = std(hnull(msk) - median(hnull(msk)))*1e9;
    dmg_say(rep, '%5g | %7d %8.2f %8.2f %9.1f\n', LAMS_NM(k), nnz(msk), R.bx, R.by, null_pm);
    % modal transfer through this color's estimator
    for m = 1:nm_modes
        p = PQ(m,1);  q = PQ(m,2);
        Ak = cos(pi*p*ii).*cos(pi*q*jj);
        aK = est(measI(dmap(AMPM*Ak)));
        gk(m,k) = (AMPM*Ak(lit)) \ aK(lit);
    end
    % rows: raw actuator maps, wrap-safe differential
    for r = 1:NR
        base = ROWS{r,2};  dev = ROWS{r,3};
        if r == 1 || ~isequal(base, ROWS{r-1,2})
            h0 = measI(dmap(base));
        end
        h1 = measI(dmap(base + dev));
        RAW{r,k} = est(wsafe(h1, h0));
    end
    col(end+1) = struct('nm',LAMS_NM(k), 'nmsk',nnz(msk), 'bx',R.bx, 'by',R.by, ...
        'null_pm',null_pm, 'tmin',toc(tk)/60); %#ok<AGROW>
    fprintf('color %g nm done in %.1f min\n', LAMS_NM(k), toc(tk)/60);
end

% ---- machinery check: K=1 radial with the record DC form == dmg_modal_corr
a_rec = dmg_modal_corr(RAW{1,1}, 'radial', fk, gk(:,1), BETA, NACT);
a_k1  = dmg_color_comb(RAW(1,1), 'radial', fk, {gk(:,1)}, BETA, NACT, 'record');
dmg_say(rep, '\nmachinery: K=1 radial (record DC) vs dmg_modal_corr, max |diff| = %.3e pm\n', ...
    max(abs(a_rec(:) - a_k1(:)))*1e9);

% ---- modal transfer table + the combination's transfer ---------------
gcell = @(ks) num2cell(gk(:,ks), 1);
G1 = gk.^2 ./ (gk.^2 + BETA^2);
Gc = sum(gk.^2, 2) ./ (sum(gk.^2, 2) + BETA^2);
dmg_say(rep, '\nmodal transfer through each color''s estimator; Gcomb = the K-color combination''s transfer on that mode\n');
dmg_say(rep, '%9s %7s |', 'mode', 'cyc/ap');
for k = 1:K, dmg_say(rep, ' %7g', LAMS_NM(k)); end
dmg_say(rep, ' | %7s %7s\n', 'G632.8', 'Gcomb');
for m = 1:nm_modes
    dmg_say(rep, '  (%2d,%2d) %7.1f |', PQ(m,1), PQ(m,2), fk(m));
    for k = 1:K, dmg_say(rep, ' %7.4f', gk(m,k)); end
    dmg_say(rep, ' | %7.4f %7.4f\n', G1(m,1), Gc(m));
end
use = fk < max(fk);                          % (96,96) is the zero command
dmg_say(rep, 'worst region over the modes below the zero command -- min |g|:');
for k = 1:K, dmg_say(rep, ' %g:%.3f', LAMS_NM(k), min(abs(gk(use,k)))); end
dmg_say(rep, '\n  min Geff (transfer after Wiener, beta %.2f):', BETA);
for k = 1:K, dmg_say(rep, ' %g:%.3f', LAMS_NM(k), min(G1(use,k))); end
dmg_say(rep, ' | K-comb %.3f\n', min(Gc(use)));

% ---- rows: per color, mean, K-comb, best pair ---------------------------
dmg_say(rep, '\nrows (differential, actuator space).  g = gain, e = rms error over lit (pm),\n');
dmg_say(rep, 'floor = rms of unpoked lit actuators (pm), SNR = mean(poked)/floor (poke-type rows).\n');
dmg_say(rep, 'columns: each single color (joint Wiener, that color''s transfer) | mean of the singles | K-color comb | best pair\n\n');
res = struct('row',{},'g',{},'e',{},'fl',{},'snr',{},'best_pair',{});
pairs = nchoosek(1:K, 2);
for r = 1:NR
    Ad = ROWS{r,3};
    A = RAW(r, :);
    gs = zeros(1,K);  es = gs;  fls = gs;  sns = gs;
    singles = cell(1,K);
    for k = 1:K
        singles{k} = dmg_color_comb(A(k), 'radial', fk, gcell(k), BETA, NACT);
        [gs(k), es(k), fls(k), sns(k)] = score_(singles{k}, Ad, lit);
    end
    amean = singles{1};  for k = 2:K, amean = amean + singles{k}; end
    amean = amean / K;
    [gm, em, flm, snm] = score_(amean, Ad, lit);
    acomb = dmg_color_comb(A, 'radial', fk, gcell(1:K), BETA, NACT);
    [gc, ec, flc, snc] = score_(acomb, Ad, lit);
    bp = [NaN NaN];  bscore = -Inf;  bg = NaN;  be = NaN;  bfl = NaN;  bsn = NaN;
    for ip = 1:size(pairs,1)
        ks = pairs(ip,:);
        ap = dmg_color_comb(A(ks), 'radial', fk, gcell(ks), BETA, NACT);
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
    res(end+1) = struct('row',ROWS{r,1}, 'g',[gs gm gc bg], 'e',[es em ec be], ...
        'fl',[fls flm flc bfl], 'snr',[sns snm snc bsn], 'best_pair',bp); %#ok<AGROW>
end
[g_rec, e_rec] = score_(a_rec, ROWS{1,3}, lit);
dmg_say(rep, '\nrecord tie-in (632.8, flat/hold20, dmg_modal_corr): g %.4f, e %.0f pm  [S3: 0.9171 / 92 pm]\n', g_rec, e_rec);
dmg_say(rep, 'S6 color complete in %.1f min\n', toc(t_all)/60);
fclose(rep);
out = struct('lams_nm',LAMS_NM, 'PQ',PQ, 'fk',fk, 'gk',gk, 'col',col, 'rows',{ROWS(:,1)}, ...
    'RAW',{RAW}, 'lit',lit, 'res',res, 'beta',BETA);
save('tg96_s6color.mat', 'out');
fprintf('wrote tg96_s6color_report.txt + tg96_s6color.mat\n');
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
