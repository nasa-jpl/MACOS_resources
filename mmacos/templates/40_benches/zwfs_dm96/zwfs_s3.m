function out = zwfs_s3()
%ZWFS_S3  Stage 3: the MODAL-calibrated battery (Dave "Go!" x2).
%   For each DM size (96x96 at 1 mm pitch; 48x48 at 2 mm -- the
%   DST-class DM on the same 96 mm bench):
%     1. registration (two pokes) + measured response kernel (S2);
%     2. the 12-mode lattice-cosine transfer THROUGH the actuator-space
%        estimator = the modal calibration;
%     3. Wiener modal correction (radial transfer, beta = 0.1) applied
%        on the actuator lattice by FFT;
%     4. battery rows in pm: null, piston 20 nm, held-out single poke,
%        held-out random 10 nm (raw vs modal-corrected -- the dense-
%        command fix the lambda scan could not provide);
%     5. THE RESCUE: grid pokes on the 30 nm base -- the sensitivity
%        stage's one undetected scenario (SNR 1.64 with the frozen
%        linear reading) -- re-measured with the PHASE-STEPPED
%        retrieval (S2b) + modal correction.
%   Frozen-linear reconstructor for the battery (small differentials);
%   stepped retrieval where range/base-crosstalk demands it -- the
%   complementary-reconstructor doctrine from S2b.
%   Run:  cd <this dir>;  matlab -batch "zwfs_s3"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
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

say_(rep, '=== ZWFS S3: modal-calibrated battery (frozen-linear + stepped rescue) ===\n');

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

% flat references + masks (== S2/S2b)
E0f = macos.complex_field(iDET);  N_WF = size(E0f,1);
dx_mask_m = abs(macos.dx_at(iMASK));
lamD_mm = LAM*(s*250)/(2*R_BEAM);  dia_mm = DIA_LAMD*lamD_mm;
If = abs(macos.complex_field(iMASK)).^2;
assert(max(If(:))/sum(If(:)) >= 1e-2, 'not focused');
[~, ipk] = max(If(:));  [pr, pc] = ind2sub(size(If), ipk);
w = 12;  rows = max(1,pr-w):min(N_WF,pr+w);  cols = max(1,pc-w):min(N_WF,pc+w);
Iw = If(rows, cols);
ctr = [sum(sum(Iw,1).*cols)/sum(Iw(:)), sum(sum(Iw,2).'.*rows)/sum(Iw(:))];
[V, D] = zwfs_mask(N_WF, dx_mask_m*1e3, dia_mm, PHI_M, ctr);
cc = exp(1i*PHI_M) - 1;
macos.intensity(iMASK);  macos.apodize_complex(iMASK, D);
Ebf = macos.complex_field(iDET, 'reset_trace', false);
b2cal = abs(Ebf).^2;
macos.intensity(iMASK);  macos.apodize_complex(iMASK, V);
I_flat = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
Kmap = cc*Ebf.*conj(E0f);  den = 2*imag(Kmap);
I0 = abs(E0f).^2;  supp = I0 > 0.1*max(I0(:));
msk = supp & (abs(den) > 0.05*max(abs(den(:))));
VK = cell(1,3);
for k = 1:3
    VK{k} = zwfs_mask(N_WF, dx_mask_m*1e3, dia_mm, PHIS(k), ctr);
end
M2 = [(2-2*cos(PHIS)).', 2*sin(PHIS).'];
M2i = pinv(M2);

% ray frame
s1t = macos.trace(iTO);   ito  = macos.get_ray_info(s1t.nRays);
s2t = macos.trace(iDET);  idet = macos.get_ray_info(s2t.nRays);
okr = ito.ok_trace(:) & ito.ok_pass(:) & idet.ok_trace(:) & idet.ok_pass(:);
psi1 = macos.get_elt_psi(iTO);  vpt1 = macos.get_elt_vpt(iTO);
u1 = macos.design.Bench.perp(psi1);  v1 = cross(psi1, u1);
xy_to = [u1.'; v1.'] * (ito.pos - vpt1);
psi2 = macos.get_elt_psi(iDET);
u2 = macos.design.Bench.perp(psi2);  v2 = cross(psi2, u2);
xy_d = [u2.'; v2.'] * (idet.pos - idet.pos(:,1));
Aaf = [xy_d(:,okr).' ones(nnz(okr),1)] \ xy_to(:,okr).';
mag = sqrt(abs(det(Aaf(1:2,:).')));
dxd_mm = abs(macos.dx_at(iDET, 'mm'));
xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
[gxd, gyd] = meshgrid(xg, xg);
PARb = [1 2 1 1];  sgn = +1;                  % S2 registration, this deck

% (p,0) probes measure the 1-D transfer (the kernel is SEPARABLE --
% the radial assumption measured anisotropy on the first run: (48,0)
% 0.62 vs (32,32) 0.74); (p,p) rows validate separability.  p = NACT
% is degenerate: cos(pi*N*(i-.5)/N) == 0, a zero command.
CFG = struct('nact', {96, 48}, 'pitch', {1.0, 2.0}, ...
    'pq', {[1 0;2 0;4 0;8 0;16 0;24 0;32 0;48 0;64 0;80 0;8 8;24 24], ...
           [1 0;2 0;4 0;8 0;12 0;16 0;24 0;32 0;40 0;8 8;16 16]});

out = struct();
for icfg = 1:2
    NACT = CFG(icfg).nact;  PITCH = CFG(icfg).pitch;  PQ = CFG(icfg).pq;
    say_(rep, '\n---- DM %dx%d, pitch %.1f mm ----\n', NACT, NACT, PITCH);
    dmap = @(act) dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', act);
    measL = @(M) measL_(M, iTO, iMASK, iDET, V, I_flat, den, msk, S_CONV, LAM, N_WF);

    % kernel from a center poke (fresh per lattice)
    ic = NACT/2;  POKE = 20e-6;
    Aa = zeros(NACT);  Aa(ic,ic) = 1;
    Ma = dmap(POKE*Aa);
    hA = measL(Ma);
    [~, im] = max(abs(Ma(:)));  [tr, tc] = ind2sub(size(Ma), im);
    tax = xg(tc);  tay = xg(tr);
    wA = abs(hA);  wA(~msk) = 0;  wA(wA < 0.5*max(wA(:))) = 0;
    [cgp, rgp] = meshgrid(1:N_WF, 1:N_WF);
    bx = sum(cgp(:).*wA(:))/sum(wA(:));  by = sum(rgp(:).*wA(:))/sum(wA(:));
    hAd = sgn*samp_(hA, PARb, gxd, gyd, tax, tay, bx, by, dxd_mm, mag, msk, N_WF);
    hAd(isnan(hAd)) = 0;
    HW = 6;
    [soff, toff] = meshgrid(-HW:HW, -HW:HW);
    stn = interp2(xg, xg.', hAd, tax + soff*PITCH, tay + toff*PITCH, 'linear', 0) / POKE;
    [axg, ayg] = meshgrid(((1:NACT)-(NACT+1)/2)*PITCH);
    [syy, sxx] = find(msk);
    rpx = hypot(sxx-mean(sxx), syy-mean(syy));
    rs = sort(rpx);  R_ILL = rs(round(0.98*numel(rs))) * dxd_mm * mag;
    lit = hypot(axg, ayg) < 0.85*R_ILL;
    est = @(h) act_fit_(sgn*samp_(h, PARb, gxd, gyd, tax, tay, bx, by, ...
                                  dxd_mm, mag, msk, N_WF), xg, axg, ayg, stn, lit);

    % ---- modal transfer through the estimator ----------------------
    AMPM = 10e-6;
    [ii, jj] = meshgrid((0.5:NACT)/NACT);
    nm_modes = size(PQ,1);
    gk = zeros(nm_modes,1);  fk = zeros(nm_modes,1);
    say_(rep, 'modal transfer (through the actuator-space estimator):\n');
    for k = 1:nm_modes
        p = PQ(k,1);  q = PQ(k,2);
        Ak = cos(pi*p*ii).*cos(pi*q*jj);
        aK = est(measL(dmap(AMPM*Ak)));
        gk(k) = (AMPM*Ak(lit)) \ aK(lit);
        fk(k) = hypot(p, q)/2;
        say_(rep, '  mode(%2d,%2d)  %5.1f cyc/ap  gain %7.4f\n', p, q, fk(k), gk(k));
    end
    % separable Wiener from the (p,0) rows; diagonals validate
    is1d = PQ(:,2) == 0;
    pk1 = PQ(is1d,1);  gk1 = gk(is1d);
    for k = find(~is1d).'
        ppp = PQ(k,1);
        g1p = interp1([0; pk1], [gk1(1); gk1], ppp, 'linear');
        say_(rep, '  separability check (%d,%d): measured %.4f, g1(%d)^2 = %.4f\n', ...
             ppp, ppp, gk(k), ppp, g1p^2);
    end
    corr = @(a) modal_corr_(a, pk1, gk1, BETA, NACT);

    % ---- battery rows ----------------------------------------------
    a_null = est(measL(dmap(zeros(NACT))));
    say_(rep, 'null: %.2f pm rms (estimator on the flat state)\n', std(a_null(lit))*1e9);
    a_pist = est(measL(dmap(20e-6*ones(NACT))));
    say_(rep, 'piston 20 nm: mean gain %.4f (piston is INVISIBLE to a ZWFS -- expected 0; the IFO reads 0.98)\n', mean(a_pist(lit))/20e-6);

    ih = round(NACT*0.625);  jh = round(NACT*0.417);
    Ah = zeros(NACT);  Ah(ih,jh) = 1;
    aH = est(measL(dmap(POKE*Ah)));
    aHc = corr(aH);
    say_(rep, 'held-out poke (%d,%d) 20 nm: raw gain %.4f, modal-corrected %.4f, err %.1f pm\n', ...
         ih, jh, aH(ih,jh)/POKE, aHc(ih,jh)/POKE, ...
         sqrt(mean((aHc(lit) - POKE*Ah(lit)).^2))*1e9);

    rng(23);
    Ar = zeros(NACT);  Ar(lit) = 10e-6*randn(nnz(lit),1);
    aR = est(measL(dmap(Ar)));
    aRc = corr(aR);
    gr_raw = Ar(lit)\aR(lit);   er_raw = sqrt(mean((aR(lit)-Ar(lit)).^2))*1e9;
    gr_cor = Ar(lit)\aRc(lit);  er_cor = sqrt(mean((aRc(lit)-Ar(lit)).^2))*1e9;
    say_(rep, 'held-out random 10 nm: raw gain %.4f / %.0f pm; modal-corrected %.4f / %.0f pm\n', ...
         gr_raw, er_raw, gr_cor, er_cor);

    % ---- THE RESCUE: grid pokes on the 30 nm base, STEPPED ----------
    Pg = zeros(NACT);  Pg(8:8:NACT, 8:8:NACT) = 1;  Pg = Pg .* lit;
    rng(7);
    Ab30 = zeros(NACT);  Ab30(lit) = 30e-6*randn(nnz(lit),1);
    AMPG = 1e-6;                                % 1 nm grid pokes
    X0 = steppedX_(dmap(Ab30), iTO, iMASK, iDET, VK, M2i, b2cal, N_WF);
    X1 = steppedX_(dmap(Ab30 + AMPG*Pg), iTO, iMASK, iDET, VK, M2i, b2cal, N_WF);
    hstep = S_CONV * angle(X1 .* conj(X0)) * LAM/(4*pi);
    aS = act_fit_(sgn*samp_(hstep, PARb, gxd, gyd, tax, tay, bx, by, ...
                            dxd_mm, mag, msk, N_WF), xg, axg, ayg, stn, lit);
    aSc = corr(aS);
    pk = Pg > 0;  un = lit & ~pk;
    snr_st = mean(aSc(pk)) / max(std(aSc(un)), eps);
    % frozen-linear comparison on the same scenario
    hlin1 = measL(dmap(Ab30 + AMPG*Pg));
    hlin0 = measL(dmap(Ab30));
    aL = act_fit_(sgn*samp_(hlin1-hlin0, PARb, gxd, gyd, tax, tay, bx, by, ...
                            dxd_mm, mag, msk, N_WF), xg, axg, ayg, stn, lit);
    aLc = corr(aL);
    snr_ln = mean(aLc(pk)) / max(std(aLc(un)), eps);
    say_(rep, 'RESCUE grid(%d)-on-30nm-base at 1 nm: stepped SNR %.2f (linear %.2f; sens stage 1.64)\n', ...
         nnz(pk), snr_st, snr_ln);
    say_(rep, '  stepped: gain %.4f, floor %.3g pm\n', mean(aS(pk))/AMPG, std(aSc(un))*1e9);

    out.(sprintf('n%d', NACT)) = struct('gk',gk, 'fk',fk, 'snr_st',snr_st, ...
        'snr_ln',snr_ln, 'er_raw',er_raw, 'er_cor',er_cor);
end
say_(rep, 'S3 complete in %.1f min\n', toc(t_all)/60);
fclose(rep);
save('zwfs_s3.mat', 'out');
fprintf('wrote zwfs_s3_report.txt + zwfs_s3.mat\n');
end

% ---- frozen-linear measurement --------------------------------------
function h = measL_(M, iTO, iMASK, iDET, V, I_flat, den, msk, S_CONV, LAM, N_WF)
    macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
    macos.intensity(iMASK);
    macos.apodize_complex(iMASK, V);
    Ia = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
    phi = zeros(N_WF);
    phi(msk) = (Ia(msk) - I_flat(msk)) ./ den(msk);
    h = S_CONV*phi*LAM/(4*pi);
end

% ---- stepped retrieval (rank-2, calibrated b2) -----------------------
function X = steppedX_(M, iTO, iMASK, iDET, VK, M2i, b2, N_WF)
    macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
    I0m = abs(macos.complex_field(iDET)).^2;
    Ik = zeros(N_WF, N_WF, 3);
    for k = 1:3
        macos.intensity(iMASK);
        macos.apodize_complex(iMASK, VK{k});
        Ik(:,:,k) = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
    end
    d1 = Ik(:,:,1)-I0m;  d2 = Ik(:,:,2)-I0m;  d3 = Ik(:,:,3)-I0m;
    p = M2i(1,1)*d1 + M2i(1,2)*d2 + M2i(1,3)*d3;
    q = M2i(2,1)*d1 + M2i(2,2)*d2 + M2i(2,3)*d3;
    X = (b2 - p) + 1i*q;
end

% ---- separable Wiener modal correction on the actuator lattice ------
function ac = modal_corr_(a, pk1, gk1, beta, NACT)
    % 1-D transfer g1(p) from the (p,0) probes; G(fx,fy) ~ g1(fx) g1(fy)
    [pks, si] = sort(pk1(:));  gks = gk1(si);
    fu = min(0:NACT-1, NACT-(0:NACT-1)).';
    g1 = interp1([0; pks], [gks(1); gks], min(fu, max(pks)), 'linear');
    w1 = g1 ./ (g1.^2 + beta^2);
    W = w1 * w1.';
    ac = real(ifft2(fft2(a) .* W));
end

function hd = samp_(h, P, gxd, gyd, tax, tay, bx, by, dxd_mm, mag, msk, N_WF)
    off = {gxd - tax, gyd - tay};
    u = P(3)*off{P(1)}/(dxd_mm*mag) + bx;
    v = P(4)*off{P(2)}/(dxd_mm*mag) + by;
    hn = h;  hn(~msk) = NaN;
    hd = interp2(1:N_WF, (1:N_WF).', hn, u, v, 'linear', NaN);
end

function a = act_fit_(hd, xg, axg, ayg, stn, lit, lam)
    if nargin < 7, lam = 0.05; end
    hd(isnan(hd)) = 0;
    m = interp2(xg, xg.', hd, axg, ayg, 'linear', 0);
    m(~lit) = 0;
    Wf = double(lit);
    l2 = (lam*max(abs(stn(:))))^2;
    Cop = @(x) reshape(Wf.*conv2(reshape(x, size(axg)), stn, 'same'), [], 1);
    Ct  = @(x) reshape(conv2(Wf.*reshape(x, size(axg)), rot90(stn,2), 'same'), [], 1);
    Aop = @(x) Ct(Cop(x)) + l2*x;
    b = Ct(m(:));
    x = pcg(Aop, b, 1e-12, 400);
    a = reshape(x, size(axg));
end

function say_(rep, fmt, varargin)
    fprintf(fmt, varargin{:});
    fprintf(rep, fmt, varargin{:});
end
