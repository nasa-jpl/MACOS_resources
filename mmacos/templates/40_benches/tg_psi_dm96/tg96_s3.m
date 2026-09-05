function out = tg96_s3()
%TG96_S3  Stage 3, IFO twin of zwfs_dm96/zwfs_s3.m: the modal-calibrated
%   battery for the interferometer.  Per DM size (96x96 at 1 mm pitch;
%   48x48 at 2 mm -- DST-class DM, same bench): the 12-mode lattice
%   transfer THROUGH the actuator-space estimator (true influence
%   kernel), the Wiener modal correction (beta 0.1), and the battery
%   rows in pm: null, piston, held-out poke, held-out random 10 nm
%   (raw vs modal-corrected).
%   Run:  cd <this dir>;  matlab -batch "tg96_s3"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
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
AT = arm_desc('tg96_test.in', G.bt, G.T, 0);
AR = arm_desc('tg96_ref.in',  G.br, G.R, 45);

say_(rep, '=== IFO S3: modal-calibrated battery ===\n');

Sr = analyzer_basis(AR, QWP, []);
S0 = analyzer_basis(AT, QWP, []);
I0 = frame(S0, Sr, 0);  msk = I0 > 0.1*max(I0(:));
p_null = fourstep(S0, Sr, THETAS);
measI = @(M) meas_surface(AT, QWP, M, Sr, p_null, THETAS, LAM);

macos.load_rx(AT.rx);
s1t = macos.trace(AT.iTO);   ito  = macos.get_ray_info(s1t.nRays);
s2t = macos.trace(AT.iDET);  idet = macos.get_ray_info(s2t.nRays);
okr = ito.ok_trace(:) & ito.ok_pass(:) & idet.ok_trace(:) & idet.ok_pass(:);
psi1 = macos.get_elt_psi(AT.iTO);  vpt1 = macos.get_elt_vpt(AT.iTO);
u1 = macos.design.Bench.perp(psi1);  v1 = cross(psi1, u1);
xy_to = [u1.'; v1.'] * (ito.pos - vpt1);
psi2 = macos.get_elt_psi(AT.iDET);
u2 = macos.design.Bench.perp(psi2);  v2 = cross(psi2, u2);
xy_d = [u2.'; v2.'] * (idet.pos - idet.pos(:,1));
Aaf = [xy_d(:,okr).' ones(nnz(okr),1)] \ xy_to(:,okr).';
mag = sqrt(abs(det(Aaf(1:2,:).')));
dxd_mm = abs(macos.dx_at(AT.iDET, 'mm'));
N_WF = size(I0,1);
xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
[gxd, gyd] = meshgrid(xg, xg);
PARb = [1 2 1 1];  sgn = -1;                  % eprime registration

CFG = struct('nact', {96, 48}, 'pitch', {1.0, 2.0}, ...
    'pq', {[1 1;2 2;4 4;8 8;16 16;24 24;32 32;48 48;64 64;80 80;96 96;48 0], ...
           [1 1;2 2;4 4;8 8;12 12;16 16;24 24;32 32;40 40;48 48;24 0]});

out = struct();
for icfg = 1:2
    NACT = CFG(icfg).nact;  PITCH = CFG(icfg).pitch;  PQ = CFG(icfg).pq;
    say_(rep, '\n---- DM %dx%d, pitch %.1f mm ----\n', NACT, NACT, PITCH);
    dmap = @(act) dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', act);

    ic = NACT/2;  POKE = 20e-6;
    Aa = zeros(NACT);  Aa(ic,ic) = 1;
    Ma = dmap(150e-6*Aa);                      % strong anchor for translation
    hA = measI(Ma);
    [~, im] = max(abs(Ma(:)));  [tr, tc] = ind2sub(size(Ma), im);
    tax = xg(tc);  tay = xg(tr);
    wA = abs(hA);  wA(~msk) = 0;  wA(wA < 0.5*max(wA(:))) = 0;
    [cgp, rgp] = meshgrid(1:N_WF, 1:N_WF);
    bx = sum(cgp(:).*wA(:))/sum(wA(:));  by = sum(rgp(:).*wA(:))/sum(wA(:));
    Mu = dmap(Aa);                             % unit influence kernel
    HW = 6;
    [soff, toff] = meshgrid(-HW:HW, -HW:HW);
    stn = interp2(xg, xg.', Mu, tax + soff*PITCH, tay + toff*PITCH, 'linear', 0);
    [axg, ayg] = meshgrid(((1:NACT)-(NACT+1)/2)*PITCH);
    [syy, sxx] = find(msk);
    rpx = hypot(sxx-mean(sxx), syy-mean(syy));
    rs = sort(rpx);  R_ILL = rs(round(0.98*numel(rs))) * dxd_mm * mag;
    lit = hypot(axg, ayg) < 0.85*R_ILL;
    est = @(h) act_fit_(sgn*samp_(h, PARb, gxd, gyd, tax, tay, bx, by, ...
                                  dxd_mm, mag, msk, N_WF), xg, axg, ayg, stn, lit);

    AMPM = 10e-6;
    [ii, jj] = meshgrid((0.5:NACT)/NACT);
    nm_modes = size(PQ,1);
    gk = zeros(nm_modes,1);  fk = zeros(nm_modes,1);
    say_(rep, 'modal transfer (through the actuator-space estimator):\n');
    for k = 1:nm_modes
        p = PQ(k,1);  q = PQ(k,2);
        Ak = cos(pi*p*ii).*cos(pi*q*jj);
        aK = est(measI(dmap(AMPM*Ak)));
        gk(k) = (AMPM*Ak(lit)) \ aK(lit);
        fk(k) = hypot(p, q)/2;
        say_(rep, '  mode(%2d,%2d)  %5.1f cyc/ap  gain %7.4f\n', p, q, fk(k), gk(k));
    end
    corr = @(a) modal_corr_(a, fk, gk, BETA, NACT);

    a_null = est(measI(dmap(zeros(NACT))));
    say_(rep, 'null: %.2f pm rms (estimator on the flat state)\n', std(a_null(lit))*1e9);
    a_pist = est(measI(dmap(20e-6*ones(NACT))));
    say_(rep, 'piston 20 nm: mean gain %.4f\n', mean(a_pist(lit))/20e-6);

    ih = round(NACT*0.625);  jh = round(NACT*0.417);
    Ah = zeros(NACT);  Ah(ih,jh) = 1;
    aH = est(measI(dmap(POKE*Ah)));
    aHc = corr(aH);
    say_(rep, 'held-out poke (%d,%d) 20 nm: raw gain %.4f, modal-corrected %.4f, err %.1f pm\n', ...
         ih, jh, aH(ih,jh)/POKE, aHc(ih,jh)/POKE, ...
         sqrt(mean((aHc(lit) - POKE*Ah(lit)).^2))*1e9);

    rng(23);
    Ar = zeros(NACT);  Ar(lit) = 10e-6*randn(nnz(lit),1);
    aR = est(measI(dmap(Ar)));
    aRc = corr(aR);
    gr_raw = Ar(lit)\aR(lit);   er_raw = sqrt(mean((aR(lit)-Ar(lit)).^2))*1e9;
    gr_cor = Ar(lit)\aRc(lit);  er_cor = sqrt(mean((aRc(lit)-Ar(lit)).^2))*1e9;
    say_(rep, 'held-out random 10 nm: raw gain %.4f / %.0f pm; modal-corrected %.4f / %.0f pm\n', ...
         gr_raw, er_raw, gr_cor, er_cor);

    out.(sprintf('n%d', NACT)) = struct('gk',gk, 'fk',fk, ...
        'er_raw',er_raw, 'er_cor',er_cor);
end
say_(rep, 'S3 complete in %.1f min\n', toc(t_all)/60);
fclose(rep);
save('tg96_s3.mat', 'out');
fprintf('wrote tg96_s3_report.txt + tg96_s3.mat\n');
end

function ac = modal_corr_(a, fk, gk, beta, NACT)
    [uu, vv] = meshgrid(0:NACT-1);
    fu = min(uu, NACT-uu);  fv = min(vv, NACT-vv);
    fr = hypot(fu, fv)/2;
    [fks, si] = sort(fk(:));  gks = gk(si);
    Gf = interp1([0; fks], [gks(1); gks], min(fr, max(fks)), 'linear');
    W = Gf ./ (Gf.^2 + beta^2);
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

function A = arm_desc(rx, b, ix, base_deg)
    nm = {b.E.name};
    A = struct('rx', rx, 'b', b, 'iPol', find(strcmp(nm,'PolIn'),1), ...
        'iQ', find(contains(nm,'QWP') & ~strcmp(nm,'OutQWP')), ...
        'base', base_deg, 'qwp_deg', base_deg, 'oq_deg', 0, 'iTO', [], ...
        'iRC', ix.iRC, 'iOQ', ix.iOutQWP, 'iAn', ix.iAnalyzer, 'iDET', ix.iDET);
    if isfield(ix,'iTO'), A.iTO = ix.iTO; end
end

function a = lax(psi, deg)
    u1 = macos.design.Bench.perp(psi(:));  u2 = cross(psi(:), u1);
    a = cosd(deg)*u1 + sind(deg)*u2;  a = a(:).';
end

function load_arm(A, QWP, an_deg, grid)
    macos.load_rx(A.rx);  b = A.b;
    if nargin >= 4 && ~isempty(grid)
        macos.set_elt_grid(A.iTO, macos.get_elt_grid_spacing(A.iTO), grid);
    end
    macos.polarizer(A.iPol, 'axis', lax(b.E(A.iPol).psi, 45));
    qa = lax(b.E(A.iQ(1)).psi, A.qwp_deg);
    for j = 1:2, macos.waveplate(A.iQ(j), 'axis', qa, 'retardance', QWP); end
    macos.waveplate(A.iOQ, 'axis', lax(b.E(A.iOQ).psi, A.oq_deg), 'retardance', QWP);
    macos.polarizer(A.iAn, 'axis', lax(b.E(A.iAn).psi, an_deg));
    macos.polarization('on', 'Ex',[1/sqrt(2) 0], 'Ey',[1/sqrt(2) 0]);
    macos.vector_diffraction(true);
end

function E = arm_field(A, QWP, an_deg, grid)
    load_arm(A, QWP, an_deg, grid);
    E = cat(3, macos.complex_field(A.iDET,'plane',1), ...
               macos.complex_field(A.iDET,'plane',2), ...
               macos.complex_field(A.iDET,'plane',3));
end

function S = analyzer_basis(A, QWP, grid)
    E0  = arm_field(A, QWP,  0, grid);
    E45 = arm_field(A, QWP, 45, grid);
    E90 = arm_field(A, QWP, 90, grid);
    S = struct('A', E0, 'C', E90, 'B', 2*E45 - E0 - E90);
end

function E = synth(S, th)
    c = cosd(th);  s = sind(th);
    E = c^2*S.A + c*s*S.B + s^2*S.C;
end

function I = frame(Sx, Sr, th)
    I = sum(abs(synth(Sx,th) + synth(Sr,th)).^2, 3);
end

function p = fourstep(Sx, Sr, th)
    I1 = frame(Sx,Sr,th(1));  I2 = frame(Sx,Sr,th(2));
    I3 = frame(Sx,Sr,th(3));  I4 = frame(Sx,Sr,th(4));
    p  = atan2(I2-I4, I1-I3);
end

function h = meas_surface(A, QWP, M, Sr, p_null, THETAS, LAM)
    d = angle(exp(1i*(fourstep(analyzer_basis(A, QWP, M), Sr, THETAS) - p_null)));
    h = d * LAM/(4*pi);
end

function say_(rep, fmt, varargin)
    fprintf(fmt, varargin{:});
    fprintf(rep, fmt, varargin{:});
end
