function out = tg96_eprime()
%TG96_EPRIME  Stage E': the IFO differential benchmark restated in
%   ACTUATOR space (Dave's scoring ruling), errors in pm (ultimate
%   target 1 pm).  Option (b) of the ruling: fit the DM influence
%   model to the measured surface map (lattice deconvolution of the
%   TRUE influence kernel, pcg); the residual is the instrument error
%   expressed in actuator commands.  The run-10 map-space Stage E
%   numbers stand as the map-space record; THESE are the head-to-head
%   currency.
%   Rows (== run-10 Stage E): bases {flat, random 30 nm rms} x devs
%   {single actuator 10 nm, random 10 nm rms}; measure base, measure
%   base+dev, difference the measurements, actuator-fit the
%   difference, score against the commanded deviation over the lit
%   actuators.  Registration re-derived by the two-poke doctrine
%   (self-contained; run 10 found parity 5, sign -1).
%   Run:  cd <this dir>;  matlab -batch "tg96_eprime"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
t_all = tic;
rep = fopen('tg96_eprime_report.txt', 'w');

s = 96/56;  LAM = 6.328e-4;  QWP = 0.25;  THETAS = [0 45 90 135];
MODEL = 1024;  NGRID = 385;  N_G = 384;  DX_G = 0.28;
NACT = 96;  PITCH = 1.0;  AOI = 7;  D_BS_TO = 700;
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

say_(rep, '=== TG96 Stage E'': differential benchmark in ACTUATOR space (pm) ===\n');

% ---- bases + ray frame ---------------------------------------------
Sr = analyzer_basis(AR, QWP, []);
S0 = analyzer_basis(AT, QWP, []);
I0 = frame(S0, Sr, 0);  msk = I0 > 0.1*max(I0(:));
p_null = fourstep(S0, Sr, THETAS);

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
say_(rep, 'ray affine: mag %.4f DM-mm/det-mm\n', mag);

% ---- registration by two pokes (150 nm -- IFO regime) --------------
PK = 150e-6;
Aa = zeros(NACT);  Aa(48,48) = 1;
Ab = zeros(NACT);  Ab(30,64) = 1;
Ma = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', PK*Aa);
Mb = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', PK*Ab);
hA = meas_surface(AT, QWP, Ma, Sr, p_null, THETAS, LAM);
hB = meas_surface(AT, QWP, Mb, Sr, p_null, THETAS, LAM);
xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
wA = abs(hA);  wA(~msk) = 0;  wA(wA < 0.5*max(wA(:))) = 0;
[cg, rg] = meshgrid(1:N_WF, 1:N_WF);
bx = sum(cg(:).*wA(:))/sum(wA(:));  by = sum(rg(:).*wA(:))/sum(wA(:));
[~, im] = max(abs(Ma(:)));  [tr, tc] = ind2sub(size(Ma), im);
tax = xg(tc);  tay = xg(tr);
[gxd, gyd] = meshgrid(xg, xg);
PAR = {[1 2 1 1],[1 2 -1 1],[1 2 1 -1],[1 2 -1 -1], ...
       [2 1 1 1],[2 1 -1 1],[2 1 1 -1],[2 1 -1 -1]};
ccs = zeros(1,8);
for p = 1:8
    hBd = samp_(hB, PAR{p}, gxd, gyd, tax, tay, bx, by, dxd_mm, mag, msk, N_WF);
    ok = ~isnan(hBd) & (abs(Mb) > 0);
    if nnz(ok) < 50, ccs(p) = 0; continue; end
    cm = corrcoef(hBd(ok), Mb(ok));  c = cm(1,2);
    if isnan(c), c = 0; end   % candidate maps the truth region off-support
    ccs(p) = c;
end
[~, pbest] = max(abs(ccs));  sgn = sign(ccs(pbest));
srt = sort(abs(ccs), 'descend');
say_(rep, 'registration: parity %d, sign %+d, |corr| %.4f (runner-up %.4f)\n', ...
     pbest, sgn, srt(1), srt(2));
assert(srt(1) >= 0.8 && srt(1)-srt(2) >= 0.3, 'registration gate FAIL');

% ---- actuator fit: TRUE influence kernel stencil -------------------
Mu = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', Aa);  % unit
HW = 6;
[soff, toff] = meshgrid(-HW:HW, -HW:HW);
stn = interp2(xg, xg.', Mu, tax + soff*PITCH, tay + toff*PITCH, 'linear', 0);
[axg, ayg] = meshgrid(((1:NACT)-(NACT+1)/2)*PITCH);
[syy, sxx] = find(msk);
rpx = hypot(sxx-mean(sxx), syy-mean(syy));
rs = sort(rpx);  R_ILL = rs(round(0.98*numel(rs))) * dxd_mm * mag;
lit = hypot(axg, ayg) < 0.85*R_ILL;
say_(rep, 'true-kernel stencil; %d lit actuators (r < 0.85 x %.1f mm)\n', nnz(lit), R_ILL);
est = @(h) act_fit_(sgn*samp_(h, PAR{pbest}, gxd, gyd, tax, tay, bx, by, ...
                              dxd_mm, mag, msk, N_WF), xg, axg, ayg, stn, lit);

% ---- Stage E' rows -------------------------------------------------
rng(7);
Abase = zeros(NACT);  Abase(lit) = 30e-6*randn(nnz(lit),1);
Mbase = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', Abase);
Asng = zeros(NACT);  Asng(48,48) = 10e-6;
rng(23);
Arnd = zeros(NACT);  Arnd(lit) = 10e-6*randn(nnz(lit),1);

bases = {zeros(NACT), 'flat'; Abase, 'random 30 nm'};
devs  = {Asng, 'single act 10 nm'; Arnd, 'random 10 nm rms'};
say_(rep, '%-14s %-18s %8s %10s %10s\n', 'base', 'deviation', 'gain', 'resid_pm', 'raw_pm');
res = struct('base',{},'dev',{},'g',{},'e_pm',{},'eraw_pm',{});
for ib = 1:2
    Mb0 = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', bases{ib,1});
    h0 = meas_surface(AT, QWP, Mb0, Sr, p_null, THETAS, LAM);
    for id = 1:2
        Ad = devs{id,1};
        M1 = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, ...
                              'act', bases{ib,1} + Ad);
        h1 = meas_surface(AT, QWP, M1, Sr, p_null, THETAS, LAM);
        dh = angle(exp(1i*(h1 - h0)*4*pi/LAM))*LAM/(4*pi);   % wrap-safe diff
        aD = est(dh);
        g  = Ad(lit) \ aD(lit);
        e  = sqrt(mean((aD(lit) - g*Ad(lit)).^2)) * 1e9;
        er = sqrt(mean((aD(lit) - Ad(lit)).^2)) * 1e9;
        say_(rep, '%-14s %-18s %8.4f %10.1f %10.1f\n', bases{ib,2}, devs{id,2}, g, e, er);
        res(end+1) = struct('base',bases{ib,2}, 'dev',devs{id,2}, ...
                            'g',g, 'e_pm',e, 'eraw_pm',er);          %#ok<AGROW>
    end
end
say_(rep, '(map-space run-10 record: single 21 pm / random 3670 pm -- different currency)\n');
say_(rep, 'E'' complete in %.1f min\n', toc(t_all)/60);
fclose(rep);
out = res;
save('tg96_eprime.mat', 'res');
fprintf('wrote tg96_eprime_report.txt + tg96_eprime.mat\n');
end

% ==== shared machinery (samp_/act_fit_ as zwfs_s2; meas as tg96) =====
function hd = samp_(h, P, gxd, gyd, tax, tay, bx, by, dxd_mm, mag, msk, N_WF)
    off = {gxd - tax, gyd - tay};
    u = P(3)*off{P(1)}/(dxd_mm*mag) + bx;
    v = P(4)*off{P(2)}/(dxd_mm*mag) + by;
    hn = h;  hn(~msk) = NaN;
    hd = interp2(1:N_WF, (1:N_WF).', hn, u, v, 'linear', NaN);
end

function a = act_fit_(hd, xg, axg, ayg, stn, lit, lam)
    % Lattice deconvolution of the response stencil, TIKHONOV-regularized:
    % the unregularized inverse amplifies modes where the kernel's
    % transfer is small (S2 G4 measured 13 nm on a dense random command
    % -- the raw-pinv-diverges lesson).  LAM is relative to the stencil
    % peak; 0.05 default, scan printed where it matters.
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
    x = pcg(Aop, b, 1e-10, 300);
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
