function out = zwfs_sens()
%ZWFS_SENS  Sensitivity test (Dave 2026-09-04): how small a change can
%   the Zernike sensor detect, and with what accuracy?  Scenarios:
%   single poke and a GRID of pokes (every 8th actuator), against the
%   flat and against a ~30 nm rms working background; deviation
%   amplitude swept 10 nm -> 0.1 pm.  Differential protocol (measure
%   base, measure base+dev, difference the frames' reconstructions),
%   actuator-space estimator from S2 (measured kernel, Tikhonov
%   lattice deconvolution).  Objective: sensitivity to 1 pm or below.
%   Detection statistic: SNR = recovered amplitude at poked sites over
%   the rms of the unpoked lit actuators; detected = SNR >= 5.
%   The model is noiseless: the floor measured here is the sensor's
%   systematic + numerical floor -- photon/detector noise rides on top
%   (a later, budgeted stage).
%   Run:  cd <this dir>;  matlab -batch "zwfs_sens"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
t_all = tic;
rep = fopen('zwfs_sens_report.txt', 'w');

s = 96/56;  LAM = 6.328e-4;
MODEL = 1024;  NGRID = 193;  N_G = 384;  DX_G = 0.28;
NACT = 96;  PITCH = 1.0;  AOI = 7;  D_BS_TO = 700;  R_BEAM = s*30;
T_FL_F = 42.5325;  T_FL_Kc = -2.58764;  T_DMF = 39.7694;  T_TRIM = -1.2473;
MASK_TRIM = -5.582;
N_FS = 1.45702;  ETCH_MM = 346.2e-6;
PHI_M = 2*pi*(N_FS-1)*ETCH_MM/LAM;
DIA_LAMD = 2.0;  S_CONV = -1;
AMPS = [10e-6 1e-6 1e-7 1e-8 1e-9 1e-10];     % mm: 10n 1n 100p 10p 1p 0.1p

say_(rep, '=== ZWFS sensitivity: smallest detectable change (differential, actuator space) ===\n');
say_(rep, 'config: model %d, NGRID %d, spot %.1f; estimator = S2 measured kernel, lambda 0.05\n', ...
     MODEL, NGRID, DIA_LAMD);

% ---- build + flat references + registration + kernel (== S2) -------
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

E0 = macos.complex_field(iDET);  N_WF = size(E0,1);
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
Eb = macos.complex_field(iDET, 'reset_trace', false);
macos.intensity(iMASK);  macos.apodize_complex(iMASK, V);
I_flat = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
Kmap = cc*Eb.*conj(E0);  den = 2*imag(Kmap);
I0 = abs(E0).^2;  supp = I0 > 0.1*max(I0(:));
msk = supp & (abs(den) > 0.05*max(abs(den(:))));

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

POKE = 20e-6;
Aa = zeros(NACT);  Aa(48,48) = 1;
Ma = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', POKE*Aa);
hA = meas_(Ma, iTO, iMASK, iDET, V, I_flat, den, msk, S_CONV, LAM, N_WF);
xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
wA = abs(hA);  wA(~msk) = 0;  wA(wA < 0.5*max(wA(:))) = 0;
[cg, rg] = meshgrid(1:N_WF, 1:N_WF);
bx = sum(cg(:).*wA(:))/sum(wA(:));  by = sum(rg(:).*wA(:))/sum(wA(:));
[~, im] = max(abs(Ma(:)));  [tr, tc] = ind2sub(size(Ma), im);
tax = xg(tc);  tay = xg(tr);
[gxd, gyd] = meshgrid(xg, xg);
PARb = [1 2 1 1];  sgn = +1;                  % S2's registration (this deck)
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

% ---- scenarios ------------------------------------------------------
Psng = zeros(NACT);  Psng(60,40) = 1;
Pgrd = zeros(NACT);  Pgrd(8:8:96, 8:8:96) = 1;  Pgrd = Pgrd .* lit;
rng(7);
Abase = zeros(NACT);  Abase(lit) = 30e-6*randn(nnz(lit),1);
bases = {zeros(NACT), 'flat'; Abase, 'rand30nm'};
scens = {Psng, 'single'; Pgrd, sprintf('grid(%d)', nnz(Pgrd))};

say_(rep, '%-9s %-10s %10s | %8s %10s %9s %9s\n', ...
     'base', 'scenario', 'amp', 'gain', 'acc_pm', 'floor_pm', 'SNR');
res = struct('base',{},'scen',{},'amp',{},'g',{},'acc',{},'flr',{},'snr',{});
for ib = 1:2
    Mb0 = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', bases{ib,1});
    h0 = meas_(Mb0, iTO, iMASK, iDET, V, I_flat, den, msk, S_CONV, LAM, N_WF);
    a0 = est(h0);
    for isc = 1:2
        P = scens{isc,1};  pk = P > 0;  un = lit & ~pk;
        for amp = AMPS
            M1 = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, ...
                                  'act', bases{ib,1} + amp*P);
            h1 = meas_(M1, iTO, iMASK, iDET, V, I_flat, den, msk, S_CONV, LAM, N_WF);
            aD = est(h1) - a0;                 % differential, actuator space
            g   = mean(aD(pk)) / amp;
            acc = (mean(aD(pk)) - amp) * 1e9;  % pm
            flr = std(aD(un)) * 1e9;           % pm, the unpoked floor
            snr = mean(aD(pk)) / max(std(aD(un)), eps);
            say_(rep, '%-9s %-10s %10.4g | %8.4f %+10.3g %9.3g %9.3g\n', ...
                 bases{ib,2}, scens{isc,2}, amp*1e6, g, acc, flr, snr);
            res(end+1) = struct('base',bases{ib,2}, 'scen',scens{isc,2}, ...
                'amp',amp, 'g',g, 'acc',acc, 'flr',flr, 'snr',snr);  %#ok<AGROW>
        end
    end
end
say_(rep, '(amp in nm; acc = recovered-minus-true at poked sites; floor = rms of unpoked\n');
say_(rep, ' lit actuators; SNR = poked/floor, detected at SNR >= 5; noiseless model)\n');
say_(rep, 'sensitivity run in %.1f min\n', toc(t_all)/60);
fclose(rep);
save('zwfs_sens.mat', 'res');
fprintf('wrote zwfs_sens_report.txt + zwfs_sens.mat\n');
out = res;
end

function h = meas_(M, iTO, iMASK, iDET, V, I_flat, den, msk, S_CONV, LAM, N_WF)
    macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
    macos.intensity(iMASK);
    macos.apodize_complex(iMASK, V);
    Ia = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
    phi = zeros(N_WF);
    phi(msk) = (Ia(msk) - I_flat(msk)) ./ den(msk);
    h = S_CONV*phi*LAM/(4*pi);
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
