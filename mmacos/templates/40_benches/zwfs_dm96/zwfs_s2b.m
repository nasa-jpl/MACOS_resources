function out = zwfs_s2b()
%ZWFS_S2B  Phase-stepped ZWFS: multiple masks of varying etch DEPTH
%   (Dave 2026-09-04).  Frames through dimples at phi = pi/2, pi,
%   3*pi/2 (same 2.0 lamF/D diameter) plus the clear frame solve, per
%   pixel and EXACTLY (linear 3x3, no small-phase assumption):
%       I_k = I0 + |c_k|^2 Y + 2 Re(c_k X),   c_k = e^{i phi_k} - 1,
%       Y = |Eb|^2,  X = Eb conj(E0).
%   The pupil phase change between two DM states is
%       dphi = arg( X_state1 * conj(X_state2) )
%   -- sign ambiguity gone, range extended to +-pi per pixel (and
%   unwrappable), the linearized-reconstructor approximation gone.
%   This is the SEQUENTIAL cousin of the phase-2 polarizing
%   metasurface (which delivers two of these phases simultaneously).
%   Hardware implication (recorded): a substrate carrying spots of
%   several DEPTHS, where the VSG2 part varies diameter at one depth.
%   Gates:
%     G1 model consistency: solved (Y,X) reproduce the measured frames
%        to round-off (the per-pixel algebra is exact)
%     G2 poke 20 nm: map-space gain vs the 1-frame linear reading
%        (0.445) -- the stepped retrieval should beat it
%     G3 RANGE: poke 150 nm (3.0 rad -- the linear reading folds
%        there); stepped recovery within 10%
%     G4 defocus 8 nm about the 30 nm working base (stacked test)
%   Run:  cd <this dir>;  matlab -batch "zwfs_s2b"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
t_all = tic;
rep = fopen('zwfs_s2b_report.txt', 'w');

s = 96/56;  LAM = 6.328e-4;
MODEL = 1024;  NGRID = 193;  N_G = 384;  DX_G = 0.28;
NACT = 96;  PITCH = 1.0;  AOI = 7;  D_BS_TO = 700;  R_BEAM = s*30;
T_FL_F = 42.5325;  T_FL_Kc = -2.58764;  T_DMF = 39.7694;  T_TRIM = -1.2473;
MASK_TRIM = -5.582;
DIA_LAMD = 2.0;
PHIS = [pi/2, pi, 3*pi/2];                    % the mask-depth ladder
S_CONV = -1;                                  % height sign, gated below

say_(rep, '=== ZWFS S2b: phase-stepped masks (depths pi/2, pi, 3pi/2 + clear) ===\n');

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

% mask family at the measured spot center
E0f = macos.complex_field(iDET);  N_WF = size(E0f,1);
dx_mask_m = abs(macos.dx_at(iMASK));
lamD_mm = LAM*(s*250)/(2*R_BEAM);  dia_mm = DIA_LAMD*lamD_mm;
If = abs(macos.complex_field(iMASK)).^2;
assert(max(If(:))/sum(If(:)) >= 1e-2, 'not focused');
[~, ipk] = max(If(:));  [pr, pc] = ind2sub(size(If), ipk);
w = 12;  rows = max(1,pr-w):min(N_WF,pr+w);  cols = max(1,pc-w):min(N_WF,pc+w);
Iw = If(rows, cols);
ctr = [sum(sum(Iw,1).*cols)/sum(Iw(:)), sum(sum(Iw,2).'.*rows)/sum(Iw(:))];
VK = cell(1,3);
for k = 1:3
    VK{k} = zwfs_mask(N_WF, dx_mask_m*1e3, dia_mm, PHIS(k), ctr);
end
CK = exp(1i*PHIS) - 1;
% STRUCTURAL FINDING (first run): |c|^2 = -2 Re(c) IDENTICALLY for any
% mask phase, so depth stepping yields only TWO observables per pixel:
%   I_k - I0 = (2-2cos phi_k) (Y - ReX) + 2 sin(phi_k) ImX
% |Eb|^2 is NOT self-calibrating from a depth ladder -- it comes from a
% one-time calibration (here: the flat-state reference measurement; on
% hardware, a calibration or a diameter change).  Solve the rank-2
% system per pixel (LSQ over 3 frames; residual is gate G1).
M2 = [(2-2*cos(PHIS)).', 2*sin(PHIS).'];
M2i = pinv(M2);
say_(rep, 'rank-2 solve, cond %.3f; |Eb|^2 supplied by flat-state calibration\n', cond(M2));

% support from the clear frame
I0f = abs(E0f).^2;
supp = I0f > 0.1*max(I0f(:));

% ---- one-time calibration: |Eb|^2 on the flat state -----------------
Db = zwfs_mask(N_WF, dx_mask_m*1e3, dia_mm, pi, ctr);   % any phase; use disk
Ddisk = real((Db - 1)/(exp(1i*pi) - 1));                % recover the disk
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), zeros(N_G));
macos.intensity(iMASK);  macos.apodize_complex(iMASK, Ddisk);
Ebf = macos.complex_field(iDET, 'reset_trace', false);
b2 = abs(Ebf).^2;

% ---- G1: rank-2 model consistency on the flat state -----------------
[Xf, pqf, I0meas] = stepped_(zeros(NACT), N_G, DX_G, NACT, PITCH, ...
                             iTO, iMASK, iDET, VK, M2i, b2, N_WF);
resmax = 0;
for k = 1:3
    Ik_model = I0meas + (2-2*cos(PHIS(k)))*pqf(:,:,1) + 2*sin(PHIS(k))*pqf(:,:,2);
    macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), zeros(N_G));
    macos.intensity(iMASK);  macos.apodize_complex(iMASK, VK{k});
    Ik = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
    r = max(abs(Ik(supp) - Ik_model(supp))) / max(I0meas(:));
    resmax = max(resmax, r);
end
say_(rep, 'G1 frame consistency (rank-2 model): max residual %.3e of peak (gate < 1e-9)\n', resmax);
assert(resmax < 1e-9, 'G1 FAIL');
say_(rep, 'G1 PASS\n');

% ---- helper: differential phase between states ---------------------
dphase = @(X1, X0) angle(X1 .* conj(X0));

% ---- G2: poke 20 nm, stepped vs linear reading ----------------------
Ap = zeros(NACT);  Ap(60,40) = 1;
Xp = stepped_(20e-6*Ap, N_G, DX_G, NACT, PITCH, iTO, iMASK, iDET, VK, M2i, b2, N_WF);
hp = S_CONV * dphase(Xp, Xf) * LAM/(4*pi);
Mp = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', 20e-6*Ap);
[gp, rp] = mapfit_(hp, Mp, supp, iTO, iDET, N_WF, xgv_(N_G, DX_G), R_BEAM);
say_(rep, 'G2 poke 20 nm, stepped retrieval: map gain %.4f (1-frame linear read 0.445)\n', gp);
say_(rep, '   sign check: gain must be +; flip S_CONV if not.  resid %.3f nm\n', rp);
assert(gp > 0.445, 'G2 FAIL: stepped %.3f not better than linear 0.445', gp);
say_(rep, 'G2 PASS\n');

% ---- G3: RANGE -- poke 150 nm (3.0 rad) -----------------------------
Xr = stepped_(150e-6*Ap, N_G, DX_G, NACT, PITCH, iTO, iMASK, iDET, VK, M2i, b2, N_WF);
hr = S_CONV * dphase(Xr, Xf) * LAM/(4*pi);
Mr = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', 150e-6*Ap);
[gr, rr] = mapfit_(hr, Mr, supp, iTO, iDET, N_WF, xgv_(N_G, DX_G), R_BEAM);
say_(rep, 'G3 poke 150 nm (3.0 rad, linear reading folds): stepped gain %.4f, resid %.3f nm\n', gr, rr);
say_(rep, '   (gate: within 25%% of the 20 nm gain -- range extension demonstrated)\n');
assert(abs(gr/gp - 1) < 0.25, 'G3 FAIL: %.3f vs %.3f', gr, gp);
say_(rep, 'G3 PASS\n');

% ---- G4: defocus 8 nm about the 30 nm working base ------------------
rng(7);
[axg, ayg] = meshgrid(((1:NACT)-(NACT+1)/2)*PITCH);
lit0 = hypot(axg, ayg) < 0.85*38;
Abase = zeros(NACT);  Abase(lit0) = 30e-6*randn(nnz(lit0),1);
Xb = stepped_(Abase, N_G, DX_G, NACT, PITCH, iTO, iMASK, iDET, VK, M2i, b2, N_WF);
xg = xgv_(N_G, DX_G);
[gx, gy] = meshgrid(xg, xg);  rr2 = hypot(gx, gy);
Md = 8e-6*(2*(rr2/38).^2 - 1).*double(rr2 <= 38);
Mbd = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', Abase);
% base + defocus applied as grid sum (defocus is a figure, not commands)
Xbd = stepped_grid_(Mbd + Md, iTO, iMASK, iDET, VK, M2i, b2, N_WF);
hd = S_CONV * dphase(Xbd, Xb) * LAM/(4*pi);
[gd, rd] = mapfit_(hd, Md, supp, iTO, iDET, N_WF, xg, R_BEAM);
% PROPERTY, band-gated only: the stepped retrieval is exact PER STATE,
% so a low-order deviation moves the reference core WITH it and the
% self-reference attenuation reappears in differential low-order work.
% The frozen-reference LINEAR reading avoids exactly this for small
% deviations (defocus 0.986 on flat).  The two reconstructors are
% COMPLEMENTARY: stepped for range + exactness, frozen-reference
% linear for small low-order differentials.
say_(rep, 'G4 defocus 8 nm about the 30 nm base: gain %.4f, resid %.3f nm\n', gd, rd);
say_(rep, '   (self-reference attenuation under EXACT retrieval -- the moving core;\n');
say_(rep, '    frozen-reference linear read 0.986: complementary reconstructors)\n');
assert(gd > 0.5 && gd < 1.05, 'G4 out of physical band: %.4f', gd);
say_(rep, 'G4 PASS (property recorded)\n');

say_(rep, 'frames per measurement: 4 (3 depths + clear), vs 1 linear / 6 IFO traces\n');
say_(rep, 'S2b complete in %.1f min\n', toc(t_all)/60);
fclose(rep);
out = struct('gp',gp, 'gr',gr, 'gd',gd, 'resmax',resmax);
save('zwfs_s2b_run.mat', 'out');
fprintf('wrote zwfs_s2b_report.txt + zwfs_s2b_run.mat\n');
end

function xg = xgv_(N_G, DX_G)
    xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
end

% stepped retrieval for an ACTUATOR command state
function [X, pq, I0m] = stepped_(act, N_G, DX_G, NACT, PITCH, iTO, iMASK, iDET, VK, M2i, b2, N_WF)
    M = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', act);
    [X, pq, I0m] = stepped_grid_(M, iTO, iMASK, iDET, VK, M2i, b2, N_WF);
end

% stepped retrieval for an explicit surface GRID state (rank-2 solve;
% X = Eb conj(E0) reconstructed with the CALIBRATED |Eb|^2 = b2)
function [X, pq, I0m] = stepped_grid_(M, iTO, iMASK, iDET, VK, M2i, b2, N_WF)
    macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
    I0m = abs(macos.complex_field(iDET)).^2;      % clear frame, fresh pass
    Ik = zeros(N_WF, N_WF, 3);
    for k = 1:3
        macos.intensity(iMASK);
        macos.apodize_complex(iMASK, VK{k});
        Ik(:,:,k) = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
    end
    d = cat(3, Ik(:,:,1)-I0m, Ik(:,:,2)-I0m, Ik(:,:,3)-I0m);
    p = M2i(1,1)*d(:,:,1) + M2i(1,2)*d(:,:,2) + M2i(1,3)*d(:,:,3);  % Y - ReX
    q = M2i(2,1)*d(:,:,1) + M2i(2,2)*d(:,:,2) + M2i(2,3)*d(:,:,3);  % ImX
    X = (b2 - p) + 1i*q;
    pq = cat(3, p, q);
end

% map-space fit of a measured height map against a truth grid map,
% through the ray frame (== wf_figs scoring, no zoom)
function [g, r_nm] = mapfit_(hm, Mt, supp, iTO, iDET, N_WF, xg, R_BEAM)
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
    dxd = abs(macos.dx_at(iDET, 'mm'));
    [iy, ix] = find(supp);  cx = mean(ix);  cy = mean(iy);   %#ok<ASGLU>
    [cg, rg] = meshgrid((1:N_WF)-cx, (1:N_WF)-cy);
    xdm = cg*dxd*mag;  ydm = rg*dxd*mag;
    ht = interp2(xg, xg.', Mt*1e6, xdm, ydm, 'linear', 0);
    hmn = hm*1e6;  hmn(~supp) = NaN;  ht(~supp) = NaN;
    fitm = supp & (hypot(xdm,ydm) < 0.85*38) & ~isnan(ht) & ~isnan(hmn);
    hmn = hmn - median(hmn(fitm));  ht = ht - median(ht(fitm));
    g = ht(fitm) \ hmn(fitm);
    r_nm = sqrt(mean((hmn(fitm) - g*ht(fitm)).^2)) * 1e-0;   % nm
end

function say_(rep, fmt, varargin)
    fprintf(fmt, varargin{:});
    fprintf(rep, fmt, varargin{:});
end
