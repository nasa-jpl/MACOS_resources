function out = zwfs_s1()
%ZWFS_S1  Stage 1 of the ZWFS campaign: mask + response gates (dev res).
%   The ZWFS instrument is the TG96 TEST ARM ALONE (non-polarizing
%   twyman_green, same geometry, same tuned tail) with the Zernike
%   dimple applied at the FocalMask element via macos.apodize_complex.
%   The mask plane gets focal-scale sampling from the NF1/NF2
%   reference-sphere sandwich ('mask_prop','nf' -- the ctb_dcr.in FPM
%   idiom); a plain geometric leg lands the wavefront there on a
%   PUPIL-scaled grid (measured 5.19 um/px, dimple 0.54 px -- the
%   first-run G0 failure that motivated the sandwich).
%   Gates:
%     G0  focal-plane sampling: dimple >= 6 px across at the mask plane
%     G1  exact superposition: E_masked == E0 + c*Eb to round-off
%         (proves the mask bites AND the field algebra, in one identity)
%     G2  reference-wave sanity (core fraction printed, bounds asserted)
%     G3  known low-order DM figure (parity-proof radial pattern, 8 nm)
%         reconstructs with gain ~ 1 against truth
%     G4  the same state WITHOUT the dimple reconstructs to nothing
%         (phase-blindness of plain pupil imaging -- the signal is the
%         dimple's, non-vacuity)
%   Dev res: model 512, NGRID 65 -- the small ray grid buys focal
%   padding (dimple px ~ dia_lamD * MODEL/NGRID); real-work sampling is
%   S2's problem (48x48 at NGRID 193, 96x96 at 385 / spot choice).
%   Run:  cd <this dir>;  matlab -batch "zwfs_s1"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
t_all = tic;
rep = fopen('zwfs_s1_report.txt', 'w');

% ---- constants ------------------------------------------------------
s   = 96/56;                 % uniform scale from the 56 mm v1 rig
LAM = 6.328e-4;              % mm
MODEL = 512;  NGRID = 65;  N_G = 256;  DX_G = 0.4;    % dev res
AOI = 7;  D_BS_TO = 700;     % TG96 clearance-solve values
R_BEAM = s*30;               % beam radius at the test optic aperture, mm
% Tuned tail (tg96_tail.m winner, tg96_report.txt "RE-TUNED set"):
T_FL_F = 42.5325;  T_FL_Kc = -2.58764;  T_DMF = 39.7694;  T_TRIM = -1.2473;
% ZWFS mask, VSG2 hardware (vsg_wip/vsg2_params.m section 9):
N_FS     = 1.45702;          % fused silica @ 632.8 nm
ETCH_MM  = 346.2e-6;         % 346.2 nm physical etch
PHI_M    = 2*pi*(N_FS - 1)*ETCH_MM/LAM;    % ~pi/2 dimple phase
DIA_LAMD = 1.06;             % spot 9 (default VSG2 spot)
% Sign of h = S_CONV * phi * lambda/(4*pi): MEASURED round 5 (2026-09-04)
% -- +1 gave gain -0.54 on the known figure; the engine's phase
% convention wants the flip.  Pinned; G3 gates the sign.
S_CONV = -1;

say_(rep, '=== ZWFS S1: mask + response gates (dev res, model %d, NGRID %d) ===\n', MODEL, NGRID);
say_(rep, 'dimple: phi = %.4f rad (%.3f waves), dia %.2f lambda/D (VSG2 spot 9)\n', ...
     PHI_M, PHI_M/(2*pi), DIA_LAMD);

macos.init(MODEL);
assert(N_G <= macos.grid_size_max(), 'N_G %d exceeds mGridMat for model %d', ...
       N_G, MODEL);
macos.write_grid_file('zwfs_flat.txt', zeros(N_G));

% ---- build helper + mask-focus scan --------------------------------
% The FocalMask sits at the THIN-LENS seed focus; the real thick-lens
% focus is a fraction of a mm away, and at lambda/D scale that spreads
% the spot over ~100s of px (round-3 finding: peak/sum 4.5e-5 vs the
% ~2e-2 of a focused core).  MASK_TRIM exists for exactly this; the
% PSI tail never needed it (its mask is a pass-through).  Scan the
% trim, maximizing the mask-plane peak = focusing the ZWFS mask stage.
C = struct('s',s, 'NGRID',NGRID, 'AOI',AOI, 'D_BS_TO',D_BS_TO, ...
           'N_G',N_G, 'DX_G',DX_G, 'T_FL_F',T_FL_F, 'T_FL_Kc',T_FL_Kc, ...
           'T_DMF',T_DMF, 'T_TRIM',T_TRIM);
% Round-5 wide scan (-30:3:9) found the focus at -5.58 mm: L2's Kr is
% the l2_trade-OPTIMIZED value, not the thin-lens seed, and the axial
% peak is only tens of um deep (peak/sum 1e-4 wings -> 1.2e-2 at
% focus), so coarse steps straddle it.  Re-find it in that bracket
% each run (self-heals if the tail geometry moves).
t_fine = fminbnd(@(t) -bld_peak_(C, t), -8.6, -2.6, ...
                 optimset('TolX',0.02, 'MaxFunEvals',25, 'Display','off'));
MASK_TRIM = t_fine;
pk_f = bld_peak_(C, MASK_TRIM);
say_(rep, 'MASK_TRIM = %+.3f mm (mask-plane peak/sum %.3e; focused class >= 1e-2)\n', ...
     MASK_TRIM, pk_f);
assert(pk_f >= 1e-2, 'mask focus not found in bracket -- re-run the wide scan');

% ---- final build at the found trim ---------------------------------
G = bld_(C, MASK_TRIM);
G.bt.emit('zwfs_test.in');
iTO = G.T.iTO;  iMASK = G.T.iMASK;  iDET = G.T.iDET;
macos.load_rx('zwfs_test.in');
say_(rep, 'bench: test arm only, iTO=%d iMASK=%d iDET=%d\n', iTO, iMASK, iDET);

% ---- ray-measured DM->detector frame (registration doctrine, DOF
% class 1; lifted from tg96 register_two_pokes) -- the support-area
% radius estimate proved ~25% off (round 8: scale sweep pinned at its
% edge), which decorrelates any q>~3 pattern.  Scale comes from rays.
s1t = macos.trace(iTO);   ito  = macos.get_ray_info(s1t.nRays);
s2t = macos.trace(iDET);  idet = macos.get_ray_info(s2t.nRays);
okr = ito.ok_trace(:) & ito.ok_pass(:) & idet.ok_trace(:) & idet.ok_pass(:);
psi1 = macos.get_elt_psi(iTO);  vpt1 = macos.get_elt_vpt(iTO);
u1 = macos.design.Bench.perp(psi1);  v1 = cross(psi1, u1);
xy_to = [u1.'; v1.'] * (ito.pos - vpt1);
psi2 = macos.get_elt_psi(iDET);
u2 = macos.design.Bench.perp(psi2);  v2 = cross(psi2, u2);
xy_d = [u2.'; v2.'] * (idet.pos - idet.pos(:,1));
xy_to = xy_to(:,okr);  xy_d = xy_d(:,okr);
Aaf = [xy_d.' ones(nnz(okr),1)] \ xy_to.';
Lm  = Aaf(1:2,:).';
[~,Ss,~] = svd(Lm);  sm = diag(Ss);
mag = sqrt(abs(det(Lm)));
nlr = xy_to - (Lm*xy_d + Aaf(3,:).');
say_(rep, 'ray frame: mag %.4f DM-mm/det-mm, anam %.2f%%, nonlin %.4f mm\n', ...
     mag, 100*(sm(1)/sm(2)-1), sqrt(mean(sum(nlr.^2,1))));

% ---- flat-DM baseline field + G0 sampling --------------------------
E0 = macos.complex_field(iDET);               % fresh full pass, no mask
N_WF = size(E0, 1);
dx_mask_m = abs(macos.dx_at(iMASK));          % SI metres
lamD_mm = LAM * (s*250) / (2*R_BEAM);         % lambda*F2/Dbeam, mm
dia_mm  = DIA_LAMD * lamD_mm;
dia_px  = (dia_mm*1e-3) / dx_mask_m;
say_(rep, 'G0 sampling: WF grid %d^2, dx(mask) %.3e m, lambda F/D %.3e mm, dimple %.3e mm = %.2f px\n', ...
     N_WF, dx_mask_m, lamD_mm, dia_mm, dia_px);
if dia_px < 6
    say_(rep, 'G0 FAIL: dimple under-resolved at the mask plane (%.2f px < 6).\n', dia_px);
    fclose(rep);
    error('zwfs_s1:G0', 'dimple %.2f px at mask plane -- under-resolved', dia_px);
end
say_(rep, 'G0 PASS\n');

% ---- locate the focal spot, center the mask on it ------------------
% (bench alignment: the BS plate's refraction walks the real chief off
% the builder's nominal axis; the spot lands tens of lambda/D from the
% grid DC pixel.  Measure it and translate the mask -- what the 9-spot
% substrate does on the real VSG2.)
If = abs(macos.complex_field(iMASK)).^2;      % field AT the mask plane
[~, ipk] = max(If(:));  [pr, pc] = ind2sub(size(If), ipk);
w = 10;  rows = max(1,pr-w):min(N_WF,pr+w);  cols = max(1,pc-w):min(N_WF,pc+w);
Iw = If(rows, cols);
ctr_row = sum(sum(Iw,2).' .* rows) / sum(Iw(:));
ctr_col = sum(sum(Iw,1)  .* cols) / sum(Iw(:));
dc = floor(N_WF/2) + 1;
say_(rep, 'spot at (col %.2f, row %.2f), DC pixel %d: offset (%.1f, %.1f) px = (%.1f, %.1f) um\n', ...
     ctr_col, ctr_row, dc, ctr_col-dc, ctr_row-dc, ...
     (ctr_col-dc)*dx_mask_m*1e6, (ctr_row-dc)*dx_mask_m*1e6);

% ---- Eb (disk-support only) and Em (the dimple) --------------------
[V, D] = zwfs_mask(N_WF, dx_mask_m*1e3, dia_mm, PHI_M, [ctr_col, ctr_row]);  % mm units
cc = exp(1i*PHI_M) - 1;

macos.intensity(iMASK);                       % fresh trace to the mask plane
macos.apodize_complex(iMASK, D);
Eb = macos.complex_field(iDET, 'reset_trace', false);

macos.intensity(iMASK);
macos.apodize_complex(iMASK, V);
Em = macos.complex_field(iDET, 'reset_trace', false);
I_flat = abs(Em).^2;

% ---- G1: exact superposition ---------------------------------------
rel = norm(Em - (E0 + cc*Eb), 'fro') / norm(E0, 'fro');
say_(rep, 'G1 superposition |Em - (E0 + c*Eb)|/|E0| = %.3e (gate < 1e-9)\n', rel);
assert(rel < 1e-9, 'G1 FAIL: %.3e', rel);
say_(rep, 'G1 PASS\n');

% ---- G2: reference-wave sanity -------------------------------------
core_frac = sum(abs(Eb(:)).^2) / sum(abs(E0(:)).^2);
say_(rep, 'G2 reference wave: core power fraction %.4f (gate in [0.01, 0.99])\n', core_frac);
assert(core_frac > 0.01 && core_frac < 0.99, 'G2 FAIL');
say_(rep, 'G2 PASS\n');

% ---- G3: known figure reconstructs ---------------------------------
% Parity-proof RADIAL COSINE at ~5 cycles across the radius -- above
% the dimple's ~0.5 cyc/pupil reference passband, where the ZWFS gain
% is ~1.  (Round 5 measured DEFOCUS at gain 0.54: aberration content
% inside the dimple passband leaks into the self-referenced core wave
% and subtracts -- real ZWFS low-order attenuation, printed below as
% an instrument property, not gated.)
A_MM = 8e-6;                                  % 8 nm in mm (grid value IS height)
Q_RAD = 2;                                    % radial cycles across R_BEAM
% q=2 (~4 cyc/pupil-diameter): 8x above the dimple's ~0.5 cyc/pup
% reference passband, and 16 px/cycle on the dev-res 65-px pupil --
% clear of BOTH the ZWFS low-order attenuation and the dev sampling
% rolloff (q=5 measured 0.675 with a verified frame: dev-res sampling,
% round 9 -- transfer curves are S3's business at full res).
xg = ((0:N_G-1) - (N_G-1)/2) * DX_G;
[gx, gy] = meshgrid(xg, xg);  rr = hypot(gx, gy);
h_true = A_MM * cos(2*pi*Q_RAD*rr/R_BEAM) .* double(rr <= R_BEAM);
h_defoc = A_MM * (2*(rr/R_BEAM).^2 - 1) .* double(rr <= R_BEAM);
sp = macos.get_elt_grid_spacing(iTO);
macos.set_elt_grid(iTO, sp, h_true);

macos.intensity(iMASK);
macos.apodize_complex(iMASK, V);
Ea = macos.complex_field(iDET, 'reset_trace', false);
Ia = abs(Ea).^2;
% no-dimple frame of the same state, for G4:
Ia0 = abs(macos.complex_field(iDET)).^2;

% Linear reconstruction about the flat state:
Kmap = cc * Eb .* conj(E0);
den  = 2 * imag(Kmap);
I0   = abs(E0).^2;
supp = I0 > 0.1*max(I0(:));
msk  = supp & (abs(den) > 0.05*max(abs(den(:))));
phi_est = zeros(N_WF);
phi_est(msk) = (Ia(msk) - I_flat(msk)) ./ den(msk);
h_est = S_CONV * phi_est * LAM/(4*pi);        % mm; single reflection = x2

% truth on the detector grid, from the measured beam support:
[iy, ix] = find(supp);                                     %#ok<ASGLU>
cx = mean(ix);  cy = mean(iy);  Rd = sqrt(nnz(supp)/pi);
[dxp, dyp] = meshgrid((1:N_WF)-cx, (1:N_WF)-cy);
rd = hypot(dxp, dyp);
% Truth mapped through the RAY frame: detector px -> det mm (dx_at,
% about the support centroid) -> DM mm (ray magnification).  The
% area-estimate radius Rd stays only for the fit region bound.
dxd_mm = macos.dx_at(iDET, 'mm');
r_dm = rd * abs(dxd_mm) * mag;                 % DM-plane radius per pixel
fitm = msk & (r_dm < 0.85*R_BEAM);
t_det = A_MM * cos(2*pi*Q_RAD*r_dm/R_BEAM);
g  = t_det(fitm) \ h_est(fitm);
res = std(h_est(fitm) - g*t_det(fitm)) * 1e6;  % nm
kimp = abs(dxd_mm)*mag*Rd/R_BEAM;
say_(rep, 'G3 frame: ray scale implies k = %.3f of the area estimate (round-8 sweep pinned at 0.80)\n', kimp);
% diagnostic sweep about the ray frame (should peak at ~1.00 now):
cbest = 0;  kstar = 1;
for k = 0.7:0.005:1.3
    tk = A_MM * cos(2*pi*Q_RAD*r_dm*k/R_BEAM);
    cm = corrcoef(tk(fitm), h_est(fitm));
    if abs(cm(1,2)) > abs(cbest), cbest = cm(1,2); kstar = k; end
end
say_(rep, 'G3 sweep about ray frame: k* = %.3f, |corr| %.3f (diagnostic)\n', kstar, abs(cbest));
say_(rep, 'G3 radial cosine (q=%d) 8 nm figure: gain %+.4f, resid %.4f nm (gate |gain-1| <= 0.15)\n', ...
     Q_RAD, g, res);
say_(rep, '   (S_CONV = %+d, pinned round 5)\n', S_CONV);
assert(abs(g - 1) <= 0.15, 'G3 FAIL: gain %+.4f', g);
say_(rep, 'G3 PASS\n');

% instrument property (ungated): the defocus response.  NOTE (round 11,
% full-res wf-figs): earlier "attenuation" readings (0.54, 0.32) were
% PATTERN-RADIUS bias -- the source cone fills ~74% of the aperture, and
% patterns defined on R_BEAM overhang the light.  On the measured
% illuminated radius, defocus reads ~0.99 (spot 2.0, full res): the
% self-reference attenuation lives at piston/tilt class, not defocus.
R_ILL = prctile_supp_(rd(msk)) * abs(dxd_mm) * mag;
say_(rep, 'illuminated radius %.2f mm (aperture %.2f)\n', R_ILL, R_BEAM);
h_defoc = A_MM * (2*(rr/R_ILL).^2 - 1) .* double(rr <= R_ILL);
macos.set_elt_grid(iTO, sp, h_defoc);
macos.intensity(iMASK);
macos.apodize_complex(iMASK, V);
Ed = macos.complex_field(iDET, 'reset_trace', false);
phi_d = zeros(N_WF);
phi_d(msk) = (abs(Ed(msk)).^2 - I_flat(msk)) ./ den(msk);
h_d = S_CONV * phi_d * LAM/(4*pi);
t_d = A_MM * (2*(r_dm/R_ILL).^2 - 1);
g_d = t_d(fitm) \ h_d(fitm);
say_(rep, 'defocus response (property, ungated): gain %+.4f -- ZWFS low-order attenuation\n', g_d);
macos.set_elt_grid(iTO, sp, h_true);           % restore G3/G4 state

% ---- G4: without the dimple the same state reads as nothing --------
phi_bad = zeros(N_WF);
phi_bad(msk) = (Ia0(msk) - I0(msk)) ./ den(msk);
h_bad = S_CONV * phi_bad * LAM/(4*pi);
g_bad = t_det(fitm) \ h_bad(fitm);
say_(rep, 'G4 no-dimple frame: gain %+.4f (gate |gain| < 0.2 -- plain pupil imaging is phase-blind)\n', g_bad);
assert(abs(g_bad) < 0.2, 'G4 FAIL: gain %+.4f', g_bad);
say_(rep, 'G4 PASS\n');

say_(rep, 'S1 complete in %.1f s\n', toc(t_all));
fclose(rep);
out = struct('dia_px',dia_px, 'rel_super',rel, 'core_frac',core_frac, ...
             'gain',g, 'resid_nm',res, 'gain_nodimple',g_bad, 's_conv',S_CONV);
save('zwfs_s1_run.mat', 'out');
fprintf('wrote zwfs_s1_report.txt + zwfs_s1_run.mat\n');
end

function v = prctile_supp_(x)
    x = sort(x(:));  v = x(max(1, round(0.98*numel(x))));
end

function say_(rep, fmt, varargin)
    fprintf(fmt, varargin{:});
    fprintf(rep, fmt, varargin{:});
end

function G = bld_(C, mask_trim)
    s = C.s;
    G = macos.design.twyman_green('polarizing',false, 'ngridpts',C.NGRID, ...
        'BS_AOI',C.AOI, ...
        'F1',s*500, 'F2',s*250, 'D_LENS',s*60, 'R_BAFFLE',s*12.5, 'D_SB',s*250, ...
        'BS_T',s*1.5, 'D_L1_BS',s*150, 'D_BS_TO',C.D_BS_TO, 'D_BS_CMP',s*100, ...
        'R_TO_AP',s*30, 'L1_Kr',s*236.866, 'L1_Kc',-0.5829, ...
        'L2_Kr',-s*124.076, 'L2_Kc',-0.5826, ...
        'to_grid_file','zwfs_flat.txt', 'to_grid_n',C.N_G, 'to_grid_dx',C.DX_G, ...
        'tail_arch','fieldlens', 'mask_prop','nf', 'MASK_TRIM',mask_trim, ...
        'FL_F',C.T_FL_F, 'FL_Kc',C.T_FL_Kc, ...
        'FL_D',s*12, 'D_MASK_FL',C.T_DMF, 'DET_TRIM',C.T_TRIM);
end

function pk = bld_peak_(C, mask_trim)
    G = bld_(C, mask_trim);
    G.bt.emit('zwfs_scan.in');
    macos.load_rx('zwfs_scan.in');
    If = abs(macos.complex_field(G.T.iMASK)).^2;
    pk = max(If(:)) / sum(If(:));
end
