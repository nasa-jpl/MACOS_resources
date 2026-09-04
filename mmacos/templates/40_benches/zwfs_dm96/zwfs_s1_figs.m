function zwfs_s1_figs()
%ZWFS_S1_FIGS  Deck-grade figures for the ZWFS S1 state (zwfs_s1.m is
%   the gate; this script re-derives the same frames and draws them).
%   Writes: zwfs_layout.png   (bench sketch, test arm)
%           zwfs_mask_fig.png (focal-plane intensity + dimple footprint;
%                              mask phase zoom)
%           zwfs_response.png (known 8 nm figure vs single-frame recovery)
%   Run:  cd <this dir>;  matlab -batch "zwfs_s1_figs"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);

% constants (== zwfs_s1.m; MASK_TRIM from the S1 focus find, provenance
% zwfs_s1_report.txt -- the gate re-finds it every run, figures may pin it)
s = 96/56;  LAM = 6.328e-4;
MODEL = 512;  NGRID = 65;  N_G = 256;  DX_G = 0.4;
AOI = 7;  D_BS_TO = 700;  R_BEAM = s*30;
T_FL_F = 42.5325;  T_FL_Kc = -2.58764;  T_DMF = 39.7694;  T_TRIM = -1.2473;
MASK_TRIM = -5.582;
N_FS = 1.45702;  ETCH_MM = 346.2e-6;
PHI_M = 2*pi*(N_FS-1)*ETCH_MM/LAM;  DIA_LAMD = 1.06;  S_CONV = -1;

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

% ---- fig 1: bench sketch -------------------------------------------
fs = G.bt.sketch('title', ...
    'ZWFS train: source - splitter - 96 mm DM - focus (dimple) - pupil image');
set(fs, 'Position', [50 50 1300 560]);
set(findobj(fs,'Type','axes'), 'XLim', [-20 1900], 'YLim', [-330 320]);
exportgraphics(fs, 'zwfs_layout.png', 'Resolution', 130);
close(fs);
fprintf('wrote zwfs_layout.png\n');

% ---- frames (== S1) ------------------------------------------------
E0 = macos.complex_field(iDET);  N_WF = size(E0,1);
dx_mask_m = abs(macos.dx_at(iMASK));
lamD_mm = LAM*(s*250)/(2*R_BEAM);  dia_mm = DIA_LAMD*lamD_mm;
If = abs(macos.complex_field(iMASK)).^2;
[~, ipk] = max(If(:));  [pr, pc] = ind2sub(size(If), ipk);
w = 10;  rows = max(1,pr-w):min(N_WF,pr+w);  cols = max(1,pc-w):min(N_WF,pc+w);
Iw = If(rows, cols);
ctr_row = sum(sum(Iw,2).'.*rows)/sum(Iw(:));
ctr_col = sum(sum(Iw,1).*cols)/sum(Iw(:));
[V, D] = zwfs_mask(N_WF, dx_mask_m*1e3, dia_mm, PHI_M, [ctr_col, ctr_row]);
cc = exp(1i*PHI_M) - 1;

macos.intensity(iMASK);  macos.apodize_complex(iMASK, D);
Eb = macos.complex_field(iDET, 'reset_trace', false);
macos.intensity(iMASK);  macos.apodize_complex(iMASK, V);
Em = macos.complex_field(iDET, 'reset_trace', false);
I_flat = abs(Em).^2;

% ---- fig 2: focal plane + mask -------------------------------------
zc = round(ctr_col);  zr = round(ctr_row);  hw = 24;
rz = zr-hw:zr+hw;  czz = zc-hw:zc+hw;
lamD_px = (lamD_mm*1e-3)/dx_mask_m;
f2 = figure('Color','w', 'Position',[50 50 1150 460]);
t = tiledlayout(f2, 1, 2, 'Padding','compact', 'TileSpacing','compact');
ax1 = nexttile(t);
imagesc(ax1, log10(If(rz,czz)/max(If(:)) + 1e-12));  axis(ax1,'image');
colormap(ax1, 'parula');  clim(ax1, [-6 0]);  colorbar(ax1);
hold(ax1,'on');
th = linspace(0,2*pi,200);
plot(ax1, hw+1+(ctr_col-zc)+cos(th)*dia_mm*1e-3/dx_mask_m/2, ...
          hw+1+(ctr_row-zr)+sin(th)*dia_mm*1e-3/dx_mask_m/2, 'w-', 'LineWidth',1.6);
title(ax1, sprintf('focal spot, log_{10} intensity; dimple footprint (%.2f \\lambda F/D = %.2f um)', ...
      DIA_LAMD, dia_mm*1e3));
xlabel(ax1, sprintf('px (%.3f um/px, %.2f px per \\lambda F/D)', dx_mask_m*1e6, lamD_px));
ax2 = nexttile(t);
imagesc(ax2, angle(V(rz,czz)));  axis(ax2,'image');
colormap(ax2, 'gray');  clim(ax2, [0 pi/2]);  cb = colorbar(ax2);
cb.Label.String = 'phase (rad)';
title(ax2, sprintf('the mask: \\phi = %.3f rad (%.1f nm etch in fused silica)', ...
      PHI_M, ETCH_MM*1e6));
xlabel(ax2, 'px (gray edge = area-weighted supersampling)');
exportgraphics(f2, 'zwfs_mask_fig.png', 'Resolution', 130);
close(f2);
fprintf('wrote zwfs_mask_fig.png\n');

% ---- fig 3: response (== S1 G3, q=2 in the ray frame) --------------
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

A_MM = 8e-6;  Q_RAD = 2;
xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
[gx, gy] = meshgrid(xg, xg);  rr = hypot(gx, gy);
h_true = A_MM*cos(2*pi*Q_RAD*rr/R_BEAM).*double(rr <= R_BEAM);
sp = macos.get_elt_grid_spacing(iTO);
macos.set_elt_grid(iTO, sp, h_true);
macos.intensity(iMASK);  macos.apodize_complex(iMASK, V);
Ia = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;

Kmap = cc*Eb.*conj(E0);  den = 2*imag(Kmap);
I0 = abs(E0).^2;  supp = I0 > 0.1*max(I0(:));
msk = supp & (abs(den) > 0.05*max(abs(den(:))));
phi_est = zeros(N_WF);  phi_est(msk) = (Ia(msk)-I_flat(msk))./den(msk);
h_est = S_CONV*phi_est*LAM/(4*pi);
[iy, ix] = find(supp);  cx = mean(ix);  cy = mean(iy);       %#ok<ASGLU>
[dxp, dyp] = meshgrid((1:N_WF)-cx, (1:N_WF)-cy);
rd = hypot(dxp, dyp);
dxd_mm = macos.dx_at(iDET, 'mm');
r_dm = rd*abs(dxd_mm)*mag;
t_det = A_MM*cos(2*pi*Q_RAD*r_dm/R_BEAM);  t_det(~msk) = NaN;
h_show = h_est;  h_show(~msk) = NaN;
bb = find(any(msk,2));  b2 = find(any(msk,1));
rzz = bb(1):bb(end);  czz2 = b2(1):b2(end);

f3 = figure('Color','w', 'Position',[50 50 1150 460]);
t = tiledlayout(f3, 1, 2, 'Padding','compact', 'TileSpacing','compact');
ax1 = nexttile(t);
imagesc(ax1, t_det(rzz,czz2)*1e6, 'AlphaData', ~isnan(t_det(rzz,czz2)));
axis(ax1,'image');  colormap(ax1,'parula');  clim(ax1, [-8 8]);  colorbar(ax1);
title(ax1, 'the commanded figure: radial cosine, 8 nm (mapped to the camera by rays)');
ax2 = nexttile(t);
imagesc(ax2, h_show(rzz,czz2)*1e6, 'AlphaData', ~isnan(h_show(rzz,czz2)));
axis(ax2,'image');  colormap(ax2,'parula');  clim(ax2, [-8 8]);  cb = colorbar(ax2);
cb.Label.String = 'nm';
fitm = msk & (r_dm < 0.85*R_BEAM);
g = t_det(fitm) \ h_est(fitm);
title(ax2, sprintf('recovered from ONE camera frame: gain %.3f', g));
exportgraphics(f3, 'zwfs_response.png', 'Resolution', 130);
close(f3);
fprintf('wrote zwfs_response.png (gain %.4f)\n', g);
end
