function zwfs_wf_figs()
%ZWFS_WF_FIGS  Deck figures: traced-rig render + wavefront-estimate
%   triptychs (applied / sensed / error), cases and amplitudes matched
%   to tg_psi_dm96/tg96_wf_figs.m so the two decks compare directly:
%   center actuator poked 20 nm (0.40 rad -- inside the linear
%   reading) and defocus at 8 nm amplitude (the class the ZWFS
%   attenuates).  Linear reconstructor (R1); alternatives are a
%   scheduled study.
%   Config: model 1024, NGRID 193 (the 96x96 sampling-budget minimum:
%   detector Nyquist 96.5 = 2.01x actuator), spot 2.0 lambdaF/D from
%   the real VSG2 table (8 px on this focal grid; spot 9 at 1.06 is
%   4 px here -- under-resolved, the S5 spot trade).
%   Writes: zwfs_render_rig.png, zwfs_poke_triptych.png,
%           zwfs_defocus_triptych.png
%   Run:  cd <this dir>;  matlab -batch "zwfs_wf_figs"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));  % dm_influence_map

s = 96/56;  LAM = 6.328e-4;
MODEL = 1024;  NGRID = 193;  N_G = 384;  DX_G = 0.28;
NACT = 96;  PITCH = 1.0;  AOI = 7;  D_BS_TO = 700;  R_BEAM = s*30;
T_FL_F = 42.5325;  T_FL_Kc = -2.58764;  T_DMF = 39.7694;  T_TRIM = -1.2473;
MASK_TRIM = -5.582;                       % S1 focus find (zwfs_s1_report.txt)
N_FS = 1.45702;  ETCH_MM = 346.2e-6;
PHI_M = 2*pi*(N_FS-1)*ETCH_MM/LAM;
DIA_LAMD = 2.0;                           % VSG2 spot 2 (see header)
S_CONV = -1;

macos.init(MODEL);
assert(N_G <= macos.grid_size_max(), 'N_G vs mGridMat');
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

% ---- fig 1: traced-rig render (demo beat-2 recipe) -----------------
macos.trace(iDET);
d0 = G.bt.src_dir(:);  [~, i0] = min(abs(d0));
xb = zeros(3,1);  xb(i0) = 1;  xb = xb - dot(xb,d0)*d0;  xb = xb/norm(xb);
yb = cross(d0, xb);
ai = deg2rad([-35 22]);
VW = { -yb, xb, 'TABLE PLANE -- looking down on the bench' ; ...
       cos(ai(2))*(cos(ai(1))*xb + sin(ai(1))*d0) + sin(ai(2))*yb, yb, 'ISO view' };
f = figure('Color','w', 'Position',[40 40 1700 560], 'Visible','off');
tl2 = tiledlayout(f, 1, 2, 'Padding','tight', 'TileSpacing','tight');
for q = 1:size(VW,1)
    ax = nexttile(tl2);
    macos.view_rx('ax', ax, 'title', VW{q,3});
    axis(ax, 'equal');
    xl = xlim(ax);  yl = ylim(ax);  zl = zlim(ax);
    tgt = [mean(xl); mean(yl); mean(zl)];
    dd  = 3*max([diff(xl), diff(yl), diff(zl)]);
    set(ax, 'CameraTarget', tgt.', 'CameraPosition', (tgt - dd*VW{q,1}).', ...
            'CameraUpVector', VW{q,2}.', 'Projection', 'orthographic');
    camva(ax, 'auto');  camzoom(ax, 1.7);  axis(ax, 'off');
end
title(tl2, 'Zernike-sensor train at 96 mm: the interferometer''s test arm alone');
print(f, 'zwfs_render_rig.png', '-dpng', '-r150');
close(f);
fprintf('wrote zwfs_render_rig.png\n');

% ---- flat-state reference frames -----------------------------------
E0 = macos.complex_field(iDET);  N_WF = size(E0,1);
dx_mask_m = abs(macos.dx_at(iMASK));
lamD_mm = LAM*(s*250)/(2*R_BEAM);  dia_mm = DIA_LAMD*lamD_mm;
dia_px = (dia_mm*1e-3)/dx_mask_m;
fprintf('dimple: %.2f lamF/D = %.2f px on the focal grid\n', DIA_LAMD, dia_px);
assert(dia_px >= 6, 'dimple under-resolved: %.2f px', dia_px);
If = abs(macos.complex_field(iMASK)).^2;
pk = max(If(:))/sum(If(:));
assert(pk >= 1e-2, 'mask plane not focused (peak/sum %.2e) -- re-run the S1 scan', pk);
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

% ---- ray frame -----------------------------------------------------
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
fprintf('ray frame: mag %.4f, det px -> DM mm scale %.5f\n', mag, dxd_mm*mag);

% ---- the two cases (== tg96_wf_figs) -------------------------------
% Pattern radius = the ILLUMINATED beam radius, measured from the flat
% support through the ray frame (the source cone fills ~41 mm of the
% 51.4 mm aperture -- patterns on the aperture radius overhang the light).
[syy, sxx] = find(msk);
scx = mean(sxx);  scy = mean(syy);
rpx = hypot(sxx-scx, syy-scy);
rs = sort(rpx);  R_PAT = rs(round(0.98*numel(rs))) * dxd_mm * mag;
fprintf('pattern radius R_PAT = %.2f mm (aperture %.2f)\n', R_PAT, R_BEAM);

Ap = zeros(NACT);  Ap(48,48) = 1;
Mp = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', 20e-6*Ap);
hp = zwfs_meas(Mp, iTO, iMASK, iDET, V, I_flat, den, msk, S_CONV, LAM, N_WF);

xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
[gx, gy] = meshgrid(xg, xg);  rr = hypot(gx, gy);
Md = 8e-6*(2*(rr/R_PAT).^2 - 1).*double(rr <= R_PAT);
hd = zwfs_meas(Md, iTO, iMASK, iDET, V, I_flat, den, msk, S_CONV, LAM, N_WF);

triptych_wf('zwfs_poke_triptych.png', Mp, hp, +1, msk, mag, dxd_mm, xg, R_PAT, ...
    'Zernike sensor: one actuator pushed 20 nm (one frame)', true, 6);
triptych_wf('zwfs_defocus_triptych.png', Md, hd, +1, msk, mag, dxd_mm, xg, R_PAT, ...
    'Zernike sensor: defocus, 8 nm amplitude (one frame)', false, 10);
fprintf('done\n');
end

function h = zwfs_meas(M, iTO, iMASK, iDET, V, I_flat, den, msk, S_CONV, LAM, N_WF)
    sp = macos.get_elt_grid_spacing(iTO);
    macos.set_elt_grid(iTO, sp, M);
    macos.intensity(iMASK);
    macos.apodize_complex(iMASK, V);
    Ia = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
    phi = zeros(N_WF);
    phi(msk) = (Ia(msk) - I_flat(msk)) ./ den(msk);
    h = S_CONV*phi*LAM/(4*pi);
end

% ==== display helper (shared design with the sibling campaign) =======
function triptych_wf(fname, Mtruth, hmeas, sgn, msk, mag, dxd_mm, xg, R_PAT, ttl, zoom_on_blob, clim_nm)
%  SGN: deck-measured measurement sign (IFO: -1 from the tg96 two-poke
%  registration; ZWFS: +1, its S_CONV is applied upstream).  R_PAT: the
%  pattern radius (measured ILLUMINATED beam radius -- the source cone
%  fills ~41 mm of the 51.4 mm aperture; defining patterns on the
%  aperture radius leaves truth beyond the light).
    N = size(hmeas,1);
    hm = sgn*hmeas*1e6;                              % nm
    hm(~msk) = NaN;
    [iy, ix] = find(msk);  cx0 = mean(ix);  cy0 = mean(iy);   %#ok<ASGLU>
    w = abs(hm - median(hm(msk)));  w(~msk | isnan(hm)) = 0;
    if zoom_on_blob
        wq = w;  wq(w < 0.5*max(w(:))) = 0;          % blob centroid
        [cg, rg] = meshgrid(1:N, 1:N);
        cx = sum(cg(:).*wq(:))/sum(wq(:));  cy = sum(rg(:).*wq(:))/sum(wq(:));
    else
        cx = cx0;  cy = cy0;
    end
    [cg, rg] = meshgrid((1:N)-cx, (1:N)-cy);
    xdm = cg*dxd_mm*mag;  ydm = rg*dxd_mm*mag;
    Mt = Mtruth*1e6;                                  % nm
    if zoom_on_blob
        [~, im] = max(abs(Mt(:)));  [tr, tc] = ind2sub(size(Mt), im);
        tx0 = xg(tc);  ty0 = xg(tr);
    else
        tx0 = 0;  ty0 = 0;
    end
    ht = interp2(xg, xg.', Mt, xdm+tx0, ydm+ty0, 'linear', 0);
    ht(~msk) = NaN;
    if zoom_on_blob
        hw = 24;  rz = max(1,round(cy)-hw):min(N,round(cy)+hw);
        czz = max(1,round(cx)-hw):min(N,round(cx)+hw);
    else
        bb = find(any(msk,2));  b2 = find(any(msk,1));
        rz = bb(1):bb(end);  czz = b2(1):b2(end);
    end
    rdm_all = hypot(xdm, ydm);
    fitm = msk & (rdm_all < 0.85*R_PAT) & ~isnan(ht) & ~isnan(hm);
    % common piston over the fit region, both maps
    hm = hm - median(hm(fitm));  ht = ht - median(ht(fitm));
    err = hm - ht;
    rmse = sqrt(mean(err(fitm).^2));
    g = ht(fitm) \ hm(fitm);
    % self-checks (frame-before-angle): applied-peak offset in the
    % window, and where the support rim maps in DM mm
    Aw = ht(rz,czz);  [~, ia] = max(abs(Aw(:)));  [ar, ac] = ind2sub(size(Aw), ia);
    fprintf('%s: gain %.3f, rms %.3f nm; applied-peak window offset (%+d,%+d) px; support rim %.1f mm (R_PAT %.1f)\n', ...
        fname, g, rmse, ac-(round(cx)-czz(1)+1), ar-(round(cy)-rz(1)+1), ...
        prctile_(rdm_all(msk), 98), R_PAT);
    f = figure('Color','w', 'Position',[40 40 1500 430], 'Visible','off');
    t = tiledlayout(f, 1, 3, 'Padding','compact', 'TileSpacing','compact');
    panels = {ht, 'applied (truth, mapped to the camera)'; ...
              hm, 'sensed (reconstructed surface)'; ...
              err, sprintf('estimate error: %.3f nm rms', rmse)};
    for k = 1:3
        ax = nexttile(t);
        A = panels{k,1}(rz,czz);
        imagesc(ax, A, 'AlphaData', ~isnan(A));  axis(ax,'image');
        colormap(ax, 'parula');
        if k < 3, clim(ax, [-clim_nm clim_nm]); else, clim(ax, [-clim_nm clim_nm]/4); end
        cb = colorbar(ax);  cb.Label.String = 'nm';
        title(ax, panels{k,2});
        set(ax, 'XTick', [], 'YTick', []);
    end
    title(t, ttl);
    exportgraphics(f, fname, 'Resolution', 130);
    close(f);
end

function v = prctile_(x, p)
    x = sort(x(~isnan(x)));  v = x(max(1, round(p/100*numel(x))));
end
