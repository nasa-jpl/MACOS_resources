function out = zwfs_sweep()
%ZWFS_SWEEP  Sampling sweep: where does the 96x96 Zernike sensor work?
%   (Dave 2026-09-04: "If sampling is starving the ZWFS, sweep the
%   sampling to find where it all works.")  At model 1024 the sensor is
%   squeezed from BOTH ends: low NGRID starves the DM (rays/actuator),
%   high NGRID starves the dimple (px/lamF/D falls as NGRID/MODEL).
%   Model 2048 cannot relieve it -- mGridMat caps at 128 there, under
%   the 384 the DM grid needs (the known relief valve is a param bump).
%   Sweep NGRID x spot over the valid combos (dimple >= ~6 px), score
%   the matched cases (center actuator 20 nm; defocus 8 nm on the
%   illuminated radius) by fitted gain + rms error.  Ultimate target
%   (Dave): measurement error ~ 1 pm -- errors printed in pm.
%   Writes: zwfs_sweep_report.txt, zwfs_sweep.png, zwfs_sweep.mat
%   Run:  cd <this dir>;  matlab -batch "zwfs_sweep"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
t_all = tic;
rep = fopen('zwfs_sweep_report.txt', 'w');

s = 96/56;  LAM = 6.328e-4;
MODEL = 1024;  N_G = 384;  DX_G = 0.28;
NACT = 96;  PITCH = 1.0;  AOI = 7;  D_BS_TO = 700;  R_BEAM = s*30;
T_FL_F = 42.5325;  T_FL_Kc = -2.58764;  T_DMF = 39.7694;  T_TRIM = -1.2473;
MASK_TRIM = -5.582;
N_FS = 1.45702;  ETCH_MM = 346.2e-6;
PHI_M = 2*pi*(N_FS-1)*ETCH_MM/LAM;
S_CONV = -1;

% valid (NGRID, spot) combos; spot diameters from the real VSG2 table
CFG = [129 1.06; 129 2.0; 129 3.0; ...
       193 2.0;  193 3.0; ...
       257 2.0;  257 3.0; ...
       385 3.0];

say_(rep, '=== ZWFS sampling sweep (model %d, DM %dx%d): where does it all work? ===\n', ...
     MODEL, NACT, NACT);
say_(rep, 'cases: center actuator 20 nm; defocus 8 nm on the illuminated radius.\n');
say_(rep, 'target context (Dave): measurement error ~ 1 pm ultimate.\n');
say_(rep, '%6s %6s | %8s %8s %8s | %7s %9s | %7s %9s\n', ...
     'NGRID', 'spot', 'dimplePx', 'illumPx', 'nyqMarg', 'pokeG', 'pokeE_pm', 'defG', 'defE_pm');

macos.init(MODEL);
macos.write_grid_file('zwfs_flat.txt', zeros(N_G));
R = struct('ngrid',{},'spot',{},'dimple_px',{},'illum_px',{},'nyq',{}, ...
           'pokeg',{},'pokee',{},'defg',{},'defe',{});

for k = 1:size(CFG,1)
    NGRID = CFG(k,1);  DIA_LAMD = CFG(k,2);
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
    G.bt.emit('zwfs_swp.in');
    iTO = G.T.iTO;  iMASK = G.T.iMASK;  iDET = G.T.iDET;
    sp0 = [];  % grid spacing, fetched after load
    macos.load_rx('zwfs_swp.in');
    macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), zeros(N_G));

    E0 = macos.complex_field(iDET);  N_WF = size(E0,1);
    dx_mask_m = abs(macos.dx_at(iMASK));
    lamD_mm = LAM*(s*250)/(2*R_BEAM);  dia_mm = DIA_LAMD*lamD_mm;
    dia_px = (dia_mm*1e-3)/dx_mask_m;
    If = abs(macos.complex_field(iMASK)).^2;
    assert(max(If(:))/sum(If(:)) >= 1e-2, 'not focused at NGRID %d', NGRID);
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

    % ray frame + illuminated radius
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
    [syy, sxx] = find(msk);
    rpx = hypot(sxx-mean(sxx), syy-mean(syy));
    rs = sort(rpx);  illum_px = 2*rs(round(0.98*numel(rs)));
    R_PAT = illum_px/2 * dxd_mm * mag;
    n_lit_act = 2*R_PAT/PITCH;                    % actuators across lit pupil
    nyq = (illum_px/2) / (n_lit_act/2);           % det Nyquist / actuator Nyquist

    % cases
    Ap = zeros(NACT);  Ap(48,48) = 1;
    Mp = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', 20e-6*Ap);
    hp = meas_(Mp, iTO, iMASK, iDET, V, I_flat, den, msk, S_CONV, LAM, N_WF);
    xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
    [gx, gy] = meshgrid(xg, xg);  rr = hypot(gx, gy);
    Md = 8e-6*(2*(rr/R_PAT).^2 - 1).*double(rr <= R_PAT);
    hd = meas_(Md, iTO, iMASK, iDET, V, I_flat, den, msk, S_CONV, LAM, N_WF);

    [gp, ep] = score_(Mp, hp, msk, mag, dxd_mm, xg, R_PAT, true);
    [gd, ed] = score_(Md, hd, msk, mag, dxd_mm, xg, R_PAT, false);
    say_(rep, '%6d %6.2f | %8.2f %8.0f %8.2f | %7.3f %9.1f | %7.3f %9.1f\n', ...
         NGRID, DIA_LAMD, dia_px, illum_px, nyq, gp, ep*1e3, gd, ed*1e3);
    R(end+1) = struct('ngrid',NGRID, 'spot',DIA_LAMD, 'dimple_px',dia_px, ...
        'illum_px',illum_px, 'nyq',nyq, 'pokeg',gp, 'pokee',ep, ...
        'defg',gd, 'defe',ed);                                        %#ok<AGROW>
end

say_(rep, 'reference: IFO at NGRID 385 -- poke 0.984 / 49 pm, defocus 1.024 / 86 pm.\n');
say_(rep, 'sweep done in %.1f min\n', toc(t_all)/60);
fclose(rep);
save('zwfs_sweep.mat', 'R');

% summary figure: gain + error vs illuminated Nyquist margin, per spot
f = figure('Color','w', 'Position',[40 40 1200 460], 'Visible','off');
t = tiledlayout(f, 1, 2, 'Padding','compact', 'TileSpacing','compact');
spots = unique([R.spot]);  mk = {'o-','s-','^-'};
ax1 = nexttile(t);  hold(ax1,'on');  grid(ax1,'on');
ax2 = nexttile(t);  hold(ax2,'on');  grid(ax2,'on');
for i = 1:numel(spots)
    ss = [R.spot] == spots(i);
    [xs, si] = sort([R(ss).nyq]);  gg = [R(ss).pokeg];  gg = gg(si);
    ee = [R(ss).pokee]*1e3;  ee = ee(si);
    plot(ax1, xs, gg, mk{i}, 'LineWidth', 1.4, ...
         'DisplayName', sprintf('spot %.2f \\lambdaF/D', spots(i)));
    plot(ax2, xs, ee, mk{i}, 'LineWidth', 1.4, ...
         'DisplayName', sprintf('spot %.2f \\lambdaF/D', spots(i)));
end
hy = yline(ax1, 0.984, 'k--', 'IFO 0.984');  hy.Annotation.LegendInformation.IconDisplayStyle = 'off';
xlabel(ax1, 'detector Nyquist margin over the lit actuator scale');
ylabel(ax1, 'single-actuator fitted gain');
title(ax1, 'poke gain vs sampling');  legend(ax1, 'Location','southeast');
hy = yline(ax2, 49, 'k--', 'IFO 49 pm');  hy.Annotation.LegendInformation.IconDisplayStyle = 'off';
hy = yline(ax2, 1, 'r:', 'target 1 pm');  hy.Annotation.LegendInformation.IconDisplayStyle = 'off';
set(ax2, 'YScale', 'log');
xlabel(ax2, 'detector Nyquist margin over the lit actuator scale');
ylabel(ax2, 'poke estimate error (pm rms)');
title(ax2, 'poke error vs sampling');  legend(ax2, 'Location','northeast');
exportgraphics(f, 'zwfs_sweep.png', 'Resolution', 130);
close(f);
out = R;
fprintf('wrote zwfs_sweep_report.txt + zwfs_sweep.png + zwfs_sweep.mat\n');
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

function [g, e] = score_(Mtruth, hmeas, msk, mag, dxd_mm, xg, R_PAT, zoom_on_blob)
    % fitted gain + rms error vs ray-mapped truth (triptych scoring, no figure)
    N = size(hmeas,1);
    hm = hmeas*1e6;  hm(~msk) = NaN;
    [iy, ix] = find(msk);  cx = mean(ix);  cy = mean(iy);   %#ok<ASGLU>
    if zoom_on_blob
        w = abs(hm - median(hm(msk)));  w(~msk | isnan(hm)) = 0;
        wq = w;  wq(w < 0.5*max(w(:))) = 0;
        [cg, rg] = meshgrid(1:N, 1:N);
        cx = sum(cg(:).*wq(:))/sum(wq(:));  cy = sum(rg(:).*wq(:))/sum(wq(:));
    end
    [cg, rg] = meshgrid((1:N)-cx, (1:N)-cy);
    xdm = cg*dxd_mm*mag;  ydm = rg*dxd_mm*mag;
    Mt = Mtruth*1e6;
    tx0 = 0;  ty0 = 0;
    if zoom_on_blob
        [~, im] = max(abs(Mt(:)));  [tr, tc] = ind2sub(size(Mt), im);
        tx0 = xg(tc);  ty0 = xg(tr);
    end
    ht = interp2(xg, xg.', Mt, xdm+tx0, ydm+ty0, 'linear', 0);
    ht(~msk) = NaN;
    fitm = msk & (hypot(xdm,ydm) < 0.85*R_PAT) & ~isnan(ht) & ~isnan(hm);
    hm = hm - median(hm(fitm));  ht = ht - median(ht(fitm));
    g = ht(fitm) \ hm(fitm);
    e = sqrt(mean((hm(fitm) - ht(fitm)).^2));
end

function say_(rep, fmt, varargin)
    fprintf(fmt, varargin{:});
    fprintf(rep, fmt, varargin{:});
end
