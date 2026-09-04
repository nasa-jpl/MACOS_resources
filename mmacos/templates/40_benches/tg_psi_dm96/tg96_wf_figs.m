function tg96_wf_figs()
%TG96_WF_FIGS  Deck figures: traced-rig render + wavefront-estimate
%   triptychs (applied / sensed / error) for the two example classes
%   (Dave 2026-09-04): a single-actuator poke (the case we care about)
%   and a low-order aberration (defocus -- the class that troubles the
%   Zernike sensor; the IFO should read it cleanly).  Cases and
%   amplitudes match zwfs_dm96/zwfs_wf_figs.m so the two decks compare
%   directly: center actuator at 20 nm; defocus 8 nm amplitude.
%   Writes: tg96_render_rig.png, tg96_poke_triptych.png,
%           tg96_defocus_triptych.png
%   Run:  cd <this dir>;  matlab -batch "tg96_wf_figs"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));  % dm_influence_map

% run-10 configuration (== tg96.m)
s = 96/56;  LAM = 6.328e-4;  QWP = 0.25;  THETAS = [0 45 90 135];
MODEL = 1024;  NGRID = 385;  N_G = 384;  DX_G = 0.28;
NACT = 96;  PITCH = 1.0;  AOI = 7;  D_BS_TO = 700;  R_BEAM = s*30;
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

% ---- fig 1: traced-rig render (demo beat-2 recipe) -----------------
load_arm(AT, QWP, 0, []);
macos.trace(AT.iDET);
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
    camva(ax, 'auto');  camzoom(ax, 3.5);  axis(ax, 'off');
end
title(tl2, 'Shallow-plate polarization-PSI Twyman-Green at 96 mm, test arm');
print(f, 'tg96_render_rig.png', '-dpng', '-r150');
close(f);
fprintf('wrote tg96_render_rig.png\n');

% ---- measurement bases ---------------------------------------------
Sr = analyzer_basis(AR, QWP, []);
S0 = analyzer_basis(AT, QWP, []);
I0 = frame(S0, Sr, 0);  msk = I0 > 0.1*max(I0(:));
p_null = fourstep(S0, Sr, THETAS);

% ray frame: DM mm per detector mm + detector px scale
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
fprintf('ray frame: mag %.4f, det px -> DM mm scale %.5f\n', mag, dxd_mm*mag);

% ---- the two cases (== zwfs_wf_figs) -------------------------------
% Pattern radius = the ILLUMINATED beam radius (see zwfs_wf_figs note).
[syy, sxx] = find(msk);
scx = mean(sxx);  scy = mean(syy);
rpx = hypot(sxx-scx, syy-scy);
rs = sort(rpx);  R_PAT = rs(round(0.98*numel(rs))) * dxd_mm * mag;
fprintf('pattern radius R_PAT = %.2f mm (aperture %.2f)\n', R_PAT, R_BEAM);
% Measurement sign: -1, the tg96 two-poke registration's measured
% meas-sign on this deck (tg96_report.txt "parity 5 ... meas sign -1").
SGN_IFO = -1;

Ap = zeros(NACT);  Ap(48,48) = 1;
Mp = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', 20e-6*Ap);
hp = meas_surface(AT, QWP, Mp, Sr, p_null, THETAS, LAM);

xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;
[gx, gy] = meshgrid(xg, xg);  rr = hypot(gx, gy);
Md = 8e-6*(2*(rr/R_PAT).^2 - 1).*double(rr <= R_PAT);
hd = meas_surface(AT, QWP, Md, Sr, p_null, THETAS, LAM);

triptych_wf('tg96_poke_triptych.png', Mp, hp, SGN_IFO, msk, mag, dxd_mm, xg, R_PAT, ...
    'IFO: one actuator pushed 20 nm', true, 6);
triptych_wf('tg96_defocus_triptych.png', Md, hd, SGN_IFO, msk, mag, dxd_mm, xg, R_PAT, ...
    'IFO: defocus, 8 nm amplitude', false, 10);
fprintf('done\n');
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

% ==== measurement helpers, copied verbatim from tg96.m ===============
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
