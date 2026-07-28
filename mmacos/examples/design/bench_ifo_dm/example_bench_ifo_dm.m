% example_bench_ifo_dm.m
% ===================================================================
%  TWYMAN-GREEN IFO TESTBED, PHASE 1: MEASURE A DM (PROPER MODEL)
% ===================================================================
%  Follow-on to examples/design/bench_ifo (read that first): the same
%  compensated Twyman-Green rig -- now built in ONE call with
%  macos.design.twyman_green -- with the test optic replaced by a
%  DEFORMABLE MIRROR carrying random actuator pokes.  The DM surface
%  comes from the PROPER DM model (proper.prop_dm influence functions,
%  |command| < POKE_NM), is loaded into MACOS as a GridData surface in
%  the optic's own local frame, and the 7-frame de Groot PSI pipeline
%  recovers the surface map, which is then compared point-by-point
%  against the PROPER truth map.
%
%  PLANNED FOLLOW-ON PHASES (switches below, staged next):
%    DO_ABERR      small errors on every optic: <10 nm translations,
%                  <10 nrad rotations, <5 nm low-order Zernikes --
%                  common-path terms cancel in the two-arm difference;
%                  quantifies the rejection.
%    DO_NEARFIELD  Fresnel / near-field diffraction propagation on the
%                  recomb->detector leg (PropType change): the
%                  recomb-plane coherent-add injection is then the
%                  physically exact wave path.
%
%  Run:  matlab -batch "run('.../example_bench_ifo_dm.m')"  (MACOS_HOME set;
%  PROPER via the pymacos venv python)
% ===================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));
if isempty(exdir), exdir = pwd; end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);                       % GridFile= resolves relative to the cwd

MODEL = 256;
macos.init(MODEL);

% ---- parameters ----------------------------------------------------
LAM     = 6.328e-4;              % mm
NFRAME  = 7;
% DM (PROPER model):
NACT    = 16;                    % actuators across
SPACING = 3.5;                   % mm actuator pitch (56 mm DM vs 45 mm beam)
POKE_NM = 50;                    % |command| bound, nm (<100 nm spec; 50 keeps
                                 %  the double-pass phase inside +/-pi: no
                                 %  unwrapping -- see bench_ifo notes)
SEED    = 7;
PATTERN = 'checker';             % 'checker' (alternating +/- actuators,
                                 %  staggered by rows) or 'random'
N_G     = 256;                   % GridData sampling (model 256 => mGridMat 256)
DX_G    = 0.35;                  % mm/sample -> 89 mm grid span
PSI_MODE = 'detector';           % how the two arms combine:
                                 %  'detector' -- each arm traced
                                 %   INDEPENDENTLY to the shared
                                 %   detector plane, fields added there
                                 %   (the real-instrument reference; the
                                 %   measurement INCLUDES the detector-
                                 %   leg retrace / pupil-imaging error)
                                 %  'recomb' -- geometric coherent-add
                                 %   injection at the Recomb reference:
                                 %   the reference rides the TEST rays,
                                 %   so the shared tail phase cancels
                                 %   -- a tail-immune surface gauge
                                 %   that isolates the DM from the
                                 %   detector-leg imaging quality
DO_ABERR     = false;            % phase 2 (staged)
DO_NEARFIELD = false;            % phase 3 (staged)

% ---- 1. DM surface from PROPER -> GridFile -------------------------
PYBIN = fullfile(getenv('HOME'), 'dev/MACOS_resources/pymacos/.venv/bin/python');
gridfile = 'dm_map.txt';
cmdline = sprintf('%s dm_map_gen.py %s %d %g %d %g %g %d %s', ...
    PYBIN, gridfile, N_G, DX_G, NACT, SPACING, POKE_NM, SEED, PATTERN);
[st, msg] = system(cmdline);
assert(st == 0, 'PROPER DM generation failed: %s', msg);
fprintf('%s', msg);
Mdm = macos.read_grid_file(gridfile);          % engine GridMat convention, mm

% ---- 2. build the rig: POKED and BASELINE test arms ----------------
%  DIFFERENTIAL PROTOCOL (how a real surface gauge runs): the PSI
%  sequence is measured TWICE -- once with the DM at zero (baseline /
%  null) and once poked -- and the two wrapped phase maps are
%  subtracted in the complex domain.  Any static inter-arm structure
%  (including the large grid-block static term that would wrap the
%  single-shot map many times over) cancels IDENTICALLY; the remaining
%  DM-induced change is < +/-pi, so no unwrapping is needed anywhere.
zerofile = 'dm_zero.txt';
macos.write_grid_file(zerofile, zeros(N_G));
G  = macos.design.twyman_green('to_grid_file', gridfile, ...
        'to_grid_n', N_G, 'to_grid_dx', DX_G);
G0 = macos.design.twyman_green('to_grid_file', zerofile, ...
        'to_grid_n', N_G, 'to_grid_dx', DX_G);
rx_test = fullfile(exdir, 'ifo_dm_test_arm.in');   G.bt.emit(rx_test);
rx_base = fullfile(exdir, 'ifo_dm_base_arm.in');   G0.bt.emit(rx_base);
rx_ref  = fullfile(exdir, 'ifo_dm_ref_arm.in');    G.br.emit(rx_ref);
T = G.T;  R = G.R;

% beam radius at the DM (physical scale for the truth comparison)
macos.load_rx(rx_test);
s = macos.trace(T.iTO);  info = macos.get_ray_info(s.nRays);
ok = info.ok_trace(:) & info.ok_pass(:);
d = info.pos(:,ok) - info.pos(:,1);
dch = info.dir(:,1)/norm(info.dir(:,1));  d = d - dch*(dch.'*d);
r_beam = max(sqrt(sum(d.^2,1)));
fprintf('beam radius at DM: %.2f mm\n', r_beam);

% ---- 3. reference-arm PZT frames -----------------------------------
%  7 traces of the reference arm with the PZT stepped by lam/8 (double
%  pass -> exactly pi/2 of fringe phase per step).  The plane where the
%  reference field is taken depends on PSI_MODE (see above).
if strcmp(PSI_MODE, 'detector'), ref_plane = R.iDET; else, ref_plane = R.iRC; end
macos.load_rx(rx_ref);
pzt_psi = macos.get_elt_psi(R.iPZT);  pzt_vpt = macos.get_elt_vpt(R.iPZT);
Er = cell(1, NFRAME);
for k = 1:NFRAME
    macos.load_rx(rx_ref);
    dz = (k - (NFRAME+1)/2) * LAM/8;
    macos.set_elt_vpt(R.iPZT, pzt_vpt + dz*pzt_psi);
    Er{k} = macos.complex_field(ref_plane);
end
macos.load_rx(rx_test);
if strcmp(PSI_MODE, 'detector'), Et = macos.complex_field(T.iDET);
else,                            Et = macos.complex_field(T.iRC);  end
msk = abs(Et) > 0.1*max(abs(Et(:)));

% ---- 4. PSI twice (poked + baseline), complex-subtract -------------
[phi_pk, vis, I4] = psi_run(rx_test, T, Er, msk, NFRAME, PSI_MODE);
fprintf('fringe visibility (median in pupil): %.3f\n', median(vis));
[phi_b, ~]    = psi_run(rx_base, T, Er, msk, NFRAME, PSI_MODE);
fprintf('static (baseline) wrapped-phase rms in pupil: %.3g rad\n', ...
    std(phi_b(msk)));
phi = angle(exp(1i*(phi_pk - phi_b)));  % differential: statics cancel
%  SIGN: a surface bump of height h toward the beam shortens the
%  double-passed path by 2h; in the engine's field-phase convention
%  (phase advances as path SHORTENS -- see psi_run) that is
%  +4*pi*h/lam, so surface height carries the + sign:
h = phi * LAM/(4*pi);                  % surface height, mm

% ---- 4b. METROLOGY-CHAIN FLOOR: PSI vs direct field phase ----------
%  Compare the PSI-recovered differential phase against the engine's
%  own differential field phase at the same detector pixels -- no
%  resampling, no grids.  In 'detector' mode this is the pure
%  algorithm error (PZT stepping + deGroot + differential) and lands
%  at MACHINE PRECISION, ~3e-5 pm rms.  In 'recomb' mode it instead
%  measures the detector-leg RETRACE TERM (~9 nm rms at 50 nm checker
%  pokes, rim-concentrated, linear in poke amplitude): the injected
%  reference rides the test rays, so the poked DM changes the tail
%  phase it accumulates, whereas the direct field difference keeps
%  that term.  A physical instrument's independent reference DOES see
%  the retrace term -- which is exactly the pupil-image-quality cost
%  of the singlet L2 leg (candidate fix: better L2 / doublet / OAP
%  relay in the Bench roadmap).
macos.load_rx(rx_test);  Ed_pk = macos.complex_field(T.iDET);
macos.load_rx(rx_base);  Ed_b  = macos.complex_field(T.iDET);
phi_dir = angle(Ed_pk .* conj(Ed_b));
dphi = angle(exp(1i*(phi - phi_dir)));
fprintf('PSI vs direct differential field phase (%s mode): %.4g pm rms\n', ...
    PSI_MODE, 1e9 * std(dphi(msk)) * LAM/(4*pi));

% ---- 5a. PIPELINE truth: engine ray-traced OPD at the detector -----
%  The engine's own geometric OPD (differential, poked minus baseline,
%  /2 for double pass) is what the optics actually deliver to the
%  detector: comparing the PSI recovery against it validates the
%  INTERFEROMETER + PROCESSING in isolation, through the same
%  pupil-mapping distortion.  The PROPER-map comparison below is the
%  separate METROLOGY question (recovered map vs physical DM), which
%  additionally needs the pixel->DM-position mapping -- i.e. the
%  PUPIL IMAGE QUALITY of the detector leg (next work item).
macos.load_rx(rx_test);  macos.trace(T.iDET);  opd_pk = macos.opd();
macos.load_rx(rx_base);  macos.trace(T.iDET);  opd_b  = macos.opd();
h_opd = (opd_pk - opd_b)/2;                    % surface-equivalent, mm
                                               % (macos OPD sign matches the
                                               % field-phase convention above)
fprintf('array sizes: OPDMat %dx%d, WF %dx%d\n', size(h_opd), size(h));
%  The OPD raster is SOURCE-GRID sized (e.g. 63x63 across the beam) while
%  the WF array is the padded diffraction grid (beam occupies only the
%  central patch) -- register the two on their BEAM DISCS, not the array
%  frames, with the axis convention resolved empirically (and printed).
sup = abs(opd_pk) > 0;                         % OPD-raster beam support
[io, jo] = find(sup);
co_j = mean(jo);  co_i = mean(io);
ro = max(sqrt((jo-co_j).^2 + (io-co_i).^2));
N = size(h,1);
[colg, rowg] = meshgrid(1:N, 1:N);
cxm = sum(colg(msk))/nnz(msk);  cym = sum(rowg(msk))/nnz(msk);
rm = max(sqrt((colg(msk)-cxm).^2 + (rowg(msk)-cym).^2));
a1 = (colg - cxm)*ro/rm;  a2 = (rowg - cym)*ro/rm;   % WF pixel -> OPD-raster px
hm0 = h - mean(h(msk));
oc = {a1,a2; a1,-a2; -a1,a2; -a1,-a2; a2,a1; a2,-a1; -a2,a1; -a2,-a1};
onames = {'+c+r','+c-r','-c+r','-c-r','+r+c','+r-c','-r+c','-r-c'};
bo = struct('c', 0, 'i', 1);
for c = 1:size(oc,1)
    ho = interpn(1:size(h_opd,1), 1:size(h_opd,2), h_opd, ...
                 co_i + oc{c,2}, co_j + oc{c,1}, 'linear', 0);
    ho = ho - mean(ho(msk));
    cc = corrcoef(hm0(msk), ho(msk));  cc = cc(1,2);
    if abs(cc) > abs(bo.c), bo = struct('c',cc, 'ho',ho, 'name',onames{c}, 'i',c); end
end
%  Sub-pixel + scale refine (the disc radius estimated from the raster
%  support is quantized to a raster pixel; the centroids to a fraction).
oco = @(p) -abs(opd_reg_corr(p, oc{bo.i,1}, oc{bo.i,2}, co_i, co_j, h_opd, hm0, msk));
p_o = fminsearch(oco, [0 0 0], optimset('TolX',1e-5,'Display','off'));
[cc2, ho2] = opd_reg_corr(p_o, oc{bo.i,1}, oc{bo.i,2}, co_i, co_j, h_opd, hm0, msk);
if abs(cc2) > abs(bo.c), bo.c = cc2;  bo.ho = ho2; end
out_pipeline_corr = bo.c;
out_pipeline_resid_nm = 1e6*std(hm0(msk) - sign(bo.c)*bo.ho(msk));
fprintf('PSI vs ray-traced OPD truth (disc-registered, axes %s): corr %.6f, residual %.3f nm rms\n', ...
    bo.name, out_pipeline_corr, out_pipeline_resid_nm);

% ---- 5b. PUPIL IMAGE QUALITY: empirical per-ray DM->detector map ---
%  The detector images the DM pupil through L2 + the folded tilted-
%  plate train -- that image is NOT an ideal (linear, radial) copy:
%  it carries magnification, anamorphic stretch, SHEAR, and nonlinear
%  distortion.  Measure it from the trace itself: every surviving ray
%  gives one (DM-plane position) <-> (detector-plane position) pair.
%  A scatteredInterpolant over those pairs IS the instrument's pupil
%  mapping; its affine part is the distortion REPORT, and the full map
%  resamples the PROPER truth onto the WF pixels, so the metrology
%  comparison no longer assumes an ideal pupil image.
macos.load_rx(rx_test);
s  = macos.trace(T.iTO);   ito  = macos.get_ray_info(s.nRays);
s2 = macos.trace(T.iDET);  idet = macos.get_ray_info(s2.nRays);
okr = ito.ok_trace(:) & ito.ok_pass(:) & idet.ok_trace(:) & idet.ok_pass(:);
% DM-plane ray coords in the GRID frame the Rx declares (pData = vpt,
% xData = perp(psi), yData = psi x xData) -- the frame Mdm lives in
psi1 = macos.get_elt_psi(T.iTO);  vpt1 = macos.get_elt_vpt(T.iTO);
u1 = macos.design.Bench.perp(psi1);  v1 = cross(psi1, u1);
xy_to = [u1.'; v1.'] * (ito.pos - vpt1);           % 2 x nRays, mm
% detector-plane ray coords about the chief (ray 1)
psi2 = macos.get_elt_psi(T.iDET);
u2 = macos.design.Bench.perp(psi2);  v2 = cross(psi2, u2);
xy_d = [u2.'; v2.'] * (idet.pos - idet.pos(:,1));
xy_to = xy_to(:,okr);  xy_d = xy_d(:,okr);
% affine part = the classical distortion numbers
Aaf = [xy_d.' ones(nnz(okr),1)] \ xy_to.';         % [xt yt] = [xd yd 1]*Aaf
Lm  = Aaf(1:2,:).';                                % 2x2 linear part
[Us,Ss,Vs] = svd(Lm);
smag = diag(Ss);                                   % principal stretches
Rrot = Us*Vs.';                                    % polar rotation part
xy_fit = Lm*xy_d + Aaf(3,:).';
nl = xy_to - xy_fit;                               % nonlinear distortion, mm
nl_rms = sqrt(mean(sum(nl.^2,1)));
fprintf('\n=== pupil image quality (detector -> DM, per-ray) ===\n');
fprintf('  magnification %.4f, anamorphic stretch %.3f%%, rotation %.3f mrad\n', ...
    sqrt(abs(det(Lm))), 100*(smag(1)/smag(2)-1), 1e3*atan2(Rrot(2,1),Rrot(1,1)));
fprintf('  shear axis %.1f deg; nonlinear distortion %.4f mm rms (%.2f%% of beam)\n', ...
    atan2d(Vs(2,1),Vs(1,1)), nl_rms, 100*nl_rms/r_beam);
Fx = scatteredInterpolant(xy_d(1,:).', xy_d(2,:).', xy_to(1,:).', 'linear', 'linear');
Fy = scatteredInterpolant(xy_d(1,:).', xy_d(2,:).', xy_to(2,:).', 'linear', 'linear');

%  WF pixel centers -> detector mm (pitch from dx_at; center = beam
%  centroid matched between mask and rays).  The discrete row/col-vs-
%  (u2,v2) axis convention of the WF array is resolved empirically over
%  the 8 candidates and PRINTED -- the classic grid-orientation trap;
%  the continuous part (shear etc.) is in the ray map, not scanned.
N = size(h,1);
[colg, rowg] = meshgrid(1:N, 1:N);
cx = sum(colg(msk))/nnz(msk);  cy = sum(rowg(msk))/nnz(msk);
dxp = macos.dx_at(T.iDET, 'mm');
a1 = (colg - cx)*dxp;  a2 = (rowg - cy)*dxp;
c_d = mean(xy_d, 2);                               % ray-side beam centroid
ax = ((1:N_G) - (N_G+1)/2)*DX_G;
hm = h - mean(h(msk));
cands = {a1,a2,'x=+col,y=+row'; a1,-a2,'x=+col,y=-row'; ...
         -a1,a2,'x=-col,y=+row'; -a1,-a2,'x=-col,y=-row'; ...
         a2,a1,'x=+row,y=+col'; a2,-a1,'x=+row,y=-col'; ...
         -a2,a1,'x=-row,y=+col'; -a2,-a1,'x=-row,y=-col'};
best = struct('c', -inf, 'i', 1);
for c = 1:size(cands,1)
    Xt = Fx(cands{c,1}+c_d(1), cands{c,2}+c_d(2));  % pixel -> det -> DM mm
    Yt = Fy(cands{c,1}+c_d(1), cands{c,2}+c_d(2));
    ht = interpn(ax, ax, Mdm, Xt, Yt, 'linear', 0);
    ht = ht - mean(ht(msk));
    cc = corrcoef(hm(msk), ht(msk));  cc = cc(1,2);
    if cc > best.c, best = struct('c',cc, 'ht',ht, 'name',cands{c,3}, 'i',c); end
end
%  Registration refine: the mask-vs-ray centroid match can be off by a
%  fraction of a WF pixel, and dx_at / the affine fit by parts in 1e-4
%  -- invisible as distortion but costly at the checker frequency.
%  Refine offset + rotation + scale by direct correlation maximization
%  (these four numbers ARE the instrument calibration).
A1 = cands{best.i,1};  A2 = cands{best.i,2};
regc = @(p) -reg_corr(p, A1, A2, c_d, Fx, Fy, ax, Mdm, hm, msk);
p_reg = fminsearch(regc, [0 0 0 0], optimset('TolX',1e-7,'TolFun',1e-10,'Display','off'));
[c_ref, ht_ref] = reg_corr(p_reg, A1, A2, c_d, Fx, Fy, ax, Mdm, hm, msk);
fprintf('  registration refine: offset [%.4f %.4f] mm, rot %.4f mrad, scale %+0.2e\n', ...
    p_reg(1), p_reg(2), 1e3*p_reg(3), expm1(p_reg(4)));
fprintf('  -> corr %.8f (was %.6f)\n', c_ref, best.c);
if c_ref > best.c, best.c = c_ref;  best.ht = ht_ref; end
res = hm(msk) - best.ht(msk);
fprintf('\n=== DM recovery (deGroot-7 vs PROPER truth through the ray map) ===\n');
fprintf('  pixel axes: %s   correlation %.6f\n', best.name, best.c);
fprintf('  truth rms %.2f nm; recovered rms %.2f nm; residual %.3f nm rms\n', ...
    1e6*std(best.ht(msk)), 1e6*std(hm(msk)), 1e6*std(res));

out = struct('visibility', median(vis), 'corr', best.c, ...
    'orientation', best.name, 'resid_nm', 1e6*std(res), ...
    'truth_rms_nm', 1e6*std(best.ht(msk)), 'seed', SEED, ...
    'nact', NACT, 'poke_nm', POKE_NM, ...
    'pipeline_corr', out_pipeline_corr, ...
    'pipeline_resid_nm', out_pipeline_resid_nm, ...
    'map_mag', sqrt(abs(det(Lm))), ...
    'map_anam_pct', 100*(smag(1)/smag(2)-1), ...
    'map_rot_mrad', 1e3*atan2(Rrot(2,1),Rrot(1,1)), ...
    'map_nonlin_rms_mm', nl_rms);

% ---- 6. figures ----------------------------------------------------
f1 = figure('Color','w','Position',[100 100 1800 460]);
subplot(1,4,1); imagesc(I4); axis image; colorbar; title('interferogram (\delta=0)');
q = nan(N); q(msk) = 1e6*hm(msk);
subplot(1,4,2); imagesc(q,'AlphaData',~isnan(q)); axis image; colorbar;
title('recovered DM surface (nm)');
q = nan(N); q(msk) = 1e6*best.ht(msk);
subplot(1,4,3); imagesc(q,'AlphaData',~isnan(q)); axis image; colorbar;
title('PROPER truth (nm)');
q = nan(N); q(msk) = 1e6*(hm(msk)-best.ht(msk));
subplot(1,4,4); imagesc(q,'AlphaData',~isnan(q)); axis image; colorbar;
title(sprintf('residual %.2f nm rms', 1e6*std(res)));
print(f1, fullfile(exdir,'ifo_dm_recovery.png'), '-dpng', '-r150');

%  Distortion report figure: the measured pupil mapping.  Left: where
%  each ray's DM position lands vs the best-fit affine image of it
%  (arrows = nonlinear distortion, exaggerated).  Right: shear/anam
%  visualized by mapping a square DM grid to the detector.
f2 = figure('Color','w','Position',[100 100 1100 520]);
subplot(1,2,1);
EX = 200;                                          % arrow exaggeration
quiver(xy_fit(1,:), xy_fit(2,:), EX*nl(1,:), EX*nl(2,:), 0, 'b'); axis equal; grid on;
title(sprintf('nonlinear distortion (x%d): %.4f mm rms = %.2f%% of beam', ...
    EX, nl_rms, 100*nl_rms/r_beam));
xlabel('DM-plane x (mm)'); ylabel('DM-plane y (mm)');
subplot(1,2,2);
g = linspace(-r_beam, r_beam, 9);  [Gx, Gy] = meshgrid(g, g);
Li = inv(Lm);                                      % DM -> detector direction
Dg = Li*([Gx(:) Gy(:)].' - Aaf(3,:).');
plot(Gx, Gy, 'Color', [0.8 0.8 0.8]); hold on; plot(Gx.', Gy.', 'Color', [0.8 0.8 0.8]);
plot(reshape(Dg(1,:),9,9),  reshape(Dg(2,:),9,9),  'r');
plot(reshape(Dg(1,:),9,9).', reshape(Dg(2,:),9,9).', 'r');
axis equal; grid on;
title(sprintf('affine pupil image: mag %.4f, anam %.3f%%, rot %.3f mrad', ...
    sqrt(abs(det(Lm))), 100*(smag(1)/smag(2)-1), 1e3*atan2(Rrot(2,1),Rrot(1,1))));
xlabel('mm'); legend({'DM grid','its detector image'}, 'Location','best');
print(f2, fullfile(exdir,'ifo_dm_pupil_map.png'), '-dpng', '-r150');

%  THE SURFACE-GAUGE PRODUCT: the recovered map resampled INTO the DM
%  plane -- pupil magnification (measured 0.810 det->DM by the ray map)
%  and image inversion CALIBRATED OUT, axes in mm ON THE DM.  This is
%  the deliverable a real instrument reports; on a bench the same
%  calibration comes from a fiducial (aperture edge / known poke) --
%  here the ray trace is the fiducial.  Shown against the PROPER truth
%  on identical axes, so scale/orientation differences read directly.
[~, ~, Xt, Yt] = reg_corr(p_reg, A1, A2, c_d, Fx, Fy, ax, Mdm, hm, msk);
Fh = scatteredInterpolant(Xt(msk), Yt(msk), hm(msk), 'natural', 'none');
ax_sub = ax(abs(ax) <= 1.08*r_beam);
[GX, GY] = meshgrid(ax_sub, ax_sub);
h_dm = Fh(GX, GY);                              % measured, DM-plane mm
t_dm = interpn(ax, ax, Mdm, GX, GY, 'linear', nan);
t_dm(isnan(h_dm)) = nan;  t_dm = t_dm - mean(t_dm(~isnan(t_dm)));
f3 = figure('Color','w','Position',[100 100 1500 520]);
ttl = {sprintf('measured DM surface, DM plane (mag %.3f calibrated)', ...
       sqrt(abs(det(Lm)))), 'PROPER truth (DM plane)', 'difference'};
Z = {h_dm, t_dm, h_dm - t_dm};
for c = 1:3
    subplot(1,3,c); imagesc(ax_sub, ax_sub, 1e6*Z{c}, 'AlphaData', ~isnan(Z{c}));
    axis image; axis xy; colorbar; title(ttl{c});
    xlabel('DM x (mm)'); ylabel('DM y (mm)');
end
print(f3, fullfile(exdir,'ifo_dm_surface_dmplane.png'), '-dpng', '-r150');

save(fullfile(exdir,'bench_ifo_dm.mat'), 'out', 'h', 'hm', 'msk', 'best', ...
    'xy_to', 'xy_d', 'Aaf', 'nl', 'r_beam', 'dxp', 'c_d', 'h_dm', 't_dm', 'ax_sub');
fprintf('\nDONE (phase 1).  DO_ABERR=%d DO_NEARFIELD=%d staged for next phases.\n', ...
    DO_ABERR, DO_NEARFIELD);

% ===================================================================
%  LOCAL FUNCTIONS
% ===================================================================
function [phi, vis, I4] = psi_run(rx, T, Er, msk, NFRAME, mode)
%PSI_RUN  One full PSI sequence on test-arm RX: 7 interference frames
%   + the de Groot windowed 7-frame estimator.  Returns the WRAPPED
%   phase map and per-pixel visibility.
%
%   mode 'detector': the frames are |Etd + Er_k|^2 with Etd the test
%   arm traced independently to the detector and Er_k the reference
%   frames AT the detector -- the physical two-beam instrument.
%   mode 'recomb': the reference is injected at the Recomb plane via
%   macos.apodize_complex (M = 1 + Er/Et) and carried to the detector
%   by the test-arm trace -- the shared tail phase cancels in the
%   fringe, making this variant immune to the detector-leg imaging
%   (see the section-4b discussion).
    LAM = 6.328e-4;
    I = cell(1, NFRAME);
    if strcmp(mode, 'detector')
        macos.load_rx(rx);
        Etd = macos.complex_field(T.iDET);
        for k = 1:NFRAME
            I{k} = abs(Etd + Er{k}).^2;
        end
    else
        for k = 1:NFRAME
            macos.load_rx(rx);
            Etk = macos.complex_field(T.iRC);
            M = ones(size(Etk));
            M(msk) = 1 + Er{k}(msk) ./ Etk(msk);
            macos.apodize_complex(T.iRC, M);
            Ed = macos.complex_field(T.iDET, 'reset_trace', false);
            I{k} = abs(Ed).^2;
        end
    end
    %  STEP SIGN (empirical, engine-verified): moving the retro PZT by
    %  +dz along its psi shortens the double-passed path by 2*dz, and
    %  the ENGINE's complex-field phase ADVANCES by +4*pi*dz/lam for
    %  that move (frame-ratio diagnostic: Er_k/Er_1 = exp(+i(k-1)pi/2)
    %  to 1e-4 pm) -- i.e. macos field phase runs OPPOSITE to optical
    %  path length, the same convention the pymacos<->PROPER work
    %  reconciles with opd_sign_flip.  With th_k = +(k-4)*pi/2 the
    %  estimator returns exactly the engine field-phase difference.
    w  = [1 2 3 4 3 2 1];  th = ((1:7)-4)*pi/2;
    num = zeros(size(I{1}));  den = num;
    for k = 1:7
        num = num + w(k)*I{k}*sin(th(k));  den = den + w(k)*I{k}*cos(th(k));
    end
    phi = atan2(-num, den);
    Imax = max(cat(3,I{:}),[],3);  Imin = min(cat(3,I{:}),[],3);
    vis = (Imax(msk)-Imin(msk)) ./ max(Imax(msk)+Imin(msk), eps);
    I4 = I{4};
end %#ok<DEFNU>

function [c, ht, Xt, Yt] = reg_corr(p, A1, A2, c_d, Fx, Fy, ax, Mdm, hm, msk)
%REG_CORR  Truth-vs-recovery correlation with a similarity adjustment
%   P = [dx dy rot log_scale] applied to the pixel->detector
%   coordinates, truth resampled through the empirical ray map with
%   SPLINE interpolation (bilinear costs 100s of pm at actuator-scale
%   structure).  Used by the registration refine in section 5b.
    s = exp(p(4));  ct = cos(p(3));  st = sin(p(3));
    X = s*(ct*A1 - st*A2) + c_d(1) + p(1);
    Y = s*(st*A1 + ct*A2) + c_d(2) + p(2);
    Xt = Fx(X, Y);  Yt = Fy(X, Y);
    ht = interpn(ax, ax, Mdm, Xt, Yt, 'spline', 0);
    ht = ht - mean(ht(msk));
    cc = corrcoef(hm(msk), ht(msk));
    c = cc(1,2);
end

function [c, ho] = opd_reg_corr(p, A1, A2, co_i, co_j, h_opd, hm0, msk)
%OPD_REG_CORR  Same idea for the 5a pipeline check: offset + scale on
%   the WF-pixel -> OPD-raster coordinates, spline resampling.
    s = exp(p(3));
    ho = interpn(1:size(h_opd,1), 1:size(h_opd,2), h_opd, ...
                 co_i + s*A2 + p(2), co_j + s*A1 + p(1), 'spline', 0);
    ho = ho - mean(ho(msk));
    cc = corrcoef(hm0(msk), ho(msk));
    c = cc(1,2);
end

