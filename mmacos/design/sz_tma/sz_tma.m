% sz_tma.m
% =====================================================================
%  SPHERE + ZERNIKE unobscured three-mirror telescope (e5mono-derived)
% =====================================================================
%  A wide-field design strategy where GEOMETRY and CORRECTION fully
%  decouple:
%    * 0th-order layout = three BASE SPHERES (Kc=0).  The sphere radii
%      set first-order power (f/#, EFL); the fold tilts unobscure the
%      beam and keep it compact.  No conics -> none of the conic/
%      telecentric coupling that bogs down a Korsch search.
%    * CORRECTION = Zernike departures on each mirror, optimized by
%      MACOS's native multi-field design optimizer (CALIB OptZern).
%      Because the departures are tiny sag perturbations they don't move
%      the chief-ray geometry, so once the layout meets packaging the
%      optimizer cannot break it.
%
%  GEOMETRY (M1, M2 lifted from e5mono; M3 designed to replace its lens
%  relay).  M2 is a CONVEX secondary that slows -- but does not reverse
%  -- the M1 cone, forming a real f/21 intermediate image ~25 m past M2.
%  M3 sits 3 m past that image (a real focus BETWEEN M2 and M3 -- the
%  metrology-injection point) and reimages it to a focal plane folded
%  out of the beam.
%
%  THE METHOD (Bauer/Schiesser/Rolland 2018 + e5mono): leave the
%  rotationally-invariant 3rd-order aberration UNCORRECTED in the
%  spheres, then optimize the Zernikes in an order that meets
%  intermediate objectives -- center first, then widen the field:
%    [2] S0  on-axis      -> diffraction-limit the center
%    [3] S1  to FOV/2     -> hold the inner field
%    [4] S2  to full FOV  -> hold the full field
%
%  Run interactively:  >> run('.../design/sz_tma/sz_tma.m')
%  (the design + figures land on disk; MATLAB stays open)
% =====================================================================

addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% ====================  KNOBS  ========================================
D          = 8.0;        % aperture diameter (m)  -- e5mono
LAM        = 1e-6;       % wavelength (m)         -- 1 um
MODEL      = 256;        % engine model size
GRIDN      = 41;         % circular ray-grid points (-> ~1300 rays)
FOV_ARCMIN = 2.0;        % design HALF-field (arcmin)
DIFFRACTION_LIMIT = 0.07;% "well-corrected" RMS WFE (waves)

% --- 0th-order all-sphere layout (|radii| m, spacings m, fold tilts deg)
R    = [51.534, 8.871, 3.0];   % M1 concave, M2 convex, M3 concave
TBET = [22.0,   28.0];         % M1->M2, M2->M3 along the folded chief ray
TILT = [-7.2,   8.46,  12.0];  % fold about x (M1/M2 reproduce e5mono)

% --- correction: e5mono's BornWolf Zernike set (piston dropped) ------
MODES = [3 4 5 9 10 11 12 13 19 20 21 22 23 24 25];
ZTYPE = 'BornWolf';
ITERS = 200;
% =====================================================================

fprintf('====================================================================\n');
fprintf(' Sphere+Zernike unobscured TMA | D=%.1f m | lambda=%.2g um | e5mono-derived\n', D, LAM*1e6);
fprintf('====================================================================\n');

%% -- build the all-sphere 0th-order layout ---------------------------
sz = macos.design.Telescope('family','TMA','aperture_diameter_m',D, ...
        'model_size',MODEL,'wavelength_m',LAM,'grid_npts',GRIDN);
sz.set_base_sphere(true);                            % hold base spheres (Kc=0)
sz.add_mirror('M1','radius_m',R(1),'spacing_after_m',TBET(1),'tilt_deg',TILT(1));
sz.add_mirror('M2','radius_m',R(2),'spacing_after_m',TBET(2),'tilt_deg',TILT(2),'convex',true);
sz.add_mirror('M3','radius_m',R(3),'spacing_after','derive', 'tilt_deg',TILT(3));
sz.add_focal_plane('FP');
sz.build();  sz.describe();
nE = numel(sz.spec.elt);
d  = sz.spec.derived;
fprintf('\n  layout: EFL=%.1f m (f/%.1f), intermediate focus ~%.0f m past M2,\n', ...
        d.EFL, d.fnum, TBET(2)-R(3));
fprintf('          M2->M3 = %.0f m leaves a real focus between M2 and M3 (met injection)\n', TBET(2));

%% -- [1] all-sphere baseline (uncorrected) ---------------------------
macos.trace(nE);  wfe0 = rms_waves(macos.opd(), LAM);
fprintf('\n[1] all-sphere baseline (Kc=0, no Zernikes): RMS WFE = %.0f waves (uncorrected)\n', wfe0);

%% -- [2] S0 -- on-axis: diffraction-limit the center -----------------
r0 = sz.optimize_freeform([1 2 3],'modes',MODES,'type',ZTYPE,'fields_arcmin',[],'max_iters',ITERS);
wfe_c = r0.wfe_after/LAM;
fprintf('[2] S0 center      : %.0f -> %.4f waves -> %s\n', r0.wfe_before/LAM, wfe_c, ...
        ternary(wfe_c<DIFFRACTION_LIMIT,'DIFFRACTION-LIMITED','residual'));

% 2-D field samples spanning +-FOV in X AND Y at the FP.  The system is
% mirror-symmetric about the y-z (fold) plane, so we sample thx>=0 only and
% AREA-WEIGHT each thx>0 point x2 (it stands for its +-thx mirror image) --
% a true area-weighted merit over the square FOV without doubling the traces.
amin_ = @(A) deg2rad(A/60);                           % arcmin -> rad
h  = FOV_ARCMIN;
F1 = (h/2)*[1 0; 0 1; 0 -1; 1 1; 1 -1];               % inner ring (~h/2)
F2 =  h   *[1 0; 0 1; 0 -1; 1 1; 1 -1];               % outer ring (~h = FOV)
Wt = @(Fam) [1, 1 + (Fam(:,1).' > 0)];                % on-axis 1; thx=0 ->1; thx>0 ->2

%% -- [3] S1 -- hold the inner 2-D field (area-weighted) --------------
r1 = sz.optimize_freeform([1 2 3],'modes',MODES,'type',ZTYPE, ...
                          'fields',amin_(F1),'weights',Wt(F1),'max_iters',ITERS);
fprintf('[3] S1 inner 2-D    : worst %.3f -> %.4f waves\n', ...
        max(r1.wfe_before)/LAM, max(r1.wfe_after)/LAM);

%% -- [4] S2 -- hold the full +-FOV 2-D field (X and Y) ---------------
F = [F1; F2];  w = Wt(F);
r2 = sz.optimize_freeform([1 2 3],'modes',MODES,'type',ZTYPE, ...
                          'fields',amin_(F),'weights',w,'max_iters',ITERS);
wfe   = r2.wfe_after(:).'/LAM;                         % per-field, waves
wfe_f = max(wfe);                                      % worst field
wfe_aw = sqrt(sum(w.*wfe.^2)/sum(w));                  % area-weighted RMS over FOV
fprintf('[4] S2 full 2-D +-%g'' : worst %.4f, area-wtd %.4f waves -> %s\n', ...
        FOV_ARCMIN, wfe_f, wfe_aw, ...
        ternary(wfe_f<DIFFRACTION_LIMIT,'DIFFRACTION-LIMITED','residual'));

%% -- [5] field scan: WFE map + clear apertures -----------------------
fprintf('\n[5] field scan over +-%g'' -> WFE map + apertures\n', FOV_ARCMIN);
scan = sz.realize_apertures('fields', ...
        macos.design.field_grid(FOV_ARCMIN,7,'units','arcmin'), ...
        'margin',0.05,'quiet',true);

%% -- [6] verify UNOBSCURED (+ FP out of beam) ------------------------
rep = sz.check_clipping('noload',true,'quiet',true);
fprintf('[6] clearance: %d/%d optics clear -> %s\n', sum([rep.ok]), numel(rep), ...
        ternary(all([rep.ok]),'UNOBSCURED','** OBSCURED (see check_clipping)'));

%% -- [7] exit pupil + save deliverable -------------------------------
sz.add_pupil(numel(sz.spec.elt));
rxfile  = fullfile(exdir,'sz_tma.in');
matfile = fullfile(exdir,'sz_tma.mat');
sz.save(rxfile);  sz.save_spec(matfile);
fprintf('[7] saved: %s\n           + %s\n', rxfile, matfile);

%% -- [8] design-report figures --------------------------------------
try
    f1 = sz.view_field_map(scan,'kind','contour');
    p1 = fullfile(exdir,'sz_tma_wfe.png');  saveas(f1,p1);
    fprintf('[8] WFE field map: %s\n', p1);
catch ME, fprintf('[8] WFE map skipped (%s)\n', ME.message); end
try
    f2 = sz.view_orthoviews({'YZ','XZ'},'nrays',9);
    p2 = fullfile(exdir,'sz_tma_layout.png');  saveas(f2,p2);
    fprintf('    orthographic layout: %s\n', p2);
catch ME, fprintf('    layout skipped (%s)\n', ME.message); end

%% -- summary ---------------------------------------------------------
fprintf('\n--------------------------------------------------------------------\n');
fprintf(' D=%.1f m | f/%.1f | %d spheres + Zernike departures (%d modes/mirror)\n', ...
        D, d.fnum, 3, numel(MODES));
fprintf(' RMS WFE: baseline %.0f -> center %.4f -> 2-D +-%g'' worst %.4f / area-wtd %.4f waves\n', ...
        wfe0, wfe_c, FOV_ARCMIN, wfe_f, wfe_aw);
fprintf(' real intermediate focus between M2 and M3 (metrology injection); FP out of beam\n');
fprintf('====================================================================\n');

% ---- helpers --------------------------------------------------------
function w = rms_waves(W, lam)
    v = W(isfinite(W) & W ~= 0);
    if isempty(v), w = NaN; else, w = std(v)/lam; end
end
function s = ternary(c,a,b), if c, s = a; else, s = b; end, end
