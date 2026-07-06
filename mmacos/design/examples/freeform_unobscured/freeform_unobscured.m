% freeform_unobscured.m  (mmacos/design/examples/freeform_unobscured/)
% =====================================================================
%  SPHERE + ZERNIKE (freeform) UNOBSCURED TELESCOPE -- the visible-band
%  3+n front end (coronagraph + imager + spectrometer).
% =====================================================================
%  Dave 2026-07-06: the conic eccentric-pupil section approach
%  (../tma_unobscured) does not meet requirements -- its AOI-safe
%  sections are shroud-expensive and the conic/telecentric coupling
%  fights every repackaging move.  The SPHERE+ZERNIKE strategy
%  (../sz_tma, e5mono-derived) is the direction for 3+n mirrors:
%    * 0th-order layout = base SPHERES placed for packaging; the fold
%      TILTS unobscure the beam while M2 stays CLOSE to the source->M1
%      beam (the tilted-fold topology is the shroud-cheap alternative
%      the eccentric section can't reach);
%    * ALL aberration correction = Zernike (freeform) departures on
%      the mirrors, staged center -> field by CALIB OptZern.  Tiny
%      departures don't move the chief ray, so geometry and correction
%      fully decouple: once the layout packages, the optimizer cannot
%      break it.
%  This example is the 3-mirror front end at 500 nm; the "+n" relay
%  mirrors (per-instrument coronagraph / imager / spectrometer feeds
%  off the shared field) are the roadmap stages that follow it.
%
%  Run AFTER building mmacos:
%    >> run('.../design/examples/freeform_unobscured/freeform_unobscured.m')
% =====================================================================
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/src'));
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% ====================  KNOBS  ========================================
D          = 8.0;        % aperture diameter (m)   -- e5mono heritage
LAM        = 0.5e-6;     % 500 nm center wavelength: VISIBLE telescope
MODEL      = 256;        % engine model size
GRIDN      = 41;         % circular ray-grid points (~1300 rays)
FOV_ARCMIN = 1.0;        % design HALF-field (arcmin) -- the visible DL
                         % bar is 35 nm RMS; hold a tighter field than
                         % sz_tma's 1-um +-2' and let [7] show the cost
DIFFRACTION_LIMIT = 0.071;

% --- 0th-order all-sphere layout (|radii| m, spacings m, fold tilts deg)
% M1 f/3.2 (slower than the centered j18's f/1.2 -- gentle AOI), convex
% M2 slows the cone to a real f/21 intermediate image between M2 and M3
% (the metrology-injection point), M3 reimages it clear of the beam.
R    = [51.534, 8.871, 3.0];   % M1 concave, M2 CONVEX, M3 concave
TBET = [22.0,   28.0];         % M1->M2, M2->M3 along the folded chief
TILT = [-7.2,   8.46,  12.0];  % fold about x (M1/M2 = e5mono)

% --- correction: e5mono's BornWolf Zernike set (piston dropped) ------
MODES = [3 4 5 9 10 11 12 13 19 20 21 22 23 24 25];
ZTYPE = 'BornWolf';
ITERS = 200;
% =====================================================================

fprintf('====================================================================\n');
fprintf(' Freeform (sphere+Zernike) unobscured telescope | D=%.1f m | %g nm\n', ...
        D, LAM*1e9);
fprintf('====================================================================\n');

%% -- [1] the all-sphere 0th-order layout ------------------------------
t = macos.design.Telescope('family','TMA','aperture_diameter_m',D, ...
        'model_size',MODEL,'wavelength_m',LAM,'grid_npts',GRIDN);
t.set_base_sphere(true);                     % hold base spheres (Kc=0)
t.add_mirror('M1','radius_m',R(1),'spacing_after_m',TBET(1),'tilt_deg',TILT(1));
t.add_mirror('M2','radius_m',R(2),'spacing_after_m',TBET(2),'tilt_deg',TILT(2), ...
             'convex',true);
t.add_mirror('M3','radius_m',R(3),'spacing_after','derive','tilt_deg',TILT(3));
t.add_focal_plane('FP');
t.build();
nE = numel(t.spec.elt);
macos.trace(nE);  wfe0 = rms_waves(macos.opd(), LAM);
fprintf(['\n[1] all-sphere layout: real intermediate focus between M2 and ' ...
         'M3 (met injection);\n    uncorrected baseline %.0f waves @ %g nm\n'], ...
        wfe0, LAM*1e9);

%% -- [2] staged freeform correction, center -> field -------------------
% Bauer-style intermediate objectives: diffraction-limit the center,
% then hold the inner 2-D field, then the full field.  Fields are
% 2-D area-weighted (mirror symmetry about the y-z fold plane: sample
% thx>=0, weight thx>0 x2).
fprintf('\n[2] staged Zernike correction (waves RMS @ %g nm):\n', LAM*1e9);
r0 = t.optimize_freeform([1 2 3],'modes',MODES,'type',ZTYPE, ...
                         'fields_arcmin',[],'max_iters',ITERS);
fprintf('    S0 center       : %.0f -> %.4f waves\n', ...
        r0.wfe_before/LAM, r0.wfe_after/LAM);

amin_ = @(A) deg2rad(A/60);
h  = FOV_ARCMIN;
F1 = (h/2)*[1 0; 0 1; 0 -1; 1 1; 1 -1];
F2 =  h   *[1 0; 0 1; 0 -1; 1 1; 1 -1];
Wt = @(Fam) [1, 1 + (Fam(:,1).' > 0)];
r1 = t.optimize_freeform([1 2 3],'modes',MODES,'type',ZTYPE, ...
                         'fields',amin_(F1),'weights',Wt(F1),'max_iters',ITERS);
fprintf('    S1 inner 2-D    : worst %.3f -> %.4f waves\n', ...
        max(r1.wfe_before)/LAM, max(r1.wfe_after)/LAM);
F = [F1; F2];  w = Wt(F);
r2 = t.optimize_freeform([1 2 3],'modes',MODES,'type',ZTYPE, ...
                         'fields',amin_(F),'weights',w,'max_iters',ITERS);
wfe   = r2.wfe_after(:).'/LAM;
wfe_f = max(wfe);  wfe_aw = sqrt(sum(w.*wfe.^2)/sum(w));
fprintf('    S2 full 2-D +-%g'': worst %.4f, area-wtd %.4f waves -> %s\n', ...
        FOV_ARCMIN, wfe_f, wfe_aw, ...
        ternary(wfe_f < DIFFRACTION_LIMIT,'DIFFRACTION-LIMITED','residual'));

%% -- [3] the TRUE focal plane (grid of field foci) ---------------------
% 2x2 for prelim, 5x5 = final design (Dave 2026-07-06); the folded
% chain's FP is tilted wrt the chief.  Runs BEFORE add_pupil.
NGRID = 5;  SPAN = min(0.25, FOV_ARCMIN/2);
fa = t.align_focal_plane('grid',NGRID, 'span_arcmin',SPAN);
fprintf(['\n[3] true FP from %dx%d field foci (+center): tilt %.3f deg, ' ...
         'defocus removed %+.3f mm,\n    field-curvature sag %+.1f to ' ...
         '%+.1f um\n'], NGRID, NGRID, fa.tilt_deg, fa.defocus_m*1e3, ...
        min(fa.sag_m)*1e6, max(fa.sag_m)*1e6);

%% -- [4] field map + clearance ------------------------------------------
% NO apertures at this design stage (Dave 2026-07-06: run without
% apertures for the first design steps; add them once the design
% approaches its objectives).  This also sidesteps the
% realize_apertures frame bug on tilted-fold designs -- footprint
% centers measured in GLOBAL XY but emitted as LOCAL ApVec offsets, so
% the saved .in loses every ray on reload (sz_tma.in carries this
% latent; see Telescope.clear_realized_apertures).  The WFE field map
% comes straight from the field diagnostic instead.
Fmap = macos.design.field_grid(FOV_ARCMIN, 7, 'units','arcmin');
dmap = wfe_field_diag(t, Fmap, 'quiet',true);
scan = struct('fields', Fmap*180*60/pi, 'wfe', dmap.rms_raw(:));
rep = t.check_clipping('noload',true,'quiet',true);
fprintf('\n[4] clearance (bodies vs beams): %d/%d optics clear -> %s\n', ...
        sum([rep.ok]), numel(rep), ...
        ternary(all([rep.ok]),'UNOBSCURED','** OBSCURED **'));

%% -- [5] deliverables ---------------------------------------------------
t.add_pupil(numel(t.spec.elt));              % EP emits PropType=FarField
rxfile  = fullfile(exdir,'freeform_unobscured.in');
matfile = fullfile(exdir,'freeform_unobscured.mat');
t.save(rxfile);  t.save_spec(matfile);
fprintf('\n[5] saved: %s\n           + %s\n', rxfile, matfile);
% Standalone verification of the SAVE (the tma_centered_foldfp lesson:
% the in-session model can be healthy while the .in is not).
macos.init(256);
nv = macos.load_rx(rxfile);  sv = macos.trace(nv);
rv = macos.get_ray_info(sv.nRays);
np = nnz(logical(rv.ok_pass) & logical(rv.ok_trace));
fprintf('    standalone reload: %d elts, %d/%d rays pass -> %s\n', ...
        nv, np, sv.nRays, ternary(np > 0.9*sv.nRays,'VERIFIED','** BROKEN **'));
t.build('', 'init', false);                  % back to the session model
try
    f1 = t.view_field_map(scan,'kind','contour');
    saveas(f1, fullfile(exdir,'freeform_unobscured_wfe.png'));
    f2 = t.view_orthoviews({'YZ','XZ'},'nrays',9);
    saveas(f2, fullfile(exdir,'freeform_unobscured_layout.png'));
    fg = figure('Visible','off');
    contourf(fa.map.thx_arcmin, fa.map.thy_arcmin, fa.map.sag_m*1e6, ...
             15, 'LineColor','none');
    axis equal tight; colormap(parula); cb = colorbar;
    cb.Label.String = 'focus sag from fitted FP  [\mum]';
    xlabel('\theta_x  [arcmin]'); ylabel('\theta_y  [arcmin]');
    title(sprintf('field curvature (FP tilt %.3f\\circ)', fa.tilt_deg));
    saveas(fg, fullfile(exdir,'freeform_unobscured_fpmap.png')); close(fg);
    fprintf('    figures: wfe + layout + fpmap PNGs in the example dir\n');
catch ME, fprintf('    figures skipped (%s)\n', ME.message); end

%% -- [6] the design report ----------------------------------------------
fprintf('\n[6] design report:\n');
rpt = design_report(t, 'rings_arcmin',[0.1 0.25 0.5 FOV_ARCMIN], ...
        'align',fa, ...
        'file',fullfile(exdir,'freeform_unobscured_report.txt'));  %#ok<NASGU>
fprintf('    report: freeform_unobscured_report.txt\n');

%% -- [7] roadmap (the "+n") ---------------------------------------------
fprintf(['\n[7] +n roadmap: relay mirrors feed each instrument a small ' ...
         'patch of the shared\n    field at the aligned FP -- coronagraph ' ...
         'at the FarField exit pupil, imager\n    and spectrometer ' ...
         'pickoffs; per-mirror AOI spread stays under the 15 deg\n    ' ...
         'polarization preference (see the report).  Next stages of this ' ...
         'example.\n']);

% ---- helpers --------------------------------------------------------
function w = rms_waves(W, lam)
    v = W(isfinite(W) & W ~= 0 & abs(W) < 1e30);
    if isempty(v), w = NaN; else, w = std(v)/lam; end
end
function s = ternary(c,a,b), if c, s = a; else, s = b; end, end
