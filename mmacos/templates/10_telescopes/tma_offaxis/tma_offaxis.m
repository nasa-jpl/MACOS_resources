% tma_offaxis.m
% ====================================================================
%  MACOS DESIGN LAYER -- UNOBSCURED OFF-AXIS CONVEX-SECONDARY TMA
%  j18mono-geometry 3-mirror anastigmat (M1 concave, M2 CONVEX, M3
%  concave, f/20): exercise the convex-secondary seidel_seed fix (K=0
%  seed + correct unfolded focus), conic-optimize to a diffraction-
%  limited on-axis anastigmat, then take it OFF-AXIS as an eccentric-
%  pupil section, clear the beam, and re-balance over a 2-D field.
% ====================================================================
%  THE CONVEX-SECONDARY FIX.  The n-flip |radii| Seidel model bakes in a
%  concave-convex-concave alternation and CANNOT model a convex secondary
%  that forms a real intermediate image -- it mis-derives the focus
%  (f/20 -> f/0.9) and returns garbage conics (brute-forcing all 8 radius
%  signs recovers neither; confirmed vs an independent Gaussian imaging
%  chain).  seidel_seed's 'convex' flag (add_mirror ...,'convex',true)
%  returns the correct UNFOLDED paraxial focus (signed curvature, convex =
%  negative lens) + a SAFE K=0 sphere seed; optimize('engine','native')
%  over the conics then refines to diffraction-limited.  The geometry +
%  recovered conics reproduce j18mono.in (f/20 JWST-class, 2.3 um).
%
%  THE OFF-AXIS RECIPE ("design on-axis, then move off-axis").  A coaxial
%  TMA self-obscures (M1 + FP sit in the M2->M3 beam).  set_offaxis('all')
%  extracts an eccentric-pupil SECTION -- RptElt != VptElt on the SAME
%  parent conic (the engine-true j18sc / JWST-segment primitive) --
%  decentering the used sub-aperture until every mirror body clears, then
%  re-optimize for the off-axis zone + balance over the field.
%
%  Run:  >> run('.../templates/10_telescopes/tma_offaxis/tma_offaxis.m')
% ====================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% ---- CONFIG: the j18mono f/20 convex-secondary TMA (edit for yours) ---
MODEL = 256;  LAM = 2.3e-6;                 % j18mono band (2.3 um)
D     = 6.605;                              % aperture (m)
R     = [15.879722 1.778913 3.016227];      % |radii| m: M1 concave, M2 CONVEX, M3 concave
TBET  = [7.169041556 7.965313479];          % M1->M2, M2->M3 (m, along the folded chief ray)
FOV_ARCMIN = 2.0;                           % design HALF-field (arcmin)
NMAP  = 7;                                  % WFE field-map / aperture-scan grid
DIFFRACTION_LIMIT = 0.07;                   % ~lambda/14 (waves)
% ======================================================================

optF = macos.design.field_grid(FOV_ARCMIN, 3, 'units','arcmin', 'origin',false);  % 3x3 AREA

fprintf('====================================================================\n');
fprintf(' Unobscured off-axis convex-secondary TMA (j18mono geometry, f/20)\n');
fprintf('====================================================================\n');

%% -- [0] build the on-axis coaxial layout (convex-secondary fix) -----
t = macos.design.Telescope('family','TMA','aperture_diameter_m',D, ...
        'model_size',MODEL,'wavelength_m',LAM,'grid_npts',41);
t.add_mirror('M1','radius_m',R(1),'spacing_after_m',TBET(1));
t.add_mirror('M2','radius_m',R(2),'spacing_after_m',TBET(2),'convex',true);
t.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
t.add_focal_plane('FP');
t.build();  d = t.spec.derived;  nE = numel(t.spec.elt);
fprintf('\n[0] built: f/%.2f (EFL=%.1f m); convex-fix seed K=[%g %g %g] + correct focus\n', ...
        d.fnum, d.EFL, t.spec.elt(1).Kc, t.spec.elt(2).Kc, t.spec.elt(3).Kc);

%% -- [1] all-sphere baseline (uncorrected) ---------------------------
macos.trace(nE);  wfe0 = rms_waves(macos.opd(), LAM);
fprintf('[1] all-sphere baseline: RMS WFE = %.0f waves (uncorrected)\n', wfe0);

%% -- [2] on-axis conic optimize -> diffraction-limited anastigmat ----
%  3 conics null 3rd-order spherical+coma+astig -- the convex-fix payoff.
r2 = t.optimize('fields_arcmin',[0.5 1.0],'dofs',[0 0 0 0 0 0 0 1],'max_iters',150);
macos.trace(nE);  wfe_c = rms_waves(macos.opd(), LAM);
fprintf('[2] on-axis conic optimize: %.0f -> %.4f waves -> %s  (K=[%.4f %.4f %.4f])\n', ...
        max(r2.wfe_before)/LAM, wfe_c, ternary(wfe_c<DIFFRACTION_LIMIT,'DIFFRACTION-LIMITED','residual'), ...
        t.spec.elt(1).Kc, t.spec.elt(2).Kc, t.spec.elt(3).Kc);

%% -- [3] take it OFF-AXIS as an eccentric-pupil section --------------
dy = t.set_offaxis('all');
macos.trace(numel(t.spec.elt));  wfe_off = rms_waves(macos.opd(), LAM);
fprintf('[3] set_offaxis(''all''): decenter %.3f m (%.2f*D); eccentric-pupil WFE = %.3f waves\n', ...
        dy, dy/D, wfe_off);

%% -- [4] refigure radius+conic for the off-axis zone (axial field) ---
t.optimize('fields_arcmin',[],'dofs',[0 0 0 0 0 0 1 1],'max_iters',100);
t.set_offaxis('none');                      % refresh section poles at the new figure
macos.trace(numel(t.spec.elt));  wfe_ax = rms_waves(macos.opd(), LAM);
fprintf('[4] off-axis axial refigure (radius+conic): %.4f waves -> %s\n', wfe_ax, ...
        ternary(wfe_ax<DIFFRACTION_LIMIT,'DIFFRACTION-LIMITED','residual'));

%% -- [5] balance over the +-FOV AREA (tip/tilt/dy+radius+conic) ------
resf = t.optimize('fields',optF,'dofs',[1 1 0 0 1 0 1 1],'max_iters',150);
t.set_offaxis('none');
wfe_fb = max(resf.wfe_before)/LAM;  wfe_fa = max(resf.wfe_after)/LAM;
fprintf('[5] +-%g'' AREA (3x3): worst %.3f -> %.3f waves -> %s\n', ...
        FOV_ARCMIN, wfe_fb, wfe_fa, ternary(wfe_fa<DIFFRACTION_LIMIT,'DIFFRACTION-LIMITED','residual'));

%% -- [6] field scan + clear apertures + clearance check -------------
scan = t.realize_apertures('fields', ...
        macos.design.field_grid(FOV_ARCMIN,NMAP,'units','arcmin'), 'margin',0.05,'quiet',true);
rep = t.check_clipping('noload',true,'quiet',true);
fprintf('[6] clearance: %d/%d optics clear -> %s\n', sum([rep.ok]), numel(rep), ...
        ternary(all([rep.ok]),'UNOBSCURED','** OBSCURED (see check_clipping)'));

%% -- [7] exit pupil + save deliverable ------------------------------
t.add_pupil(numel(t.spec.elt));
rxfile = fullfile(exdir,'tma_offaxis.in');  matfile = fullfile(exdir,'tma_offaxis.mat');
t.save(rxfile);  t.save_spec(matfile);
fprintf('[7] saved: %s\n           + %s\n', rxfile, matfile);

%% -- [8] design-report figures --------------------------------------
try
    f1 = t.view_field_map(scan,'kind','contour');
    p1 = fullfile(exdir,'tma_offaxis_wfe.png');  saveas(f1,p1);
    fprintf('[8] WFE field map: %s\n', p1);
catch ME, fprintf('[8] WFE map skipped (%s)\n', ME.message); end
try
    f2 = t.view_orthoviews({'YZ','XZ'},'nrays',9);
    p2 = fullfile(exdir,'tma_offaxis_layout.png');  saveas(f2,p2);
    fprintf('    orthographic layout: %s\n', p2);
catch ME, fprintf('    layout skipped (%s)\n', ME.message); end

%% -- summary --------------------------------------------------------
fprintf('\n--------------------------------------------------------------------\n');
fprintf(' D=%.1f m | f/%.1f | convex-secondary TMA, j18mono geometry, %s lambda=%.2g um\n', ...
        D, d.fnum, 'unobscured off-axis', LAM*1e6);
fprintf(' RMS WFE: baseline %.0f -> on-axis %.4f -> off-axis +-%g'' worst %.4f waves\n', ...
        wfe0, wfe_c, FOV_ARCMIN, wfe_fa);
fprintf(' convex-secondary fix: K=0 seed + correct f/20 focus; conics from optimize.\n');
fprintf('====================================================================\n');

% ---- helpers --------------------------------------------------------
function w = rms_waves(W, lam)
    v = W(isfinite(W) & W ~= 0);
    if isempty(v), w = NaN; else, w = std(v)/lam; end
end
function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
