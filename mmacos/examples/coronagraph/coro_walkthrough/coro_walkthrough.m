function coro_walkthrough(rx_path, outdir)
%CORO_WALKTHROUGH  Illustrated coronagraph propagation example (for the manual).
%   coro_walkthrough() propagates the HCIT-style Lyot coronagraph
%   (Rx_Coro_FPM.in) through MACOS and saves annotated figures at the
%   KEY SURFACES of the relay, demonstrating the coronagraph principle
%   end-to-end:
%
%     entrance pupil (Elt 2)  -> DM pupil (Elt 4)
%       -> focal-plane mask / FPM (Elt 9, CorMask)
%       -> Lyot pupil (Elt 14, LyotStop)   <- where starlight is rejected
%       -> exit pupil (Elt 20)
%       -> science focal plane (Elt 21)    <- the dark hole
%
%   Figures written to OUTDIR (default ./figures):
%     coro_surfaces.png   2x3 montage at the key surfaces
%     coro_darkzone.png   radial contrast vs lambda/D (no-mask vs coronagraph)
%     coro_broadband.png  monochromatic vs COMPOSE broadband PSF
%
%   The COMPOSE broadband step is shown as ONE step in the walkthrough
%   (macos.compose assembles per-wavelength PSFs on a fixed pixel grid).
%
%   Runs headless (MATLAB -batch): figures are created invisible and
%   exported to PNG.  Reuses the dark-zone scoring helpers in
%   ../coro/.
arguments
    rx_path (1,:) char = ''
    outdir  (1,:) char = ''
end

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'design', 'coro'));   % radial_contrast, dark_zone_metrics
if isempty(rx_path)
    rx_path = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
        'pymacos', 'tests', 'Rx', 'Rx_Coro_FPM.in');
end
if isempty(outdir)
    outdir = fullfile(here, 'figures');
end
if ~exist(outdir, 'dir'), mkdir(outdir); end

MODEL = 1024;
LAM_NM = 850;                       % central wavelength (nm)
DET   = 21;                         % science focal plane
NO_MASK_RX = fullfile(fileparts(rx_path), 'Rx_Coro_noLyot.in');

macos.init(MODEL);
macos.load_rx(rx_path);
macos.trace();

% ---- Key surfaces: (element, kind, title) ----------------------------
%   'pupil' -> amplitude |field| (linear);  'focal' -> intensity (log10)
panels = {
    2,  'pupil', 'Entrance pupil (Elt 2)'
    4,  'pupil', 'DM pupil (Elt 4)'
    9,  'focal', 'FPM / CorMask (Elt 9)'
   14,  'pupil', 'Lyot pupil (Elt 14)'
   20,  'pupil', 'Exit pupil (Elt 20)'
   21,  'focal', 'Science focal plane (Elt 21)'
};

fig = figure('Visible','off','Position',[80 80 1280 820], 'Color','w');
tl = tiledlayout(fig, 2, 3, 'TileSpacing','compact', 'Padding','compact');
title(tl, sprintf('MACOS coronagraph walkthrough (Rx\\_Coro\\_FPM, %d nm, model %d)', ...
    LAM_NM, MODEL), 'FontWeight','bold');

for i = 1:size(panels,1)
    elt  = panels{i,1};  kind = panels{i,2};  ttl = panels{i,3};
    nexttile(tl);
    if strcmp(kind, 'pupil')
        cf  = macos.complex_field(elt);
        img = abs(cf);                         % amplitude shows the aperture
        % TODO(hot-pixel): pupil complex_field carries a spurious hot
        % pixel (see memory project_pupil_hot_pixel) -- root cause not
        % yet found.  Normalise to the 99.5th percentile so it doesn't
        % wash out the real pupil structure.  DISPLAY workaround only.
        nrm = prctile(img(:), 99.5);
        img = min(img / max(nrm, eps), 1.0);
        imagesc(crop_center(img, 360)); axis image off;
        colormap(gca, gray); clim([0 1]);
        title(ttl);
    else
        I   = macos.intensity(elt);
        I   = I / max(I(:) + eps);
        L   = log10(max(I, 1e-12));
        imagesc(crop_center(L, 220)); axis image off;
        colormap(gca, parula); clim([-8 0]);
        cb = colorbar; cb.Label.String = 'log_{10} norm. intensity';
        title(ttl);
    end
end
exportgraphics(fig, fullfile(outdir,'coro_surfaces.png'), 'Resolution',150);
close(fig);
fprintf('[walkthrough] wrote coro_surfaces.png\n');

% ---- Dark-zone radial contrast: no-mask vs coronagraph ---------------
% (reuse the Sprint-1 coro scoring helpers)
macos.load_rx(NO_MASK_RX);
I_no    = macos.intensity(DET);
peak_no = max(I_no(:));
lamD    = lambda_over_D_pixels(I_no);
[r_no, c_no] = radial_contrast(I_no, peak_no, lamD, 20.0);

macos.load_rx(rx_path);
I_co = macos.intensity(DET);
[r_co, c_co] = radial_contrast(I_co, peak_no, lamD, 20.0);
m_co = dark_zone_metrics(I_co, peak_no, lamD, 7, 10);

fig = figure('Visible','off','Position',[80 80 760 560], 'Color','w');
semilogy(r_no, max(c_no,1e-14), 'LineWidth',1.5); hold on;
semilogy(r_co, max(c_co,1e-14), 'LineWidth',1.5);
grid on; xlim([0 18]); ylim([1e-12 2]);
xlabel('separation (\lambda/D)'); ylabel('radial contrast (norm. to no-mask peak)');
legend({'no mask (baseline)','FPM + Lyot coronagraph'}, 'Location','northeast');
title(sprintf(['Rx\\_Coro radial contrast at science focal plane\n', ...
    'dark-zone (7-10 \\lambda/D): mean %.2e, floor %.2e'], m_co.mean, m_co.floor));
exportgraphics(fig, fullfile(outdir,'coro_darkzone.png'), 'Resolution',150);
close(fig);
fprintf('[walkthrough] wrote coro_darkzone.png (dz mean %.2e, floor %.2e)\n', ...
    m_co.mean, m_co.floor);

% ---- Before / after: mask plane (Elt 9) + science focal plane (Elt 21)
% The key "did the mask do anything?" figure.  BOTH panels in each row
% share ONE normalisation (the no-coronagraph reference), so the
% suppression is visible rather than hidden by per-panel auto-scaling.
% Focal FOV extends past the 7-10 lambda/D dark hole (rings overlaid).
FPM_ELT = 9;
macos.load_rx(NO_MASK_RX); I_pre_mask  = macos.intensity(FPM_ELT);
macos.load_rx(rx_path);    I_post_mask = macos.intensity(FPM_ELT);
% I_no (no-coro @21) and I_co (coro @21) + peak_no + lamD reused from above.
ref_mask = max(I_pre_mask(:));
% The FPM is a Circle of radius 0.4 mm (ObsVec); at the mask plane's fine
% sampling that disk is ~hundreds of px across -- size the crop to the
% FPM's OWN scale (not the science-plane lambda/D) so the mask EDGE and
% the starlight returning outside it are both visible.
dx9 = abs(macos.dx_at(FPM_ELT));            % mask-plane pitch (m)
r_fpm_px = 4.0e-4 / max(dx9, eps);          % FPM radius (0.4 mm) in pixels
wm = min(max(round(2.8 * r_fpm_px), 120), 760);   % +/- ~1.4 FPM radii
wf = round(2 * 15 * lamD);            % focal FOV: +/- 15 lambda/D (past dark hole)

fig = figure('Visible','off','Position',[80 80 1080 1000], 'Color','w');
tl = tiledlayout(fig, 2, 2, 'TileSpacing','compact','Padding','compact');
title(tl, 'Coronagraph before / after (shared normalisation per row)', ...
    'FontWeight','bold');

nexttile(tl);
show_log(crop_center(I_pre_mask /ref_mask, wm), [-6 0]);
title('Focal-mask plane: NO FPM (stellar core)');
nexttile(tl);
show_log(crop_center(I_post_mask/ref_mask, wm), [-6 0]);
title('Focal-mask plane: FPM (core blocked)');

nexttile(tl);
show_log(crop_center(I_no/peak_no, wf), [-10 0]);
title('Science focal plane: NO coronagraph');
add_dz_rings(wf, lamD, 7, 10);
nexttile(tl);
show_log(crop_center(I_co/peak_no, wf), [-10 0]);
title('Science focal plane: FPM + Lyot coronagraph');
add_dz_rings(wf, lamD, 7, 10);
exportgraphics(fig, fullfile(outdir,'coro_beforeafter.png'), 'Resolution',150);
close(fig);
fprintf('[walkthrough] wrote coro_beforeafter.png\n');

% ---- COMPOSE: monochromatic vs broadband composite (ONE step) --------
macos.load_rx(rx_path);
macos.intensity(DET);                          % establish dxElt(DET) before dx_at
% macos's focal-plane dxElt is signed (plane-orientation convention; the
% FF output plane reports a negative dx).  A pixel SIZE is a magnitude.
dx   = abs(macos.dx_at(DET));                  % SI metres
npix = 128;
lam0 = macos.get_src_wvl();                     % WaveUnits
I_mono = macos.compose(DET, lam0,                       npix, dx);
lams   = lam0 * linspace(0.90, 1.10, 7);        % 20% band, 7 samples
I_band = macos.compose(DET, lams,                        npix, dx);

fig = figure('Visible','off','Position',[80 80 1100 520], 'Color','w');
tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact','Padding','compact');
title(tl, 'COMPOSE: monochromatic vs broadband PSF (science focal plane)', ...
    'FontWeight','bold');
nexttile(tl);
imagesc(log10(max(I_mono/max(I_mono(:)+eps),1e-10))); axis image off;
colormap(parula); clim([-10 0]); colorbar;
title(sprintf('monochromatic (%d nm)', LAM_NM));
nexttile(tl);
imagesc(log10(max(I_band/max(I_band(:)+eps),1e-10))); axis image off;
colormap(parula); clim([-10 0]); colorbar;
title(sprintf('broadband COMPOSE (%d nm \\pm 10%%, %d \\lambda)', LAM_NM, numel(lams)));
exportgraphics(fig, fullfile(outdir,'coro_broadband.png'), 'Resolution',150);
close(fig);
fprintf('[walkthrough] wrote coro_broadband.png\n');

% ---- Planet injection: on-axis star + off-axis "planet" via COMPOSE --
% COMPOSE adds two SCENES (not wavelengths): the coronagraph-suppressed
% on-axis star plus a faint off-axis planet that slips past the FPM edge
% into the dark hole.  Uses the low-level COMPOSE primitives directly
% (macos.compose loops wavelengths; here we vary source pointing + flux
% between the two ADDs).
np3          = 256;
% PRELIMINARY planet-injection demo -- to be redone on a better-scaled
% model.  Two known limitations on this heavily SCALED Rx_Coro (tiny
% ~0.22 mm pupil):
%   1) N lambda/D maps to a LARGE source tilt (8 l/D ~ 1.8 deg) that
%      vignettes the off-axis beam through the relay.
%   2) macos's focal-plane diffraction grid re-centers on the chief ray,
%      so an off-axis source's PSF stays CENTERED in its grid -- COMPOSE
%      then stacks the planet on the (suppressed) star instead of
%      offsetting it.  Correct off-axis placement needs the WINDOW
%      command (ifPixLoc -> CPIXILATE uses the chief-ray offset), which
%      is not yet wrapped.
% TODO: redo on a realistically-scaled coronagraph (arcsec tilts) WITH a
% WINDOW reference so the planet lands at its true sky offset.
PLANET_LAMD  = 3;            % planet separation (lambda/D)
PLANET_RATIO = 1e-3;         % planet flux relative to the star

% Source-tilt for the planet, from the first-order plate scale:
% tilt-per-(lambda/D) = lamD_rad (SYSPROP).  Exact, needs no calibration
% ramp.  Computed on the no-mask Rx (clean EFL marginal-ray trace; the
% FPM can vignette the marginal ray); same optics -> same lamD_rad for
% the coronagraph.
macos.load_rx(NO_MASK_RX);
fop = macos.first_order_properties(DET);
th_planet = PLANET_LAMD * fop.lamD_rad;
fprintf('[walkthrough] planet at %d lambda/D -> tilt %.3e rad (lamD_rad=%.3e)\n', ...
        PLANET_LAMD, th_planet, fop.lamD_rad);

% COMPOSE the two scenes on the coronagraph.
macos.load_rx(rx_path);
macos.intensity(DET);
dxp_bu = abs(macos.dx_at(DET, 'native'));          % BaseUnits for raw compose_start
fovC = macos.get_src_fov();  dirC = fovC.src_dir;  fluxC = macos.get_src_flux();
% Build the composite incrementally so we can capture the star-only
% scene and then star+planet -- their difference isolates the planet.
mmacos('compose_start', double(DET), double(np3), double(dxp_bu));
macos.set_src_fov('src_dir', dirC);                        macos.set_src_flux(fluxC);
macos.intensity(DET);  mmacos('compose_add', 0.0);                  % on-axis star
I_star = mmacos('compose_get', double(np3));                        % star only
macos.set_src_fov('src_dir', dirC + [th_planet; 0; 0]);    macos.set_src_flux(PLANET_RATIO*fluxC);
macos.intensity(DET);  mmacos('compose_add', 0.0);                  % + off-axis planet
I_scene = mmacos('compose_get', double(np3));                       % star + planet
macos.set_src_fov('src_dir', dirC);  macos.set_src_flux(fluxC);     % restore
I_diff = I_scene - I_star;                                          % planet light, star cancelled

fig = figure('Visible','off','Position',[80 80 1180 560], 'Color','w');
tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact', 'Padding','compact');
title(tl, sprintf(['Planet injection via COMPOSE: suppressed on-axis ', ...
   'star + planet at %d \\lambda/D, %g\\times fainter'], ...
   PLANET_LAMD, PLANET_RATIO), 'FontWeight','bold');
nexttile(tl);
show_log(I_scene / max(I_scene(:)), [-6 0]);
title('star + planet (composite)');
add_dz_rings(np3, lamD, 7, 10);
nexttile(tl);
% Difference (star+planet) - (star) cancels the star residual and
% sharply exposes the planet.  Normalise to the planet peak.
show_log(max(I_diff,0) / max(I_diff(:)), [-6 0]);
title('difference: (star+planet) - star  ->  planet isolated');
add_dz_rings(np3, lamD, 7, 10);
exportgraphics(fig, fullfile(outdir,'coro_planet.png'), 'Resolution',150);
close(fig);
fprintf('[walkthrough] wrote coro_planet.png (planet tilt %.2e rad)\n', th_planet);

fprintf('[walkthrough] done -> %s\n', outdir);
end

% ----------------------------------------------------------------------
function out = crop_center(img, w)
% Center-crop an NxN array to w x w (for display); no-op if w >= N.
    n = size(img,1);
    if w >= n, out = img; return; end
    c  = floor(n/2) + 1;
    lo = c - floor(w/2);  hi = lo + w - 1;
    lo = max(lo,1); hi = min(hi,n);
    out = img(lo:hi, lo:hi);
end

% ----------------------------------------------------------------------
function show_log(img, climv)
% imagesc of log10(img) with a fixed colour range + colorbar.
    L = log10(max(img, 10^climv(1)));
    imagesc(L); axis image off;
    colormap(gca, parula); clim(climv);
    cb = colorbar; cb.Label.String = 'log_{10} norm. intensity';
end

% ----------------------------------------------------------------------
function add_dz_rings(w, lamD, inner_lamD, outer_lamD)
% Overlay dashed rings at inner/outer lambda/D on a w x w display whose
% centre is the optical axis (the crop is centred on the array centre).
    c = (w + 1) / 2;   hold on;
    for rr = [inner_lamD outer_lamD] * lamD
        rectangle('Position', [c-rr, c-rr, 2*rr, 2*rr], ...
                  'Curvature', [1 1], 'EdgeColor', 'w', ...
                  'LineStyle', '--', 'LineWidth', 1.0);
    end
    hold off;
end
