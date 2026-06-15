function out = coro_planet_demo(opts)
%CORO_PLANET_DEMO  Illustrated coronagraph + off-axis planet injection.
%   out = CORO_PLANET_DEMO() runs the canonical CoroExample recipe through
%   the mmacos framework and writes two 4-panel figures (matching the
%   CoroExample.jou "pgp 4" plot pages) into the example folder:
%     coro_surfaces.png  the coronagraph surfaces  (jou: int 1/6/11/16)
%       [entrance pupil | star occulted at FPM | pupil at Lyot | star at det]
%     coro_planet.png    the planet injection      (jou: ffp + add)
%       [suppressed star | planet alone | star+planet | difference]
%
%   The planet injection assembles two scenes on one fixed COMPOSE pixel
%   grid at the detector: the suppressed on-axis star, and a dim OFF-AXIS
%   planet placed with FFP/PFP.  The planet lands at its TRUE sky offset
%   (not re-centred) because macos.window(...) turns on the WINDOW pixel-
%   location option first; ADD always composes at the COMPOSE element
%   (macos_cmd_loop.inc:4814), and WINDOW places each source by its chief-
%   ray offset from the element vertex (~5236).
%
%   The off-axis position is caller-chosen via opts.offset + opts.unit --
%   the two engine-native placement conventions, which differ by the plate
%   scale:
%     'sky_angle'  direction cosines [dx dy] (~= field angle in rad)
%                                              -> macos.ffp
%     'pixels'     focal-plane pixels [dx dy] on the COMPOSE grid (pitch
%                  dxpix)                       -> macos.pfp
%   To place a planet at N lambda/D on a system whose EFL marginal-ray
%   trace succeeds (so SYSPROP returns a focal plate scale), convert first:
%     p = macos.first_order_properties(det_elt);
%     coro_planet_demo('unit','pixels','offset', N*p.lamD_px*[1 0]);
%   NOTE: a lambda/D *unit* is deliberately NOT offered.  SYSPROP's
%   lamD_rad is the ENTRANCE-pupil lambda/D (sky); a de-magnifying relay
%   (like CoroExample's) maps a few entrance-lambda/D to a sub-pixel shift
%   at the science detector, so 'lamD' via lamD_rad would silently mislead.
%
%   Name-value options (all have CoroExample defaults):
%     'rx'          prescription path (default: CoroExample.in)
%     'model'       model size            (default 512 -- CoroExample needs it)
%     'place_elt'   element to position the planet at (default 6, the FPM)
%     'det_elt'     detector / compose plane          (default 16)
%     'stop_elt'    system stop for FFP/PFP / Lyot     (default 11)
%     'mask_elt'    focal-plane mask element           (default 6)
%     'reset_ors'   Return elements to re-establish    (default [5 7])
%     'reset_fex'   re-find exit pupil after FFP       (default true)
%     'npix'        composite grid size                (default 64)
%     'dxpix'       composite pixel pitch, BaseUnits   (default 1.392542e-6)
%     'offset'      planet offset [dx dy]              (default [2e-5 2e-5])
%     'unit'        'sky_angle' | 'pixels'             (default 'sky_angle')
%     'frame'       WINDOW frame 'tout' | 'beam'       (default 'tout')
%     'planet_flux' probe source flux (the planet image is then rescaled
%                   to planet_contrast; value itself is unobservable)  (0.01)
%     'planet_contrast' planet peak / suppressed-star residual peak
%                   (default 1.0 = on par, the realistic detection regime)
%     'outdir'      where to write the figures          (default: example dir)
%     'visible'     show the figures                    (default false)
%
%   Returns a struct with the surface intensities, the composite scenes,
%   the measured planet peak offset, and the list of figures written.
%
%   See also: macos.window, macos.ffp, macos.pfp, macos.compose,
%             macos.first_order_properties, macos.stop.
arguments
    opts.rx          (1,:) char = default_rx()
    opts.model       (1,1) double {mustBeInteger,mustBePositive} = 512
    opts.place_elt   (1,1) double {mustBeInteger,mustBePositive} = 6
    opts.det_elt     (1,1) double {mustBeInteger,mustBePositive} = 16
    opts.stop_elt    (1,1) double {mustBeInteger,mustBePositive} = 11
    opts.mask_elt    (1,1) double {mustBeInteger,mustBePositive} = 6
    opts.spot_elt    (1,1) double {mustBeInteger,mustBePositive} = 16
    opts.reset_ors   (1,:) double = [5 7]
    opts.reset_fex   (1,1) logical = true
    opts.npix        (1,1) double {mustBeInteger,mustBePositive} = 91
    opts.dxpix       (1,1) double {mustBePositive} = 1.392542e-6
    opts.offset      (1,2) double = [2e-5 2e-5]
    opts.unit        (1,:) char {mustBeMember(opts.unit,{'sky_angle','pixels'})} = 'sky_angle'
    opts.frame       (1,:) char {mustBeMember(opts.frame,{'tout','beam'})} = 'tout'
    opts.planet_flux (1,1) double {mustBePositive} = 0.01
    opts.planet_contrast (1,1) double {mustBePositive} = 1.0
    opts.outdir      (1,:) char = default_outdir()
    opts.visible     (1,1) logical = false
end

DET   = opts.det_elt;
PLACE = opts.place_elt;
MASK  = opts.mask_elt;
LYOT  = opts.stop_elt;
SPOTE = opts.spot_elt;       % element for the geometric spot diagrams
NPIX  = opts.npix;
DXP   = opts.dxpix;          % BaseUnits pixel pitch (matches COMPOSE/WINDOW)
vis   = ternary(opts.visible, 'on', 'off');
if ~exist(opts.outdir,'dir'); mkdir(opts.outdir); end
figs  = {};

% --- load + establish the ray grid ------------------------------------
macos.init(opts.model);
nElt = macos.load_rx(opts.rx);
assert(nElt >= DET, 'rx has %d elts, need >= %d', nElt, DET);
macos.intensity(DET);                       % first trace -> grid established
flux0 = macos.get_src_flux();               % star (nominal) flux

% --- FIGURE 1: coronagraph surfaces (CoroExample.jou pgp-4 page) -------
% Entrance pupil, on-axis star occulted at the FPM, pupil at the Lyot
% stop, and the suppressed star at the detector -- the jou int 1/6/11/16.
I_pupil = macos.intensity(1);
I_mask  = macos.intensity(MASK);
I_lyot  = macos.intensity(LYOT);
I_starD = macos.intensity(DET);

fig1 = figure('Visible',vis,'Position',[60 60 1180 940],'Color','w');
tl1  = tiledlayout(fig1, 2, 2, 'TileSpacing','compact', 'Padding','compact');
title(tl1, 'CoroExample coronagraph surfaces', 'FontWeight','bold');
panel(tl1, I_pupil, 'sqrt', [],      sprintf('entrance pupil (Elt %d)', 1));
panel(tl1, I_mask,  'log',  [-6 0],  sprintf('star occulted at FPM (Elt %d)', MASK));
panel(tl1, I_lyot,  'sqrt', [],      sprintf('pupil at Lyot stop (Elt %d)', LYOT));
panel(tl1, I_starD, 'log',  [-6 0],  sprintf('suppressed star at detector (Elt %d)', DET));
figs{end+1} = fullfile(opts.outdir, 'coro_surfaces.png');
exportgraphics(fig1, figs{end}, 'Resolution',150);
if ~opts.visible; close(fig1); end
fprintf('[coro_planet] wrote coro_surfaces.png\n');

% --- resolve the requested placement into an ffp/pfp call -------------
switch opts.unit
    case 'sky_angle'
        cosine_xy = opts.offset;            % direction cosines ~= field angle
        place = @() macos.ffp(PLACE, opts.offset);
    case 'pixels'
        % pixels -> cosines via the focal plate scale (informational only;
        % EFL may be unavailable on complex relays -> NaN).
        fop = macos.first_order_properties(DET);
        if fop.efl_baseunits > 0
            cosine_xy = opts.offset * DXP / fop.efl_baseunits;
        else
            cosine_xy = [NaN NaN];
        end
        place = @() macos.pfp(PLACE, DXP, opts.offset);
end

% --- WINDOW on: place each source at its TRUE offset on the grid ------
macos.window(opts.frame, DXP);

% --- the suppressed on-axis star (its own composite) -----------------
mmacos('compose_start', double(DET), double(NPIX), double(DXP));
macos.intensity(DET);  mmacos('compose_add', 0.0);
I_star = mmacos('compose_get', double(NPIX));

% --- star geometric spot (on-axis), ALL rays --------------------------
% Every ray is obscured at a coronagraph focal plane, so set OBS ALL
% (iObsOpt=0) to plot the geometric image regardless of the FPM/Lyot --
% Dave's option (2).  Centred on the chief ray so the comparison is of
% spot SHAPE/SIZE.  Restore OBS POSITIVE so the composites stay masked.
mmacos('obs_set', 0);                            % ALL rays
star_spot = macos.spot(SPOTE, 'ref','tout', 'at','chief');
mmacos('obs_set', 1);                            % restore unobscured-only

% --- the off-axis planet: tilt source, reset references, dim it ------
macos.stop(LYOT);                            % FFP/PFP need the system stop
place();                                     % tilt source off-axis
for ie = opts.reset_ors                      % re-establish Return surfaces
    mmacos('ors_run', double(ie));
end
if opts.reset_fex && nElt > 3
    macos.fex(1);                            % re-find exit pupil (Srf nElt-1)
end
macos.set_src_flux(opts.planet_flux * flux0);

% Planet light BY ITSELF, on its own fresh composite (CoroExample.jou's
% "int 6" analogue): the off-axis planet image alone.
mmacos('compose_start', double(DET), double(NPIX), double(DXP));
macos.intensity(DET);  mmacos('compose_add', 0.0);
I_planet0 = mmacos('compose_get', double(NPIX));   % probe (rescaled below)

% --- planet geometric spot (off-axis), ALL rays -----------------------
% Source is still tilted off-axis here; the broad geometric spot vs the
% tight on-axis star spot confirms FIELD ABERRATION (off-axis coma /
% astigmatism) as the cause of the planet's PSF broadening.
mmacos('obs_set', 0);                            % ALL rays
planet_spot = macos.spot(SPOTE, 'ref','tout', 'at','chief');
mmacos('obs_set', 1);                            % restore unobscured-only

% --- restore engine state --------------------------------------------
macos.set_src_flux(flux0);
macos.window_off();

% --- scale the planet to the requested contrast vs the suppressed star -
% Put the planet PEAK at opts.planet_contrast x the suppressed-star
% residual peak (default 1.0 = on par -- the realistic detection regime,
% where the planet sits in the coronagraph noise floor).  Intensity is
% linear in source flux, so scaling the probe image is exact; eff_flux is
% the physical planet/star flux ratio it corresponds to.
star_peak = max(I_star(:));
p0        = max(I_planet0(:));
pscale    = opts.planet_contrast * star_peak / max(p0, eps);
I_planet  = I_planet0 * pscale;
eff_flux  = opts.planet_flux * pscale;

% COMPOSE is incoherent-additive, so the full scene is star + planet, and
% (scene - star) recovers the planet -- the diff confirms the isolation.
I_scene = I_star + I_planet;
I_diff  = I_scene - I_star;

% --- VALIDATION: where did the planet land? (PEAK, not just centroid) -
cen          = (NPIX + 1) / 2;                % grid centre (1-based)
[~, ipk]     = max(I_planet(:));
[pr, pc]     = ind2sub(size(I_planet), ipk);
peak_xy      = [pc pr];                        % [col row] of brightest planet px
peak_off     = peak_xy - cen;                  % pixels from centre
cen_xy       = centroid(max(I_planet,0));      % intensity-weighted centre
cen_off      = cen_xy - cen;
offmag       = hypot(peak_off(1), peak_off(2));
placed_off   = offmag > 2.0;                   % clearly displaced from centre
fprintf(['[coro_planet] unit=%s offset=[%g %g] (cosines=[%.3e %.3e])\n', ...
         '              planet peak at [%d %d] = [%+.1f %+.1f] px from centre ', ...
         '(|%.1f| px), centroid [%+.1f %+.1f] -> %s\n'], ...
        opts.unit, opts.offset(1), opts.offset(2), cosine_xy(1), cosine_xy(2), ...
        pc, pr, peak_off(1), peak_off(2), offmag, cen_off(1), cen_off(2), ...
        ternary(placed_off,'OFF-AXIS (WINDOW OK)','CENTRED (placement failed)'));
fprintf(['[coro_planet] planet peak = %.2gx suppressed-star residual peak ', ...
         '(effective flux %.2e x star; star peak %.2e, planet peak %.2e)\n'], ...
        opts.planet_contrast, eff_flux, star_peak, max(I_planet(:)));

% --- FIGURE 2: planet injection (CoroExample.jou pgp-4 page) ----------
% ABSOLUTE, SHARED LINEAR scale across all four panels (not per-panel
% normalised) so the suppressed star and the on-par off-axis planet can
% be compared directly -- the colourbars all read the same absolute
% intensity, 0 .. global peak.
gmax     = max([max(I_star(:)), max(I_planet(:)), max(I_scene(:)), eps]);
clim_abs = [0, gmax];
fig2 = figure('Visible',vis, 'Position',[60 60 1180 940], 'Color','w');
tl2  = tiledlayout(fig2, 2, 2, 'TileSpacing','compact', 'Padding','compact');
title(tl2, sprintf(['Planet injection (WINDOW + %s, absolute linear scale): ', ...
    'planet at [%g %g] %s, %.2g\\times the star-residual peak ', ...
    '(flux %.1e \\times star)'], upper(opts.unit), opts.offset(1), ...
    opts.offset(2), opts.unit, opts.planet_contrast, eff_flux), 'FontWeight','bold');

panel(tl2, I_star,        'linabs', clim_abs, 'suppressed (occulted) star');
mark(cen, cen, peak_xy, false);
panel(tl2, I_planet,      'linabs', clim_abs, 'planet light, by itself');
mark(cen, cen, peak_xy, true);
panel(tl2, I_scene,       'linabs', clim_abs, 'star + planet (composite)');
mark(cen, cen, peak_xy, true);
panel(tl2, max(I_diff,0), 'linabs', clim_abs, 'difference: (star+planet) - star');
mark(cen, cen, peak_xy, true);

figs{end+1} = fullfile(opts.outdir, 'coro_planet.png');
exportgraphics(fig2, figs{end}, 'Resolution',150);
if ~opts.visible; close(fig2); end
fprintf('[coro_planet] wrote coro_planet.png\n');

% --- FIGURE 3: geometric spot diagrams (field aberration) -------------
% On-axis star vs off-axis planet, each centred on its chief ray, shared
% axes.  A broad planet spot vs a tight star spot is the geometric-optics
% confirmation that field aberration drives the planet PSF broadening.
rms_star   = spot_rms(star_spot.pts);
rms_planet = spot_rms(planet_spot.pts);
lim = 1.1 * max([spot_ext(star_spot.pts), spot_ext(planet_spot.pts), eps]);
fig3 = figure('Visible',vis, 'Position',[80 80 1180 560], 'Color','w');
tl3  = tiledlayout(fig3, 1, 2, 'TileSpacing','compact', 'Padding','compact');
title(tl3, sprintf(['Geometric spot at Elt %d (OBS ALL): on-axis star vs ', ...
    'off-axis planet'], SPOTE), 'FontWeight','bold');
spot_panel(tl3, star_spot.pts,   lim, sprintf('star (on-axis)\\newlineRMS = %.2e', rms_star));
spot_panel(tl3, planet_spot.pts, lim, sprintf('planet (off-axis)\\newlineRMS = %.2e (%.1f\\times)', ...
    rms_planet, rms_planet/max(rms_star,eps)));
figs{end+1} = fullfile(opts.outdir, 'coro_spots.png');
exportgraphics(fig3, figs{end}, 'Resolution',150);
if ~opts.visible; close(fig3); end
fprintf('[coro_planet] wrote coro_spots.png (RMS star=%.2e planet=%.2e, %.1fx broader)\n', ...
        rms_star, rms_planet, rms_planet/max(rms_star,eps));

out = struct('I_pupil',I_pupil, 'I_mask',I_mask, 'I_lyot',I_lyot, ...
             'I_starD',I_starD, 'I_star',I_star, 'I_planet',I_planet, ...
             'I_scene',I_scene, 'I_diff',I_diff, 'peak_offset_px',peak_off, ...
             'centroid_offset_px',cen_off, 'offset_mag_px',offmag, ...
             'placed_off_axis',placed_off, 'cosines',cosine_xy, ...
             'star_spot',star_spot, 'planet_spot',planet_spot, ...
             'rms_spot_star',rms_star, 'rms_spot_planet',rms_planet, ...
             'planet_contrast',opts.planet_contrast, 'eff_flux',eff_flux, ...
             'unit',opts.unit, 'figures',{figs});
end

% ====================================================================
function p = default_rx()
% CoroExample.in lives in the macos source tree's manual examples.
here = fileparts(mfilename('fullpath'));
p = fullfile(here, '..','..','..','..','macos','docs','macos-manual', ...
             'examples','CoroExample.in');
end

function d = default_outdir()
% Write artifacts next to this example by default (committed with it).
d = fileparts(mfilename('fullpath'));
end

function c = centroid(I)
% intensity-weighted centre of mass, returned as [col row] (1-based).
I = double(I); s = sum(I(:));
if s <= 0; c = ((size(I,2)+1)/2) * [1 1]; return; end
[X,Y] = meshgrid(1:size(I,2), 1:size(I,1));
c = [sum(X(:).*I(:)), sum(Y(:).*I(:))] / s;
end

function panel(tl, I, stretch, clim, ttl)
% draw one surface into the next tile of a 2x2 layout.
%   'log'    per-panel normalised log10 (clim relative to the panel peak)
%   'logabs' ABSOLUTE log10 (clim are absolute log10 values)
%   'linabs' ABSOLUTE linear (clim are absolute intensities; use a shared
%            clim to compare panels directly)
%   'sqrt'   per-panel normalised sqrt
nexttile(tl);
I = double(I);
switch stretch
    case 'log';    show_log(I / (max(I(:)) + eps), clim);
    case 'logabs'; show_log(I, clim);
    case 'linabs'; show_lin(I, clim);
    case 'sqrt';   show_sqrt(I / (max(I(:)) + eps));
end
title(ttl);
end

function show_log(I, clim)
imagesc(log10(max(I, 10^clim(1)))); axis image off;
caxis(clim); colormap(gca, 'hot'); colorbar;
end

function show_sqrt(I)
imagesc(sqrt(max(I, 0))); axis image off;
colormap(gca, 'hot'); colorbar;
end

function show_lin(I, clim)
imagesc(I); axis image off;
caxis(clim); colormap(gca, 'hot'); colorbar;
end

function r = spot_rms(pts)
% RMS spot radius about the spot centroid (BaseUnits).
if isempty(pts); r = 0; return; end
c = mean(pts, 1);
r = sqrt(mean(sum((pts - c).^2, 2)));
end

function e = spot_ext(pts)
% max ray distance from the spot centroid (BaseUnits).
if isempty(pts); e = 0; return; end
c = mean(pts, 1);
e = max(sqrt(sum((pts - c).^2, 2)));
end

function spot_panel(tl, pts, lim, ttl)
% scatter one spot, centred on its centroid, with shared axis limits.
nexttile(tl);
if isempty(pts)
    text(0.5, 0.5, '(no rays)', 'Units','normalized', ...
         'HorizontalAlignment','center'); axis off; title(ttl); return;
end
c = mean(pts, 1);
plot(pts(:,1)-c(1), pts(:,2)-c(2), 'k.', 'MarkerSize', 4);
axis equal; xlim([-lim lim]); ylim([-lim lim]); grid on; box on;
xlabel('x (BaseUnits)'); ylabel('y (BaseUnits)'); title(ttl);
end

function mark(cx, cy, peak_xy, show_peak)
% overlay the grid centre (star location) and, optionally, the planet peak.
hold on;
plot(cx, cy, 'wx', 'MarkerSize', 9, 'LineWidth', 1.2);   % star / grid centre
if show_peak
    plot(peak_xy(1), peak_xy(2), 'c+', 'MarkerSize', 14, 'LineWidth', 1.6);
end
hold off;
end

function v = ternary(cond, a, b)
if cond; v = a; else; v = b; end
end
