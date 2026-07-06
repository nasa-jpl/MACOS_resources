% tma_centered_foldfp.m  (mmacos/design/examples/tma_centered/)
% =====================================================================
%  THE FOLDED, FIELD-BIASED CENTERED TMA -- the buildable j18.
% =====================================================================
%  Companion to tma_centered_fold_search.m (run that FIRST -- it scans
%  the M3-pushback x field-bias ladders and saves the compliant
%  geometry).  This script rebuilds the chosen design, balances the
%  conics over the science ring AROUND THE BIASED FIELD CENTER,
%  re-verifies the full 3-D clearance, and saves the deliverables.
%
%  The design story (Dave's knobs + packaging, 2026-07-05):
%   - the coaxial Korsch's FP lands ON AXIS in the middle of the beam;
%   - a FIELD BIAS (source tilt) separates the feed and return bundles;
%   - M3 pushed back opens fold room behind the primary;
%   - a FLAT FOLD in the M2->M3 FEED (add_fold -- exactly WFE-neutral)
%     turns the beam 90 deg into +x with its normal in the X-Z plane,
%     so M3, the image, and the FP all sit on a FLAT X-Y BENCH behind
%     the primary: no body left in any beam except M2's own accepted
%     central obscuration, and the whole back end packages flat.
%
%  Run:  >> run('.../tma_centered/tma_centered_foldfp.m')
% =====================================================================
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/src'));
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

gfile = fullfile(exdir,'tma_centered_fold_geometry.mat');
if ~exist(gfile,'file')
    error('run tma_centered_fold_search.m first (no %s).', gfile);
end
g = load(gfile);  c = g.chosen;
DIFFRACTION_LIMIT = 0.071;                   % Marechal, waves RMS

fprintf('====================================================================\n');
fprintf(' Folded centered TMA | M3 back %.1f m, bias %.1f'', fold at z=%.2f m\n', ...
        c.dm3, c.bias, c.zfold);
fprintf('====================================================================\n');

%% -- [1] rebuild the chosen geometry ----------------------------------
t = macos.design.Telescope('family','TMA','aperture_diameter_m',c.D, ...
        'model_size',256,'wavelength_m',c.lambda,'grid_npts',41);
t.add_mirror('M1','radius_m',c.R(1),'spacing_after_m',c.TBET(1),'conic',c.K(1));
t.add_mirror('M2','radius_m',c.R(2),'spacing_after_m',c.TBET(2),'convex',true, ...
             'conic',c.K(2));
t.add_mirror('M3','radius_m',c.R(3),'spacing_after','derive','conic',c.K(3));
t.add_focal_plane('FP','ap_r',c.fp_ap_r);
t.set_field_bias(c.bias);
t.add_fold('FM','after','M2','dist_m',c.fold_dist,'to',[1 0 0], ...
           'ap_r',c.fold_ap_r);
t.set_hole('M1', c.hole_r);
t.build();
t.center_focal_plane();
e = t.spec.elt;
fprintf('\n[1] chain (M3 + image + FP on the X-Y bench behind the PM):\n');
for k = 1:numel(e)
    fprintf('    %-4s %-10s Vpt=[%8.3f %8.3f %8.3f]\n', ...
            e(k).name, e(k).kind, e(k).Vpt);
end

%% -- [2] performance-first correction at the biased field --------------
% Dave 2026-07-05: optical performance over FOV.  The OPD at the bias
% point is dominated by the symmetric parent's FIELD ASTIGMATISM
% (~bias^2) -- the earlier ring balance spread 6 DOFs over an 11-point
% ring and left the bias point astigmatic.  Two corrections stack, each
% shown as its own step:
%   [2a] diagnose the raw bias-point ladder (the "before");
%   [2b] conics+ROC re-derived AT THE BIAS FIELD ONLY -- the classic
%        annular-field-anastigmat solve (three conics can null coma +
%        astig at ONE field radius; the FOV pays, stage [3] shows it);
%   [2c] freeform Zernike departures on M2+M3 -- at a single field the
%        residual is a FIXED pupil map, so a static mirror figure nulls
%        it; only the DIFFERENTIAL aberration across the field remains.
% Pointing DOFs stay out throughout (tip/tilt/decenter would re-center
% the telescope INTO the bias and put the feed back on the fold --
% check_clipping guards that trade).
ladder = @(F) wfe_field_diag(t, F, 'quiet',true);
lprint = @(tag,d) fprintf( ...
    '    %-22s raw %7.3f | -tilt %7.3f | -focus %7.3f | -astig %7.3f\n', ...
    tag, max(d.rms_raw), max(d.rms_tilt), max(d.rms_focus), max(d.rms_astig));
fprintf('\n[2] bias-point correction ladder (waves RMS @ %g um):\n', c.lambda*1e6);
d0 = ladder([0 0]);
lprint('[2a] as found:', d0);

rb1 = t.optimize('fields_arcmin',[],'dofs',[0 0 0 0 0 0 1 1], ...
                 'max_iters',120);                       %#ok<NASGU>
d1 = ladder([0 0]);
lprint('[2b] conics at bias:', d1);

% ONE mirror, center-only -- and the mirror is M1 (Dave 2026-07-05:
% a Zernike surface on M1 is fine).  M1 IS the aperture stop, so its
% figure is exactly pupil-conjugate: a field-constant correction with
% the least field damage, and it is FULLY lit (no normalization
% degeneracy).  Four lessons from the ways this failed first, kept
% visible per Dave:
%  (i)  a RING-balanced freeform trades the center away (0.061 -> 0.122
%       over a 0.5' ring, mode 4 fighting the detector conjugate);
%  (ii) TWO mirrors at a single field over-fit -- 14 DOFs for one field
%       put huge canceling figures on M2/M3 (center 0.044 but the 0.25'
%       ring collapsed 0.16 -> 0.85 waves);
%  (iii) modes normalized to a 3.3 m BODY over a 0.32 m lit patch (M2)
%       are degenerate there -- ill-conditioned OptZern nulls the
%       center with steep canceling figures that beam walk turns into
%       field damage.  'lmon' = the measured footprint conditions the
%       basis when a small-patch mirror must be used;
%  (iv) M2's conditioned modes 5:11 could not reach the post-conic
%       higher-order residual at all (0.061 -> 0.060, a no-op) -- the
%       stop surface with a deeper mode set can.
iM1 = find(strcmp({t.spec.elt.name},'M1'), 1);
rf = t.optimize_freeform(iM1, 'modes',5:15, ...
        'fields_arcmin',[], 'max_iters',100);            %#ok<NASGU>
d2 = ladder([0 0]);
lprint('[2c] + freeform M1:', d2);

% [2d] the TRUE focal plane (Dave 2026-07-06): for a biased field the
% focal plane is TILTED wrt the chief ray, and one focus point cannot
% identify the tilt -- align_focal_plane maps best-focus points over a
% FIELD GRID (2x2 for prelim analysis; 5x5 here = final design), fits
% the detector plane through the foci, and sets FP Vpt + psi from it.
% Replaces the translate-only center_focal_plane; defocus_m answers
% "do the plots show a defocused spot?" honestly, and sag_m is the
% residual FIELD-CURVATURE map the flat detector cannot follow.
NGRID = 5;  SPAN = 0.25;                     % arcmin half-span
fa = t.align_focal_plane('grid',NGRID, 'span_arcmin',SPAN);
fprintf(['[2d] true FP from %dx%d field foci (+center): ' ...
         'tilt %.3f deg wrt chief,\n' ...
         '     defocus removed %+.3f mm; field-curvature sag ' ...
         '%+.1f to %+.1f um (rms %.1f um);\n' ...
         '     best-focus blur RMS %.2e m at the field center\n'], ...
        NGRID, NGRID, fa.tilt_deg, fa.defocus_m*1e3, ...
        min(fa.sag_m)*1e6, max(fa.sag_m)*1e6, fa.fit_rms_m*1e6, ...
        fa.spot_rms_m(1));
try
    fg = figure('Visible','off');
    contourf(fa.map.thx_arcmin, fa.map.thy_arcmin, ...
             fa.map.sag_m*1e6, 15, 'LineColor','none');
    axis equal tight; colormap(parula); cb = colorbar;
    cb.Label.String = 'focus sag from fitted FP  [\mum]';
    xlabel('\theta_x  [arcmin]'); ylabel('\theta_y  [arcmin]');
    title(sprintf('field-curvature map about the %g'' bias (FP tilt %.3f\\circ)', ...
          c.bias, fa.tilt_deg));
    saveas(fg, fullfile(exdir,'tma_centered_foldfp_fpmap.png')); close(fg);
    fprintf('     field map: tma_centered_foldfp_fpmap.png\n');
catch ME, fprintf('     field map skipped (%s)\n', ME.message); end
w23 = max(d2.rms_raw)*c.lambda/2.3e-6;
fprintf(['    bias point at j18''s own 2.3 um yardstick: %.3f waves -> %s\n'], ...
        w23, ternary(w23 < DIFFRACTION_LIMIT,'DIFFRACTION-LIMITED','residual'));

%% -- [3] what the correction costs in FOV ------------------------------
% The bias-point solve nulls the aberration AT the bias; the field
% around it keeps the differential part.  Judge BLUR on the -tilt
% column: the raw number at a ring field is dominated by the field-
% dependent TILT the single-point solve no longer balances -- that is
% distortion / plate scale (calibrated out in imaging), not image
% quality.  This is the honest performance-vs-FOV curve.
fprintf(['\n[3] FOV ladder (worst on ring, waves RMS @ %g um):\n' ...
         '    %9s %9s %9s %12s\n'], c.lambda*1e6, ...
        'ring','raw','-tilt','-tilt @2.3um');
for rr = [0.25 0.5 1.0 2.5]
    dr = ladder(macos.design.field_ring(rr,'units','arcmin'));
    w1 = max(dr.rms_raw);  wt = max(dr.rms_tilt);
    fprintf('    %8.2f'' %9.3f %9.3f %12.3f%s\n', rr, w1, wt, ...
            wt*c.lambda/2.3e-6, ...
            ternary(wt*c.lambda/2.3e-6 < DIFFRACTION_LIMIT,'  <- DL @2.3um',''));
end

%% -- [4] the buildability verdict (re-checked AFTER the correction) ----
rep = t.check_clipping('noload',true);
i2 = find(strcmp({rep.name},'M2'),1);
others_ok = all([rep([1:i2-1, i2+1:end]).obstructs] == 0);
fprintf(['\n[4] M2''s central obscuration is the centered family''s accepted\n' ...
         '    price; every OTHER body %s.\n'], ...
        ternary(others_ok,'is CLEAR of every beam','still CONFLICTS'));

aoi = aoi_report(t);
pk  = packaging_report(t);
fprintf('    fold AOI %.1f deg (spread %.1f); shroud %.2f x D\n', ...
        aoi(strcmp({aoi.name},'FM')).aoi_chief_deg, ...
        aoi(strcmp({aoi.name},'FM')).aoi_spread_deg, pk.shroud_over_D);

%% -- [5] deliverables --------------------------------------------------
t.add_pupil(numel(t.spec.elt));
rxfile = fullfile(exdir,'tma_centered_foldfp.in');
matfile = fullfile(exdir,'tma_centered_foldfp.mat');
t.save(rxfile);  t.save_spec(matfile);
fprintf('\n[5] saved: %s\n           + %s\n', rxfile, matfile);

try
    % ALL figures land in the example directory (usual practice, Dave
    % 2026-07-05) -- the layout, the bench detail, AND the verification
    % views of the pupil retrace.
    % (1) layout + X-Y bench detail.  XY panels render TRUE ray
    % positions from ray_bundle.  Slice choice matters (Dave
    % 2026-07-05): the fold swaps z<->x and PRESERVES y, so the pupil-X
    % slice's spread maps into z -- invisible in an XY view -- while
    % the pupil-Y slice fans out in y along the bench and SHOWS the
    % beam width.  All XY panels start AT THE FOLD: the M1-M2-FM legs
    % project on top of the bench and bury the detail.
    ib = find(ismember({t.spec.elt.name}, {'FM','M3','FP'}));
    xx = arrayfun(@(k) t.spec.elt(k).Vpt(1), ib);
    yy = arrayfun(@(k) t.spec.elt(k).Vpt(2), ib);
    lo = min([xx yy]) - 0.6;  hi = max([xx yy]) + 0.6;
    hid  = find(strcmp({t.spec.elt.kind}, 'Return'));
    iFM  = find(strcmp({t.spec.elt.name}, 'FM'), 1);
    iFP  = find(strcmp({t.spec.elt.name}, 'FP'), 1);
    f2 = t.view_orthoviews({'YZ','XZ'},'nrays',9,'hide',hid, ...
                           'iend',iFP,'zoom',{'XY',[lo hi lo hi],[iFM iFP]}, ...
                           'zoom_fans','y');
    saveas(f2, fullfile(exdir,'tma_centered_foldfp_layout.png'));
    % (2) XY from the fold onward incl. the FP -> ExitPupil -> FP pupil
    % retrace (Returns drawn): the retrace legs must OVERLAY the M3->FP
    % band -- a Return retro-reflects (rhat = -ihat), verified by eye.
    f3 = t.view_orthoviews({'XY'},'nrays',9,'istart',iFM,'fans','y', ...
                           'zoom',{'XY',[lo hi lo hi]},'zoom_fans','y');
    saveas(f3, fullfile(exdir,'tma_centered_foldfp_xy_retrace.png'));
    fprintf(['    figures: tma_centered_foldfp_layout.png (bench detail)\n' ...
             '             tma_centered_foldfp_xy_retrace.png (pupil retrace)\n']);
catch ME, fprintf('    figures skipped (%s)\n', ME.message); end

function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
