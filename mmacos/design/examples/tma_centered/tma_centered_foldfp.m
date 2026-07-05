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

%% -- [2] balance the conics over the science ring at the biased field --
% ROC+conic ONLY: giving the balance tip/tilt/decenter lets CALIB re-point
% the telescope INTO the bias -- the WFE improves but the lateral
% separation the source tilt bought is silently optimized away and the
% feed beam lands back on the fold (check_clipping catches it).  Hold the
% pointing geometry; let the figures do the field work.
optF = macos.design.field_ring(c.field_rad,'units','arcmin');
rb = t.optimize('fields',optF,'dofs',[0 0 0 0 0 0 1 1],'max_iters',120);
t.center_focal_plane();                      % re-aim the detector body
wworst = max(rb.wfe_after)/c.lambda;
% the LADDER separates what the raw number mixes: field-dependent TILT
% is distortion/plate-scale (calibrated out in imaging), and FOCUS is
% the field curvature a biased ring pays (refocusable / a curved or
% stepped detector / the 3+1's field mirror).  Judge blur on -tilt,
% and the freeform-reachable floor on -focus.
dc = wfe_field_diag(t, optF, 'quiet',true);
fprintf(['\n[2] balanced over the %g''-dia ring about the %g'' bias ' ...
         '(waves RMS @ %g um):\n' ...
         '    worst raw %.3f | -tilt %.3f | -focus %.3f | -astig %.3f\n' ...
         '    (at j18''s own 2.3 um yardstick: -tilt %.3f waves -> %s)\n'], ...
        2*c.field_rad, c.bias, c.lambda*1e6, wworst, ...
        max(dc.rms_tilt), max(dc.rms_focus), max(dc.rms_astig), ...
        max(dc.rms_tilt)*c.lambda/2.3e-6, ...
        ternary(max(dc.rms_tilt)*c.lambda/2.3e-6 < DIFFRACTION_LIMIT, ...
                'DIFFRACTION-LIMITED','residual'));
fprintf(['    The residual is the price of USING the field off-axis: the\n' ...
         '    ring rides at %g''+/-%g'' where field curvature and astig grow\n' ...
         '    as r*dr, beyond what ROC+conic can null (and the pointing DOFs\n' ...
         '    that would null it re-center the telescope and put the beam\n' ...
         '    back on the fold -- check_clipping guards that).  Buy-downs:\n' ...
         '    per-field refocus / curved detector (the -focus row), freeform\n' ...
         '    M2/M3 (partial -- see tma_freeform), or a 4th powered mirror\n' ...
         '    (the tma_3plus1 route; the real wide-field answer).\n'], ...
        c.bias, c.field_rad);

%% -- [3] the buildability verdict -------------------------------------
rep = t.check_clipping('noload',true);
iM2 = find(strcmp({rep.name},'M2'),1);
others_ok = all([rep([1:iM2-1, iM2+1:end]).obstructs] == 0);
fprintf(['[3] M2''s central obscuration is the centered family''s accepted\n' ...
         '    price; every OTHER body %s.\n'], ...
        ternary(others_ok,'is CLEAR of every beam','still CONFLICTS'));

aoi = aoi_report(t);
pk  = packaging_report(t);
fprintf('    fold AOI %.1f deg (spread %.1f); shroud %.2f x D\n', ...
        aoi(strcmp({aoi.name},'FM')).aoi_chief_deg, ...
        aoi(strcmp({aoi.name},'FM')).aoi_spread_deg, pk.shroud_over_D);

%% -- [4] deliverables --------------------------------------------------
t.add_pupil(numel(t.spec.elt));
rxfile = fullfile(exdir,'tma_centered_foldfp.in');
matfile = fullfile(exdir,'tma_centered_foldfp.mat');
t.save(rxfile);  t.save_spec(matfile);
fprintf('\n[4] saved: %s\n           + %s\n', rxfile, matfile);

try
    % zoom panel: the X-Y BENCH behind the PM (fold / M3 / image / FP),
    % framed square over both bench coordinates so it survives knob
    % changes; the add_pupil Return surfaces are hidden and rays stop at
    % the FP (the pupil retrace clutters the detail view)
    ib = find(ismember({t.spec.elt.name}, {'FM','M3','FP'}));
    xx = arrayfun(@(k) t.spec.elt(k).Vpt(1), ib);
    yy = arrayfun(@(k) t.spec.elt(k).Vpt(2), ib);
    lo = min([xx yy]) - 0.6;  hi = max([xx yy]) + 0.6;
    hid  = find(strcmp({t.spec.elt.kind}, 'Return'));
    iFM  = find(strcmp({t.spec.elt.name}, 'FM'), 1);
    iFP  = find(strcmp({t.spec.elt.name}, 'FP'), 1);
    % the detail panel draws ONLY the bench legs (fold -> M3 -> FP): the
    % front-end beams otherwise project through the crop and bury it
    f2 = t.view_orthoviews({'YZ','XZ'},'nrays',9,'hide',hid, ...
                           'iend',iFP,'zoom',{'XY',[lo hi lo hi],[iFM iFP]});
    saveas(f2, fullfile(exdir,'tma_centered_foldfp_layout.png'));
    fprintf('    figure: tma_centered_foldfp_layout.png (+ X-Y bench detail)\n');
catch ME, fprintf('    figures skipped (%s)\n', ME.message); end

function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
