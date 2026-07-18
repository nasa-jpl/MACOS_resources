% s1_telescope.m  (mmacos/design/examples/e2e/ -- stage 1 of 6)
% =====================================================================
%  E2E STAGE 1 -- TELESCOPE DESIGN (TMA + freeform, on-axis taken
%  slightly off-axis, back end FOLDED behind the primary) with views
%  and a thorough design report.
% =====================================================================
%  From the BASIC telescope parameters in e2e_params.m (aperture, both
%  f/#s, feed magnification, packaging fractions) this runner:
%    [1] derives the first-order Korsch layout (macos.design.tma_layout
%        -- closed-form Cassegrain feed + M3 relay, f/#s are FREE
%        inputs, the radii come out);
%    [2] adds the 90-deg FOLD after M2 (Dave 2026-07-17): the feed is
%        folded into +x at a station just behind M1, so M3, the image,
%        and the focal plane all sit on a flat bench BEHIND the
%        primary.  Then SWEEPS the off-axis field bias: the image walk
%        (bias x EFL) is what separates the M3->FP return from the
%        fold body and the detector from the beams, so the LEAST bias
%        whose folded design fully clears (only M2's accepted central
%        obscuration remains) wins -- aberration grows ~bias^2;
%    [3] solves the conics at the recommended bias (CALIB multi-field);
%    [4] refines with FREEFORM Zernike departures on the three powered
%        mirrors (the "+FF"), with per-mirror field-zone normalization
%        radii (field_zone_lmon -- the sphere+Zernike solve doctrine)
%        and a coefficient-sanity verification;
%    [5] re-sizes the M1 central hole from the measured beam and fits
%        the TRUE focal plane from a grid of field foci;
%    [6] draws the standard views (macos.view_std) + WFE field map +
%        field-curvature map;
%    [7] saves the deliverables (s1_telescope.in/.mat) and verifies the
%        saved prescription standalone;
%    [8] emits the design report (design_report + a stage-1 addendum:
%        requested-vs-achieved first order, bias sweep, fold/bench
%        geometry, solve provenance) to s1_report.txt.
%  Stage 2 (s2_instrument.m) consumes s1_telescope.mat and adds the
%  imaging relay that widens the corrected field.
%
%  Run AFTER building mmacos:
%    >> run('.../design/examples/e2e/s1_telescope.m')
% =====================================================================
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/src'));
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
P = e2e_params();
D = P.D_m;  LAM = P.lambda_m;

fprintf('====================================================================\n');
fprintf(' E2E stage 1: telescope | D=%.2f m | primary f/%.3g | system f/%.3g | %g nm\n', ...
        D, P.primary_fnum, P.system_fnum, LAM*1e9);
fprintf('====================================================================\n');

%% -- [1] first-order layout from the basics (f/#s are free inputs) ----
[R, tsp, lay] = macos.design.tma_layout(D, P.primary_fnum, P.system_fnum, ...
        'secondary_mag', P.secondary_mag, ...
        'int_focus_m',   P.int_focus_frac*D, ...
        'm3_behind_m',   P.m3_behind_frac*D);
fno_int = P.primary_fnum * P.secondary_mag;
fprintf(['\n[1] layout: R=[%.4f %.4f %.4f] m  t=[%.4f %.4f] m  ', ...
         'EFL=%.2f m (f/%.2f)\n'], R(1),R(2),R(3), tsp(1),tsp(2), ...
        lay.EFL, lay.fnum);
fprintf(['    intermediate focus z=%+.3f m (f/%.1f cone -- field stop / ', ...
         'met injection); M3 z=%+.3f m\n'], lay.int_focus_z, fno_int, lay.m3_z);
fprintf('    fold FM at z=%+.3f m folds the feed into +x: back end behind M1\n', ...
        P.fold_frac*D);

%% -- [2] bias sweep: least bias whose FOLDED design fully clears ------
% Clearance judge per candidate: measure the M1 through-hole from the
% ray history, declare it, then require every body EXCEPT M2 (the
% accepted central obscuration) clear of every beam.
fprintf('\n[2] bias sweep (on-axis design taken slightly off-axis):\n');
fprintf('      bias(arcmin)   RMS WFE(waves)   clears (fold+FP)?\n');
nb  = numel(P.bias_sweep_arcmin);
wfe = nan(1,nb);  clr = false(1,nb);  pick = 0;
for i = 1:nb
    bias = P.bias_sweep_arcmin(i);
    ti = build_tma_(R, tsp, lay, P, bias);
    pm = powered_(ti);
    ti.optimize('fields_arcmin', [], ...           % bias point only
                'dofs', [0 0 0 0 0 0 0 1], 'max_iters', P.max_iters);
    macos.trace(numel(ti.spec.elt));  wfe(i) = rms_waves(macos.opd(), LAM);
    rh = m1_hole_radius_(ti, P.hole_margin);
    if isfinite(rh), ti.set_hole('M1', rh); end
    rep = ti.check_clipping('noload', true, 'quiet', true);
    bad = {rep(~[rep.ok]).name};
    clr(i) = all(strcmp(bad, 'M2'));            % only M2 may obstruct
    fprintf('      %8.1f       %10.4f       %s\n', bias, wfe(i), ...
            ternary(clr(i), 'YES', ['no  (' strjoin(bad,',') ')']));
end
% Among the CLEARING biases, pick the one whose solved WFE is BEST --
% not simply the least bias: with the M3 extraction tilt the tilt/bias
% interplay is non-monotonic (the tilt astigmatism dominates at small
% bias and a too-small bias lands the conic solve in a bad basin).
ic = find(clr);
if ~isempty(ic)
    [~, j] = min(wfe(ic));  pick = ic(j);
else
    [~, pick] = min(wfe + 1e3*(~clr));
    fprintf('    (no bias fully cleared -- taking the best candidate; inspect!)\n');
end
bias = P.bias_sweep_arcmin(pick);
fprintf('    RECOMMENDED bias = %g'' (best solved WFE among clearing biases)\n', bias);

%% -- [3] conic solve AT THE BIAS POINT (annular-field anastigmat) -----
% Three conics null spherical + coma + astig at ONE field radius (the
% bias); the spread across the science field is stage [4]'s freeform
% job.  Radii stay FIXED so the requested f/# holds.
t = build_tma_(R, tsp, lay, P, bias);
pm = powered_(t);                               % powered mirrors (FM is flat)
% BIAS CONTINUATION: the conic solve at a small bias (where the M3
% extraction-tilt astigmatism dominates) repeatably lands in a BAD LM
% basin (K3 -> -3..-4).  Seed it by solving first at the sweep's
% best-conditioned bias, then walking the bias to the picked value and
% re-solving from those conics.
[~, ib] = min(wfe);  bias_seed = P.bias_sweep_arcmin(ib);
if bias_seed ~= bias
    t.set_field_bias(bias_seed);  t.build('', 'init', false);
    t.optimize('fields_arcmin', [], ...
               'dofs', [0 0 0 0 0 0 0 1], 'max_iters', P.max_iters);
    t.set_field_bias(bias);  t.build('', 'init', false);
    fprintf('    (conics seeded by continuation from bias %g'')\n', bias_seed);
end
r3 = t.optimize('fields_arcmin', [], ...
                'dofs', [0 0 0 0 0 0 0 1], 'max_iters', P.max_iters);
% the FP still sits at the ON-AXIS paraxial focus; the biased field
% focuses mm away (pure defocus in the merit).  Put the detector at the
% TRUE biased focus and re-null the conics there.
t.align_focal_plane('grid', 3, 'span_arcmin', P.fov_arcmin/2);
r3 = t.optimize('fields_arcmin', [], ...
                'dofs', [0 0 0 0 0 0 0 1], 'max_iters', P.max_iters);
nE = numel(t.spec.elt);
macos.trace(nE);  wfe_con = rms_waves(macos.opd(), LAM);
K = arrayfun(@(k) t.spec.elt(k).Kc, pm);
fprintf('\n[3] conics @ bias %g'' (bias point, FP aligned): %.0f -> %.4f waves (K=[%.4f %.4f %.4f])\n', ...
        bias, max(r3.wfe_before)/LAM, wfe_con, K);

%% -- [4] freeform refinement (the "+FF"), doctrine-compliant ----------
% Per-mirror FIELD-ZONE Zernike normalization radii, fixed ONCE before
% the solve; the joint solve is a FIELD solve (2-D set, mirror-symmetric
% weights about the y-z plane).  Coefficient sanity is verified after.
h  = P.fov_arcmin;
F  = [ (h/2)*[1 0; 0 1; 0 -1; 1 1; 1 -1] ;
        h   *[1 0; 0 1; 0 -1; 1 1; 1 -1] ] * pi/180/60;
w  = [1, 1 + (F(:,1).' > 0)];              % thx>0 sampled once, weighted x2
lz = field_zone_lmon(t, pm, F);
fprintf('\n[4] freeform refinement: field-zone lMon = [%.3f %.3f %.3f] m\n', lz);
% [4a] joint FIELD solve on all powered mirrors.  (Seeding this from a
% pre-nulled bias point lands in a WORSE basin -- solve from the conic
% state, then clean up.)
r4 = t.optimize_freeform(pm, 'modes', P.modes, 'type', P.ztype, ...
                         'fields', F, 'weights', w, 'lmon', lz, ...
                         'max_iters', P.max_iters_ff);
fprintf('    [4a] joint field solve: worst %.4f -> %.4f waves\n', ...
        max(r4.wfe_before)/LAM, max(r4.wfe_after)/LAM);
% [4b] common-mode null on the STOP (M1): a stop-surface figure
% subtracts the SAME pupil map from every field point, so nulling the
% remaining bias-point residual there makes the telescope essentially
% perfect at its design field point and hands stage 2 a PURE
% field-differential residual -- the clean interface for the imaging
% instrument, whose field-conjugate mirrors correct exactly that.
% (The corners pay a little: the common mode partially canceled the
% differential there.  The patch spread itself is architecture-limited
% at this bias -- widening the corrected field is stage 2's job.)
ff1_pre = t.spec.elt(pm(1)).freeform;          % for the guard below
r4b = t.optimize_freeform(pm(1), 'modes', P.modes, 'type', P.ztype, ...
                          'fields_arcmin', [], 'lmon', lz(1), ...
                          'max_iters', P.max_iters);
d4 = wfe_field_diag(t, F, 'quiet', true);
% GUARD: the common-mode null trades bias-point perfection against the
% corners; when the joint solve left a larger residual the trade can
% LOSE.  Revert the M1 figure if the worst-field degraded.
if max(d4.rms_tilt) > 1.2 * max(r4.wfe_after)/LAM
    if isstruct(ff1_pre) && ~isempty(ff1_pre) && isfield(ff1_pre,'modes')
        t.set_freeform(pm(1), ff1_pre.modes, ff1_pre.coef, ...
                       'type', ff1_pre.type, 'lmon', ff1_pre.lmon);
    else
        t.set_freeform(pm(1), P.modes, zeros(1,numel(P.modes)), ...
                       'type', P.ztype, 'lmon', lz(1));
    end
    t.build('', 'init', false);
    d4 = wfe_field_diag(t, F, 'quiet', true);
    fprintf('    [4b] M1 null REVERTED (worst-field degraded; guard)\n');
end
wfe_ff = max(d4.rms_raw);  wfe_ft = max(d4.rms_tilt);
fprintf(['    [4b] M1 common-mode null: bias point %.4f -> %.4f waves; ', ...
         'worst-field %.4f raw / %.4f -tilt -> %s\n'], ...
        r4b.wfe_before(1)/LAM, r4b.wfe_after(1)/LAM, wfe_ff, wfe_ft, ...
        ternary(wfe_ft < P.dl_waves, 'DIFFRACTION-LIMITED', 'residual'));
cmax = zeros(1,numel(pm));  c3 = zeros(1,numel(pm));
for j = 1:numel(pm)
    ff = t.spec.elt(pm(j)).freeform;
    cmax(j) = max(abs(ff.coef));
    i3 = find(ff.modes == 3, 1);  if ~isempty(i3), c3(j) = ff.coef(i3); end
end
fprintf(['    coefficient sanity: max|coef| = [%.2e %.2e %.2e] m, ', ...
         'tilt mode = [%.1e %.1e %.1e] m\n'], cmax, c3);
if any(cmax > 1e-2)
    warning('e2e:s1:coef', ['metre/cm-scale Zernike coefficients -- the ', ...
        'canceling-pair pathology; revisit lMon / staging.']);
end

%% -- [5] M1 hole + the true focal plane + clearance -------------------
r_hole = m1_hole_radius_(t, P.hole_margin);
if isfinite(r_hole)
    t.set_hole('M1', r_hole);
    fprintf(['\n[5] M1 central hole: r = %.3f m (%.1fx the measured ', ...
             'through-beam at the M1 plane)\n'], r_hole, P.hole_margin);
else
    fprintf('\n[5] no beam crosses the M1 plane -- no hole needed\n');
end
fa = t.align_focal_plane('grid', 5, 'span_arcmin', min(0.25, h/2));
fprintf(['    true FP from 5x5 field foci: tilt %.3f deg, defocus removed ', ...
         '%+.3f mm,\n    field-curvature sag %+.1f to %+.1f um\n'], ...
        fa.tilt_deg, fa.defocus_m*1e3, min(fa.sag_m)*1e6, max(fa.sag_m)*1e6);
rep = t.check_clipping('noload', true, 'quiet', true);
bad = {rep(~[rep.ok]).name};
fprintf('    clearance: %d/%d bodies clear%s\n', sum([rep.ok]), numel(rep), ...
        ternary(isempty(bad), '', [' (' strjoin(bad,',') ...
        ' -- M2 central obscuration is the design)']));

%% -- [6] views --------------------------------------------------------
try
    fv = macos.view_std('args', {'show','beam'}, 'visible', false, ...
            'title', sprintf(['e2e s1: D=%.1f m f/%.0f folded Korsch ', ...
                              'TMA, bias %g'''], D, P.system_fnum, bias), ...
            'save', fullfile(exdir, 's1_views.png'));
    close(fv);
    fprintf('\n[6] standard views: s1_views.png\n');
catch ME, fprintf('\n[6] view_std skipped (%s)\n', ME.message); end
try
    Fmap = macos.design.field_grid(h, 7, 'units','arcmin');
    dmap = wfe_field_diag(t, Fmap, 'quiet', true);
    scan = struct('fields', Fmap*180*60/pi, 'wfe', dmap.rms_raw(:));
    f1 = t.view_field_map(scan, 'kind', 'contour');
    saveas(f1, fullfile(exdir, 's1_wfe_field.png'));  close(f1);
    fg = figure('Visible','off');
    contourf(fa.map.thx_arcmin, fa.map.thy_arcmin, fa.map.sag_m*1e6, ...
             15, 'LineColor','none');
    axis equal tight; colormap(parula); cb = colorbar;
    cb.Label.String = 'focus sag from fitted FP  [\mum]';
    xlabel('\theta_x  [arcmin]'); ylabel('\theta_y  [arcmin]');
    title(sprintf('field curvature (FP tilt %.3f\\circ)', fa.tilt_deg));
    saveas(fg, fullfile(exdir, 's1_fpmap.png'));  close(fg);
    fprintf('    field maps: s1_wfe_field.png + s1_fpmap.png\n');
catch ME, fprintf('    field maps skipped (%s)\n', ME.message); end

%% -- [7] deliverables + standalone verification -----------------------
t.add_pupil(numel(t.spec.elt));              % EP emits PropType=FarField
rxfile  = fullfile(exdir, 's1_telescope.in');
matfile = fullfile(exdir, 's1_telescope.mat');
t.save(rxfile);  t.save_spec(matfile);
fprintf('\n[7] saved: %s\n           + %s\n', rxfile, matfile);
macos.init(P.model_size);
nv = macos.load_rx(rxfile);  sv = macos.trace(nv);
rv = macos.get_ray_info(sv.nRays);
np = nnz(logical(rv.ok_pass) & logical(rv.ok_trace));
fprintf('    standalone reload: %d elts, %d/%d rays pass -> %s\n', ...
        nv, np, sv.nRays, ternary(np > 0.9*sv.nRays, 'VERIFIED', '** BROKEN **'));
t.build('', 'init', false);                  % back to the session model

%% -- [8] the design report (+ stage-1 addendum) -----------------------
fprintf('\n[8] design report:\n');
rpt = design_report(t, 'rings_arcmin', [0.25 0.5 h P.inst.fov_arcmin], ...
                    'align', fa);
e = t.spec.elt;
im3 = pm(end);  ifp = find(strcmp({e.kind},'FocalPlane'), 1, 'last');
add = { ...
 ' -- stage-1 addendum: requested vs achieved --'
 sprintf('   requested: D %.2f m | primary f/%.4g | system f/%.4g | m2 %.4g', ...
         D, P.primary_fnum, P.system_fnum, P.secondary_mag)
 sprintf('   first-order: EFL %.2f m (f/%.2f); achieved live EFL %.2f m (f/%.2f)', ...
         lay.EFL, lay.fnum, rpt.EFL_m, rpt.fno_fp)
 sprintf('   fold FM at z=%+.3f m -> bench behind M1: M3 [%.2f %.2f %.2f], FP [%.2f %.2f %.2f]', ...
         P.fold_frac*D, e(im3).Vpt, e(ifp).Vpt)
 sprintf('   bias sweep %s'' -> recommended %g'' (least whose folded design clears)', ...
         mat2str(P.bias_sweep_arcmin), bias)
 sprintf('   conic solve (bias point): K = [%.5f %.5f %.5f], %.4f waves', ...
         K, wfe_con)
 sprintf('   freeform:  lMon = [%.3f %.3f %.3f] m (field-zone), max|coef| = [%.1e %.1e %.1e] m', ...
         lz, cmax)
 sprintf(['   bias point %.4f waves (DL); worst-field @ +-%g'': %.4f raw / ', ...
          '%.4f -tilt waves @ %g nm'], r4b.wfe_after(1)/LAM, h, wfe_ff, wfe_ft, LAM*1e9)
 sprintf(['   the +-%g'' spread is the telescope''s field differential -- ', ...
          'stage 2''s instrument widens toward +-%g'''], h, P.inst.fov_arcmin)
 sprintf('   M1 hole r %.3f m | intermediate focus f/%.1f at z=%+.3f m', ...
         r_hole, fno_int, lay.int_focus_z)
 '======================================================='};
addtxt = sprintf('%s\n', add{:});
fprintf('%s', addtxt);
fid = fopen(fullfile(exdir, 's1_report.txt'), 'w');
fprintf(fid, '%s%s', rpt.text, addtxt);  fclose(fid);
fprintf('    report: s1_report.txt\n');

save(matfile, 'P', 'R', 'tsp', 'lay', 'bias', 'r3', 'r4', 'lz', 'fa', ...
     'rpt', 'pm', 'r_hole', '-append');
fprintf('\nStage 1 complete.  Next: s2_instrument.m (widen the field).\n');

% ---- helpers --------------------------------------------------------
function t = build_tma_(R, tsp, lay, P, bias)
%BUILD_TMA_  Fresh folded, field-biased Korsch TMA (K=0 seed) from the
%   first-order layout.  The 90-deg fold FM sits P.fold_frac*D behind M1
%   in the M2->M3 feed and turns it into +x (fold normal in the X-Z
%   plane), so M3 + image + FP land on a flat bench behind the primary.
    D = P.D_m;
    zf = P.fold_frac*D;
    r_fold = P.fold_margin * (zf - lay.int_focus_z) / (2*P.primary_fnum*P.secondary_mag);
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_m', D, 'wavelength_m', P.lambda_m, ...
            'model_size', P.model_size, 'grid_npts', P.grid_npts);
    t.add_mirror('M1','radius_m',R(1),'spacing_after_m',tsp(1));
    t.add_mirror('M2','radius_m',R(2),'spacing_after_m',tsp(2),'convex',true);
    t.add_mirror('M3','radius_m',R(3),'spacing_after','derive', ...
                 'tilt_deg',P.m3_tilt_deg);   % extraction tilt: return off the feed axis
    t.add_focal_plane('FP','ap_r',P.fp_body_r);
    t.set_field_bias(bias);
    t.add_fold('FM','after','M2','dist_m', tsp(1)+zf, 'to',[1 0 0], ...
               'ap_r', r_fold);
    t.build();
    t.center_focal_plane();                  % detector onto the walked image
end

function pm = powered_(t)
%POWERED_  Element indices of the powered Reflectors (skips the flat FM).
    e = t.spec.elt;
    pm = find(arrayfun(@(x) strcmp(x.kind,'Reflector') && abs(x.Kr) < 1e21, e));
end

function r = m1_hole_radius_(t, margin)
%M1_HOLE_RADIUS_  Radius the M1 perforation needs: the largest crossing
%   radius of ANY post-M2 beam leg (M2->FM, FM->M3, ...) through the M1
%   plane, from the engine ray history.  NaN when nothing crosses.
    macos.ray_hist('on');
    s = macos.trace();
    hh = macos.ray_hist(s.nRays);
    macos.ray_hist('off');
    p0 = t.spec.elt(1).Vpt(:);  ps = t.spec.elt(1).psi(:);
    r = NaN;  hit = [];
    for leg = 3:size(hh.P,3)-1                  % slot 3 = M2 (slot 1 = source)
        A = squeeze(hh.P(:,:,leg));  B = squeeze(hh.P(:,:,leg+1));
        ok = hh.ok(:,leg) & hh.ok(:,leg+1);
        for i = find(ok(:)).'
            d  = B(:,i) - A(:,i);
            dn = dot(ps, d);
            if abs(dn) < 1e-12, continue; end
            sfr = dot(ps, p0 - A(:,i)) / dn;
            if sfr > 0 && sfr < 1
                q = A(:,i) + sfr*d;
                hit(end+1) = norm(q - p0 - ps*dot(ps, q - p0));  %#ok<AGROW>
            end
        end
    end
    if ~isempty(hit), r = margin * max(hit); end
end

function w = rms_waves(W, lam)
    v = W(isfinite(W) & W ~= 0 & abs(W) < 1e30);
    if isempty(v), w = NaN; else, w = std(v)/lam; end
end
function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
