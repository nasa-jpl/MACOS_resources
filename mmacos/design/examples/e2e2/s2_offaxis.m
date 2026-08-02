% s2_offaxis.m  (mmacos/design/examples/e2e2/ -- stage 2 of 5)
% =====================================================================
%  E2E2 STAGE 2 -- TAKE THE FIELD OFF THE AXIS
% =====================================================================
%  The coaxial telescope of stage 1 has its detector sitting in its own
%  beam.  Carrying the image off the axis is what makes the design
%  buildable, and it is the single most expensive thing the flow does to
%  the wavefront.  This stage measures that cost and then recovers it, in
%  three sub-stages that are reported separately BECAUSE the difference
%  between them is the result:
%
%    (a) COLLAPSE   stage 1's optics, frozen, pointed off axis, with only
%                   the detector re-fitted.  This is the "why we must
%                   re-solve" number and it is meant to look bad.
%    (b) CONICS     the same three conics re-solved AT the biased field,
%                   jointly with the detector.  How much of the collapse
%                   the original degrees of freedom can take back.
%    (c) RIGID      M2 and M3 decenter + tilt added to the SAME joint DOF
%                   set.  What the extra freedom is worth.
%
%  THE BIAS IS A REQUIREMENT, NOT AN OPTIMIZER OUTPUT (Dave 2026-08-01).
%  What forces it is CLEARANCE -- the image and the detector body have to
%  leave the beam -- and clearance is tested in this stage and sharpened
%  by the fold in stage 3.  Scoring bias candidates on wavefront alone is
%  degenerate: WFE grows monotonically with bias, so the smallest one
%  always wins.  So the requirement comes in as P.offset_ratio x the
%  half-field, the sweep is run to PRICE it rather than to pick it, and
%  the clearance measurement says whether it is enough.
%
%  Consumes s1_axial.{in,mat}.  Emits one .in per sub-stage, a parameter
%  table with the delta against stage 1, the bias cost curve, and the
%  field-center scan.
%
%    >> run('.../design/examples/e2e2/s2_offaxis.m')
% =====================================================================
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
mmroot = fileparts(fileparts(fileparts(exdir)));
run(fullfile(mmroot,'mmacos_setup.m'));
addpath(exdir);

P   = e2e2_params();
LAM = P.lambda_m;   D = P.D_m;   h = P.fov_half_deg;
S1  = load(fullfile(exdir,'s1_axial.mat'));

BIAS = P.offset_ratio * P.fov_arcmin;        % DERIVED here, not in params
fprintf('\n====================================================================\n');
fprintf(' E2E2 stage 2: off-axis | bias %.1f'' (%.3f deg) = %.1f x the %.3f deg half-field\n', ...
        BIAS, BIAS/60, P.offset_ratio, h);
fprintf('====================================================================\n');
fprintf(['    The bias is a REQUIREMENT (clearance), priced below and ' ...
         'sharpened in stage 3.\n']);

Fsolve = macos.design.field_grid(P.fov_arcmin, P.solve_n, 'units','arcmin', ...
                                 'origin', false);
score_ = @(deck, ttl) stage_score(deck, 'lambda', LAM, 'fov_half_deg', h, ...
        'n', P.score_n, 'solve_fields', Fsolve, 'rung', P.score_rung, ...
        'dl_waves', P.dl_waves, 'strehl_min', P.strehl_min, 'title', ttl);

%% -- [1] (a) THE COLLAPSE: stage 1 frozen, pointed off axis ------------
% Only the detector is allowed to move, and only after the bias is
% applied -- a real program would at least re-focus.  Everything else is
% stage 1's solved design, verbatim.
fprintf('\n[1] (a) COLLAPSE -- stage 1 frozen at bias %.1f'', detector re-fitted only\n', BIAS);
ta = build_from_s1_(S1, P, BIAS);
gA = pupil_gate('elt', 1, 'rtol', P.pupil_tol_rel);
assert(gA.ok, 'e2e2:s2:pupil', 'PUPIL GATE FAILED: %s', gA.msg);
fa = fit_detector_(ta, P, '(a)');
ta.add_pupil();
deckA = fullfile(exdir,'s2a_collapse.in');   ta.save(deckA);
scA = score_(deckA, sprintf('S2(a) COLLAPSE -- frozen S1 at %.1f'' bias', BIAS));

%% -- [2] (b) CONICS re-solved AT the biased field ----------------------
% SOLVE AT THE BIAS POINT.  The conic basin is path-dependent, and the
% stage-1 sweep priced getting this wrong at 3.5x; re-scoring a design
% solved elsewhere is not the same thing as solving here.
fprintf('\n[2] (b) CONICS re-solved AT the bias, jointly with the detector\n');
tb = build_from_s1_(S1, P, BIAS);
fit_detector_(tb, P, '(b)');
tb.add_pupil();
nE = numel(tb.spec.elt);
rb = tb.optimize('fields', Fsolve, 'dofs', P.dofs_conic, ...
                 'fpa_dofs', P.fpa_dofs, 'max_iters', P.max_iters);
Kb = conics_(tb);
fprintf('    converged=%d, merit %.4g -> %.4g waves;  K = [%.9f %.9f %.9f]\n', ...
        rb.converged, max(rb.wfe_before)/LAM, max(rb.wfe_after)/LAM, Kb);
deckB = fullfile(exdir,'s2b_conics.in');   tb.save(deckB);
scB = score_(deckB, 'S2(b) CONICS re-solved at the bias');

%% -- [3] (c) RIGID: M2/M3 decenter + tilt in the SAME DOF set ----------
% ONE joint solve, not a second pass on top of (b): the conics and the
% rigid DOFs trade against each other, so solving them sequentially finds
% a different -- and worse-conditioned -- point than solving them
% together.  M1 stays conic-only: it IS the stop, so its decenter and
% tilt are degenerate with pointing the whole telescope.
fprintf('\n[3] (c) RIGID -- M2/M3 decenter+tilt joined to the conic DOF set\n');
tc = build_from_s1_(S1, P, BIAS);
fit_detector_(tc, P, '(c)');
tc.add_pupil();
dofs = [P.dofs_conic; P.dofs_rigid; P.dofs_rigid];      % M1 | M2 | M3
rc = tc.optimize('fields', Fsolve, 'dofs', dofs, ...
                 'fpa_dofs', P.fpa_dofs, 'max_iters', P.max_iters);
Kc = conics_(tc);
fprintf('    converged=%d, merit %.4g -> %.4g waves;  K = [%.9f %.9f %.9f]\n', ...
        rc.converged, max(rc.wfe_before)/LAM, max(rc.wfe_after)/LAM, Kc);
rg = rigid_of_(tc);
fprintf('    rigid body reached: M2 Ydec %+8.4f mm / alpha %+8.5f deg\n', rg(1,:));
fprintf('                        M3 Ydec %+8.4f mm / alpha %+8.5f deg\n', rg(2,:));
deckC = fullfile(exdir,'s2c_rigid.in');   tc.save(deckC);
scC = score_(deckC, 'S2(c) RIGID -- conics + M2/M3 tilt/decenter, joint');

%% -- [4] what the three sub-stages actually say ------------------------
r  = P.score_rung;
tab = { 'stage 1 (on axis)',  S1.sc.uniform.max_m(r), S1.sc.uniform.max_m(2)
        '(a) collapse',       scA.uniform.max_m(r),   scA.uniform.max_m(2)
        '(b) conics',         scB.uniform.max_m(r),   scB.uniform.max_m(2)
        '(c) + rigid',        scC.uniform.max_m(r),   scC.uniform.max_m(2) };
fprintf('\n[4] the stage-2 story, max RMS over the used box:\n');
fprintf('    %-20s %12s %12s %10s\n','', '+LStilt nm','centroid nm','x stage 1');
for i = 1:size(tab,1)
    fprintf('    %-20s %12.3f %12.3f %10.2f\n', tab{i,1}, tab{i,2}*1e9, ...
            tab{i,3}*1e9, tab{i,2}/tab{1,2});
end
fprintf(['    collapse %.1fx -> conics recover %.1fx of it -> rigid a further ' ...
         '%.2fx\n'], scA.uniform.max_m(r)/tab{1,2}, ...
        scA.uniform.max_m(r)/scB.uniform.max_m(r), ...
        scB.uniform.max_m(r)/scC.uniform.max_m(r));

%% -- [5] the bias COST CURVE -- pricing the requirement ----------------
% Not a selection.  Each candidate gets the (c) recipe and is scored on a
% coarser grid; what the table shows is the exponent, which is the thing
% a program negotiating its offset actually needs.
fprintf('\n[5] bias cost curve (the (c) recipe at each candidate, %dx%d scoring grid)\n', ...
        P.bias_curve_n, P.bias_curve_n);
fprintf('    %10s %12s %12s %11s   %s\n', 'bias[arcmin]','max nm','avg nm', ...
        'minStrehl','clears?');
BS = P.bias_sweep_arcmin;
cost = nan(numel(BS),3);   clr = false(1,numel(BS));   bad = cell(1,numel(BS));
for i = 1:numel(BS)
    ti = build_from_s1_(S1, P, BS(i));
    ti.align_focal_plane('grid', 3, 'span_arcmin', P.fov_arcmin);   % seed
    ti.add_pupil();
    ti.optimize('fields', Fsolve, 'dofs', dofs, 'fpa_dofs', P.fpa_dofs, ...
                'max_iters', P.max_iters);
    di = fullfile(exdir, sprintf('s2_bias_%03darcmin.in', BS(i)));
    ti.save(di);
    F = macos.design.field_grid(P.fov_arcmin, P.bias_curve_n, 'units','arcmin');
    [L, info] = strict_ladder_deck(di, F, 'lambda', LAM);
    ok = all(isfinite(L),2);
    cost(i,:) = [max(L(ok,r)), mean(L(ok,r)), min(info.strehl(ok,r))];
    % CLEARANCE -- the criterion that actually sets the bias.  Only M2's
    % central obscuration is accepted; anything else in the beam is a
    % genuine conflict at this bias.
    rep = ti.check_clipping('noload', true, 'quiet', true);
    bd  = {rep(~[rep.ok]).name};
    bad{i} = bd;   clr(i) = all(ismember(bd, {'M2'}));
    fprintf('    %10d %12.3f %12.3f %11.4f   %s\n', BS(i), cost(i,1)*1e9, ...
            cost(i,2)*1e9, cost(i,3), clear_str_(clr(i), bd));
    delete(di);
end
gb = polyfit(log(BS(:)), log(cost(:,1)), 1);
fprintf('    measured cost of bias: RMS ~ bias^%.2f\n', gb(1));
ic = find(clr, 1);
if isempty(ic)
    fprintf(['    NO candidate clears on the unfolded design -- expected, and it ' ...
             'is the\n    argument for stage 3: the FOLD is what buys clearance, ' ...
             'not more bias.\n']);
else
    fprintf(['    least CLEARING bias in the sweep: %d'' -- stage 3 re-tests this ' ...
             'with the fold\n'], BS(ic));
end

%% -- [6] where the good field actually sits ----------------------------
% The science box need not be centred on the bias point.  Map the WFE
% over a patch wider than the box and report the centroid of the region
% inside the bar; a shift larger than the threshold is a real finding for
% stage 3 to act on (re-scoring a shifted centre undersells re-solving
% there, so this stage REPORTS rather than adopts).
fprintf('\n[6] field-centre scan over +-%g'' (%dx%d)\n', ...
        P.map_fov_arcmin, P.map_n, P.map_n);
Fm = macos.design.field_grid(P.map_fov_arcmin, P.map_n, 'units','arcmin');
[Lm, ~] = strict_ladder_deck(deckC, Fm, 'lambda', LAM);
wm = Lm(:,r);   good = isfinite(wm) & (wm <= P.dl_rms_m);
if nnz(good) >= 3
    cxy = mean(Fm(good,:),1) * 180*60/pi;
    fprintf(['    %d of %d map points inside the %.1f nm bar; their centroid is ' ...
             '(%+.2f, %+.2f)''\n'], nnz(good), numel(wm), P.dl_rms_m*1e9, cxy);
    if norm(cxy) > P.center_move_min_arcmin
        fprintf(['    -> the good region is %.2f'' off the chief.  Stage 3 should ' ...
                 'RE-SOLVE there,\n       not merely re-score (e2e s2 [4g]).\n'], ...
                norm(cxy));
    else
        fprintf('    -> the chief is already on it (%.2f'' off); nothing to move.\n', ...
                norm(cxy));
    end
else
    cxy = [NaN NaN];
    fprintf('    fewer than 3 map points inside the bar -- no centroid to report.\n');
end

%% -- [7] views + field map ---------------------------------------------
macos.load_rx(deckC);
gC = pupil_gate('elt', 1, 'rtol', P.pupil_tol_rel);
tc.build('', 'init', false);
try
    fv = macos.view_std('args', {'show','beam'}, 'visible', false, ...
            'title', sprintf('e2e2 s2: off-axis TMA, bias %.1f''', BIAS), ...
            'save', fullfile(exdir,'s2_views.png'));
    close(fv);  fprintf('\n[7] standard views: s2_views.png\n');
catch ME, fprintf('\n[7] view_std skipped (%s)\n', ME.message); end
try
    scan = struct('fields', scC.fields*180*60/pi, ...
                  'wfe', scC.uniform.waves(:,2), 'metric','strict-centroid');
    f1 = tc.view_field_map(scan, 'kind','contour');
    saveas(f1, fullfile(exdir,'s2_wfe_field.png'));  close(f1);
    fprintf('    field map (strict-centroid): s2_wfe_field.png\n');
catch ME, fprintf('    field map skipped (%s)\n', ME.message); end

%% -- [8] parameter provenance + report ---------------------------------
pt = param_table(tc, 'prev', S1.pt, ...
     'title', 'S2 OFF-AXIS -- PARAMETER PROVENANCE (delta vs stage 1)', ...
     'held', {'R1,R2,R3 (first-order layout)', 't12,t23 (spacings)', ...
              'M1 hole radius', 'M1 pose (it is the stop)'});
rpt = design_report(tc, 'rings_arcmin', [P.fov_arcmin/4, P.fov_arcmin/2, ...
                    P.fov_arcmin], 'dl_waves', P.dl_waves);
add = { ...
 ' -- stage-2 addendum --'
 sprintf('   bias %.1f'' (%.3f deg) = P.offset_ratio %.2f x the %.3f deg half-field,', ...
         BIAS, BIAS/60, P.offset_ratio, h)
 '          a REQUIREMENT priced by [5] and sharpened by the fold in stage 3'
 sprintf('   pupil gate: %.6f x semi, %d rays outside (saved deck %.6f x, %d)', ...
         gA.r_ratio, gA.n_outside, gC.r_ratio, gC.n_outside)
 sprintf('   (a) collapse   %9.3f nm  (%.1fx stage 1)', ...
         scA.uniform.max_m(r)*1e9, scA.uniform.max_m(r)/S1.sc.uniform.max_m(r))
 sprintf('   (b) conics     %9.3f nm  (recovers %.1fx of the collapse)', ...
         scB.uniform.max_m(r)*1e9, scA.uniform.max_m(r)/scB.uniform.max_m(r))
 sprintf('   (c) + rigid    %9.3f nm  (a further %.2fx)', ...
         scC.uniform.max_m(r)*1e9, scB.uniform.max_m(r)/scC.uniform.max_m(r))
 sprintf('   conic<->rigid trade: (b) K = [%.6f %.6f %.6f]', Kb)
 sprintf('                        (c) K = [%.6f %.6f %.6f]', Kc)
 '          equal wavefront from different DOF magnitudes on one compensation'
 '          branch is a PROPERTY of the joint solve, not a disagreement'
 sprintf('   measured cost of bias: RMS ~ bias^%.2f over %d''..%d''', ...
         gb(1), min(P.bias_sweep_arcmin), max(P.bias_sweep_arcmin))
 center_line_(cxy, P)
 sprintf('   budget: stage 1 left %.1f nm in quadrature; (c) uses %.1f nm of it', ...
         sqrt(max(0,(P.dl_rms_m*1e9)^2 - (S1.sc.uniform.max_m(r)*1e9)^2)), ...
         scC.uniform.max_m(r)*1e9)
 '======================================================='};
addtxt = sprintf('%s\n', add{:});
fprintf('\n[8] %s', addtxt);
fid = fopen(fullfile(exdir,'s2_report.txt'),'w');
fprintf(fid, '%s%s%s%s%s%s', rpt.text, scA.text, scB.text, scC.text, ...
        pt.text, addtxt);
fclose(fid);
matfile = fullfile(exdir,'s2_offaxis.mat');
tc.save_spec(matfile);
save(matfile, 'P','BIAS','scA','scB','scC','Kb','Kc','rg','cost','BS', ...
     'clr','gb','cxy','pt','rpt','Fsolve','-append');
fprintf('    report: s2_report.txt   artifacts: s2{a,b,c}_*.in, s2_offaxis.mat\n');
fprintf('\nStage 2 complete.  Next: s3_fold.m (fold the back end behind M1).\n');

% ---- helpers --------------------------------------------------------
function s = center_line_(cxy, P)
%CENTER_LINE_  Report the good-region centroid, or say why there is none.
%   An all-NaN centroid is not a missing measurement -- it means NO map
%   point landed inside the bar, which is itself the finding at this bias.
    if any(~isfinite(cxy))
        s = sprintf(['   good-region centroid: NONE -- no point on the ' ...
                     '+-%g'' map is inside the %.1f nm bar at this bias'], ...
                    P.map_fov_arcmin, P.dl_rms_m*1e9);
    else
        s = sprintf(['   good-region centroid: (%+.2f, %+.2f)'' -- stage 3 ' ...
                     'acts on it'], cxy);
    end
end

function fa = fit_detector_(t, P, tag)
%FIT_DETECTOR_  Seed the detector pose, and gate the fit STRUCTURALLY.
%
%   WHY THIS IS NOT A BLUR GATE.  Stage 1 gates the fit on best-focus
%   blur, because there the failure was a fit locking onto the spherical
%   caustic of an unsolved K = 0 design.  That bar does not transfer here.
%   Every sub-stage of stage 2 fits the detector on optics that are not
%   yet solved AT THIS FIELD -- sub-stage (a) deliberately never will be,
%   since being bad is its entire job -- so a large blur is the
%   measurement, not a malfunction.  On this design at 1.5 deg the fit
%   reports 2.4 mm, and that is the honest size of the off-axis image
%   stage 1's conics deliver.
%
%   Giving (a) the best detector we can find is also the CONSERVATIVE
%   choice: it makes the collapse look as small as possible, so an
%   imperfect fit overstates the recovery rather than the damage.
%
%   What IS worth gating is the structural failure -- a plane fit that
%   comes back degenerate, as it does on a rotationally symmetric design
%   where the two in-plane singular values are equal and the returned
%   normal ends up 90 deg from the arriving chief, i.e. a detector edge-on
%   to its own beam.  That is scale-free and it is what this checks.
%
%   FRAME BEFORE ANGLE.  align_focal_plane's tilt_deg is measured against
%   the ARRIVING CHIEF, which at a 1.5 deg field bias is itself far off
%   the optical axis -- the exit pupil sits near the image, so the
%   image-space chief angle is large.  A ~41 deg reading here is that
%   frame, not a 41 deg detector.  Both angles are printed, and the
%   parameter table reports both for the final design.
    fa = t.align_focal_plane('grid', 5, 'span_arcmin', P.fov_arcmin);
    nrm = fa.psi(:)/norm(fa.psi);
    ax  = acosd(min(1, abs(dot(nrm, [0;0;1]))));
    fprintf(['    %s detector fit: %.4f deg vs the ARRIVING CHIEF / %.4f deg vs ' ...
             'the AXIS,\n        defocus %+.4f mm, best-focus blur %.3f mm max ' ...
             '(reported, not gated -- see the source)\n'], ...
            tag, fa.tilt_deg, ax, fa.defocus_m*1e3, max(fa.spot_rms_m)*1e3);
    assert(all(isfinite(fa.psi)) && all(isfinite(fa.fp_vpt)), ...
        'e2e2:s2:fpfit', '%s detector fit returned a non-finite plane.', tag);
    assert(fa.tilt_deg < 89, 'e2e2:s2:fpdegenerate', ...
        ['%s DETECTOR-FIT GATE FAILED: the fitted normal is %.2f deg from the ' ...
         'arriving chief -- a detector edge-on to its own beam.  That is the ' ...
         'degenerate plane fit, not a pose.'], tag, fa.tilt_deg);
end

function t = build_from_s1_(S1, P, bias_arcmin)
%BUILD_FROM_S1_  Stage 1's SOLVED optics, rebuilt at a given field bias.
%   The conics come from stage 1, so what changes between the sub-stages
%   is only which DOFs the solve is allowed to touch -- and the bias.
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_m', P.D_m, 'wavelength_m', P.lambda_m, ...
            'model_size', P.model_size, 'grid_npts', P.grid_npts);
    t.add_mirror('M1','radius_m',S1.R(1),'conic',S1.K(1), ...
                 'spacing_after_m',S1.tsp(1));
    t.add_mirror('M2','radius_m',S1.R(2),'conic',S1.K(2), ...
                 'spacing_after_m',S1.tsp(2),'convex',true);
    t.add_mirror('M3','radius_m',S1.R(3),'conic',S1.K(3), ...
                 'spacing_after','derive');
    t.add_focal_plane('FP','ap_r',P.fp_body_r);
    t.set_hole('M1', P.M1_hole_m);
    t.set_field_bias(bias_arcmin);
    t.build();
end

function K = conics_(t)
    e = t.spec.elt;
    pm = find(arrayfun(@(x) strcmp(x.kind,'Reflector') && abs(x.Kr) < 1e21, e));
    K = arrayfun(@(k) e(k).Kc, pm);
end

function v = rigid_of_(t)
%RIGID_OF_  M2/M3 decenter (mm) and alpha tilt (deg), the frame the
%   rodgers1 study decodes CODE V's YDE/ADE into.
    e = t.spec.elt;
    pm = find(arrayfun(@(x) strcmp(x.kind,'Reflector') && abs(x.Kr) < 1e21, e));
    v = zeros(2,2);
    for i = 1:2
        k = pm(i+1);
        psi = e(k).psi(:)/norm(e(k).psi);
        v(i,:) = [e(k).Vpt(2)*1e3, atan2d(psi(2), -psi(3))];
    end
end

function s = clear_str_(ok, bad)
    if ok, s = 'YES';
    else,  s = sprintf('no  (%s)', strjoin(bad, ','));
    end
end
