% s3_relay.m  (mmacos/templates/80_end_to_end/e2e2/ -- stage 3 of 4)
% =====================================================================
%  E2E2 STAGE 3 -- RELAY + FOCAL PLANE, AND THE FIELD CORRECTOR
% =====================================================================
%  Stage 2 left 56.6 nm and, more usefully, left a DIAGNOSIS of what it
%  is.  Decomposed over the used box the residual runs
%
%      raw 4.02 -> -tilt 3.99 -> -focus 1.74 -> -astig 0.66 waves
%
%  i.e. field curvature plus field astigmatism -- and the astigmatism
%  REVERSES SIGN ACROSS THE FIELD (z5 mean |coef| 6.3e-7 m against a
%  field spread of 2.8e-6 m, spread/mean 4.48).
%
%  THAT IS WHY MORE FREEFORM ON M1-M3 CANNOT HELP, and stage 2 measured
%  it doing exactly nothing useful under two different solvers.  A fixed
%  figure subtracts the SAME map at every field.  All three Korsch
%  mirrors sit near the PUPIL, so they see every field point on the same
%  patch of glass and have no field-differential authority at all.
%
%  A mirror near a FOCUS is the opposite: each field point lands on a
%  DIFFERENT part of it, so its figure acts field-dependently.  That is
%  the lever this stage adds, and it is e2e's rule 11 -- "M4 near the
%  focus is the reflective field-corrector".  The relay is therefore not
%  here to re-image (a 1:1 Offner would re-image the residual faithfully
%  and change nothing); it is here to put glass at a conjugate the
%  telescope does not have.
%
%  This stage also absorbs the old off-axis stage (Dave 2026-08-01, "fold
%  S3 into S4").  The bias is already fixed at 13' by stage 2's clearance
%  frontier; adding the relay changes the geometry, so clearance and the
%  bias are RE-CHECKED here rather than re-searched, and the rigid DOFs
%  on M2/M3 -- which stage 2 never used -- join the solve.
%
%  Consumes s2_fold.{in,mat}.  Emits s3_relay.{in,mat}, report and views.
%
%    >> run('.../templates/80_end_to_end/e2e2/s3_relay.m')
% =====================================================================
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
% NOTE: this file sits one level BELOW the other e2e2 stages, so mmroot
% needs four fileparts, not the three they use.  Its e2e2_params /
% s1_axial.mat / s2_fold.mat reaches still assume exdir == the e2e2
% directory -- pre-existing, left for the rewalk.
mmroot = fileparts(fileparts(fileparts(fileparts(exdir))));
run(fullfile(mmroot,'mmacos_setup.m'));
addpath(exdir);

P   = e2e2_params();
LAM = P.lambda_m;   D = P.D_m;   h = P.fov_half_deg;
S1  = load(fullfile(exdir,'s1_axial.mat'));
S2  = load(fullfile(exdir,'s2_fold.mat'));

fprintf('\n====================================================================\n');
fprintf(' E2E2 stage 3: RELAY | %s | field corrector at the telescope focus\n', ...
        upper(char(P.relay.type)));
fprintf('====================================================================\n');
fprintf(['    stage 2 left %.3f nm, diagnosed as FIELD-VARYING (astig ' ...
         'spread/mean 4.48).\n    Pupil-conjugate figure cannot reach it; a ' ...
         'FOCUS-conjugate mirror can.\n'], S2.sc.uniform.max_m(P.score_rung)*1e9);

Fsolve = macos.design.field_grid(P.fov_arcmin, P.solve_n, 'units','arcmin', ...
                                 'origin', false);
Fdense = macos.design.field_grid(P.fov_arcmin, P.ff_field_n, 'units','arcmin');

%% -- [0] BRANCH POINT: how big does the relay have to be? ---------------
% RECORDED BECAUSE IT COST A FAILED RUN, and because the answer depends
% on a packaging judgement rather than on optics alone.
%
% The first attempt scaled e2e's Offner along with the aperture (x 3/5:
% R 2.0 -> 1.2 m, ring h 0.25 -> 0.15 m) and the trace collapsed -- 318
% rays lost to SURFACE MISS at the convex stop mirror, then CALIB
% singular.  Scaling the relay with the APERTURE is the wrong invariant.
% A relay is sized by the IMAGE it has to accept:
%
%     image half-height = EFL x tan(half-field)
%       e2e   : 72 m x tan(2')    = 0.042 m   -- h = 0.25 m, ample
%       e2e2  : 60 m x tan(0.3 deg) = 0.314 m -- h = 0.15 m, 0.48x.  Dead.
%
% The field went up 9x in angle and the image with it; the aperture went
% DOWN.  Three ways out, all measured or measurable:
%
%   (1) SIZE THE OFFNER TO THE FIELD.  h >~ 0.35 m, and the Offner's
%       concentricity wants h << R/2, so R >~ 2.1 m and realistically
%       2.5-3 m -- a concave mirror of 1.25-1.5 m radius, comparable to
%       the 3 m primary.  Optically the cleanest answer; the cost is a
%       large and expensive relay assembly.
%
%   (2) BENCH (ZIGZAG) RELAY.  Tilted spheres, no concentricity
%       constraint, so NO ring-radius limit at all.  It pays tilt
%       astigmatism instead -- which is precisely what the near-focus
%       field corrector below exists to absorb, so the two fit together.
%
%   (3) NARROW THE FIELD.  The image scales with it directly.
%
% ADOPTED (Dave 2026-08-02): (2) AND (3) together -- the bench relay, at
% a smaller field.  The bench keeps the door open for other instruments
% on the same focus, which a ring-field Offner does not.  Set
% P.relay.type and P.fov_half_deg to change branch; (1) remains available
% by setting type "offner" and raising P.relay.offner_R/offner_h to the
% sizes above.
%
% THIS IS THE SECOND TIME THE WIDE FIELD HAS COST SOMETHING STRUCTURAL:
% it took 46%% of the wavefront budget at stage 1, and it drove the relay
% past the size of the primary here.  Both are recorded rather than
% absorbed silently.

%% -- [1] the telescope focus, and how far past it the corrector sits ----
% Measured off stage 2's own geometry, not recomputed from first order.
e2 = S2.spec.elt;
i3 = find(strcmp({e2.name},'M3'), 1);
ifp= find(strcmp({e2.kind},'FocalPlane'), 1, 'last');
b  = norm(e2(ifp).Vpt(:) - e2(i3).Vpt(:));     % M3 -> telescope focus
fprintf('\n[1] telescope focus sits %.4f m past M3; corrector %.4f m past THAT\n', ...
        b, P.relay.dpast_m);
r_beam = P.relay.dpast_m/(2*P.system_fnum);
fprintf(['    beam radius on the corrector: %.4f m at f/%.4g -- small, so each ' ...
         'field\n    point lands on its OWN patch: that is the field-conjugate ' ...
         'authority\n'], r_beam, P.system_fnum);

%% -- [2] build the folded telescope + the relay -------------------------
is_off = strcmpi(char(P.relay.type), 'offner');
if is_off
    [olegs, otilts, og] = offner_layout(P.relay.offner_R, P.relay.offner_h, ...
                                        'fno', P.system_fnum);
    Rrel  = [P.relay.corrector_R, P.relay.offner_R, P.relay.offner_R/2, ...
             P.relay.offner_R];
    rlegs = [olegs(1) - P.relay.dpast_m, olegs(2), olegs(3)];
    rtilts= [P.relay.corrector_tilt_deg, otilts];
    fprintf('    Offner: concave R=%.3f used twice + convex R/2 at the stop, concentric\n', ...
            P.relay.offner_R);
    fprintf('            convex-body daylight %.4f m (the classic vignetting check)\n', ...
            og.conv_clear_m);
else
    f5    = P.relay.dpast_m + P.relay.legs_m(1);
    Rrel  = [P.relay.corrector_R, 2*f5, 2*f5];
    rlegs = P.relay.legs_m;
    rtilts= [P.relay.corrector_tilt_deg, P.relay.tilt_deg(2:3)];
end
t = build_relay_(S1, S2, P, b, Rrel, rlegs, rtilts, is_off);
pm = powered_(t);
nrel = numel(pm) - 3;                       % relay mirrors after M1-M3
fprintf('    built: %d elements, %d powered (%d telescope + %d relay)\n', ...
        numel(t.spec.elt), numel(pm), 3, nrel);

g0 = pupil_gate('elt', 1, 'rtol', P.pupil_tol_rel);
assert(g0.ok, 'e2e2:s3:pupil', 'PUPIL GATE FAILED: %s', g0.msg);

%% -- [3] clearance + AOI, RE-CHECKED because the geometry changed -------
rep = t.check_clipping('noload', true, 'quiet', true);
bad = {rep(~[rep.ok]).name};
fprintf('\n[3] clearance at bias %g'' with the relay in: %d/%d clear%s\n', ...
        S2.BIAS, sum([rep.ok]), numel(rep), tern_(isempty(bad), '', ...
        sprintf(' -- %s', strjoin(bad,','))));
aoi0 = powered_aoi_spread_(t);
fprintf('    max AOI spread at a powered surface: %.2f deg (bar %.1f)\n', ...
        aoi0, P.aoi_max_deg);

%% -- [4] the JOINT solve -- the full DOF set, in ONE CALIB set ----------
% Conics on M1-M3, RIGID on M2/M3 (which stage 2 never used), and the
% detector.  The Offner triple is HELD: a ROC or conic solve would
% un-Offner the concentricity that zeroes its Seidel sums over the ring
% field, and the convex stop mirror is pupil-conjugate anyway (e2e rule
% 12).  The corrector's job is FIGURE, not power, and it is done in [5].
vary = pm(1:3);
dofs = [P.dofs_conic; P.dofs_rigid; P.dofs_rigid];
fprintf('\n[4] joint solve: conics M1-M3 + rigid M2/M3 + FPA, %d explicit FoV\n', ...
        size(Fsolve,1));
fprintf('    (relay held: concentricity is what makes the Offner work)\n');
t.add_pupil();
res = t.optimize('fields', Fsolve, 'elts', vary, 'dofs', dofs, ...
                 'fpa_dofs', P.fpa_dofs, 'max_iters', P.max_iters);
K = arrayfun(@(k) t.spec.elt(k).Kc, pm);
fprintf('    converged=%d, merit %.4g -> %.4g waves\n', res.converged, ...
        max(res.wfe_before)/LAM, max(res.wfe_after)/LAM);
deckA = fullfile(exdir,'s3_relay_nocorr.in');   t.save(deckA);
scA = stage_score(deckA, 'lambda', LAM, 'fov_half_deg', h, 'n', P.score_n, ...
        'rung', P.score_rung, 'dl_waves', P.dl_waves, ...
        'strehl_min', P.strehl_min, 'quiet', true, 'title','relay, no corrector');
fprintf('    relay in, corrector still flat: %.3f nm (stage 2: %.3f nm)\n', ...
        scA.uniform.max_m(P.score_rung)*1e9, S2.sc.uniform.max_m(P.score_rung)*1e9);

%% -- [5] THE FIELD CORRECTOR: freeform on the FOCUS-conjugate mirror ----
% This is the whole point of the stage.  The SVD engine on a dense grid,
% per e2e rule 7 -- and note what is being asked of it now: a mirror at a
% FIELD conjugate, where a fixed figure IS a field-dependent correction,
% instead of the pupil-conjugate mirrors stage 2 proved cannot help.
icorr = pm(4);
fprintf('\n[5] field corrector: freeform on %s (%d modes, SVD, %d dense fields)\n', ...
        t.spec.elt(icorr).name, numel(P.modes), size(Fdense,1));
lzc = field_zone_lmon(t, icorr, Fsolve);
rc  = zern_jacobian_solve(t, icorr, 'modes', P.modes, 'type', P.ztype, ...
        'fields', Fdense, 'lmon', lzc, 'iters', P.ff_iters, ...
        'svd_rel', P.ff_svd_rel, 'quiet', true);
fprintf('    lMon %.4f m, rank %d/%d, worst %.4g -> %.4g waves\n', ...
        lzc, rc.rank, numel(P.modes), max(rc.wfe(1,:))/LAM, max(rc.wfe(end,:))/LAM);
cmax = max(abs(t.spec.elt(icorr).freeform.coef));
fprintf('    max |coef| %.3e m\n', cmax);
if cmax > 1e-2
    warning('e2e2:s3:coef', ['metre/cm-scale coefficients -- the ' ...
        'canceling-pair pathology; revisit lMon.']);
end

deck = fullfile(exdir,'s3_relay.in');   t.save(deck);
sc = stage_score(deck, 'lambda', LAM, 'fov_half_deg', h, 'n', P.score_n, ...
        'solve_fields', Fsolve, 'rung', P.score_rung, 'dl_waves', P.dl_waves, ...
        'strehl_min', P.strehl_min, ...
        'title', sprintf('S3 RELAY -- %s + field corrector', upper(char(P.relay.type))));

% KEEP THE BETTER OF THE TWO, per the stage-2 rule: more DOFs must never
% make the reported design worse.
r = P.score_rung;
if scA.uniform.max_m(r) < sc.uniform.max_m(r)
    fprintf(['\n    ** the corrector LOST (%.3f nm vs %.3f nm without it) -- ' ...
             'keeping the\n       uncorrected design.  A focus-conjugate ' ...
             'mirror SHOULD have authority here,\n       so this means the ' ...
             'basis or lMon is wrong, not that the lever is absent.\n'], ...
            sc.uniform.max_m(r)*1e9, scA.uniform.max_m(r)*1e9);
    corr_won = false;
else
    corr_won = true;
    fprintf(['\n    field corrector: %.3f nm -> %.3f nm (%.2fx), and stage 2 ' ...
             'was %.3f nm\n'], scA.uniform.max_m(r)*1e9, ...
            sc.uniform.max_m(r)*1e9, scA.uniform.max_m(r)/sc.uniform.max_m(r), ...
            S2.sc.uniform.max_m(r)*1e9);
end

%% -- [6] the residual, decomposed again -- did the CHARACTER change? ----
% Stage 2's diagnosis was field-varying astigmatism.  If the corrector
% worked, that term is what should have moved.
d = wfe_field_diag(t, Fdense, 'quiet', true);
sp5 = max(d.z56(:,1)) - min(d.z56(:,1));   m5 = mean(abs(d.z56(:,1)));
fprintf('\n[6] residual decomposition (waves): raw %.4f -> -tilt %.4f -> -focus %.4f -> -astig %.4f\n', ...
        max(d.rms_raw), max(d.rms_tilt), max(d.rms_focus), max(d.rms_astig));
fprintf('    astig z5 spread/mean %.2f (stage 2: 4.48 -- >>1 means it still reverses)\n', ...
        sp5/max(m5,eps));

%% -- [7] views + field map ----------------------------------------------
macos.load_rx(deck);
gv = pupil_gate('elt', 1, 'rtol', P.pupil_tol_rel);
t.build('', 'init', false);
try
    fv = macos.view_std('args', {'show','beam'}, 'visible', false, ...
            'title', sprintf('e2e2 s3: %s relay + field corrector', ...
                             upper(char(P.relay.type))), ...
            'save', fullfile(exdir,'s3_views.png'));
    close(fv);  fprintf('\n[7] standard views: s3_views.png\n');
catch ME, fprintf('\n[7] view_std skipped (%s)\n', ME.message); end
try
    scan = struct('fields', sc.fields*180*60/pi, ...
                  'wfe', sc.uniform.waves(:,2), 'metric','strict-centroid');
    f1 = t.view_field_map(scan, 'kind','contour');
    saveas(f1, fullfile(exdir,'s3_wfe_field.png'));  close(f1);
    fprintf('    field map: s3_wfe_field.png\n');
catch ME, fprintf('    field map skipped (%s)\n', ME.message); end

%% -- [8] parameter provenance + report ----------------------------------
pt = param_table(t, 'prev', S2.pt, ...
     'title', 'S3 RELAY -- PARAMETER PROVENANCE (delta vs stage 2)', ...
     'held', {'R1,R2,R3 + spacings', 'the Offner triple (concentricity)', ...
              sprintf('field bias %g'' (set by stage 2''s clearance frontier)', S2.BIAS)});
rpt = design_report(t, 'rings_arcmin', [P.fov_arcmin/4, P.fov_arcmin/2, ...
                    P.fov_arcmin], 'dl_waves', P.dl_waves);
add = { ...
 ' -- stage-3 addendum: a conjugate the telescope does not have --'
 sprintf('   relay %s; corrector %.3f m past the telescope focus, beam radius %.4f m', ...
         upper(char(P.relay.type)), P.relay.dpast_m, r_beam)
 '   stage 2 proved pupil-conjugate figure cannot reach a field-VARYING residual'
 '   (astig reversing sign across the field, spread/mean 4.48); this stage puts'
 '   glass at a FIELD conjugate, where a fixed figure IS a field-dependent correction'
 sprintf('   %-28s %10.3f nm', 'stage 2 (fold, no relay)', S2.sc.uniform.max_m(r)*1e9)
 sprintf('   %-28s %10.3f nm', 'relay in, corrector flat', scA.uniform.max_m(r)*1e9)
 sprintf('   %-28s %10.3f nm  %s', '+ field corrector', sc.uniform.max_m(r)*1e9, ...
         tern_(corr_won,'(kept)','(LOST -- uncorrected kept)'))
 sprintf('   astig z5 spread/mean %.2f (was 4.48)', sp5/max(m5,eps))
 sprintf('   clearance %s | AOI spread %.2f deg (bar %.1f)', ...
         tern_(isempty(bad),'all clear',strjoin(bad,',')), aoi0, P.aoi_max_deg)
 sprintf('   pupil gate %.6f x semi, %d outside (saved deck %.6f x, %d)', ...
         g0.r_ratio, g0.n_outside, gv.r_ratio, gv.n_outside)
 '======================================================='};
addtxt = sprintf('%s\n', add{:});
fprintf('\n[8] %s', addtxt);
fid = fopen(fullfile(exdir,'s3_report.txt'),'w');
fprintf(fid, '%s%s%s%s', rpt.text, sc.text, pt.text, addtxt);   fclose(fid);
matfile = fullfile(exdir,'s3_relay.mat');
t.save_spec(matfile);
save(matfile, 'P','K','res','rc','sc','scA','pt','rpt','Fsolve','Fdense', ...
     'b','Rrel','rlegs','rtilts','corr_won','d','-append');
fprintf('    report: s3_report.txt   artifacts: s3_relay.{in,mat}\n');
fprintf('\nStage 3 complete.  Next: s4_score.m (final scoring + docs).\n');

% ---- helpers --------------------------------------------------------
function t = build_relay_(S1, S2, P, b, Rrel, rlegs, rtilts, is_off)
%BUILD_RELAY_  Stage 2's folded telescope with the relay chain appended.
    D = P.D_m;
    zf = P.fold_frac*D;
    r_fold = P.fold_margin * (zf - S1.lay.int_focus_z) / ...
             (2*P.primary_fnum*P.secondary_mag);
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_m', D, 'wavelength_m', P.lambda_m, ...
            'model_size', P.model_size, 'grid_npts', P.grid_npts);
    t.add_mirror('M1','radius_m',S1.R(1),'conic',S2.K(1), ...
                 'spacing_after_m',S1.tsp(1));
    t.add_mirror('M2','radius_m',S1.R(2),'conic',S2.K(2), ...
                 'spacing_after_m',S1.tsp(2),'convex',true);
    t.add_mirror('M3','radius_m',S1.R(3),'conic',S2.K(3), ...
                 'spacing_after_m', b + P.relay.dpast_m);
    t.add_mirror('M4','radius_m',Rrel(1),'spacing_after_m',rlegs(1), ...
                 'tilt_deg',rtilts(1),'conic',0);
    t.add_mirror('M5','radius_m',Rrel(2),'spacing_after_m',rlegs(2), ...
                 'tilt_deg',rtilts(2),'conic',0);
    if is_off
        t.add_mirror('M6','radius_m',Rrel(3),'spacing_after_m',rlegs(3), ...
                     'tilt_deg',rtilts(3),'convex',true,'conic',0);
        t.add_mirror('M7','radius_m',Rrel(4),'spacing_after','derive', ...
                     'tilt_deg',rtilts(4),'conic',0);
    else
        t.add_mirror('M6','radius_m',Rrel(3),'spacing_after','derive', ...
                     'tilt_deg',rtilts(3),'conic',0);
    end
    t.add_focal_plane('FP','ap_r',P.fp_body_r);
    t.set_field_bias(S2.BIAS);
    t.add_fold('FM','after','M2','dist_m', S1.tsp(1)+zf, 'to',[1 0 0], ...
               'ap_r', r_fold);
    t.set_hole('M1', S2.r_hole);
    t.build();
    t.center_focal_plane();
end

function pm = powered_(t)
    e = t.spec.elt;
    pm = find(arrayfun(@(x) strcmp(x.kind,'Reflector') && abs(x.Kr) < 1e21, e));
end

function a = powered_aoi_spread_(t)
%POWERED_AOI_SPREAD_  Worst AOI SPREAD across the beam, powered mirrors
%   only -- a flat fold runs at 45 deg by construction and is
%   aberration-free there, so including it just reports the fold.
    a = NaN;
    try, r = aoi_report(t, 'quiet', true); catch, return; end
    e = t.spec.elt;
    pw = arrayfun(@(x) strcmp(x.kind,'Reflector') && abs(x.Kr) < 1e21, e);
    keep = false(1,numel(r));
    for k = 1:numel(r), keep(k) = r(k).elt <= numel(pw) && pw(r(k).elt); end
    if any(keep), a = max([r(keep).aoi_spread_deg]); end
end

function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
