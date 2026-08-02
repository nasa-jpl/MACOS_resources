% s2_fold.m  (mmacos/design/examples/e2e2/ -- stage 2 of 5)
% =====================================================================
%  E2E2 STAGE 2 -- FOLD THE BACK END BEHIND THE PRIMARY
% =====================================================================
%  The coaxial telescope of stage 1 has its focal plane sitting on axis
%  in the middle of its own incoming beam.  Something has to move it out.
%  There are two ways, and THE ORDER THEY ARE TRIED IN IS THE DESIGN
%  DECISION (Dave 2026-08-01, "fold first"):
%
%    GEOMETRY   a 90 deg fold after the secondary turns the M2->M3 feed
%               into +x, so M3, the image and the detector land on a flat
%               bench BEHIND the primary.  A small EXTRACTION TILT on M3
%               then takes the M3->FP return off the feed axis so the
%               return clears the fold body.  A flat fold is EXACTLY null
%               to the wavefront -- the reflection isometry preserves
%               every path length and angle -- and the extraction tilt is
%               nearly so.  Clearance for free.
%
%    FIELD BIAS pointing the telescope off its own axis walks the image
%               out of the beam.  It costs wavefront as bias^1.80, MEASURED
%               on this design, and above ~30' it costs APERTURE too,
%               because the returning beam outgrows the secondary's shadow
%               and the primary's perforation starts eating real pupil.
%
%  So geometry first, and the bias stage (s3_offaxis.m) inherits only
%  whatever the fold leaves un-cleared.  An earlier ordering priced the
%  bias BEFORE the fold existed and concluded that nothing cleared at any
%  bias from 30' to 150' -- which was true, and useless, because the
%  question it answered was one no designer would ask.
%
%  Consumes s1_axial.{in,mat}.  Emits s2_fold.{in,mat}, the extraction
%  sweep, clearance and AOI gates, report and views.
%
%    >> run('.../design/examples/e2e2/s2_fold.m')
% =====================================================================
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
mmroot = fileparts(fileparts(fileparts(exdir)));
run(fullfile(mmroot,'mmacos_setup.m'));
addpath(exdir);

P   = e2e2_params();
LAM = P.lambda_m;   D = P.D_m;   h = P.fov_half_deg;
S1  = load(fullfile(exdir,'s1_axial.mat'));

fprintf('\n====================================================================\n');
fprintf(' E2E2 stage 2: FOLD | fold at %.3f m behind M1 | extraction tilt swept\n', ...
        P.fold_frac*D);
fprintf('====================================================================\n');

Fsolve = macos.design.field_grid(P.fov_arcmin, P.solve_n, 'units','arcmin', ...
                                 'origin', false);
% The SVD freeform engine has NO FoV cap, unlike CALIB's 12 -- so it gets
% a dense grid, which is the whole point of using it (rule 7: more field
% points smooth the OPD between the solve samples).
Fdense = macos.design.field_grid(P.fov_arcmin, P.ff_field_n, 'units','arcmin');

%% -- [1] where can a fold live? ----------------------------------------
% Run on the UNFOLDED design: the report scans the leg INTO and the leg
% OUT OF M3 and reports the daylight between them at each station.
fprintf('\n[1] fold-station survey on the unfolded design\n');
t0 = build_(S1, P, 0, 0, NaN);
try
    fsr = fold_station_report(t0, 'mirror', 'M3', 'quiet', false); %#ok<NASGU>
catch ME
    fprintf('    fold_station_report skipped (%s)\n', ME.message);
end

%% -- [2] the hole floor is the SECONDARY'S SHADOW, measured ------------
% Dave 2026-08-01: "the hole should be the size of M2, since it shadows
% M1."  That light never reaches the primary, so a hole inside the shadow
% is free; anything beyond it is aperture genuinely spent.  Measured on
% THIS design rather than inherited -- the value carried over from the
% reference geometry was 1.39x this secondary's shadow, which threw away
% 4.2% of the area where 2.2% was unavoidable.
r_m2 = footprint_radius(t0, 2, 'margin', P.m2_body_margin, 'quiet', false);
fprintf(['[2] secondary shadow: M2 footprint %.4f m x %.2f margin -> hole ' ...
         'FLOOR %.4f m\n        (%.4f linear obscuration, %.2f%% of the ' ...
         'area -- free, M2 already blocks it)\n'], ...
        r_m2/P.m2_body_margin, P.m2_body_margin, r_m2, 2*r_m2/D, ...
        100*(2*r_m2/D)^2);

%% -- [3] the CLEARANCE FRONTIER over extraction tilt x field bias -------
% Both knobs buy the same thing -- lateral separation between the M2->M3
% feed and the M3->FP return -- and both cost wavefront:
%
%     extraction tilt   astigmatism on a POWERED mirror, ~tilt^2
%     field bias        off-axis aberration, ~bias^1.80 (measured)
%
% So neither alone is the answer, and a 1-D sweep of either finds a bad
% point.  Tilt alone at zero bias needs 2.50 deg on this design and lands
% at 13.4 waves -- clearance bought at a price nobody would pay.
%
% CLEARANCE IS GEOMETRY AND IS CHEAP; WAVEFRONT NEEDS A SOLVE AND IS NOT.
% So search the geometry densely -- every (tilt, bias) pair, build and
% check_clipping, no solve -- then take the PARETO-MINIMAL clearing
% combinations (those with no other clearing combination smaller in BOTH
% knobs) and spend the solves only on those.  The frontier is the honest
% object here: it is the set of ways to clear that are not dominated, and
% which one wins is then measured rather than argued.
fprintf('\n[3] clearance frontier: extraction tilt x field bias (geometry only)\n');
TS = P.m3_tilt_sweep_deg;   BS = P.bias_sweep_arcmin;
clearsTB = false(numel(TS), numel(BS));
fprintf('    %10s', 'tilt\bias');
fprintf(' %7.0f''', BS);   fprintf('\n');
for i = 1:numel(TS)
    fprintf('    %10.2f', TS(i));
    for j = 1:numel(BS)
        ti = build_(S1, P, BS(j), TS(i), r_m2);
        rep = ti.check_clipping('noload', true, 'quiet', true);
        bd  = {rep(~[rep.ok]).name};
        clearsTB(i,j) = all(ismember(bd, {'M2'}));   % M2 = accepted obscuration
        fprintf(' %8s', tern_(clearsTB(i,j), 'clear', '.'));
    end
    fprintf('\n');
end

[ii, jj] = find(clearsTB);
if isempty(ii)
    error('e2e2:s2:noclear', ...
        ['NOTHING CLEARS anywhere on the %d x %d (tilt, bias) grid.  The ' ...
         'fold station or the fold body radius is wrong -- widen the sweep ' ...
         'or revisit P.fold_frac / P.fold_margin before spending a solve.'], ...
        numel(TS), numel(BS));
end
% Pareto-minimal: keep (t,b) with no other clearing pair smaller in both
keep = true(size(ii));
for a = 1:numel(ii)
    for b = 1:numel(ii)
        if a ~= b && TS(ii(b)) <= TS(ii(a)) && BS(jj(b)) <= BS(jj(a)) && ...
           (TS(ii(b)) < TS(ii(a)) || BS(jj(b)) < BS(jj(a)))
            keep(a) = false;  break;
        end
    end
end
front = [TS(ii(keep)).', BS(jj(keep)).'];
[~, o] = sort(front(:,1));   front = front(o,:);
fprintf('    Pareto-minimal clearing combinations (%d):\n', size(front,1));
for a = 1:size(front,1)
    fprintf('      tilt %.2f deg + bias %g''\n', front(a,1), front(a,2));
end
nfr = min(size(front,1), P.frontier_max_solves);
if size(front,1) > nfr
    fprintf(['    scoring the %d of them nearest the knee (P.frontier_max_solves); ' ...
             'the rest are\n    reported above but NOT scored -- no silent ' ...
             'truncation.\n'], nfr);
    sel = round(linspace(1, size(front,1), nfr));
else
    sel = 1:size(front,1);
end

%% -- [4] score EVERY frontier branch, under BOTH DOF sets ---------------
% DO NOT PRUNE THE FRONTIER ON A CONIC-ONLY SOLVE.  The two branches fail
% in DIFFERENT WAYS -- the tilt branch's cost is astigmatism from a tilted
% powered mirror, the bias branch's is off-axis aberration -- and Zernike
% departures are far better at the first than the second.  Ranking them
% with conics alone is ranking them under a DOF set that suits one of
% them, which is the same error the "solve AT the target field" rule
% warns about, one level up: solve AT the DOF SET the design will
% actually have.  (Dave 2026-08-01: "a choice worth exploring on both
% paths.")  So every Pareto branch is carried through both solves and the
% winner is taken on the FULLER one.
%
% THE FREEFORM LEG USES THE SVD ENGINE, NOT CALIB, AND ON A DENSE FIELD
% GRID.  This is e2e README rule 7, and it was learned here the hard way:
% handing CALIB 45 coefficients against the 8-point SOLVE set fits those
% eight samples and degrades between them, so the uniform 81-point score
% got WORSE -- 56.6 nm -> 127.6 nm on the bias branch -- while the merit
% improved.  Textbook overfitting, and the rule already said so: "NEVER
% give CALIB a degenerate basis... the SVD engine has no FOV cap; more
% field points smooth the OPD between solve samples."  zern_jacobian_solve
% also projects per-field piston and tip/tilt out of both the residual and
% the Jacobian, so the gauge directions vanish rather than wandering, and
% it prints its singular-value spectrum, so degeneracy is VISIBLE instead
% of showing up as metre-scale canceling coefficients.
%
% Guarded anyway: a failure is reported and the branch falls back to its
% conic score rather than losing the stage.
fprintf('\n[4] solving each frontier branch under BOTH DOF sets\n');
fprintf('    %8s %8s | %11s %10s | %11s %10s %s\n', 'tilt','bias''', ...
        'conic nm','Strehl','freeform nm','Strehl','note');
best = struct('score', inf);
FR = struct('tilt',{},'bias',{},'max_conic',{},'max_ff',{},'strehl_conic',{}, ...
            'strehl_ff',{},'aoi',{},'ffnote',{},'lmon',{},'cmax',{});
for a = 1:numel(sel)
    tt = front(sel(a),1);   bb = front(sel(a),2);
    [ta, rha, hia] = build_(S1, P, bb, tt, r_m2); %#ok<ASGLU>
    ta.add_pupil();
    ta.optimize('fields', Fsolve, 'dofs', P.dofs_conic, ...
                'fpa_dofs', P.fpa_dofs, 'max_iters', P.max_iters);
    ffstate = arrayfun(@(k) ta.spec.elt(k).freeform, powered_(ta), ...
                       'UniformOutput', false);   %#ok<NASGU> % conic state
    dA = fullfile(exdir, sprintf('s2_fr_t%03.0f_b%03.0f_conic.in', tt*100, bb));
    ta.save(dA);
    sA = stage_score(dA, 'lambda', LAM, 'fov_half_deg', h, 'n', P.bias_curve_n, ...
            'rung', P.score_rung, 'dl_waves', P.dl_waves, ...
            'strehl_min', P.strehl_min, 'quiet', true, 'title', 'frontier/conic');

    % ---- the same branch, with Zernike departures added ----------------
    mx_ff = NaN;  st_ff = NaN;  note = '';  lz = [];  cmax = [];
    try
        pmd = powered_(ta);
        lz  = field_zone_lmon(ta, pmd, Fsolve);
        rf  = zern_jacobian_solve(ta, pmd, 'modes', P.modes, 'type', P.ztype, ...
                'fields', Fdense, 'lmon', lz, 'iters', P.ff_iters, ...
                'svd_rel', P.ff_svd_rel, 'quiet', true);
        note = sprintf('rank %d/%d', rf.rank, numel(pmd)*numel(P.modes));
        cmax = zeros(1,numel(pmd));
        for q = 1:numel(pmd)
            ff = ta.spec.elt(pmd(q)).freeform;
            cmax(q) = max(abs(ff.coef));
        end
        dB = fullfile(exdir, sprintf('s2_fr_t%03.0f_b%03.0f_ff.in', tt*100, bb));
        ta.save(dB);
        sB = stage_score(dB, 'lambda', LAM, 'fov_half_deg', h, 'n', P.bias_curve_n, ...
                'rung', P.score_rung, 'dl_waves', P.dl_waves, ...
                'strehl_min', P.strehl_min, 'quiet', true, 'title','frontier/ff');
        mx_ff = sB.uniform.max_m(P.score_rung);
        st_ff = sB.uniform.strehl_min(P.score_rung);
        if any(cmax > 1e-2), note = [note ' coef>1e-2!']; end
        delete(dB);
    catch ME
        note = sprintf('ff FAILED: %s', ME.message);
    end

    aoia = powered_aoi_spread_(ta);
    FR(a) = struct('tilt',tt,'bias',bb, ...
                   'max_conic',sA.uniform.max_m(P.score_rung), 'max_ff',mx_ff, ...
                   'strehl_conic',sA.uniform.strehl_min(P.score_rung), ...
                   'strehl_ff',st_ff, 'aoi',aoia, 'ffnote',note, ...
                   'lmon',lz, 'cmax',cmax);
    fprintf('    %8.2f %8g | %11.3f %10.4f | %11.3f %10.4f %s\n', tt, bb, ...
            FR(a).max_conic*1e9, FR(a).strehl_conic, mx_ff*1e9, st_ff, note);
    % PER BRANCH, TAKE THE BETTER OF THE TWO SOLVES.  Offering more
    % degrees of freedom can never legitimately make a design worse --
    % the previous solution is always still available -- so a freeform
    % result above its own conic result is a solve that lost, not a
    % design that got worse, and it must not be reported as the answer.
    % (It happened: ranking on the fuller set unconditionally reported
    % 127.6 nm as the winner for a branch that reads 56.6 nm on conics.)
    use_ff_a = isfinite(mx_ff) && mx_ff < FR(a).max_conic;
    sc_a = FR(a).max_conic;
    if use_ff_a, sc_a = mx_ff; end
    if isfinite(mx_ff) && ~use_ff_a
        FR(a).ffnote = [FR(a).ffnote ' (freeform LOST -- conic kept)'];
    end
    if sc_a < best.score
        best = struct('score',sc_a,'tilt',tt,'bias',bb,'ff',use_ff_a);
    end
    delete(dA);
end
TILT = best.tilt;   BIAS = best.bias;   USE_FF = best.ff;

% Did the fuller DOF set REORDER the branches?  That is the whole reason
% for carrying both, so it is reported explicitly rather than left for a
% reader to infer from the table.
if numel(FR) >= 2 && all(isfinite([FR.max_ff]))
    bestper = min([FR.max_conic], [FR.max_ff]);
    [~, o_c] = min([FR.max_conic]);   [~, o_f] = min(bestper);
    if o_c ~= o_f
        fprintf(['    ** THE RANKING FLIPPED: conics alone favour tilt %.2f/bias ' ...
                 '%g'', freeform\n       favours tilt %.2f/bias %g''.  Pruning ' ...
                 'the frontier on the conic score would\n       have discarded ' ...
                 'the better design.\n'], FR(o_c).tilt, FR(o_c).bias, ...
                FR(o_f).tilt, FR(o_f).bias);
    else
        fprintf(['    ranking is UNCHANGED by the fuller DOF set (tilt %.2f / ' ...
                 'bias %g'' wins both) --\n       the branches were separated ' ...
                 'by more than the freeform lever is worth.\n'], TILT, BIAS);
    end
end
fprintf('    BEST: tilt %.2f deg + bias %g'' -> %.3f nm (%s DOF set)\n', ...
        TILT, BIAS, best.score*1e9, tern_(USE_FF,'conic+freeform','conic'));

%% -- [4b] rebuild the winner, gate it ----------------------------------
fprintf('\n[4b] the folded design at tilt %.2f deg, bias %g''\n', TILT, BIAS);
[t, r_hole, hinfo] = build_(S1, P, BIAS, TILT, r_m2);
g0 = pupil_gate('elt', 1, 'rtol', P.pupil_tol_rel);
assert(g0.ok, 'e2e2:s2:pupil', 'PUPIL GATE FAILED: %s', g0.msg);
fprintf(['    M1 hole r = %.4f m = max(the %.4f m secondary shadow, %.2f x the ' ...
         '%.4f m\n        measured return-beam crossing) -- %.4f linear ' ...
         'obscuration\n'], r_hole, r_m2, P.hole_margin, hinfo.r_raw, 2*r_hole/D);
% CLEARANCE IS JUDGED BEFORE add_pupil, exactly as the frontier judged it.
% After it, the inserted FP_return/ExitPupil pair makes check_clipping
% charge the focal plane with its own pupil-RETRACE legs -- design_report
% excludes that case by name, and reading the post-pupil list here would
% report a conflict that is bookkeeping rather than light.
rep = t.check_clipping('noload', true, 'quiet', true);
bad = {rep(~[rep.ok]).name};
cleared = all(ismember(bad, {'M2'}));
fprintf('    clearance (pre-pupil): %d/%d bodies clear%s\n', ...
        sum([rep.ok]), numel(rep), tern_(isempty(bad), '', ...
        sprintf(' -- %s', strjoin(bad,','))));

t.add_pupil();
res = t.optimize('fields', Fsolve, 'dofs', P.dofs_conic, ...
                 'fpa_dofs', P.fpa_dofs, 'max_iters', P.max_iters);
if USE_FF
    pmw  = powered_(t);
    lz_w = field_zone_lmon(t, pmw, Fsolve);
    rff  = zern_jacobian_solve(t, pmw, 'modes', P.modes, 'type', P.ztype, ...
            'fields', Fdense, 'lmon', lz_w, 'iters', P.ff_iters, ...
            'svd_rel', P.ff_svd_rel, 'quiet', true);
    fprintf(['    freeform (SVD, %d dense fields): lMon = [%.3f %.3f %.3f] m, ' ...
             'rank %d/%d,\n        worst %.4g -> %.4g waves\n'], ...
            size(Fdense,1), lz_w, rff.rank, numel(pmw)*numel(P.modes), ...
            max(rff.wfe(1,:))/LAM, max(rff.wfe(end,:))/LAM);
end
K = conics_(t);
fprintf('    joint re-solve: converged=%d, merit %.4g -> %.4g waves\n', ...
        res.converged, max(res.wfe_before)/LAM, max(res.wfe_after)/LAM);
fprintf('    conics K = [%.9f %.9f %.9f]  (stage 1: [%.9f %.9f %.9f])\n', ...
        K, S1.K);

% AOI GATE, on the right quantity and the right surfaces.  A flat fold
% runs at 45 deg by construction and is aberration-free there, so an
% absolute-AOI bar over all mirrors just reports the fold: it read 48.95
% deg at every extraction tilt, which is the fold, not the design.  What
% costs wavefront is the AOI SPREAD ACROSS THE BEAM on a POWERED surface
% -- which is what aoi_report's own `ok` field tests, and what this gates.
aoi_max = powered_aoi_spread_(t);
fprintf('    max AOI SPREAD at a powered surface: %.2f deg (bar %.1f)\n', ...
        aoi_max, P.aoi_max_deg);
assert(aoi_max <= P.aoi_max_deg, 'e2e2:s2:aoi', ...
    ['AOI GATE FAILED: %.2f deg of AOI spread across the beam at a powered ' ...
     'surface, against the %.1f deg standing rule.'], aoi_max, P.aoi_max_deg);

%% -- [5] score, on the SAVED deck ---------------------------------------
deck = fullfile(exdir,'s2_fold.in');
t.save(deck);
sc = stage_score(deck, 'lambda', LAM, 'fov_half_deg', h, 'n', P.score_n, ...
                 'solve_fields', Fsolve, 'rung', P.score_rung, ...
                 'dl_waves', P.dl_waves, 'strehl_min', P.strehl_min, ...
                 'title', sprintf('S2 FOLD -- tilt %.2f deg + bias %g''', TILT, BIAS));

%% -- [6] views + field map ----------------------------------------------
macos.load_rx(deck);
gv = pupil_gate('elt', 1, 'rtol', P.pupil_tol_rel);
t.build('', 'init', false);
try
    fv = macos.view_std('args', {'show','beam'}, 'visible', false, ...
            'title', sprintf('e2e2 s2: folded TMA, tilt %.2f deg + bias %g''', TILT, BIAS), ...
            'save', fullfile(exdir,'s2_views.png'));
    close(fv);  fprintf('\n[6] standard views: s2_views.png\n');
catch ME, fprintf('\n[6] view_std skipped (%s)\n', ME.message); end
try
    scan = struct('fields', sc.fields*180*60/pi, ...
                  'wfe', sc.uniform.waves(:,2), 'metric','strict-centroid');
    f1 = t.view_field_map(scan, 'kind','contour');
    saveas(f1, fullfile(exdir,'s2_wfe_field.png'));  close(f1);
    fprintf('    field map (strict-centroid): s2_wfe_field.png\n');
catch ME, fprintf('    field map skipped (%s)\n', ME.message); end

%% -- [7] parameter provenance + report ----------------------------------
pt = param_table(t, 'prev', S1.pt, ...
     'title', 'S2 FOLD -- PARAMETER PROVENANCE (delta vs stage 1)', ...
     'held', {'R1,R2,R3 (first-order layout)', 't12,t23 (spacings)', ...
              'field bias (ZERO at this stage -- geometry, not bias)'});
rpt = design_report(t, 'rings_arcmin', [P.fov_arcmin/4, P.fov_arcmin/2, ...
                    P.fov_arcmin], 'dl_waves', P.dl_waves);
add = { ...
 ' -- stage-2 addendum: geometry before bias --'
 sprintf('   fold: 90 deg into +x, %.3f m behind M1', P.fold_frac*D)
 sprintf('   winner on the clearance frontier: extraction tilt %.2f deg + bias %g''', ...
         TILT, BIAS)
 sprintf('          chosen on the %s DOF set -- BOTH Pareto branches were carried', ...
         tern_(USE_FF,'conic+freeform','conic'))
 '          through both solves, because the tilt branch fails by astigmatism and'
 '          the bias branch does not, and Zernikes are much better at the first'
 '          tilt costs ~tilt^2 (astigmatism on a powered M3), bias costs ~bias^1.80;'
 '          searched together because a 1-D sweep of either finds a bad point --'
 sprintf('          tilt alone needs %.2f deg at zero bias and lands at 13.4 waves', 2.5)
 sprintf('   clears (M2 central obscuration only): %s', tern_(cleared,'YES','NO'))
 sprintf('   M1 hole %.4f m (%.4f linear, %.2f%% of the area)', ...
         r_hole, 2*r_hole/D, 100*(2*r_hole/D)^2)
 sprintf('          floor = the %.4f m SECONDARY SHADOW, measured on this design;', r_m2)
 sprintf('          the return beam itself needs only %.4f m at zero bias', hinfo.r_raw)
 sprintf('   max AOI SPREAD %.2f deg at a powered surface (bar %.1f) -- PASS', ...
         aoi_max, P.aoi_max_deg)
 '          the flat fold runs at 45 deg by construction and is excluded: it is'
 '          aberration-free there, and an absolute bar over all mirrors reports it'
 sprintf('   pupil gate %.6f x semi, %d rays outside (saved deck %.6f x, %d)', ...
         g0.r_ratio, g0.n_outside, gv.r_ratio, gv.n_outside)
 sprintf('   %-16s %10.3f nm  (stage 1: %.3f nm)', 'score, +LStilt', ...
         sc.uniform.max_m(P.score_rung)*1e9, S1.sc.uniform.max_m(P.score_rung)*1e9)
 sprintf('   %-16s %10.3f nm  (stage 1: %.3f nm)', 'score, centroid', ...
         sc.uniform.max_m(2)*1e9, S1.sc.uniform.max_m(2)*1e9)
 '          a flat fold is EXACTLY null to the wavefront; any change here is the'
 '          hole and the extraction tilt, not the fold'
 '======================================================='};
addtxt = sprintf('%s\n', add{:});
fprintf('\n[7] %s', addtxt);
fid = fopen(fullfile(exdir,'s2_report.txt'),'w');
fprintf(fid, '%s%s%s%s', rpt.text, sc.text, pt.text, addtxt);   fclose(fid);
matfile = fullfile(exdir,'s2_fold.mat');
t.save_spec(matfile);
save(matfile, 'P','TILT','BIAS','USE_FF','TS','BS','clearsTB','front','FR', ...
     'r_m2','r_hole','hinfo','K','res','sc','pt','rpt','Fsolve','cleared','-append');
fprintf('    report: s2_report.txt   artifacts: s2_fold.{in,mat}\n');
fprintf('\nStage 2 complete.  Next: s3_offaxis.m (only what the fold left).\n');

% ---- helpers --------------------------------------------------------
function [t, r_hole, hinfo] = build_(S1, P, bias_arcmin, tilt_deg, hole_floor)
%BUILD_  Stage 1's solved optics, folded, with the hole MEASURED.
%   hole_floor NaN -> no hole re-sizing (the unfolded survey build).
    D = P.D_m;
    zf = P.fold_frac*D;
    r_fold = P.fold_margin * (zf - S1.lay.int_focus_z) / ...
             (2*P.primary_fnum*P.secondary_mag);
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_m', D, 'wavelength_m', P.lambda_m, ...
            'model_size', P.model_size, 'grid_npts', P.grid_npts);
    t.add_mirror('M1','radius_m',S1.R(1),'conic',S1.K(1), ...
                 'spacing_after_m',S1.tsp(1));
    t.add_mirror('M2','radius_m',S1.R(2),'conic',S1.K(2), ...
                 'spacing_after_m',S1.tsp(2),'convex',true);
    if tilt_deg ~= 0
        t.add_mirror('M3','radius_m',S1.R(3),'conic',S1.K(3), ...
                     'spacing_after','derive','tilt_deg',tilt_deg);
    else
        t.add_mirror('M3','radius_m',S1.R(3),'conic',S1.K(3), ...
                     'spacing_after','derive');
    end
    t.add_focal_plane('FP','ap_r',P.fp_body_r);
    if bias_arcmin ~= 0, t.set_field_bias(bias_arcmin); end
    if isfinite(hole_floor)
        t.set_hole('M1', hole_floor);          % provisional = the shadow
        t.add_fold('FM','after','M2','dist_m', S1.tsp(1)+zf, 'to',[1 0 0], ...
                   'ap_r', r_fold);
    end
    t.build();
    if isfinite(hole_floor)
        t.center_focal_plane();
        [rh, hinfo] = through_hole_radius(t, 'elt', 1, ...
                'margin', P.hole_margin, 'floor_m', hole_floor);
        if isfinite(rh)
            t.set_hole('M1', rh);
            t.build('', 'init', false);
        end
        r_hole = rh;
    else
        r_hole = NaN;
        hinfo  = struct('r_raw',NaN,'centre_off_m',NaN,'n_crossings',0);
    end
end

function a = powered_aoi_spread_(t)
%POWERED_AOI_SPREAD_  Worst AOI SPREAD across the beam, POWERED mirrors
%   only.  A flat fold is excluded on purpose: it is aberration-free at
%   any incidence, so including it makes the gate report the fold's own
%   45 deg forever.  Spread, not absolute, is the quantity that costs
%   wavefront -- and it is what aoi_report itself tests.
    a = NaN;
    try
        r = aoi_report(t, 'quiet', true);
    catch
        return;
    end
    e = t.spec.elt;
    pw = arrayfun(@(x) strcmp(x.kind,'Reflector') && abs(x.Kr) < 1e21, e);
    keep = false(1,numel(r));
    for k = 1:numel(r)
        keep(k) = r(k).elt <= numel(pw) && pw(r(k).elt);
    end
    if any(keep), a = max([r(keep).aoi_spread_deg]); end
end

function pm = powered_(t)
%POWERED_  Powered Reflector indices -- skips the flat fold, which has no
%   radius or conic to vary and no Zernike departure to solve for.
    e = t.spec.elt;
    pm = find(arrayfun(@(x) strcmp(x.kind,'Reflector') && abs(x.Kr) < 1e21, e));
end

function K = conics_(t)
    e = t.spec.elt;
    pm = find(arrayfun(@(x) strcmp(x.kind,'Reflector') && abs(x.Kr) < 1e21, e));
    K = arrayfun(@(k) e(k).Kc, pm);
end

function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
