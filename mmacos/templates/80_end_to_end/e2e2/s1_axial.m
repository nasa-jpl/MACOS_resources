% s1_axial.m  (mmacos/templates/80_end_to_end/e2e2/ -- stage 1 of 5)
% =====================================================================
%  E2E2 STAGE 1 -- THE KORSCH AXIAL STARTING POINT
% =====================================================================
%  The anchor every later stage is measured against: a coaxial, on-axis
%  three-mirror anastigmat at the design point in e2e2_params.m, solved
%  jointly with its detector and scored on a uniform field grid.
%
%  Three gates run BEFORE any wavefront number is believed, in order of
%  what they would invalidate:
%    [1] the CONIC SOLVER, against the shared TMA fixture -- if the
%        closed-form Seidel solve has moved, nothing downstream means
%        anything.  A mismatch is stop-and-fix, never a widened bar.
%    [2] the FIRST-ORDER LAYOUT, against the Rodgers geometry this design
%        point scales from.  The builder derives M3's radius from the
%        f/# constraint; CODE V derived his from a `CUY UMY -0.025`
%        marginal-angle solve.  They must agree, and that agreement is
%        an independent check of the builder no de-novo layout can make.
%    [3] the PUPIL, against the declared Aperture -- greatest chord, not
%        a span.  See design/rodgers1/PACKET.md Addendum 10 for why a
%        span check cannot see a 5% pupil error.
%
%  Then: joint solve (conics + FPA in ONE CALIB DOF set, never
%  alternating), score, report, save.
%
%  Run AFTER building mmacos:
%    >> run('.../templates/80_end_to_end/e2e2/s1_axial.m')
% =====================================================================
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
mmroot = fileparts(fileparts(fileparts(exdir)));     % .../mmacos
run(fullfile(mmroot,'mmacos_setup.m'));              % src + design/src + runners
addpath(exdir);

P   = e2e2_params();
D   = P.D_m;   LAM = P.lambda_m;
h   = P.fov_half_deg;

fprintf('\n====================================================================\n');
fprintf(' E2E2 stage 1: axial Korsch TMA | D=%.2f m | f/%.4g | %g nm | box %g deg\n', ...
        D, P.system_fnum, LAM*1e9, 2*h);
fprintf('====================================================================\n');

%% -- [1] GATE: the conic solver, against the shared TMA fixture --------
% The fixture's conics null S_I/S_II/S_III by construction for its stated
% layout, so reproducing them is a check on the SOLVER, not on this
% design.  It runs first because a solver that has drifted would produce
% a plausible-looking telescope that is wrong for reasons no field map
% would reveal.
fxdir = fullfile(fileparts(mmroot), 'optical_design', 'fixtures');
if ~isfolder(fxdir)
    fxdir = fullfile(getenv('HOME'),'dev','MACOS_resources','optical_design','fixtures');
end
fx  = jsondecode(fileread(fullfile(fxdir,'tma_fixture.json')));
Rfx = [fx.layout_m.R1, fx.layout_m.R2, fx.layout_m.R3];
tfx = [fx.layout_m.t_M1_M2, fx.layout_m.t_M2_M3];
Kfx = [fx.conics.K1, fx.conics.K2, fx.conics.K3];
[Kgot, tf_fx, EFL_fx] = macos.design.seidel_seed(Rfx, tfx, fx.layout_m.D);
dK  = abs(Kgot - Kfx);
fprintf('\n[1] conic-solver fixture gate (%s)\n', 'tma_fixture.json');
fprintf('    K fixture  = [%12.7f %12.7f %12.7f]\n', Kfx);
fprintf('    K solved   = [%12.7f %12.7f %12.7f]   max|dK| = %.2e (bar %.0e)\n', ...
        Kgot, max(dK), P.fixture_tol);
fprintf('    EFL %.4f m (fixture %.4f), t_focus %.6f m (fixture %.6f)\n', ...
        EFL_fx, fx.derived.EFL_m, tf_fx, fx.layout_m.t_M3_image);
assert(max(dK) <= P.fixture_tol, 'e2e2:s1:fixture', ...
    ['CONIC SOLVER GATE FAILED: max|dK| = %.3e > %.0e.  STOP AND FIX -- ' ...
     'do NOT widen this bar and do NOT hand-edit the fixture; regenerate ' ...
     'it with optical_design/make_tma_fixture.py and explain what moved.'], ...
    max(dK), P.fixture_tol);
fprintf('    GATE PASS\n');

%% -- [2] first-order layout, and the gate against the scaled geometry --
% The f/#s and the feed magnification are the FREE inputs; the radii come
% out.  R1 and R2 follow directly from them, so the informative check is
% R3, which the builder derives from the system-f/# constraint alone.
[R, tsp, lay] = macos.design.tma_layout(D, P.primary_fnum, P.system_fnum, ...
        'secondary_mag', P.secondary_mag, ...
        'int_focus_m',   P.int_focus_m, ...
        'm3_behind_m',   P.m3_behind_m);
dR = abs(R - P.R_ref_m) ./ P.R_ref_m;
dt = abs(tsp - P.t_ref_m) ./ P.t_ref_m;
fprintf('\n[2] first-order layout (radii DERIVED from the f/#s):\n');
fprintf('    %-4s %14s %14s %12s\n','', 'derived [m]', 'reference [m]', 'rel delta');
nmR = {'R1','R2','R3'};
for k = 1:3
    fprintf('    %-4s %14.6f %14.6f %12.2e\n', nmR{k}, R(k), P.R_ref_m(k), dR(k));
end
fprintf('    %-4s %14.6f %14.6f %12.2e\n','t12', tsp(1), P.t_ref_m(1), dt(1));
fprintf('    %-4s %14.6f %14.6f %12.2e\n','t23', tsp(2), P.t_ref_m(2), dt(2));
fprintf('    paraxial EFL %.3f m (f/%.4f); intermediate focus z=%+.4f m; M3 z=%+.4f m\n', ...
        lay.EFL, lay.fnum, lay.int_focus_z, lay.m3_z);
assert(dR(3) <= P.R3_tol_rel, 'e2e2:s1:R3', ...
    ['FIRST-ORDER GATE FAILED: the builder derives R3 = %.6f m from the ' ...
     'f/%.4g constraint; the reference geometry (a CODE V marginal-angle ' ...
     'solve at the same f/#) says %.6f m -- %.2e relative, bar %.0e.  ' ...
     'The layout is wrong, or the design point is not the one it claims.'], ...
    R(3), P.system_fnum, P.R_ref_m(3), dR(3), P.R3_tol_rel);
fprintf('    GATE PASS -- the builder reproduces the reference f/#-constrained R3\n');
fprintf(['    NOTE the paraxial seeder only PLACES the focal plane; it is ' ...
         'wildly wrong on\n         long-EFL designs (96%% on the 100 m ' ...
         'Rodgers deck).  Every EFL quoted below\n         is TRACED.\n']);

%% -- [3] build the axial telescope, and gate the pupil -----------------
t  = build_axial_(R, tsp, P);
pm = powered_(t);
fprintf('\n[3] built: %d elements\n', numel(t.spec.elt));
g0 = pupil_gate('elt', 1, 'rtol', P.pupil_tol_rel);
assert(g0.ok, 'e2e2:s1:pupil', 'PUPIL GATE FAILED: %s', g0.msg);

%% -- [4] seed the conics, then solve them JOINTLY with the detector ----
% TWO SOLVE-ORDER RULES, both earned by a failed run of this stage.
%
% RULE 1 -- NEVER FIT A DETECTOR ON AN UNSOLVED DESIGN.  align_focal_plane
% fits each field's best focus FROM RAYS.  Called on the K = 0 spherical
% seed -- where this design carries 15 mm RMS of spherical aberration --
% it locks onto the spherical caustic 1.796 m from the paraxial focus,
% reports an 11 mm best-focus blur, and moves the detector there.  The
% joint solve then spent its FPA piston undoing the excursion, and
% add_pupil's FP_return and ExitPupil were left stranded at the abandoned
% station, so the SAVED deck declared an exit pupil belonging to a design
% that no longer existed.
%
% RULE 2 -- AND DO NOT FIT ONE ON A ROTATIONALLY SYMMETRIC DESIGN AT ALL.
% Fixing rule 1 (solve the conics first) does not rescue the fit here: on
% a coaxial on-axis telescope the per-field foci form a rotationally
% symmetric DISC, so the plane fit's two in-plane singular values are
% EQUAL and its basis is arbitrary.  The fit then returned a normal 90 deg
% from the arriving chief -- a detector edge-on to its own beam.  There is
% nothing for it to find: symmetry already fixes the detector normal along
% the axis, the builder's paraxial 'derive' places the station (correct to
% 20 um here, measured below), and the FPA piston in the joint DOF set
% refines what is left.  The fit belongs in stage 2, where the field bias
% breaks the symmetry and the focal surface genuinely tilts.
%
% What remains is the doctrine's own shape: ONE seeding solve to leave the
% spherical regime, then ONE joint solve.  Not an alternation -- looping
% solve <-> refit chases two objectives that need not contract (measured
% drifting 0.6-13 mm per round on the rodgers1 TMA).
fpz = @() t.spec.elt(find(strcmp({t.spec.elt.kind},'FocalPlane'),1,'last')).Vpt(3);
fprintf('\n[4] detector station provenance (m along the axis):\n');
z_build = fpz();
fprintf('    after build (builder paraxial ''derive'')   z = %+10.6f\n', z_build);

% [4a] on-axis conic seed, detector HELD -- get out of the K = 0 regime
r4a = t.optimize('fields_arcmin', [], 'dofs', P.dofs_conic, ...
                 'max_iters', P.max_iters);
fprintf('    [4a] on-axis conic seed: %.4g -> %.4g waves; z held  = %+10.6f\n', ...
        max(r4a.wfe_before)/LAM, max(r4a.wfe_after)/LAM, fpz());

% THE HOLE IS MEASURED, NOT INHERITED, and measured on the SOLVED design.
% Both halves matter.  Inherited: the reference geometry's scaled value is
% 1.39x this design's own secondary shadow, so declaring it threw away
% 4.2% of the area where 1.9% was unavoidable -- and it was simultaneously
% too SMALL for the returning beam once the field is biased, which made
% check_clipping report the primary as an obstruction.  One stale constant,
% both errors, in opposite directions.
% Solved: measured on the K = 0 spherical seed instead, M2's footprint
% reads 0.195 m against 0.222 m on the solved conics -- a 14% error in the
% shadow, and the shadow is the hole's floor.
r_m2 = footprint_radius(t, 2, 'margin', P.m2_body_margin);
[r_hole, hinfo] = through_hole_radius(t, 'elt', 1, ...
        'margin', P.hole_margin, 'floor_m', r_m2);
if isfinite(r_hole)
    t.set_hole('M1', r_hole);
    t.build('', 'init', false);
end
fprintf(['    M1 hole r = %.4f m = max(%.2f x the %.4f m measured return-beam\n' ...
         '         crossing, the %.4f m measured SECONDARY SHADOW) -> %.4f ' ...
         'linear, %.2f%% of the area\n'], r_hole, P.hole_margin, hinfo.r_raw, ...
        r_m2, 2*r_hole/D, 100*(2*r_hole/D)^2);

% [4b] the exit pupil, derived from a detector that is already correct
t.add_pupil();
nE = numel(t.spec.elt);
fprintf('    [4b] exit pupil inserted: %d elements, EP at %d, detector at %d\n', ...
        nE, t.spec.pupil.ep_elt, t.spec.pupil.fp_elt);
ir = find(strcmp({t.spec.elt.name},'FP_return'), 1);
fprintf(['         FP_return z = %+10.6f -- must TRACK the detector; a ' ...
         'mismatch means the\n         pupil pair was derived from a ' ...
         'station the design has since left\n'], t.spec.elt(ir).Vpt(3));
assert(abs(t.spec.elt(ir).Vpt(3) - fpz()) < 1e-9, 'e2e2:s1:pupilstation', ...
    ['EXIT-PUPIL STATION GATE FAILED: FP_return sits at %.6f and the ' ...
     'detector at %.6f.  add_pupil derives both from the focal-plane ' ...
     'station at the moment it runs, so this means the detector moved ' ...
     'after it -- re-order the stage.'], t.spec.elt(ir).Vpt(3), fpz());

% [4c] the JOINT solve: conics + detector tip/focus in ONE CALIB DOF set
Fsolve = macos.design.field_grid(P.fov_arcmin, P.solve_n, 'units','arcmin', ...
                                 'origin', false);
Vseed  = t.spec.elt(nE).Vpt(:);
Nseed  = t.spec.elt(nE).psi(:);  Nseed = Nseed/norm(Nseed);
res = t.optimize('fields', Fsolve, 'dofs', P.dofs_conic, ...
                 'fpa_dofs', P.fpa_dofs, 'max_iters', P.max_iters);
V = t.spec.elt(nE).Vpt(:);  N = t.spec.elt(nE).psi(:);  N = N/norm(N);
K = arrayfun(@(k) t.spec.elt(k).Kc, pm);
fp_move_mm = norm(V-Vseed)*1e3;
fprintf(['    [4c] JOINT solve (%d explicit + 1 implicit FoV): converged=%d, ' ...
         'merit %.4g -> %.4g waves\n'], size(Fsolve,1), res.converged, ...
        max(res.wfe_before)/LAM, max(res.wfe_after)/LAM);
fprintf('         conics K = [%.9f %.9f %.9f]\n', K);
ep_radius = abs(t.spec.pupil.ep_radius);
fprintf(['         detector moved %.4f mm / %.6f deg -> z = %+10.6f\n' ...
         '         (that move IS the builder''s paraxial placement error, ' ...
         'measured)\n'], fp_move_mm, ...
        acosd(min(1,abs(dot(N,Nseed)))), fpz());

% The conics this on-axis anastigmat lands on are an INDEPENDENT check on
% the whole chain.  The design point is the reference geometry scaled by
% 3/5, and conic constants are DIMENSIONLESS -- they do not scale -- so a
% correct solve must reproduce the reference stage-1 conics without ever
% having been given them.  Reported, not gated: the reference was solved
% at 1 um over a 15-point half box and this solve runs at 500 nm over a
% uniform box, so exact agreement is not the expectation.
dKr = abs(K - P.K_ref);
fprintf('    conic cross-check vs the reference design (conics are scale-free):\n');
for k = 1:3
    fprintf('      K_M%d  solved %14.9f   reference %14.9f   |d| %.2e\n', ...
            k, K(k), P.K_ref(k), dKr(k));
end

%% -- [5] save the deck, then score it on an INDEPENDENT path -----------
% Scoring reads the SAVED prescription through strict_ladder_deck, not the
% live session: a solve that games its own objective shows up here.
rxfile  = fullfile(exdir, 's1_axial.in');
matfile = fullfile(exdir, 's1_axial.mat');
t.save(rxfile);  t.save_spec(matfile);
fprintf('\n[5] saved %s\n', rxfile);

sc = stage_score(rxfile, 'lambda', LAM, 'fov_half_deg', h, 'n', P.score_n, ...
                 'solve_fields', Fsolve, 'rung', P.score_rung, ...
                 'dl_waves', P.dl_waves, 'strehl_min', P.strehl_min, ...
                 'title', 'S1 AXIAL -- FIELD SCORE');

%% -- [6] views + the WFE field map -------------------------------------
macos.load_rx(rxfile);
gv = pupil_gate('elt', 1, 'rtol', P.pupil_tol_rel);   % gate the SAVED deck too
t.build('', 'init', false);
try
    fv = macos.view_std('args', {'show','beam'}, 'visible', false, ...
            'title', sprintf('e2e2 s1: axial Korsch TMA, D=%.1f m f/%.0f', ...
                             D, P.system_fnum), ...
            'save', fullfile(exdir,'s1_views.png'));
    close(fv);
    fprintf('\n[6] standard views: s1_views.png\n');
catch ME, fprintf('\n[6] view_std skipped (%s)\n', ME.message); end
try
    scan = struct('fields', sc.fields*180*60/pi, ...
                  'wfe',    sc.uniform.waves(:,2), ...
                  'metric', 'strict-centroid');
    f1 = t.view_field_map(scan, 'kind', 'contour');
    saveas(f1, fullfile(exdir,'s1_wfe_field.png'));  close(f1);
    fprintf('    field map (strict-centroid): s1_wfe_field.png\n');
catch ME, fprintf('    field map skipped (%s)\n', ME.message); end

%% -- [7] the parameter-provenance table --------------------------------
pt = param_table(t, 'title', 'S1 AXIAL -- PARAMETER PROVENANCE', ...
     'held', {'R1,R2,R3 (first-order layout)', 't12,t23 (spacings)', ...
              'all rigid-body poses (coaxial)'});

%% -- [8] the design report ---------------------------------------------
fprintf('\n[8] design report:\n');
rpt = design_report(t, 'rings_arcmin', [h*60/4, h*60/2, h*60], ...
                    'dl_waves', P.dl_waves);
add = { ...
 ' -- stage-1 addendum: gates, provenance, and what this stage is --'
 sprintf('   design point: D %.3f m | primary f/%.4f | system f/%.4g | m2 %.6f | %g nm', ...
         D, P.primary_fnum, P.system_fnum, P.secondary_mag, LAM*1e9)
 sprintf('   GATE 1 conic solver vs tma_fixture.json : max|dK| %.2e (bar %.0e) PASS', ...
         max(dK), P.fixture_tol)
 sprintf('   GATE 2 derived R3 vs the reference f/#-constrained solve: %.2e rel (bar %.0e) PASS', ...
         dR(3), P.R3_tol_rel)
 sprintf('   GATE 3 pupil: r_max %.6f x semi, chord %.6f x Aperture, %d rays outside -- PASS', ...
         g0.r_ratio, g0.chord_ratio, g0.n_outside)
 sprintf('          (re-checked on the SAVED deck: %.6f x semi, %d outside)', ...
         gv.r_ratio, gv.n_outside)
 sprintf('   detector: builder placed it at z %+.6f m; the joint solve moved it %.4f mm', ...
         z_build, fp_move_mm)
 sprintf(['   FP_return is frozen at the pre-solve station, so it trails the ' ...
          'detector by that %.1f um'], fp_move_mm*1e3)
 sprintf(['            -- %.1e of the %.3f m exit-pupil radius, and the reason ' ...
          'the deck-level'], fp_move_mm*1e-3/ep_radius, ep_radius)
 '            gate is a bar and not an equality (tE2E2Axial)'
 sprintf('   conic cross-check vs the reference design: |dK| = [%.2e %.2e %.2e] (reported, not gated)', ...
         dKr)
 sprintf('   first order: paraxial EFL %.3f m (f/%.3f) | TRACED EFL %.3f m (f/%.3f)', ...
         lay.EFL, lay.fnum, rpt.EFL_m, rpt.fno_fp)
 sprintf('   solve: JOINT, %d explicit FoV + implicit on-axis, conics + FPA(tip,focus) in ONE DOF set', ...
         size(Fsolve,1))
 sprintf('   conics reached K = [%.9f %.9f %.9f]', K)
 sprintf('   scoring: UNIFORM %dx%d grid over the +-%g deg box; the %d-point solve set is', ...
         P.score_n, P.score_n, h, size(Fsolve,1))
 '            reported alongside ONLY to show what the optimizer saw'
 sprintf('   what stage 2 receives: an axial anastigmat whose residual over the box is')
 sprintf('            %.3f nm max at the centroid reference -- the number the field bias will spoil', ...
         sc.uniform.max_m(2)*1e9)
 '======================================================='};
addtxt = sprintf('%s\n', add{:});
fprintf('%s', addtxt);
fid = fopen(fullfile(exdir,'s1_report.txt'), 'w');
fprintf(fid, '%s%s%s%s', rpt.text, sc.text, pt.text, addtxt);  fclose(fid);
fprintf('    report: s1_report.txt\n');

save(matfile, 'P','R','tsp','lay','K','res','r4a','sc','pt','rpt','pm', ...
     'Fsolve','g0','dK','dR','dKr','z_build','fp_move_mm','ep_radius', ...
     'r_hole','hinfo','r_m2','-append');
fprintf('\nStage 1 complete.  Next: s2_fold.m (fold the back end behind M1).\n');

% ---- helpers --------------------------------------------------------
function t = build_axial_(R, tsp, P)
%BUILD_AXIAL_  The coaxial on-axis Korsch TMA, K=0 seed, hole declared.
%   The secondary is CONVEX by geometry (it sits before the M1 focus), so
%   the closed-form Seidel conic seed is not valid here and the builder
%   returns K=0; the conics are found by the CALIB solve in [4].  This is
%   the documented convex-reimager path, not a shortcut.
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_m', P.D_m, 'wavelength_m', P.lambda_m, ...
            'model_size', P.model_size, 'grid_npts', P.grid_npts);
    t.add_mirror('M1','radius_m',R(1),'spacing_after_m',tsp(1));
    t.add_mirror('M2','radius_m',R(2),'spacing_after_m',tsp(2),'convex',true);
    t.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
    t.add_focal_plane('FP','ap_r',P.fp_body_r);
    t.build();                          % [3] measures and declares the hole
end

function pm = powered_(t)
%POWERED_  Element indices of the powered Reflectors (skips flat folds).
    e = t.spec.elt;
    pm = find(arrayfun(@(x) strcmp(x.kind,'Reflector') && abs(x.Kr) < 1e21, e));
end
