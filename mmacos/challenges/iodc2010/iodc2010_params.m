function p = iodc2010_params()
%IODC2010_PARAMS  Defaults for the IODC 2010 "green lens" challenge runner.
%   p = iodc2010_params() returns the documented knob struct; edit a field
%   or pass name/value pairs to iodc2010_run.  Every number a report prints
%   traces to one of these or to the prescription.
%
%   The problem (Juergens, Proc. SPIE 7652, 76522I, Sec. 2.1-2.2):
%     lambda_nm        532      monochromatic, doubled YAG
%     glass            N-BK7    Schott, the only glass
%     efl_mm           100      |EFL| = 100.00 +- 0.01 mm, paraxial
%     efl_tol_mm       0.01
%     rms_limit        0.070    waves RMS, chief-ray reference, tilt & focus kept
%     dist_limit       5.0      percent, |100 (yR - yP)/yP|, at the image plane used
%   Scoring (ours; README "The metric"):
%     nGridpts         97       rays across the paraxial entrance-pupil diameter
%                               (7081 in the disc).  The RMS is a lattice sum, so
%                               it depends on the grid: the f/10 singlet on axis
%                               reads 0.09095 / 0.09330 / 0.09439 / 0.09436 waves
%                               at 33 / 65 / 97 / 127 across (probe 2026-10-30);
%                               97 is converged to 4e-4 relative and costs 0.02 s
%                               per field.  The 2006 evaluator quotes CODE V with
%                               512 rays across the pupil.
%     field_scan_deg   the semi-fields scanned in order; the scan stops at the
%                      first failing field and bisects back to theta_tol_deg
%     theta_tol_deg    0.01     the semi-field limit's resolution
%     semidia_default  the clear semi-diameter written when a lens does not
%                      give its own and the footprint sizer is off (mm)
%     size_apertures   true: size every clear aperture to the beam footprint
%                      at the scan's largest PASSING field (the "no mounting
%                      material beyond the clear aperture" loophole), then
%                      check the edge thickness there
%     stop_clips       false: the stop is the PARAXIAL construct the 2006
%                      evaluator used -- the rays are uniform across the
%                      paraxial entrance pupil and "any internal aperture
%                      stops were fully filled" (Sec. 4.4: the pupil
%                      aberration this puts at the stop was accepted, not
%                      clipped); a ray is lost only to a surface miss, TIR or
%                      a declared clear aperture.  true: the stop's radius is
%                      a hard circular aperture (the first ladder's convention,
%                      2026-10-30 a.m., which failed lenses by a ring of edge
%                      rays a fraction of a percent outside the stop)
%     src_standoff_mm  how far upstream of the lens the source plane sits
%     model_size       128
%     outdir           where reports, decks and figures go (this directory)
%   The outer design loop (p.outer, passed to iodc2010_outer by the runner):
%     families         'forms' (iodc2010_forms) and/or 'groups' (iodc2010_groups)
%     n_elts           element counts the forms family enumerates
%     groups           iodc2010_groups overrides (a_range, c_range, b_max, gap_grid)
%     D_target         entrance pupil the seeds are sized for (mm)
%     design_field_deg the floor of the CALIB design field (the seed's own limit above it)
%     top_k            seeds per element count that go to the solve
%     prescreen_grid   rays across the pupil in the pre-screen
%     air_grid         seed air spaces in units of the radius
%     ladder_D / ladder_theta  the (D, theta) ladder the best lenses are re-solved on
%     n_ladder         how many of the best solved lenses climb the ladder
%     solve            iodc2010_solve overrides (rounds, iters_per_round,
%                      floor_glass_R, floor_air_R, var_stop, ray_grid)
%     n_climb          how many of the best solved lenses get the merit-shaped
%                      climb (iodc2010_climb: the design field bracketed on the
%                      lens's own semi-field, the fields re-weighted toward the
%                      worst one), at the lens's own D and at every climb_D
%     climb_D          pupils (mm) the climb is run at besides the lens's own
%     climb            iodc2010_climb overrides (steps, n_fields, grow, tol,
%                      reweight, w_floor, score_grid)
%     chain            iodc2010_chain overrides (n_units, unit_gap, link_gap,
%                      symmetric, stop_places) when 'chain' is among families
%   The on-axis stage (p.axis; stage 'axis'): retro (iodc2010_retro grids),
%     top_k (per element count), D_targets, ladder, capacity_grid, relax
%     (iodc2010_relax overrides).
%   The chain runs (p.chain, a cell of entries; stage 'chain'): each entry
%   names its output directory under search/ and carries the outer-loop
%   fields that differ from p.outer -- at least 'chain', the pair-unit grids.
p = struct();
p.lambda_nm       = 532;
p.glass           = 'N-BK7';
p.efl_mm          = 100;
p.efl_tol_mm      = 0.01;
p.rms_limit       = 0.070;
p.dist_limit      = 5.0;
p.nGridpts        = 97;
p.field_scan_deg  = [0 0.5 1 2 3 4 5 7.5 10 12.5 15 17.5 20 25 30 35 40 45 50 55 60];
p.theta_tol_deg   = 0.01;
p.semidia_default = 60;
p.size_apertures  = true;
p.aperture_margin = 1e-6;      % relative margin on a footprint-sized aperture
p.stop_clips      = false;
p.src_standoff_mm = 100;
p.model_size      = 128;
p.outdir          = fileparts(mfilename('fullpath'));
% The values are the 2026-10-30 production run's (search/outer/: the air
% spaces 0.2 / 0.7 / 2.5 R, the groups' gaps 0.3-10 R, the ladder 4-20 mm x
% 3-20 deg), so iodc2010_run('stage', 'outer') reproduces that record.  A
% seed's NAME carries its gap INDEX (f3_..._g3_...), so a table written under
% one air grid names different lenses under another; the tables carry gaps_R.
p.outer = struct('families', {{'forms', 'groups'}}, 'n_elts', [2 3], ...
    'groups', struct('a_range', 0:2, 'c_range', 0:2, 'b_max', 4, 'gap_grid', [0.3 1 3 10]), ...
    'D_target', 10, 'design_field_deg', 1, 'top_k', 8, ...
    'prescreen_grid', 21, 'air_grid', [0.2 0.7 2.5], 'ladder_D', [4 7 12 20], ...
    'ladder_theta', [3 6 12 20], 'n_ladder', 3, ...
    'solve', struct('rounds', 4, 'iters_per_round', 12, 'floor_glass_R', 0.03, 'floor_air_R', 0.01, 'var_stop', false, 'ray_grid', 31), ...
    'n_climb', 3, 'climb_D', [4 7 12], ...
    'climb', struct('steps', 8, 'n_fields', 4, 'grow', 1.3, 'tol', 0.03, 'reweight', false, 'w_floor', 0.1, 'score_grid', 21));
% The chain runs (iodc2010_run('stage', 'chain')): four and six elements as
% chains of zero-Petzval pair units (iodc2010_chain), each run the outer loop
% with p.outer's options, 'families' {'chain'} and the entry's fields over
% them, in search/<name>/.  Six units' grids are coarser: the count is the
% grids' product (2 orders x unit gaps x links per unit).
% The on-axis construction stage (iodc2010_run('stage', 'axis'); search/axis/):
% the retrofocus family (iodc2010_retro, the grids below) plus the thick
% equi-convex singlet, each seed's on-axis capacity measured
% (iodc2010_capacity), the top_k by capacity relaxed and snapped
% (iodc2010_relax) at every D in D_targets; relax = iodc2010_relax overrides.
% The baseline the stage must beat (2026-10-31, README "The on-axis
% construction"): 10.4 mm (the thick equi-convex), 5.3-6.4 (retrofocus
% trains), 2.5 (the chain relays).
p.axis = struct('retro', struct('n_front', [1 2], 'gap_R', [3 10], 'n_men', [2 4 8], 'tau_R', [0.3 0.6], 'g_R', [0.05 0.3]), ...
    'top_k', 1, 'D_targets', [20 40 80], 'capacity_grid', 33, 'ladder', true, ...
    'relax', struct('iters_free', 120, 'iters', 25, 'score_grid', 21));
% top_k is PER ELEMENT COUNT (1, 4, 6, 10, 12 elements); ladder = each pupil
% target relaxed from the previous target's snapped lens.  The first run
% (2026-10-31, top_k 4 overall, no ladder, 60 free iterations, the
% nearest-value snap): best 16.92 mm, README "Relax and snap".
p.chain = { ...
    struct('name', 'chain4', 'chain', struct('n_units', 2, 'unit_gap', [0.1 0.3 1 3], 'link_gap', [0.5 1.5 4 10])), ...
    struct('name', 'chain6', 'chain', struct('n_units', 3, 'unit_gap', [0.3 1], 'link_gap', [1.5 4 10]))};
end
