function P = e2e2_params()
%E2E2_PARAMS  Every user knob of the improved TMA design flow, in one place.
%
%   The e2e2 stages (s1_axial ... s5_score) read THIS file and nothing
%   else.  Hack it for your system, then re-run the stages in order; each
%   stage consumes the previous stage's saved artifacts, so a knob change
%   re-runs from the first stage it affects.
%
%   NO COMPUTATION HAPPENS HERE.  Every derived quantity -- radii, conics,
%   bias, hole size, detector pose, relay geometry -- is solved or measured
%   inside a stage runner and reported with its provenance.  A number that
%   appears in a report and not in this file was DERIVED; a number that
%   appears here was CHOSEN.  That split is the point of the flow.
%
% =====================================================================
%  THE DESIGN POINT (Dave, 2026-08-01)
% =====================================================================
%  The Rodgers offset-field coaxial TMA (design/rodgers1/), SCALED to a
%  D = 3 m aperture: every length x 3/5 from his 5 m .seq geometry.  f/#
%  is scale-invariant, so the scaled system is still f/20 (EFL 60 m) off
%  an f/1.2358 primary, and his `CUY UMY -0.025` marginal-angle solve --
%  the constraint that pins M3's radius -- carries over unchanged.
%
%  Scaling a validated design instead of inventing one buys a FIRST-ORDER
%  GATE that no de-novo layout has: the builder's own R3 (derived from the
%  f/20 constraint) must reproduce his CODE V solve's R3 x 3/5.  s1 checks
%  exactly that before it believes any WFE number.
%
%  What is NOT scaled: the wavelength.  His study ran at 1 um; this one
%  runs at 500 nm and asks for DIFFRACTION-LIMITED performance there --
%  RMS <= lambda/14 ~ 36 nm and Strehl >= 0.8 across the used box.
%
%  NOR IS THE FIELD.  His box was 0.2 deg; this one is 0.6 deg, THREE
%  TIMES WIDER, and that is a measurement rather than an aspiration.
%  s1_fov_sweep.m re-solved the axial telescope at half-fields from 0.10
%  to 0.30 deg and scored each on its own uniform box: the residual grows
%  as theta^2.96 and the full 0.6 deg box lands at 27.7 nm at the primary
%  (centroid) reference, inside the 35.7 nm bar with 22%% margin.  Run the
%  sweep again after any change to the design point -- it is cheap and it
%  is the only honest way to set this number.

% ================= aperture, wavelength, field =======================
P.D_m        = 3.0;         % aperture diameter (m) = 5 m x 3/5
P.lambda_m   = 500e-9;      % design wavelength (m) -- visible
P.fov_half_deg = 0.3;       % HALF-field: the 0.6 deg x 0.6 deg used box.
                            % Measured, not assumed -- see s1_fov_sweep.m
                            % and the header note above.  0.25 deg is the
                            % conservative alternative: it passes every
                            % rung including the strictest (chief) with
                            % room, where 0.30 deg is 4%% OVER the bar at
                            % chief while passing at the centroid primary.
P.fov_arcmin   = 18.0;      % the same half-field in arcmin (0.3 deg)

% ---- performance target (reported, never silently relaxed) ----------
P.dl_waves   = 1/14;        % Marechal-class RMS bar, waves (~0.0714)
P.dl_rms_m   = 500e-9/14;   % ... the same in metres (35.7 nm)
P.strehl_min = 0.80;        % Strehl bar at the design wavelength

% ================= first-order layout (S1) ===========================
% The builder (macos.design.tma_layout) takes the f/#s and the feed
% magnification as FREE inputs and derives the radii.  These four numbers
% ARE the Rodgers geometry, read back from his .seq prescription:
%   primary_fnum   = |R1|/(2 D)         = 12357.51782 / (2 x 5000)
%   secondary_mag  = 1 + 2 d_int/|R2|   (Cassegrain feed magnification)
%   int_focus_m    = -t1 + d_int        (intermediate focus station)
%   m3_behind_m    = -t1 + t2           (M3 vertex station)
% all evaluated on his unfolded paraxial trace, then scaled x 3/5.
P.primary_fnum  = 1.235752;   % M1 f/#  (f1 = fno1*D, R1 = 2*f1)
P.system_fnum   = 20;         % system f/# at the FP (EFL = fno_sys*D = 60 m)
P.secondary_mag = 5.798319;   % Cassegrain feed magnification m2 (>1)
P.int_focus_m   = +0.008116;  % intermediate-focus z (m).  POSITIVE and
                              % tiny: this feed focuses ~8 mm BEHIND the
                              % M1 vertex -- a near-telecentric Cassegrain,
                              % which is why the M1 hole is large.
P.m3_behind_m   = 1.103412;   % M3 vertex z, behind the primary (m)
P.M1_hole_m     = 0.308365;   % M1 central perforation SEMI-diameter (m)
                              % = his CIR HOL 513.9423 mm x 3/5.  0.2056
                              % linear obscuration, 4.23% of the area.

% ---- the first-order GATE (S1 stops here if these miss) -------------
% His radii, scaled.  R1 and R2 are INPUTS above (via the f/# and the
% feed magnification); R3 is DERIVED by the builder from the f/20
% constraint and must land on his CODE V `CUY UMY` solve.
P.R_ref_m       = [7.414511, 1.320820, 1.612784];  % [R1 R2 R3] x 3/5
P.t_ref_m       = [3.160742, 4.264155];            % [M1->M2 M2->M3] x 3/5
P.R3_tol_rel    = 2e-3;      % |R3_derived - R3_ref| / R3_ref bar.  The
                             % paraxial reproduction of his f/20 solve is
                             % 1.1e-5 on the marginal angle; 2e-3 leaves
                             % room for the builder's own root selection
                             % without letting a wrong branch through.
% The reference design's CONIC constants.  Conics are DIMENSIONLESS, so
% unlike the radii they do not scale -- a correct on-axis anastigmat solve
% at this design point must land on them without ever being given them.
% s1 reports the comparison; it is a cross-check on the whole chain
% (layout -> builder -> CALIB), not a gate, because the reference conics
% were solved at 1 um over a 15-point half box and this solve runs at
% 500 nm over a uniform box.
P.K_ref         = [-0.9929244714356076, -1.926467376849899, ...
                   -0.7072161814228599];
P.fp_blur_max_m = 1.0e-3;    % detector-fit gate, used from S2 on (S1 does
                             % not fit a detector -- see its [4]).  The
                             % best-focus blur align_focal_plane reports
                             % must be small: it is a RAY fit, so on an
                             % unsolved (K = 0, spherical) design it locks
                             % onto the spherical caustic instead -- 1.1e-2
                             % m of blur, detector walked 1.796 m off the
                             % correct station.  1 mm is ~80x the Airy
                             % radius here and ~11x below that failure.
P.fixture_tol   = 1e-6;      % conic mismatch bar on the shared TMA
                             % fixture (optical_design/fixtures/
                             % tma_fixture.json).  STOP-AND-FIX, never
                             % widen: a miss here means the conic solver
                             % moved, not that the tolerance was tight.

% ================= off-axis stage (S2) ===============================
% The field is biased off the axis so the focal plane clears the beam.
% THE BIAS IS A PROGRAM REQUIREMENT, not an optimization output (Dave
% 2026-08-01): what forces it is CLEARANCE, and clearance is tested in
% S2/S3, so scoring candidates on WFE alone is degenerate -- the smallest
% bias always wins.  The starting requirement grows in PROPORTION to the
% field, at the ratio the reference design used:
%
%     offset / half-field = 0.5 deg / 0.1 deg = 5      (P.offset_ratio)
%
% so the 0.3 deg half-field above asks for a 1.5 deg offset.  S2 solves
% there and S3's fold + M3 extraction tilt then sharpen it: the e2e VIS
% lesson is that GEOMETRY buys clearance and bias only buys it at a
% cost of ~bias^2, so expect the sharpened value to come back DOWN.
P.offset_ratio = 5.0;       % offset / half-field; S2 derives the offset
                            % from it and reports the number
P.bias_sweep_arcmin = [30 60 90 120 150];  % 0.5 .. 2.5 deg, bracketing
                            % the 90' (1.5 deg) proportional requirement
P.field_center = "auto";     % "auto": after the pass-1 solve, map the WFE
                             % over a patch wider than the science box,
                             % take the centroid of the good region, and
                             % RE-SOLVE with the chief there if it moved
                             % (the e2e s2 [4g] loop -- re-scoring a
                             % shifted center undersells re-solving there).
                             % "manual" keeps field_dy_arcmin as given.
P.field_center_thresh = 1/14;% waves, the "good region" bar for that scan
P.field_dy_arcmin  = 0;      % starting science-center shift off the bias
P.map_fov_arcmin   = 27;     % WFE-map half-width for the center scan
                             % (1.5x the half-field)
P.map_n            = 13;     % map sampling (13x13)
P.center_move_min_arcmin = 0.15;  % re-solve only if the chief is farther
                             % than this off the measured centroid

% ================= fold (S3) =========================================
P.fold_frac      = 0.075;    % 90-deg fold station z, fraction of D behind
                             % M1: the M2->M3 feed turns into +x there, so
                             % M3 + image + FP sit on a bench BEHIND M1
P.m3_tilt_deg    = 1.2;      % EXTRACTION TILT on M3 about the bench
                             % normal: the M3->FP return leaves the feed
                             % axis geometrically, so the fold clears
                             % WITHOUT paying bias^2 for it
P.fold_margin    = 1.15;     % fold body radius = margin x local feed radius
P.fp_body_r      = 0.056;    % focal-plane body radius (m) for the
                             % clearance judge = 0.075 x 3/5
P.hole_margin    = 1.3;      % hole radius = margin x measured through-beam
                             % radius at the M1 plane; the DECLARED hole is
                             % max(this, P.M1_hole_m)
P.aoi_max_deg    = 15;       % standing rule: angle of incidence bar at
                             % every powered surface

% ================= relay + focal plane (S4) ==========================
P.relay = struct( ...
    'type',       "offner", ...  % "offner" (concentric 1:1 ring-field
    ...                          % relay, pure spheres -- no tilted
    ...                          % powered surfaces, so the tilt-astig
    ...                          % floor is deleted at the root) |
    ...                          % "bench" (the 3-mirror zigzag relay) |
    ...                          % "none" (score the telescope alone)
    'offner_R',   1.2, ...       % concave radius (m); convex = R/2
    'offner_h',   0.15, ...      % ring radius (m): object/image offset
    'dpast_m',    0.27, ...      % first relay mirror past the telescope
    ...                          % focus (m) = 0.45 x 3/5
    'legs_m',     [0.9 1.2], ... % bench-relay legs (m) = e2e x 3/5
    'tilt_deg',   [5 -5 5], ...  % bench-relay zigzag tilts (deg)
    'nfield_svd', 5);            % NxN dense field grid for the SVD solve
                                 % stages (CALIB caps at 12 FoV; the SVD
                                 % engine does not)

% ================= solve control =====================================
P.modes        = [3 4 5 9 10 11 12 13 19 20 21 22 23 24 25];
P.ztype        = 'BornWolf';  % ONE Zernike type per mirror for the life
                              % of its coefficients (solve doctrine)
P.max_iters    = 150;         % CALIB iterations per conic/rigid solve
P.max_iters_ff = 300;         % ... for the joint freeform field solve
% FPA DOF mask for the JOINT solve, in the engine's
% [TIP TILT CLOCK DX DY PIST ROC CONIC] order: TIP (rotation about the
% detector's local x = the alpha tilt) and PIST (translation along its
% normal = focus).  NOT CLOCK (a null direction on a detector) and not DX
% (a lateral shift the reference tie absorbs).  Verified against
% macos_ops.F:CPERTURB_2, where PV(1:3) is rotation and PV(4:6)
% translation in the element frame.
P.fpa_dofs     = [1 0 0 0 0 1 0 0];
% Per-element DOF masks, same order.  Conics only for the anastigmat
% solve; conics + tilt + decenter for the rigid-body stage.
P.dofs_conic   = [0 0 0 0 0 0 0 1];
P.dofs_rigid   = [1 0 0 0 1 0 0 1];

% ================= scoring ===========================================
% SOLVE SET != SCORING SET.  The solve runs on the program's field points
% (CALIB caps the TOTAL at 12 FoV with the on-axis field IMPLICIT, so at
% most 11 explicit); the STATISTICS are computed on a uniform grid.  An
% edge-weighted solve sampling biases the average ~8% at an identical max
% (rodgers1 dense_field_check).
P.solve_n      = 3;          % NxN solve grid over the box (3x3 = 8
                             % explicit + 1 implicit = 9 FoV)
P.score_n      = 9;          % NxN UNIFORM scoring grid (81 points)
P.score_rung   = 4;          % which strict_rungs column leads the
                             % headline: 1 chief, 2 centroid (the
                             % primary reference), 3 +bestfoc,
                             % 4 +LS tip/tilt.  ALL FOUR are always
                             % computed and tabled; this only picks the
                             % one the pass/fail gate reads.  4 is the
                             % rung external field-map RMS numbers
                             % (CODE V's included) are consistent with.
P.report_rungs = 1:4;        % rungs to table

% ================= engine sampling ===================================
P.model_size = 256;          % engine model size (>= nGridpts)
P.grid_npts  = 41;           % circular ray-grid points (~1300 rays).
                             % 41 is the SOLVE density; the final maps
                             % re-run at score_grid_npts.
P.score_grid_npts = 81;      % denser grid for the final scoring pass
P.pupil_tol_rel   = 1e-3;    % pupil gate: |greatest chord of spot(1) /
                             % declared Aperture - 1| bar, and zero rays
                             % outside the declared radius.  The engine
                             % is correct since macos PR #70; the gate
                             % stays because the defect hid for decades
                             % by reading exactly right along the axes.
end
