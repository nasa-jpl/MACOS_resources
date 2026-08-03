function P = afocal4_params()
%AFOCAL4_PARAMS  The one parameter struct for the afocal4 study (PLAN S0).
%
%   Everything the 30x afocal 4-mirror work is specified by, in ONE place
%   and in SI: no script below this file invents a number.  The benchmark it
%   answers is J.M. Rodgers' coaxial 3-mirror afocal (design/rodgers2/), so
%   the aperture, magnification, wavelength, field box and stop position are
%   HIS -- a form study that changed them would not be answering him.
%
%   THE PARENT is his S1 on-axis variant, and the choice is not cosmetic.
%   S1 is the only member of his ladder whose FIRST ORDER closes: it is
%   paraxially afocal to 3e-6 rad, M = 30.000 to five figures, and its exit
%   pupil lands 343.4 mm past M3 against the 344.2 mm coldstop he placed by
%   hand.  His S3/S4 re-solves moved R3 by 3.8% to recollimate the REAL
%   marginal ray at f/1.25, which leaves the paraxial marginal ray 41 urad
%   convergent and the paraxial exit beam 4.8% small (31.80 vs 33.33 mm).
%   Both are right -- his is a real-ray solution, this is a paraxial study --
%   but seeding a first-order form study from S3 would build a 31.4x
%   telescope.  The conics carried below are S1's.
%
%   THE TARGETS are the S3 gate-review numbers (PLAN_AFOCAL4 S3), set at
%   >= 10x his BEST 3-mirror variant.  The breathing target is CHIEF-NORMAL
%   -- a magnification read on the placed coldstop carries that plane's own
%   1/cos obliquity (rodgers2 PACKET section 4 refinement) and is not a
%   pupil-imaging defect.
%
%   TWO DIFFERENT THINGS ARE CALLED "STANDOFF" and conflating them wastes an
%   afternoon, so they are named apart here and everywhere below:
%     P.iface        the INTERFACE standoff -- how far past the last mirror
%                    the exit pupil (the instrument's cold stop) sits.  This
%                    is the S4 ruling's PARAMETER: it selects the operating
%                    point, it is not a spec, and the targets-versus-iface
%                    curve is a reported output (afocal4_ladder 'trade').
%                    140 mm is the flagged default -- the breathing-meeting
%                    point of the S3 phi4 scan.
%     P.fm_standoff  the FIELD MIRROR's standoff -- how far BEFORE the
%                    intermediate image the fourth mirror sits.  A solve DOF;
%                    it buys beam footprint on a mirror that would otherwise
%                    have none, which is what makes its conic worth having.
%                    NEGATIVE puts the mirror PAST the image, which is just
%                    as physical -- the marginal ray has crossed the axis and
%                    the closure handles the sign -- so the bound is
%                    two-sided and the solve picks the side.

    mm = 1e-3;
    P = struct();
    P.name        = 'afocal4';
    P.D           = 1.0;            % entrance pupil diameter, m
    P.M           = 30.0;           % angular magnification (exit beam D/M)
    P.lambda      = 1.0e-6;         % m
    P.stop_ahead  = 50*mm;          % stop AHEAD of M1 (his STO)
    P.model_size  = 256;
    P.ngrid       = 41;             % circular grid -> 1185 launched

    % --- field: 0.5 x 0.5 deg box biased +0.6 deg in Y (his) -------------
    P.fov_half_deg = 0.25;
    P.bias_deg     = 0.6;
    [gx, gy]  = meshgrid([-1 0 1]*P.fov_half_deg, [-1 0 1]*P.fov_half_deg);
    P.Fsolve_deg = [gx(:), gy(:)];              % his 3x3 SOLVE set
    P.Fsolve     = deg2rad(P.Fsolve_deg);
    P.grid_n     = 9;                            % uniform SCORING grid

    % --- the parent: rodgers2 S1 on-axis, metres -------------------------
    % Vertex stations in his global frame (M1 at 0): M2 at -1.049239294,
    % M3 at +0.640415896; the beam folds -z, +z, -z, so the inter-mirror
    % spacings are the |dz| between vertices.
    zM2 = -1.049239293684764;
    zM3 = +0.640415896;
    P.parent = struct( ...
        'name',    {{'M1','M2','M3'}}, ...
        'R',       [2.5, 468.7799802942544*mm, 580.8105879437068*mm], ...
        'convex',  [false, true, false], ...
        'K',       [-1.0, -1.782495505768868, -1.001753914266608], ...
        't',       [-zM2, zM3 - zM2], ...
        'z',       [0, zM2, zM3], ...
        'coldstop_dist', 344.173*mm);      % his recenter t, S1

    % --- interface spec --------------------------------------------------
    % The distance from the LAST powered mirror to the interface pupil.
    % His coldstop is 344.17 mm past M3 and the traced first-order exit
    % pupil lands at 343.36 mm -- 0.8 mm apart on a 33 mm beam.  So the
    % 3-mirror ALREADY closes the first-order pupil condition, and that is
    % the single most important input to the form study (FORM_STUDY.md
    % section 1).  The spec is held at the PUPIL, not at his coldstop, so
    % every candidate is compared at its own closed condition.
    P.iface_dist = 343.363*mm;

    % --- S3 targets (gate review 2026-08-03) -----------------------------
    P.targets = struct( ...
        'wfe_rung2_nm',   71.0,  ...   % DL at 1 um, in-box max
        'blur_um',        47.0,  ...   % from 469 (his S4)
        'wander_um',      56.0,  ...   % at the REFIT plane, from 557
        'breathe_pct',    0.4,   ...   % CHIEF-NORMAL half-range, from 3.63
        'surface_pv_mm',  0.2,   ...   % net of the imaged primary sag
        'mag',            30.0,  ...
        'mag_pct',        0.1);        % tolerance on M at box centre

    % =====================================================================
    %  S4 -- the joint solve
    % =====================================================================

    % --- the operating point (the S4 ruling: a PARAMETER, not a spec) ----
    % The interface standoff rides the field mirror's power: 343 mm at
    % phi4 = 0 (the three-mirror's own pupil), 140 mm at phi4 = +2 /m where
    % the S3 breathing null sits, 27 mm at phi4 = +4.  Held at the flagged
    % default through the answer ladder; swept by afocal4_ladder('trade').
    P.iface        = 140*mm;
    P.fm_standoff  = 0.20;             % field-mirror standoff SEED (a DOF)
    P.iface_trade  = [50 90 140 220 343]*mm;   % the reported trade curve

    % --- the form under solve --------------------------------------------
    % 'field' is the S3 ruling.  'mersenne' is the ONE bounded hedge
    % experiment (task 3): confocal SPACINGS held -- that is where the pupil
    % relay lives -- and all four conics relaxed.
    P.form = 'field';
    P.mersenne = struct('m1',6.0, 'stage2_fnum',2.0, ...
                        'stage2_type','gregorian', 'gap',0.5);

    % --- solve sampling: SOLVE SET ~= SCORING SET, in field AND in rays ---
    % The doctrine rule is about the FIELD set (his 3x3 to solve, a uniform
    % 9x9 to score).  The same separation pays in ray count and pupil-node
    % count, and it is worth stating rather than leaving as tuning: the
    % merit only has to RANK designs, the score has to be quotable.  Measured
    % on the S3 field_p2 layout, nodes 11 vs 21 move the blur by 1% and the
    % wander by 0.2%, for 10% of the time.
    P.solve = struct( ...
        'ngrid',       21,   ...  % ray grid during the solve (score at P.ngrid)
        'nodes',       11,   ...  % pupil_map lattice during the solve
        'nodes_score', 21,   ...  % ... and when scoring
        'max_iter',    40,   ...
        'tol_fun',     1e-4, ...
        'tol_x',       1e-6, ...
        'fd_step',     1e-3);     % forward-difference step, in SCALED DOFs

    % --- merit weights (knobs, per the S4 brief) --------------------------
    % Every term is scored in the LOG of its ratio to target, not the ratio
    % itself; see AFOCAL4_SCORE for why (an unoptimised four-mirror layout
    % misses the WFE target by 200x and the pupil targets by 6x, and a
    % linear normalisation lets the WFE term own the solve outright).
    P.weights = struct( ...
        'wfe',      1.0,  ...   % applied per solve field
        'blur',     1.0,  ...
        'breathe',  1.0,  ...
        'wander',   1.0,  ...
        'surf_pv',  0.3,  ...   % already 14x inside target on the parent
        'mag',      1.0);

    % A term better than FLOOR x its target stops earning credit: the solve
    % should spend its remaining freedom on whatever is still missing, not
    % drive one satisfied metric to zero.
    P.merit_floor = 0.5;

    % --- rung-4 rigid-body DOFs -------------------------------------------
    % M2 / FM / M3, y-decenter and x-tilt: the plane of the field bias, which
    % is the only plane a coaxial design biased in +y has any asymmetry in.
    % M1 carries no rigid body -- it is the datum (and his study holds it
    % too).  Scales below make every DOF O(1) to the solver.
    P.rb_elts   = [2 3 4];             % build order: M2, FM, M3
    P.dof_scale = struct('conic',1.0, 'standoff',0.05, 'radius',0.05, ...
                         'spacing',0.05, 'dec',1e-3, 'tilt',1e-3);
    P.bounds    = struct('fm_standoff',[-0.50 0.60], 'conic',[-30 30], ...
                         'R2',[0.25 1.20], 't1',[0.70 1.40], ...
                         'dec',[-0.05 0.05], 'tilt',[-0.05 0.05]);
end
