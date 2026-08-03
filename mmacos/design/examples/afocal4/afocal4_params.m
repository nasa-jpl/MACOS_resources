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
        'wander_um',      56.0,  ...   % at the placed plane, from 557
        'breathe_pct',    0.4,   ...   % CHIEF-NORMAL half-range, from 3.63
        'surface_pv_mm',  0.2,   ...   % net of the imaged primary sag
        'mag',            30.0);
end
