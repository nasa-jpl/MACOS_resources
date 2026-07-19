function P = e2e_params()
%E2E_PARAMS  Every user knob of the end-to-end example, in one place.
%   The e2e stages (s1_telescope ... s6_simulator) all read THIS file --
%   hack it for your system, then re-run the stages in order.  Each
%   stage consumes the previous stage's saved artifacts, so a knob
%   change re-runs from the first stage it affects.
%
%   Stage 1 derives the telescope from the BASIC parameters below via
%   the first-order solver (macos.design.tma_layout): aperture, both
%   f/#s, the Cassegrain-feed magnification and the packaging fractions
%   are free inputs.  The telescope is an ON-AXIS Korsch TMA taken
%   SLIGHTLY OFF-AXIS: the science field is biased just far enough off
%   the axis to carry the focal plane out of the beam (the runner
%   sweeps the bias and picks the least that clears -- off-axis
%   aberration grows ~quadratically with bias, so least is best).

% ================= telescope basics (stage 1) =======================
P.D_m          = 4.0;       % aperture diameter (m); packaging scales with D
P.lambda_m     = 0.5e-6;    % design wavelength (m) -- visible telescope
P.fov_arcmin   = 1.0;       % HALF-field (about the bias) the telescope
                            % holds; stage 2's imaging instrument pushes
                            % the corrected field toward P.inst.fov_arcmin
P.primary_fnum = 1.75;      % M1 f/#  (f1 = fno1*D, R1 = 2*f1)
P.system_fnum  = 18;        % final f/# at the FP  (EFL = fno_sys*D)
P.secondary_mag  = 8;       % Cassegrain feed magnification m2 (>1).
                            % Lower mag relaxes M2's curvature (a
                            % field-aberration source) at the cost of
                            % a somewhat longer bench; the f/1.25+m2=16
                            % starting point could not reach VIS blur
                            % (Dave 2026-07-17: f/1.75, m2=8)
P.int_focus_frac = -0.125;  % intermediate-focus z, fraction of D
                            % (NEGATIVE = in front of M1, between M1
                            % and M2 -- the field-stop / metrology-
                            % injection plane stays accessible)
P.m3_behind_frac = 0.6;     % M3 vertex z, fraction of D behind M1
P.fold_frac    = 0.075;     % 90-deg fold (FM) station z, fraction of D
                            % behind M1: the M2->M3 feed is folded into
                            % +x there, so M3 and everything after live
                            % on a flat bench BEHIND M1 (Dave: move the
                            % back end behind the primary)
P.m3_tilt_deg  = 1.2;       % EXTRACTION TILT on M3, about the bench
                            % normal (the builder's tilt axis maps
                            % there through the fold isometry): the
                            % M3->FP return leaves the feed axis
                            % geometrically, so the fold clears WITHOUT
                            % field bias -- the bias sweep then settles
                            % near its minimum.  (Bias was the VIS
                            % killer: aberration ~ bias^2.)
P.fold_margin  = 1.15;      % fold body radius = margin * local feed
                            % radius.  A slim mount: the M3->FP return
                            % passes the fold body with mm-scale
                            % daylight at the chosen tilt/bias -- 1.4x
                            % grazed the clearance judge
P.fp_body_r    = 0.075;     % focal-plane BODY radius (m) for the
                            % clearance judge (detector + mount)
P.bias_sweep_arcmin = [1 2 3 4 6];  % off-axis biases to explore; the
                            % least whose folded design fully CLEARS
                            % (only M2's central obscuration allowed)
                            % wins -- aberration grows ~bias^2
P.hole_margin  = 1.3;       % M1 central hole radius = margin * the
                            % measured beam radius at the M1 plane

% ---- freeform refinement on top of the conic solve (the "+FF") -----
P.modes     = [3 4 5 9 10 11 12 13 19 20 21 22 23 24 25];
P.ztype     = 'BornWolf';   % Zernike ordering for the departure basis
P.max_iters = 150;          % CALIB iterations per solve
P.max_iters_ff = 300;       % ... for the joint freeform field solve
P.dl_waves  = 0.071;        % Marechal diffraction-limit bar (waves RMS)

% ---- engine sampling ----------------------------------------------
P.model_size = 256;         % stage 1-2 engine model
P.grid_npts  = 41;          % circular ray-grid points (~1300 rays)

% ================= stage 2: imaging instrument ======================
% A THREE-mirror bench relay behind the telescope focus (M4 weak
% corrector at an intermediate conjugate / M5 collimator / M6 camera,
% unit magnification -> final f/# preserved).  Field correction =
% freeform on surfaces at staggered intermediate conjugates; no active
% control in this part of the telescope (Dave 2026-07-17).  A 4-mirror
% variant (extra weak corrector near the relayed pupil) was probed and
% conditioned WORSE -- near-pupil weak correctors act common-mode and
% collapse to near-zero field-solve rank.
% The solve is JOINT: M1-M3 keep refining with the instrument.
P.inst = struct( ...
    'type', "offner", ...       % "offner" (concentric ring-field 1:1
    ...                         % relay -- no tilted powered surfaces,
    ...                         % zero Seidel over the ring; the biased
    ...                         % patch is a ring arc, and several
    ...                         % small-field instruments can pick off
    ...                         % arcs of the same ring) | "zigzag"
    ...                         % (the tilted-sphere bench relay)
    'offner_R',   2.0, ...      % Offner concave radius (m); convex = R/2
    'offner_h',   0.25, ...     % ring radius (m): object/image offset
    'fov_arcmin', 2.0, ...      % widened half-field target
    'field_center', "auto", ... % "auto" (Dave 2026-07-18): after the
    ...                         % pass-1 solve, point the nominal chief
    ...                         % at the y-CENTROID of the region where
    ...                         % raw WFE < field_center_thresh on the
    ...                         % +-map_fov map, and re-solve once (the
    ...                         % [4f] scan then verifies optimality).
    ...                         % "manual" keeps field_dy_arcmin as-is.
    ...                         % set_field_bias is a +y scalar, so
    ...                         % only the y-centroid is adoptable; the
    ...                         % x-centroid is reported.
    'field_center_thresh', 0.02, ...  % waves, the "good region" bar
    'map_fov_arcmin', 3, ...    % WFE-map half-width (Dave: +-3')
    'map_n', 13, ...            % map sampling (13x13 = 0.5' pitch)
    'field_dy_arcmin', -0.7, ...% STARTING science-field-center shift
    ...                         % (+y bias
    ...                         % units) applied to the s1 bias for the
    ...                         % instrument stage: the s2 WFE map's
    ...                         % sweet spot sat below (0,0) (Dave
    ...                         % 2026-07-18: center at (0,-0.7)' or
    ...                         % lower).  BOTH candidates were solved
    ...                         % end-to-end and kept (s2_variants/):
    ...                         %  -0.70' -> best FIELD EDGE: worst +-2'
    ...                         %    0.0231 -tilt, 2' ring Strehl 0.965
    ...                         %  -1.05' -> FLATTER interior (Strehl
    ...                         %    >=0.985 through 1.5') but the 2'
    ...                         %    edge pays (0.038 -tilt, 0.945)
    ...                         % Adopted -0.7 on the worst-field
    ...                         % criterion; switch here if the
    ...                         % instruments live inside the ring.
    ...                         % The runner prints a center-scan table
    ...                         % after the solve; clearance is
    ...                         % re-checked at the shifted bias.
    'dpast_m',    0.45, ...     % M4 corrector past the telescope focus
    'R_m',        [], ...      % [] = DERIVE: M4 weak (20 m); M5 from
    ...                         % the collimator condition f5 = dpast +
    ...                         % leg1 (exact, whatever the telescope
    ...                         % conjugates); M6 = M5 (unit mag)
    'legs_m',     [1.5 2.0], ...       % M4->M5, M5->M6 (M6->FP derives)
    'tilt_deg',   [5 -5 5], ...        % bench zigzag tilts (astig ~
    ...                         % tilt^2; 6 deg OVERLAPS the in/out
    ...                         % beams at these legs and diverges)
    'modes',      [], ...       % relay Zernike modes ([] -> P.modes)
    'nfield_svd', 5, ...        % NxN dense field grid for the SVD solve
    ...                         % stages + final scoring (CALIB caps at 12)
    'max_iters_ff', 150);       % joint freeform field-solve iterations

% ================= stage 3: segmentation ============================
% TWO variants, both kept as artifacts (Dave 2026-07-18): "pie" =
% 1-ring PIE (7 segments: center HEXAGON + 6 chorded wedges) and
% "hex2" = 2-ring HEX (19 hexagonal segments).  Segment size is NOT a
% knob: SegMirMaker defaults it to Aperture/(2*rings+1), so the tiling
% scales with P.D_m automatically (4 m here = half the e5 fixtures;
% the gap halves with it, 50 -> 25 mm).
P.seg = struct( ...
    'variants', ["pie" "hex2"], ... % s3 builds ALL of these ->
    ...                         % e2e_pie.in / e2e_hex2.in
    'variant',  "pie", ...      % the one stages 4-6 consume (one-knob
    ...                         % switch; both artifacts always exist)
    'gap_mm',         25, ...   % inter-segment gap (BaseUnits are m in
    ...                         % this family -- the runner converts)
    'dofs',           6, ...    % per-segment DOFs
    'meas_config',    2, ...    % edge sensors: 1 inner edges, 2 all
    'emit_apertures', true, ... % declare physical PolyApVec boundaries
    'ap_pad',         0, ...    % 0 = physical edge (gap rays clip)
    'grid_npts',      128, ...  % segmented-source sampling (e5-corpus
    ...                         % density; the parent's 41 gives only
    ...                         % ~50 rays/segment at 2 rings -- Dave
    ...                         % 2026-07-18: sampling seemed coarse)
    'model_size',     512);     % segmented stages need the big model

% ================= stage 5: MET =====================================
P.met = struct( ...
    'nf',               6, ...  % fiducials on the hub (>=3, likely 6)
    'fid_rim_inset_mm', 25, ... % fiducial ring inset inside the hub rim
    'edge_off_mm',      5, ...  % launcher clearance outward of seg edge
    'min_sep_mm',       50, ... % min separation between ANY 2 launchers
    'sig_rot',   1e-6, ...      % prior (deploy) sigma, rad per rot DOF
    'sig_trans', 1e-6, ...      % ... metres per trans DOF
    'sig_edge',  1e-9, ...      % edge-sensor noise, metres
    'sig_met',   1e-9);         % MET gauge noise, metres

% ================= stage 6: simulator ===============================
P.sim = struct( ...
    'engine',  "linear", ...    % "linear" | "mmacos" PSF engine switch
    'nsteps',  200, ...         % time-history length
    'dt_s',    1.0, ...         % time step
    'walk_rot',   2e-8, ...     % random-walk step, rad (rigid rot)
    'walk_trans', 2e-8, ...     % random-walk step, m (rigid trans)
    'grid_nm_rms', 5);          % per-segment figure drift, nm RMS
end
