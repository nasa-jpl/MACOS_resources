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
P.primary_fnum = 1.25;      % M1 f/#  (f1 = fno1*D, R1 = 2*f1) -- FAST
P.system_fnum  = 18;        % final f/# at the FP  (EFL = fno_sys*D)
P.secondary_mag  = 16;      % Cassegrain feed magnification m2 (>1).
                            % HIGH mag = compact: intermediate cone
                            % fno1*m2 = f/20, so the M3 relay is
                            % near-unit magnification (short bench),
                            % M2 is small (~6% linear obscuration),
                            % and the M1 hole is cm-scale
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
P.fold_margin  = 1.4;       % fold body radius = margin * local feed radius
P.fp_body_r    = 0.075;     % focal-plane BODY radius (m) for the
                            % clearance judge (detector + mount)
P.bias_sweep_arcmin = [3 4 5 6 8];  % off-axis biases to explore; the
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
P.inst = struct( ...
    'fov_arcmin', 2.0);     % widened half-field target for the 3-4
                            % mirror imaging relay (rest of the stage-2
                            % knobs land with s2_instrument)

% ================= stage 3: segmentation ============================
P.seg = struct( ...
    'rings',          1, ...    % 1 -> 7 segments, 2 -> 19
    'grid',           "Hex", ...% segment tiling (Hex|Pie)
    'gap_mm',         25, ...   % inter-segment gap
    'dofs',           6, ...    % per-segment DOFs
    'meas_config',    2, ...    % edge sensors: 1 inner edges, 2 all
    'emit_apertures', true, ... % declare physical PolyApVec boundaries
    'ap_pad',         0, ...    % 0 = physical edge (gap rays clip)
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
