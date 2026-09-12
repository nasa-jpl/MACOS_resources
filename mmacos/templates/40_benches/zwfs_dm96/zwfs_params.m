function P = zwfs_params()
%ZWFS_PARAMS  Default parameters for zwfs_run, the ZWFS DM-gauge runner.
%   P = zwfs_params() returns the documented default struct: every knob the
%   campaign turned, with the value of record.  Edit a copy and pass it,
%       P = zwfs_params;  P.NGRID = 385;  P.mask.DIA_LAMD = 3;  zwfs_run(P)
%   or override on the call line with top-level or dotted names,
%       zwfs_run('NGRID',385, 'mask.DIA_LAMD',3, 'stages',{'battery','figs'})
%
%   UNITS: lengths mm (the bench decks' BaseUnits), wavelengths mm, DM
%   heights mm (20e-6 = 20 nm), phases rad, photons per MEASUREMENT (one DM
%   shape measured once; a reading's frames share the count).
%
%   The defaults reproduce the S7 record (zwfs_s7iter_report.txt) when
%   run with stages {'bench','battery'}: same bench, same seeds, same
%   readings -- the runner's own equivalence gate (zwfs_dm96/README.md,
%   "Run it yourself").

% ---- run control ------------------------------------------------------
P.tag    = 'run';            % names the output dir + files: runs/<tag>/<tag>_report.txt
P.outdir = '';               % '' = fullfile(<this dir>, 'runs', P.tag)
P.stages = {'bench', 'battery', 'figs'};
                             % 'bench'   build + gates + frame + registration (always run)
                             % 'battery' kernels, modal transfer, the five rows, break ladder
                             %           (per DM config in P.dm)
                             % 'color'   the multi-wavelength stage (P.color; P.dm(1) only)
                             % 'noise'   photon-shot pricing of every reading (P.noise)
                             % 'loop'    the closed-loop HOLD metric (P.loop): on-orbit servo mode
                             % 'figs'    PNGs of whatever ran
P.readings = {'L', 'F', 'I', 'I+', 'S', 'V'};
                             % L  frozen-reference linear, 1 frame
                             % F  exact solve, frozen reference wave, 1 frame
                             % I  exact solve, iterated reference wave, 1 frame
                             % I+ I with the base's refined stepped branch prior
                             %    (the base costs 4 frames ONCE; each differential frame is 1)
                             % S  phase-stepped (3 depths + clear), 4 frames
                             % V  VECTOR pair: the polarized (geometric-phase) dimple gives a +phi
                             %    and a -phi pupil image AT ONCE (one per circular polarization);
                             %    exact per-pixel solve with no branch fold, 2 simultaneous frames
                             %    (the photons split between them).  Ideal metasurface: each image
                             %    is the scalar sensor with its own dimple sign (stage V1, 2026-09-11)

% ---- engine + sampling ------------------------------------------------
P.MODEL = 1024;              % engine model size (ONE per MATLAB process -- see README)
P.param_file = '';           % '' = the engine's own macos_param.txt (MACOS_HOME); or a path to
                             % a custom size table, copied into the run dir as macos_param.txt
                             % where the engine looks FIRST -- e.g. 'macos_param_2048.txt'
                             % (this dir): MODEL 2048 trimmed to fit a 30 GB box
P.NGRID = 193;               % ray grid across the source aperture; the pupil image spans
                             % ~NGRID px at the detector.  193 = the dev/record grid,
                             % 385 = the 1 Mpix-class detector (tg96 rule)
                             % THE SAMPLING TRADE (Dave 2026-09-10; measured): the dimple's
                             % size at the mask plane is fill*MODEL/NGRID px per lam/D
                             % (fill 0.74 here: 3.96 px/(lam/D) at 1024/193 and 2048/385,
                             % 1.98 at 1024/385), while the detector's px per actuator go
                             % as NGRID.  Halving NGRID at fixed MODEL doubles the focal
                             % resolution and halves the pupil sampling -- the two budget
                             % lines pull opposite ways; only MODEL buys both.
P.LAM   = 6.328e-4;          % wavelength of record, mm (632.8 nm)

% ---- the bench (macos.design.twyman_green options, test arm only) ------
s = 96/56;                   % uniform scale of the 56 mm v1 rig to the 96 mm DM
P.bench.polarizing = false;  % the ZWFS is the test arm alone: no polarizers, no reference arm
P.bench.BS_AOI   = 7;        % beam-splitter incidence, deg (the tg96 clearance solve)
P.bench.F1       = s*500;    P.bench.F2 = s*250;      % collimator / focusing lens
P.bench.D_LENS   = s*60;     P.bench.R_BAFFLE = s*12.5;  P.bench.D_SB = s*250;
P.bench.BS_T     = s*1.5;    P.bench.D_L1_BS = s*150;    P.bench.D_BS_TO = 700;
P.bench.D_BS_CMP = s*100;    P.bench.R_TO_AP = s*30;     % test-optic (DM) aperture radius
P.bench.L1_Kr = s*236.866;   P.bench.L1_Kc = -0.5829;    % tuned lens figures (l2_trade)
P.bench.L2_Kr = -s*124.076;  P.bench.L2_Kc = -0.5826;
P.bench.tail_arch = 'fieldlens';                         % pupil-relay field lens behind the mask
P.bench.MASK_TRIM = -5.582;  % thin-lens seed -> true focus (S1 rounds 2-5)
P.bench.FL_F   = 42.5325;    P.bench.FL_Kc = -2.58764;  P.bench.FL_D = s*12;
P.bench.D_MASK_FL = 39.7694; P.bench.DET_TRIM = -1.2473; % tuned tail (tg96_tail)
P.bench.mask_prop = 'nf';    % 'nf' = SYMMETRIC reference-sphere sandwich about the mask
                             % (the corrected model, S7); 'nf_legacy' reproduces the
                             % Fresnel-DEFOCUSED S1-S6 sensor (record only)

% ---- the DM surface grid (GridData on the test optic) ------------------
P.grid.N_G  = 384;           % grid points across (needs MODEL 1024: mGridMat caps at 256 below it)
P.grid.DX_G = 0.28;          % grid pitch, mm (384 x 0.28 = 107.5 mm > the 96 mm DM)
P.grid.flat_file = 'zwfs_flat.txt';

% ---- the mask (VSG2 hardware, vsg_wip/vsg2_params.m section 9) ---------
P.mask.ETCH_MM  = 346.2e-6;  % etch depth in fused silica (346.2 nm ~ quarter wave at 632.8)
P.mask.index    = 'malitson';% substrate index: 'malitson' (fused silica, chromatic) or a number
P.mask.DIA_LAMD = 2.0;       % dimple diameter in lambda/D AT P.LAM (fixed physical size across colors)
P.mask.PHIS_REC = [pi/2, pi, 3*pi/2];
                             % phase-stepped depth ladder, phases at P.LAM; each depth is
                             % fixed glass, so at other colors they scale as (n-1)/lambda
P.mask.S_CONV = -1;          % height sign convention (pinned by the S1 sign gate)
P.mask.NITER  = 5;           % reference-wave iterations of the exact readings (I, V)
P.mask.v_gate_nm = 100;      % G4 (V only): single-actuator pokes (every 8th actuator) of this height
                             % put their pixels beyond the one-frame fold (peak 1.9 rad, 3% of msk);
                             % the pair must reproduce them (< 0.1%), the single frame must not

% ---- sampling budget (asserted at the bench stage) ---------------------
P.samp.min_dimple_px  = 6;   % dimple diameter at the mask plane, px (S1 G0 rule)
P.samp.min_px_per_act = 2;   % detector px per actuator on the reimaged pupil (Nyquist)
P.samp.enforce = 'warn';     % 'warn' | 'error' when a budget line is not met

% ---- registration (two-poke doctrine: 4 DOF classes, deck-dependent) ---
P.reg.mode = 'search';       % 'search' = parity + sign from an off-center poke (dmg_register);
                             % 'record' = use P.reg.PARb / P.reg.sgn as given
P.reg.PARb = [1 2 1 1];      % the record's parity for this deck (S2)
P.reg.sgn  = +1;             % the record's measurement sign
P.reg.POKE = 20e-6;          % registration/kernel poke, mm (20 nm: inside the linear range)
P.reg.min_corr = 0.4;        % selection gate: |corr| of the winning parity
P.reg.min_sep  = 0.3;        % and its separation from the runner-up
P.reg.hw = 6;                % kernel stencil half-width, actuators
P.reg.stencil_site = 'lattice';
                             % where the kernel stencil is sampled about the poke: 'lattice' =
                             % the exact actuator centre (default since 2026-09-10: own-site
                             % gain 0.958 -> 0.992, test-actuator gain 0.900 -> 0.946 at 193);
                             % 'grid' = the map-grid point nearest the poke's peak (up to half
                             % a grid pitch, 0.14 mm, off centre) -- the S1-S8 record; use it to
                             % reproduce zwfs_s7iter_report.txt
P.reg.kernel_site = 'center';% where the response kernel is measured: 'center' (the record),
                             % 'hold' (the test actuator, P.dm(i).hold) or [row col].  The
                             % registration anchor always comes from the centre poke.

% ---- DM configurations (each gets the full battery) ---------------------
P.dm(1).nact  = 96;  P.dm(1).pitch = 1.0;  P.dm(1).hold = [60 40];
P.dm(1).PQ = [1 0;2 0;4 0;8 0;16 0;24 0;32 0;40 0;48 0;56 0;64 0;72 0;80 0;8 8;24 24];
P.dm(2).nact  = 48;  P.dm(2).pitch = 2.0;  P.dm(2).hold = [30 20];
P.dm(2).PQ = [1 0;2 0;4 0;8 0;12 0;16 0;24 0;32 0;40 0;8 8;16 16];
P.dm_use = [];               % which P.dm entries to run ([] = all; e.g. 1 = 96x96 only)
P.hold   = [];               % if set, replaces every P.dm(i).hold (the test actuator)
                             % hold = the held-out single actuator (row, col);
                             % PQ = modal probes cos(pi p x) cos(pi q y): (p,0) rows carry the
                             % separable transfer, (p,p) rows are the separability check

% ---- the battery ---------------------------------------------------------
P.battery.AMPM     = 10e-6;  % modal-probe amplitude, mm
P.battery.base_rms = 30e-6;  % the random working state, mm rms (seed P.battery.seed_base)
P.battery.dev_single = 10e-6;% single-actuator differential, mm
P.battery.dev_rand = 10e-6;  % random differential, mm rms (seed P.battery.seed_dev)
P.battery.grid_amp = 1e-6;   % grid-poke differential, mm (1 nm: the sensitivity-stage floor test)
P.battery.grid_step = 8;     % grid pokes every N actuators
P.battery.seed_base = 7;     P.battery.seed_dev = 23;
P.battery.BETA = 0.1;        % Wiener modal-correction damping
P.battery.act_lam = 0.05;    % Tikhonov weight of the actuator fit (relative to the stencil peak)
P.battery.ladder = [30 40 50 60 120 240 480]*1e-6;
                             % break-scale ladder: working-state rms, mm (same field, scaled)
P.battery.ladder_sites = 'hold';
                             % the ladder's differential: 'hold' = dev_single on the hold-out
                             % actuator alone (the record; one site, so its fold status is
                             % that site's); 'grid' = dev_single on every grid site
                             % (grid_step) -- gain and floor over ~50 sites, the robust form
P.battery.rows = {'flat/hold', 'flat/rand', 'base/single', 'base/grid', 'base/rand'};
                             % the differential rows; any subset in this order
P.battery.calib_mode = 'matrix';
                             % 'matrix' (default since 2026-09-10, Dave) = the MEASURED response
                             % matrix dw/da: every lit actuator poked once in sparse multiplexed
                             % grids (matrix_step), its response cut from its own detector
                             % window, the sensor's piston null carried as a rank-one term;
                             % estimator = regularized least squares on that matrix -- no
                             % single-site kernel, no frequency correction (measured response
                             % 0.98-1.07 at every frequency; single-actuator test 0.994 / 4 pm);
                             % 'kernel' = one measured response kernel (at reg.kernel_site) +
                             % lattice deconvolution + the modal correction -- the S1-S9 record
P.battery.matrix_step = 8;   % grid step of the multiplexed pokes (no response overlap at 8)
P.battery.matrix_lam  = 1e-3;% Tikhonov weight relative to the median column energy of J
P.battery.matrix_sign = 'same';
                             % 'same' = all pokes positive (default: single-actuator test 0.994 /
                             % 4 pm vs 0.987 / 11 pm alternating; the linear reading's +/-
                             % asymmetry costs it 0.67 under alternation); 'alternate' =
                             % checkerboard of +/- pokes over each grid -- zero-mean pattern, no
                             % shared pedestal, and on a real bench common-mode drift cancels
                             % between the sets (Dave 2026-09-10); the piston-null term handles
                             % the pedestal in either case
P.battery.calib_surface = 'flat';
                             % 'flat' = kernel + modal transfer measured on the flat DM (the
                             % record); 'base' = measured on the working surface itself
                             % (base_rms, seed_base), differentially, the exact class read
                             % with the base's refined sign map -- the calibration a bench
                             % would make in place

% ---- multi-color stage (Dave 2026-09-08) ---------------------------------
P.color.lams_nm  = [632.8 480 532 700 780];   % P.LAM's color FIRST (lit + bases + record tie-in)
P.color.readings = {'L', 'I+', 'S'};          % readings to combine (subset of P.readings)
P.color.rows     = {'flat/hold', 'flat/rand', 'base/single', 'base/grid', 'base/rand'};
P.color.BETA     = 0.1;
P.color.dc       = 'unit';                    % combiner DC form: 'unit' (g(0)=1) | 'record'

% ---- photon-noise stage ----------------------------------------------------
P.noise.nstates = 10.^(6:2:14);               % photons per MEASUREMENT of one DM shape (split over a
                                              % reading's frames; the knob keeps its historical name)
P.noise.nreal   = 8;                          % Monte-Carlo realizations per point
P.noise.readings = {'L', 'F', 'I', 'I+', 'S', 'V'};
P.noise.prior   = {'split', 'noiseless'};     % I+ prior frames: 'split' = the base's 4 stepped
                                              % frames share ONE state budget (N/4 each);
                                              % 'full' = N per frame; 'noiseless' = the prior
                                              % treated as calibration
P.noise.seed = 1000;

% ---- closed-loop hold stage (Dave 2026-09-11: the on-orbit metric) ----------
% The DM is held at the working surface by a proportional loop closed through
% ONE reading: each cycle the state is traced, photon noise injected, the
% differential to the set point's frames fitted through the measured matrix,
% and g times the estimate removed.  Shared loop code: dm_gauge_lib/dmg_loop
% (the IFO runs the identical loop, drift realizations and scoring).  The
% metric = steady-state rms surface error over lit (pm) against each drift,
% as a curve in photons per cycle; the ONE number = photons per cycle to hold
% P.loop.hold_spec.  Cost: K+1 traced states per (reading, drift, photon level).
P.loop.readings = {'L', 'I+', 'S', 'V'};      % subset of P.readings (1 / 1 / 4 / 2 frames per measurement)
P.loop.surface  = 'base';                     % the set point: 'base' = the working surface
                                              % (battery.base_rms, seed_base) with the matrix
                                              % calibrated ON it (S10) | 'flat'
P.loop.g        = 0.5;                        % loop gain
P.loop.K        = 60;                         % cycles (steady state = the last K/2)
P.loop.nph      = [1e12 1e13 1e14 1e15];      % photons per MEASUREMENT, one per cycle (a reading's frames share it)
P.loop.drifts   = {'walk', 'thermal'};        % drift models run at every photon level
P.loop.walk_sigma   = 2e-9;                   % mm per actuator per cycle (2 pm random walk)
P.loop.thermal_rate = 5e-9;                   % mm rms per cycle of a defocus + astigmatism ramp (5 pm)
P.loop.steps    = [1e-6 10e-6];               % mm rms: NOISELESS step disturbances at cycle 1 -- the
                                              % time constant (G1) and the dynamic range (a step the
                                              % reading cannot track leaves a residual)
P.loop.floor    = true;                       % also the noise-only loop at every photon level (G2)
P.loop.ref      = 'noiseless';                % set-point frames: 'noiseless' (calibration-grade,
                                              % averaged) | 'noisy' (ONE exposure at nph: its noise
                                              % is a fixed bias the loop converges to)
P.loop.seed     = 77;                         % the drift realization (the IFO uses the same seed)
P.loop.hold_spec = 3e-9;                      % mm: the hold level priced in photons per cycle (3 pm)
P.loop.rmax     = 1e-3;                       % mm: a residual above this declares the run DIVERGED and
                                              % stops it (the exact one-frame reading I+ diverges on the
                                              % 30 nm surface: runs/loop193); Inf = never
end
