function P = tg96_params()
%TG96_PARAMS  Parameterized knobs for the 96x96 Twyman-Green DM gauge, in the
%   ONE-runner form (edit this + rerun tg96_run, no AI).  Every knob is at its
%   value of record from tg_psi_dm96/tg96.m; the ONLY additions are the
%   reflective ('optics') knobs.  Edit fields and call tg96_run(tg96_params()),
%   or override on the fly: tg96_run('bench.optics','oap','tag','oap').
%
%   The refractive record lives (untouched) in ../tg_psi_dm96/.  This runner
%   drives BOTH the lens rig (equivalence gate vs that record) and the OAP rig
%   (the reflective variant) through the SAME code path -- flip bench.optics.

% ---- run control -----------------------------------------------------
P.tag    = 'lens';                 % names runs/<tag>/<tag>_report.txt
P.outdir = '';                     % '' => <this dir>/runs/<tag>
P.stages = {'bench','battery','figs'};   % bench | battery | figs

% ---- engine + sampling (LOAD-BEARING; see tg96.m Stage A2) -----------
P.MODEL  = 1024;                   % mGridMat caps grids at 256 on model 512;
                                   %  the 384 DM grid needs 1024 (heap guard)
P.NGRID  = 385;                    % pupil image ~385 px (Nyquist ~192 cyc/pup)
P.LAM    = 6.328e-4;               % HeNe, mm
P.param_file = '';                 % '' => engine macos_param.txt; else a trim
                                   %  table copied into the run dir (memory)

% ---- the DM + surface grid -------------------------------------------
P.dm(1).nact = 96;  P.dm(1).pitch = 1.0;   % 96 mm aperture, 1 mm pitch
P.dm(2).nact = 48;  P.dm(2).pitch = 2.0;   % DST-class twin, same 96 mm
P.grid.N_G  = 384;                 % nGridMat (>= max DM grid; mGridMat guard)
P.grid.DX_G = 0.28;                % GridSrfdx mm (3.57 px/actuator at 96x96)
P.grid.flat_file = 'tg96_flat.txt';
P.POKE = 50e-6;                    % mm (50 nm calibration commands)

% ---- PSI ------------------------------------------------------------
P.QWP    = 0.25;                   % quarter-wave retardance
P.THETAS = [0 45 90 135];          % analyzer four-step

% ---- Stage-A clearance solve (folded layout re-solve for OAP) --------
P.clear.beam_r  = [];              % [] => s*30 (scaled R_TO_AP)
P.clear.HW_DM   = 90;   P.clear.HW_REF = 60;  P.clear.HW_CAM = 50;
P.clear.MARGIN  = 25;   P.clear.LEG_CAP = 700;

% ---- the bench (macos.design.twyman_green options; s = 96/56 applied
%      in tg96_run so the whole rig scales uniformly off the 56 mm v1) --
P.bench.polarizing = true;
P.bench.BS_AOI     = [];           % [] => Stage-A solved AOI (=7 at record)
P.bench.F1 = 500;   P.bench.F2 = 250;      % *s in the runner
P.bench.D_LENS = 60;  P.bench.R_BAFFLE = 12.5;  P.bench.D_SB = 250;
P.bench.BS_T = 1.5;   P.bench.D_L1_BS = 150;    P.bench.D_BS_CMP = 100;
P.bench.D_BS_TO = [];              % [] => Stage-A solved DM leg
P.bench.R_TO_AP = 30;
P.bench.L1_Kr = 236.866;  P.bench.L1_Kc = -0.5829;   % lens seeds (ignored oap)
P.bench.L2_Kr = -124.076; P.bench.L2_Kc = -0.5826;
P.bench.qwp_ret = 0.25;  P.bench.pol_in_deg = 45;
P.bench.qwp_test_deg = 0;  P.bench.qwp_ref_deg = 45;
P.bench.out_qwp_deg = 0;   P.bench.analyzer_deg = 0;
P.bench.tail_arch = 'fieldlens';
% l2_trade tail winner (scaled *s in the runner); re-tuned per optics from
% tg96_tail.mat when present (the tail was fit to L2 -- MUST re-run for OAP)
P.bench.FL_F = 25.02100857;  P.bench.FL_Kc = -2.11278288;
P.bench.FL_D = 12;  P.bench.D_MASK_FL = 6.277463741;  P.bench.DET_TRIM = 1.085330067;

% ---- reflective knobs (the ONLY additions vs the record) -------------
%   'lens' reproduces the record; 'oap' is the all-reflective variant.
%   The fold AOIs are re-solved by Stage A for the folded source->OAP1 and
%   OAP2->detector legs (near-normal preferred; must clear the bodies inside
%   LEG_CAP).  [] => the Stage-A solved value; a number pins it.
P.bench.optics    = 'lens';        % 'lens' | 'oap'
P.oap.OAP1_AOI    = [];            % [] => Stage-A solved; deg
P.oap.OAP2_AOI    = [];
P.oap.OAP1_SIDE   = 1;   P.oap.OAP2_SIDE = 1;

% ---- battery selection ----------------------------------------------
P.battery.piston_nm  = 20;
P.battery.single_nm  = 150;        % Stage-C single-actuator poke
P.battery.reg_act    = [30 64];    % off-center poke for registration parity
P.battery.transfer_PQ = [1 1; 2 2; 4 4; 8 8; 16 16; 24 24; 32 32; ...
                         48 48; 64 64; 80 80; 96 96; 48 0];
P.battery.rand_seed  = 11;         % held-out random command seed
% Stage-E differential rows (the pm product the ZWFS comparison needs)
P.battery.diff_single_nm = 10;     % single-act deviation
P.battery.diff_rand_nm   = 10;     % dense-random deviation
P.battery.diff_rand_seed = 23;
P.battery.base_rand_nm   = 16;     % working-state random base (rms-ish)
P.battery.base_rand_seed = 11;

% ---- calibration mode (Dave 2026-09-10; the ZWFS S10 default) ---------
%   'matrix' = the MEASURED response matrix dw/da: poke every matrix_step-th
%     actuator on a sparse grid (no response overlap), step through the
%     matrix_step^2 offsets so every lit actuator is poked once, cut each
%     response from its OWN detector-pixel window placed by the ray affine
%     (dmg_frame + tg96_place), assemble J (detector px x lit act) and
%     estimate commands by regularized least squares.  Registration only
%     PLACES the windows -- the columns carry the actual response, so the
%     fold's flip/rotation/scale and a real DM's irregularities are in the
%     calibration by construction.  Comparisons are in ACTUATOR units (pm).
%   'kernel' = the record: register_two_pokes + one interpolated truth map,
%     compared in detector-pixel space (Stage C-E as first shipped).
P.battery.calib_mode  = 'matrix';  % 'matrix' (default) | 'kernel' (the record)
P.battery.matrix_step = 8;         % sparse-poke grid step (no overlap at 8; hw < step/2 pitch)
P.battery.matrix_lam  = 1e-3;      % Tikhonov weight, relative to median column energy of J
P.battery.matrix_sign = 'same';    % 'same' | 'alternate' (zero-mean checkerboard; halos cancel)
P.battery.matrix_states = inf;     % cap on J-build states (inf = all step^2 = every lit act once)
P.battery.matrix_window = 'box';   % 'box' (+/-half-step window) | 'voronoi' (nearest-poke cells; item 3a)
P.battery.matrix_lam_sweep = [1e-3 1e-4 1e-5];  % reg sweep on the dense-random row (bright vs dark; item 3b)
P.battery.break_ladder = [30 60 120 240 480];   % base working-state rms (nm) for the break ladder
% ---- D4 alignment sensitivity (OAP rig): perturb OAP1/OAP2, re-read --------
P.battery.d4 = false;              % true => Stage D4 (OAP1/OAP2 decenter + tilt sensitivity)
P.battery.d4_dec_um   = 10;        % decenter perturbation (micron)
P.battery.d4_tilt_urad = 10;       % tilt perturbation (microradian)
P.battery.calib_surface = 'flat';  % 'flat' (the record) | 'base' (differential on a working state)
P.battery.base_rms    = 30e-6;     % working-state rms (mm) for calib_surface 'base' (seed_base)
P.battery.seed_base   = 7;

% ---- window placement (the affine route; Dave 2026-09-11) ------------
P.place.mode      = 'affine';      % 'affine' = ray-affine + resolved field parity (both rigs)
P.place.gate_px   = 2;             % D1 gate: response CoM within gate_px of predicted (u,v)
P.place.gate_frac = 0.99;          % ... for >= this fraction of lit actuators
P.place.resolve   = true;          % resolve the field-array parity against a reference poke
P.place.poly_deg  = 1;             % refit degree: 1=affine (both rigs; fit is robust to outliers)
P.place.gate_max_states = inf;     % cap the D1-gate sweep states (dev: sample a few)
P.place.boot_states = 8;           % states for the placement bootstrap/refit (few suffice)
P.place.gate_assert = true;        % dev: false continues past a failed gate (saves .mat)

% ---- closed-loop hold metric (D7; Dave 2026-09-11, BRIEF_loop_metric) ----
%   The on-orbit servo mode: the DM held at the working surface by a
%   proportional loop closed through the four-step reading and its measured
%   matrix (calibrated ON the working surface, S10). ONE reading here (the
%   four-step PSI map), so no readings dimension -- the ZWFS runs L/I+/S/V.
%   The loop code is shared: ../dm_gauge_lib/dmg_loop.m (gated by
%   tests/tDmgLoop.m). Same knobs and seed as the ZWFS P.loop so the two
%   instruments run the IDENTICAL drift realizations. Cost: K+1 traced
%   states per (drift, photon level) -- an hour-class job at MODEL 1024.
P.loop.surface  = 'base';          % set point: 'base' = the 30 nm working
                                   %   surface with the matrix ON it (base_rms,
                                   %   seed_base); 'flat' = the flat DM
P.loop.g        = 0.5;             % loop gain
P.loop.K        = 60;              % cycles (steady state = the last K/2)
P.loop.nph      = [1e12 1e13 1e14 1e15];  % photons per MEASUREMENT, one per
                                   %   cycle (the four frames share it, nph/4 each)
P.loop.drifts   = {'walk', 'thermal'};    % drift models run at every photon level
P.loop.walk_sigma   = 2e-9;        % mm per actuator per cycle (2 pm random walk)
P.loop.thermal_rate = 5e-9;        % mm rms per cycle of a defocus + astigmatism ramp (5 pm)
P.loop.steps    = [1e-6 10e-6];    % mm rms: NOISELESS step disturbances at cycle 1
                                   %   (time constant + dynamic range)
P.loop.floor    = true;            % also the noise-only loop at every photon level (G2)
P.loop.ref      = 'noiseless';     % set-point frames: 'noiseless' (calibration-grade)
                                   %   | 'noisy' (single-shot; a fixed bias)
P.loop.seed     = 77;              % drift realization (SHARED with the ZWFS)
P.loop.hold_spec = 3e-9;           % mm: the hold level priced in photons per cycle (3 pm)
P.loop.rmax     = 1e-3;            % mm: a residual above this declares the run DIVERGED

% ---- dev / smoke -----------------------------------------------------
P.smoke = false;                   % true => Stage-A2 sampling asserts become warnings
                                   %   (code-path checks at coarse MODEL/NGRID; NOT a result)
end
