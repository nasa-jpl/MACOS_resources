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
P.stages = {'bench','battery','figs'};   % clearance | bench | battery | figs
                                   %  'clearance' prints dmg_bench_clearance's
                                   %  part-by-part table into the report (it traces
                                   %  both arms at P.MODEL before Stage B builds)

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

% ---- phase-shift form (deck item 3): the four frames are the same in the
%      model; the forms differ in what they get wrong.  The PZT four-step is
%      SEQUENTIAL, so it carries a phase-step miscalibration and within-scan
%      drift; the polarization snapshot takes all four at once (no within-scan
%      drift) but carries polarization systematics.  step_err is TO's
%      pdi.step_err pattern: a fractional error applied to the step sizes in
%      the FRAMES only (the atan2 solve assumes the nominal pi/2 quadrature).
%      Default 0 => frames at the nominal steps => byte-identical to the record.
P.pzt.step_err = 0;                % fractional four-step phase-step error (0 | 0.02 | 0.05 ...)

% ---- Stage-A clearance solve (folded layout re-solve for OAP) --------
P.clear.beam_r  = [];              % [] => s*30 (scaled R_TO_AP)
P.clear.HW_DM   = 90;   P.clear.HW_REF = 60;  P.clear.HW_CAM = 50;
P.clear.MARGIN  = 25;   P.clear.LEG_CAP = 700;
P.clear.MOUNT   = 8;               % mount ring beyond a part's aperture radius --
                                   %  the same 8 mm dmg_bench_clearance uses, so the
                                   %  Stage-A rule and the tool's table agree
P.clear.node    = true;            % solve the splitter angle against the NODE parts
                                   %  too (L1, input polarizer, compensator, output
                                   %  QWP, analyzer, L2), not just the three end
                                   %  bodies.  Dave 2026-09-15: at the record's 7 deg
                                   %  eight of nine node parts sat in another beam --
                                   %  "this is not buildable".
P.clear.BODY = struct();           % the parts' PHYSICAL bodies (part stem ->
                                   %  radius before the mount, mm) for the
                                   %  measured clearance table.  EMPTY = the
                                   %  record: apertures only, which omits the
                                   %  SOURCE head entirely (its builder element
                                   %  is an Obscuring baffle, not an optic) and
                                   %  scores the camera at its pupil-image
                                   %  size.  The reflective runs pass the
                                   %  Stage-A rule's own half-widths --
                                   %  struct('Baffle',50,'Detector',50,
                                   %  'TestOptic',90,'PZT',60) -- so the screen
                                   %  and the measurement describe the same
                                   %  parts.  Left empty by default so the
                                   %  lens rig's recorded table (REPORT_bench_
                                   %  realism section 2) reproduces exactly.
P.clear.plate_over = 5;            % a builder plate carries no aperture: its radius
                                   %  is the beam + this (dmg_bench_clearance's rule)

% ---- the bench (macos.design.twyman_green options; s = 96/56 applied
%      in tg96_run so the whole rig scales uniformly off the 56 mm v1) --
P.bench.polarizing = true;
P.bench.BS_AOI     = 22.5;         % Dave 2026-09-15: pinned (the Stage-A solve's 7 deg cleared only the
                                   % end bodies; the node parts need >= 22.5: dmg_bench_clearance)
P.bench.D_RECOMB   = 150;          % physical mm (NOT scaled by s): the recomb plane and the output
P.bench.D_RC_L2    = 55;           % optics 150 mm behind the splitter, L2 at 150 + 55 = 205 as before
P.bench.F1 = 500;   P.bench.F2 = 250;      % *s in the runner
P.bench.D_LENS = 60;  P.bench.R_BAFFLE = 12.5;  P.bench.D_SB = 250;
P.bench.BS_T = 1.5;   P.bench.D_L1_BS = 150;    P.bench.D_BS_CMP = 200/(96/56);   % compensator at 200 mm physical (x s in the runner)
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
% Where the input polarizer lives.  'collimated' (the record) puts it D_POL
% past the collimator.  That works for a LENS, whose conjugate leg is on-axis;
% an OAP collimator's conjugate leg comes BACK along the collimated axis, and
% at 10 mm past the pole the two legs are 10*tan(2*AOI) apart, so the
% polarizer sits inside the incoming cone at EVERY fold angle -- measured
% -102 mm of clearance at 5 deg and still -80 mm at 30 deg (oap_fold_solve,
% runs/fold1).  'source' puts it in the diverging leg, D_POL past the baffle,
% which is where a real reflective bench polarizes anyway.  Ignored for 'lens'.
P.bench.POL_IN    = 'collimated';  % 'collimated' | 'source' (oap only)
% Feed the collimator at its TRUE focus.  Bench emits zSource (25 mm) and the
% engine puts the real point source at ChfRayPos + zSource*ChfRayDir, so the
% source sits 25 mm inside the parabola's focus -- measured 926 urad rms of
% residual convergence (a 28.8 m focus), which an OAP turns into coma LINEAR in
% the fold angle: 0.13 / 0.37 / 0.65 / 1.08 lambda F/D of best-focus blur at
% 1 / 5 / 9 / 15 deg, and a 6.46 mm mask-seat trim at EVERY angle -- CCMac's
% 6.14 mm.  Corrected: 0.000 lambda F/D and 0.00 mm trim at every angle
% (oap_conj_probe, runs/conj).  The LENS rig hides the same error in its tuned
% L1 figures, so this is 'oap' only and default false = the record.
P.bench.SRC_AT_FOCUS = false;      % true => the collimator is fed at its focus
P.bench.tail_from_mat = true;      % false => use the GEOMETRIC SEED tail even if
                                   %  <tag>_tail.mat / <optics>_tail.mat exists.
                                   %  The seed-vs-tuned A/B when a reading
                                   %  misbehaves; without it the lookup falls
                                   %  back to another bench's tail.
P.oap.OAP1_AOI    = [];            % [] => Stage-A solved; deg
P.oap.OAP2_AOI    = [];
P.oap.OAP1_SIDE   = 1;   P.oap.OAP2_SIDE = 1;

% ---- OAP coating (D5 / brief item B): the polarization-cost row ------
%   'none'        ideal reflector (RS=-1, RP=+1, zero retardance) -- the D3
%                 baseline; the lens/OAP comparison stays geometric.
%   'bareAl'      a single opaque aluminium layer (n - i*kappa at HeNe
%                 632.8 nm, Rakic 1998); the physical bare-metal reflection.
%   'protectedAl' MgF2 half-wave overcoat over opaque Al (the realistic
%                 mirror). Applied via macos.coating (= coat_set) to BOTH
%                 OAPs (L1 collimator, L2 focuser) in BOTH arms -- shared
%                 tail optics, so its retardance is a common-mode term.
%   Thickness in mm (bench BaseUnits). Ignored when bench.optics ~= 'oap'.
P.bench.coat_oap  = 'none';        % 'none' | 'bareAl' | 'protectedAl'
P.bench.coat_bareAl      = struct('index',1.373, 'extinc',7.62, 'thickness',1.0e-4);
P.bench.coat_protectedAl = struct('index',[1.38 1.373], 'extinc',[0 7.62], ...
                                  'thickness',[2.293e-4 1.0e-4]);  % [MgF2 lambda/2 ; Al opaque]

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
% ---- deck rows (item 1) + capture range (item 2), the ZWFS convention -------
%   Stage DECK reproduces the ZWFS currency on the SAME 30 nm working surface
%   (base_rms, seed_base) with the matrix measured ON it: rows scored by
%   gain / floor / SNR exactly as zwfs_run's score_, the 47-site grid row
%   (dmg_lit + every-8th actuator), the capture-range-to-10% print, the
%   re-measured ladder, and the S5 photons-for-1-pm fit.  Off by default; the
%   deck runs turn it on ('battery.deck',true) at MODEL 1024.
P.battery.deck        = false;                   % true => Stage DECK (rows + capture + photons)
P.battery.dev_single  = 10e-6;                   % single-actuator differential, mm (10 nm at the hold-out site)
P.battery.dev_rand    = 10e-6;                   % dense-random differential, mm rms (seed seed_dev)
P.battery.grid_amp    = 1e-6;                    % grid-poke differential, mm (1 nm on the 47 sites)
P.battery.grid_step   = 8;                       % grid pokes every N actuators (Afig(4:8:end,4:8:end))
P.battery.seed_dev    = 23;                      % dense-random differential seed
P.battery.cap_ladder  = [30 40 50 60 80 100 120 160 240 480]*1e-6;  % aging ladder (matrix once on 30 nm)
P.battery.recap_surf  = [60 90 120 160]*1e-6;    % re-measured surfaces (matrix rebuilt on each; the 1 nm grid row)
% ---- photons for 1 pm (item 2; the S5 noise-stage form) ---------------------
%   sigma ~ c/sqrt(N) fit of the single-10-nm-on-the-surface estimate noise,
%   matrix on the surface; N(1 pm) = c^2.  Run at each of noise_surf.  This is
%   NOT the loop's sig_n (a single-shot estimate at one photon level) -- the
%   report states both.
P.battery.noise       = false;                   % true => append the S5 photon fit to Stage DECK
P.battery.noise_nph   = [1e11 1e12 1e13 1e14 1e15];  % photons per measurement swept for the fit
P.battery.noise_nreal = 24;                      % Monte-Carlo realizations per photon level
P.battery.noise_seed  = 91;                      % noise realization seed
P.battery.noise_surf  = [30 60 120 160]*1e-6;    % working-surface rms at which N(1 pm) is fit
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
% ---- item 3: within-scan drifts of the PZT four-step (sequential form) ------
%   The polarization snapshot takes the four frames at once and is immune; the
%   PZT four-step steps them in time and is not.  Two drifts sit within a scan:
%   the CAMERA 1/f offset (dmg_loop 'cam'; Dube 2024) and -- once TO lands
%   loop.intra in dmg_loop -- the DM's own walk within the four frames.  A
%   zero-sum four-step is exactly immune to a within-scan-CONSTANT offset
%   (cam_intra 0); cam_intra > 0 develops the offset frame-to-frame and breaks
%   the immunity.  Add 'cam' to P.loop.drifts to price it.  The PZT step error
%   (P.pzt.step_err) is priced in the loop too when set.
P.loop.cam_walk  = 0.13;           % CAMERA offset random-walk, per cycle (unit below)
P.loop.cam_unit  = 'rel';          % 'e' = electrons per pixel per cycle | 'rel' = fraction of the
                                   %   scan's mean photons per lit pixel per frame (a signal-scaled bias)
P.loop.cam_intra = 0;              % fraction of each camera step that develops WITHIN a scan (0 = immune)
% ---- item 3 within-scan DM drift + item 5 descent (TO's shared dmg_loop knobs,
%      landed 2026-09-13; mirrored here verbatim into the loop stage) ----------
P.loop.intra     = 0;              % fraction of the NEXT cycle's DM drift that develops WITHIN a scan
                                   %   (the four-step steps its frames in time and pays for it; 0 = DM still)
P.loop.ref_walk  = 0;              % rms (rad/cycle) of a reference-arm (PZT-flat) phase walk -- the IFO's
                                   %   non-common-path term (default off; not asked for the deck)
% ---- descent (item 5): capture the DM's initial figure, ~100-200 nm WFE -------
P.loop.start_rms   = [];           % [] = no descent; else the loop STARTS from a surface of this rms
                                   %   (mm; a vector runs the ladder), matrix measured AT the start
P.loop.start_shape = [];           % [] = the set point's own field rescaled ("the same field, scaled")
P.loop.recal_every = 0;            % cycles between on-surface re-calibrations (0 = never)
P.loop.recal_list  = [];           % descent: recal_every values to compare ([] => [recal_every])
P.loop.reach       = [10e-6 3e-9]; % descent columns: first cycle to 10 nm, to 3 pm (mm)
P.loop.unwrap      = 'auto';       % 'auto' = battery.unwrap OR a descent (start_rms set); true/false force
% ---- unwrap the wrapped four-step differential before the estimator (item 5) --
P.battery.unwrap   = false;        % dm_gauge_lib/dmg_unwrap (2-D least squares on the lit mask); default OFF

% ---- dev / smoke -----------------------------------------------------
P.smoke = false;                   % true => Stage-A2 sampling asserts become warnings
                                   %   (code-path checks at coarse MODEL/NGRID; NOT a result)
end
