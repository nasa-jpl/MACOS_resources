function P = dyson5_params(over)
%DYSON5_PARAMS  Single source of truth for the dyson5 challenge runner.
%
%   P = DYSON5_PARAMS() returns the default parameter set: Joe's EMIT-class
%   VSWIR imaging-spectrometer spec (dyson5_guidance.txt), the fixed problem
%   this challenge is scored against.  P = DYSON5_PARAMS(OVER) applies the
%   fields of struct OVER on top (the offset_imager pattern) -- how another
%   instrument is run without touching this file:
%
%       P = dyson5_params(struct('Fno',2.2,'band_m',[2040 2380]*1e-9));
%
%   Every stage of DYSON5_RUN reads ONLY this struct.  Fields:
%
%   The spec (Joe, 2026-09; recorded verbatim, EMIT to the digit)
%     Fno          image-space (air-equivalent) F-number           1.8
%     npix         [spatial spectral] detector format             [3000 500]
%     pixel_m      pixel pitch, m                                 18e-6
%     band_m       [lambda_min lambda_max], m                     [380 2500]e-9
%     smile_px     smile requirement, px (0.2 "might be ok")      0.1
%     keystone_px  keystone requirement, px                       0.1
%     srf_px       SRF FWHM requirement band, px                  [1.5 2.0]
%     xrf_px       XRF FWHM requirement, px                       1.5
%     slit_px      slit width in pixels (Jim: 2-px slits common)  2
%   Jim's realism, recorded alongside the spec (not scored against):
%     as-built SRF 2.5-3 px; photon-limited, not diffraction-limited;
%     smile/keystone are the drivers; the GRATING is the stop.
%
%   Derived (computed by the stages, never typed): slit length = npix(1) *
%   pixel_m = 54 mm; FPA spectral height = npix(2)*pixel_m = 9 mm; spectral
%   sampling = (band span)/npix(2) = 4.24 nm/px.
%
%   Form (both chains built by design/src/spectrometer_geom)
%     glass        block material (engine GlassElt name)          'Silica'
%     lambda_ref_m index evaluation wavelength for the layout, m  1.0e-6
%     ap_margin_m, mount_margin_m, slit_mask_m, pkg_*  mechanics: declared
%                  apertures (footprint + margin), mount margin for the
%                  clearance gate, slit mask plate, FPA package model
%     y_slit_offner_m  the Offner's slit ring radius (0.22 R, addendum 6)
%     offner_*     the Offner's solved corrections (offner_solve): convex
%                  radius factor, M3 radius factor, M3 centre offsets
%     twin_rung    which s3 rung's deck the propagation twin runs on
%     y_slit_m     slit centre offset from the concentric axis
%                  along the dispersion direction, m (the FPA lands
%                  on the far side; the two must clear physically)  6e-3
%     block_r_m    Dyson block radius, m (s0's scaling law says
%                  >= 0.213 for a 0.25 px corner blur)             0.22
%     face_offset_m Dyson flat face stands this far beyond the
%                  common centre, so slit and FPA sit in AIR at the
%                  centre plane (the classical seed has 0)          0.5e-3
%     Rg_factor    grating radius / Dyson-condition radius          1
%     order        |diffraction order| (sign solved: dispersion
%                  pushes the FPA away from the slit)               1
%     offner_R_m   Offner concave radius, m (convex grating R/2)    0.5
%     Fno_offner   the Offner sibling's own speed (the paper's
%                  long-slit Offner runs F/2.8; F/1.8 is a Dyson
%                  number)                                          2.8
%     blur_px      s0 layout blur budget: transverse rms spot at the
%                  slit CORNER, in pixels, that the concentric seed
%                  must meet before any element is added           0.25
%
%   Published reference columns (see the block below): mg_rules, mg_table2,
%   mg_table5 -- reported beside Joe's numbers, never scored against.
%
%   Numerics / bookkeeping
%     r_grid_m     block radii swept by the s0 scaling stage, m
%     model        MACOS model size (engine stages)               128
%     tag, outdir  artifact naming; outdir '' = this directory
%     ngridpts     deck ray grid (41 -> ~1200 rays per field point)
%     score_nx, score_nlam   the s2 scoring grid (slit positions x lambdas)
%     blaze_m      scalar blaze wavelength of the radiometric chain
%     qe           [lambda QE] table -- a PLACEHOLDER curve until a real
%                  detector is named; reported, never scored
%     wave_*       the s2w propagation twin: grid, model size, deck ray
%                  grid (odd; the FPA window is ngridpts x lambda F),
%                  reference-sphere radius (m)
%     ladder_*     the s3 departure ladder: rungs, chain scoring grid,
%                  merit weights (distortion vs blur, px), clearance wall,
%                  iteration cap
%     slitloss_*   the s2l slit-loss measurement (wavelengths, model 1024,
%                  grid, modelled slit length, slit-to-grating distance)
%     stages       which stages to run (default {'s0','s1','s2'}; add
%                  's3' for the departure ladder, 's2w' for the twin, 's2l'
%                  for the slit loss -- its own MATLAB at model 1024)
    arguments
        over struct = struct()
    end
    P.Fno         = 1.8;
    P.npix        = [3000 500];
    P.pixel_m     = 18e-6;
    P.band_m      = [380e-9 2500e-9];
    P.smile_px    = 0.1;
    P.keystone_px = 0.1;
    P.srf_px      = [1.5 2.0];
    P.xrf_px      = 1.5;
    P.slit_px     = 2;

    P.glass        = 'Silica';
    P.lambda_ref_m = 1.0e-6;
    P.y_slit_m     = 6e-3;
    P.block_r_m    = 0.22;
    P.face_offset_m = 0.5e-3;
    P.Rg_factor    = 1;
    P.order        = 1;
    P.offner_R_m   = 0.5;
    P.Fno_offner   = 2.8;
    P.blur_px      = 0.25;

    % Published reference columns (Mouroulis & Green 2018, Opt. Eng. 57(4)
    % 040901 -- NOTE_mg2018_digest.md; the PDF is local, never committed).
    % Reported BESIDE Joe's numbers, never scored against:
    %   mg_rules   Sec. 5.3 design principles: distortion ~1 % px at design
    %              (~3 % after tolerancing), > 75 % diffraction energy in the
    %              pixel, degraded spots OK for uniformity, grating = stop
    %   mg_table2  the Fig. 13 long-slit OFFNER's design table (F/2.8, 48 mm
    %              slit, 30 um px, 10 nm/px) -- the performance CLASS
    %   mg_table5  the freeform PRISM Dyson at Joe's regime (3200 px, 18 um,
    %              F/2, 57.6 mm slit, 54 cm long, 19.2 cm prism diameter)
    P.mg_rules  = struct('distortion_px_design', 0.01, 'distortion_px_toleranced', 0.03, ...
                         'ensquared_min', 0.75, 'stop', 'grating');
    P.mg_table2 = struct('form', 'Offner (Fig. 13)', 'Fno', 2.8, 'slit_m', 48e-3, ...
                         'pixel_m', 30e-6, 'sampling_m', 10e-9, ...
                         'smile_px', 0.003, 'keystone_px', 0.02, 'ensquared_min', 0.76, ...
                         'srf_fwhm_x_sampling', 1.35, 'crf_fwhm_x_sampling', 1.10, ...
                         'srf_var_field', 0.045, 'crf_var_lambda', 0.02);
    P.mg_table5 = struct('form', 'freeform prism Dyson (BPDS, Table 5)', 'npix_spatial', 3200, ...
                         'pixel_m', 18e-6, 'Fno', 2.0, 'slit_m', 57.6e-3, 'length_m', 0.54, ...
                         'prism_diam_m', 0.192, 'smile_m_achieved', 0.6e-6, ...
                         'keystone_m_achieved', 0.2e-6, 'uniformity_min', 0.90);

    P.r_grid_m = [0.05 0.075 0.10 0.15 0.20 0.25 0.30 0.40 0.50];
    P.model    = 128;
    P.tag      = 'dyson5';
    P.outdir   = '';
    P.ngridpts   = 41;
    P.score_nx   = 7;                 % slit positions scored (over the 54 mm)
    P.score_nlam = 7;                 % wavelengths scored (over the band)
    P.blaze_m    = 1.0e-6;            % scalar blaze wavelength for the chain
    P.qe         = [380e-9 0.55; 600e-9 0.80; 1000e-9 0.85; 2000e-9 0.80; 2500e-9 0.65];  % PLACEHOLDER QE table
    P.wave_nx = 3;  P.wave_nlam = 3;  P.wave_model = 512;  P.wave_ngridpts = 127;  P.wave_L_ref = 0.1;
    P.ladder_rungs = 0:5;  P.ladder_nx = 5;  P.ladder_nlam = 5;
    P.ladder_w_dist = 10;  P.ladder_w_blur = 1;  P.ladder_clear_m = 3e-3;  P.ladder_max_iter = 60;
    P.ladder_free_r = false;              % true frees the block radius (it walks to its bound)
    % mechanics (addendum 6): declared apertures = footprint + margin; bodies
    % scored with a mount margin; the slit mask and the FPA package at the face
    P.ap_margin_m    = 5e-3;              % aperture beyond the multi-field, multi-lambda footprint
    P.mount_margin_m = 5e-3;              % mount beyond the aperture, for clearance
    P.slit_mask_m    = [0.064 0.004 0.001];   % slit mask plate: length (x), height (y), thickness
    P.pkg_margin_m   = 5e-3;              % FPA carrier beyond the 54 x 9 mm active area
    P.pkg_depth_m    = 10e-3;             % carrier depth on the far side of the face
    P.pkg_shield_m   = 0;                 % cold shield / window height toward the block (0 = none)
    P.y_slit_offner_m = 0.110;            % the Offner's slit ring radius 0.22 R: beside the grating, not through it (addendum 6)
    % the Offner's classical corrections at that ring, solved by offner_solve
    % (2026-10-01, dyson5_s1_offner_solve.txt): convex grating radius factor
    % (x R/2), second concave zone's radius factor and centre offsets
    P.offner_Rg_factor = 1.00340;  P.offner_M3_factor = 0.95097;
    P.offner_M3_dy = 0.294e-3;     P.offner_M3_dz = 0.153e-3;
    P.twin_rung      = 'R4';              % s2w runs the twin on this s3 rung's deck ('' = the s1 seed)
    % s4, the native optimize (dyson_native): CALIB on the rung of record
    P.native_rung = 'R4';  P.native_nx = 5;  P.native_nlam = 6;    % <= 12 FOV x 6 lambda (CALIB's cap)
    P.native_chunk = 5;  P.native_max_chunks = 8;  P.native_tol_px = 0.002;
    P.native_wall_px = 0.05;              % smile/keystone wall on the chain between chunks: half the spec
    P.native_asph = false;                % the block's h^4/h^6 as CALIB DOFs: OFF -- the slice is fixed (0d257ff) but the asphere derivative columns come out EMPTY (gaussj singular; CC)
    P.native_varset = 'blur';             % 'blur' = block face, meniscus, focus; 'all' adds the grating's position (keystone 15 px in 5 iterations, wall-rejected)
    P.native_enabled = true;              % CALIB's SPOT derivative stride fixed (macos 0d257ff); the stage runs
    % s5, R5's fold prism (addendum 10): entrance plate + mirror-coated fold
    % prism cemented to the block, the FPA folded away from the slit; the
    % COLD-SHIELD HEIGHT is the parameter -- each height needs an air gap of
    % height + clearance between the prism's exit face and the FPA, and the
    % design is re-solved (R5 rung) at each; the record is at fold_shield_m
    P.fold_h_m = 16e-3;                   % fold plane depth below the face: the package (9 mm + 2 x 5 mm carrier, centred on
                                          % the fold) must stay below the face plane by the mount margin -- 8 mm stood 1.5 mm INSIDE the block
    P.fold_slit_gap_m = 0.5e-3;           % slit in air before the entrance plate
    P.fold_face_offset_m = 25e-3;         % seed face offset (plate thickness + slit gap); the R5 rung solves it (>= fold + 7.5 + gap)
    P.fold_shield_sweep_m = [0 1e-3 2e-3 3e-3 5e-3];   % cold-shield heights swept (air gap = h + 1 mm, both sides)
    P.fold_shield_m = [];                 % the shield height of record: [] = the TALLEST height that closes (spec + clearance)
    P.fold_shield_clear_m = 1e-3;         % air beyond the shield to the prism's exit face
    P.fold_max_iter = 40;                 % lsqnonlin iterations per sweep point (warm-started along the sweep; 12 variables, ~15 min each)
    % s4env, the closure envelope (addendum 11): R4 re-solved from the record,
    % one axis at a time, judged against the spec; the FPA stays 54 x 9 mm
    P.env_Fno       = [1.6 1.8 2.0 2.2 2.8];
    P.env_block_r_m = [0.15 0.18 0.22 0.26 0.30];
    P.env_slit_m    = [30e-3 40e-3 54e-3 60e-3];   % pixel count follows
    P.env_pixel_m   = [18e-6 30e-6];               % the paper's 30 um; pixel count follows
    P.env_glass     = {'Silica', 'CaF2'};
    P.env_max_iter  = 30;
    P.slitloss_lams = [380e-9 700e-9 1440e-9 2500e-9];  P.slitloss_model = 1024;  P.slitloss_ngrid = 255;
    P.slitloss_len = 0.083e-3;  P.slitloss_z = 0.7;        % s2l: modelled slit length (-> window = 2 x the acceptance at 380 nm), slit-to-grating distance
    P.slitloss_propagating = true;                          % normalise to |sin theta| <= 1 (the planar FFT carries evanescent energy)
    % t1 / t2, THE TELESCOPE (beat 5, BRIEF_to_dyson5 addendum 7: EMIT's
    % parameters): the fore-optics that feed the slit, a three-mirror
    % anastigmat solved on the exact chain (telescope_geom / telescope_ladder)
    % and scored at the slit in the engine (telescope_score), then the
    % instrument traced END TO END as one deck (e2e_geom) and scored by the
    % spectrometer's scorer with the grating as the stop
    P.tel_alt_m      = 420e3;             % orbit altitude (EMIT, ISS)
    P.tel_gsd_m      = 60;                % ground sample distance -> IFOV = gsd/alt = 0.143 mrad per pixel
    %   derived by the stage: f = pixel/IFOV = 126 mm, D = f/Fno = 70 mm, field = npix(1)*IFOV = 24.6 deg
    P.tel_npix_xt    = NaN;               % cross-track pixels the telescope's field covers (NaN = P.npix(1); 1500 = Jim's 1.5k module, addendum 26)
    P.tel_dyson      = 'R4';              % the spectrometer the telescope feeds (pupil match): 'R4' | 'R5' | 'size:<family>:<r_mm>' (a dyson5_size.mat row: 'size:F:240' 3k CaF2, 'size:D:130' 1.5k silica)
    P.tel_t1_m       = 0.14;              % seed: M1 -> M2 spacing (the first-order family's free knob)
    P.tel_y2         = 0.6;               % seed: beam compression at M2 (t2 = f y2 follows from telecentricity + a flat field)
    P.tel_bias_deg   = 0;                 % seed: the field bias across the slit (the coaxial section's knob; the folds do the unobscuring)
    P.tel_tilt_deg   = [32 -32 24];       % seed: the chief's FOLD angle at M1, M2, M3 (Bauer; the scan's open layout, +5.4 mm)
    P.tel_fold_gap_m = 25e-3;             % the fold flat this far before the slit (it moves with the back focus)
    P.tel_fold_dir   = [0 1 0];           % the folded beam's direction in the telescope's local frame (+y: away from the sky side)
    P.tel_rungs      = {'T0', 'T1', 'T2', 'T3'};   % layout (bias, spacings, M2/M3 decentre + tilt, fold distance, wall dominant); conics + radii + spacings + bias; + h^4/h^6 aspheres; + everything
    P.tel_nfield     = 7;  P.tel_nring = 3;  P.tel_max_iter = 120;
    P.tel_w          = struct('blur', 1, 'v', 1, 'map', 1, 'ftheta', 0.1, 'pupil', 1, 'flat', 1, 'clear', 20);   % merit weights (px; walk mm; focus 100 um; wall mm)
    P.tel_clear_m    = 2e-3;              % the wall: legs this far beyond the mount margin from every body
    P.tel_score_nfield = 9;               % the engine score at the slit: fields along the slit
    P.tel_oversize   = 1.0;               % launched bundle / 70 mm (the grating is the stop; > 1 overfills it)
    P.e2e_rungs      = {'R4', 'R5'};      % the spectrometers the telescope is traced into (s3 / s5 records)
    P.e2e_nfield     = 7;  P.e2e_nlam = 7;   % the end-to-end score grid (fields along the slit x wavelengths)
    % t3, THE TELESCOPE THROUGH THE OFFSET_IMAGER LADDER (beat 5b, addenda
    % 19-20): rodgers3's template called (not copied) with the box = the
    % slit's 24.6 deg cross-track x a thin along-track strip, pushed OFF AXIS
    % along-track; each CASE is an ENVELOPE (t1 = M1 -> M2 spacing, through
    % telescope_seed's first-order family: EFL 126 mm, Petzval 0, y2 =
    % P.tel_y2) x an offset, mapped to the template's signed CODE V
    % convention (R1 = -|R1|, spacings [-t1 0 +t2], stop at M2).  Step 1
    % (addendum 19, t1 = 140 mm at 4-10 deg) failed clearance by ~-55 mm at
    % every offset AND on axis: the envelope, not the offset (addendum 20)
    P.tel3_cases     = [0.30 15; 0.45 10; 0.60 8; 0.60 10];   % rows [t1_m offset_deg] (y2 = P.tel_y2; addendum 20's scan, walk >= 80 mm) or [t1_m offset_deg y2] (a t3s-screened row)
    P.tel3_stages    = 1:3;               % template stages run (addendum 20: S1-S3 only until a case packages)
    P.tel3_pack_m    = 5e-3;              % addendum 20's packaging verdict: clearance floor >= this
    P.tel3_s1_conv_nm = 1000;             % S1 dense-map max above this = not converged -> no verdict (addendum 22; converged S1 here: 172 / 289 nm)
    P.tel3_box_al_deg = 0.3;              % along-track full width of the box (the cross-track width is the slit's field)
    P.tel3_clear_m   = [0.005 0.005];     % template clearance list (min = hard knee, max = WARN; template gate PASS at min - 1.5 mm)
    P.tel3_exit_dir  = [0 0 -1];          % exit chief pin in the template frame (three mirrors exit REVERSED; the flat fold turns it to the slit)
    P.tel3_z_m1_m    = 0.2;               % template frame: global z of M1 (arbitrary; beam enters +z)
    P.tel3_lambda_m  = 1.0e-6;            % template WFE wavelength
    P.tel3_nsolve    = 3;  P.tel3_nsolve_s5 = 5;   % odd solve grids (S5 spends on fields)
    P.tel3_gn_iters  = 12;  P.tel3_model = 256;  P.tel3_sampling = 41;
    P.tel3_reuse     = true;              % reuse a finished case (t3/<tag>_t3_t<mm>_off<deg>_sum.mat)
    % t3w, THE y2 CONTINUATION (addendum 23): S1 walked in y2 at t1 fixed,
    % each step warm-started from the previous solved S1 (conics,
    % aspheres, FPA refit carried; spacings + R1 = the family point at the
    % new y2; the R2/R3 branch held by seed_R_m), solved to the LM's own
    % stop (cap tel3w_iters); then S3 at the offset seeded FROM that S1,
    % accepted only if the gate still reads >= tel3_pack_m, else S4 (the
    % clearance hinge) from it
    P.tel3w_t1_m     = 0.14;
    P.tel3w_y2       = [0.6 0.55 0.5 0.45 0.4];   % the walk (a failed step is halved once)
    P.tel3w_off_deg  = 14;                        % the S3 / S4 offset (the screen's packaging row)
    P.tel3w_hold_R1  = true;                      % S1 / S3 hold R1 at the family point (y2 sets M1's power; a free R1 ran 0.70 -> 2.61 m with K1 -141 on the first run)
    P.tel3w_iters    = 40;                        % S1 / S3 / S4 cap; a solve ending AT the cap is reported as capped
    P.tel3w_nsolve   = [5 3];                     % solve set: 5 across the slit x 3 along the strip (oi_fieldset [nx ny])
    P.tel3w_vig_max  = 0.05;                      % hard stop: > 5 % rays lost at the cross-track edge
    P.tel3w_img_max_nm = 250;                     % hard stop: S3 / S4 dense-map max above this with the gate satisfied
    P.tel3w_screen_pass_m = NaN;                  % the screen's pass threshold for choosing the S3 base step (NaN = tel3_pack_m)
    P.tel3w_xtrack_deg = NaN;                     % cross-track box width, deg (NaN = the slit's field, 24.6; 12.3 = one of two modules, addendum 24)
    P.tel3w_s1_only  = false;                     % true: stop after the walk (no S3 / S4 -- addendum 24's S1-only run)
    P.tel3w_suffix   = '';                        % record name suffix (dyson5_t3w<suffix>.*, t3/dyson5_t3w<suffix>_y*)
    % t3o, THE OFFSET SOLVE FROM A t3w PARENT (addendum 25): S3 at the offset
    % seeded from a recorded walk step; a STALL (stated in advance below)
    % switches to a walk in the OFFSET; S4 if S3 loses the gate; S5 once if
    % the result converges above the image bar with the gate held
    P.tel3o_from      = 'dyson5_t3w_x12.mat';   % the walk record whose step seeds S3 (in P.outdir)
    P.tel3o_y2        = 0.40;                   % which step (the packaging corner)
    P.tel3o_xtrack_deg = NaN;                   % the box's cross-track width (NaN = the telescope's field, tel_npix_xt x IFOV; must match the walk's)
    P.tel3o_off_deg   = 14;                     % the target offset
    P.tel3o_off_walk  = [5 10 14];              % the offset walk if the direct S3 stalls (0 = the S1 parent)
    P.tel3o_stall_iters = 5;  P.tel3o_stall_gain = 0.20;   % STALL: stops within 5 iterations, < 20 %% gain, last step rejected
    P.tel3o_s5        = true;                   % run S5 (Zernike freeform) once if converged above the bar with the gate held
    P.tel3o_suffix    = '_x12';                 % record dyson5_t3o<suffix>.*
    P.tel3o_resume    = '';                     % a t3o record (in P.outdir) whose solved S3 is the start: skip S3, run S4 then S5 (addendum 28)
    % t4, THE TWO-MIRROR MODIFIED SCHWARZSCHILD (beat 5c, addenda 29-30): the
    % review's TMS at Jim's numbers (tel_alt_m / tel_gsd_m: 550 km, 30 m ->
    % f 330 mm, D 183 mm at F/1.8); tms_paraxial (flat by equal radii,
    % telecentric by the virtual stop at M2's front focus) + tms_geom (the
    % exact chain), solved on the chain, scored in the engine
    P.tms_d_m        = 0.165;             % M1 -> M2 spacing (the family's one knob)
    P.tms_npix_solve = 3000;              % the strip the rungs SOLVE (3k, 9.4 deg); 1.5k scored on the same decks at half field
    P.tms_npix_score = [3000 1500];       % strips scored in the engine per rung
    P.tms_rungs      = {'seed', 'R1a', 'R1b'};   % aplanat conics; + M2 h^4 (conics, focus re-solved); + M2 h^6 (only if R1a misses a pixel); 'R2' (addendum 31): + d, R2 free (EFL by R1), M1 h^4 h^6, Petzval as a row
    P.tms_nfield_solve = 7;  P.tms_ngrid_solve = 11;   % solve: the slit's 7 fields x an 11 x 11 pupil grid
    P.tms_nfield_score = 9;               % engine score: fields across each strip
    P.tms_max_iter   = 200;               % lsqnonlin iterations per rung
    P.tms_from       = '';                % a t4 record whose rung (tms_from_rung) seeds R2 when the earlier rungs are not run
    P.tms_from_rung  = 'R1a';
    P.tms_r2_npix    = 3000;              % R2 solves THIS module's strip alone (addendum 31: each module its own telescope)
    P.tms_r2_maxfev  = 20000;             % R2 function-evaluation limit (raised to converge)
    P.tms_petzval_w  = 1;                 % R2's Petzval row: the image sag at the strip edge -> geometric blur (um) x sqrt(rays per field)
    P.tms_bias_deg   = 0;                 % R3: the off-axis section -- field bias (along track) held during the solve (R3w walks it: tms_bias_walk)
    P.tms_dec_ep_m   = 0;                 % R3: entrance-pupil decentre in y held during the solve
    P.tms_bias_walk  = [10 20 30 40];     % R3w: the bias walk (deg), each step from the previous solve, the pupil decentre free
    P.tms_clear_req_m = 5e-3;             % R3w: the clearance wall -- every tms_clear pair >= this
    P.tms_wall_w     = 1e4;               % R3w: wall weight, um of residual per mm of deficit, x sqrt(rays per field): dominant over the image
    P.tms_display    = 'off';             % lsqnonlin Display ('final-detailed' to see why a solve stopped)
    P.tms_score_only = false;             % true: re-score the tms_from design (no solve) -- R2c's stepwise engine score
    P.tms_suffix     = '';                % record dyson5_t4<suffix>.*
    P.tms_areal_kg_m2 = 40;               % lightweighted-mirror areal density assumed for the mass line (kg/m^2)
    P.tms_solid_rho  = 2530;  P.tms_solid_aspect = 6;   % and the SOLID alternative: Zerodur, thickness = D/6
    % t3e, END TO END FROM A t3o RECORD (addendum 27: the verdict): the
    % template's final design mapped onto the exact chain (telescope_geom:
    % |R|, spacings, back focus = |BFD| - the refit dz, K, aspheres, the
    % offset as the field bias), the mapping GATED template-engine vs chain,
    % the telescope scored at the module's slit (chain + engine), then
    % telescope + the module's own Dyson as ONE deck (e2e_geom, the grating
    % the stop) through the spectrometer's scorer and the clearance gate
    P.tel3e_from     = '';                % the t3o record (in P.outdir), e.g. 'dyson5_t3o_3k.mat'
    P.tel3e_fold_gap_m = 0;               % > 0: a flat fold this far before the slit (packaging), 0 = none
    P.tel3e_suffix   = '';                % record dyson5_t3e<suffix>.*
    % t5e, END TO END FROM A TELESCOPE DECK (TMA step 2, CC): a coaxial
    % Telescope-emitted deck with an object-space ApStop (tel_deck_geom),
    % its FocalPlane replaced by the Dyson's slit (P.tel_dyson); the strip the
    % slit admits is scored, the plate scale and slit vignetting stated.
    P.tel5e_deck     = '';                % the deck (path; relative = this folder), e.g. 'dyson5_tma_step2b_linux_d205_1k5.in'
    P.tel5e_suffix   = '';                % record dyson5_t5e<suffix>.*
    P.tel5e_roll_deg = 0;                 % the telescope rolled about the exit chief in the join (180: bodies to the other side of the Dyson)
    % tA, TMA STAGE A (BRIEF_dyson5_tma.md, stage A hand-off): the eccentric
    % section of the TELECENTRIC Korsch parent (tma_layout 'telecentric', M1
    % stop), bias x decenter scanned AS IS (no figure), the three first-order
    % numbers read first -- chief spread / pupil distance / traced plate scale
    % at the working bias -- clearance by Telescope.check_clipping; then the
    % parent EFL calibrated so the SECTION's local plate scale is the spec
    % (pixel/IFOV), the step-2 conic ladder on the pick, the deck through t5e.
    P.tA_strip_half_deg = NaN;            % cross-track half strip (NaN = the module's: tel_npix_xt x IFOV / 2)
    P.tA_bias_deg    = [-8 -6 -5 -4 -3 -2];  % along-track field bias scanned (deg; negative = away from the decenter)
    P.tA_dec_m       = [0.08 0.10 0.12 0.14 0.16];  % eccentric pupil decenter scanned (m; Telescope.set_offaxis('none','dist',d))
    P.tA_nfield      = 7;                 % strip fields for the three numbers and the spot
    P.tA_plate_tol   = 0.005;             % plate-scale calibration tolerance (relative)
    P.tA_cal_iters   = 6;                 % parent-EFL calibration passes
    P.tA_cal_solved  = false;             % true: the plate scale is read on the STRIP-SOLVED section inside the calibration loop (the conics re-power the sub-pupil)
    P.tA_rungs       = [0 2];             % ladder on the pick: 0 as-is, 1 inner half-strip conics, 2 full-strip conics, 3 STAGE B (aspheres + per-field position rows, one row per tA_beam_wt)
    P.tA_beam_wt     = [1e-2 1e-1 1];     % stage B: weights of the position rows (f*tan(theta) per strip field -- plate scale AND distortion) against the WFE rows
    P.tA_asph_terms  = [1 2];             % stage B: even-radial terms on M1-M3 (1 = h^4, 2 = h^6, 3 = h^8)
    P.tA_max_iters   = 150;               % CALIB iterations per conic rung
    P.tA_pick        = [];                % [bias_deg dec_m] to force the working point (else: clear, then plate nearest the spec, then chief spread)
    P.tA_suffix      = '';                % record dyson5_tA<suffix>.*
    P.tA_t5e         = true;              % score the stage's deck end to end (t5e, P.tel_dyson) at the end
    % tEP, ADDENDUM 42: the strict (reference-sphere) metric vs the FP-OPD metric on one section, per field
    P.tEP_bias_deg   = -4;  P.tEP_dec_m = 0.19;   % the section (the stage-B row of 50ed36b)
    P.tEP_fsys       = NaN;               % parent F/# (NaN = read from dyson5_tA_B3_1k5_b4d190.mat)
    P.tEP_R_m        = 1.0;               % reference-sphere radius about each field's best-focus point (m; virtual pupil allowed)
    P.tEP_rungs      = {'seed', 'B1'};    % the seed (as-is conics) and the stage-B rung re-solved with the FP merit
    P.tEP_suffix     = '';                % record dyson5_tA_EP<suffix>.*
    % tPZ, ADDENDUM 43: the Petzval scan over the parent's secondary_mag at the working point
    P.tPZ_m2         = 2.5:0.5:8;         % secondary_mag values scanned
    P.tPZ_bias_deg   = -4;  P.tPZ_dec_m = 0.19;
    P.tPZ_suffix     = '';                % record dyson5_tA_pz<suffix>.*
    P.tEP_maxfev     = 400;               % S1 (strict-merit lsqnonlin) function-evaluation budget
    P.tEP_from       = 'seed';            % S1 start: 'seed' (as-is conics, zero aspheres) | 'B1' (the FP-merit solution of the same run)
    P.tEP_center     = 'chief';           % S1 reference-sphere centre per field: 'chief' (detector intercept) | 'focus' (best focus)
    % t3s, THE FIRST-ORDER CLEARANCE SCREEN (addendum 21): tma_screen's nine
    % OI_CLEAR pairs evaluated paraxially (engine-free, ms per row) over
    % telescope_seed's family -- solve ONLY rows it passes
    P.tel3s_t1_m     = [0.14 0.20 0.25 0.30:0.05:0.60];   % M1 -> M2 (stop) spacing
    P.tel3s_y2       = 0.3:0.1:0.9;                       % beam compression at M2 (sets t2 = f y2 and the back focus)
    P.tel3s_off_deg  = 8:2:30;                            % along-track offset of the strip
    P.tel3s_off_max  = 15;                                % addendum 21's offset ceiling for "packages"
    P.tel3s_validate = [0.14 0; 0.14 8; 0.14 10; 0.30 0; 0.30 15];   % [t1 off] seeds checked screen vs engine oi_clear (y2 = P.tel_y2)
    P.tel3s_suffix   = '';                            % record name suffix (dyson5_t3s<suffix>.*)
    P.stages   = {'s0','s1','s2'};        % 's3' (ladder), 's4' (native, blocked), 's5' (fold prism), 's2w' (twin, model 512), 's2l', 't1' (telescope), 't2' (end to end), 't3' (offset_imager ladder), 't3s' (first-order screen), 't3w' (y2 continuation), 't3o' (offset solve), 't3e' (end to end from t3o), 't4' (two-mirror Schwarzschild), 't5e' (end to end from a telescope deck), 'tA' (TMA stage A: telecentric section), 'tEP' (addendum 42: strict vs FP merit), 'tPZ' (addendum 43: Petzval scan) are opt-in

    f = fieldnames(over);
    for k = 1:numel(f)
        if ~isfield(P, f{k})
            error('dyson5_params:unknown', 'unknown parameter "%s"', f{k});
        end
        P.(f{k}) = over.(f{k});
    end
end
