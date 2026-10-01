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
    P.native_asph = false;                % the block's h^4/h^6 as CALIB DOFs -- OFF until the OptAsph slice fix (CC)
    P.native_enabled = false;             % s4 refuses to run until CALIB's SPOT derivative stride is fixed (beat 4c 3.4, CC)
    % s5, R5's fold prism (addendum 10): entrance plate + mirror-coated fold
    % prism cemented to the block, the FPA folded away from the slit; the
    % COLD-SHIELD HEIGHT is the parameter -- each height needs an air gap of
    % height + clearance between the prism's exit face and the FPA, and the
    % design is re-solved (R5 rung) at each; the record is at fold_shield_m
    P.fold_h_m = 8e-3;                    % fold plane depth below the face (the beam half-height there is ~7 mm)
    P.fold_slit_gap_m = 0.5e-3;           % slit in air before the entrance plate
    P.fold_face_offset_m = 17e-3;         % seed face offset (plate thickness + slit gap); the R5 rung solves it
    P.fold_shield_sweep_m = [0 2e-3 5e-3 10e-3];   % cold-shield heights swept
    P.fold_shield_m = 2e-3;               % the shield height of record
    P.fold_shield_clear_m = 1e-3;         % air beyond the shield to the prism's exit face
    P.fold_max_iter = 30;                 % lsqnonlin iterations per sweep point (warm-started along the sweep)
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
    P.stages   = {'s0','s1','s2'};        % 's3' (ladder), 's4' (native, blocked), 's5' (fold prism), 's2w' (twin, model 512), 's2l' are opt-in

    f = fieldnames(over);
    for k = 1:numel(f)
        if ~isfield(P, f{k})
            error('dyson5_params:unknown', 'unknown parameter "%s"', f{k});
        end
        P.(f{k}) = over.(f{k});
    end
end
