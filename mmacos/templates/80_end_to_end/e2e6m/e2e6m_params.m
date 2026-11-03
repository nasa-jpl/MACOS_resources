function P = e2e6m_params(over)
%E2E6M_PARAMS  Single source of truth for the e2e6m end-to-end example.
%
%   The Keysight end-to-end use case: design a 6 m UNOBSCURED telescope,
%   segment its primary, hang an imager and an APLC coronagraph off it
%   with Bench, harvest sensitivities, and run a random time series.
%   Every stage runner reads ONLY this struct (the e2e2 pattern), so a
%   knob change re-runs from the first stage it affects.
%
%   P = E2E6M_PARAMS() returns the defaults; P = E2E6M_PARAMS(OVER)
%   applies the fields of OVER on top (unknown names error).
%
%   ---- the three hard numbers (Dave, 2026-08-24) ----------------------
%     lambda   500 nm -- the diffraction-limit claim is at 500
%     OTA f/#  in [12, 20]
%     shroud   8 m DIAMETER, deployed, length free and stated; measured
%              by packaging_report as the radial extent of every body and
%              beam about the incoming-beam axis.  The entry corridor is
%              reported SEPARATELY as the sunshade keep-out -- it is not
%              counted against the diameter.
%
%   ---- the four aperture rules (Dave, 2026-08-24) ---------------------
%   realize_apertures has an open frame defect (footprint centres
%   measured in global XY, emitted as local ApVec), so a saved
%   tilted-fold .in loses every ray on reload.  Therefore:
%     1  design apertures-off;
%     2  apertures enter only via the S2 segmentation machinery (the PM)
%        and aperture_full_field (the rest of the train);
%     3  every save() is gated by a reload-ray-count check;
%     4  if aperture_full_field shares the defect, the proper
%        ray_bundle-based frame fix is in scope, not a carried stopgap.
%
%   See also S1_TELESCOPE, S1_LAYOUT_SEARCH, FREEFORM_UNOBSCURED.

    arguments
        over struct = struct()
    end

    % ---- identity -------------------------------------------------------
    P.name        = 'e2e6m';
    P.D_m         = 6.0;
    P.lambda_m    = 500e-9;
    P.model       = 256;
    P.gridn       = 41;            % circular ray grid (~1300 rays)

    % ---- gates ----------------------------------------------------------
    P.fno_band    = [12 20];
    P.shroud_D_m  = 8.0;
    P.aoi_limit   = 15;            % deg spread per mirror (coronagraph
                                   % polarization preference; reported,
                                   % not a hard gate)
    P.dl_waves    = 0.071;         % Marechal

    % ---- S1: the 0th-order unobscured layout ----------------------------
    % Picked by s1_layout_search against the three gates above; see
    % s1_layout_search.txt for the sweep and e2e6m_LOG.md for why this
    % topology (sphere+Zernike tilted folds) and not the field-offset one.
    P.tel.R_m     = [38.400, 3.800, 2.000];   % M1 concave, M2 CONVEX, M3
                                              % R3 = 2.0 is the DESIGN
                                              % POINT (2026-08-24): the
                                              % layout search picks 3.5 on
                                              % the base spheres, but the
                                              % freeform stage spends
                                              % power, so the design point
                                              % is chosen on the CORRECTED
                                              % system.  See the LOG's
                                              % trade table -- and note
                                              % the corrected f/# is
                                              % DISCONTINUOUS in R3
                                              % (1.4000 -> f/15.5,
                                              % 1.4050 -> f/25.7), so this
                                              % is a measured pick, not a
                                              % point on a smooth curve.
    P.tel.T_m     = [17.500, 14.000];         % M1->M2, M2->M3 along the
                                              % folded chief
    P.tel.tilt_deg = [-5.60, 5.90, 8.00];     % fold about x
    P.tel.convex  = [false true false];

    % ---- S1: the freeform correction ------------------------------------
    P.tel.fov_arcmin = 0.35;       % design HALF-field (a 0.7' = 42"
                                   % box).  Well inside the brief's
                                   % "<= 0.1 deg", and generous for the
                                   % use case: the coronagraph science
                                   % field is a few lambda/D, which at
                                   % f/25 and 500 nm is ARCSECONDS.  The
                                   % visible DL bar is 35 nm RMS and the
                                   % field is where the fight is; +-1'
                                   % was tried and costs a factor of a
                                   % few in the residual (LOG).
    P.tel.modes   = [3 4 5 9 10 11 12 13 19 20 21 22 23 24 25];
    P.tel.ztype   = 'BornWolf';
    P.tel.iters   = 200;
    P.tel.lmon_mode   = 'auto';    % Zernike normalization radius:
                                   % 'body' = set_freeform's default (the
                                   % element ap_r) | 'auto' = the measured
                                   % full-field footprint radius about the
                                   % element's Mon origin, times the margin
    P.tel.lmon_margin = 1.15;
    P.tel.conic_stage = false;     % solve the mirror CONICS before the
                                   % Zernike departures.  Conics are
                                   % first-order neutral, so they buy
                                   % aberration without spending EFL --
                                   % and they keep the freeform
                                   % departures small enough that the
                                   % sphere+Zernike doctrine's premise
                                   % (departures do not move the chief
                                   % ray) actually holds.  MEASURED
                                   % 2026-08-24: on this train it is
                                   % first-order neutral (EFL 81.311 m
                                   % across the stage, exactly the base
                                   % value) and takes the field worst
                                   % from 8793 to 1313 waves -- but it
                                   % drives K to [-11 -123 +215] and the
                                   % Zernike stage then lands 16 waves
                                   % against 0.066 without it.  Off by
                                   % default; kept as a knob because the
                                   % neutrality result is worth having.
    P.tel.conic_iters = 120;
    P.tel.map_n   = 7;             % dense WFE-vs-field map (n x n)
    P.tel.fp_grid = 5;             % align_focal_plane field-foci grid

    % ---- S2: segmentation -----------------------------------------------
    P.seg.kind    = 'hex';
    P.seg.rings   = 2;             % 19 segments, ~1.2 m flat-to-flat
    P.seg.gap_m   = 0.025;
    P.seg.ng      = 256;           % per-segment grid size

    % ---- S3: the back end (Bench, in METRES so it splices) --------------
    % A 4-OAP coronagraph relay off the telescope focus: collimate to an
    % accessible pupil (the apodizer site), focus to the FPM, re-collimate
    % to the Lyot pupil, focus to the science detector.  Near-normal folds
    % (small AOI) keep each section barely off-axis -- minimum off-axis
    % astigmatism, and they keep the back end inside the annulus the
    % telescope already fills.
    P.bk.tag      = 'seg';         % artifact-name suffix, so the SEGMENTED
                                   % and MONOLITHIC trains can both exist
                                   % (the gap cost is the difference)
    P.bk.base_in  = 's2_segmented.in';  % deck the back end splices onto
                                   % (the SEGMENTED telescope; falls back
                                   % to s1_telescope.in when S2 has not
                                   % run)
    P.bk.fno_in   = NaN;           % f/# feeding the back end
                                   % (NaN = read the measured value out
                                   % of s1_run.mat, which is the CORRECTED
                                   % f/#, not the base-sphere one)
    P.bk.back_m   = 0.30;          % start the bench this far BEFORE the
                                   % telescope focus (add_oap's collimate
                                   % mode wants the incoming chief
                                   % diverging from a focus one conjugate
                                   % back)
    P.bk.f_oap1   = 1.20;          % collimator: sets the pupil diameter
                                   % = f_oap1 / fno_in
    P.bk.d_apod   = 0.25;          % OAP1 -> apodizer pupil
    P.bk.d_oap2   = 0.25;          % apodizer -> OAP2
    P.bk.f_oap2   = 0.90;          % OAP2 focal length (-> FPM focus)
    P.bk.f_oap3   = 0.90;          % FPM -> OAP3 = its focal length
    P.bk.d_lyot   = 0.25;          % OAP3 -> Lyot pupil
    P.bk.d_oap4   = 0.25;          % Lyot -> OAP4
    P.bk.f_oap4   = 0.90;          % OAP4 -> science focus
    P.bk.aoi_deg  = [6 6 6 6];     % per-OAP angle of incidence
    P.bk.drop_tail = 3;            % telescope elements dropped at the
                                   % splice: its terminal quartet
                                   % (FP_return / ExitPupil / FP) -- the
                                   % back end re-images that focus, and a
                                   % FocalPlane mid-train terminates it

    % ---- S3b: the coronagraph -------------------------------------------
    P.co.model      = 1024;        % >= ngridpts, and the grid must span
                                   % ~4 beam diameters for 4 samples per
                                   % lambda/D at the focal mask (samples
                                   % per lambda/D = model*dx / D_beam,
                                   % independent of the sphere radius)
    P.co.ngridpts   = 255;
    P.co.r_occ_lamD = 2.8;         % hard occulter radius (Soummer 2011
                                   % GPI: 5.6 lambda/D diameter)
    P.co.r_lyot_frac= 0.90;        % APLC uses a near-full Lyot -- the
                                   % apodizer, not the Lyot, suppresses
    P.co.prolate_iter = 5000;      % power-iteration cap for the prolate
                                   % apodizer.  The ctb default of 200 is
                                   % NOT enough here: at 200 the solver
                                   % reports Lambda0 = 1.0017, above the
                                   % eigenvalue's physical bound of 1, and
                                   % flags itself unconverged.  This pupil
                                   % converges at 2387 (Lambda0 0.999994).
    % ---- S3c: the imager leg (the demo's second instrument) -----------
    % A DEPLOYABLE PICK-OFF at the shared collimated pupil, not a
    % beamsplitter: a permanent BS would put two transmitting surfaces in
    % the coronagraph deck, which would invalidate the S4 sensitivities
    % and the S5 series already built on it.  Two configurations of one
    % observatory, both counted in the shroud gate.
    P.im.f_cam      = 0.90;     % camera focal length, m -> f/19 on the
                                % 47 mm shared pupil, lambda/D ~ 9.5 um,
                                % Nyquist on a 5 um pixel
    P.im.aoi_deg    = 6;        % pick-off and camera fold AOI (matches
                                % the coronagraph leg's 6 deg)
    P.im.d_pick     = 0.15;     % shared pupil -> pick-off
    P.im.d_cam      = 0.25;     % pick-off -> camera OAP
    P.im.strehl_min = 0.80;     % image-quality gate
    P.im.s1_wfe_ref = 0.0473;   % S1's telescope-only record at the
                                % TELESCOPE best-focus XP, reprinted for
                                % reference -- a DIFFERENT anchor from
                                % this stage's imager-leg exit pupil

    % ---- S3b: the LP apodizer (Carlotti/Vanderbei/Kasdin) -------------
    % Targets are a LADDER, not a single number: the LP is always
    % feasible (A=0 is), so a pupil that cannot reach a target does not
    % fail -- its throughput collapses.  The ladder is how you find out.
    P.ap.targets     = [1e-5 3e-6 1e-6];   % the band the design
                                % model can actually support; deeper
                                % targets are optimizing its error
                                % (measured, see the report)
    P.ap.thru_floor  = 0.05;    % pick the deepest target still above this
    P.ap.nvar_target = 2500;    % block-constant tiles; the OPERATOR keeps
                                % the pupil at full resolution (the 25 mm
                                % gaps are 1.06 px at model 1024 and are
                                % erased by any coarser pupil grid)
    P.ap.dz_per_lamD = 2.0;     % dark-zone samples per lambda/D
    P.ap.n_fpm       = 48;      % occulter-grid samples across its diameter
    P.ap.onesided    = false;   % annular zone; the escalation if the LP
                                % gets tight is Por's D-shaped zone
    P.ap.gate_factor = 3.0;     % model-vs-engine agreement bar (gate 1).
                                % MEASURED at ~5x on this train -- the gate
                                % FAILS, deliberately recorded rather than
                                % relaxed; see s3b_report [5] and the LOG.
    P.ap.verify_dense = 2;      % re-score the LP solution on a 2x finer
                                % dark-zone grid: an LP bounds the field
                                % only AT ITS SAMPLES
    P.ap.prolate_iter = 6000;   % power-iteration cap for the aperture-
                                % specific prolate (the segmented support
                                % converges near 1300; the circular one on
                                % this pupil needs 2392)
    P.ap.pupil_phase_rms = 0.108;  % rad, MEASURED at the apodizer plane --
                                % quoted in the report so the reason the
                                % operator uses the complex field is on the
                                % page next to the number

    P.co.inner_lamD = 3.0;
    P.co.outer_lamD = 15.0;

    % ---- S4: sensitivities ----------------------------------------------
    P.sn.rx          = 's3_seg_prop.in';  % the FULL train (segmented
                                   % primary + coronagraph back end), and
                                   % the same deck S5 propagates
    P.sn.zmodes_fig  = 4:11;       % dwdz segment MonZernike modes
    P.sn.zmodes_grid = 4:9;        % dwdgrid influence-basis modes
    P.sn.ng          = 256;        % per-segment grid size
    P.sn.model       = 512;

    % ---- S5: the drift time series --------------------------------------
    % MET IS NOT IN SCOPE (see s5_timeseries' header): the control is
    % IMAGE-BASED, so the corrected leg is an optimistic bound.
    P.ts.seed        = 6;
    P.ts.frames      = 41;         % history length
    P.ts.dt          = 10;         % s per frame -> a ~400 s soak
    P.ts.every       = 5;          % score CONTRAST every Nth frame: each
                                   % point is a full diffraction
                                   % propagation plus the APLC chain
    P.ts.walk_trans  = 0.3e-9;     % random walk, m per step
    P.ts.walk_rot    = 0.3e-9;     % rad per step
    P.ts.drift_trans = 6e-9;       % correlated drift, m per 100 s
    P.ts.drift_rot   = 6e-9;       % rad per 100 s
    P.ts.control_elts = 1:19;      % the segments are the control bodies
    P.ts.control_dofs = 5;         % 0-based: 5 = Tz, segment PISTON.
                                   % The only DOF the engine-vs-model
                                   % check reproduces (0.03% over three
                                   % decades); the rotation columns are
                                   % an OPEN discrepancy, see the LOG.
                                   % Piston is also the physical control
                                   % DOF for a segmented primary --
                                   % phasing -- so this is a real demo,
                                   % not a fallback.
    P.ts.wfc_frame   = 2;          % "no system starts perfect": control
                                   % turns on at frame 2, so the first
                                   % points stand at the as-deployed state
    P.ts.wfc_iters   = 3;
    P.ts.ridge       = 1e-6;       % Tikhonov ridge, relative to |A|^2
    P.ts.n_check     = 6;          % engine-vs-model sample DOFs
    P.ts.d_trans     = 1e-9;
    P.ts.d_rot       = 1e-9;
    P.ts.tol_linear  = 0.05;       % engine vs linear model, relative

    % ---- output ---------------------------------------------------------
    P.outdir      = '';            % '' = the example directory

    % ---- overrides -------------------------------------------------------
    fn = fieldnames(over);
    for i = 1:numel(fn)
        if ~isfield(P, fn{i})
            error('e2e6m_params:unknown','unknown parameter "%s"', fn{i});
        end
        P.(fn{i}) = over.(fn{i});
    end
end
