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
