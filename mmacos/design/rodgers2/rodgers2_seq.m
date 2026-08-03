function S = rodgers2_seq()
%RODGERS2_SEQ  The CODE V .seq TRUTH for the Rodgers2 30x afocal TMA study.
%
%   Every number below is transcribed VERBATIM from the four sequence files
%   J.M. Rodgers supplied on 2026-08-02, in
%   ~/dev/MACOS_sandbox/Design/Rodgers2/ :
%
%     CentAfo_Coaxial_OnAxisFOV.seq                      -> variant 1 (S1)
%     CentAfo_Coaxial_06degOffsetFOV.seq                 -> variant 2 (S2)
%     CentAfo_Coaxial_06degOffsetFOV_NewConics.seq       -> variant 3 (S3)
%     CentAfo_TiltDecM2M3_06degOffsetFOV_NewConics.seq   -> variant 4 (S4)
%
%   (all four: "VERSION: 2025.03  LENS VERSION: 92  Creation Date:
%   2-Aug-2026".)  Transcription, not parsing -- no .seq reader exists and
%   none is to be written (PLAN_AFOCAL4 doctrine 8).  RODGERS2_DECK renders
%   this struct to MACOS .in text; PACKET.md section 2 is the element-by-
%   element conversion audit.
%
%   CODE V SYNTAX DECODED (only what these files use; the rodgers1
%   RODGERS_SEQ header covers the shared subset):
%     RDM;LEN          radius mode -- the first number on an S line is the
%                      signed vertex RADIUS
%     DIM M            lens units = MILLIMETRES
%     EPD              entrance-pupil diameter (lens units)
%     WL / REF / WTW   wavelength (nm) / reference index / weight
%     INI 'ORA'        initialisation macro; no prescription content
%     CA APE           clear apertures FROM the aperture definition
%     XAN, YAN         field angles, DEGREES, object space
%     S r t [REFL]     surface: radius, thickness to the NEXT surface
%     SO / SI          object / image surface
%     STO              this surface is the aperture stop
%     SLB "..."        surface label
%     CON ; K          conic surface ; conic constant (same sign as MACOS Kc)
%     CIR HOL r        CIRcular HOLe of SEMI-diameter r (central obscuration)
%     DAR              DecenterAndReturn -- perturb THIS surface only; the
%                      axis reverts for the next one (= a MACOS single-
%                      element perturbation).  WITHOUT DAR the decenter and
%                      tilt PERSIST for every downstream surface (= a
%                      coordinate break); the "recenter" surface is that
%                      case and the "coldstop" surface is the DAR case.
%     XDE/YDE/ZDE      decenters (lens units); ADE/BDE/CDE tilts (deg) about
%                      the local x/y/z, decenter applied first
%     AFI d            AFocal Image: evaluate the afocal output at d
%
%   DECODED HERE, AND NEW SINCE rodgers1: the per-surface
%       CUM <c>; THM <t>
%   pairs are MECHANICAL substrate data -- CUM is the curvature of the
%   mounting/back surface, THM its thickness.  M1 carries CUM -0.0004
%   (= 1/-2500, i.e. the back face is concentric with the optical surface)
%   and THM 75.0; M2 THM 40.136; M3 THM 13.379 -- monotone with the beam
%   footprint, as substrate thicknesses are.  They carry NO prescription
%   content and are not transcribed into the .in.  (rodgers1's files
%   repeated ONE byte-identical THM on every surface and were flagged
%   undecoded; these files resolve what the datum is.)
%
%   PARTIALLY UNDECODED, FLAGGED FOR MIKE: the stop surface carries
%   `CIR EDG 0.1`.  The stop's optical semi-diameter is set by EPD (500 mm),
%   so this is a drawing/edge datum, but the qualifier is not confirmed.
%   No prescription content is assumed from it.
%
%   THE SIGN CONVENTIONS ARE REUSED, NOT RE-MEASURED (rodgers1 Addendum 5,
%   `convention_decode.m`, 16 sign combinations at a 30x margin):
%     his YDE  == our Vpt(2)                          (matches)
%     his ADE  == -(our alpha), alpha = atan2d(psi_y, -psi_z)
%   so a CODE V surface whose local z is R_x(-ADE)*zhat emits in MACOS as
%     psiElt = (0, -sin(ADE), -cos(ADE)),   VptElt = (0, YDE, z).
%   RODGERS2_DECK carries an independent CHECK of that reuse on this
%   design: the "recenter" coordinate break places the coldstop vertex on
%   the exit chief ray only for this sign (the other lands it ~222 mm off
%   a 33 mm beam), which PACKET.md section 2.4 records as witness #5.
%
%   Returns S with:
%     .EPD_mm .lambda_nm .dim .obj_dist_mm .M_afocal .stop_ahead_of_M1_mm
%     .M1_hole_semi_mm .z (station table, mm)
%     .v(4)   per-variant struct array:
%        .name .file .title .YAN_abs_deg
%        .ROC_mm(3) .K(3) .s_img_to_M3_mm
%        .recenter (t_mm, YDE_mm, ADE_deg)
%        .coldstop_ADE_deg
%        .rb  ([] or rows [iElt, YDE_mm, ADE_deg], VERBATIM CODE V sign)
%        .gt_max_nm .gt_avg_nm   his reported in-box RMS WFE
%     .Frel_deg (9x2) .Frel (9x2 rad)   the field set, BOX-RELATIVE
%     .fov_half_deg .offset_deg

    S = struct();
    S.src_dir  = '~/dev/MACOS_sandbox/Design/Rodgers2';
    S.deck     = '260802-AfocalTMA_Offsetfield-jmr.pptx';

    % ---- system, identical in all four files ---------------------------
    S.dim         = 'mm';                 % DIM M
    S.EPD_mm      = 1000.0;               % EPD 1000.0
    S.lambda_nm   = 1000.0;               % WL 1000.0 (REF 1, WTW 1)
    S.obj_dist_mm = 367915496283.038;     % SO thickness -- infinity proxy
    S.M_afocal    = 30.0;                 % design angular magnification
    S.exit_beam_mm = S.EPD_mm / S.M_afocal;   % 33.333 mm nominal

    % ---- the fore-optics train (identical in all four) ------------------
    %   SO  0.0  367915496283.038  AIR
    %   S   0.0  1100.0
    %   S   0.0  0.0               SLB "tilt"   (placeholder; NO decenter
    %                                            in ANY of the four files)
    %   S   0.0  50.0  AIR         STO, CIR EDG 0.1
    %   S  -2500.0 ...  REFL       SLB "m1"
    % Stations are quoted with the M1 VERTEX at z = 0 (the rodgers1
    % convention: vertex z = cumulative sum of thicknesses).
    S.stop_ahead_of_M1_mm = 50.0;
    S.dummy1_to_tilt_mm   = 1100.0;
    S.z = struct('SO',   -(1150.0 + S.obj_dist_mm), ...
                 'dummy1', -1150.0, ...
                 'tilt',   -50.0, ...
                 'STO',    -50.0, ...
                 'M1',       0.0);
    S.M1_ROC_mm       = -2500.0;          % fixed in all four
    S.M1_K            = -1.0;             % parabola, fixed in all four
    S.M1_hole_semi_mm =  130.0;           % CIR HOL 130.0 (SEMI-diameter)
    S.s_M1_M2_mm      = -1049.239293684764;   % M1 thickness, all four
    S.s_M2_thru_mm    =  1049.239293684764;   % M2 thickness -> "thru" at z=0
    S.s_thru_img_mm   =  350.0;               % "thru" thickness (all four)

    % ---- mechanical substrate data (decoded, not transcribed) -----------
    S.CUM = [-0.0004, 0.0, 0.0];
    S.THM = [75.0, 40.13623428344727, 13.37874507904053];

    % ---- the field set --------------------------------------------------
    % XAN/YAN, 9 points, a FULL 3x3 box (unlike rodgers1's 15-point half
    % box).  Identical RELATIVE pattern in all four files; only the YAN
    % centre moves (0.0 on-axis, 0.6 offset).
    %   XAN  0.0 0.0 0.0 0.25 0.25 0.25 -0.25 -0.25 -0.25
    %   YAN  c-.25 c c+.25  (x3)
    xan = [0 0 0 0.25 0.25 0.25 -0.25 -0.25 -0.25].';
    yrel = [-0.25 0 0.25 -0.25 0 0.25 -0.25 0 0.25].';
    S.Frel_deg     = [xan, yrel];
    S.Frel         = deg2rad(S.Frel_deg);
    S.fov_half_deg = 0.25;                 % 0.5 x 0.5 deg box
    S.offset_deg   = 0.6;

    % ---- his reported ladder (deck; max / avg RMS WFE in the used FOV) --
    gt_max_nm = [ 15, 430, 160, 119 ];
    gt_avg_nm = [ 4.0, 154,  93,  48 ];

    % =====================================================================
    %  VARIANT 1 -- CentAfo_Coaxial_OnAxisFOV.seq
    %  TITLE 'Coaxial.  On-axis FOV'
    % =====================================================================
    v(1) = mkvar('S1_onaxis', 'CentAfo_Coaxial_OnAxisFOV.seq', ...
        'Coaxial.  On-axis FOV', 0.0, ...
        [-2500.0, -468.7799802942544, -580.8105879437068], ...
        [-1.0,    -1.782495505768868, -1.001753914266608], ...
        290.4158962406167, ...          % "thru"+350 dummy -> M3
        -344.173, 0.0, 0.0, ...         % recenter: t, YDE, ADE (none here)
        0.0, [], gt_max_nm(1), gt_avg_nm(1));

    % =====================================================================
    %  VARIANT 2 -- CentAfo_Coaxial_06degOffsetFOV.seq
    %  TITLE 'Coaxial.  0.6 deg offset FOV'
    %  Same optics as S1; only the field box and the exit-side pose move.
    % =====================================================================
    v(2) = mkvar('S2_offset', 'CentAfo_Coaxial_06degOffsetFOV.seq', ...
        'Coaxial.  0.6 deg offset FOV', 0.6, ...
        [-2500.0, -468.7799802942544, -580.8105879437068], ...
        [-1.0,    -1.782495505768868, -1.001753914266608], ...
        290.486221424707, ...
        -365.779766, 110.5639166839551, 17.67063712218155, ...
        4.289, [], gt_max_nm(2), gt_avg_nm(2));

    % =====================================================================
    %  VARIANT 3 -- CentAfo_Coaxial_06degOffsetFOV_NewConics.seq
    %  TITLE 'Coaxial.  0.6 deg offset FOV. New conics.'
    %  M2/M3 radii AND M2/M3 conics re-solved; M1 stays the R=-2500
    %  parabola.
    % =====================================================================
    v(3) = mkvar('S3_newconics', 'CentAfo_Coaxial_06degOffsetFOV_NewConics.seq', ...
        'Coaxial.  0.6 deg offset FOV. New conics.', 0.6, ...
        [-2500.0, -468.1589934385201, -558.9832733211452], ...
        [-1.0,    -1.778931782803361, -0.9569802075823867], ...
        290.486221424707, ...
        -351.03178, 110.9415904849413, 18.49984656262884, ...
        3.576783, [], gt_max_nm(3), gt_avg_nm(3));

    % =====================================================================
    %  VARIANT 4 -- CentAfo_TiltDecM2M3_06degOffsetFOV_NewConics.seq
    %  TITLE 'Tilt/dec M2 M3.  0.6 deg offset FOV. New conics.'
    %  M2 and M3 carry DAR YDE + ADE (single-element perturbations).
    %  .rb rows are [iElt, YDE mm, ADE deg] in VERBATIM CODE V sign.
    % =====================================================================
    v(4) = mkvar('S4_tiltdec', 'CentAfo_TiltDecM2M3_06degOffsetFOV_NewConics.seq', ...
        'Tilt/dec M2 M3.  0.6 deg offset FOV. New conics.', 0.6, ...
        [-2500.0, -468.2687654028678, -564.6825868509671], ...
        [-1.0,    -1.777290021452283, -0.9820445490283405], ...
        290.486221424707, ...
        -355.257136, 130.030367037749, 21.99503063542316, ...
        -0.355818, ...
        [2,  1.760802316111543, 0.5114423167490506
         3, 36.8890230448649,   4.023132834823657], ...
        gt_max_nm(4), gt_avg_nm(4));

    S.v = v;

    % ---- derived notes ---------------------------------------------------
    % Object distance.  SO's thickness is 3.679e11 mm = 3.679e8 m -- CODE V's
    % large-number stand-in for infinity.  For an AFOCAL system the relevant
    % departure is an angular one: a source at L subtends the 500 mm pupil
    % semi-diameter at 500/L = 1.36e-9 rad = 2.8e-4 arcsec, which is 1.4e-7
    % of the 0.25 deg half-field.  A collimated MACOS source (zSource=1e22)
    % is exact for this study, not an approximation.  Quantified, not assumed.
    S.obj_subtense_rad = (S.EPD_mm/2) / S.obj_dist_mm;

    % First-order check on the printed layout, all four variants: with
    % f1 = |R1|/2 = 1250 mm and the M1->M2 spacing 1049.239 mm, the marginal
    % ray height at M2 is 500*(1 - 1049.239/1250) = 80.30 mm, and the
    % intermediate image sits 1399 mm past M2 -- i.e. AT the second dummy
    % (z = +350.0), which is why that dummy exists.  The remaining 290.4 mm
    % to M3 is M3's focal length, so M3 recollimates and the exit semi-beam
    % is 16.67 mm => M = 30.  (Verified numerically by RODGERS2_DECK's
    % first-order gate.)
end

% =====================================================================
function v = mkvar(name, file, ttl, yan_abs, ROC, K, s_img_M3, ...
                   rec_t, rec_YDE, rec_ADE, cs_ADE, rb, gtmax, gtavg)
    v = struct('name', name, 'file', file, 'title', ttl, ...
               'YAN_abs_deg', yan_abs, ...
               'ROC_mm', ROC, 'K', K, ...
               's_img_to_M3_mm', s_img_M3, ...
               'recenter', struct('t_mm', rec_t, 'YDE_mm', rec_YDE, ...
                                  'ADE_deg', rec_ADE), ...
               'coldstop_ADE_deg', cs_ADE, ...
               'rb', rb, ...
               'gt_max_nm', gtmax, 'gt_avg_nm', gtavg);
end
