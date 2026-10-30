function S = rodgers_seq()
%RODGERS_SEQ  The CODE V .seq TRUTH for the Rodgers offset-field TMA study.
%
%   Every number below is transcribed VERBATIM from the four sequence files
%   J.M. Rodgers supplied on 2026-07-31, in
%   ~/dev/MACOS_sandbox/Design/Rodgers/ :
%
%     Coaxial_OnAxisFOV.seq                    -> stage 1
%     Coaxial_05degOffsetFOV.seq               -> stage 2
%     Coaxial_05degOffsetFOV_NewConics.seq     -> stage 3
%     TiltDecM2M3_05degOffsetFOV_NewConics.seq -> stage 4
%
%   (all four: "VERSION: 2025.03  LENS VERSION: 92  Creation Date:
%   31-Jul-2026").  This file supersedes the slide transcriptions in
%   RODGERS_COMMON for anything it covers -- see the reconciliation table in
%   PACKET.md Addendum 8.  RODGERS_COMMON is left untouched so the committed
%   EPD-4060 baselines reproduce bit-for-bit.
%
%   CODE V SYNTAX DECODED (only what these files use):
%     DIM M            lens units = MILLIMETRES
%     RDM              radius mode: the first number on an S line is the
%                      signed vertex RADIUS, not the curvature
%     EPD              entrance pupil diameter (lens units)
%     WL / REF / WTW   wavelength (nm) / reference index / weight
%     XAN, YAN         field angles, DEGREES, object space
%     S r t [REFL]     surface: radius, thickness to the NEXT surface
%     SO / SI          object / image surface
%     STO              this surface is the aperture stop
%     SLB "..."        surface label
%     CON ; K          conic surface ; conic constant (same sign as MACOS Kc)
%     CIR HOL r        CIRcular HOLe of SEMI-diameter r (a central hole)
%     CUY UMY u        curvature-Y held by a SOLVE on the marginal ray angle
%     PIM              paraxial image distance solve
%     DAR              DecenterAndReturn: perturb this surface only; the
%                      axis reverts for the next one (= a MACOS element
%                      perturbation)
%     XDE/YDE/ZDE      decenters (lens units); ADE/BDE/CDE tilts (deg) about
%                      the local x/y/z, decenter applied first
%     <x>C n           the variable/constraint flag on datum <x>:
%                      0 = VARIABLE in the optimisation, 100 = held.
%                      Hence KC 0 / CCY 0 / THC 0 / ADC 0 / YDC 0 mark the
%                      conic / curvature / thickness / alpha-tilt / y-decenter
%                      as design variables, and BDC 100 / CDC 100 / XDC 100 /
%                      ZDC 100 mark the rest as held.
%
%   UNDECODED, AND CARRIES NO PRESCRIPTION CONTENT: every reflective surface
%   in all four files repeats, byte-identically,
%       CUM 0.0; THM 256.9711383237419
%   The S lines are a complete, self-consistent prescription without it.  THM
%   is exactly the design's length scale: the object thickness is 1e9 x THM,
%   the M1 hole semi-diameter is 2 x THM and the first dummy thickness is
%   22 x THM, all to the last written digit -- i.e. the design was built in a
%   normalised unit system and SCALEd by 256.9711383237419.  Flagged for Mike
%   rather than guessed at.
%
%   Returns S with:
%     .EPD_mm .lambda_nm .dim .obj_dist_mm
%     .ROC_mm .K(4,3) .s12_mm .s23_mm .s3f_mm .dummy_before_M1_mm
%     .M1_hole_semi_mm .stop_at
%     .umy_solve .img_ADE_deg(4) .img_defocus_mm(4)
%     .rb  (stage-4 M2/M3 rigid body, VERBATIM CODE V sign)
%     .Frel_deg (15 x 2) .Frel (15 x 2 rad)  -- the field set, BOX-RELATIVE
%     .YAN_abs(4)  the absolute YAN box centre per stage
%     .vars  per-stage list of the CODE V variables

    S = struct();
    S.src_dir     = '~/dev/MACOS_sandbox/Design/Rodgers';
    S.files       = {'Coaxial_OnAxisFOV.seq', ...
                     'Coaxial_05degOffsetFOV.seq', ...
                     'Coaxial_05degOffsetFOV_NewConics.seq', ...
                     'TiltDecM2M3_05degOffsetFOV_NewConics.seq'};

    % ---- system, identical in all four files ---------------------------
    S.dim          = 'mm';                    % DIM M
    S.EPD_mm       = 5000.0;                  % EPD 5000.0
    S.lambda_nm    = 1000.0;                  % WL 1000.0 (REF 1, WTW 1)
    S.obj_dist_mm  = 256971138323.7418;       % SO thickness (see note below)
    S.stop_at      = 'M1';                    % STO surface is coincident with M1

    % ---- layout (S lines, signed CODE V) --------------------------------
    S.dummy_before_M1_mm = 5653.36504312232;  % S 0.0 5653.365... -> "tilt" -> STO
    S.ROC_mm  = [-12357.51781954069, -2201.366731591871, -2687.973160418888];
    S.s12_mm  = -5267.903548419933;           % M1 -> M2
    S.s23_mm  =  7106.92433775584;            % M2 -> M3
    S.s3f_mm  = -5095.367898318288;           % "recenter" (at M3) -> image, PIM

    S.M1_hole_semi_mm = 513.9422766474837;    % CIR HOL on M1 (SEMI-diameter)

    % ---- conics, per stage (rows 1..4) ----------------------------------
    S.K = [ -0.9929244714356076, -1.926467376849899, -0.7072161814228599   % S1
            -0.9929244714356076, -1.926467376849899, -0.7072161814228599   % S2 (held)
            -0.9933526733207213, -1.932084692329995, -0.7024948533063915   % S3
            -0.9936265324358577, -1.93518426306974,  -0.7059332564340962]; % S4

    % ---- solves ---------------------------------------------------------
    % M3: CUY UMY -0.025 -- M3's curvature is NOT a free number, it is held by
    % a paraxial solve on the final marginal ray angle u' = -0.025.  With the
    % stop at M1 and EPD 5000 that pins EFL = (5000/2)/0.025 = 100 000 mm
    % exactly, i.e. f/20.  The RADIUS printed on the M3 line is the solved
    % value at THIS EPD; at any other EPD the solve would move it.
    S.umy_solve  = -0.025;
    S.EFL_mm     = (S.EPD_mm/2) / abs(S.umy_solve);
    S.Fno        = S.EFL_mm / S.EPD_mm;

    % ---- image surface, per stage ---------------------------------------
    % THC 0 -> the image defocus is a VARIABLE in every stage.
    % ADC 0 -> the image alpha-tilt is a VARIABLE in stages 2,3,4 (absent in
    %          stage 1, which is on-axis and needs none).
    % NOTE the sign convention: these are CODE V ADE, VERBATIM.  Our reporting
    % frame is rigid_of's alpha = atan2d(psi_y, -psi_z), which
    % convention_decode.m measured to be the OPPOSITE sense (30x margin, 16
    % combinations).  Use img_ADE_ours() below to get our-frame values.
    S.img_ADE_deg     = [ NaN, -0.07353267154497573, 0.2399974965374732, 4.440677563287398 ];
    S.img_defocus_mm  = [ 0.02133007044788927, 0.8312083952697579, -0.6316458675136252, -0.8317449958110119 ];

    % ---- stage-4 rigid body, VERBATIM CODE V ----------------------------
    %   rows = [element, YDE mm, ADE deg];  XDE = ZDE = 0, BDE = CDE = 0.
    %   Flags: XDC 100, YDC 0, ZDC 100 / ADC 0, BDC 100, CDC 100 -> YDE and
    %   ADE are the variables, everything else held.  DAR = decenter-and-
    %   return, i.e. a pure single-element perturbation.
    S.rb = [ 2,   8.344662391469104, 0.5169423948334376
             3, 121.8674206663581,   2.329691666359868 ];

    % ---- the field set --------------------------------------------------
    % XAN/YAN, 15 points, IDENTICAL relative pattern in all four files.
    % It is a HALF box: XAN >= 0 only (the design is symmetric about the y-z
    % plane -- every decenter is YDE and every tilt is ADE), sampled on a
    % quincunx: 3 y-points at XAN = 0, 0.05, 0.1 and 2 at XAN = 0.025, 0.075,
    % plus 2 extra interior y-points on the XAN = 0 column.
    xan = [0 0 0 0 0 0.025 0.025 0.05 0.05 0.05 0.075 0.075 0.1 0.1 0.1].';
    yrel= [-0.1 0.0 0.1 -0.05 0.05 -0.05 0.05 -0.1 0.0 0.1 ...
           -0.05 0.05 -0.1 0.0 0.1].';
    S.Frel_deg = [xan, yrel];
    S.Frel     = deg2rad(S.Frel_deg);
    S.YAN_abs  = [0.0, 0.5, 0.5, 0.5];       % box centre per stage (deg)
    S.offset_deg = 0.5;
    S.fov_half_deg = 0.1;

    % ---- what CODE V actually varied, per stage -------------------------
    S.vars = { {'K_M1','K_M2','K_M3','CU_M2 (CCY 0)','image defocus (THC 0)'}
               {'image defocus (THC 0)','image ADE (ADC 0)'}
               {'K_M1','K_M2','K_M3','image defocus (THC 0)','image ADE (ADC 0)'}
               {'K_M1','K_M2','K_M3','M2 YDE','M2 ADE','M3 YDE','M3 ADE', ...
                'image defocus (THC 0)','image ADE (ADC 0)'} };

    % ---- derived notes ---------------------------------------------------
    % Object distance.  SO's thickness is 2.5697e11 mm = 2.5697e8 m.  That is
    % CODE V's large-number stand-in for infinity, scaled with the design
    % (= 1e9 x THM).  Departure from a true collimated source, expressed as a
    % focus shift, is EFL^2 / L:
    S.obj_focus_shift_mm = S.EFL_mm^2 / S.obj_dist_mm;   % ~3.9e-2 mm
    % ... which is smaller than the image defocus CODE V solved for in three
    % of the four stages, and the image defocus is a FREE VARIABLE in all
    % four -- so a collimated source is exact for this study, not an
    % approximation.  Quantified rather than assumed.
end
