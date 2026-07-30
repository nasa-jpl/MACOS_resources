function P = rodgers_common()
%RODGERS_COMMON  Shared constants for the Rodgers offset-field TMA study.
%   Verbatim prescription + ground-truth field-map statistics transcribed
%   from macos_sandbox/Design/Rodgers/260728-TMA_Offsetfield-jmr.pptx
%   (J.M. Rodgers, ORA/CODE V, 28-Jul-2026).  lambda = 1000 nm.
%
%   Sign/units notes (CODE V -> MACOS):
%     * ROC given SIGNED in CODE V; MACOS emits KrElt = -|R| for every
%       mirror (convex vs concave is geometry, not radius sign).  We pass
%       add_mirror the MAGNITUDE abs(ROC).
%     * conic K identical convention (KcElt = K directly), pass verbatim.
%     * spacings given SIGNED (reduced thickness, sign flips each
%       reflection); the builder folds z with (-1)^k, so we pass the
%       absolute separation magnitudes.
%     * FOV / field offset are in DEGREES; M2/M3 tilts in DEGREES.
%     * Rodgers' field-map stats are in WAVES @ 1000 nm; *1000 -> nm.
%
%   EPD: NOT stated on any slide.  M1 measures ~2003 mm from the slide-1
%   layout drawing (scale bar 590 units = 1000 mm); Dave set EPD = 2000 mm.

    P.lambda_m   = 1000e-9;         % 1 micron
    P.EPD_mm     = 2000;            % entrance pupil diameter (Dave, 2026-07-29)
    P.model_size = 256;

    % --- Slide-1 / Slide-2 nominal conics (his starting design) ----------
    P.ROC_mm = [-12357.51782, -2201.36673, -2687.97316];   % signed CODE V
    P.K_nom  = [-0.992924471436, -1.92646737685, -0.707216181423];

    % --- spacings (signed CODE V reduced thickness) ----------------------
    P.s12_mm   = -5267.903548;      % M1 -> M2
    P.s23_mm   =  7106.924338;      % M2 -> M3
    P.s3f_mm   = -5095.367898;      % M3 -> focus (paraxial)

    % --- field definition ------------------------------------------------
    P.fov_full_deg  = 0.2;          % 0.2 deg x 0.2 deg square FOV
    P.fov_half_deg  = 0.1;
    P.offset_deg    = 0.5;          % offset FOV: +y 0.5 deg

    % --- Rodgers' solved parameters (Stages 3, 4) ------------------------
    % Stage 3: reoptimized conics + FPA tilt/focus (used-box <= 92 nm)
    P.K_s3 = [-0.993352673321, -1.93208469233, -0.702494853307];
    % Stage 4: tilt/dec M2,M3 + reoptimized conics + FPA (used-box <= 40 nm)
    P.K_s4       = [-0.99362653306, -1.935184265266, -0.705933233554];
    P.Ydec_M2_mm = 8.344733;    P.tilt_M2_deg = 0.516947;
    P.Ydec_M3_mm = 121.868248;  P.tilt_M3_deg = 2.329710;

    % --- ground-truth field-map statistics (WAVES @ 1000 nm) -------------
    % each: [min max avg std].  "wide" = survey grid; "box" = 0.2x0.2 used FOV.
    P.gt.s1_onaxis_box = [0.00044626, 0.0014628,  0.00060631, 0.00019359];
    P.gt.s2_wide       = [0.026545,   0.37459,    0.10897,    0.097297 ];
    P.gt.s2_box        = [0.079381,   0.37459,    0.19995,    0.088749 ];
    P.gt.s3_wide       = [0.016808,   0.21069,    0.085412,   0.048906 ];
    P.gt.s3_box        = [0.01783,    0.091617,   0.046379,   0.019586 ];
    P.gt.s4_wide       = [0.011241,   0.27777,    0.11739,    0.090993 ];
    P.gt.s4_box        = [0.01092,    0.039802,   0.022493,   0.0070409];
    P.gt.units = 'waves@1000nm';
end
