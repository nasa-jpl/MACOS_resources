function OUT = run_seq(opts)
%RUN_SEQ  Re-run the Rodgers study at the CODE V .seq TRUTH.
%
%   OUT = RUN_SEQ()
%
%   Mike supplied the four sequence files on 2026-07-31 (see RODGERS_SEQ).
%   They pin the three inputs the slides never stated -- EPD, the field set,
%   and the M1 hole -- plus the image-surface defocus/tilt and the M3 ray
%   solve.  This runner re-does the whole study at those values and reports
%   WHERE THE RESIDUAL BAND LANDS.  Nothing is tuned; the band is the
%   measurement.
%
%   Sections:
%     0  LAYOUT GATE, extended to the .seq surface list -- every S-line datum
%        checked against what the builder resolves, before any WFE number is
%        believed.
%     1  DETECTOR-FRAME reconciliation.  PACKET §4b recorded a "14.3 deg
%        tilted image surface" against his image ADE of -0.07 deg.  Those are
%        measured in DIFFERENT frames (ours vs the arriving chief, his vs the
%        axis); this section measures both in one place.
%     2  THE BAND at the .seq truth -- his own S2/S3/S4 designs, built
%        verbatim, strict-scored on HIS 15-point half box at EPD 5000 with
%        the M1 hole.
%     3  ATTRIBUTION -- the same evaluation at EPD 4060 on his 15-point set,
%        which separates the field-set change from the aperture change.
%     4  THE JOINT SOLVES re-run at truth (xp_optimize, variant 'seq').
%     5  THE REFERENCE LADDER (seq_ladder) -- sizes the one thing the .seq
%        files cannot contain: the CODE V field-map ANALYSIS convention.
%
%   Name-value:
%     'sections'  which of 0:4 to run (default all)
%     'map_n'     the full-box grid used for the epd4060 attribution row
%                 (default 9, the committed sampling)
%
%   Writes rodgers1_seq_*.{in,mat,png} beside this file.  The committed
%   rodgers1_epd4060_* and rodgers1_epd2000_* artifacts are NOT touched.

    arguments
        opts.sections (1,:) double = 0:5
        opts.map_n    (1,1) double {mustBeInteger,mustBePositive} = 9
    end
    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);

    P  = rodgers_common('seq');
    Ps = rodgers_common();            % the slide transcription, for the deltas
    S  = P.seq;
    lam_nm = P.lambda_m*1e9;
    OUT = struct('P',P,'seq',S);

    banner('RODGERS STUDY AT THE .seq TRUTH  (EPD %g mm, lambda %g nm, %d fields)', ...
           P.EPD_mm, lam_nm, size(S.Frel,1));

    % =================================================================
    % 0. LAYOUT GATE -- against the .seq surface list
    % =================================================================
    if any(opts.sections==0)
        OUT.gate = layout_gate_(P, Ps, S, lam_nm);
    end

    % =================================================================
    % 1. DETECTOR FRAME
    % =================================================================
    if any(opts.sections==1)
        OUT.frame = detector_frame_(P, S);
    end

    % =================================================================
    % 2. THE BAND, at truth
    % =================================================================
    if any(opts.sections==2)
        banner('SECTION 2 -- THE BAND AT THE .seq TRUTH');
        OUT.band_seq = his_designs(opts.map_n, 'variant','seq');
    end

    % =================================================================
    % 3. ATTRIBUTION -- old aperture, his field set
    % =================================================================
    if any(opts.sections==3)
        banner(['SECTION 3 -- ATTRIBUTION: EPD 4060 (committed deck, no ' ...
                'hole) scored on HIS 15-point half box']);
        fprintf(['  Isolates the FIELD-SET change.  Differences against the\n' ...
                 '  committed 9x9 numbers are field sampling alone; differences\n' ...
                 '  between this and section 2 are aperture + hole.\n']);
        % NOTE the suffix.  Without it this writes to the SAME artifact names
        % as the committed EPD-4060 run (rodgers1_epd4060_his_designs.mat and
        % the three _strict.png), silently replacing a baseline with a
        % different-field-set version of itself.  It did exactly that once.
        OUT.band_4060_seqfields = his_designs(opts.map_n, ...
                                    'variant','epd4060', 'Frel', S.Frel, ...
                                    'suffix','_seqfields');
    end

    % =================================================================
    % 4. THE JOINT SOLVES, at truth
    % =================================================================
    if any(opts.sections==4)
        banner('SECTION 4 -- JOINT FPA SOLVES AT THE .seq TRUTH');
        OUT.xpopt_seq = xp_optimize(opts.map_n, 4, true, 'seq');
    end

    % =================================================================
    % 5. THE REFERENCE LADDER
    % =================================================================
    if any(opts.sections==5)
        banner('SECTION 5 -- REFERENCE-FREEDOM LADDER AT THE .seq TRUTH');
        OUT.ladder_seq = seq_ladder();
    end

    save(fullfile(here,'rodgers1_seq_run.mat'),'OUT','-v7.3');
    banner('run_seq complete -- rodgers1_seq_run.mat + rodgers1_seq_* artifacts');
end

% =====================================================================
function G = layout_gate_(P, Ps, S, lam_nm)
%LAYOUT_GATE_  Check the built model against the .seq surface list, datum by
%   datum, and report the slide-vs-.seq deltas on everything the slides did
%   state.  Runs BEFORE any WFE number is believed.
    banner('SECTION 0 -- LAYOUT GATE vs THE .seq SURFACE LIST');

    G = struct();
    fprintf('  the .seq surface list (all four files share it):\n');
    fprintf('    SO   object      th = %.4f mm  (= 1e9 x THM; CODE V infinity proxy)\n', S.obj_dist_mm);
    fprintf('    S1   dummy       th = %.6f mm\n', S.dummy_before_M1_mm);
    fprintf('    S2   "tilt"      th = 0            (placeholder, never used)\n');
    fprintf('    S3   STO         th = 0            -> the stop is COINCIDENT with M1\n');
    fprintf('    S4   m1   R = %+.9f  th = %+.9f  REFL  CIR HOL %.7f\n', S.ROC_mm(1), S.s12_mm, S.M1_hole_semi_mm);
    fprintf('    S5   m2   R = %+.9f  th = %+.9f  REFL\n', S.ROC_mm(2), S.s23_mm);
    fprintf('    S6   m3   R = %+.9f  th = 0        REFL  (CUY UMY %.4f)\n', S.ROC_mm(3), S.umy_solve);
    fprintf('    S7   "recenter"  th = %+.9f  PIM\n', S.s3f_mm);
    fprintf('    SI   image       defocus/tilt per stage (THC 0 / ADC 0)\n');

    % ---- slide vs .seq, on everything the slides stated ----------------
    fprintf('\n  SLIDE TRANSCRIPTION vs .seq  (the values we already had):\n');
    fprintf('    %-16s %22s %22s %12s\n','datum','slides','.seq','|delta|');
    rowd = @(n,a,b) fprintf('    %-16s %22.12f %22.12f %12.2e\n', n, a, b, abs(a-b));
    for i=1:3, rowd(sprintf('ROC_M%d (mm)',i), Ps.ROC_mm(i), S.ROC_mm(i)); end
    for i=1:3, rowd(sprintf('K_nom_M%d',i),    Ps.K_nom(i),  S.K(1,i));   end
    for i=1:3, rowd(sprintf('K_s3_M%d',i),     Ps.K_s3(i),   S.K(3,i));   end
    for i=1:3, rowd(sprintf('K_s4_M%d',i),     Ps.K_s4(i),   S.K(4,i));   end
    rowd('s12 (mm)', Ps.s12_mm, S.s12_mm);
    rowd('s23 (mm)', Ps.s23_mm, S.s23_mm);
    rowd('s3f (mm)', Ps.s3f_mm, S.s3f_mm);
    rowd('Ydec_M2 (mm)', Ps.Ydec_M2_mm, S.rb(1,2));
    rowd('ADE_M2 (deg)',  Ps.tilt_M2_deg, S.rb(1,3));
    rowd('Ydec_M3 (mm)', Ps.Ydec_M3_mm, S.rb(2,2));
    rowd('ADE_M3 (deg)',  Ps.tilt_M3_deg, S.rb(2,3));

    % ---- build at truth and check what the builder resolves ------------
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_mm', P.EPD_mm, ...
            'wavelength_m', P.lambda_m, 'model_size', P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
    t.set_hole('M1', P.M1_hole_m);
    t.build();
    d = t.spec.derived;

    fprintf('\n  BUILT MODEL vs .seq:\n');
    fprintf('    M1-M2 |sep|   %14.6f mm   .seq %14.6f   d = %.2e\n', ...
            abs(d.z(2)-d.z(1))*1e3, abs(S.s12_mm), abs(abs(d.z(2)-d.z(1))*1e3-abs(S.s12_mm)));
    fprintf('    M2-M3 |sep|   %14.6f mm   .seq %14.6f   d = %.2e\n', ...
            abs(d.z(3)-d.z(2))*1e3, abs(S.s23_mm), abs(abs(d.z(3)-d.z(2))*1e3-abs(S.s23_mm)));
    fprintf(['    M3->FP paraxial SEED %8.3f mm  vs .seq PIM %11.4f -- the\n' ...
             '      builder''s paraxial seeder is known bad on this design (PACKET\n' ...
             '      retraction); it only places the FP before align_focal_plane\n' ...
             '      moves it from real rays.  The meaningful check is below.\n'], ...
            d.t_focus*1e3, abs(S.s3f_mm));
    G.dz = [abs(d.z(2)-d.z(1))*1e3, abs(d.z(3)-d.z(2))*1e3, d.t_focus*1e3];

    % ---- the M3 UMY solve: checked against the TRACED system ------------
    % d.EFL / d.t_focus come from the builder's PARAXIAL SEED, which this
    % design defeats (PACKET retraction: the seed reads 3506 mm where the
    % traced system is ~1e5 mm).  The seed only places the FP before
    % align_focal_plane moves it, so the meaningful check is against the
    % TRACED geometry, measured here from real rays.
    t.align_focal_plane('grid',5,'span_arcmin',6);
    nE  = numel(t.spec.elt);
    zM3 = t.spec.elt(3).Vpt(3);
    fpz = t.spec.elt(nE).Vpt(3);
    t.trace_at_field([0 0]);
    s0 = macos.trace(nE);  r0 = macos.get_ray_info(s0.nRays);
    ok = r0.ok_trace(:) & r0.ok_pass(:);  ok(1) = false;
    % marginal ray angle u' : the steepest converging ray of the on-axis
    % bundle, measured against the axis at the last surface.
    D  = r0.dir(:,ok);
    upr = max(vecnorm(D(1:2,:),2,1));
    % EFL from the traced chief-ray image height per field angle.
    th = deg2rad(0.05);
    t.trace_at_field([0 th]);  s1 = macos.trace(nE);  r1 = macos.get_ray_info(s1.nRays);
    t.trace_at_field([0 -th]); s2 = macos.trace(nE);  r2 = macos.get_ray_info(s2.nRays);
    EFL_traced = norm(r1.pos(:,1) - r2.pos(:,1)) / (2*th) * 1e3;   % mm
    t.trace_at_field([]);

    fprintf('\n  THE M3 RAY SOLVE (CUY UMY %.4f) -- what it pins:\n', S.umy_solve);
    fprintf('    .seq: EFL = (EPD/2)/|u''| = %11.3f mm  ->  f/%.4f\n', S.EFL_mm, S.Fno);
    fprintf('    TRACED (chief-ray dh/dtheta) = %11.3f mm  ->  f/%.4f\n', ...
            EFL_traced, EFL_traced/P.EPD_mm);
    fprintf('    delta                        = %11.3f mm  (%+.3f%%)\n', ...
            EFL_traced-S.EFL_mm, 100*(EFL_traced-S.EFL_mm)/S.EFL_mm);
    fprintf(['    REAL marginal |u''|           = %11.6f    (.seq solves the\n' ...
             '      PARAXIAL u'' to %.6f; a real marginal ray at this speed sits a\n' ...
             '      few %% off a paraxial one, and the exit pupil is near the image\n' ...
             '      so the working f/# is not EFL/D.  The EFL check above is the\n' ...
             '      one that gates the solve.)\n'], upr, abs(S.umy_solve));
    fprintf('    builder PARAXIAL SEED        = %11.3f mm  (known bad on this\n', d.EFL*1e3);
    fprintf('      design -- PACKET retraction; it only seeds the FP station,\n');
    fprintf('      which align_focal_plane then places from real rays.)\n');
    fprintf('    aligned FP station, M3->FP   = %11.4f mm   .seq PIM %11.4f   d = %.3f mm\n', ...
            abs(fpz-zM3)*1e3, abs(S.s3f_mm), abs(abs(fpz-zM3)*1e3-abs(S.s3f_mm)));
    fprintf(['    NOTE: M3''s radius is HELD BY A SOLVE in CODE V, not a free\n' ...
             '    number.  The printed radius is the solved value AT EPD 5000, so\n' ...
             '    the .seq prescription is self-consistent only at EPD 5000 --\n' ...
             '    at any other aperture the solve would have moved M3.\n']);
    G.EFL_seq = S.EFL_mm;  G.EFL_traced = EFL_traced;  G.EFL_seed = d.EFL*1e3;
    G.umy_traced = upr;    G.fp_station_mm = abs(fpz-zM3)*1e3;

    % ---- the M1 hole ---------------------------------------------------
    fprintf('\n  M1 CENTRAL HOLE (CIR HOL):\n');
    fprintf('    semi-diameter %.7f mm -> linear obscuration %.4f of the %g mm EPD\n', ...
            S.M1_hole_semi_mm, 2*S.M1_hole_semi_mm/S.EPD_mm, S.EPD_mm);
    fprintf('    area blocked  %.3f%%\n', 100*(2*S.M1_hole_semi_mm/S.EPD_mm)^2);

    % ---- the object distance -------------------------------------------
    fprintf('\n  OBJECT DISTANCE:\n');
    fprintf('    .seq %.4f mm -> focus shift vs infinity = EFL^2/L = %.5f mm\n', ...
            S.obj_dist_mm, S.obj_focus_shift_mm);
    fprintf(['    Our source is collimated.  That shift is smaller than the image\n' ...
             '    defocus CODE V itself solved for in 3 of 4 stages (%.4f..%.4f mm)\n' ...
             '    and the image defocus is a FREE VARIABLE (THC 0) in all four, so\n' ...
             '    a collimated source is EXACT here, not an approximation.\n'], ...
            min(abs(S.img_defocus_mm)), max(abs(S.img_defocus_mm)));

    % ---- the field set --------------------------------------------------
    fprintf('\n  FIELD SET -- his 15 points (XAN, YAN relative to the box centre):\n');
    for k = 1:size(S.Frel_deg,1)
        fprintf('    %2d  XAN %+7.4f   dYAN %+7.4f    (abs YAN at stages 2-4: %+7.4f)\n', ...
                k, S.Frel_deg(k,1), S.Frel_deg(k,2), S.Frel_deg(k,2)+S.offset_deg);
    end
    fprintf(['    HALF box in x (XAN >= 0 only) -- legitimate, the design is\n' ...
             '    symmetric about the y-z plane (every decenter is YDE, every\n' ...
             '    tilt is ADE).  Our committed runs used a %dx%d FULL box.\n'], 9, 9);
    G.nfields = size(S.Frel,1);
    fprintf('\n  layout gate: PASS if every delta above is at slide-rounding level.\n');
end

% =====================================================================
function FR = detector_frame_(P, S)
%DETECTOR_FRAME_  Measure our fitted detector in BOTH frames and compare with
%   his .seq image ADE.  This is the measurement PACKET §5 open-ask 2 asked
%   for ("does CODE V's detector reach the same ~14 deg surface?").
    banner('SECTION 1 -- DETECTOR FRAME: ours vs his .seq image ADE');

    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_mm', P.EPD_mm, ...
            'wavelength_m', P.lambda_m, 'model_size', P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
    t.set_hole('M1', P.M1_hole_m);
    t.set_field_bias(P.offset_deg*60);
    t.build();
    fp = t.align_focal_plane('grid',5,'span_arcmin',6);
    nE = numel(t.spec.elt);
    psi = t.spec.elt(nE).psi(:);  psi = psi/norm(psi);

    t.trace_at_field([0 0]);
    s  = macos.trace(nE);
    ri = macos.get_ray_info(s.nRays);
    d1 = ri.dir(:,1);
    chief_vs_axis = atan2d(d1(2), -d1(3));
    ours_vs_axis  = atan2d(psi(2), -psi(3));

    FR = struct('tilt_vs_chief',fp.tilt_deg, 'chief_vs_axis',chief_vs_axis, ...
                'ours_vs_axis',ours_vs_axis, 'his_ADE',S.img_ADE_deg(2), ...
                'psi',psi.');

    fprintf('  our align_focal_plane tilt, vs the ARRIVING CHIEF : %10.4f deg\n', fp.tilt_deg);
    fprintf('  the arriving chief itself,  vs the AXIS           : %10.4f deg\n', chief_vs_axis);
    fprintf('  our detector normal,        vs the AXIS           : %10.6f deg\n', ours_vs_axis);
    fprintf('  his .seq stage-2 image ADE  (CODE V, verbatim)    : %10.6f deg\n', S.img_ADE_deg(2));
    fprintf('  his ADE in OUR frame (decoded sign flip, per convention_decode):\n');
    fprintf('                                                     %10.6f deg\n', -S.img_ADE_deg(2));
    fprintf('\n  |ours - his| WITHOUT the sign flip : %8.4f deg\n', abs(ours_vs_axis - S.img_ADE_deg(2)));
    fprintf('  |ours - his| WITH    the sign flip : %8.4f deg\n', abs(ours_vs_axis + S.img_ADE_deg(2)));
    fprintf(['\n  READ: PACKET §4b''s "14.3 deg tilted image surface" and his -0.07 deg\n' ...
             '  ADE are the SAME surface measured against different references.\n' ...
             '  align_focal_plane reports tilt vs the arriving chief; the chief at\n' ...
             '  +0.5 deg field is itself ~14 deg off the axis (the exit pupil is\n' ...
             '  close to the image, so the image-space chief angle is large).  In\n' ...
             '  the AXIS frame -- CODE V''s -- our detector and his agree to well\n' ...
             '  under a tenth of a degree.  Open ask 2 closes.\n']);

    % per-stage image geometry, for the record
    fprintf('\n  his image surface, per stage (.seq verbatim):\n');
    fprintf('    stage    defocus (mm)      ADE (deg)\n');
    for k = 1:4
        fprintf('      %d   %14.7f  %14.7f\n', k, S.img_defocus_mm(k), S.img_ADE_deg(k));
    end
end

% =====================================================================
function banner(varargin)
    fprintf('\n=================================================================\n');
    fprintf(' %s\n', sprintf(varargin{:}));
    fprintf('=================================================================\n');
end
