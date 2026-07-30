function out = rodgers1(varargin)
%RODGERS1  Reproduce J.M. Rodgers' offset-field coaxial-TMA study in MACOS.
% =====================================================================
%  A parameter-driven driver that rebuilds the four-stage TMA study in
%  macos_sandbox/Design/Rodgers/260728-TMA_Offsetfield-jmr.pptx (ORA /
%  CODE V, 28-Jul-2026) inside the MACOS design layer, and compares --
%  stage by stage -- the achieved RMS wavefront error field map and the
%  solved optical parameters against Rodgers' CODE V results.
%
%  THE SYSTEM.  A coaxial three-mirror anastigmat (all mirrors on one
%  axis), 2 m aperture, EFL ~3.5 m (f/1.75), imaging a small 0.2deg x
%  0.2deg field.  lambda = 1000 nm.  The interesting part is what
%  happens when the used field is pushed 0.5deg OFF axis: the on-axis
%  corrections no longer serve it, WFE explodes, and successively richer
%  degree-of-freedom sets buy it back.
%
%  THE FOUR STAGES (each = an evaluation OR an optimization):
%    1  On-axis, conics optimized      -> essentially perfect (~1 nm).
%    2  0.5deg-offset field, SAME conics, only the focal-plane (FPA)
%       tilt+focus re-fit                -> WFE high in the used box.
%    3  Re-OPTIMIZE the three conics (+FPA tilt/focus) for the offset
%       field                            -> used-box WFE back down (~92 nm).
%    4  Also let M2,M3 DECENTER (Ydec) + TILT (alpha), re-optimize conics
%       + FPA                            -> used-box WFE lower still (~40 nm).
%
%  WHAT THIS DRIVER SHOWS.  Stages 1-2 are a PURE EVALUATION of Rodgers'
%  verbatim prescription -- the gate is reproducing his field-map
%  min/max/avg.  Stages 3-4 run OUR native optimizer (MACOS CALIB) with
%  his exact DOF sets and compare BOTH the achieved WFE and the solved
%  parameter values (his conics to ~4 decimals, Ydec_M3 = 121.868 mm,
%  alpha-tilt = 2.330deg).
%
%  CODE V -> MACOS conventions (see rodgers_common.m):
%    * ROC: signed in CODE V; MACOS emits KrElt = -|R| (convexity is
%      geometry, not sign).  We pass abs(ROC).
%    * conic K: identical (KcElt = K).  spacings: pass magnitudes; the
%      builder folds z with (-1)^k.  FOV / offset / tilts in DEGREES.
%    * lambda = 1 um; WFE reported in nm (Rodgers' slides are in waves).
%
%  USAGE (all name/value, all optional -- defaults reproduce the study):
%    rodgers1                              % run all 4 stages, save artifacts
%    rodgers1('stages',[1 2])              % only stages 1-2
%    rodgers1('EPD_mm',2000,'lambda_nm',1000)   % change aperture / lambda
%    rodgers1('map_n',9)                   % finer field-map grid (odd)
%    rodgers1('save',false,'plots',false)  % numbers only, no files
%    out = rodgers1(...)                   % struct: per-stage stats + params
%
%  Batch:  matlab -batch "run('.../rodgers1.m'); exit(0)"   (uses defaults)
% =====================================================================

    here = fileparts(mfilename('fullpath'));
    run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
    addpath(fullfile(fileparts(fileparts(here)),'design','src'));  % wfe_field_diag
    P = rodgers_common();

    % ----- knobs -----------------------------------------------------
    ip = inputParser;
    ip.addParameter('stages', 1:4);
    ip.addParameter('EPD_mm',   P.EPD_mm);
    ip.addParameter('lambda_nm', P.lambda_m*1e9);
    ip.addParameter('model_size', P.model_size);
    ip.addParameter('map_n',  9);        % odd NxN field-map grid (eval)
    ip.addParameter('opt_n',  3);        % odd NxN optimize grid (<=3 -> 9 FoV)
    ip.addParameter('max_iters', 120);
    ip.addParameter('save',  true);
    ip.addParameter('plots', true);
    ip.addParameter('outdir', here);
    ip.parse(varargin{:});
    A = ip.Results;
    P.EPD_mm = A.EPD_mm;  P.lambda_m = A.lambda_nm*1e-9;  P.model_size = A.model_size;
    lam_nm = A.lambda_nm;

    banner('RODGERS OFFSET-FIELD COAXIAL-TMA STUDY  (lambda = %g nm, EPD = %g mm)', ...
           lam_nm, P.EPD_mm);

    out = struct('P',P,'A',A,'stage',struct([]));

    % ================================================================
    % STAGE 1 -- on-axis, verbatim conics.  PURE EVALUATION.
    % ================================================================
    if any(A.stages==1)
        banner('STAGE 1 -- on-axis 0.2x0.2 FOV, verbatim conics (EVALUATION)');
        t = build_tma(P, P.K_nom, 0);           % no field bias
        fp = t.align_focal_plane('grid',5,'span_arcmin',6);  % FPA focus (on-axis)
        fprintf('  FPA focus set: FP z = %.2f mm  (Rodgers paraxial %.2f mm)\n', ...
                fp.fp_vpt(3)*1e3, -abs(P.s3f_mm));
        s1 = eval_box(t, 0, P, A.map_n);        % 0 offset
        report_map('S1 on-axis box', s1, P.gt.s1_onaxis_box, lam_nm);
        out.stage(1).ladder = metric_ladder(t, P, A.map_n, lam_nm, P.gt.s1_onaxis_box);
        out.stage(1).name='S1 on-axis (verbatim conics)';
        out.stage(1).scan=s1; out.stage(1).gt=P.gt.s1_onaxis_box;
        if A.plots, save_map(t,s1,'stage1_onaxis',A,lam_nm); end
        if A.save,  t.save(fullfile(A.outdir,'rodgers1_stage1.in')); end
    end

    % ================================================================
    % STAGE 2 -- 0.5deg offset FOV, SAME conics, only FPA tilt/focus.
    %            PURE EVALUATION (his slide 2).
    % ================================================================
    if any(A.stages==2)
        banner('STAGE 2 -- 0.5deg-offset FOV, verbatim conics, FPA re-fit (EVALUATION)');
        t = build_tma(P, P.K_nom, P.offset_deg);            % bias +y 0.5deg
        fp = t.align_focal_plane('grid',5,'span_arcmin',6); % FPA tilt+focus
        fprintf('  FPA re-fit at offset: tilt %.4f deg, defocus %.2f mm\n', ...
                fp.tilt_deg, fp.defocus_m*1e3);
        s2 = eval_box(t, 0, P, A.map_n);        % box centered on the bias
        report_map('S2 offset box', s2, P.gt.s2_box, lam_nm);
        out.stage(2).ladder = metric_ladder(t, P, A.map_n, lam_nm, P.gt.s2_box);
        out.stage(2).name='S2 offset (verbatim conics, FPA re-fit)';
        out.stage(2).scan=s2; out.stage(2).gt=P.gt.s2_box;
        if A.plots, save_map(t,s2,'stage2_offset',A,lam_nm); end
        if A.save,  t.save(fullfile(A.outdir,'rodgers1_stage2.in')); end
    end

    % ================================================================
    % STAGE 3 -- offset FOV, RE-OPTIMIZE conics + FPA.  OUR optimizer.
    % ================================================================
    if any(A.stages==3)
        banner('STAGE 3 -- offset FOV, re-optimize CONICS + FPA (NATIVE OPTIMIZER)');
        t = build_tma(P, P.K_nom, P.offset_deg);
        t.align_focal_plane('grid',5,'span_arcmin',6);      % FPA to true focus FIRST
        optF = macos.design.field_grid(P.fov_half_deg*60, A.opt_n, ...
                    'units','arcmin', 'origin',false);      % 0.1deg half = 6'
        r = t.optimize('fields', optF, 'dofs',[0 0 0 0 0 0 0 1], ... % conic only
                       'max_iters', A.max_iters);
        t.align_focal_plane('grid',5,'span_arcmin',6);      % re-fit FPA tilt/focus
        s3 = eval_box(t, 0, P, A.map_n);
        report_map('S3 offset box', s3, P.gt.s3_box, lam_nm);
        out.stage(3).ladder = metric_ladder(t, P, A.map_n, lam_nm, P.gt.s3_box);
        Ksol = [t.spec.elt(1).Kc t.spec.elt(2).Kc t.spec.elt(3).Kc];
        report_conics('S3', Ksol, P.K_s3);
        out.stage(3).name='S3 offset (reopt conics + FPA)';
        out.stage(3).scan=s3; out.stage(3).gt=P.gt.s3_box;
        out.stage(3).K=Ksol; out.stage(3).K_gt=P.K_s3;
        if A.plots, save_map(t,s3,'stage3_reoptconics',A,lam_nm); end
        if A.save,  t.save(fullfile(A.outdir,'rodgers1_stage3.in')); end
    end

    % ================================================================
    % STAGE 4 -- also TILT+DECENTER M2,M3; re-optimize conics + FPA.
    %            PER-ELEMENT DOF: M1 held rigid (conic only), M2/M3 add
    %            Ydec (DY) + alpha-tilt (TIP).  OUR optimizer.
    % ================================================================
    if any(A.stages==4)
        banner('STAGE 4 -- tilt/dec M2,M3 + reopt CONICS + FPA (NATIVE OPTIMIZER)');
        t = build_tma(P, P.K_nom, P.offset_deg);
        t.align_focal_plane('grid',5,'span_arcmin',6);      % FPA to true focus FIRST
        optF = macos.design.field_grid(P.fov_half_deg*60, A.opt_n, ...
                    'units','arcmin', 'origin',false);
        % per-element mask (rows aligned to var_elts = [1 2 3]):
        %   M1: conic only (held rigid -> no global-tilt gauge freedom)
        %   M2,M3: TIP (alpha-tilt about x) + DY (Ydec) + CONIC
        perM = [0 0 0 0 0 0 0 1;
                1 0 0 0 1 0 0 1;
                1 0 0 0 1 0 0 1];
        r = t.optimize('fields', optF, 'dofs', perM, 'max_iters', A.max_iters);
        t.align_focal_plane('grid',5,'span_arcmin',6);
        s4 = eval_box(t, 0, P, A.map_n);
        report_map('S4 offset box', s4, P.gt.s4_box, lam_nm);
        out.stage(4).ladder = metric_ladder(t, P, A.map_n, lam_nm, P.gt.s4_box);
        Ksol = [t.spec.elt(1).Kc t.spec.elt(2).Kc t.spec.elt(3).Kc];
        report_conics('S4', Ksol, P.K_s4);
        % solved rigid-body params (LOCAL element frame): Ydec = Vpt y shift,
        % alpha-tilt = angle of psi about x from (0,0,-1).
        [yd2,al2] = rigid_of(t.spec.elt(2));
        [yd3,al3] = rigid_of(t.spec.elt(3));
        report_rigid('M2', yd2, al2, P.Ydec_M2_mm, P.tilt_M2_deg);
        report_rigid('M3', yd3, al3, P.Ydec_M3_mm, P.tilt_M3_deg);
        out.stage(4).name='S4 offset (tilt/dec M2,M3 + reopt)';
        out.stage(4).scan=s4; out.stage(4).gt=P.gt.s4_box;
        out.stage(4).K=Ksol; out.stage(4).K_gt=P.K_s4;
        out.stage(4).rigid=[yd2 al2; yd3 al3];
        out.stage(4).rigid_gt=[P.Ydec_M2_mm P.tilt_M2_deg; P.Ydec_M3_mm P.tilt_M3_deg];
        if A.plots, save_map(t,s4,'stage4_tiltdec',A,lam_nm); end
        if A.save,  t.save(fullfile(A.outdir,'rodgers1_stage4.in')); end
    end

    % ----- summary table + packet -----------------------------------
    print_summary(out, lam_nm);
    if A.save
        save(fullfile(A.outdir,'rodgers1_results.mat'),'out');
        fprintf('\nSaved results + .in files + figures to %s\n', A.outdir);
    end
end

% =====================================================================
%  helpers
% =====================================================================
function t = build_tma(P, K, bias_deg)
%BUILD_TMA  Rodgers' verbatim coaxial TMA, conics = K, field bias in +y deg.
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_mm', P.EPD_mm, ...
            'wavelength_m', P.lambda_m, 'model_size', P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',K(1), ...
                 'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',K(2), ...
                 'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',K(3), ...
                 'spacing_after','derive');
    if bias_deg ~= 0, t.set_field_bias(bias_deg*60); end   % arcmin
    t.build();
end

function L = metric_ladder(t, P, n, lam_nm, gt)
%METRIC_LADDER  Per-field RMS WFE over the box under three metric definitions
%   (raw / piston+tilt removed / +defocus removed), to show WHICH CODE V
%   field-map convention the numbers match.  CODE V's field-map RMS WFE
%   references each field to its own image-point sphere; mmacos realize_
%   apertures uses std(OPD) at ONE global image plane, which leaves the
%   fast anastigmat's field-curvature defocus in the off-center corners.
    F = macos.design.field_grid(P.fov_half_deg*60, n, 'units','arcmin');
    d = wfe_field_diag(t, F, 'quiet', true);
    raw = d.rms_raw(isfinite(d.rms_raw))*lam_nm;
    tl  = d.rms_tilt(isfinite(d.rms_tilt))*lam_nm;
    fo  = d.rms_focus(isfinite(d.rms_focus))*lam_nm;
    if isempty(raw)
        % all fields lost rays -- happens on the STEEPLY-TILTED offset FP
        % (the per-field decomposition traces to the physical FP surface; a
        % ~14 deg tilt at 0.5 deg off-axis vignettes the box).  Itself a
        % finding: see the packet's FPA-tilt note.
        fprintf('  metric ladder: all fields lost rays on the tilted FP ');
        fprintf('(see FPA-tilt note in the packet)\n');
        L = struct('raw',[],'tilt',[],'focus',[]);  return;
    end
    fprintf('  metric ladder (nm)         min      max      avg\n');
    fprintf('    raw (piston only)     %8.3f %8.3f %8.3f\n', min(raw),max(raw),mean(raw));
    fprintf('    - piston/tip/tilt     %8.3f %8.3f %8.3f\n', min(tl),max(tl),mean(tl));
    fprintf('    - + defocus (per fld) %8.3f %8.3f %8.3f\n', min(fo),max(fo),mean(fo));
    fprintf('    Rodgers (CODE V)      %8.3f %8.3f %8.3f\n', gt(1)*lam_nm,gt(2)*lam_nm,gt(3)*lam_nm);
    L = struct('raw',raw,'tilt',tl,'focus',fo);
end

function scan = eval_box(t, extra_off_deg, P, n)
%EVAL_BOX  RMS WFE over the 0.2x0.2 used FOV (centered on the field bias).
%   n x n grid, half-field 0.1 deg.  extra_off_deg shifts the box center
%   in +y beyond the bias (unused here; kept for reuse).
    F = macos.design.field_grid(P.fov_half_deg*60, n, 'units','arcmin');
    if extra_off_deg ~= 0
        F(:,2) = F(:,2) + deg2rad(extra_off_deg);
    end
    scan = t.realize_apertures('fields', F, 'quiet', true);
end

function [ydec_mm, alpha_deg] = rigid_of(e)
%RIGID_OF  Recover Ydec (mm) and alpha-tilt (deg) of a moved coaxial mirror.
%   Ydec = the y-component of the vertex shift off the nominal axis.
%   alpha = tilt of psi about x, measured from the nominal (0,0,-1).
    ydec_mm = e.Vpt(2) * 1e3;
    psi = e.psi(:) / norm(e.psi);
    % nominal psi = (0,0,-1); alpha about x rotates psi into the y-z plane:
    % psi = (0, sin(alpha)*(+/-), -cos(alpha)).  angle from -z:
    alpha_deg = atan2d(psi(2), -psi(3));
end

function banner(varargin)
    s = sprintf(varargin{:});
    fprintf('\n==================================================================\n');
    fprintf(' %s\n', s);
    fprintf('==================================================================\n');
end

function report_map(name, scan, gt, lam_nm)
%REPORT_MAP  min/max/avg WFE (nm) vs Rodgers ground truth (waves->nm).
    w  = scan.wfe(isfinite(scan.wfe));
    mn = min(w)*lam_nm;  mx = max(w)*lam_nm;  av = mean(w)*lam_nm;
    g  = gt*lam_nm;      % [min max avg std] in nm
    fprintf('  %s (nm):        MACOS      Rodgers    ratio\n', name);
    fprintf('     min WFE  %12.3f %11.3f   %6.2fx\n', mn, g(1), safe(mn,g(1)));
    fprintf('     max WFE  %12.3f %11.3f   %6.2fx\n', mx, g(2), safe(mx,g(2)));
    fprintf('     avg WFE  %12.3f %11.3f   %6.2fx\n', av, g(3), safe(av,g(3)));
    flag_dev('max', mx, g(2));  flag_dev('avg', av, g(3));
end

function report_conics(tag, K, Kgt)
    fprintf('  %s conics:        MACOS            Rodgers        |diff|\n', tag);
    nm = {'K_M1','K_M2','K_M3'};
    for i=1:3
        fprintf('     %-5s %16.9f %16.9f   %.2e\n', nm{i}, K(i), Kgt(i), abs(K(i)-Kgt(i)));
    end
end

function report_rigid(name, yd, al, yd_gt, al_gt)
    fprintf('  %s rigid-body:    MACOS        Rodgers      |diff|\n', name);
    fprintf('     Ydec (mm)  %11.4f %11.4f   %.3e\n', yd, yd_gt, abs(yd-yd_gt));
    fprintf('     tilt (deg) %11.4f %11.4f   %.3e\n', al, al_gt, abs(al-al_gt));
    flag_dev([name ' Ydec'], yd, yd_gt);
    flag_dev([name ' tilt'], al, al_gt);
end

function r = safe(a,b)
    if b==0, r=Inf; else, r=a/b; end
end

function flag_dev(what, val, ref)
%FLAG_DEV  Flag a >2x deviation for Fable (per the task: do not tune past it).
    if ref==0, return; end
    rr = val/ref;
    if rr > 2 || rr < 0.5
        fprintf('     >>> FLAG (%s): %.3g vs %.3g = %.2fx deviation -- FOR FABLE, not tuning\n', ...
                what, val, ref, rr);
    end
end

function save_map(t, scan, tag, A, lam_nm)
    f = fullfile(A.outdir, ['rodgers1_' tag '.png']);
    sc = scan;  sc.wfe = scan.wfe;   % view_field_map plots in waves
    fig = t.view_field_map(sc, 'kind','contour', 'save',f, 'visible',false);
    close(fig);
    fprintf('  field map -> %s\n', f);
end

function print_summary(out, lam_nm)
    banner('SUMMARY -- used-box RMS WFE: MACOS vs Rodgers (nm @ %g nm)', lam_nm);
    fprintf('  %-34s  MACOS max   Rodgers max   ratio\n','stage');
    for i=1:numel(out.stage)
        s = out.stage(i);
        if isempty(s) || ~isfield(s,'scan') || isempty(s.scan), continue; end
        w = s.scan.wfe(isfinite(s.scan.wfe));
        mx = max(w)*lam_nm;  g = s.gt(2)*lam_nm;
        fprintf('  %-34s %9.3f %12.3f   %6.2fx\n', s.name, mx, g, safe(mx,g));
    end
end
