function run_epd4060(EPD_mm, tag)
%RUN_EPD4060  Full four-stage Rodgers sequence at a given EPD, both metrics,
%   bias-relative box (the §D convention).  Records conics, rigid-body, and
%   the bias-relative stage table for S2/S3/S4 under {global,refsphere}, plus
%   the on-axis S1 anchor.  Emits per-stage field maps + the four .in decks.
%
%   run_epd4060(2000,'epd2000')   % validation gate (reproduce committed)
%   run_epd4060(4060,'epd4060')   % the true-aperture reproduction
%
%   Writes rodgers1_<tag>_results.mat and rodgers1_<tag>_stage{1..4}.in and
%   rodgers1_<tag>_stage{1..4}_*.png in this directory.

    here = fileparts(mfilename('fullpath'));
    run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
    addpath(fullfile(fileparts(fileparts(here)),'design','src'));  % wfe_field_diag
    P = rodgers_common();
    P.EPD_mm = EPD_mm;
    lam_nm   = P.lambda_m*1e9;
    map_n    = 9;
    opt_n    = 3;
    max_iters= 120;
    metrics  = {'global','refsphere'};

    banner('RODGERS EPD SWEEP  tag=%s  EPD=%g mm  (lambda=%g nm)', tag, EPD_mm, lam_nm);

    R = struct('tag',tag,'EPD_mm',EPD_mm,'lam_nm',lam_nm);

    % ---- bias-relative box helper (§D convention) --------------------
    Frel = macos.design.field_grid(P.fov_half_deg*60, map_n, 'units','arcmin');
    biasbox = @(t) [Frel(:,1), t.spec.field_bias + Frel(:,2)];   % centred on +30'
    optF = macos.design.field_grid(P.fov_half_deg*60, opt_n, 'units','arcmin','origin',false);

    % ================= STAGE 1 (on-axis anchor) =======================
    banner('STAGE 1 -- on-axis anchor');
    t1 = build_tma(P, P.K_nom, 0);
    fp1 = t1.align_focal_plane('grid',5,'span_arcmin',6);
    fprintf('  S1 FPA focus z = %.2f mm (Rodgers paraxial %.2f)\n', ...
            fp1.fp_vpt(3)*1e3, -abs(P.s3f_mm));
    F1 = macos.design.field_grid(P.fov_half_deg*60, map_n, 'units','arcmin');  % about 0'
    R.s1 = eval_metrics(t1, F1, metrics, lam_nm, P.gt.s1_onaxis_box);
    report_stage('S1 on-axis box', R.s1, metrics, P.gt.s1_onaxis_box, lam_nm);
    save_maps(t1, F1, metrics, sprintf('%s_stage1',tag), here, lam_nm);
    t1.save(fullfile(here, sprintf('rodgers1_%s_stage1.in',tag)));

    % ================= STAGE 2 (verbatim conics, FPA re-fit) ==========
    banner('STAGE 2 -- offset, verbatim conics, FPA re-fit (bias-relative box)');
    t2 = build_tma(P, P.K_nom, P.offset_deg);
    fp2 = t2.align_focal_plane('grid',5,'span_arcmin',6);
    fprintf('  S2 FPA re-fit: tilt %.4f deg, defocus %.2f mm\n', fp2.tilt_deg, fp2.defocus_m*1e3);
    R.s2_fpa_tilt = fp2.tilt_deg;
    R.s2 = eval_metrics(t2, biasbox(t2), metrics, lam_nm, P.gt.s2_box);
    report_stage('S2 bias-rel box', R.s2, metrics, P.gt.s2_box, lam_nm);
    save_maps(t2, biasbox(t2), metrics, sprintf('%s_stage2',tag), here, lam_nm);
    t2.save(fullfile(here, sprintf('rodgers1_%s_stage2.in',tag)));

    % ================= STAGE 3 (reopt conics + FPA) ===================
    banner('STAGE 3 -- reopt CONICS + FPA (native optimizer)');
    t3 = build_tma(P, P.K_nom, P.offset_deg);
    t3.align_focal_plane('grid',5,'span_arcmin',6);
    t3.optimize('fields', optF, 'dofs',[0 0 0 0 0 0 0 1], 'max_iters', max_iters);
    t3.align_focal_plane('grid',5,'span_arcmin',6);
    R.K_s3 = [t3.spec.elt(1).Kc t3.spec.elt(2).Kc t3.spec.elt(3).Kc];
    report_conics('S3', R.K_s3, P.K_s3);
    R.s3 = eval_metrics(t3, biasbox(t3), metrics, lam_nm, P.gt.s3_box);
    report_stage('S3 bias-rel box', R.s3, metrics, P.gt.s3_box, lam_nm);
    save_maps(t3, biasbox(t3), metrics, sprintf('%s_stage3',tag), here, lam_nm);
    t3.save(fullfile(here, sprintf('rodgers1_%s_stage3.in',tag)));

    % ================= STAGE 4 (tilt/dec M2,M3 + reopt) ===============
    banner('STAGE 4 -- tilt/dec M2,M3 + reopt CONICS + FPA');
    t4 = build_tma(P, P.K_nom, P.offset_deg);
    t4.align_focal_plane('grid',5,'span_arcmin',6);
    perM = [0 0 0 0 0 0 0 1; 1 0 0 0 1 0 0 1; 1 0 0 0 1 0 0 1];
    t4.optimize('fields', optF, 'dofs', perM, 'max_iters', max_iters);
    t4.align_focal_plane('grid',5,'span_arcmin',6);
    R.K_s4 = [t4.spec.elt(1).Kc t4.spec.elt(2).Kc t4.spec.elt(3).Kc];
    report_conics('S4', R.K_s4, P.K_s4);
    [yd2,al2] = rigid_of(t4.spec.elt(2));
    [yd3,al3] = rigid_of(t4.spec.elt(3));
    R.rigid = [yd2 al2; yd3 al3];
    R.rigid_gt = [P.Ydec_M2_mm P.tilt_M2_deg; P.Ydec_M3_mm P.tilt_M3_deg];
    fprintf('  M2 Ydec=%.4f mm (gt %.4f)  tilt=%.4f deg (gt %.4f)\n', yd2,P.Ydec_M2_mm,al2,P.tilt_M2_deg);
    fprintf('  M3 Ydec=%.4f mm (gt %.4f)  tilt=%.4f deg (gt %.4f)\n', yd3,P.Ydec_M3_mm,al3,P.tilt_M3_deg);
    R.s4 = eval_metrics(t4, biasbox(t4), metrics, lam_nm, P.gt.s4_box);
    report_stage('S4 bias-rel box', R.s4, metrics, P.gt.s4_box, lam_nm);
    save_maps(t4, biasbox(t4), metrics, sprintf('%s_stage4',tag), here, lam_nm);
    t4.save(fullfile(here, sprintf('rodgers1_%s_stage4.in',tag)));

    % ---- §D-style table --------------------------------------------
    banner('BIAS-RELATIVE STAGE TABLE  tag=%s  (nm @ %g nm)', tag, lam_nm);
    fprintf('  stage | global max/avg | refsphere max/avg | Rodgers max/avg | g x | rs x\n');
    print_row('S2', R.s2, P.gt.s2_box, lam_nm);
    print_row('S3', R.s3, P.gt.s3_box, lam_nm);
    print_row('S4', R.s4, P.gt.s4_box, lam_nm);

    save(fullfile(here, sprintf('rodgers1_%s_results.mat',tag)),'R');
    fprintf('\nSaved rodgers1_%s_results.mat + .in + png to %s\n', tag, here);
end

% ============================ helpers =============================
function t = build_tma(P, K, bias_deg)
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_mm', P.EPD_mm, ...
            'wavelength_m', P.lambda_m, 'model_size', P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',K(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',K(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',K(3),'spacing_after','derive');
    if bias_deg ~= 0, t.set_field_bias(bias_deg*60); end
    t.build();
end

function R = eval_metrics(t, F, metrics, lam_nm, gt)
    R = struct('gt',gt);
    for m = 1:numel(metrics)
        s = t.realize_apertures('fields', F, 'quiet', true, 'metric', metrics{m});
        w = s.wfe(isfinite(s.wfe));
        R.(metrics{m}) = struct('min',min(w)*lam_nm,'max',max(w)*lam_nm, ...
            'avg',mean(w)*lam_nm,'nfin',numel(w),'ntot',numel(s.wfe));
    end
end

function report_stage(tag, R, metrics, gt, lam_nm)
    for m=1:numel(metrics)
        s = R.(metrics{m}); lost = s.ntot - s.nfin;
        fprintf('  %-20s [%9s] min=%.3f max=%.3f avg=%.3f nm (%d/%d%s)  Rodgers max/avg=%.1f/%.1f\n', ...
            tag, metrics{m}, s.min, s.max, s.avg, s.nfin, s.ntot, ...
            tern(lost>0,sprintf(', %d LOST',lost),''), gt(2)*lam_nm, gt(3)*lam_nm);
    end
end

function report_conics(tag, K, Kgt)
    nm = {'K_M1','K_M2','K_M3'};
    fprintf('  %s conics:      MACOS           Rodgers        |diff|\n', tag);
    for i=1:3
        fprintf('     %-5s %16.9f %16.9f   %.2e\n', nm{i}, K(i), Kgt(i), abs(K(i)-Kgt(i)));
    end
end

function print_row(tag, R, gt, lam_nm)
    g = R.global; rs = R.refsphere;
    gx = g.max/(gt(2)*lam_nm); rx = rs.max/(gt(2)*lam_nm);
    fprintf('  %-4s | %8.1f/%-8.1f | %8.1f/%-8.1f | %6.1f/%-6.1f | %4.1fx | %4.1fx\n', ...
        tag, g.max,g.avg, rs.max,rs.avg, gt(2)*lam_nm,gt(3)*lam_nm, gx, rx);
end

function [ydec_mm, alpha_deg] = rigid_of(e)
    ydec_mm = e.Vpt(2) * 1e3;
    psi = e.psi(:) / norm(e.psi);
    alpha_deg = atan2d(psi(2), -psi(3));
end

function save_maps(t, F, metrics, tagstem, outdir, lam_nm) %#ok<INUSD>
    for m=1:numel(metrics)
        s = t.realize_apertures('fields', F, 'quiet', true, 'metric', metrics{m});
        f = fullfile(outdir, sprintf('rodgers1_%s_%s.png', tagstem, metrics{m}));
        fig = t.view_field_map(s, 'kind','contour', 'save',f, 'visible',false);
        close(fig);
    end
end

function y = tern(c,a,b), if c, y=a; else, y=b; end, end

function banner(varargin)
    s = sprintf(varargin{:});
    fprintf('\n==================================================================\n');
    fprintf(' %s\n', s);
    fprintf('==================================================================\n');
end
