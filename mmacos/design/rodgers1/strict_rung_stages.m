function strict_rung_stages()
%STRICT_RUNG_STAGES  Score stages 2-4 (EPD=4060) with Dave's strict metric:
%   per-field TRUE reference sphere at the chief-ray detector intercept,
%   evaluated at the exit pupil, piston-only removal.  Head-to-head vs
%   Rodgers <=375/<=92/<=40 nm.  Stage 2 is the VALIDATION (frozen design,
%   no optimizer).
    here = fileparts(mfilename('fullpath'));
    run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
    addpath(fullfile(fileparts(fileparts(here)),'design','src'));
    P = rodgers_common();  P.EPD_mm = 4060;  lam_nm = P.lambda_m*1e9;
    Frel = macos.design.field_grid(P.fov_half_deg*60, 9, 'units','arcmin');

    banner('STRICT-RUNG (per-field true sphere @ chief intercept, XP, piston-only) EPD=4060');
    gt = struct('s2',[0.079381 0.37459 0.19995], ...   % Rodgers box [min max avg] waves
                's3',[0.01783 0.091617 0.046379], ...
                's4',[0.01092 0.039802 0.022493]);
    R = struct('lam_nm',lam_nm);

    for st = [2 3 4]
        t = build_stage(P, st);
        t.add_pupil(numel(t.spec.elt));
        s = strict_rung(t, Frel, lam_nm);
        gtv = gt.(sprintf('s%d',st)) * lam_nm;   % nm
        R.(sprintf('s%d',st)) = s;
        fprintf('\n  STAGE %d strict rung: max %.2f  avg %.2f nm   (n %d..%d/%d)\n', ...
                st, s.max, s.avg, min(s.n), max(s.n), numel(s.n));
        fprintf('    Rodgers S%d box: max %.1f  avg %.1f nm   -> ratio max %.2fx avg %.2fx\n', ...
                st, gtv(2), gtv(3), s.max/gtv(2), s.avg/gtv(3));
    end

    banner('SUMMARY -- strict rung vs Rodgers (nm @ %g nm, EPD 4060)', lam_nm);
    fprintf('  stage   strict max/avg    Rodgers max/avg    ratio max\n');
    prow('S2', R.s2, gt.s2*lam_nm);
    prow('S3', R.s3, gt.s3*lam_nm);
    prow('S4', R.s4, gt.s4*lam_nm);
    fprintf('\n  GATE (step 2): S2 strict within ~1.5x of Rodgers 374.6 nm -> metric validated\n');

    save(fullfile(here,'rodgers1_epd4060_strict_rung.mat'),'R');
    fprintf('  saved rodgers1_epd4060_strict_rung.mat\n');
end

function t = build_stage(P, st)
    t = macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm, ...
            'wavelength_m',P.lambda_m,'model_size',P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
    t.set_field_bias(P.offset_deg*60);
    t.build();
    t.align_focal_plane('grid',5,'span_arcmin',6);
    if st == 2, return; end                       % frozen conics, FPA re-fit only
    optF = macos.design.field_grid(P.fov_half_deg*60, 3, 'units','arcmin','origin',false);
    if st == 3
        t.optimize('fields', optF, 'dofs',[0 0 0 0 0 0 0 1], 'max_iters',120);
    else % st == 4
        t.optimize('fields', optF, 'dofs',[0 0 0 0 0 0 0 1;1 0 0 0 1 0 0 1;1 0 0 0 1 0 0 1], 'max_iters',120);
    end
    t.align_focal_plane('grid',5,'span_arcmin',6);
end

function prow(tag, s, gt)
    fprintf('  %-4s   %8.2f/%-8.2f  %8.1f/%-8.1f   %6.2fx\n', tag, s.max, s.avg, gt(2), gt(3), s.max/gt(2));
end
function banner(varargin)
    fprintf('\n==================================================================\n');
    fprintf(' %s\n', sprintf(varargin{:}));
    fprintf('==================================================================\n');
end
