function diag_grid()
% diag_grid -- is the on-axis box-edge WFE (8nm vs Rodgers 1.46) numerical
% sampling noise on the fast f/1.75 beam?  Sweep model_size.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'mmacos_setup.m'));
P = rodgers_common(); lam=P.lambda_m; lam_nm=lam*1e9;
for ms = [128 256 512 1024]
    P.model_size = ms;
    t = macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm, ...
            'wavelength_m',lam,'model_size',ms);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
    t.build();
    t.align_focal_plane('grid',5,'span_arcmin',6);
    F = macos.design.field_grid(P.fov_half_deg*60, 9, 'units','arcmin');
    s = t.realize_apertures('fields',F,'quiet',true);
    w = s.wfe(isfinite(s.wfe))*lam_nm;
    % also the single on-axis (center) field
    t.trace_at_field([0 0]); W=macos.opd(); v=W(isfinite(W)&W~=0); wc=std(v)/lam*lam_nm;
    fprintf('ms=%4d  box min/max/avg = %7.3f %7.3f %7.3f nm   center=%.4f nm\n', ...
            ms, min(w), max(w), mean(w), wc);
end
fprintf('(Rodgers on-axis box: min 0.45  max 1.46  avg 0.61 nm)\n');
exit(0);
end
