function diag_apert()
% diag_apert -- the EPD is not on any slide.  Sweep D: which aperture makes
% the on-axis 0.2x0.2 box (min/max/avg) match Rodgers (0.446/1.463/0.606 nm)?
% On-axis center is spherical-dominated (~D^4); corners add coma/astig
% (~D^3/D^2), so the D that fits BOTH resolves the unstated aperture.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'mmacos_setup.m'));
P = rodgers_common(); lam=P.lambda_m; lam_nm=lam*1e9;
fprintf('  D(mm)   f/#    box: min    max    avg  (nm)   [Rodgers 0.446/1.463/0.606]\n');
for D = [800 1000 1200 1400 1600 2000]
    t = macos.design.Telescope('family','TMA','aperture_diameter_mm',D, ...
            'wavelength_m',lam,'model_size',256);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
    t.build();
    t.align_focal_plane('grid',5,'span_arcmin',6);
    F = macos.design.field_grid(P.fov_half_deg*60, 9, 'units','arcmin');
    s = t.realize_apertures('fields',F,'quiet',true);
    w = s.wfe(isfinite(s.wfe))*lam_nm;
    fprintf('  %5.0f  %5.2f    %7.3f %7.3f %7.3f\n', D, t.spec.derived.fnum, min(w),max(w),mean(w));
end
exit(0);
end
