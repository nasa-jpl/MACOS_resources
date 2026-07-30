function diag_ladder()
% diag_ladder -- is CODE V's field-map RMS WFE the piston/tip/tilt-removed
% RMS (rms_tilt)?  Compare the box corner + center under raw vs tilt-removed.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'mmacos_setup.m'));
addpath(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'design','src'));
P = rodgers_common(); lam=P.lambda_m; lam_nm=lam*1e9;

t = macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm, ...
        'wavelength_m',lam,'model_size',P.model_size);
t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
t.build();
t.align_focal_plane('grid',5,'span_arcmin',6);

% STAGE 1 on-axis box, 9x9
F = macos.design.field_grid(P.fov_half_deg*60, 9, 'units','arcmin');
d = wfe_field_diag(t, F, 'quiet', true);
raw = d.rms_raw*lam_nm;  tl = d.rms_tilt*lam_nm;  fo = d.rms_focus*lam_nm;
fprintf('=== STAGE 1 on-axis box (9x9), nm ===\n');
fprintf('             min     max     avg\n');
fprintf(' raw       %6.3f %6.3f %6.3f\n', min(raw),max(raw),mean(raw));
fprintf(' -tilt     %6.3f %6.3f %6.3f   <- CODE V field-map RMS?\n', min(tl),max(tl),mean(tl));
fprintf(' -focus    %6.3f %6.3f %6.3f\n', min(fo),max(fo),mean(fo));
fprintf(' Rodgers   %6.3f %6.3f %6.3f\n', 0.446, 1.463, 0.606);
exit(0);
end
