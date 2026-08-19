% smoke_layout.m -- Stage-0 gate: build Rodgers' verbatim Rx, dump layout.
% Verify vertex z-fold + derived focus vs his signed CODE V spacings BEFORE
% trusting any WFE number.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'mmacos_setup.m'));
P = rodgers_common();

t = macos.design.Telescope('family','TMA', ...
        'aperture_diameter_mm', P.EPD_mm, ...
        'wavelength_m', P.lambda_m, 'model_size', P.model_size);
t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1), ...
             'spacing_after_mm',abs(P.s12_mm));
t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2), ...
             'spacing_after_mm',abs(P.s23_mm));
t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3), ...
             'spacing_after','derive');
t.build();
t.describe();

d = t.spec.derived;
fprintf('\n=== LAYOUT CHECK ===\n');
fprintf('EFL = %.3f mm   f/# = %.4f\n', d.EFL*1e3, d.fnum);
fprintf('vertex z (mm): '); fprintf('%.4f ', d.z*1e3); fprintf('\n');
fprintf('derived t_focus (M3->FP) = %.4f mm  (Rodgers paraxial: %.4f mm)\n', ...
        d.t_focus*1e3, P.s3f_mm);
fprintf('|separations| (mm): M1-M2=%.3f  M2-M3=%.3f\n', ...
        abs(d.z(2)-d.z(1))*1e3, abs(d.z(3)-d.z(2))*1e3);
fprintf('Rodgers |sep|:      M1-M2=%.3f  M2-M3=%.3f\n', abs(P.s12_mm), abs(P.s23_mm));
fprintf('conics resolved: '); fprintf('%.6f ', d.K); fprintf('\n');
exit(0);
