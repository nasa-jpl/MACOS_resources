% smoke_focus.m -- find the true best focus (align_focal_plane, seed-independent
% ray-bundle closest-point) and compare to Rodgers' verbatim paraxial focus.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'mmacos_setup.m'));
P = rodgers_common();
t = macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm, ...
        'wavelength_m',P.lambda_m,'model_size',P.model_size);
t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
t.build();
nE = numel(t.spec.elt);
z3 = t.spec.elt(3).Vpt(3);
lam = t.spec.wavelength;
fprintf('seed FP z = %.2f mm\n', t.spec.elt(nE).Vpt(3)*1e3);
fprintf('Rodgers FP z (M3 %.1f - 5095.37) = %.2f mm\n', z3*1e3, (z3-abs(P.s3f_mm)*1e-3)*1e3);

% on-axis best focus (grid over tiny span to identify plane; on-axis dominated)
r = align_wrap(t);
fprintf('\nalign_focal_plane -> FP z = %.2f mm, defocus from seed = %.2f mm, tilt %.4f deg\n', ...
        r.fp_vpt(3)*1e3, r.defocus_m*1e3, r.tilt_deg);

macos.trace(nE); W = macos.opd(); v = W(isfinite(W)&W~=0);
fprintf('on-axis (biased? no) RMS WFE at best focus = %.5f waves = %.3f nm\n', ...
        std(v)/lam, std(v)/lam*lam*1e9);
exit(0);

function r = align_wrap(t)
    r = t.align_focal_plane('grid',3,'span_arcmin',3);
end
