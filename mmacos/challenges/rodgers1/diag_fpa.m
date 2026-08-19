function diag_fpa()
% diag_fpa -- why is the field-map WFE off? isolate the FPA model.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'mmacos_setup.m'));
P = rodgers_common(); lam = P.lambda_m; lam_nm = lam*1e9;

fprintf('=== STAGE 1 on-axis (bias 0) ===\n');
t = mk(P,0);
fp = t.align_focal_plane('grid',5,'span_arcmin',6);
fprintf(' align: tilt=%.4f deg defocus=%.2f mm\n', fp.tilt_deg, fp.defocus_m*1e3);
w = boxstats(t,P,lam_nm); fprintf(' box(align) min/max/avg = %.3f %.3f %.3f nm  (Rodgers 0.45/1.46/0.61)\n', w);
fprintf(' center WFE = %.4f nm\n', centerwfe(t,lam,lam_nm));

fprintf('\n=== STAGE 2 offset 0.5deg ===\n');
t = mk(P,P.offset_deg);
w = boxstats(t,P,lam_nm); fprintf(' (i)  seed focus:        box min/max/avg = %.2f %.2f %.2f nm\n', w);
fprintf('       center WFE = %.3f nm  (Rodgers box min=79)\n', centerwfe(t,lam,lam_nm));

t = mk(P,P.offset_deg);
fp = t.align_focal_plane('grid',5,'span_arcmin',6);
fprintf(' (iii) align tilt=%.3f deg defocus=%.2f mm\n', fp.tilt_deg, fp.defocus_m*1e3);
w = boxstats(t,P,lam_nm); fprintf('       box min/max/avg = %.2f %.2f %.2f nm  (Rodgers 79/375/200)\n', w);
fprintf('       center WFE = %.3f nm\n', centerwfe(t,lam,lam_nm));

% ALSO: what half-field does the box span, and is realize_apertures resizing
% apertures (clipping) mid-scan? print the field extremes it evaluated.
F = macos.design.field_grid(P.fov_half_deg*60, 9, 'units','arcmin');
fprintf('\n box field half-extent: %.4f deg (x), %.4f deg (y about bias)\n', ...
        max(F(:,1))*180/pi, max(F(:,2))*180/pi);
exit(0);
end

function t = mk(P,bias_deg)
    t = macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm, ...
            'wavelength_m',P.lambda_m,'model_size',P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
    if bias_deg~=0, t.set_field_bias(bias_deg*60); end
    t.build();
end
function w = boxstats(t,P,lam_nm)
    F = macos.design.field_grid(P.fov_half_deg*60, 9, 'units','arcmin');
    s = t.realize_apertures('fields',F,'quiet',true);
    w = s.wfe(isfinite(s.wfe))*lam_nm;
    w = [min(w) max(w) mean(w)];
end
function wc = centerwfe(t,lam,lam_nm)
    t.trace_at_field([0 0]);
    W = macos.opd(); v = W(isfinite(W)&W~=0);
    wc = std(v)/lam*lam_nm;
end
