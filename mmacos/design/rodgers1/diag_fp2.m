function diag_fp2()
% diag_fp2 -- is align_focal_plane's 14deg tilt at the offset field spurious
% (fast f/1.75 beam, near-collinear-foci warning)?  Compare box stats under
% NO align, focus-only (grid=0 minimal), and grid tilt-fit; and check whether
% the field-CENTER loses rays (NaN) under the tilted FP.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'mmacos_setup.m'));
P = rodgers_common(); lam=P.lambda_m; lam_nm=lam*1e9;

t = mk(P,P.offset_deg);
fprintf('seed FP z=%.2f mm\n', t.spec.elt(end).Vpt(3)*1e3);
box(t,P,lam_nm,'seed (no align)');
cwfe(t,lam_nm,'seed');

t = mk(P,P.offset_deg);
r = t.align_focal_plane('grid',0,'span_arcmin',6);   % minimal 4-pt cross
fprintf('\ngrid=0 cross: tilt=%.4f deg defocus=%.2f mm\n', r.tilt_deg, r.defocus_m*1e3);
box(t,P,lam_nm,'align grid=0');
cwfe(t,lam_nm,'align0');

t = mk(P,P.offset_deg);
r = t.align_focal_plane('grid',5,'span_arcmin',6);
fprintf('\ngrid=5: tilt=%.4f deg defocus=%.2f mm\n', r.tilt_deg, r.defocus_m*1e3);
box(t,P,lam_nm,'align grid=5');
cwfe(t,lam_nm,'align5');
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
function box(t,P,lam_nm,tag)
    F = macos.design.field_grid(P.fov_half_deg*60, 9, 'units','arcmin');
    s = t.realize_apertures('fields',F,'quiet',true);
    w = s.wfe(isfinite(s.wfe))*lam_nm;
    fprintf('  [%s] box min/max/avg = %.3f %.3f %.3f nm (n=%d finite)\n', ...
            tag, min(w),max(w),mean(w), numel(w));
end
function cwfe(t,lam_nm,tag)
    t.trace_at_field([0 0]);
    W = macos.opd(); v = W(isfinite(W)&W~=0 & abs(W)<1e30);
    if isempty(v)
        fprintf('  [%s] center WFE = NaN (all rays lost!)\n',tag);
    else
        fprintf('  [%s] center WFE = %.4f nm (n=%d rays)\n', tag, std(v)*1e9, numel(v));
    end
end
