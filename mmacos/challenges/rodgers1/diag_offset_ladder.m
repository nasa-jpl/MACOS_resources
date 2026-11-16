function diag_offset_ladder()
% Does the CODE V-consistent metric (per-field defocus-removed RMS) reconcile
% the OFFSET stage too?  Run wfe_field_diag on a FRESH stage-2 build (no
% realize_apertures aperture-resizing first) with the tilted FPA.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'mmacos_setup.m'));
addpath(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'design','src'));
P = rodgers_common(); lam_nm = P.lambda_m*1e9;

t = macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm, ...
        'wavelength_m',P.lambda_m,'model_size',P.model_size);
t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
t.set_field_bias(P.offset_deg*60);
t.build();
r = t.align_focal_plane('grid',5,'span_arcmin',6);
fprintf('FPA tilt=%.3f deg\n', r.tilt_deg);

F = macos.design.field_grid(P.fov_half_deg*60, 9, 'units','arcmin');
d = wfe_field_diag(t, F, 'quiet', true);
show('raw       ', d.rms_raw, lam_nm);
show('-tilt     ', d.rms_tilt, lam_nm);
show('-focus    ', d.rms_focus, lam_nm);
show('-astig    ', d.rms_astig, lam_nm);
fprintf('Rodgers S2 box: 79.381 / 374.590 / 199.950 nm\n');
exit(0);
end
function show(tag, w, lam_nm)
    v = w(isfinite(w))*lam_nm;
    if isempty(v), fprintf('  %s: all NaN (%d/%d finite)\n', tag, 0, numel(w)); return; end
    fprintf('  %s: min/max/avg = %8.3f %8.3f %8.3f nm  (%d/%d finite)\n', ...
            tag, min(v),max(v),mean(v), numel(v),numel(w));
end
