function fixedfp()
here=fileparts(mfilename('fullpath')); run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
P=rodgers_common(); P.EPD_mm=4060; lam_nm=P.lambda_m*1e9;
Frel=macos.design.field_grid(P.fov_half_deg*60,9,'units','arcmin');
% STAGE 2 frozen, NO align_focal_plane (detector stays at builder's paraxial FP)
t=macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm,'wavelength_m',P.lambda_m,'model_size',P.model_size);
t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
t.set_field_bias(P.offset_deg*60); t.build();
nE=numel(t.spec.elt);
fpz = t.spec.elt(nE).Vpt(3);
fprintf('NO-ALIGN: detector z=%.4f m (paraxial ~ -3.256; Rodgers box max should be large)\n', fpz);
% engine detector rmsWFE across box, NO align, NO pupil
W=nan(size(Frel,1),1);
for j=1:size(Frel,1)
  t.trace_at_field(Frel(j,:)); s=macos.trace(nE); Wm=macos.opd(); v=Wm(isfinite(Wm)&Wm~=0&abs(Wm)<1e30);
  if numel(v)>=8, W(j)=std(v)*lam_nm; end
end
t.trace_at_field([]);
fprintf('STAGE2 NO-ALIGN engine detector rmsWFE box: max %.2f avg %.2f nm (Rodgers 374.6/199.9)\n', max(W), mean(W(isfinite(W))));
% now WITH align, same measure
t.align_focal_plane('grid',5,'span_arcmin',6);
W2=nan(size(Frel,1),1);
for j=1:size(Frel,1)
  t.trace_at_field(Frel(j,:)); s=macos.trace(nE); Wm=macos.opd(); v=Wm(isfinite(Wm)&Wm~=0&abs(Wm)<1e30);
  if numel(v)>=8, W2(j)=std(v)*lam_nm; end
end
t.trace_at_field([]);
fprintf('STAGE2 WITH-ALIGN engine detector rmsWFE box: max %.2f avg %.2f nm\n', max(W2), mean(W2(isfinite(W2))));
end
