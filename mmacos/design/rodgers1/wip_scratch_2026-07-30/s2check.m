function s2check()
here=fileparts(mfilename('fullpath')); run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
P=rodgers_common(); P.EPD_mm=4060; lam_nm=P.lambda_m*1e9;
t=macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm,'wavelength_m',P.lambda_m,'model_size',P.model_size);
t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
t.set_field_bias(P.offset_deg*60); t.build(); t.align_focal_plane('grid',5,'span_arcmin',6);
nFP=numel(t.spec.elt);
% Engine's OWN detector rmsWFE, stage 2, at box corner vs center (NO pupil, NO refsphere)
h=deg2rad(6/60);
for F={[0 0],[h h],[0 h],[h 0]}
  f=F{1}; t.trace_at_field(f); s=macos.trace(nFP);
  W=macos.opd(); v=W(isfinite(W)&W~=0&abs(W)<1e30);
  fprintf('  stage2 (%+.0f,%+.0f)'': detector rmsWFE=%.4f nm  std(opd)=%.4f nm  ptp=%.4f nm  n=%d\n',...
     rad2deg(f(1))*60,rad2deg(f(2))*60, s.rmsWFE*lam_nm, std(v)*lam_nm, (max(v)-min(v))*lam_nm, numel(v));
end
t.trace_at_field([]);
end
