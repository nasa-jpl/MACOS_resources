function common()
% Strict metric but referenced to ONE common sphere (box-center field's chief
% intercept) for ALL box fields -- tests whether Rodgers' numbers live in the
% field-to-field walk against a common surface.
here=fileparts(mfilename('fullpath')); run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
P=rodgers_common(); P.EPD_mm=4060; lam_nm=P.lambda_m*1e9;
Frel=macos.design.field_grid(P.fov_half_deg*60,9,'units','arcmin');
gt=struct('s2',[374.6 199.9],'s3',[91.6 46.4],'s4',[39.8 22.5]);
for st=[2 3 4]
  t=build_stage(P,st); t.add_pupil(numel(t.spec.elt)); pu=t.spec.pupil; iEP=pu.ep_elt;
  % FEX once at the box CENTER field -> fixes ONE common chief-tied sphere
  t.trace_at_field([0 0]); macos.stop(1); try, macos.fex(1); catch, end
  nF=size(Frel,1); W=nan(nF,1);
  for j=1:nF
    t.trace_at_field(Frel(j,:));      % change field, but DO NOT re-run FEX
    macos.trace(iEP); Wm=macos.opd(); v=Wm(isfinite(Wm)&Wm~=0&abs(Wm)<1e30);
    if numel(v)>=8, W(j)=std(v)*lam_nm; end
  end
  t.trace_at_field([]);
  g=gt.(sprintf('s%d',st));
  fprintf('S%d COMMON-sphere (box-center FEX): max %.2f avg %.2f nm | Rodgers %.1f/%.1f -> ratio %.2fx\n',...
    st, max(W), mean(W(isfinite(W))), g(1),g(2), max(W)/g(1));
end
end
function t=build_stage(P,st)
  t=macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm,'wavelength_m',P.lambda_m,'model_size',P.model_size);
  t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
  t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
  t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
  t.set_field_bias(P.offset_deg*60); t.build(); t.align_focal_plane('grid',5,'span_arcmin',6);
  if st==2, return; end
  optF=macos.design.field_grid(P.fov_half_deg*60,3,'units','arcmin','origin',false);
  if st==3, t.optimize('fields',optF,'dofs',[0 0 0 0 0 0 0 1],'max_iters',120);
  else, t.optimize('fields',optF,'dofs',[0 0 0 0 0 0 0 1;1 0 0 0 1 0 0 1;1 0 0 0 1 0 0 1],'max_iters',120); end
  t.align_focal_plane('grid',5,'span_arcmin',6);
end
