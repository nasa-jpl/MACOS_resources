function sgdiag()
here = fileparts(mfilename('fullpath'));
run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
P = rodgers_common(); P.EPD_mm=4060; lam_nm=P.lambda_m*1e9;
t = macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm,'wavelength_m',P.lambda_m,'model_size',P.model_size);
t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
t.set_field_bias(P.offset_deg*60); t.build(); t.align_focal_plane('grid',5,'span_arcmin',6);
t.add_pupil(numel(t.spec.elt)); pu=t.spec.pupil; iEP=pu.ep_elt; iFP=pu.fp_elt;
h=deg2rad(6/60);
for F={[0 0],[h h]}
  f=F{1}; t.trace_at_field(f);
  sfp=macos.trace(iFP); rf=macos.get_ray_info(sfp.nRays);
  ok=rf.ok_trace&rf.ok_pass; Pp=rf.pos; Pp(:,~ok)=NaN; c=mean(Pp(:,ok),2);
  d2=sum((Pp-c).^2,1); d2(~ok)=inf; [~,ic]=min(d2); Cf=rf.pos(:,ic);
  sep=macos.trace(iEP); re=macos.get_ray_info(sep.nRays);
  ok2=re.ok_trace&re.ok_pass&isfinite(re.opl); pos=re.pos(:,ok2); opl=re.opl(ok2);
  dist=sqrt(sum((pos-Cf).^2,1)).';
  raw=opl-mean(opl); conv=opl+dist; conv=conv-mean(conv);
  fprintf('(%+.0f,%+.0f): nOK=%d  opl ptp=%.4e m  |EP-Cf|~%.4f m\n',rad2deg(f(1))*60,rad2deg(f(2))*60,nnz(ok2),max(opl)-min(opl),mean(dist));
  fprintf('   std(opl-mean)=%.4f nm   std(opl+dist-mean)=%.4f nm\n',std(raw)*1e9,std(conv)*1e9);
end
t.trace_at_field([]);
end
