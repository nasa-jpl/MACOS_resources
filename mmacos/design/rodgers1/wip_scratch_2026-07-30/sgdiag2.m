function sgdiag2()
here=fileparts(mfilename('fullpath')); run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
P=rodgers_common(); P.EPD_mm=4060; lam_nm=P.lambda_m*1e9;
t=macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm,'wavelength_m',P.lambda_m,'model_size',P.model_size);
t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
t.set_field_bias(P.offset_deg*60); t.build(); t.align_focal_plane('grid',5,'span_arcmin',6);
t.add_pupil(numel(t.spec.elt)); pu=t.spec.pupil; iEP=pu.ep_elt; iFP=pu.fp_elt;
% ON-AXIS field (design field = bias). Engine per-field WFE at detector:
t.trace_at_field([0 0]);
sfp=macos.trace(iFP); fprintf('detector rmsWFE (engine, on-axis) = %.4f nm\n', sfp.rmsWFE*lam_nm);
rf=macos.get_ray_info(sfp.nRays); ok=rf.ok_trace&rf.ok_pass; Pp=rf.pos; Pp(:,~ok)=NaN;
c=mean(Pp(:,ok),2); d2=sum((Pp-c).^2,1); d2(~ok)=inf; [~,ic]=min(d2); Cf=rf.pos(:,ic);
% fit a BEST-focus distance along chief: sweep the sphere center along chief axis, min RMS
sep=macos.trace(iEP); re=macos.get_ray_info(sep.nRays);
ok2=re.ok_trace&re.ok_pass&isfinite(re.opl); pos=re.pos(:,ok2); opl=re.opl(ok2);
% chief dir at detector ~ unit(Cf - EPvpt)
EPv=t.spec.elt(iEP).Vpt(:); ax=(Cf-EPv)/norm(Cf-EPv);
best=inf; bshift=0;
for sh=-0.01:0.0005:0.01
  Cs=Cf+sh*ax; dist=sqrt(sum((pos-Cs).^2,1)).'; w=opl+dist; r=std(w-mean(w))*1e9;
  if r<best, best=r; bshift=sh; end
end
dist0=sqrt(sum((pos-Cf).^2,1)).'; w0=opl+dist0;
fprintf('conv-sphere @Cf: %.2f nm ; best-focus (shift %.4f m): %.2f nm\n', std(w0-mean(w0))*1e9, bshift, best);
t.trace_at_field([]);
end
