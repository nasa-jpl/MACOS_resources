function opd_test()
here=fileparts(mfilename('fullpath')); run(fullfile(fileparts(fileparts(here)),'mmacos_setup.m'));
P=rodgers_common(); P.EPD_mm=4060; lam_nm=P.lambda_m*1e9;
t=macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm,'wavelength_m',P.lambda_m,'model_size',P.model_size);
t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
t.set_field_bias(P.offset_deg*60); t.build(); t.align_focal_plane('grid',5,'span_arcmin',6);
t.add_pupil(numel(t.spec.elt)); pu=t.spec.pupil; iEP=pu.ep_elt;
h=deg2rad(6/60);
for F={[0 0],[0 h],[h 0],[h h]}
  f=F{1}; t.trace_at_field(f); macos.trace(iEP); W=macos.opd();
  [ny,nx]=size(W); [X,Y]=meshgrid(linspace(-1,1,nx),linspace(-1,1,ny));
  m=isfinite(W)&W~=0&abs(W)<1e30; x=X(m);y=Y(m);w=W(m);
  x=x-mean(x);y=y-mean(y);s=max(hypot(x,y));if s>0,x=x/s;y=y/s;end
  Bp=ones(size(x)); Bt=[Bp,x,y]; r2=x.^2+y.^2; Bf=[Bp,x,y,2*r2-1];
  raw=std(w-Bp*(Bp\w))*lam_nm; tl=std(w-Bt*(Bt\w))*lam_nm; fo=std(w-Bf*(Bf\w))*lam_nm;
  fprintf('(%+.0f,%+.0f): opd() raw=%.4f | -tilt=%.4f | -tilt-foc=%.4f nm  (n=%d)\n',...
     rad2deg(f(1))*60,rad2deg(f(2))*60, raw, tl, fo, nnz(m));
end
t.trace_at_field([]);
end
