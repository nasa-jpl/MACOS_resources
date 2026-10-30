run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
ap='/home/dcr/dev/MACOS_res_dev/mmacos/challenges/afocal4';
addpath(ap); addpath(fullfile(ap,'descent')); addpath(fullfile(ap,'offaxis'));
P=afocal4_params(); macos.init(P.model_size);
fprintf('\n  P.iface = %.4f m; the committed deck is 343 mm.\n', P.iface);
fprintf(['  A bare Mersenne cannot MOVE its pupil, so its interface plane is\n' ...
         '  only a reporting plane -- and its output is COLLIMATED, so the\n' ...
         '  wavefront should be invariant to where that plane sits.  Measured,\n' ...
         '  not assumed:\n\n']);
fprintf('  %-6s %8s %11s %11s %9s %11s\n','form','iface m','rung2 nm','rung3 nm','traced M','union mm');
for f1=[2.5 5.0]
 for ifc=[0.140 0.343]
  f2=f1/P.M; sep=f1-f2;
  t=macos.design.Telescope('family','tma','aperture_diameter_m',P.D, ...
      'wavelength_m',P.lambda,'grid_npts',P.ngrid,'model_size',P.model_size);
  t.add_mirror('M1','radius_m',2*f1,'spacing_after_m',sep,'convex',false,'conic',-1);
  t.add_mirror('M2','radius_m',2*f2,'spacing_after_m',ifc,'convex',true,'conic',-1);
  t.add_exit_reference('ColdStop','dist_m',ifc);
  if P.bias_deg~=0, t.set_field_bias(P.bias_deg*60); end
  d=sprintf('/tmp/ifc_%g_%g.in',f1,ifc); t.build(d);
  oa=offaxis_decenter(d,1.5,'fields',P.Fsolve,'quiet',true);
  S=afocal4_score(P,d,'fields',P.Fsolve,'nodes',P.solve.nodes_score,'pupil',false);
  K=afocal4_union(d,'fields',P.Fsolve,'body_k',1.15,'body_pad',0.015,'quiet',true);
  fprintf('  f1=%-3.1f %8.3f %11.1f %11.1f %9.4f %11.1f\n', f1, ifc, ...
          S.wfe_max_nm, S.wfe_rung3_max_nm, oa.traced.mag, K.floor_m*1e3);
 end
end
exit(0);
