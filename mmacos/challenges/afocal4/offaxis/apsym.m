run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
ap='/home/dcr/dev/MACOS_res_dev/mmacos/challenges/afocal4';
addpath(ap); addpath(fullfile(ap,'descent')); addpath(fullfile(ap,'offaxis'));
addpath(fullfile(ap,'wall')); addpath(fullfile(ap,'clearing'));
P=afocal4_params(); macos.init(P.model_size);
D0=wall_recover(P, fullfile(ap,'afocal4_b2long_343mm.in'));
fo=afocal_first_order([abs(P.parent.R(1)) D0.R2],D0.t1,[false true], ...
                      'D',P.D,'stop_ahead',P.stop_ahead);
a=-fo.y_marginal(2)/fo.u_marginal(2)-D0.fm_standoff;
D=struct('N',4,'R',[abs(P.parent.R(1)) D0.R2],'convex',[false true], ...
   't',[D0.t1 a],'K',D0.K(:).','iface',D0.iface,'tilt_deg',zeros(1,4), ...
   'ngrid',P.ngrid,'bias_deg',P.bias_deg);
Pr=P; Pr.pack.enforce=false;
for h=[0.55 1.0]
  % (A) fitted apertures -- what the runs in flight are doing
  Dh=D; Dh.decenter=h; dA=sprintf('/tmp/apA_%g.in',h);
  oA=descent_build(Pr,Dh,dA,'oa_fields',P.Fsolve,'verify',true,'quiet',true);
  SA=afocal4_score(P,dA,'fields',P.Fsolve,'nodes',P.solve.nodes_score,'pupil',false);
  % (B) same decenter, apertures REMOVED -- symmetric with the h=0 control
  dB=sprintf('/tmp/apB_%g.in',h); copyfile(dA,dB);
  txt=regexprep(fileread(dB),'(?m)(^\s*ApType=\s*)\S+','$1None');
  fid=fopen(dB,'w');fprintf(fid,'%s',txt);fclose(fid);
  SB=afocal4_score(P,dB,'fields',P.Fsolve,'nodes',P.solve.nodes_score,'pupil',false);
  macos.load_rx(dA); tA=macos.trace(macos.num_elt()); rA=macos.get_ray_info(tA.nRays);
  macos.load_rx(dB); tB=macos.trace(macos.num_elt()); rB=macos.get_ray_info(tB.nRays);
  fprintf(['  h %.2f : fitted WFE %10.2f nm (%4d pass) | none WFE %10.2f nm ' ...
           '(%4d pass) | delta %.3e rel\n'], h, SA.wfe_max_nm, ...
           nnz(rA.ok_trace&rA.ok_pass), SB.wfe_max_nm, ...
           nnz(rB.ok_trace&rB.ok_pass), abs(SA.wfe_max_nm/SB.wfe_max_nm-1));
end
exit(0);
