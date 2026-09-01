run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
ap='/home/dcr/dev/MACOS_res_dev/mmacos/challenges/afocal4';
addpath(ap); addpath(fullfile(ap,'descent')); addpath(fullfile(ap,'offaxis'));
P=afocal4_params(); macos.init(P.model_size);
function n=lost(d), macos.load_rx(d); t=macos.trace(macos.num_elt());
  ri=macos.get_ray_info(t.nRays); n=nnz(~(ri.ok_trace(:)&ri.ok_pass(:))); end
function s=v3(v), s=sprintf('%.16E  %.16E  %.16E',v(1),v(2),v(3)); end
function v=g3(txt,k), tk=regexp(txt,['(?m)^\s*' k '=\s*([^\n]*)'],'tokens','once');
  v=sscanf(strrep(tk{1},'D','E'),'%f',3); v=v(:); end
h=0.55;
for f1=[1.25 2.5], for t23=[0.625 1.0]
  S=offaxis_seed(P,'cass','N',5,'f1',f1); S.t(2)=t23; S.t(3)=t23; S.decenter=0;
  b=sprintf('/tmp/trap_%g_%g.in',f1,t23);
  try, descent_build(P,S,b,'defer_union',true,'verify',false,'quiet',true);
  catch ME, fprintf('  f1 %.2f t %.3f  BUILD %s\n',f1,t23,ME.identifier); continue; end
  n0=lost(b);
  % (a) decenter + WIDEN
  txt=fileread(b); cp=g3(txt,'ChfRayPos'); st=g3(txt,'ApStop');
  s=regexprep(txt,'(ChfRayPos=\s*)[^\n]*',['$1' v3(cp+[0;h;0])]);
  s=regexprep(s,'(ApStop=\s*)[^\n]*',['$1' v3(st+[0;h;0])]);
  sw=regexprep(s,'(?m)(^\s*ApVec=\s*)[^\n]*',['$1' v3([4*(h+1) 0 0])]);
  sw=regexprep(sw,'(?m)(^\s*ApType=\s*)None','$1  Circular');
  dw=sprintf('/tmp/trapW_%g_%g.in',f1,t23); fid=fopen(dw,'w');fprintf(fid,'%s',sw);fclose(fid);
  nw=lost(dw);
  % (b) decenter + ApType=None
  sn=regexprep(s,'(?m)(^\s*ApType=\s*)\S+','$1None');
  dn=sprintf('/tmp/trapN_%g_%g.in',f1,t23); fid=fopen(dn,'w');fprintf(fid,'%s',sn);fclose(fid);
  nn=lost(dn);
  % (c) decenter + untouched coaxial apertures
  dc=sprintf('/tmp/trapC_%g_%g.in',f1,t23); fid=fopen(dc,'w');fprintf(fid,'%s',s);fclose(fid);
  nc=lost(dc);
  fprintf(['  f1 %.2f t %.3f : coaxial-lost %4d | decentered: WIDE %4d, ' ...
           'None %4d, as-emitted %4d\n'], f1,t23,n0,nw,nn,nc);
end, end
exit(0);
