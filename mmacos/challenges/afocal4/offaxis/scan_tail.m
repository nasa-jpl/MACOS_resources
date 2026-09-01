run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
ap='/home/dcr/dev/MACOS_res_dev/mmacos/challenges/afocal4';
addpath(ap); addpath(fullfile(ap,'descent')); addpath(fullfile(ap,'offaxis'));
P=afocal4_params(); macos.init(P.model_size);
form=getenv('SC_FORM'); if isempty(form), form='cass'; end
f1=str2double(getenv('SC_F1')); if isnan(f1), f1=1.25; end
S0=offaxis_seed(P,form,'N',5,'f1',f1);
rows=struct([]); d=sprintf('/tmp/sc_%s_%g.in',form,f1);
R3g=[0.4 0.6 0.9 1.2 1.6 2.2 3.0]; t2g=[0.3 0.5 0.8 1.2]; t3g=[0.3 0.5 0.8 1.2];
fprintf('\n  %-6s %5s %5s %5s | %10s %9s %9s %8s %6s\n', ...
   'form','R3','t2','t3','M','err %','coll urad','behind','lost');
for R3=R3g, for t2=t2g, for t3=t3g
  S=S0; S.R(3)=R3; S.t(2)=t2; S.t(3)=t3; S.decenter=S0.decenter;
  try
    o=descent_build(P,S,d,'defer_union',true,'oa_fields',P.Fsolve, ...
                    'quiet',true,'verify',true);
  catch, continue; end
  e=(o.traced.mag/P.M-1)*100;
  fprintf('  %-6s %5.2f %5.2f %5.2f | %10.4f %9.3f %9.1f %8.0f %6d %s\n', ...
     form,R3,t2,t3,o.traced.mag,e,o.traced.collimation_urad, ...
     o.behind_m1*1e3,o.offaxis.nlost, ...
     tern(abs(e)<2 && o.traced.collimation_urad<2000,'<== GOOD',''));
  r=struct('R3',R3,'t2',t2,'t3',t3,'M',o.traced.mag,'err',e, ...
           'coll',o.traced.collimation_urad,'behind',o.behind_m1, ...
           'lost',o.offaxis.nlost);
  if isempty(rows), rows=r; else, rows(end+1)=r; end
end, end, end
save(sprintf('/tmp/scan_tail_%s_%g.mat',form,f1),'rows','P','S0');
exit(0);
function s=tern(c,a,b), if c,s=a; else,s=b; end, end
