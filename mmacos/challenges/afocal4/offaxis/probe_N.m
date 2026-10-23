run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
ap='/home/dcr/dev/MACOS_res_dev/mmacos/challenges/afocal4';
addpath(ap); addpath(fullfile(ap,'descent')); addpath(fullfile(ap,'offaxis'));
P=afocal4_params(); macos.init(P.model_size);
fprintf('\n  wall: behind_m1 >= %+.0f mm ; min spacing 20 mm\n\n', P.pack.m3_behind_min*1e3);
fprintf('  %-5s %-5s %6s %7s %7s %10s %9s %9s\n', ...
        'form','f1','N','t2','t3','behind mm','tmin mm','ok');
for form={'cass','greg'}
 for f1=[0.75 1.25 2.50]
  for N=[4 5 6]
   S0 = offaxis_seed(P, form{1}, 'N',N, 'f1',f1);
   grid2 = [0.3 0.6 1.0 1.5 2.0]; if N==4, grid2=1.0; end
   for t2=grid2
    for t3=[0.3 0.6 1.0 1.5 2.0]
      S=S0; if numel(S.t)>=2, S.t(2)=t2; end
      if numel(S.t)>=3, S.t(3)=t3; elseif t3~=0.3, continue; end
      try
        C=descent_close(P,struct('N',N,'R',S.R,'convex',S.convex,'t',S.t, ...
              'iface',S.iface,'K',S.K),'window',[-1.5 9],'npts',241);
      catch, continue; end
      if ~isfield(C,'found')||~C.found, continue; end
      tmin=min(abs(diff(C.z)))*1e3; ok = C.behind_m1>=P.pack.m3_behind_min && tmin>=20;
      if ok || (N==4&&t3==0.3) || (N>4&&t2==1.0&&t3==1.0)
        fprintf('  %-5s %5.2f %6d %7.2f %7.2f %10.1f %9.1f %9s\n', ...
          form{1},f1,N,t2,t3,C.behind_m1*1e3,tmin,ternx(ok,'COMPLIANT',''));
      end
    end
   end
  end
 end
end
exit(0);
function s=ternx(c,a,b), if c,s=a; else,s=b; end, end
