run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
ap='/home/dcr/dev/MACOS_res_dev/mmacos/challenges/afocal4';
addpath(ap); addpath(fullfile(ap,'descent')); addpath(fullfile(ap,'offaxis'));
P=afocal4_params(); macos.init(P.model_size);
fprintf('\n  packaging wall: last powered mirror >= %+.0f mm behind M1\n', ...
        P.pack.m3_behind_min*1e3);
for f1 = [1.25 2.5]
  for form = {'cass','greg'}
    S0 = offaxis_seed(P, form{1}, 'N',4, 'f1',f1);
    fprintf('\n  --- %s f1 %.3f (sep %.4f) ---\n', upper(form{1}), f1, S0.sep);
    best = [];
    for t2 = [0.3 0.6 1.0 1.5 2.0 2.5 3.0 3.5 4.0 5.0 6.0]
      S = S0; S.t(2) = t2;
      try
        C = descent_close(P, struct('N',4,'R',S.R,'convex',S.convex, ...
              't',S.t,'iface',S.iface,'K',S.K), 'window',[-1.5 9],'npts',241);
      catch ME, fprintf('    t2 %5.2f  closure error %s\n',t2,ME.identifier); continue; end
      if ~isfield(C,'found')||~C.found, fprintf('    t2 %5.2f  no root\n',t2); continue; end
      fprintf(['    t2 %5.2f  t3 %8.4f  behind_m1 %+9.1f mm  tmin %7.1f mm  ' ...
               'R3 %9.3f R4 %9.4f  %s\n'], t2, C.t(3), C.behind_m1*1e3, ...
               min(abs(diff(C.z)))*1e3, C.R(3), C.R(4), ...
               ternx(C.behind_m1>=P.pack.m3_behind_min,'COMPLIANT',''));
      if C.behind_m1>=P.pack.m3_behind_min && isempty(best), best=t2; end
    end
    if isempty(best), fprintf('    -> no compliant t2 on this grid\n');
    else, fprintf('    -> first compliant t2 = %.2f m\n', best); end
  end
end
exit(0);
function s=ternx(c,a,b), if c, s=a; else, s=b; end, end
