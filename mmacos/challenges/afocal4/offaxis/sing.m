run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
ap='/home/dcr/dev/MACOS_res_dev/mmacos/challenges/afocal4';
addpath(ap); addpath(fullfile(ap,'descent')); addpath(fullfile(ap,'offaxis'));
P=afocal4_params(); macos.init(P.model_size);
for form={'cass','greg'}
 S=offaxis_seed(P,form{1},'N',5,'f1',1.25);
 C=descent_close(P,struct('N',5,'R',S.R,'convex',S.convex,'t',S.t, ...
       'iface',S.iface,'K',S.K),'window',[-1.5 9],'npts',241);
 fprintf('\n  %s  found %d  resid [%.2e %.2e %.2e]\n', upper(form{1}), ...
         C.found, C.residual);
 fprintf('    R  = %s\n', mat2str(round(C.R,4)));
 fprintf('    t  = %s\n', mat2str(round(C.t,4)));
 fprintf('    phi= %s  /m\n', mat2str(round(C.phi,4)));
 fprintf('    paraxial mag %.6f, u_out %.3e\n', C.fo.mag, C.fo.u_out);
 % marginal state after the FREE mirrors, by hand from the same recipe
 y=P.D/2; u=0;
 for k=1:numel(S.R)
   phi = 2/abs(C.R(k)); if C.convex(k), phi=-phi; end
   u = u - y*phi;  y = y + C.t(k)*u;
   fprintf('    after free mirror %d: y %10.6f m, u %11.6f\n',k,y,u);
 end
 fprintf('    yout target = %.6f m ;  numerator (yout-ym) = %.3e\n', ...
         (P.D/2)/P.M, (P.D/2)/P.M - y);
end
exit(0);
