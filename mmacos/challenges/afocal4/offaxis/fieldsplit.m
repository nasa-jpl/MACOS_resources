run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
ap='/home/dcr/dev/MACOS_res_dev/mmacos/challenges/afocal4';
addpath(ap); addpath(fullfile(ap,'descent')); addpath(fullfile(ap,'offaxis'));
addpath(fullfile(ap,'wall')); addpath(fullfile(ap,'clearing'));
P=afocal4_params(); macos.init(P.model_size);
fprintf(['\n  WHERE DOES THE WAVEFRONT ERROR LIVE?  rung-2 max, scored at\n' ...
         '  ONE field (box centre) vs the full 3x3 box.  If the centre is\n' ...
         '  already micron-class the error is APERTURE aberration and no\n' ...
         '  amount of field correction reaches it; if the centre is small\n' ...
         '  the whole problem is FIELD.\n\n']);
fprintf('  bias = %.3f deg; TRUE on-axis is the field offset (0, -bias).\n\n', P.bias_deg);
fprintf(['  RUNG 2 removes piston + per-field tip/tilt.  RUNG 3 also removes\n' ...
         '  POWER -- so rung2 minus rung3 IS the residual defocus, i.e. the\n' ...
         '  amount by which the real traced beam fails to be collimated.\n\n']);
fprintf('  %-30s %10s %10s %10s %10s %9s\n','deck','r2 bias','r3 bias', ...
        'r2 box','r3 box','coll urad');
D={'committed 4-mirror', fullfile(ap,'afocal4_b2long_343mm.in')};
lst = {D};
for N=[5 6 7]
  f=fullfile(ap,'descent',sprintf('descent_ASC_N%d.mat',N));
  if isfile(f)
    Z=load(f,'R'); Dd=Z.R.D; Dd.ngrid=P.ngrid; Dd.bias_deg=P.bias_deg;
    dk=sprintf('/tmp/fs_N%d.in',N); Pr=P; Pr.pack.enforce=false;
    descent_build(Pr,Dd,dk,'verify',false,'quiet',true);
    lst{end+1}={sprintf('descent ascent rung N=%d',N), dk};
  end
end
% and the best off-axis Mersenne from the sweep
mf=fullfile(ap,'offaxis','decks','om_cass_f5_h1.5.in');
if isfile(mf), lst{end+1}={'off-axis Mersenne f1=5 h=1.5', mf}; end
for i=1:numel(lst)
  nm=lst{i}{1}; dk=lst{i}{2};
  try
    Sc=afocal4_score(P,dk,'fields',[0 0],'nodes',P.solve.nodes_score,'pupil',false);
    Sb=afocal4_score(P,dk,'fields',P.Fsolve,'nodes',P.solve.nodes_score,'pupil',false);
    macos.load_rx(dk); tr=macos.trace(macos.num_elt()); ri=macos.get_ray_info(tr.nRays);
    ok=ri.ok_trace(:)&ri.ok_pass(:); ok(1)=false;
    dd=ri.dir(:,ok); dd=dd./vecnorm(dd); dm=mean(dd,2); dm=dm/norm(dm);
    cl=max(acos(min(1,dm.'*dd)))*1e6;
    fprintf('  %-30s %10.1f %10.1f %10.1f %10.1f %9.1f\n', nm, ...
            Sc.wfe_max_nm, Sc.wfe_rung3_max_nm, ...
            Sb.wfe_max_nm, Sb.wfe_rung3_max_nm, cl);
  catch ME
    fprintf('  %-34s FAILED %s\n', nm, ME.message);
  end
end
exit(0);
