run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
ap='/home/dcr/dev/MACOS_res_dev/mmacos/challenges/afocal4';
addpath(ap); addpath(fullfile(ap,'descent')); addpath(fullfile(ap,'offaxis'));
P=afocal4_params(); macos.init(P.model_size);
fprintf(['\n  IS THE MERSENNE ERROR FIELD CURVATURE?  rung2 vs rung3 (power\n' ...
         '  removed).  cass and greg pairs have OPPOSITE-sign Petzval, so if\n' ...
         '  the residual is field curvature their defocus-vs-field slopes must\n' ...
         '  have opposite SIGN -- which is what a double Mersenne cancels.\n\n']);
fprintf('  %-32s %10s %10s %8s\n','deck','rung2 nm','rung3 nm','%% power');
lst={};
d=dir(fullfile(ap,'afocal4_mersenne*.in'));
for i=1:numel(d), lst{end+1}={d(i).name, fullfile(d(i).folder,d(i).name)}; end
dd=dir(fullfile(ap,'offaxis','decks','om_*_h1.5.in'));
for i=1:numel(dd), lst{end+1}={dd(i).name, fullfile(dd(i).folder,dd(i).name)}; end
for i=1:numel(lst)
  try
    S=afocal4_score(P,lst{i}{2},'fields',P.Fsolve,'nodes',P.solve.nodes_score,'pupil',false);
    fprintf('  %-32s %10.1f %10.1f %8.1f\n', lst{i}{1}, S.wfe_max_nm, ...
            S.wfe_rung3_max_nm, 100*(1-(S.wfe_rung3_max_nm/max(S.wfe_max_nm,eps))^2));
  catch ME, fprintf('  %-32s FAILED %s\n', lst{i}{1}, ME.message); end
end
% the sign test: per-field rung2-vs-rung3 gap across the field box, cass vs greg
fprintf('\n  DEFOCUS vs FIELD (per-field rung2, nm) -- the sign test\n');
for f={'om_cass_f5_h1.5.in','om_greg_f5_h1.5.in'}
  dk=fullfile(ap,'offaxis','decks',f{1});
  if ~isfile(dk), continue; end
  Fy=[-0.25 -0.125 0 0.125 0.25]; F=[zeros(5,1) deg2rad(Fy(:))];
  S=afocal4_score(P,dk,'fields',F,'nodes',P.solve.nodes_score,'pupil',false);
  fprintf('  %-24s %s\n', f{1}, sprintf('%10.1f', S.wfe_nm));
end
exit(0);
