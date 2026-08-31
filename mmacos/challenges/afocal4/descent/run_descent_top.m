% RUN_DESCENT_TOP  One top-of-ladder attempt at a given N, one process.
%   DESC_N      powered mirrors (7)
%   DESC_SEED   seed index among the compliant closures (1)
%   DESC_EVALS  evaluations per round (600)
%   DESC_ROUNDS restart rounds (3)
%   DESC_DOFS   comma list ("conic,spacing,tilt")
%   DESC_TAG    artifact suffix
run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
here = fileparts(mfilename('fullpath'));   up = fileparts(here);
addpath(here); addpath(up); addpath(fullfile(up,'clearing')); addpath(fullfile(up,'wall'));

dN   = str2double(getenv('DESC_N'));       if isnan(dN),   dN = 7;   end
dEv  = str2double(getenv('DESC_EVALS'));   if isnan(dEv),  dEv = 600; end
dRd  = str2double(getenv('DESC_ROUNDS'));  if isnan(dRd),  dRd = 3;  end
dTag = getenv('DESC_TAG');                 if isempty(dTag), dTag = sprintf('N%d',dN); end
dDof = getenv('DESC_DOFS');
if isempty(dDof), dofs = {'conic','spacing','tilt'}; else, dofs = strsplit(dDof,','); end

macos.init(256);
P = afocal4_params();
P.pack.enforce = true;                 % the S4b station, on
P.solve.fd_type = 'central';  P.solve.fd_step = 1e-4;
P.solve.tol_fun = 1e-8;  P.solve.tol_x = 1e-9;  P.solve.tol_opt = 1e-8;
P.solve.max_fev = dEv;

fprintf('\n==== DESCENT TOP  N = %d  [%s] ====\n', dN, dTag);
[S0, si] = descent_seed(P, dN, 'quiet',false);
if ~si.ok
    error('run_descent_top:seed', 'no compliant %d-mirror seed.', dN);
end
D = S0;  D.tilt_deg = zeros(1,dN);  D.ngrid = P.ngrid;  D.bias_deg = P.bias_deg;

deck = fullfile(here, sprintf('afocal4_descent_%s.in', dTag));
Dc = D;  mprev = Inf;  rounds = struct('k',{},'exitflag',{},'nfev',{},'merit',{},'gain',{});
for k = 1:dRd
    try
        Rk = descent_solve(P, Dc, 'dofs',dofs, 'deck',deck, ...
                           'label',sprintf('round %d', k), 'max_iter',400, ...
                           'quiet',true);
    catch ME
        fprintf('  round %d FAILED (%s): %s\n', k, ME.identifier, ME.message);
        if k == 1, rethrow(ME); end
        break;
    end
    m = Rk.S.merit;
    if isfinite(mprev), g = (mprev-m)/max(abs(mprev),eps); else, g = Inf; end
    rounds(end+1) = struct('k',k,'exitflag',Rk.exitflag,'nfev',Rk.nfev, ...
                           'merit',m,'gain',g); %#ok<SAGROW>
    fprintf('  round %d: %4d evals, %6.1f min, exitflag %d, merit %.6f (gain %.2e)\n', ...
            k, Rk.nfev, Rk.seconds/60, Rk.exitflag, m, g);
    Dc = Rk.D;  mprev = m;
    if Rk.exitflag == 1, fprintf('  -> optimality reached\n'); break; end
    if k > 1 && g < 1e-6, fprintf('  -> plateau\n'); break; end
end

Dr = Dc;  Dr.ngrid = P.ngrid;
Pr = P;   Pr.pack.enforce = false;
descent_build(Pr, Dr, deck, 'verify',false);
Q = descent_require(P, deck);
R = struct('N',dN, 'tag',dTag, 'seed',si, 'D',Dc, 'rounds',rounds, 'Q',Q, ...
           'deck',deck, 'dofs',{dofs}); %#ok<NASGU>
save(fullfile(here, sprintf('descent_%s.mat', dTag)), 'R', '-v7.3');
fprintf('\n  wrote %s\n', fullfile(here, sprintf('descent_%s.mat', dTag)));
exit(0);
