% RUN_DESCENT_RUNG  One rung DOWN from a parent checkpoint, one process.
%
%   RUNG_FROM    parent checkpoint tag (e.g. 'N7a')
%   RUNG_K       which powered mirror to remove
%   RUNG_MODE    'retain' | 'delete'
%   RUNG_EVALS   evaluations per round (600)
%   RUNG_ROUNDS  restart rounds (3)
%   RUNG_DOFS    comma list ("conic,spacing,tilt")
%   RUNG_TAG     artifact suffix
%
%   The walk: take the parent's CONVERGED design, remove one mirror by the
%   named mechanism, re-solve warm-started from what is left.  Every rung is
%   a finished, checked design -- so if the time box runs out mid-ladder the
%   completed rungs stand (the walk doctrine's virtue, and the descent
%   brief's instruction).
run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
here = fileparts(mfilename('fullpath'));   up = fileparts(here);
addpath(here); addpath(up); addpath(fullfile(up,'clearing')); addpath(fullfile(up,'wall'));

rFrom = getenv('RUNG_FROM');   if isempty(rFrom), error('set RUNG_FROM'); end
rK    = str2double(getenv('RUNG_K'));
rMode = getenv('RUNG_MODE');   if isempty(rMode), rMode = 'retain'; end
rEv   = str2double(getenv('RUNG_EVALS'));   if isnan(rEv),  rEv = 600; end
rRd   = str2double(getenv('RUNG_ROUNDS'));  if isnan(rRd),  rRd = 3;   end
rTag  = getenv('RUNG_TAG');
rDof  = getenv('RUNG_DOFS');
if isempty(rDof), dofs = {'conic','spacing','tilt'}; else, dofs = strsplit(rDof,','); end

macos.init(256);
P = afocal4_params();
P.pack.enforce = true;
P.solve.fd_type='central'; P.solve.fd_step=1e-4;
P.solve.tol_fun=1e-8; P.solve.tol_x=1e-9; P.solve.tol_opt=1e-8;
P.solve.max_fev = rEv;

Z = load(fullfile(here, sprintf('descent_%s.mat', rFrom)), 'R');
Dp = Z.R.D;
fprintf('\n==== DESCENT RUNG  from %s (elements %d), remove mirror %d by %s ====\n', ...
        rFrom, Dp.N, rK, upper(rMode));

% the parent's own spec, then the removal
Sp = struct('N',Dp.N, 'R',Dp.R, 'convex',Dp.convex, 't',Dp.t, ...
            'iface',Dp.iface, 'K',Dp.K);
if isfield(Dp,'n_flat'), Sp.n_flat = Dp.n_flat; end
if isfield(Dp,'flat_at'), Sp.flat_at = Dp.flat_at; end
[S2, ri] = descent_remove(P, Sp, rK, rMode, 'allow', 3:(Dp.N-2));
fprintf('  %s: elements %d -> %d, powered %d, flats %d; parity flips: %d\n', ...
        ri.mode, Dp.N, S2.N, ri.n_powered, ri.n_flat, ri.parity_flips);
fprintf('  (%s)\n', ri.why);

C2 = descent_close(P, S2);
if ~isfield(C2,'found') || ~C2.found
    fprintf(['\n  THE REMOVAL DOES NOT CLOSE at all: no penultimate power puts ' ...
             'the exit pupil at %.0f mm.\n  That is a LADDER DATUM (the rung ' ...
             'has no first-order solution by this mechanism), not a solver ' ...
             'failure.\n'], S2.iface*1e3);
    R = struct('from',rFrom, 'k',rK, 'mode',rMode, 'removal',ri, 'ok',false, ...
               'why','no closure after removal', 'tag',rTag); %#ok<NASGU>
    save(fullfile(here, sprintf('descent_%s.mat', rTag)), 'R', '-v7.3');
    exit(0);
end
fprintf('  closes: behind M1 %+.3f m, resid %.1e, powered %d\n', ...
        C2.behind_m1, max(abs(C2.residual)), C2.n_powered);
compliant = C2.behind_m1 >= P.pack.m3_behind_min;
fprintf('  packaging station: %s (%.0f mm against a %.0f mm minimum)\n', ...
        tern(compliant,'CLEARS','FAILS'), C2.behind_m1*1e3, P.pack.m3_behind_min*1e3);

D = S2;
D.tilt_deg = zeros(1, S2.N);
if isfield(Dp,'tilt_deg')
    n = min(numel(Dp.tilt_deg), S2.N);
    if strcmp(rMode,'delete')
        tl = Dp.tilt_deg;  tl(rK) = [];   D.tilt_deg = tl;
    else
        D.tilt_deg(1:n) = Dp.tilt_deg(1:n);
    end
end
D.ngrid = P.ngrid;   D.bias_deg = P.bias_deg;

deck = fullfile(here, sprintf('afocal4_descent_%s.in', rTag));
Dc = D;  mprev = Inf;  rounds = struct('k',{},'exitflag',{},'nfev',{},'merit',{},'gain',{});
for k = 1:rRd
    try
        Rk = descent_solve(P, Dc, 'dofs',dofs, 'deck',deck, ...
                           'label',sprintf('round %d',k), 'max_iter',400, 'quiet',true);
    catch ME
        fprintf('  round %d FAILED (%s): %s\n', k, ME.identifier, ME.message);
        if k == 1
            R = struct('from',rFrom,'k',rK,'mode',rMode,'removal',ri,'ok',false, ...
                       'why',ME.message,'tag',rTag); %#ok<NASGU>
            save(fullfile(here, sprintf('descent_%s.mat',rTag)),'R','-v7.3');
            exit(0);
        end
        break;
    end
    m = Rk.S.merit;
    if isfinite(mprev), g=(mprev-m)/max(abs(mprev),eps); else, g=Inf; end
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
R = struct('from',rFrom, 'k',rK, 'mode',rMode, 'removal',ri, 'ok',true, ...
           'why','', 'N',S2.N, 'n_powered',ri.n_powered, 'n_flat',ri.n_flat, ...
           'D',Dc, 'rounds',rounds, 'Q',Q, 'deck',deck, 'dofs',{dofs}, ...
           'tag',rTag); %#ok<NASGU>
save(fullfile(here, sprintf('descent_%s.mat', rTag)), 'R', '-v7.3');
fprintf('\n  wrote %s\n', fullfile(here, sprintf('descent_%s.mat', rTag)));
exit(0);

function s = tern(c,a,b), if c, s=a; else, s=b; end, end
