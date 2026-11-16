% RUN_DESCENT_ASCENT  Build the top of the ladder UP from the committed
% four-mirror design, one mirror at a time, each step warm-started.
%
%   ASC_TO     element count to build up to (7)
%   ASC_EVALS  evaluations per round (600)
%   ASC_ROUNDS restart rounds per rung (2)
%   ASC_AT     where to insert, as a free-element index (3)
%   ASC_TAG    artifact prefix ('ASC')
run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
here = fileparts(mfilename('fullpath'));   up = fileparts(here);
addpath(here); addpath(up); addpath(fullfile(up,'clearing')); addpath(fullfile(up,'wall'));

aTo  = str2double(getenv('ASC_TO'));      if isnan(aTo),  aTo = 7;  end
aEv  = str2double(getenv('ASC_EVALS'));   if isnan(aEv),  aEv = 600; end
aRd  = str2double(getenv('ASC_ROUNDS'));  if isnan(aRd),  aRd = 2;  end
aAt  = str2double(getenv('ASC_AT'));      if isnan(aAt),  aAt = 3;  end
aTag = getenv('ASC_TAG');                 if isempty(aTag), aTag = 'ASC'; end

macos.init(256);
P = afocal4_params();
P.pack.enforce = true;
P.solve.fd_type='central'; P.solve.fd_step=1e-4;
P.solve.tol_fun=1e-8; P.solve.tol_x=1e-9; P.solve.tol_opt=1e-8;
P.solve.max_fev = aEv;

% ---- the parent: the committed four-mirror design, in N-mirror form ----
src = fullfile(up,'afocal4_b2long_343mm.in');
D0  = wall_recover(P, src);
fo  = afocal_first_order([abs(P.parent.R(1)) D0.R2], D0.t1, [false true], ...
                         'D',P.D, 'stop_ahead',P.stop_ahead);
a   = -fo.y_marginal(2)/fo.u_marginal(2) - D0.fm_standoff;
S = struct('N',4, 'R',[abs(P.parent.R(1)) D0.R2], 'convex',[false true], ...
           't',[D0.t1 a], 'iface',D0.iface, 'K',D0.K(:).');
C = descent_close(P, S);
fprintf('\n==== ASCENT from the committed 4-mirror design up to N = %d ====\n', aTo);
fprintf('  parent: powered %d, behind M1 %+.3f m, resid %.1e\n', ...
        C.n_powered, C.behind_m1, max(abs(C.residual)));

D = S;  D.tilt_deg = zeros(1,4);  D.ngrid = P.ngrid;  D.bias_deg = P.bias_deg;
rungs = struct('N',{},'merit',{},'wfe',{},'blur',{},'behind',{},'ok',{},'deck',{});
for Ntarget = 5:aTo
    fprintf('\n  ---- adding a mirror: %d -> %d elements ----\n', D.N, Ntarget);
    Sp = struct('N',D.N,'R',D.R,'convex',D.convex,'t',D.t,'iface',D.iface,'K',D.K);
    [S2, ai] = descent_add(P, Sp, min(aAt,D.N-2), 'search',true);
    if isempty(S2) || ~ai.ok
        fprintf('  %s\n  stopping.\n', ai.why);
        break;
    end
    kk = ai.k;
    fprintf(['  inserted after free element %d at split %.2f: elements %d, ' ...
             'behind M1 %+.3f m (%d of %d closures compliant, warmth %.3f)\n'], ...
            kk, ai.split, ai.n_elements, ai.behind_m1, ai.n_compliant, ...
            ai.n_closed, ai.warmth);
    D2 = S2;  D2.tilt_deg = [D.tilt_deg(1:kk), 0, D.tilt_deg(kk+1:end)];
    D2.ngrid = P.ngrid;  D2.bias_deg = P.bias_deg;
    tag  = sprintf('%s_N%d', aTag, Ntarget);
    deck = fullfile(here, sprintf('afocal4_descent_%s.in', tag));
    Dc = D2;  mprev = Inf;
    for r = 1:aRd
        try
            Rr = descent_solve(P, Dc, 'dofs',{'conic','radius','spacing','tilt'}, ...
                     'deck',deck, 'label',sprintf('N%d round %d',Ntarget,r), ...
                     'max_iter',400, 'quiet',true);
        catch ME
            fprintf('  N%d round %d FAILED: %s\n', Ntarget, r, ME.message);
            break;
        end
        m = Rr.S.merit;
        if isfinite(mprev), g=(mprev-m)/max(abs(mprev),eps); else, g=Inf; end
        fprintf('  N%d round %d: %4d evals, %5.1f min, xfl %d, merit %.4f (gain %.2e)\n', ...
                Ntarget, r, Rr.nfev, Rr.seconds/60, Rr.exitflag, m, g);
        Dc = Rr.D;  mprev = m;
        if Rr.exitflag == 1 || (r>1 && g < 1e-6), break; end
    end
    Pr = P;  Pr.pack.enforce = false;
    Dr = Dc; Dr.ngrid = P.ngrid;
    descent_build(Pr, Dr, deck, 'verify',false);
    Q = descent_require(P, deck, 'quiet',true);
    fprintf('  N%d: WFE %.1f nm, blur %.1f um, M err %.4f %%, floor %+.2f mm -- targets %s\n', ...
            Ntarget, Q.rows(1).value, Q.rows(2).value, Q.rows(6).value, ...
            Q.floor_mm, tern(Q.ok,'MET','missed'));
    rungs(end+1) = struct('N',Ntarget,'merit',mprev,'wfe',Q.rows(1).value, ...
        'blur',Q.rows(2).value,'behind',Q.z.behind_m1,'ok',Q.ok,'deck',deck); %#ok<SAGROW>
    R = struct('N',Ntarget,'tag',tag,'D',Dc,'Q',Q,'deck',deck,'add',ai); %#ok<NASGU>
    save(fullfile(here, sprintf('descent_%s.mat', tag)), 'R', '-v7.3');
    D = Dc;
end
save(fullfile(here, sprintf('descent_%s_summary.mat', aTag)), 'rungs', '-v7.3');
fprintf('\n  ascent complete: %d rungs\n', numel(rungs));
exit(0);

function s = tern(c,a,b), if c, s=a; else, s=b; end, end
