% RUN_OFFAXIS_RESTART  Is the off-axis gain REAL, or is it re-seeding?
%
%   THE CONFOUND THIS EXISTS TO KILL.  Section O.7e leaves two readings of the
%   same three rows standing:
%
%     (1) a decentered pupil genuinely reaches a better design, or
%     (2) decentering merely RE-SEEDS the solver -- it perturbs the starting
%         point, the solver lands in a different basin, and on a rough
%         landscape a different basin is sometimes better and sometimes worse.
%
%   Reading (2) predicts the whole pattern: two arms improved, one degraded,
%   and the off-axis results clustered more tightly than the coaxial ones.  It
%   also predicts the N = 5 result, which was the largest gain and whose
%   control section D.3 had already flagged as a basin-scatter outlier.
%
%   THE DISCRIMINATOR IS A COAXIAL MULTI-START, and it is the control the
%   experiment was missing.  Perturb the STARTING DESIGN, solve at h = 0, and
%   collect the spread of floors:
%
%     * if jittered COAXIAL starts span the same range the off-axis arms did
%       (2847-4546 nm), then the decenter bought nothing a random kick would
%       not have bought, and reading (2) stands;
%     * if they cluster near the unjittered 4497.7 and none of them reaches
%       2847, then the decenter is doing something a perturbation cannot, and
%       reading (1) survives.
%
%   WHAT IS JITTERED, AND WHY IT IS THE RIGHT KNOB.  The free radii and
%   spacings, by a few per cent -- NOT the conics alone, because a conic is a
%   pure aberration knob that leaves the layout untouched, and the point is to
%   move the solver's STARTING POINT in the same coarse way a 0.55 m decenter
%   does.  The first-order closure is re-imposed exactly by DESCENT_CLOSE at
%   every iterate regardless, so a jittered start is still an exactly afocal,
%   exactly 30x, exactly pupil-correct design: the jitter moves where the
%   solver begins, never what it is required to deliver.
%
%   SEEDED, so the control is reproducible: rng(OR_SEED + k).
%
%   OR_N       mirror count (4 -- the arm whose control reproduces D.3 and
%              whose off-axis result is the headline)
%   OR_STARTS  how many jittered starts (3)
%   OR_JITTER  fractional sigma on free radii and spacings (0.03)
%   OR_SEED    base RNG seed (1000)
%   OR_EVALS / OR_ROUNDS  as the other runners (400 / 2)
run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
here = fileparts(mfilename('fullpath'));   up = fileparts(here);
addpath(here); addpath(up); addpath(fullfile(up,'clearing'));
addpath(fullfile(up,'wall')); addpath(fullfile(up,'descent'));

N    = str2double(getenv_d('OR_N','4'));
nst  = str2double(getenv_d('OR_STARTS','3'));
jit  = str2double(getenv_d('OR_JITTER','0.03'));
seed = str2double(getenv_d('OR_SEED','1000'));
wEv  = str2double(getenv_d('OR_EVALS','400'));
wRd  = str2double(getenv_d('OR_ROUNDS','2'));

macos.init(256);
P = afocal4_params();
P.pack.enforce = true;
P.solve.fd_type='central';  P.solve.fd_step=1e-4;
P.solve.tol_fun=1e-8;  P.solve.tol_x=1e-9;  P.solve.tol_opt=1e-8;
P.solve.max_fev = wEv;
dofs = {'conic','radius','spacing','tilt'};

% the same starting design the isolation experiment used, unjittered
src = fullfile(up,'afocal4_b2long_343mm.in');
D0  = wall_recover(P, src);
fo  = afocal_first_order([abs(P.parent.R(1)) D0.R2], D0.t1, [false true], ...
                         'D',P.D, 'stop_ahead',P.stop_ahead);
a   = -fo.y_marginal(2)/fo.u_marginal(2) - D0.fm_standoff;
Dbase = struct('N',4, 'R',[abs(P.parent.R(1)) D0.R2], 'convex',[false true], ...
               't',[D0.t1 a], 'K',D0.K(:).', 'iface',D0.iface, ...
               'tilt_deg',zeros(1,4), 'ngrid',P.ngrid, 'bias_deg',P.bias_deg);

fprintf('\n==== COAXIAL MULTI-START: is the off-axis gain re-seeding? ====\n');
fprintf(['  %d jittered starts at h = 0, sigma %.1f %% on free radii and\n' ...
         '  spacings.  Reference points: unjittered coaxial floor 4497.7 nm;\n' ...
         '  off-axis arms spanned 2846.9 - 4545.5 nm.\n\n'], nst, jit*100);
fprintf('  %-8s %12s %12s %10s %9s\n','start','start nm','floor nm','x target','M');

rows = struct('k',{},'wfe0',{},'wfe',{},'M',{},'deck',{},'R',{},'t',{});
for k = 0:nst
    D = Dbase;
    if k > 0
        rng(seed + k);
        D.R = Dbase.R .* (1 + jit*randn(size(Dbase.R)));
        D.t = Dbase.t .* (1 + jit*randn(size(Dbase.t)));
    end
    lbl = sprintf('J%d', k);
    d0  = fullfile(here, sprintf('afocal4_RST_%s_start.in', lbl));
    Pr  = P;   Pr.pack.enforce = false;
    try
        descent_build(Pr, D, d0, 'verify',false, 'quiet',true);
        S0 = afocal4_score(P, d0, 'fields',P.Fsolve, ...
                           'nodes',P.solve.nodes_score, 'pupil',false);
    catch ME
        fprintf('  %-8s BUILD/SCORE FAILED: %s\n', lbl, ME.message);   continue;
    end

    deck = fullfile(here, sprintf('afocal4_RST_%s.in', lbl));
    Dc = D;   mprev = Inf;
    for r = 1:wRd
        try
            Rr = descent_solve(P, Dc, 'dofs',dofs, 'deck',deck, ...
                     'pupil',false, 'max_iter',400, ...
                     'label',sprintf('%s r%d',lbl,r), 'quiet',true);
        catch ME
            fprintf('  %-8s round %d FAILED: %s\n', lbl, r, ME.message);   break;
        end
        m = Rr.S.merit;   Dc = Rr.D;
        if isfinite(mprev) && (mprev-m)/max(abs(mprev),eps) < 1e-6, break; end
        mprev = m;
    end
    Dr = Dc;   Dr.ngrid = P.ngrid;
    o1 = descent_build(Pr, Dr, deck, 'verify',true, 'quiet',true);
    S1 = afocal4_score(P, deck, 'fields',P.Fsolve, ...
                       'nodes',P.solve.nodes_score, 'pupil',false);
    fprintf('  %-8s %12.1f %12.1f %10.0f %9.4f%s\n', lbl, S0.wfe_max_nm, ...
            S1.wfe_max_nm, S1.wfe_max_nm/71, o1.traced.mag, ...
            tern_(k==0,'   (unjittered reference)',''));
    rows(end+1) = struct('k',k,'wfe0',S0.wfe_max_nm,'wfe',S1.wfe_max_nm, ...
        'M',o1.traced.mag,'deck',deck,'R',D.R,'t',D.t); %#ok<SAGROW>
    save(fullfile(here,'offaxis_restart.mat'),'rows','P','-v7.3');
end

% ---- the verdict, stated against the off-axis range ---------------------
if numel(rows) >= 3
    w = [rows.wfe];   jr = w(2:end);
    fprintf('\n  coaxial jittered floors: min %.1f, max %.1f nm (n = %d)\n', ...
            min(jr), max(jr), numel(jr));
    fprintf('  off-axis arms spanned  : 2846.9 - 4545.5 nm\n');
    reaches = min(jr) <= 2846.9*1.05;
    fprintf(['\n  %s\n'], tern_(reaches, ...
      ['RE-SEEDING SUFFICES: a coaxial jitter reaches the off-axis best, so ' ...
       'the decenter bought nothing a random kick would not have.'], ...
      ['RE-SEEDING DOES NOT SUFFICE: no coaxial jitter reached the off-axis ' ...
       'best, so the decenter is doing something a perturbation cannot.']));
end
fprintf('\n');
exit(0);

function v = getenv_d(k,d), v = getenv(k); if isempty(v), v = d; end, end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
