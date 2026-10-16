function R = afocal4_basin2(opts)
%AFOCAL4_BASIN2  Basin 2, solved long: is 10.5 um a design or a stall?
%
%   R = AFOCAL4_BASIN2() re-solves the second S4b basin -- the one with the
%   intermediate image behind M1 and the field mirror back on it -- at every
%   point of the interface-standoff trade, from three independent seeds, with
%   the solver settings the S4b sweep did NOT have.  It exists because that
%   basin was hand-seeded and every one of its solves stopped on `exitflag 3`
%   at ~11 iterations, and its wavefront column (10.5-17 um) is suspiciously
%   flat across a curve on which nothing else is flat.  A caveat stated in
%   S4b; a measurement here.
%
%   WHY THE OLD SOLVES STOPPED, MEASURED BEFORE ANYTHING WAS CHANGED.  The
%   merit is smooth in the DOFs: at the 343 mm basin-2 design the
%   central-difference slope in K_M2 is -1.4490 at a 3e-3 scaled step and
%   -1.4490 at 1e-5, i.e. four figures over two decades, with no noise floor
%   in between.  The study's default FORWARD difference at 3e-3 reads -1.198
%   -- 17% low -- and that error, not the objective's shape, is what stalls
%   lsqnonlin on its FunctionTolerance.  The slope itself is the finding:
%   -1.45 is not a stationary point, so the point the sweep delivered was
%   never a minimum of anything.
%
%   WHAT THIS DOES DIFFERENTLY, and nothing else:
%     * forward differences at a 3e-4 scaled step (1% gradient error, from
%       the probe above) instead of 3e-3;
%     * FunctionTolerance 1e-8 and StepTolerance 1e-9 instead of 1e-4/1e-6,
%       so `exitflag 3` has to be earned;
%     * RESTART rounds -- a fresh trust region from the converged point,
%       repeated until the solver returns exitflag 1 or a round buys less
%       than 'plateau' of the merit;
%     * a THIRD, independent seed: the second compliant closure of the
%       image-behind-M1 front end (a different field-mirror branch, not a
%       neighbouring radius);
%     * and, on the winner, a central-difference polish round followed by an
%       explicit GRADIENT PROBE, so "converged" and "plateau" are told apart
%       by a number rather than by an exit code.
%   The MERIT, the sampling and the DOF set are untouched: same
%   afocal4_score, same P.solve.ngrid / nodes, same {conic, standoff, front}.
%   A long solve that changed the objective would not be comparable to the
%   curve it is correcting.
%
%   Standard gates, all printed: the anchoring residual (a solver-integrity
%   check -- 0.1 um on a sound design, tens of mm on a scrambled one), the
%   two-plus-seed basin report, and the paraxial verification ON AXIS of
%   every delivered design.
%
%   Name-value:
%     'iface'   which trade points (default P.iface_trade)
%     'seeds'   subset of {'converged','fresh','second'} (all three)
%     'rounds'  restart rounds per seed (2)
%     'evals'   function evaluations per round (250)
%     'plateau' relative merit gain below which a restart round counts as a
%               plateau rather than progress (1e-6)
%     'polish'  central-difference round on the winner (true)
%     'tag'     suffix for the .mat / .in artifacts, so one process per
%               trade point can run in parallel and be merged after
%     'probe'   do not solve: re-open this tag's .mat and re-interrogate the
%               designs it delivered -- the gradient, and the merit along
%               -g, which is what separates a narrow valley from a wall
%     'save'    write .in / .mat (true)
%
%   Run:  >> afocal4_basin2                                  (all 5, hours)
%         >> afocal4_basin2('iface',0.343,'tag','343mm')     (one point)
%   Merge:   afocal4_basin2_merge (below) collects the per-tag .mat files.
%
%   See also AFOCAL4_S4B, AFOCAL4_SOLVE, AFOCAL4_SCORE, AFOCAL4_PHI4.

    arguments
        opts.iface   (1,:) double  = []
        opts.seeds   (1,:) cell    = {'converged','fresh','second'}
        opts.rounds  (1,1) double  = 2
        opts.evals   (1,1) double  = 250
        opts.plateau (1,1) double  = 1e-6
        opts.polish  (1,1) logical = true
        opts.tag     (1,:) char    = ''
        opts.probe   (1,1) logical = false
        opts.save    (1,1) logical = true
    end
    here = fileparts(mfilename('fullpath'));
    P = afocal4_params();
    if isempty(opts.iface), opts.iface = P.iface_trade; end
    macos.init(P.model_size);

    sfx = '';
    if ~isempty(opts.tag), sfx = ['_' opts.tag]; end
    matf = fullfile(here, sprintf('afocal4_basin2%s.mat', sfx));
    R = struct('P',P, 'when',datestr(now,31), 'opts',opts, 'pt',[]); %#ok<TNOW1,DATST>

    if opts.probe
        R = probe_only_(matf, P, opts.save);   return;
    end

    % the S4b record: its trade2 designs are seed 'converged'
    prior = [];
    f4 = fullfile(here,'afocal4_s4b.mat');
    if isfile(f4)
        q = load(f4);
        if isfield(q.R,'trade2'), prior = q.R.trade2; end
    end

    % ONE field list, declared once and built once (RESULTS rule 7: `arr(k) =
    % s` on dissimilar structs fails only when REACHED, i.e. after the
    % expensive part is already spent).
    PT = struct('iface',{},'seed',{},'rounds',{},'D',{},'S',{},'merit',{}, ...
                'exitflag',{},'nfev',{},'seconds',{},'all',{},'grad',{}, ...
                'build',{},'pack',{},'onaxis',{},'deck',{});
    for iq = 1:numel(opts.iface)
        q = opts.iface(iq);
        banner_(sprintf('BASIN 2 -- interface standoff %.0f mm', q*1e3));
        cand = seeds_(P, q, opts.seeds, prior);
        res = struct('seed',{},'D',{},'S',{},'merit',{},'rounds',{}, ...
                     'exitflag',{},'nfev',{},'seconds',{});
        for j = 1:numel(cand)
            fprintf('\n  --- seed "%s" ---------------------------------\n', ...
                    cand(j).name);
            try
                r = long_solve_(P, here, cand(j), q, opts);
            catch ME
                fprintf('  seed "%s" FAILED: %s\n', cand(j).name, ME.message);
                continue;
            end
            res(end+1) = r; %#ok<AGROW>
            if opts.save, R.pt = pack_(PT, res, q);  save(matf,'R','-v7.3'); end
        end
        if isempty(res)
            fprintf('  iface %.0f mm: NO POINT from any seed\n', q*1e3);
            continue;
        end

        % ---- the winner, and the basin report ---------------------------
        [~,kb] = min([res.merit]);
        [~,kw] = min(arrayfun(@(r) r.S.worst, res));
        fprintf('\n  BASIN REPORT, iface %.0f mm (%d seeds)\n', q*1e3, numel(res));
        fprintf('  %-12s %10s %9s %9s %9s %9s %9s %6s %6s\n', 'seed', ...
                'merit','WFE nm','blur um','breathe%','wander um','anchor um', ...
                'exit','nfev');
        for j = 1:numel(res)
            S = res(j).S;
            fprintf('  %-12s %10.4f %9.1f %9.1f %9.4f %9.1f %9.4f %6d %6d%s\n', ...
                res(j).seed, res(j).merit, S.wfe_max_nm, S.blur_um, ...
                S.breathe_pct, S.wander_um, S.anchor_resid_um, ...
                res(j).exitflag, res(j).nfev, tick_(j==kb));
        end
        sp = spread_(res);
        if sp.n_sound < 2
            fprintf(['  seed-to-seed spread: NOT COMPUTABLE -- %d of %d seeds ' ...
                     'reached a sound design\n'], sp.n_sound, numel(res));
        else
            fprintf(['  seed-to-seed spread over the %d SOUND seeds: merit ' ...
                     '%.2f%%, WFE %.2f%% -- %s%s\n'], sp.n_sound, ...
                     100*sp.merit, 100*sp.wfe, ...
                     ternary_(sp.merit < 0.01, 'ONE basin', 'MORE THAN ONE basin'), ...
                     ternary_(sp.n_invalid > 0, sprintf([' (%d seed(s) ' ...
                        'excluded: anchoring residual says no design)'], ...
                        sp.n_invalid), ''));
        end
        if kw ~= kb
            fprintf(['  NOTE: ranking by merit picks "%s", by worst-miss ' ...
                     '"%s" -- both reported\n'], res(kb).seed, res(kw).seed);
        end

        % ---- polish + the gradient probe, on the winner -----------------
        w = res(kb);
        if opts.polish
            fprintf('\n  --- central-difference polish on "%s" ---\n', w.seed);
            Pc = tighten_(P, 1e-4, 'central', opts.evals);
            dk = fullfile(here, sprintf('afocal4_b2long_%03.0fmm.in', q*1e3));
            if ~opts.save, dk = [tempname '.in']; end
            s = afocal4_solve(Pc, w.D, 'dofs',{'conic','standoff','front'}, ...
                    'label',sprintf('polish %.0f mm', q*1e3), 'deck',dk, ...
                    'max_iter', 10*opts.evals);
            m = merit_(P, s.D);
            fprintf('  polish: merit %.6f -> %.6f (%+.3f%%), exitflag %d\n', ...
                    w.merit, m, 100*(m/w.merit - 1), s.exitflag);
            if m <= w.merit
                w.D = s.D;  w.S = s.S;  w.merit = m;  w.exitflag = s.exitflag;
                w.nfev = w.nfev + s.nfev;   w.seconds = w.seconds + s.seconds;
                w.rounds(end+1) = struct('kind','polish', 'merit',m, ...
                    'exitflag',s.exitflag, 'nfev',s.nfev, 'seconds',s.seconds);
            else
                fprintf('  polish did not improve the merit -- winner kept\n');
            end
        end
        g = grad_probe_(P, w.D);
        gs = '';
        for i = 1:numel(g.names)
            gs = [gs sprintf('%s %+.3g  ', g.names{i}, g.g(i))]; %#ok<AGROW>
        end
        fprintf(['\n  GRADIENT PROBE at the delivered design (central, 1e-4 ' ...
                 'scaled):\n    |g| = %.4g,  max|g_i| = %.4g  [%s]\n'], ...
                norm(g.g), max(abs(g.g)), strtrim(gs));
        fprintf('    verdict: %s\n', verdict_(w.exitflag, g));

        % ---- the deck, the on-axis paraxial check, the packaging gate ---
        dk = fullfile(here, sprintf('afocal4_b2long_%03.0fmm.in', q*1e3));
        if ~opts.save, dk = [tempname '.in']; end
        b = afocal4_build(P, w.D, dk, 'verify',true, 'quiet',false);
        Don = w.D;   Don.bias_deg = 0;
        onax = [tempname '.in'];
        bon = afocal4_build(P, Don, onax, 'verify',true, 'quiet',true);
        fprintf(['  PARAXIAL, ON AXIS: traced M %.5fx against the closure''s ' ...
                 '%.5f (%+.3f%%), collimation %.2f urad\n'], ...
                bon.traced.mag, bon.C.fo.mag, ...
                100*(bon.traced.mag/bon.C.fo.mag - 1), ...
                bon.traced.collimation_urad);
        if exist(onax,'file'), delete(onax); end
        K = afocal4_pack(P, dk, 'quiet',true);
        afocal4_score_print(P, w.S, sprintf('basin2 %.0f mm, delivered', q*1e3));

        PT(end+1) = struct('iface',q, 'seed',{{res.seed}}, ...
            'rounds',{{res.rounds}}, 'D',w.D, 'S',w.S, 'merit',w.merit, ...
            'exitflag',w.exitflag, 'nfev',sum([res.nfev]), ...
            'seconds',sum([res.seconds]), 'all',{res}, 'grad',g, ...
            'build',b, 'pack',K, ...
            'onaxis',struct('mag_traced',bon.traced.mag, ...
                            'mag_paraxial',bon.C.fo.mag, ...
                            'collimation_urad',bon.traced.collimation_urad), ...
            'deck',dk); %#ok<AGROW>
        R.pt = PT;
        if opts.save, save(matf,'R','-v7.3');  fprintf('  saved %s\n', matf); end
    end
    R.pt = PT;
    if opts.save, save(matf,'R','-v7.3'); end
end

% =====================================================================
function r = long_solve_(P, here, seed, q, opts)
%LONG_SOLVE_  One seed, taken as far as the solver will go: a conics-only
%   pre-pass where the seed is cold (the S4 rule -- a cold joint solve sat
%   at 2246 nm where three conics reached 1391), then restart rounds from a
%   tightened trust region until exitflag 1 or a demonstrated plateau.
    dofs = {'conic','standoff','front'};
    D = seed.D;
    rounds = struct('kind',{},'merit',{},'exitflag',{},'nfev',{},'seconds',{});
    nfev = 0;   secs = 0;
    if seed.pre
        Pp = tighten_(P, 3e-4, 'forward', opts.evals);
        pre = afocal4_solve(Pp, D, 'dofs',{'conic'}, 'max_iter',12, ...
                            'quiet',true, 'label','pre');
        fprintf('  conic pre-pass: WFE %.1f nm, worst %.2fx\n', ...
                pre.S.wfe_max_nm, pre.S.worst);
        D = pre.D;   nfev = nfev + pre.nfev;   secs = secs + pre.seconds;
        rounds(end+1) = struct('kind','pre', 'merit',NaN, ...
            'exitflag',pre.exitflag, 'nfev',pre.nfev, 'seconds',pre.seconds);
    end
    m = merit_(P, D);
    fprintf('  seed merit %.6f\n', m);
    S = [];   exitflag = NaN;
    for k = 1:opts.rounds
        Pk = tighten_(P, 3e-4, 'forward', opts.evals);
        dk = [tempname '.in'];
        s = afocal4_solve(Pk, D, 'dofs',dofs, 'deck',dk, ...
                'label',sprintf('%s round %d', seed.name, k), ...
                'max_iter', 10*opts.evals);
        if exist(dk,'file'), delete(dk); end
        mk = merit_(P, s.D);
        gain = (m - mk)/max(m, realmin);
        fprintf('  round %d: merit %.6f -> %.6f (%+.4f%%), exitflag %d, %d evals\n', ...
                k, m, mk, -100*gain, s.exitflag, s.nfev);
        rounds(end+1) = struct('kind',sprintf('round%d',k), 'merit',mk, ...
            'exitflag',s.exitflag, 'nfev',s.nfev, 'seconds',s.seconds); %#ok<AGROW>
        nfev = nfev + s.nfev;   secs = secs + s.seconds;
        if mk <= m, D = s.D;  S = s.S;  m = mk;  end
        exitflag = s.exitflag;
        if s.exitflag == 1
            fprintf('  first-order optimality reached -- rounds stopped\n');
            break;
        end
        % A ROUND THAT RAN OUT OF BUDGET IS NOT A PLATEAU.  exitflag 0 means
        % lsqnonlin hit MaxFunctionEvaluations, so "no progress" there says
        % the budget was too small, not that the design is stationary --
        % calling that convergence is exactly the mistake this driver
        % exists to correct.
        if gain < opts.plateau && s.exitflag ~= 0
            fprintf(['  round bought %.2e of the merit (< %.0e) at exitflag ' ...
                     '%d -- PLATEAU, rounds stopped\n'], gain, opts.plateau, ...
                    s.exitflag);
            break;
        end
        if s.exitflag == 0
            fprintf('  round exhausted its %d-evaluation budget -- continuing\n', ...
                    opts.evals);
        end
    end
    if isempty(S)
        dk = [tempname '.in'];
        afocal4_build(P, D, dk, 'verify',false);
        S = afocal4_score(P, dk, 'nodes',P.solve.nodes_score, 'grid',P.grid_n);
        if exist(dk,'file'), delete(dk); end
    end
    r = struct('seed',seed.name, 'D',D, 'S',S, 'merit',m, 'rounds',rounds, ...
               'exitflag',exitflag, 'nfev',nfev, 'seconds',secs);
end

function Q = tighten_(P, fd, type, evals)
%TIGHTEN_  The long-solve settings, in one place.  Everything else about P
%   -- targets, weights, sampling, bounds, the packaging wall -- is the
%   study's, unchanged, because the point is to solve the SAME problem
%   properly rather than to solve a different one.
    Q = P;
    Q.solve.fd_step = fd;
    Q.solve.fd_type = type;
    Q.solve.tol_fun = 1e-8;
    Q.solve.tol_x   = 1e-9;
    Q.solve.tol_opt = 1e-8;
    Q.solve.max_fev = evals;
end

function m = merit_(P, D)
%MERIT_  The objective the solver actually minimises, at SOLVE sampling.
%   Rounds are compared on this and not on the scoring-sampling score: a
%   restart that improved the merit must be recognised as progress even if
%   the quotable number moves the other way.
    tmp = [tempname '.in'];
    D.ngrid = P.solve.ngrid;
    afocal4_build(P, D, tmp, 'verify',false);
    S = afocal4_score(P, tmp, 'fields',P.Fsolve, 'nodes',P.solve.nodes);
    m = S.merit;
    if exist(tmp,'file'), delete(tmp); end
end

function R = probe_only_(matf, P, dosave)
%PROBE_ONLY_  Re-open a finished run and ask the delivered designs the two
%   questions an exit code cannot answer: what is the gradient here, and is
%   there anything to be had by walking down it?
%
%   THE SECOND QUESTION IS THE POINT.  A large |g| at a design the solver
%   would not leave has two very different explanations -- a narrow valley
%   the trust region cannot follow, or a WALL (the packaging constraint, a
%   closure that stops existing) blocking the descent direction.  Walking
%   the gradient by hand tells them apart: if the merit falls, the solve was
%   short; if every step is worse or unbuildable, the design is against
%   something and |g| is measuring the constraint, not the objective.
    if ~isfile(matf)
        error('macos:design:afocal4_basin2:probe', ...
              'no run to probe: %s does not exist', matf);
    end
    q = load(matf);   R = q.R;
    for i = 1:numel(R.pt)
        g = grad_probe_(P, R.pt(i).D);
        d = descent_probe_(P, R.pt(i).D, g);
        R.pt(i).grad    = g;
        R.pt(i).descent = d;
        fprintf(['\n  iface %.0f mm: |g| = %.4g over the free DOFs  ->  best ' ...
                 'merit along -g %.6f at step %.3g (from %.6f, %+.4f%%)\n'], ...
                R.pt(i).iface*1e3, norm(g.g(isfinite(g.g))), d.best_merit, ...
                d.best_step, d.merit0, 100*(d.best_merit/d.merit0 - 1));
        fprintf('    %s\n', d.verdict);
    end
    if dosave, save(matf,'R','-v7.3'); end
end

function d = descent_probe_(P, D, g)
%DESCENT_PROBE_  Merit along the steepest-descent direction, by hand.
%
%   IT IS A LOWER BOUND, NOT A CONVERGENCE TEST.  On a badly-conditioned
%   least-squares problem -g is a poor direction, and this probe found
%   "nothing available" at every S4b design while a Gauss-Newton restart of
%   the same designs then took 0.1-1.3% off the merit.  Read a small gain
%   here as "the floor is not obviously escapable by walking downhill",
%   never as "this is a minimum".
    m0 = merit_(P, D);
    steps = [1e-3 3e-3 1e-2 3e-2 1e-1 3e-1];
    m = nan(size(steps));   ok = false(size(steps));
    w = gwalled_(g);
    if ~isempty(w)
        % -g does not exist: at least one DOF's finite difference left the
        % feasible set.  Do not fabricate a direction out of the DOFs that
        % remain -- report the wall.
        d = struct('steps',steps, 'merit',m, 'ok',ok, 'merit0',m0, ...
                   'best_merit',m0, 'best_step',0, 'gain',0, 'walled',{w});
        d.verdict = sprintf(['-g is not defined here: [%s] leave the feasible ' ...
                             'set, so the steepest-descent probe says nothing'], ...
                            strjoin(w,' '));
        fprintf('    %s\n', d.verdict);
        return;
    end
    u  = -g.g(:)/max(norm(g.g), realmin);
    for i = 1:numel(steps)
        Dt = D;
        for j = 1:numel(u)
            Dt = bump_(Dt, j, steps(i)*u(j)*g.scale(j));
        end
        try,  m(i) = merit_(P, Dt);  ok(i) = true;
        catch ME
            fprintf('    step %.3g: WALL (%s)\n', steps(i), ME.message);
        end
    end
    for i = 1:numel(steps)
        fprintf('    step %8.3g -> merit %12.6f%s\n', steps(i), m(i), ...
                ternary_(ok(i) && m(i) < m0, '   <- better', ''));
    end
    [mb, kb] = min(m);
    d = struct('steps',steps, 'merit',m, 'ok',ok, 'merit0',m0, ...
               'best_merit',mb, 'best_step',steps(kb), ...
               'gain', (m0-mb)/max(m0,realmin), 'walled', {{}});
    if ~any(ok)
        d.verdict = 'every descent step is a WALL -- the design is against the constraint';
    elseif d.gain < 1e-4
        d.verdict = sprintf(['nothing along -g: best gain %.2e -- a genuine ' ...
                             'floor, whatever the exit code says'], d.gain);
    elseif d.gain < 0.01
        d.verdict = sprintf(['a narrow valley: %.3f%% is available along -g, ' ...
                             'no more'], 100*d.gain);
    else
        d.verdict = sprintf(['UNDER-SOLVED: %.2f%% of the merit is available ' ...
                             'in one hand-walked step'], 100*d.gain);
    end
end

function g = grad_probe_(P, D)
%GRAD_PROBE_  The gradient of the merit at the delivered design, by CENTRAL
%   differences at a 1e-4 scaled step -- the step the smoothness probe showed
%   returns the same slope as 1e-5.  This is what separates "converged" from
%   "the solver gave up": an exit code is the solver's opinion, |g| is a
%   measurement.
    names = {'K2','K3','K4','s_FM','R_M2','t_M1M2'};
    sc = [P.dof_scale.conic, P.dof_scale.conic, P.dof_scale.conic, ...
          P.dof_scale.standoff, P.dof_scale.radius, P.dof_scale.spacing];
    h = 1e-4;
    gv = nan(1,numel(names));
    for i = 1:numel(names)
        try
            gv(i) = (merit_(P, bump_(D, i, +h*sc(i))) - ...
                     merit_(P, bump_(D, i, -h*sc(i)))) / (2*h);
        catch ME
            fprintf('    gradient probe on %s hit a wall: %s\n', names{i}, ...
                    ME.message);
        end
    end
    g = struct('names',{names}, 'g',gv, 'step',h, 'scale',sc, ...
               'walled', {names(~isfinite(gv))});
end

function w = gwalled_(g)
%GWALLED_  Which DOFs the gradient probe could not measure because the step
%   left the feasible set.  A NaN here is a CONSTRAINT, not a failure.
    w = {};
    if isfield(g,'walled') && ~isempty(g.walled), w = g.walled;  return; end
    if isfield(g,'g'), w = g.names(~isfinite(g.g)); end
end

function D = bump_(D, i, dv)
    switch i
    case 1, D.K(2) = D.K(2) + dv;
    case 2, D.K(3) = D.K(3) + dv;
    case 3, D.K(4) = D.K(4) + dv;
    case 4, D.fm_standoff = D.fm_standoff + dv;
    case 5, D.R2 = D.R2 + dv;
    case 6, D.t1 = D.t1 + dv;
    end
end

function s = verdict_(exitflag, g)
%VERDICT_  What the gradient says, in the language the answer needs.
%   A NaN component is not a failed measurement: it is the finite-difference
%   step falling OUTSIDE the feasible set, i.e. the design sitting ON the
%   packaging wall in that degree of freedom.  Reporting that as "not
%   converged" would be exactly backwards -- there is no interior minimum to
%   converge to along it.
    if isfield(g,'walled') && ~isempty(g.walled)
        s = sprintf(['ON THE PACKAGING WALL in [%s]: the merit still has a ' ...
                     'gradient there (max|g_i| = %.3g over the free DOFs) ' ...
                     'but descending it leaves the feasible set'], ...
                    strjoin(g.walled, ' '), max(abs(g.g(isfinite(g.g)))));
        return;
    end
    if exitflag == 1
        s = 'CONVERGED (first-order optimality)';
    elseif all(isfinite(g.g)) && norm(g.g) < 1e-3
        s = sprintf('CONVERGED by measurement (|g| = %.2e, exitflag %d)', ...
                    norm(g.g), exitflag);
    elseif all(isfinite(g.g)) && norm(g.g) < 1e-1
        s = sprintf(['PLATEAU: |g| = %.2e -- shallow, but not a stationary ' ...
                     'point (exitflag %d)'], norm(g.g), exitflag);
    else
        s = sprintf(['NOT CONVERGED: |g| = %.3g at the delivered design ' ...
                     '(exitflag %d)'], norm(g.g), exitflag);
    end
end

% ---------------------------------------------------------------------
function C = seeds_(P, q, want, prior)
%SEEDS_  Three independent starting points for the same basin.
%   'converged'  the S4b sweep's own delivered design at this standoff -- a
%                restart, which is the only one of the three that can show
%                the old answer was under-solved rather than wrong;
%   'fresh'      the shortest front end that puts the intermediate image far
%                enough behind M1 (the S4b seeder, verbatim);
%   'second'     the NEXT compliant closure after that one, chosen so its
%                field-mirror branch or its radius genuinely differs -- a
%                neighbouring grid point is not an independent seed.
    C = struct('name',{},'D',{},'pre',{});
    if any(strcmp(want,'converged')) && ~isempty(prior)
        k = find(abs([prior.iface] - q) < 1e-9, 1);
        if ~isempty(k)
            D = prior(k).D;   D.iface = q;
            C(end+1) = struct('name','converged', 'D',D, 'pre',false);
        end
    end
    L = compliant_(P, q);
    if any(strcmp(want,'fresh')) && ~isempty(L)
        D = afocal4_seed(P, 'bias_deg',P.bias_deg, 'iface',q);
        D.R2 = L(1).R2;   D.fm_standoff = L(1).s;
        C(end+1) = struct('name','fresh', 'D',D, 'pre',true);
    end
    if any(strcmp(want,'second')) && numel(L) >= 2
        % genuinely different: a different field-mirror branch if one exists,
        % otherwise a radius at least 10 mm away
        j = find(arrayfun(@(z) abs(z.s - L(1).s) > 1e-6, L(2:end)), 1) + 1;
        if isempty(j)
            j = find(arrayfun(@(z) abs(z.R2 - L(1).R2) > 0.010, L(2:end)), 1) + 1;
        end
        if ~isempty(j)
            D = afocal4_seed(P, 'bias_deg',P.bias_deg, 'iface',q);
            D.R2 = L(j).R2;   D.fm_standoff = L(j).s;
            C(end+1) = struct('name','second', 'D',D, 'pre',true);
        end
    end
    for i = 1:numel(C)
        fprintf('  seed %-10s R_M2 %8.4f m  s_FM %+8.4f m\n', ...
                C(i).name, C(i).D.R2, C(i).D.fm_standoff);
    end
end

function L = compliant_(P, iface)
%COMPLIANT_  Every (R_M2, s_FM) on the scan that closes with the whole back
%   end behind M1, in scan order.  This is AFOCAL4_S4B's IMAGE_BEHIND_SEED_
%   opened up: that helper returns the FIRST hit, and a second seed needs the
%   list.  Descending in radius because the variant's cost IS length -- the
%   image station runs away steeply as the secondary slows, so the first
%   compliant radius is the shortest telescope that complies.
    L = struct('R2',{},'s',{},'phi4',{},'behind_m1',{},'z_img',{});
    for R2 = 0.465:-0.0025:0.425
        Q = P;   Q.parent.R(2) = R2;
        p = afocal_first_order(Q.parent.R, Q.parent.t, Q.parent.convex, ...
                               'D',P.D, 'stop_ahead',P.stop_ahead);
        a0 = -p.y_marginal(2)/p.u_marginal(2);
        if ~(a0 > 0) || a0 > 6, continue; end
        z_img = -Q.parent.t(1) + a0;
        for s = [0 0.05 -0.05 0.10 -0.10]
            [phi4, C, found] = afocal4_phi4(Q, s, iface);
            if ~found || C.behind_m1 < P.pack.m3_behind_min + 0.05, continue; end
            L(end+1) = struct('R2',R2, 's',s, 'phi4',phi4, ...
                              'behind_m1',C.behind_m1, 'z_img',z_img); %#ok<AGROW>
        end
        if numel(L) >= 6, break; end
    end
end

function s = spread_(res)
%SPREAD_  How far apart the seeds landed -- OVER THE SOUND ONES ONLY.
%   A seed that never reached a sound design does not report a worse
%   design, it reports no design: pupil_map's anchoring residual is 0.1 um
%   on every sound afocal4 solve here and 151 MM on the 343 mm cold start
%   (S4 rule 8, the solver-integrity gate).  Averaging that into a
%   seed-to-seed spread turns a failed seed into a second "basin", which is
%   the opposite of what the number is for.
    ar = arrayfun(@(r) r.S.anchor_resid_um, res);
    ok = isfinite(ar) & ar < 1e3;                    % 1 mm, i.e. 4 decades out
    s = struct('n_sound',nnz(ok), 'n_invalid',nnz(~ok), 'merit',NaN, 'wfe',NaN);
    if nnz(ok) < 2, return; end
    m = [res(ok).merit];   w = arrayfun(@(r) r.S.wfe_max_nm, res(ok));
    s.merit = (max(m)-min(m))/max(min(m),realmin);
    s.wfe   = (max(w)-min(w))/max(min(w),realmin);
end

function PTo = pack_(PT, res, q) %#ok<INUSD>
%PACK_  A partial checkpoint: whatever seeds have finished at this point.
    PTo = PT;
    if isempty(res), return; end
    [~,k] = min([res.merit]);
    PTo(end+1).iface = q;
    PTo(end).seed    = {res.seed};
    PTo(end).rounds  = {res.rounds};
    PTo(end).D       = res(k).D;
    PTo(end).S       = res(k).S;
    PTo(end).merit   = res(k).merit;
    PTo(end).exitflag = res(k).exitflag;
    PTo(end).nfev    = sum([res.nfev]);
    PTo(end).seconds = sum([res.seconds]);
    PTo(end).all     = res;
end

function s = tick_(b),  if b, s = '   <- delivered'; else, s = ''; end,  end
function v = ternary_(c,a,b),  if c, v = a; else, v = b; end,  end

function banner_(s)
    fprintf('\n%s\n%s\n%s\n', repmat('=',1,74), s, repmat('=',1,74));
end
