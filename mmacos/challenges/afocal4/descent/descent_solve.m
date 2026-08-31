function R = descent_solve(P, D0, opts)
%DESCENT_SOLVE  The outer loop for an N-mirror rung.
%
%   R = DESCENT_SOLVE(P, D0) is AFOCAL4_SOLVE / CLEAR_SOLVE generalized: same
%   scaled-DEVIATION parameterisation, same log-domain merit, same
%   wall-not-penalty treatment of an unbuildable iterate, over a DOF set that
%   grows with N.
%
%   THE DOF SET IS EXPLICIT IN THE RECORD, and tilts are in it FROM THE START
%   (the descent brief's ruling, and the wall slice's lesson):
%
%     'conic'    N conics
%     'radius'   the N-2 FREE radii (the closure consumes the last two)
%     'spacing'  the N-2 FREE spacings (the closure consumes the last one)
%     'tilt'     N extraction tilts, degrees
%     'iface'    the interface standoff -- OFF by default: it is the S4
%                ruling's OPERATING POINT, carried as a parameter and
%                reported, not optimised away.
%
%   FOUR RULES THIS FILE INHERITS RATHER THAN REDISCOVERS, each paid for:
%
%   1  SOLVE IN SCALED DEVIATIONS, not scaled values (RESULTS rule 2).
%      Values size the trust region by the largest DOF and rung 1 wandered
%      2323 -> 2516 nm.
%   2  A WALL IS ONLY A WALL WHILE IT DOMINATES THE MERIT'S SCALE (rule 32).
%      The rejected-iterate residual scales with the largest merit weight in
%      play, so a re-weighting or a regularizer cannot quietly turn the wall
%      into an attractor -- which it did once, returning a converged design
%      with a mirror 1051 mm on the wrong side of the primary.
%   3  A WALL BELONGS ON ITERATES, NEVER ON THE REPORT (rule 31).  The final
%      build here measures and does not judge; walls applied to it throw the
%      whole solve away at the last step.
%   4  CENTRAL DIFFERENCES (S4c).  A 3e-3 forward difference reads this
%      merit's gradient 17 % low, and the stalls that produces were misread
%      as convergence twice in this arc.
%
%   Name-value:
%     'dofs'      cell of the groups above ({'conic','spacing','tilt'})
%     'deck'      where to write the converged design ('' = temporary)
%     'label'     for the progress line
%     'max_iter'  lsqnonlin iterations (P.solve.max_iter)
%     'axis'      tilt axis handed to DESCENT_BUILD ([1 0 0])
%     'quiet'     (false)
%
%   Returns R with .D .S .x .names .scale .base .hist .exitflag .nfev
%   .seconds .deck .dofs.
%
%   See also DESCENT_BUILD, DESCENT_REQUIRE, AFOCAL4_SCORE, CLEAR_SOLVE.

    arguments
        P (1,1) struct
        D0 (1,1) struct
        opts.dofs     (1,:) cell = {'conic','spacing','tilt'}
        opts.deck     (1,:) char = ''
        opts.label    (1,:) char = 'descent solve'
        opts.max_iter (1,1) double = 0
        opts.axis     (1,3) double = [1 0 0]
        opts.quiet    (1,1) logical = false
    end
    if opts.max_iter <= 0, opts.max_iter = P.solve.max_iter; end
    N = D0.N;

    [x0, names, scale, base, lo, hi] = pack_(P, D0, opts.dofs);
    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>
    hist = struct('x',{},'merit',{},'worst',{},'wfe',{},'blur',{}, ...
                  'breathe',{},'wander',{});
    nfev = 0;   t0 = tic;
    % rule 32: the wall residual has to out-scale the merit it is bounding.
    wallr = 20 * max(1, max(cell2mat(struct2cell(P.weights))));
    nres  = size(P.Fsolve,1) + 5;

    function r = obj_(xs)
        nfev = nfev + 1;
        D = unpack_(P, D0, opts.dofs, xs, scale, base);
        D.ngrid = P.solve.ngrid;
        try
            descent_build(P, D, tmp, 'axis',opts.axis, 'verify',false);
            S = afocal4_score(P, tmp, 'fields',P.Fsolve, 'nodes',P.solve.nodes);
        catch ME
            r = repmat(wallr, nres, 1);
            hist(end+1) = struct('x',xs(:).', 'merit',sum(r.^2), 'worst',Inf, ...
                'wfe',NaN, 'blur',NaN, 'breathe',NaN, 'wander',NaN); %#ok<SETNU>
            if ~opts.quiet && mod(nfev,50) == 1
                fprintf('    [%4d] WALL: %s\n', nfev, one_line_(ME.message));
            end
            return;
        end
        r = S.resid;
        % AFOCAL4_SCORE HAS ITS OWN FAILURE PATH and it returns a MINIMAL
        % struct -- resid / merit / worst and nothing else -- when the deck
        % will not load or the ladder will not build.  Reaching straight for
        % S.wfe_max_nm there crashes the solve on an "Unrecognized field
        % name", which turns a scored-as-bad iterate into a dead run.  Cost:
        % one ascent rung.  The history is diagnostic, so a missing column is
        % a NaN and never an exception.  (CLEAR_SOLVE carries the same
        % unguarded access; it has simply never been handed that struct.)
        hist(end+1) = struct('x',xs(:).', 'merit',sum(r.^2), ...
            'worst',fld_(S,'worst',Inf), 'wfe',fld_(S,'wfe_max_nm',NaN), ...
            'blur',fld_(S,'blur_um',NaN), 'breathe',fld_(S,'breathe_pct',NaN), ...
            'wander',fld_(S,'wander_um',NaN)); %#ok<SETNU>
        if ~opts.quiet && (nfev <= 2 || mod(nfev, 50) == 0)
            fprintf(['    [%4d] merit %9.4f  worst %8.2fx | WFE %9.1f nm  ' ...
                     'blur %7.1f  breathe %6.3f%%  wander %7.1f um\n'], ...
                    nfev, sum(r.^2), fld_(S,'worst',Inf), ...
                    fld_(S,'wfe_max_nm',NaN), fld_(S,'blur_um',NaN), ...
                    fld_(S,'breathe_pct',NaN), fld_(S,'wander_um',NaN));
        end
    end

    fdt = 'central';   topt = 1e-8;
    if isfield(P.solve,'fd_type'), fdt = P.solve.fd_type; end
    if isfield(P.solve,'tol_opt'), topt = P.solve.tol_opt; end
    o = optimoptions('lsqnonlin', 'Display','off', ...
        'MaxIterations',opts.max_iter, ...
        'MaxFunctionEvaluations',opts.max_iter*(numel(x0)+2), ...
        'FunctionTolerance',P.solve.tol_fun, 'StepTolerance',P.solve.tol_x, ...
        'FiniteDifferenceStepSize',P.solve.fd_step, ...
        'FiniteDifferenceType',fdt, 'OptimalityTolerance',topt);
    if isfield(P.solve,'max_fev') && P.solve.max_fev > 0
        o = optimoptions(o, 'MaxFunctionEvaluations', P.solve.max_fev);
    end
    if ~opts.quiet
        fprintf('  %s: N %d, %d DOFs [%s]\n', opts.label, N, numel(x0), ...
                strjoin(opts.dofs,' '));
    end
    [x, ~, ~, exitflag] = lsqnonlin(@obj_, x0, lo, hi, o);

    D = unpack_(P, D0, opts.dofs, x, scale, base);
    deck = opts.deck;   if isempty(deck), deck = tmp; end
    % rule 31: the report measures, it does not judge.
    Pr = P;   Pr.pack.enforce = false;
    if isfield(Pr.pack,'union_enforce'), Pr.pack.union_enforce = false; end
    descent_build(Pr, D, deck, 'axis',opts.axis, 'verify',false);
    S = afocal4_score(P, deck, 'fields',P.Fsolve, 'nodes',P.solve.nodes_score, ...
                      'grid',P.grid_n);
    R = struct('D',D, 'S',S, 'x0',x0, 'x',x, 'names',{names}, 'scale',scale, ...
               'base',base, 'lo',lo, 'hi',hi, 'hist',hist, 'exitflag',exitflag, ...
               'nfev',nfev, 'seconds',toc(t0), 'deck',deck, 'dofs',{opts.dofs});
    if ~opts.quiet
        fprintf('  %s: %d evaluations, %.1f min, exitflag %d\n', opts.label, ...
                nfev, R.seconds/60, exitflag);
    end
end

% =====================================================================
function [x, nm, sc, ba, lo, hi] = pack_(P, D, dofs)
    x = [];  nm = {};  sc = [];  ba = [];  lo = [];  hi = [];
    N = D.N;   nf = N - 2;
    if any(strcmp(dofs,'conic'))
        for k = 1:N
            [x,nm,sc,ba,lo,hi] = add_(x,nm,sc,ba,lo,hi, sprintf('K%d',k), ...
                D.K(k), P.dof_scale.conic, P.bounds.conic);
        end
    end
    if any(strcmp(dofs,'radius'))
        for k = 1:nf
            [x,nm,sc,ba,lo,hi] = add_(x,nm,sc,ba,lo,hi, sprintf('R%d',k), ...
                D.R(k), P.dof_scale.radius, [0.15 12]);
        end
    end
    if any(strcmp(dofs,'spacing'))
        for k = 1:nf
            [x,nm,sc,ba,lo,hi] = add_(x,nm,sc,ba,lo,hi, sprintf('t%d',k), ...
                D.t(k), P.dof_scale.spacing, [0.05 6]);
        end
    end
    if any(strcmp(dofs,'tilt'))
        % scale 1 deg: the clearance curve moves over ~10 deg and the pupil
        % terms over ~2, so one solver unit is one degree (the wall slice's
        % own scaling, kept).
        for k = 1:N
            [x,nm,sc,ba,lo,hi] = add_(x,nm,sc,ba,lo,hi, sprintf('a%d',k), ...
                D.tilt_deg(k), 1.0, [-20 20]);
        end
    end
    if any(strcmp(dofs,'iface'))
        [x,nm,sc,ba,lo,hi] = add_(x,nm,sc,ba,lo,hi, 'iface', D.iface, ...
                                  0.05, [0.05 0.60]);
    end
    x = x(:);  sc = sc(:);  ba = ba(:);  lo = lo(:);  hi = hi(:);
end

function [x,nm,sc,ba,lo,hi] = add_(x,nm,sc,ba,lo,hi, name, val, s, bnd)
    x(end+1)  = 0;      nm{end+1} = name;   sc(end+1) = s;   ba(end+1) = val;
    lo(end+1) = (bnd(1) - val)/s;           hi(end+1) = (bnd(2) - val)/s;
end

function D = unpack_(P, D, dofs, xs, scale, base) %#ok<INUSL>
%UNPACK_  Read together with PACK_: same order, or the solve moves a DOF
%   nobody asked it to.
    xs = base(:) + xs(:).*scale(:);   j = 0;
    N = D.N;   nf = N - 2;
    if any(strcmp(dofs,'conic'))
        for k = 1:N, j = j + 1;  D.K(k) = xs(j); end
    end
    if any(strcmp(dofs,'radius'))
        for k = 1:nf, j = j + 1;  D.R(k) = xs(j); end
    end
    if any(strcmp(dofs,'spacing'))
        for k = 1:nf, j = j + 1;  D.t(k) = xs(j); end
    end
    if any(strcmp(dofs,'tilt'))
        for k = 1:N, j = j + 1;  D.tilt_deg(k) = xs(j); end
    end
    if any(strcmp(dofs,'iface')), j = j + 1;  D.iface = xs(j); end
end

function v = fld_(s, f, d)
    if isstruct(s) && isfield(s,f), v = s.(f); else, v = d; end
end

function s = one_line_(m)
    s = regexprep(m, '\s+', ' ');
    if numel(s) > 100, s = [s(1:100) '...']; end
end

function del_(p),  if exist(p,'file'), delete(p); end,  end
