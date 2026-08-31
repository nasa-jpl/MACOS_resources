function R = clear_solve(P, D0, opts)
%CLEAR_SOLVE  AFOCAL4_SOLVE with the extraction tilt in the loop.
%
%   R = CLEAR_SOLVE(P, D0) is AFOCAL4_SOLVE's outer loop built on
%   CLEAR_BUILD instead of AFOCAL4_BUILD, so every iterate is a design that
%   has already been swung by D.tilt_deg before it is scored.  Same DOF
%   packing, same scaled-DEVIATION parameterisation, same log-domain merit,
%   same wall-not-penalty treatment of an unbuildable iterate -- the only
%   changes are the builder and one extra DOF.
%
%   WHY A SEPARATE SOLVE AND NOT A FLAG ON AFOCAL4_SOLVE.  AFOCAL4_SOLVE and
%   AFOCAL4_BUILD are the committed evidence for the S4/S4b/S4c trade rows;
%   a new branch inside them would put every one of those numbers at the
%   mercy of a code path they never ran.  This file is additive and the
%   committed ones are untouched.  The two are pinned against each other:
%   with 'dofs' not containing 'tilt' and D0.tilt_deg = 0, CLEAR_SOLVE and
%   AFOCAL4_SOLVE must agree exactly -- AFOCAL4_CLEARING's null section
%   asserts it.
%
%   THE EXTRA DOF:
%     'tilt'   the field mirror's extraction tilt, DEGREES.  Bounded by
%              P.clear.tilt_bounds (default [-20 20] deg).  It is offered
%              as a DOF for the ONE experiment that needs it -- does the
%              solver, given the choice, pay for clearance or spend the
%              tilt on the pupil? -- and is NOT in the default set.  The
%              delivered design fixes the tilt at a stated operating value
%              (CLEAR_PRICE reports the curve) and re-solves the rest,
%              because a merit that cannot see the clearance would trade it
%              away: clearance is a WALL, and a wall is not a merit term
%              (the S4b earned rule).
%
%   Name-value: as AFOCAL4_SOLVE, plus
%     'axis'   tilt axis handed to CLEAR_BUILD (default [1 0 0])
%
%   Returns R with .D .S .x0 .x .names .scale .base .hist .exitflag .nfev
%   .seconds .deck .dofs .tilt_deg.
%
%   See also CLEAR_BUILD, CLEAR_PRICE, AFOCAL4_SOLVE, AFOCAL4_SCORE.

    arguments
        P (1,1) struct
        D0 (1,1) struct
        opts.dofs     (1,:) cell = {'conic'}
        opts.deck     (1,:) char = ''
        opts.label    (1,:) char = 'clear solve'
        opts.max_iter (1,1) double = 0
        opts.pupil    (1,1) logical = true
        opts.axis     (1,3) double = [1 0 0]
        opts.quiet    (1,1) logical = false
    end
    if opts.max_iter <= 0, opts.max_iter = P.solve.max_iter; end
    if ~isfield(D0,'tilt_deg'), D0.tilt_deg = 0; end
    tb = [-20 20];
    if isfield(P,'clear') && isfield(P.clear,'tilt_bounds'), tb = P.clear.tilt_bounds; end

    [x0, names, scale, base, lo, hi] = pack_(P, D0, opts.dofs, tb);
    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>

    hist = struct('x',{},'merit',{},'worst',{},'wfe',{},'blur',{}, ...
                  'breathe',{},'wander',{});
    nfev = 0;   t0 = tic;

    function r = obj_(xs)
        nfev = nfev + 1;
        D = unpack_(P, D0, opts.dofs, xs, scale, base);
        D.ngrid = P.solve.ngrid;
        try
            clear_build(P, D, tmp, 'axis',opts.axis, 'verify',false);
            S = afocal4_score(P, tmp, 'fields',P.Fsolve, ...
                              'nodes',P.solve.nodes, 'pupil',opts.pupil);
        catch ME
            % A WALL IS ONLY A WALL WHILE IT DOMINATES THE MERIT'S OWN SCALE.
            % A rejected iterate returns a large finite residual so lsqnonlin
            % backs out of it -- but "large" is relative.  At the study's own
            % weights a sound design scores ~30 and a constant 20 per
            % component (merit 5600) is an impassable barrier.  Raise the
            % pupil weights x16 to measure the addendum's slack and a sound
            % design scores ~4e4: the SAME constant now looks ATTRACTIVE, and
            % the solver walks THROUGH the wall on purpose.  Measured, once:
            % a pupil-weighted run returned a converged x whose closure put
            % M3 1051 mm in FRONT of the primary, and died in the report
            % build on the S4b packaging wall.
            % So the residual scales with the largest merit weight in play.
            % With the study's own weights max(w) = 1 and this is 20 exactly,
            % bit-identical to every committed clearing-stage solve.
            wallr = 20 * max(1, max(cell2mat(struct2cell(P.weights))));
            r = repmat(wallr, size(P.Fsolve,1) + 5*opts.pupil, 1);
            hist(end+1) = struct('x',xs(:).', 'merit',sum(r.^2), 'worst',Inf, ...
                'wfe',NaN, 'blur',NaN, 'breathe',NaN, 'wander',NaN); %#ok<SETNU>
            if ~opts.quiet && mod(nfev,25) == 1
                fprintf('    [%3d] WALL: %s\n', nfev, ME.message);
            end
            return;
        end
        r = S.resid;
        hist(end+1) = struct('x',xs(:).', 'merit',sum(r.^2), 'worst',S.worst, ...
            'wfe',S.wfe_max_nm, 'blur',S.blur_um, 'breathe',S.breathe_pct, ...
            'wander',S.wander_um); %#ok<SETNU>
        if ~opts.quiet && (nfev <= 2 || mod(nfev, 20) == 0)
            fprintf(['    [%3d] merit %9.4f  worst %7.2fx | WFE %8.1f nm  ' ...
                     'blur %7.1f  breathe %6.3f%%  wander %7.1f um\n'], ...
                    nfev, sum(r.^2), S.worst, S.wfe_max_nm, S.blur_um, ...
                    S.breathe_pct, S.wander_um);
        end
    end

    fdt = 'forward';   topt = [];
    if isfield(P.solve,'fd_type'), fdt = P.solve.fd_type; end
    if isfield(P.solve,'tol_opt'), topt = P.solve.tol_opt; end
    o = optimoptions('lsqnonlin', 'Display','off', ...
        'MaxIterations',opts.max_iter, ...
        'MaxFunctionEvaluations',opts.max_iter*(numel(x0)+2), ...
        'FunctionTolerance',P.solve.tol_fun, 'StepTolerance',P.solve.tol_x, ...
        'FiniteDifferenceStepSize',P.solve.fd_step, ...
        'FiniteDifferenceType',fdt);
    if ~isempty(topt), o = optimoptions(o, 'OptimalityTolerance', topt); end
    if isfield(P.solve,'max_fev') && P.solve.max_fev > 0
        o = optimoptions(o, 'MaxFunctionEvaluations', P.solve.max_fev);
    end
    if ~opts.quiet
        fprintf('  %s: %d DOFs [%s], tilt %+.3f deg\n', opts.label, numel(x0), ...
                strjoin(names,' '), D0.tilt_deg);
    end
    [x, ~, ~, exitflag] = lsqnonlin(@obj_, x0, lo, hi, o);

    D = unpack_(P, D0, opts.dofs, x, scale, base);
    deck = opts.deck;   if isempty(deck), deck = tmp; end
    % THE FINAL BUILD IS A REPORT, SO IT MEASURES AND DOES NOT JUDGE.  Walls
    % belong on ITERATES: inside obj_ a violation is turned into a large
    % finite residual and the solver backs out of it.  Here there is nobody
    % to back out -- and the union wall in particular is evaluated at SOLVE
    % sampling inside the loop and would be re-evaluated at REPORTING
    % sampling here, where a bigger ray grid makes a bigger union hull and
    % the floor reads ~2 mm lower.  A converged design sitting ON its wall
    % therefore throws out of the report path and the whole multi-hour solve
    % is lost with it.  Measured, once: an hour of a -8 deg walled run.
    % With every wall off (the default) this is the same call as before.
    Pr = P;
    if isfield(Pr,'pack') && isfield(Pr.pack,'union_enforce')
        Pr.pack.union_enforce = false;
    end
    clear_build(Pr, D, deck, 'axis',opts.axis, 'verify',false);
    S = afocal4_score(P, deck, 'fields',P.Fsolve, 'pupil',opts.pupil, ...
                      'nodes',P.solve.nodes_score, 'grid',P.grid_n);

    R = struct('D',D, 'S',S, 'x0',x0, 'x',x, 'names',{names}, 'scale',scale, ...
               'base',base, 'lo',lo, 'hi',hi, 'hist',hist, 'exitflag',exitflag, ...
               'nfev',nfev, 'seconds',toc(t0), 'deck',deck, ...
               'dofs',{opts.dofs}, 'tilt_deg',D.tilt_deg, 'axis',opts.axis);
    if ~opts.quiet
        fprintf('  %s: %d evaluations, %.1f s, exitflag %d\n', ...
                opts.label, nfev, R.seconds, exitflag);
        afocal4_score_print(P, S, opts.label);
    end
end

% =====================================================================
function [x, names, scale, base, lo, hi] = pack_(P, D, dofs, tb)
    x = [];  names = {};  scale = [];  base = [];  lo = [];  hi = [];
    ki = conic_free_(D.form);
    if any(strcmp(dofs,'conic'))
        for k = ki
            [x,names,scale,base,lo,hi] = add_(x,names,scale,base,lo,hi, ...
                sprintf('K%d',k), D.K(k), P.dof_scale.conic, P.bounds.conic);
        end
    end
    if any(strcmp(dofs,'standoff'))
        [x,names,scale,base,lo,hi] = add_(x,names,scale,base,lo,hi, ...
            's_FM', D.fm_standoff, P.dof_scale.standoff, P.bounds.fm_standoff);
    end
    if any(strcmp(dofs,'front'))
        [x,names,scale,base,lo,hi] = add_(x,names,scale,base,lo,hi, ...
            'R_M2', D.R2, P.dof_scale.radius, P.bounds.R2);
        [x,names,scale,base,lo,hi] = add_(x,names,scale,base,lo,hi, ...
            't_M1M2', D.t1, P.dof_scale.spacing, P.bounds.t1);
    end
    if any(strcmp(dofs,'tilt'))
        % scale 1 deg: the clearance curve moves over ~10 deg and the pupil
        % terms over ~2, so one solver unit is one degree.
        [x,names,scale,base,lo,hi] = add_(x,names,scale,base,lo,hi, ...
            'tilt', D.tilt_deg, 1.0, tb);
    end
    if any(strcmp(dofs,'rb'))
        for i = 1:numel(P.rb_elts)
            [x,names,scale,base,lo,hi] = add_(x,names,scale,base,lo,hi, ...
                sprintf('dy%d',P.rb_elts(i)), D.rb(i,1), P.dof_scale.dec, P.bounds.dec);
            [x,names,scale,base,lo,hi] = add_(x,names,scale,base,lo,hi, ...
                sprintf('tx%d',P.rb_elts(i)), D.rb(i,2), P.dof_scale.tilt, P.bounds.tilt);
        end
    end
    x = x(:);  scale = scale(:);  base = base(:);  lo = lo(:);  hi = hi(:);
end

function [x,nm,sc,ba,lo,hi] = add_(x,nm,sc,ba,lo,hi, name, val, s, bnd)
    x(end+1)  = 0;
    nm{end+1} = name;
    sc(end+1) = s;
    ba(end+1) = val;
    lo(end+1) = (bnd(1) - val)/s;
    hi(end+1) = (bnd(2) - val)/s;
end

function D = unpack_(P, D, dofs, xs, scale, base)
%UNPACK_  Read together with PACK_: same order, or the solve moves a DOF
%   nobody asked it to.
    xs = base(:) + xs(:).*scale(:);   j = 0;
    ki = conic_free_(D.form);
    if any(strcmp(dofs,'conic'))
        for k = ki, j = j + 1;  D.K(k) = xs(j); end
    end
    if any(strcmp(dofs,'standoff')), j = j + 1;  D.fm_standoff = xs(j); end
    if any(strcmp(dofs,'front')),    D.R2 = xs(j+1);  D.t1 = xs(j+2);  j = j + 2; end
    if any(strcmp(dofs,'tilt')),     j = j + 1;  D.tilt_deg = xs(j); end
    if any(strcmp(dofs,'rb'))
        for i = 1:numel(P.rb_elts)
            D.rb(i,1) = xs(j+1);   D.rb(i,2) = xs(j+2);   j = j + 2;
        end
    end
end

function ki = conic_free_(form)
    switch form
    case 'field',    ki = [2 3 4];
    case 'mersenne', ki = [1 2 3 4];
    otherwise, error('macos:design:clear_solve:form','unknown form "%s".', form);
    end
end

function del_(p),  if exist(p,'file'), delete(p); end,  end
