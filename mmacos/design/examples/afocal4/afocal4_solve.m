function R = afocal4_solve(P, D0, opts)
%AFOCAL4_SOLVE  The S4 joint solve: one DOF set, nested inside an exact
%   first-order closure.
%
%   R = AFOCAL4_SOLVE(P, D0) moves the outer degrees of freedom of the
%   design D0 to minimise AFOCAL4_SCORE's residual, re-closing the afocal
%   condition, the collimator's station and the pupil station EXACTLY at
%   every iterate (AFOCAL4_BUILD).  Never alternates: one joint DOF set per
%   rung, seeds are seeds (rodgers1 doctrine rule 1).
%
%   THE OUTER DOFs, and what is held:
%     'conic'     the conics.  'field' form: K_M2, K_FM, K_M3 -- M1 is HELD
%                 at -1, a parabola, matching his study and stated as held
%                 in every provenance table.  'mersenne' form: all four.
%     'standoff'  the FIELD MIRROR's standoff before the intermediate image.
%                 Not the interface standoff -- see AFOCAL4_PARAMS.
%     'front'     the front end: M2's radius and the M1-M2 spacing.  His own
%                 rung 3 re-optimises "conics and radii", and holding the
%                 whole front end answers a smaller question than he asked.
%                 Everything downstream is re-derived by the closure.
%     'rb'        rigid bodies: y-decenter and x-tilt on each of P.rb_elts.
%
%   NOT A DOF: the field mirror's POWER.  The inner closure consumes it to
%   hold the pupil station at D.iface, which the S4 ruling carries as a
%   PARAMETER (the operating point) rather than a spec.  Sweeping it is the
%   trade curve, not the solve -- afocal4_ladder('trade').
%
%   NOT A DOF EITHER: M1.  Its radius sets the aperture and the f-number the
%   whole benchmark is posed at, and its conic is HELD at -1, a parabola, as
%   in his study.  R_FM and R_col are not free either -- they are whatever
%   the closure needs them to be at the current (iface, standoff, front end),
%   which moves them ~29% from the parent at the flagged operating point.
%   So those radii DO change rung to rung; they are just never free.
%
%   SCALING, AND IT IS NOT A DETAIL.  The solver works in DEVIATIONS from the
%   seed, divided by P.dof_scale:  x = (value - seed) / scale,  so every DOF
%   starts at 0 and one unit of x means the same size of physical change on
%   all of them.  Conics are O(1), the M1-M2 spacing is O(1 m), a tilt is
%   O(1 mrad); feeding those to lsqnonlin as raw scaled VALUES makes the
%   initial trust region proportional to the largest of them, and the first
%   step throws the design across the parameter space.  Measured: the value
%   parameterisation left rung 1 wandering at 2345 nm after 40 evaluations,
%   worse than a three-conic solve had already reached.
%
%   Name-value:
%     'dofs'     cellstr subset of {'conic','standoff','rb'}
%                (default {'conic'})
%     'deck'     where the converged design is written (required to keep it)
%     'label'    for the progress lines
%     'merit'    'log' (default) | 'linear' -- the A/B of AFOCAL4_SCORE's
%                earned rule.  'linear' scores each term by m/t rather than
%                log(m/t), which is what "normalise every target to 1" reads
%                as and which the WFE term then owns outright.
%     'max_iter' (P.solve.max_iter)
%     'quiet'    (false)
%
%   Returns R with .D (the converged design), .S (its score at SCORING
%   sampling), .x0 .x .names .scale, .hist (every evaluation: x, merit,
%   worst), .exitflag, .nfev, .seconds.
%
%   See also AFOCAL4_BUILD, AFOCAL4_SCORE, AFOCAL4_LADDER.

    arguments
        P (1,1) struct
        D0 (1,1) struct
        opts.dofs     (1,:) cell = {'conic'}
        opts.deck     (1,:) char = ''
        opts.label    (1,:) char = 'solve'
        opts.merit    (1,:) char {mustBeMember(opts.merit,{'log','linear'})} = 'log'
        opts.max_iter (1,1) double = 0
        opts.pupil    (1,1) logical = true
        opts.quiet    (1,1) logical = false
    end
    if opts.max_iter <= 0, opts.max_iter = P.solve.max_iter; end

    [x0, names, scale, base, lo, hi] = pack_(P, D0, opts.dofs);
    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>

    hist = struct('x',{},'merit',{},'worst',{},'wfe',{},'blur',{}, ...
                  'breathe',{},'wander',{});
    nfev = 0;
    t0 = tic;

    function r = obj_(xs)
        nfev = nfev + 1;
        D = unpack_(P, D0, opts.dofs, xs, scale, base);
        D.ngrid = P.solve.ngrid;
        try
            afocal4_build(P, D, tmp, 'verify',false);
            S = afocal4_score(P, tmp, 'fields',P.Fsolve, ...
                              'nodes',P.solve.nodes, 'pupil',opts.pupil);
        catch ME
            % An unbuildable iterate (a standoff past the image, a closure
            % with no root) is a WALL, not a datum.  Large finite residuals
            % turn it back; an error would end the solve.
            r = repmat(20, size(P.Fsolve,1) + 5*opts.pupil, 1);
            hist(end+1) = struct('x',xs(:).', 'merit',sum(r.^2), 'worst',Inf, ...
                'wfe',NaN, 'blur',NaN, 'breathe',NaN, 'wander',NaN); %#ok<SETNU>
            if ~opts.quiet && mod(nfev,25) == 1
                fprintf('    [%3d] WALL: %s\n', nfev, ME.message);
            end
            return;
        end
        if strcmp(opts.merit,'linear')
            r = resid_linear_(P, S);
        else
            r = S.resid;
        end
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

    o = optimoptions('lsqnonlin', 'Display','off', ...
        'MaxIterations',opts.max_iter, ...
        'MaxFunctionEvaluations',opts.max_iter*(numel(x0)+2), ...
        'FunctionTolerance',P.solve.tol_fun, 'StepTolerance',P.solve.tol_x, ...
        'FiniteDifferenceStepSize',P.solve.fd_step);
    if ~opts.quiet
        fprintf('  %s: %d DOFs [%s], merit "%s"\n', opts.label, numel(x0), ...
                strjoin(names, ' '), opts.merit);
    end
    [x, ~, ~, exitflag] = lsqnonlin(@obj_, x0, lo, hi, o);

    D = unpack_(P, D0, opts.dofs, x, scale, base);
    deck = opts.deck;
    if isempty(deck), deck = tmp; end
    afocal4_build(P, D, deck, 'verify',false);
    S = afocal4_score(P, deck, 'fields',P.Fsolve, 'pupil',opts.pupil, ...
                      'nodes',P.solve.nodes_score, 'grid',P.grid_n);

    R = struct('D',D, 'S',S, 'x0',x0, 'x',x, 'names',{names}, 'scale',scale, ...
               'base',base, ...
               'lo',lo, 'hi',hi, 'hist',hist, 'exitflag',exitflag, ...
               'nfev',nfev, 'seconds',toc(t0), 'merit_kind',opts.merit, ...
               'deck',deck, 'dofs',{opts.dofs});
    if ~opts.quiet
        fprintf('  %s: %d evaluations, %.1f s, exitflag %d\n', ...
                opts.label, nfev, R.seconds, exitflag);
        afocal4_score_print(P, S, opts.label);
    end
end

% =====================================================================
function [x, names, scale, base, lo, hi] = pack_(P, D, dofs)
%PACK_  Physical DOFs -> the scaled DEVIATION vector the solver works in.
%   BASE is the seed value of each DOF; the solver's x is (value-base)/scale
%   and therefore starts at zero on every DOF.
    x = [];  names = {};  scale = [];  base = [];  lo = [];  hi = [];
    ki = conic_free_(D.form);
    if any(strcmp(dofs,'conic'))
        for k = ki
            [x,names,scale,base,lo,hi] = add_(x,names,scale,base,lo,hi, ...
                sprintf('K%d',k), D.K(k), P.dof_scale.conic, P.bounds.conic);
        end
    end
    if any(strcmp(dofs,'standoff'))
        if ~strcmp(D.form,'field')
            error('macos:design:afocal4_solve:standoff', ...
                  'the field-mirror standoff is a DOF of the "field" form only.');
        end
        [x,names,scale,base,lo,hi] = add_(x,names,scale,base,lo,hi, ...
            's_FM', D.fm_standoff, P.dof_scale.standoff, P.bounds.fm_standoff);
    end
    if any(strcmp(dofs,'front'))
        if ~strcmp(D.form,'field')
            error('macos:design:afocal4_solve:front', ...
                  'the front-end DOFs belong to the "field" form only.');
        end
        [x,names,scale,base,lo,hi] = add_(x,names,scale,base,lo,hi, ...
            'R_M2', D.R2, P.dof_scale.radius, P.bounds.R2);
        [x,names,scale,base,lo,hi] = add_(x,names,scale,base,lo,hi, ...
            't_M1M2', D.t1, P.dof_scale.spacing, P.bounds.t1);
    end
    if any(strcmp(dofs,'rb'))
        for i = 1:numel(P.rb_elts)
            [x,names,scale,base,lo,hi] = add_(x,names,scale,base,lo,hi, ...
                sprintf('dy%d',P.rb_elts(i)), D.rb(i,1), P.dof_scale.dec, ...
                P.bounds.dec);
            [x,names,scale,base,lo,hi] = add_(x,names,scale,base,lo,hi, ...
                sprintf('tx%d',P.rb_elts(i)), D.rb(i,2), P.dof_scale.tilt, ...
                P.bounds.tilt);
        end
    end
    x = x(:);  scale = scale(:);  base = base(:);  lo = lo(:);  hi = hi(:);
end

function [x,nm,sc,ba,lo,hi] = add_(x,nm,sc,ba,lo,hi, name, val, s, bnd)
    x(end+1)  = 0;                    % the solver starts at the seed
    nm{end+1} = name;
    sc(end+1) = s;
    ba(end+1) = val;
    lo(end+1) = (bnd(1) - val)/s;
    hi(end+1) = (bnd(2) - val)/s;
end

function D = unpack_(P, D, dofs, xs, scale, base)
%UNPACK_  The solver's deviation vector -> physical values, in the SAME
%   order PACK_ built it.  The two must be read together; a mismatch here is
%   silent and produces a design nobody asked for.
    xs = base(:) + xs(:).*scale(:);   j = 0;
    ki = conic_free_(D.form);
    if any(strcmp(dofs,'conic'))
        for k = ki, j = j + 1;  D.K(k) = xs(j); end
    end
    if any(strcmp(dofs,'standoff'))
        j = j + 1;   D.fm_standoff = xs(j);
    end
    if any(strcmp(dofs,'front'))
        D.R2 = xs(j+1);   D.t1 = xs(j+2);   j = j + 2;
    end
    if any(strcmp(dofs,'rb'))
        for i = 1:numel(P.rb_elts)
            D.rb(i,1) = xs(j+1);   D.rb(i,2) = xs(j+2);   j = j + 2;
        end
    end
end

function ki = conic_free_(form)
%CONIC_FREE_  M1 is HELD a parabola on the field form -- his study holds it
%   too, and it is the one surface whose figure the whole benchmark shares.
%   The Mersenne hedge exists precisely to spend all four, so it does.
    switch form
    case 'field',    ki = [2 3 4];
    case 'mersenne', ki = [1 2 3 4];
    otherwise, error('macos:design:afocal4_solve:form','unknown form.');
    end
end

function r = resid_linear_(P, S)
%RESID_LINEAR_  The rejected alternative, kept runnable so the rule that
%   replaced it can be MEASURED rather than asserted (afocal4_ladder
%   'merit_ab').  Each term enters as m/t, so a 200x WFE miss is 200 and a
%   7x pupil miss is 7: in a sum of squares the pupil terms are 1 part in
%   1000 of the merit and the solve is free to trade them away entirely.
    T = P.targets;   W = P.weights;
    K = numel(S.wfe_nm);
    r = [ (W.wfe/sqrt(K)) * (S.wfe_nm(:)/T.wfe_rung2_nm); ...
          W.blur    * S.norm.blur; ...
          W.breathe * S.norm.breathe; ...
          W.wander  * S.norm.wander; ...
          W.surf_pv * S.norm.surf_pv; ...
          W.mag     * S.norm.mag ];
    r(~isfinite(r)) = 1e4;
end

function del_(p),  if exist(p,'file'), delete(p); end,  end
