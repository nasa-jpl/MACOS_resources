function [X, hist] = oi_solve(X, P, stage, opts)
%OI_SOLVE  Damped Gauss-Newton stage solve of the offset_imager ladder.
%
%   [X, HIST] = OI_SOLVE(X, P, STAGE) optimizes the design struct X for
%   one template stage.  The residual vector is the STACKED PER-RAY
%   centroid-rung residual wavefronts (nm; strict_rungs column 2, piston
%   out per field) over the SOLVE SET (an nsolve x nsolve grid across
%   the field box; solve set != scoring set) -- true Gauss-Newton: the
%   per-ray OPD is nearly linear in the surface coefficients, where a
%   per-field RMS is not.  When P.exit_dir is set, the exit-chief
%   direction error rides as one weighted residual row (an equality
%   constraint, S3+).  Every iterate passes through the first-order
%   closure (OI_CLOSE), so EFL (and Petzval = 0 in the symmetric
%   stages) and the stop/FP constructions are IDENTITIES of every
%   design the optimizer ever sees.
%
%   STAGE and its variable set:
%     'S1'   on-axis box:  K(1:3), asph(3x3), R(1:2), fpa [dz tilt]
%     'fpa'  offset box:   fpa [dz tilt] ONLY (the S2 "refit only" rung)
%     'S3'   offset box:   the S1 set, solved AT the offset field
%     'S4'   offset box:   S3 set + yde(1:3) + ade(1:3)
%     'S5'   offset box:   Zernike coefs (P.zern_modes per mirror,
%            REPLACING the aspheres) + K + R(1:2) + fpa.  lMon is FROZEN
%            at stage start from the traced footprints (the solve
%            doctrine: lMon fixed, power pinned to radii).
%
%   Damped chord-GN with per-variable natural scales (asphere / conic /
%   Zernike scales are waves-of-sag at the mirror's own footprint edge),
%   forward-FD Jacobian reused for up to 3 accepted steps, Levenberg
%   damping, trust radius.  Genuine packaging walls reject steps
%   (afocal4 rule); smooth equality constraints ride the residual.
%
%   Name-value:
%     'iters'     iteration cap (default P.gn_iters)
%     'offset'    box-centre YAN, deg (default: 0 for S1, else P.offset_deg)
%     'walls'     function handle w(X,G) -> true if X violates a wall
%     'quiet'     false
%
%   HIST: .rms0/.rms (nm, quadratic-mean over solve set), .iters,
%   .accepted, .vars (names), .x0/.x (scaled).
%
%   See also OI_CLOSE, OI_SCORE, OI_DECK, OFFSET_IMAGER.

    arguments
        X struct
        P struct
        stage (1,:) char
        opts.iters (1,1) double = NaN
        opts.offset (1,1) double = NaN
        opts.walls = []
        opts.quiet (1,1) logical = false
    end
    if isnan(opts.iters), opts.iters = P.gn_iters; end
    if isnan(opts.offset)
        if strcmp(stage,'S1'), opts.offset = 0; else, opts.offset = P.offset_deg; end
    end

    % ---- solve set: nsolve x nsolve over the box -------------------------
    F = oi_fieldset(P, opts.offset, P.nsolve);

    % ---- footprints for the natural scales (frozen at stage start) -------
    h = footprints_(X, P, opts.offset);
    lam = P.lambda_m;

    % ---- variable spec -----------------------------------------------------
    V = varspec_(stage, X, P, h, lam);

    % ---- residual function --------------------------------------------------
    rfun = @(u) residual_(u, V, X, P, opts.offset, F, opts.walls, 0);

    % ---- Levenberg-damped GN -------------------------------------------------
    u = zeros(numel(V),1);                 % scaled deltas about the incoming X
    [r, Xbest, rq] = rfun(u);
    len0 = numel(r);
    rfun = @(u) residual_(u, V, X, P, opts.offset, F, opts.walls, len0);
    rms0 = rq;  rmsb = rms0;
    lamLM = 1e-3;  du_fd = 1e-2;
    hist = struct('rms0',rms0,'iters',0,'accepted',0, ...
                  'vars',{{V.name}}, 'rms_path',rms0);
    if ~opts.quiet
        fprintf('  oi_solve %-4s: %2d vars, %d fields, start %10.3f nm\n', ...
                stage, numel(V), size(F,1), rms0);
    end

    for it = 1:opts.iters
        % Jacobian (forward FD in the scaled space)
        J = zeros(numel(r), numel(V));
        for j = 1:numel(V)
            up = u;  up(j) = up(j) + du_fd;
            rj = rfun(up);
            J(:,j) = (rj - r)/du_fd;
        end
        % LM steps: try decreasing damping first.  CHORD GN: one J is
        % reused for up to 3 accepted steps (the Zernike stage is near-
        % linear, so the Jacobian barely moves between iterates).
        ok = false;
        dJ = diag(J.'*J);
        dJ = max(dJ, 1e-6*max(dJ) + 1e-12);   % floor null directions
        nacc = 0;
        for k = 1:8
            A  = J.'*J + lamLM*diag(dJ);
            du = -A\(J.'*r);
            % trust region in the scaled space (scales are natural units)
            nd = norm(du);
            if nd > 30, du = du*(30/nd); end
            [rn, Xn, rqn] = rfun(u + du);
            if qmean_(rn) < qmean_(r)
                u = u + du;  r = rn;  Xbest = Xn;  rmsb = rqn;
                lamLM = max(lamLM/3, 1e-7);
                ok = true;  nacc = nacc + 1;
                if nacc >= 3, break; end
            else
                lamLM = lamLM*8;
                if ok, break; end      % chord exhausted -- fresh J next
            end
        end
        hist.iters = it;  hist.rms_path(end+1) = rmsb; %#ok<AGROW>
        if ok, hist.accepted = hist.accepted + 1; end
        if ~opts.quiet
            fprintf('    it %2d: %10.3f nm   (lam %.1e%s)\n', it, rmsb, lamLM, ...
                    tern_(ok,'',' -- step rejected'));
        end
        if ~ok, break; end
        if it > 2 && (hist.rms_path(end-1) - rmsb) < 1e-3*rmsb, break; end
    end

    X = Xbest;
    hist.rms = rmsb;
end

% =========================================================================
function [r, Xc, rq] = residual_(u, V, X, P, offset, F, walls, len0)
%RESIDUAL_  Stacked per-ray centroid-rung residual wavefronts (nm) --
%   TRUE Gauss-Newton residuals (per-ray OPD is nearly linear in the
%   surface coefficients; a per-field RMS is not, and GN on it stalls).
%   rq = quadratic mean of the per-field strict RMS (the human-readable
%   merit).  Walls return 1e9 rows of the base length so J stays
%   well-formed across evals.
    if len0 == 0, len0 = size(F,1); end
    rq = 1e9;
    Xc = apply_(u, V, X);
    try
        [Xc, G] = oi_close(Xc, P, 'offset_deg', offset);
    catch
        % a candidate the closure cannot even close is a wall, not an error
        r = 1e9*ones(len0,1);
        return
    end
    Xc.fpa = oi_apply_fpa(Xc);       % FPA refit deltas ride the base pose
    G.fpa  = Xc.fpa;
    if ~isempty(walls) && walls(Xc, G)
        r = 1e9*ones(len0,1);        % wall: reject the step
        return
    end
    D = fill_(Xc, P);
    if isfield(P,'solve_sampling') && ~isempty(P.solve_sampling)
        D.sampling = P.solve_sampling;   % coarse grid in the loop only
    end
    txt = oi_deck(D);
    sc = oi_score(txt, G, F, 'anchor','center', 'resid',true);
    r = sc.resid;
    % the exit-direction row (appended below) is part of the base length
    nx = double(isfield(P,'exit_dir') && ~isempty(P.exit_dir) && abs(offset) > 1e-12);
    if isempty(r) || (len0 > size(F,1) && numel(r) ~= len0 - nx)
        r = 1e9*ones(len0,1);
        return
    end
    r(~isfinite(r)) = 1e9;
    % exit-direction EQUALITY constraint as a weighted residual row
    % (offset stages only -- on axis it is satisfied by symmetry)
    if nx > 0
        [~, ic] = min(vecnorm(F - [0 offset], 2, 2));
        ed = P.exit_dir(:)/norm(P.exit_dir);
        dc = sc.chief_dir(:,ic);
        if any(~isfinite(dc))
            err_deg = 1e3;
        else
            err_deg = acosd(max(-1, min(1, dot(ed, dc))));
        end
        r = [r; P.exit_wt * err_deg];
    end
    w = sc.wfe_cen_nm;
    rq = sqrt(mean(w(isfinite(w)).^2));
    if ~isfinite(rq), rq = 1e9; end
end

function X = apply_(u, V, X)
    for j = 1:numel(V)
        v = V(j);
        val = v.get(X) + u(j)*v.scale;
        X = v.set(X, val);
    end
end

function V = varspec_(stage, X, P, h, lam)
%VARSPEC_  Stage variable list with natural scales.
%   Asphere term i scale: coefficient giving 0.2 waves of sag at the
%   footprint edge h: s = 0.2*lam/h^(2i+2).  Zernike coef scale: 0.2*lam.
    V = struct('name',{},'scale',{},'get',{},'set',{});
    addv = @(V,name,scale,get,set) [V, struct('name',name,'scale',scale, ...
                                              'get',get,'set',set)];
    geom = ~strcmp(stage,'fpa');
    symm = strcmp(stage,'S1') || strcmp(stage,'S3');
    if geom
        for m = 1:3
            % conic natural scale: dK giving 0.2 waves of sag at the
            % footprint edge (sag_K ~ K*c^3*h^4/8); capped -- a nearly
            % flat mirror's conic is optically irrelevant at any K
            sK = min(50, 0.2*lam*8*abs(X.R(m))^3 / h(m)^4);
            V = addv(V, sprintf('K%d',m), sK, ...
                @(X) X.K(m), @(X,v) setfield_(X,'K',m,v)); %#ok<*GFLD>
        end
        if symm
            % S1/S3: Petzval = 0 is a closure IDENTITY (R2,R3 eliminated)
            V = addv(V, 'R1', 5e-2, ...
                @(X) X.R(1), @(X,v) setfield_(X,'R',1,v));
        else
            for m = 1:2                      % R3 is the EFL eliminator
                V = addv(V, sprintf('R%d',m), 1e-2, ...
                    @(X) X.R(m), @(X,v) setfield_(X,'R',m,v));
            end
        end
        if strcmp(stage,'S5')
            nm = numel(P.zern_modes);
            for m = 1:3
                for q = 1:nm
                    V = addv(V, sprintf('Z%d_%d',m,P.zern_modes(q)), 0.2*lam, ...
                        @(X) X.zern{m}.coef(q), ...
                        @(X,v) setzern_(X,m,q,v));
                end
            end
        else
            for m = 1:3
                for i = 1:3
                    V = addv(V, sprintf('A%d_%d',m,i), 0.2*lam/h(m)^(2*i+2), ...
                        @(X) X.asph(m,i), @(X,v) setasph_(X,m,i,v));
                end
            end
        end
        if strcmp(stage,'S4') || strcmp(stage,'S5')
            for m = 1:3
                V = addv(V, sprintf('yde%d',m), 1e-3, ...
                    @(X) X.yde(m), @(X,v) setfield_(X,'yde',m,v));
                V = addv(V, sprintf('ade%d',m), 5e-2, ...
                    @(X) X.ade(m), @(X,v) setfield_(X,'ade',m,v));
            end
        end
    end
    % FPA refit rides every stage (dz along the chief-normal, tilt about x)
    V = addv(V, 'fpa_dz',   1e-3, @(X) fpa_get_(X,1), @(X,v) fpa_set_(X,1,v));
    V = addv(V, 'fpa_tilt', 1e-1, @(X) fpa_get_(X,2), @(X,v) fpa_set_(X,2,v));
end

function X = setfield_(X, f, m, v),  X.(f)(m) = v;  end
function X = setasph_(X, m, i, v),   X.asph(m,i) = v;  end
function X = setzern_(X, m, q, v),   X.zern{m}.coef(q) = v;  end

% The FPA refit deltas live in X.fpa_refit = [dz_m, tilt_deg]; OI_CLOSE
% constructs the base pose, then OFFSET_IMAGER applies the refit AFTER
% closure via oi_apply_fpa.  Getter/setter here only touch the deltas.
function v = fpa_get_(X, i)
    if ~isfield(X,'fpa_refit'), v = 0; else, v = X.fpa_refit(i); end
end
function X = fpa_set_(X, i, v)
    if ~isfield(X,'fpa_refit'), X.fpa_refit = [0 0]; end
    X.fpa_refit(i) = v;
end

function h = footprints_(X, P, offset)
%FOOTPRINTS_  Traced beam footprint radius at each mirror (box centre).
    tmp = [tempname '.in'];
    cu  = onCleanup(@() delete_if_(tmp));
    [Xc, ~] = oi_close(X, P, 'offset_deg', offset);
    txt = oi_deck(fill_(Xc, P));
    cdir = [tand(0); tand(offset); 1];  cdir = cdir/norm(cdir);
    v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));
    z_stop = Xc.z_m1 + Xc.spacings(1);
    cdR = [cdir(1); cdir(2); -cdir(3)];
    tq  = (Xc.z_m1 - z_stop)/cdir(3);
    q   = Xc.stopC - tq*cdR;
    p0  = q - (0.75/cdir(3))*cdir;
    s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3(cdir)]);
    s = regexprep(s,    '(ChfRayPos=\s*)[^\n]*', ['$1' v3(p0)]);
    fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
    macos.load_rx(tmp);
    macos.stop(2,[0 0]);
    h = nan(1,3);
    ie = [1 3 4];                        % M1, M2, M3 in the 5-elt train
    for m = 1:3
        tr = macos.trace(ie(m));
        ri = macos.get_ray_info(tr.nRays);
        ok = ri.ok_trace(:) & ri.ok_pass(:);  ok(1) = false;
        Q = ri.pos(:,ok);
        h(m) = max(vecnorm(Q(1:2,:) - mean(Q(1:2,:),2), 2, 1));
    end
end

function D = fill_(X, P)
    D = X;
    D.EPD_m = P.EPD_m;  D.WL_m = P.lambda_m;
    D.sampling = P.sampling;  D.name = P.name;
end

function q = qmean_(r), q = sqrt(mean(r.^2)); end
function s = tern_(c,a,b), if c, s=a; else, s=b; end, end
function delete_if_(p), if exist(p,'file'), delete(p); end, end
