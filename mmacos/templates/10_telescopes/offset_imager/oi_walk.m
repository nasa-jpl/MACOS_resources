function OUT = oi_walk(over, varargin)
%OI_WALK  Solve a hard offset-imager instance by PARAMETER CONTINUATION.
%
%   OUT = OI_WALK(OVER) solves the offset_imager instance described by the
%   override struct OVER (the same struct OI_STORY/OFFSET_IMAGER take) by
%   WALKING the field box open: it solves an easy NARROW-box instance at
%   the offset first, then widens the box in steps, carrying the solved
%   design X as the WARM START of each step.  The staged ladder
%   (OFFSET_IMAGER S1..S5) is pedagogy; this runner is a solution-finder.
%
%   OUT = OI_WALK(OVER, 'steps', [5 8 11 13 15], ...) sets the walk
%   schedule.  Name-value:
%     'walk'      the continuation axis (default 'box_deg' -- the ONLY
%                 supported axis: box width shrinks aberration span AND
%                 clearance need monotonically; walking the offset down
%                 re-enters the t4 field-walk infeasibility, PACKET
%                 2026-08-21 addendum).
%     'steps'     vector of box FULL-WIDTHS (deg) to walk through, low to
%                 high.  Each width w scales the target box:
%                 box_deg = target_box * (w / max(target_box)).  The
%                 target width is appended if the schedule omits it, so
%                 the FINAL step is always the target instance.  Default:
%                 the target width alone (a single cold solve -- no walk).
%     'min_step'  smallest width increment (deg) the adaptive screen will
%                 take before declaring the box untraceable (default 0.5).
%     'baseline_nm'  a reference map max (nm) to quote the final against in
%                 the report (default [] = no comparison line).
%   Any other name-value pair is folded into OVER as a parameter override
%   (OFFSET_IMAGER_PARAMS errors on unknown keys, so 'walk'/'steps'/
%   'min_step'/'baseline_nm' are consumed here and never passed through).
%
%   THE STEP RULE (the F8 lesson, t5 unguided experiment): the step size
%   is bounded by the TRACEABILITY radius.  Before solving at a widened
%   box, the carried X is SCREENED at that box (START_QMEAN_ over the
%   solve set, without solving).  If it scores the 1e9 no-rays sentinel
%   the step is too big; the increment is HALVED (a waypoint inserted) and
%   re-screened, adaptively, until it traces or the increment falls below
%   'min_step' -- then the walk stops and reports an honest failure at
%   that box.  It never proceeds from a design it has already scored
%   untraceable.
%
%   FULL FREEDOM FROM STEP 1: every step is an S5 solve (conics + Zernike
%   departures + tilt/decenter + radii R1,R2 + stop_y), with the exit-
%   direction equality row and the signed clearance hinge rows active.
%   Identities (EFL, stop/FP poses) are re-derived per step via OI_CLOSE;
%   the stop pose is CONSTRUCTED ONCE at step 1 (POSE_STOP_ONCE_) and
%   FROZEN thereafter (the frozen-pose lesson) -- from then on the stop
%   decenter is an explicit solve variable.
%
%   Metric (quoted with every number, the packet contract): strict RMS
%   WFE, reference sphere on the spot centroid on the step's frozen FPA,
%   anchored at the exit pupil, piston-only removal (design/src strict
%   kernel); the headline per step is the MAXIMUM of the dense P.map_n x
%   P.map_n map over the box (solve set != scoring set).
%
%   Artifacts in P.outdir: per-step decks <tag>_k*.in, figures
%   <tag>_k*_{layout,map}.png, the walk report <tag>_REPORT.md, and
%   <tag>_run.mat (re-saved after EACH step -- a killed walk keeps its
%   finished steps).
%
%   Example (the t5 redemption instance the cold start could not solve):
%     OUT = oi_walk(struct('name','t5-walk','tag','t5_walk', ...
%             'EPD_m',0.150,'Fno',3.3,'box_deg',[15 15],'offset_deg',22.5, ...
%             'z_m1_m',0.6649568*1.65,'spacings_m',[-0.7228968 0 0.7408280]*1.65, ...
%             'seed_R1_m',8.8*1.65,'clear_m',[0.040 0.025],'exit_dir',[0 0 -1], ...
%             's1_target_nm',159,'nsolve_s5',5,'gn_iters',30), ...
%             'steps',[5 8 11 13 15]);
%
%   See also OFFSET_IMAGER, OI_STORY, OI_SOLVE, OI_CLOSE, OFFSET_IMAGER_PARAMS,
%   run_s5_budget (the leg_ solve-from-X pattern this reuses).

    if nargin < 1 || isempty(over), over = struct(); end
    if ~isstruct(over)
        error('oi_walk:over', 'the first argument must be an override struct');
    end

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    addpath(here);

    % ---- parse name-value: 'walk'/'steps'/'min_step'/'baseline_nm' are ours;
    %      everything else folds into the override struct ----------------------
    walk_axis   = 'box_deg';
    steps       = [];
    min_step    = 0.5;
    baseline_nm = [];
    kv = varargin;
    if mod(numel(kv), 2) ~= 0
        error('oi_walk:nv', 'name-value arguments must come in pairs');
    end
    for i = 1:2:numel(kv)
        key = kv{i};  val = kv{i+1};
        switch key
            case 'walk',        walk_axis   = val;
            case 'steps',       steps       = val;
            case 'min_step',    min_step    = val;
            case 'baseline_nm', baseline_nm = val;
            otherwise,          over.(key)  = val;   % a parameter override
        end
    end
    if ~strcmp(walk_axis, 'box_deg')
        error('oi_walk:axis', ...
            ['walk axis "%s" is not supported -- only ''box_deg''.  Walking ' ...
             'the offset re-enters the t4 field-walk infeasibility (PACKET ' ...
             '2026-08-21).'], walk_axis);
    end

    % ---- the target parameter set ------------------------------------------
    P = offset_imager_params(over);
    if isempty(P.outdir), P.outdir = here; end
    if ~exist(P.outdir,'dir'), mkdir(P.outdir); end
    tag = fullfile(P.outdir, P.tag);

    % every walk step is an S5 (Zernike) solve; size the solve set to the
    % variable count (the S5 lesson: a 3x3 grid under-determines 80+ Zernike
    % vars and the dense map stalls).  nsolve_s5 drives it when set.
    if ~isempty(P.nsolve_s5), P.nsolve = P.nsolve_s5; end

    % ---- the walk schedule: box FULL-WIDTHS up to the target ----------------
    full = max(P.box_deg);
    if isempty(steps), steps = full; end
    steps = sort(unique(steps(:).'));
    steps = steps(steps > 0 & steps <= full + 1e-9);
    if isempty(steps) || abs(steps(end) - full) > 1e-9
        steps = [steps(steps < full - 1e-9), full];   % final step = target
    end
    boxfor = @(w) P.box_deg * (w / full);   % width -> (possibly non-square) box

    macos.init(P.model);

    fprintf(['\n=== oi_walk: %s -- continuation over box width ' ...
             '[%s] deg (target %gx%g at YAN %+g) ===\n'], P.name, ...
            strtrim(sprintf('%g ', steps)), P.box_deg, P.offset_deg);

    OUT = struct('P',P, 'steps',steps, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    rec = rec_init_();
    stall = [];
    Xbest = [];

    % ======================= step 1: bootstrap ==============================
    % Delegate the narrow-box bootstrap to OFFSET_IMAGER's own S1..S5 ladder.
    % Its S3 TWO-CANDIDATE (carry-vs-fresh) start is what adapts the on-axis
    % symmetric seed to the offset field; a hand-rolled S1->S5 jump loses every
    % ray at a large offset (on-axis aspheres are poison off axis -- exactly
    % the mechanism the ladder's S3 screen exists to handle).  The narrow box
    % keeps the bootstrap inside the convergent basin (the walk's whole
    % premise: small aberration span, easy clearances).  The solved S5 state
    % is the walk's warm start; the bootstrap's own artifacts land in
    % <tag>_boot/.
    w  = steps(1);
    Pk = P;  Pk.box_deg = boxfor(w);
    banner_(sprintf('step 1/%d  bootstrap via the offset_imager ladder: box %gx%g deg', ...
            numel(steps), Pk.box_deg));
    bootover = over;                            % the target instance overrides
    bootover.box_deg = boxfor(w);               % ... at the narrow box
    bootover.tag     = [P.tag '_boot'];
    bootover.outdir  = fullfile(P.outdir, [P.tag '_boot']);
    B = [];
    try
        B = offset_imager(bootover);
    catch e
        stall = struct('step',1, 'width',w, 'box_deg',Pk.box_deg, 'qmean',NaN, ...
                       'n_halvings',0, ...
            'why', sprintf(['the narrowest box (%gx%g deg) could not be ' ...
                'bootstrapped (%s: %s) -- narrow the first step or revise the ' ...
                'envelope/offset/S1 cap'], Pk.box_deg, e.identifier, e.message));
    end
    if isempty(stall) && ~isfield(B,'s5')
        stall = struct('step',1, 'width',w, 'box_deg',Pk.box_deg, 'qmean',NaN, ...
                       'n_halvings',0, ...
            'why', 'the bootstrap ladder did not reach S5 (check P.stages)');
    end
    if isempty(stall)
        X = B.s5.X;
        g = leg_(X, Pk, [tag sprintf('_k%02d',1)], ...
                 sprintf('walk step 1 (bootstrap): box %gx%g deg', Pk.box_deg), ...
                 B.s5.hist);                    % pre-solved: score, do not re-solve
        X = g.X;  Xbest = X;
        rec(1) = mk_rec_(1, w, Pk, B.s5.hist.rms0, 0, g);
        report_step_(rec(1));
        save([tag '_run.mat'], 'OUT', 'rec', 'steps');  %#ok<*NASGU>
    end

    % ======================= steps 2..N: continuation ======================
    if isempty(stall)
        s_cur   = steps(1);
        stepnum = 1;
        ti      = 1;  targets = steps(2:end);
        while ti <= numel(targets)
            s_tgt = targets(ti);
            % adaptive screen: the largest reachable width in (s_cur, s_tgt]
            s_try = s_tgt;  nhalv = 0;
            Pt = P;  Pt.box_deg = boxfor(s_try);
            q = start_qmean_(X, Pt);
            while q >= 1e9
                s_new = 0.5*(s_cur + s_try);
                if (s_new - s_cur) < min_step
                    stall = struct('step',stepnum+1, 'width',s_try, ...
                        'box_deg',boxfor(s_try), 'qmean',q, 'n_halvings',nhalv, ...
                        'why', sprintf(['carried design will not trace a box ' ...
                            'wider than ~%gx%g deg (reached from %gx%g); the ' ...
                            'increment fell below min_step %g deg -- the walk ' ...
                            'is at the traceability radius'], boxfor(s_cur), ...
                            boxfor(s_cur), min_step));
                    break;
                end
                s_try = s_new;  nhalv = nhalv + 1;
                Pt.box_deg = boxfor(s_try);
                fprintf('  screen: box too big -> halve to %gx%g deg (halving %d)\n', ...
                        Pt.box_deg, nhalv);
                q = start_qmean_(X, Pt);
            end
            if ~isempty(stall), break; end

            stepnum = stepnum + 1;
            banner_(sprintf('step %d  box %gx%g deg  (target %g of %g%s)', ...
                    stepnum, Pt.box_deg, s_tgt, full, ...
                    tern_(nhalv>0, sprintf(', %d halving(s)', nhalv), '')));
            fprintf('  screen (carried start): qmean %.1f nm\n', q);
            g = leg_(X, Pt, [tag sprintf('_k%02d',stepnum)], ...
                     sprintf('walk step %d: box %gx%g deg', stepnum, Pt.box_deg));
            X = g.X;  Xbest = X;
            rec(end+1) = mk_rec_(stepnum, s_try, Pt, q, nhalv, g); %#ok<AGROW>
            report_step_(rec(end));
            save([tag '_run.mat'], 'OUT', 'rec', 'steps');

            s_cur = s_try;
            if abs(s_try - s_tgt) < 1e-9
                ti = ti + 1;      % target reached; advance the schedule
            end                   % else: re-attempt s_tgt from the new s_cur
        end
    end

    % ======================= verdict + report ==============================
    OUT.walk        = rec;
    nh = 0;  if ~isempty(rec), nh = sum([rec.n_halvings]); end
    if ~isempty(stall), nh = nh + stall.n_halvings; end
    OUT.n_halvings  = nh;                      % total across steps AND a stall
    OUT.stalled     = ~isempty(stall);
    OUT.stall       = stall;
    if ~isempty(Xbest), OUT.final = rec(end); end
    OUT.verdict     = verdict_(rec, stall);
    OUT.baseline_nm = baseline_nm;
    save([tag '_run.mat'], 'OUT', 'rec', 'steps');
    write_report_(tag, P, steps, rec, stall, OUT.verdict, baseline_nm);
    fprintf('\noi_walk: verdict %s -- saved %s_REPORT.md + %s_run.mat\n', ...
            OUT.verdict, tag, tag);
end

% =========================================================================
% The per-step solve-from-X body (the run_s5_budget leg_ pattern).  Warm
% start = the carried X struct itself; oi_solve optimizes scaled deltas
% about it.  Do NOT reset X.fpa_refit and do NOT re-seed the Zernike here
% (both are part of the warm state; oi_zern_seed would zero the carried
% coefficients -- it refits from aspheres, which are 0 after step 1).
function g = leg_(X0, P, stem, lbl, presolved)
    if nargin < 5, presolved = []; end
    X = X0;
    if isempty(presolved)
        [X, hist] = oi_solve(X, P, 'S5', 'clear', true);
    else
        hist = presolved;              % step 1: solved by the bootstrap ladder
    end
    [X, Gc]   = oi_close(X, P, 'offset_deg', P.offset_deg, 'repose_stop', false);
    X.fpa = oi_apply_fpa(X);  Gc.fpa = X.fpa;
    [png_m, mp] = oi_map_fig(X, Gc, P, P.offset_deg, lbl, [stem '_map.png']);
    png_l = oi_layout_fig(X, Gc, P, P.offset_deg, lbl, [stem '_layout.png']);
    txt = oi_deck(fill_(X, P));
    fdeck = [stem '.in'];
    fid = fopen(fdeck,'w');  fprintf(fid,'%s',txt);  fclose(fid);
    gt = oi_gates(X, Gc, P, P.offset_deg);
    g = struct('X0',X0, 'X',X, 'G',Gc, 'hist',hist, 'map',mp, 'gates',gt, ...
               'fig_map',png_m, 'fig_layout',png_l, 'deck',fdeck);
    fprintf('  step done: map max %s nm, clearance %.1f mm (%s), exit %s\n', ...
            mapmax_str_(mp), gt.clear_min_m*1e3, pf_(gt.clear_pass), pf_(gt.exit_pass));
end

% ---- record bookkeeping --------------------------------------------------
function r = rec_init_()
    r = struct('step',{},'width',{},'box_deg',{},'start_qmean',{}, ...
               'end_qmean',{},'map_max_nm',{},'map_valid',{},'map_nfailed',{}, ...
               'map_nfields',{},'clear_min_mm',{},'clear_pass',{}, ...
               'exit_err_deg',{},'exit_pass',{},'n_halvings',{},'iters',{}, ...
               'X0',{},'X',{},'hist',{},'gates',{},'map',{});
end

function r = mk_rec_(stepnum, width, P, qstart, nhalv, g)
    mp = g.map;  gt = g.gates;
    valid = ~isfield(mp,'valid') || mp.valid;
    r = struct( ...
        'step',stepnum, 'width',width, 'box_deg',P.box_deg, ...
        'start_qmean',qstart, 'end_qmean',g.hist.rms, ...
        'map_max_nm',mp.max_nm, 'map_valid',valid, ...
        'map_nfailed',getf_(mp,'n_failed',0), 'map_nfields',getf_(mp,'n_fields',NaN), ...
        'clear_min_mm',gt.clear_min_m*1e3, 'clear_pass',gt.clear_pass, ...
        'exit_err_deg',gt.exit_err_deg, 'exit_pass',gt.exit_pass, ...
        'n_halvings',nhalv, 'iters',g.hist.iters, ...
        'X0',g.X0, 'X',g.X, 'hist',g.hist, 'gates',gt, 'map',mp);
end

function report_step_(r)
    fprintf(['  [step %d] box %gx%g  start %.1f -> end %.1f nm (solve set), ' ...
             'map max %s nm, clear %.1f mm %s\n'], r.step, r.box_deg, ...
            r.start_qmean, r.end_qmean, mapmax_str_(r.map), r.clear_min_mm, ...
            pf_(r.clear_pass));
end

% ---- verdict -------------------------------------------------------------
function v = verdict_(rec, stall)
    if ~isempty(stall) || isempty(rec)
        v = 'FAIL';  return;
    end
    f = rec(end);
    if ~f.map_valid
        v = 'FAIL';                            % final map lost fields
    elseif f.exit_pass && f.clear_pass
        v = 'PASS';                            % completed, both gates clear
    else
        v = 'PARTIAL';                         % completed + valid, a gate fails
    end
end

% ---- report file ---------------------------------------------------------
function write_report_(tag, P, steps, rec, stall, verdict, baseline_nm)
    f = fopen([tag '_REPORT.md'], 'w');
    cs = onCleanup(@() fclose(f));
    pr = @(varargin) fprintf(f, varargin{:});

    pr('# %s -- offset_imager continuation walk\n\n', P.name);
    pr(['%s.  EPD %.0f mm, F/%.4g (EFL %.3f m held as an identity), ' ...
        'lambda %.2f um, target box %gx%g%c offset %+g%c, spacings [%g %g %g] m, ' ...
        'model %d, nGridpts %d.\n'], datestr(now,31), P.EPD_m*1e3, P.Fno, ...
        P.EFL_m, P.lambda_m*1e6, P.box_deg, char(176), P.offset_deg, char(176), ...
        P.spacings_m, P.model, P.sampling); %#ok<TNOW1,DATST>
    pr(['\nMetric (every number below): strict RMS WFE, sphere centred on the ' ...
        'spot centroid on the step''s frozen FPA, anchored at the exit pupil, ' ...
        'piston-only removal (design/src strict kernel); headline = dense %dx%d ' ...
        'map MAXIMUM over the box.  Solve set %dx%d != scoring set.\n\n'], ...
        P.map_n, P.map_n, P.nsolve, P.nsolve);
    pr(['Continuation: walk the box FULL-WIDTH open along [%s] deg at fixed ' ...
        'offset %+g%c, carrying the solved X as each step''s warm start; every ' ...
        'step is a full-freedom S5 solve (conics+Zernike+tilt/dec+radii+stop_y) ' ...
        'with the exit row + signed clearance rows active.  A carried design is ' ...
        'SCREENED at the widened box before solving; a 1e9 no-rays score halves ' ...
        'the step (F8 rule).\n\n'], strtrim(sprintf('%g ', steps)), ...
        P.offset_deg, char(176));

    pr('## Verdict\n\n**%s.**  ', verdict);
    switch verdict
        case 'PASS'
            pr(['The walk reached the target box and both gates (exit ' ...
                'direction, signed clearance) pass.\n']);
        case 'PARTIAL'
            pr(['The walk reached the target box with a valid dense map but a ' ...
                'gate still fails (see the final row) -- the aberration walk ' ...
                'succeeded; the failing gate is the remaining, separately ' ...
                'stated, constraint.\n']);
        case 'FAIL'
            if ~isempty(stall)
                pr(['The walk stopped short: %s.  It did not proceed blind ' ...
                    '(the F8 rule).\n'], stall.why);
            else
                pr(['The final step''s dense map lost fields (INVALID) -- the ' ...
                    'target box is not traceable by the design the walk ' ...
                    'reached.\n']);
            end
    end
    if ~isempty(rec)
        last = rec(end);
        if ~isempty(baseline_nm) && last.map_valid
            % report the improvement as a factor: a walk that turns a
            % 6e5 nm cold-start stall into tens of nm is 1/8000x, which
            % %.2f rounds to a misleading "0.00x" -- quote the factor.
            r = last.map_max_nm / max(baseline_nm, eps);
            if r < 0.1
                pr(['\nFinal dense-map max **%.1f nm** vs the cold-start ' ...
                    'baseline %.1f nm -- **%.0fx better** (ratio %.2g).\n'], ...
                    last.map_max_nm, baseline_nm, baseline_nm/max(last.map_max_nm,eps), r);
            else
                pr(['\nFinal dense-map max **%.1f nm** vs the cold-start ' ...
                    'baseline %.1f nm (%.2fx).\n'], last.map_max_nm, baseline_nm, r);
            end
        elseif last.map_valid
            pr('\nFinal dense-map max **%.1f nm** at the target box.\n', last.map_max_nm);
        end
    end

    pr('\n## The walk, step by step\n\n');
    pr(['| step | box (deg) | halvings | start qmean (nm) | end qmean (nm) | ' ...
        'map max (nm) | clear floor (mm) | exit err (deg) | gates |\n']);
    pr('|---|---|---|---|---|---|---|---|---|\n');
    for i = 1:numel(rec)
        r = rec(i);
        pr('| %d | %g x %g | %d | %s | %s | %s | %.1f | %s | exit %s / clear %s |\n', ...
           r.step, r.box_deg(1), r.box_deg(2), r.n_halvings, ...
           qfmt_(r.start_qmean), qfmt_(r.end_qmean), mapmax_str_(r.map), ...
           r.clear_min_mm, efmt_(r.exit_err_deg), pf_(r.exit_pass), pf_(r.clear_pass));
    end
    if ~isempty(stall)
        pr('| %d | %g x %g | %d | %s | -- | STALL | -- | -- | untraceable |\n', ...
           stall.step, stall.box_deg(1), stall.box_deg(2), stall.n_halvings, ...
           qfmt_(stall.qmean));
    end
    pr(['\nGate thresholds: exit-chief direction within %.1f%c of the pin; ' ...
        'signed clearance floor >= %.0f mm (WARN < %.0f mm).  The clearance ' ...
        'model is SIGNED (design/src/oi_clear): a negative floor is beam-mirror ' ...
        'interference depth, not a distance.\n'], P.exit_tol_deg, char(176), ...
        min(P.clear_m)*1e3, max(P.clear_m)*1e3);
end

% ---- small local helpers (copied from offset_imager.m / run_s5_budget.m) --
function q = start_qmean_(X, P)
%START_QMEAN_  Quadratic-mean strict WFE of a carried design over the SOLVE
%   set at the offset, WITHOUT solving (copied from offset_imager.m).  The
%   traceability screen: 1e9 = the no-rays sentinel (step too big).
    try
        [Xc, G] = oi_close(X, P, 'offset_deg', P.offset_deg);
        Xc.fpa = oi_apply_fpa(Xc);  G.fpa = Xc.fpa;
        D = fill_(Xc, P);
        if isfield(P,'solve_sampling') && ~isempty(P.solve_sampling)
            D.sampling = P.solve_sampling;
        end
        sc = oi_score(oi_deck(D), G, oi_fieldset(P, P.offset_deg, P.nsolve), ...
                      'anchor','center');
        w = sc.wfe_cen_nm;
        q = sqrt(mean(w(isfinite(w)).^2));
        if ~isfinite(q), q = 1e9; end
    catch
        q = 1e9;
    end
end

function D = fill_(X, P)
    D = X;
    D.EPD_m = P.EPD_m;  D.WL_m = P.lambda_m;
    D.sampling = P.sampling;  D.name = P.name;
end

function banner_(s)
    fprintf('\n-----------------------------------------------------------------\n');
    fprintf(' %s\n', s);
    fprintf('-----------------------------------------------------------------\n');
end

function s = mapmax_str_(mp)
    if isfield(mp,'valid') && ~mp.valid
        s = sprintf('INVALID(%d/%d lost)', mp.n_failed, mp.n_fields);
    else
        s = sprintf('%.1f', mp.max_nm);
    end
end

function s = qfmt_(q)
    if q >= 1e9, s = '1e9 (no rays)'; else, s = sprintf('%.1f', q); end
end

function s = efmt_(e)
    if isnan(e), s = 'n/a'; else, s = sprintf('%.3f', e); end
end

function v = getf_(s, f, d)
    if isfield(s,f), v = s.(f); else, v = d; end
end

function s = tern_(c,a,b), if c, s=a; else, s=b; end, end
function s = pf_(p), if p, s = 'PASS'; else, s = 'FAIL'; end, end
