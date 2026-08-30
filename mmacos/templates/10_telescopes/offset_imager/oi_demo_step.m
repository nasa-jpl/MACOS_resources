function OUT = oi_demo_step(width_deg, varargin)
%OI_DEMO_STEP  ONE warm-started continuation step to an ADJACENT field box.
%
%   OUT = OI_DEMO_STEP(WIDTH_DEG) takes a field-box FULL WIDTH in degrees,
%   warm-starts from the nearest committed step of the OI_WALK frontier
%   (t5_walk/t5_walk_run.mat), states the PREDICTED outcome from that
%   frontier BEFORE solving, runs ONE full-freedom S5 continuation step at
%   the asked box, and prints a compact verdict block (dense-map max,
%   clearance floor, exit error, gates) with the frontier bracket beside
%   it.  It is the live-demo face of the walk: the committed table is
%   COMPILED DESIGN KNOWLEDGE and this driver extends it, on demand, to a
%   box width nobody solved in advance.
%
%   The instrument is FIXED (the t5 instance of `run_t5_walk`): EPD 150 mm,
%   F/3.3, lambda 1 um, offset +22.5 deg, the x1.65 rodgers3 W-fold
%   envelope, exit chief pinned to [0 0 -1], clearances 40/25 mm.  The BOX
%   WIDTH is the only knob -- walking the offset re-enters the t4 field-walk
%   infeasibility (PACKET 2026-08-21), and `oi_walk` documents `box_deg` as
%   the only supported continuation axis.
%
%   Name-value:
%     'range_deg'  [lo hi] admissible box full-width, deg (default [5 15]
%                  -- the committed walk's own span).  An ask outside it is
%                  REFUSED with one accurate sentence; nothing is solved.
%     'gn_iters'   Gauss-Newton cap for the step (default 1 -- the pinned
%                  demo knob; the walk's record runs used 30).  A step off a
%                  solved neighbour is a POLISH, so iterations are the CHEAP
%                  axis to give up: measured at 12 deg, iterations 2..8 at a
%                  9-field grid bought 6% of dense-map max for +43 min,
%                  while going 9 -> 25 fields bought 14% for +16 min.
%     'solve_sampling'  nGridpts inside the solve loop (default 11 here vs
%                  the template's 21).  It is NOT a cost lever -- the solve
%                  is deck write/parse bound, and 11 vs 21 measured 29.51 vs
%                  29.49 min -- but at 12 deg it was better on all three
%                  reported axes (33.55 vs 35.23 nm, 24.94 vs 23.89 mm,
%                  0.0005 vs 0.0427 deg), so it is pinned on quality.
%     'nsolve'     solve grid is nsolve x nsolve (default 5, and it must be
%                  ODD -- see the two measured notes below).  Do NOT drop it
%                  to 3 to save time: 83 Zernike variables against 9 fields
%                  under-determine the solve, and the dense map stalls while
%                  the solve set converges (the recorded S5 lesson, and it
%                  reproduces here -- see the knob study in README).
%     'run_mat'    path to the committed walk record (default
%                  t5_walk/t5_walk_run.mat beside this file)
%     'outdir'     artifact directory (default demo_adjacent/ beside this
%                  file)
%     'tag'        artifact prefix.  DEFAULT IS TIMESTAMPED
%                  (oi_demo_<width>deg_<yyyymmdd_HHMMSS>) so a rerun never
%                  overwrites a figure a viewer may already be holding --
%                  pass an explicit tag only for a pre-generated fallback.
%     'quiet'      suppress the narration blocks (default false)
%   Any other name-value pair folds into the instance override struct
%   (OFFSET_IMAGER_PARAMS errors on unknown keys).
%
%   THE WARM-START RULE: the start is the LARGEST committed step STRICTLY
%   BELOW the ask (and the lowest committed step if the ask is at or below
%   it).  Strictly below, so that an ask AT a committed width is a real
%   continuation step rather than a re-score of an already-solved design --
%   the demo must show the driver working, not replaying.  It is never a
%   cold start: cold at this offset is the documented 595 um failure
%   (t5_redemption/t5r_REPORT.md).
%
%   TWO REFUSAL PATHS, both answering with one accurate sentence and both
%   returning OUT.refused = true rather than raising:
%     1  RANGE -- the ask is outside 'range_deg'.
%     2  TRACEABILITY -- the carried design does not trace the asked box
%        (the screen returns the 1e9 no-rays sentinel).  This is OI_WALK's
%        F8 rule: never proceed from a design already scored untraceable.
%
%   Metric, quoted with every number (the packet contract): strict RMS WFE,
%   reference sphere on the spot centroid on the frozen FPA, anchored at
%   the exit pupil, piston-only removal; the headline is the MAXIMUM of the
%   dense P.map_n x P.map_n map over the box.  Solve set != scoring set.
%
%   OUT: .width_deg .box_deg .refused .why .warm (step/width/source)
%        .screen_qmean_nm .hist .map .gates .verdict .predicted
%        .files (map/layout/fields/deck/verdict/mat) .elapsed_min .P
%
%   Example (the demo default):
%     OUT = oi_demo_step(12);
%
%   See also OI_WALK, RUN_T5_WALK, OI_SOLVE, OI_CLOSE, OFFSET_IMAGER_PARAMS,
%   demo_adjacent/REHEARSAL.md.

    t_all = tic;
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    addpath(here);

    % ---- our name-values; everything else folds into the instance override --
    range_deg = [5 15];
    gn_iters  = 1;
    nsolve    = 5;
    run_mat   = fullfile(here, 't5_walk', 't5_walk_run.mat');
    outdir    = fullfile(here, 'demo_adjacent');
    tag       = '';
    quiet     = false;
    show_     = usejava('desktop');   % pop the reveal windows on a desktop
    over      = struct();
    kv = varargin;
    if mod(numel(kv), 2) ~= 0
        error('oi_demo_step:nv', 'name-value arguments must come in pairs');
    end
    for i = 1:2:numel(kv)
        switch kv{i}
            case 'range_deg', range_deg = kv{i+1};
            case 'gn_iters',  gn_iters  = kv{i+1};
            case 'nsolve',    nsolve    = kv{i+1};
            case 'run_mat',   run_mat   = kv{i+1};
            case 'outdir',    outdir    = kv{i+1};
            case 'tag',       tag       = kv{i+1};
            case 'quiet',     quiet     = kv{i+1};
            case 'show',      show_     = kv{i+1};
            otherwise,        over.(kv{i}) = kv{i+1};
        end
    end
    say = @(varargin) fprintf_if_(~quiet, varargin{:});

    if nargin < 1 || isempty(width_deg), width_deg = 12; end
    if ~isnumeric(width_deg) || ~isscalar(width_deg) || ~isfinite(width_deg)
        error('oi_demo_step:width', ...
              'the box full-width must be a finite numeric scalar (deg)');
    end
    width_deg = double(width_deg);

    % nsolve MUST be ODD.  OI_SOLVE imposes the exit-direction equality on
    % the solve field NEAREST the box centre; an even grid has no centre
    % field, so the exit chief is pinned at an off-centre point and walks.
    % Measured at 12 deg (nsolve 4 vs 5, same warm start): exit error
    % 0.6981 vs 0.0005 deg and dense-map max 103.85 vs 33.55 nm.
    if mod(nsolve, 2) == 0
        warning('oi_demo_step:evenGrid', ...
            ['nsolve = %d is EVEN: the solve grid has no box-centre field, ' ...
             'so the exit-direction row is imposed off centre (measured at ' ...
             '12 deg: exit 0.70 deg, map 103.9 nm, vs 0.0005 deg / 33.6 nm ' ...
             'at nsolve 5).  Use an odd nsolve.'], nsolve);
    end

    if isempty(tag)
        tag = sprintf('oi_demo_%sdeg_%s', numtag_(width_deg), ...
                      datestr(now,'yyyymmdd_HHMMSS')); %#ok<TNOW1,DATST>
    end
    if ~exist(outdir,'dir'), mkdir(outdir); end
    stem = fullfile(outdir, tag);

    OUT = struct('width_deg',width_deg, 'refused',false, 'why','', ...
                 'when',datestr(now,31), 'tag',tag, 'outdir',outdir); %#ok<TNOW1,DATST>

    say('\n=================================================================\n');
    say(' oi_demo_step: adjacent-problem continuation -- box %.4g x %.4g deg\n', ...
        width_deg, width_deg);
    say('=================================================================\n');

    % ================== screen 1: the validated range =======================
    if width_deg < range_deg(1) - 1e-9 || width_deg > range_deg(2) + 1e-9
        OUT.refused = true;
        OUT.why = sprintf(['REFUSED: %.4g deg is outside this driver''s ' ...
            'validated continuation range (%g-%g deg box full-width at ' ...
            '+22.5 deg offset, the span the committed t5_walk frontier ' ...
            'actually traced), so there is no warm start to continue from ' ...
            'and a cold solve there is the documented 595 um failure -- ' ...
            'the honest answer is a full re-instance, which is an ' ...
            'overnight run.'], width_deg, range_deg(1), range_deg(2));
        say('\n%s\n\n', wrap_(OUT.why, 72));
        OUT.elapsed_min = toc(t_all)/60;
        return
    end

    % ================== the committed frontier ==============================
    if ~exist(run_mat,'file')
        error('oi_demo_step:runmat', ...
              'committed walk record not found: %s', run_mat);
    end
    S = load(run_mat);
    if ~isfield(S,'rec') || isempty(S.rec)
        error('oi_demo_step:rec', ...
              '%s carries no walk record (rec)', run_mat);
    end
    rec = S.rec;
    W   = [rec.width];

    % warm start = largest committed step STRICTLY below the ask (see help)
    k = find(W < width_deg - 1e-9, 1, 'last');
    if isempty(k), k = 1; end
    X = rec(k).X;
    OUT.warm = struct('step',rec(k).step, 'width',rec(k).width, ...
                      'box_deg',rec(k).box_deg, 'map_max_nm',rec(k).map_max_nm, ...
                      'clear_min_mm',rec(k).clear_min_mm, 'source',run_mat);

    say(['\nWarm start: committed walk step %d (%g x %g deg, %.1f nm map max, ' ...
         '%.1f mm floor)\n            from %s\n'], rec(k).step, rec(k).box_deg, ...
        rec(k).map_max_nm, rec(k).clear_min_mm, run_mat);

    % ================== the instance (fixed) + the demo knobs ===============
    base = struct( ...
        'name', sprintf('t5-demo %gx%g deg', width_deg, width_deg), ...
        'tag', tag, 'outdir', outdir, ...
        'EPD_m',0.150, 'Fno',3.3, ...
        'box_deg',[width_deg width_deg], 'offset_deg',22.5, ...
        'z_m1_m',0.6649568*1.65, ...
        'spacings_m',[-0.7228968 0 0.7408280]*1.65, ...
        'seed_R1_m',8.8*1.65, ...
        'clear_m',[0.040 0.025], 'exit_dir',[0 0 -1], ...
        's1_target_nm',159, 'nsolve_s5',nsolve, 'gn_iters',gn_iters, ...
        'solve_sampling',11);      % pinned on QUALITY, not cost -- see help
    fn = fieldnames(over);
    for i = 1:numel(fn), base.(fn{i}) = over.(fn{i}); end
    P = offset_imager_params(base);
    if ~isempty(P.nsolve_s5), P.nsolve = P.nsolve_s5; end   % the oi_walk rule
    OUT.P = P;  OUT.box_deg = P.box_deg;

    macos.init(P.model);

    % ================== screen 2: traceability (the F8 rule) ================
    q = start_qmean_(X, P);
    OUT.screen_qmean_nm = q;
    if q >= 1e9
        OUT.refused = true;
        OUT.why = sprintf(['REFUSED: the committed %g x %g deg design does ' ...
            'not trace a %.4g x %.4g deg box -- the screen returns the ' ...
            'no-rays sentinel, so this ask is past the traceability radius ' ...
            'from the nearest warm start, and the walk''s own F8 rule stops ' ...
            'here rather than reporting a partial map as if it were the ' ...
            'metric.'], rec(k).box_deg, width_deg, width_deg);
        say('\n%s\n\n', wrap_(OUT.why, 72));
        OUT.elapsed_min = toc(t_all)/60;
        return
    end
    say('Screen (carried design, no solve): qmean %.1f nm -- it traces.\n', q);

    % ================== the prediction, BEFORE the solve ====================
    pred = predict_(rec, width_deg);
    OUT.predicted = pred;
    if ~quiet, print_prediction_(pred, width_deg); end

    % ================== the step ============================================
    say(['\n--- solving: ONE full-freedom S5 continuation step ' ...
         '(%dx%d solve set, <=%d iters) ---\n'], P.nsolve, P.nsolve, P.gn_iters);
    t_solve = tic;
    [X, hist] = oi_solve(X, P, 'S5', 'clear', true);
    t_solve = toc(t_solve);

    % first-order closure, frozen stop (the carried pose is packaging)
    [X, G] = oi_close(X, P, 'offset_deg', P.offset_deg, 'repose_stop', false);
    X.fpa = oi_apply_fpa(X);  G.fpa = X.fpa;

    lbl = sprintf('adjacent step: box %g x %g deg', P.box_deg);
    t_score = tic;
    [png_m, mp] = oi_map_fig(X, G, P, P.offset_deg, lbl, [stem '_map.png']);
    [png_l, png_f] = oi_layout_fig(X, G, P, P.offset_deg, lbl, [stem '_layout.png']);
    gt = oi_gates(X, G, P, P.offset_deg);
    t_score = toc(t_score);

    fdeck = [stem '.in'];
    txt = oi_deck(fill_(X, P));
    fid = fopen(fdeck,'w');  fprintf(fid,'%s',txt);  fclose(fid);

    OUT.X = X;  OUT.G = G;  OUT.hist = hist;  OUT.map = mp;  OUT.gates = gt;
    OUT.verdict = verdict_(mp, gt);
    OUT.t_solve_min = t_solve/60;
    OUT.t_score_min = t_score/60;
    OUT.elapsed_min = toc(t_all)/60;
    OUT.files = struct('map',png_m, 'layout',png_l, 'fields',png_f, ...
                       'deck',fdeck, 'verdict',[stem '_verdict.txt'], ...
                       'mat',[stem '_run.mat']);

    % ================== the verdict block ===================================
    L = verdict_block_(OUT, P, rec, k, pred);
    fid = fopen(OUT.files.verdict,'w');
    fprintf(fid, '%s\n', L{:});
    fclose(fid);
    if ~quiet, fprintf('%s\n', L{:}); end

    save(OUT.files.mat, 'OUT');
    say('\nSaved: %s\n', OUT.files.mat);

    % ---- the reveal windows (a display error must never kill the run) ---
    if show_
        try
            oi_demo_show(OUT);
        catch me
            warning('oi_demo_step:show', 'reveal render failed: %s', ...
                    me.message);
        end
    end
end

% =========================================================================
% the frontier prediction
% =========================================================================
function pred = predict_(rec, w)
%PREDICT_  Bracket the ask with the committed walk rows and interpolate.
%   Both bracket rows are the COMMITTED numbers (never re-derived); the
%   expectation between them is a straight line, and is labelled as such.
    W = [rec.width];
    lo = find(W <= w + 1e-9, 1, 'last');
    hi = find(W >= w - 1e-9, 1, 'first');
    if isempty(lo), lo = 1; end
    if isempty(hi), hi = numel(W); end
    pred = struct('lo',rec(lo), 'hi',rec(hi), 'exact', lo == hi);
    if lo == hi
        pred.wfe_nm   = rec(lo).map_max_nm;
        pred.floor_mm = rec(lo).clear_min_mm;
        pred.how      = 'the committed row itself';
    else
        f = (w - W(lo)) / (W(hi) - W(lo));
        pred.wfe_nm   = rec(lo).map_max_nm   + f*(rec(hi).map_max_nm   - rec(lo).map_max_nm);
        pred.floor_mm = rec(lo).clear_min_mm + f*(rec(hi).clear_min_mm - rec(lo).clear_min_mm);
        pred.how      = 'linear between the bracketing committed rows';
    end
    % the walk's own caveats (t5_walk_REPORT.md + PACKET Sec B endgame).
    % Two DIFFERENT notes: the clearance knee (any wide ask) and the
    % under-solved 15 deg row (only when that row is actually quoted).
    pred.caveat = {};
    if w >= 13 - 1e-9
        pred.caveat{end+1} = ['the committed floors cross the 25 mm spec ' ...
            'knee between the 11 and 13 deg rows, so at this width ' ...
            'CLEARANCE is the binding story, not the wavefront -- and it ' ...
            'is priced: the endgame restart re-solve in this same envelope ' ...
            'lands 47.1 nm at a 30.89 mm floor'];
    end
    if max(W(lo), W(hi)) >= 15 - 1e-9
        pred.caveat{end+1} = ['the walk''s 15 deg row is UNDER-SOLVED ' ...
            '(6 iters -- the WFE-only plateau break fired while the ' ...
            'clearance hinge still pulled), so read it as an upper bound ' ...
            'on the WFE, not as the frontier''s best'];
    end
end

function print_prediction_(pred, w)
    fprintf('\n--- frontier prediction, stated BEFORE the solve ---\n');
    fprintf('  committed walk rows (t5_walk_REPORT.md), this envelope:\n');
    rows = {pred.lo};
    if ~pred.exact, rows{end+1} = pred.hi; end
    for i = 1:numel(rows)
        r = rows{i};
        fprintf('    step %d:  %-11s deg  ->  %6.1f nm map max, %5.1f mm floor\n', ...
                r.step, sprintf('%g x %g', r.box_deg), r.map_max_nm, r.clear_min_mm);
    end
    fprintf('  ASK %g x %g deg  ->  EXPECT ~%.1f nm and a floor ~%.1f mm\n', ...
            w, w, pred.wfe_nm, pred.floor_mm);
    fprintf('       (%s)\n', pred.how);
    for i = 1:numel(pred.caveat)
        fprintf('  NOTE: %s.\n', wrap_(pred.caveat{i}, 68));
    end
end

% =========================================================================
% the verdict block (printed AND written next to the figures)
% =========================================================================
function L = verdict_block_(OUT, P, rec, k, pred)
    mp = OUT.map;  gt = OUT.gates;  h = OUT.hist;
    L = {};
    a = @(varargin) sprintf(varargin{:});
    L{end+1} = '';
    L{end+1} = '=================================================================';
    L{end+1} = a(' VERDICT -- box %g x %g deg at YAN %+g deg   [%s]', ...
                 P.box_deg, P.offset_deg, OUT.verdict);
    L{end+1} = '=================================================================';
    L{end+1} = a(' metric  : strict RMS WFE, sphere on the spot centroid on the');
    L{end+1} = a('           frozen FPA, anchored at the exit pupil, piston-only');
    L{end+1} = a('           removal; headline = dense %dx%d map MAXIMUM over the', ...
                 P.map_n, P.map_n);
    L{end+1} = a('           box (solve set %dx%d != scoring set).', P.nsolve, P.nsolve);
    L{end+1} = '';
    L{end+1} = a(' warm start        : committed walk step %d (%g x %g deg)', ...
                 rec(k).step, rec(k).box_deg);
    L{end+1} = a(' screen (carried)  : qmean %.1f nm -- traces', OUT.screen_qmean_nm);
    L{end+1} = a(' solve             : S5 full freedom (conics+Zernike+tilt/dec');
    L{end+1} = a('                     +radii+stop_y), %d of <=%d iters used', ...
                 h.iters, P.gn_iters);
    L{end+1} = a(' solve-set qmean   : %.1f -> %.1f nm', h.rms0, h.rms);
    L{end+1} = '';
    L{end+1} = a(' DENSE MAP MAX     : %s nm      <-- the headline', mapmax_str_(mp));
    L{end+1} = a(' map avg / valid   : %s nm / %s', avg_str_(mp), valid_str_(mp));
    % Quote the threshold the gate ACTUALLY applies.  oi_gates passes at
    % min(clear_m) less a 1.5 mm hinge knee, so a bare "spec >= 25 ... PASS"
    % beside a 24.7 mm floor reads as a contradiction on stage.
    L{end+1} = a(' clearance floor   : %.1f mm   (spec %.0f, gate >= %.1f with the', ...
                 gt.clear_min_m*1e3, min(P.clear_m)*1e3, min(P.clear_m)*1e3 - 1.5);
    L{end+1} = a('                     1.5 mm hinge knee; WARN < %.0f)  %s%s', ...
                 max(P.clear_m)*1e3, pf_(gt.clear_pass), ...
                 tern_(gt.clear_warn && gt.clear_pass, ' (WARN)', ''));
    L{end+1} = a(' exit direction err: %s deg  (tol %.1f)  %s', ...
                 efmt_(gt.exit_err_deg), P.exit_tol_deg, pf_(gt.exit_pass));
    L{end+1} = a(' gates             : exit %s / clear %s', ...
                 pf_(gt.exit_pass), pf_(gt.clear_pass));
    L{end+1} = '';
    L{end+1} = a(' vs the frontier   : bracketing committed rows %g deg (%.1f nm /', ...
                 pred.lo.width, pred.lo.map_max_nm);
    L{end+1} = a('                     %.1f mm) and %g deg (%.1f nm / %.1f mm)', ...
                 pred.lo.clear_min_mm, pred.hi.width, pred.hi.map_max_nm, ...
                 pred.hi.clear_min_mm);
    if isfinite(mp.max_nm)
        L{end+1} = a('                     predicted ~%.1f nm -> measured %.1f nm (%.2fx)', ...
                     pred.wfe_nm, mp.max_nm, mp.max_nm/max(pred.wfe_nm,eps));
    else
        L{end+1} = a('                     predicted ~%.1f nm -> measured INVALID', pred.wfe_nm);
    end
    L{end+1} = a('                     predicted ~%.1f mm -> measured %.1f mm floor', ...
                 pred.floor_mm, gt.clear_min_m*1e3);
    % Name the trade when it happens.  The frontier's straight line between
    % two committed rows cannot re-solve; the driver can, and the clearance
    % hinge (200 nm per mm of deficit, kneed at the spec) is what makes a
    % wide ask buy floor with wavefront.  Say so rather than leaving a
    % worse-than-predicted WFE looking like a miss.
    if isfinite(mp.max_nm)
        dW = mp.max_nm - pred.wfe_nm;
        dC = gt.clear_min_m*1e3 - pred.floor_mm;
        % Only claim the hinge is LIVE when the floor is actually near it.
        % At a narrow box the floor runs ~90 mm and the hinge contributes
        % nothing -- the extra floor is not "bought", it is just more room
        % than a straight line between two committed rows implies.
        near_hinge = gt.clear_min_m*1e3 < max(P.clear_m)*1e3;
        if dC > 0.05 && dW > 0 && near_hinge
            L{end+1} = a(['                     -> the solve bought %.1f mm of ' ...
                          'floor with %.1f nm of WFE'], dC, dW);
            L{end+1} = a('                        (the %.0f mm clearance hinge is live at this width)', ...
                         min(P.clear_m)*1e3);
        elseif dC > 0.05 && dW > 0
            L{end+1} = a(['                     -> floor %.1f mm ABOVE the frontier line ' ...
                          '(hinge dormant here);'], dC);
            L{end+1} = a('                        wavefront %.1f nm above it', dW);
        elseif dC >= -0.05 && dW < 0
            L{end+1} = a('                     -> better than the frontier line on BOTH axes');
        end
    end
    L{end+1} = '';
    L{end+1} = a(' wall time         : %.1f min total (%.1f solve + %.1f score/figs)', ...
                 OUT.elapsed_min, OUT.t_solve_min, OUT.t_score_min);
    L{end+1} = a(' artifacts         : %s', shortname_(OUT.files.map));
    L{end+1} = a('                     %s', shortname_(OUT.files.layout));
    L{end+1} = a('                     %s', shortname_(OUT.files.fields));
    L{end+1} = a('                     %s', shortname_(OUT.files.deck));
    L{end+1} = '=================================================================';
end

function v = verdict_(mp, gt)
    if ~(~isfield(mp,'valid') || mp.valid)
        v = 'FAIL';                             % the map lost fields
    elseif gt.exit_pass && gt.clear_pass
        v = 'PASS';
    else
        v = 'PARTIAL';                          % solved, a gate still binds
    end
end

% =========================================================================
% helpers (the oi_walk copies -- same contract, same wording)
% =========================================================================
function q = start_qmean_(X, P)
%START_QMEAN_  Carried-design strict WFE over the SOLVE set at the offset,
%   WITHOUT solving (copied from OI_WALK).  1e9 = the no-rays sentinel.
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

function s = mapmax_str_(mp)
    if isfield(mp,'valid') && ~mp.valid
        s = sprintf('INVALID(%d/%d lost)', mp.n_failed, mp.n_fields);
    else
        s = sprintf('%.1f', mp.max_nm);
    end
end

function s = avg_str_(mp)
    if isfinite(mp.avg_nm), s = sprintf('%.1f', mp.avg_nm); else, s = 'n/a'; end
end

function s = valid_str_(mp)
    nf = 0;  nt = NaN;
    if isfield(mp,'n_failed'),  nf = mp.n_failed;  end
    if isfield(mp,'n_fields'),  nt = mp.n_fields;  end
    if ~isfield(mp,'valid') || mp.valid
        s = sprintf('VALID (%d/%d fields)', nt-nf, nt);
    else
        s = sprintf('INVALID (%d/%d fields lost)', nf, nt);
    end
end

function s = efmt_(e)
    if isnan(e), s = 'unmeasurable'; else, s = sprintf('%.3f', e); end
end

function s = pf_(p), if p, s = 'PASS'; else, s = 'FAIL'; end, end

function s = tern_(c, a, b), if c, s = a; else, s = b; end, end

function s = numtag_(w)
    s = strrep(sprintf('%g', w), '.', 'p');
end

function s = shortname_(f)
    [d, n, e] = fileparts(f);
    [~, dn] = fileparts(d);
    s = fullfile(dn, [n e]);
end

function fprintf_if_(c, varargin)
    if c, fprintf(varargin{:}); end
end

function out = wrap_(s, n)
%WRAP_  Wrap a sentence to n columns, indented to line up under a label.
    words = strsplit(strtrim(s));
    out = '';  line = '';
    for i = 1:numel(words)
        if isempty(line)
            line = words{i};
        elseif numel(line) + 1 + numel(words{i}) <= n
            line = [line ' ' words{i}]; %#ok<AGROW>
        else
            out = [out line newline '  ']; %#ok<AGROW>
            line = words{i};
        end
    end
    out = [out line];
end
