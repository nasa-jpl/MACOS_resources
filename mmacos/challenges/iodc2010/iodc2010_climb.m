function [Lb, info] = iodc2010_climb(L, p, opts)
%IODC2010_CLIMB  The merit-shaped inner objective: the largest semi-field at RMS 0.070 for a given D.
%   [Lb, info] = iodc2010_climb(L, p, 'D_target', D) takes a green lens and
%   returns the lens of the largest MERIT (D x semi-field, iodc2010_score's
%   metric) it found by re-solving the spacings (iodc2010_solve) for a
%   design field it raises while the lens holds it and lowers when it does
%   not -- a bracket on the design field with CALIB inside.
%
%   Why.  CALIB minimizes a SUM of squared wavefront errors at fixed fields;
%   the merit is a THRESHOLD, the largest field at which every field up to it
%   has RMS <= p.rms_limit (and |distortion| <= p.dist_limit, no ray lost).
%   The sum buys the design fields with the field's edge (the 2026-10-30 run:
%   one seed went 81 -> 53), and a design field asked for beyond reach drives
%   the lens off the merit altogether (the ladder's zeros).  Two things make
%   the solve merit-shaped here:
%     * the design fields track the achievable field: the step k solve asks
%       for 0 .. theta_k at opts.n_fields equal spacings; the solved lens is
%       SCORED (a quick metric on opts.score_grid rays across); if its own
%       semi-field reaches theta_k (to opts.tol) the field is raised
%       (x opts.grow, or to the geometric mean of the bracket), else lowered
%       (to the geometric mean, or by opts.grow when nothing has passed yet);
%       the bracket closes at opts.tol or after opts.steps solves;
%     * optionally (opts.reweight) the field weights are re-weighted toward
%       the worst field: after every solve each field's weight is
%       multiplied by (its RMS / p.rms_limit)^2 (floored at opts.w_floor,
%       normalized to a mean of 1, clipped to 1e-3 .. 1e3) -- iteratively
%       re-weighted least squares toward the minimax the threshold wants.
%       MEASURED on the three production seeds (2026-10-31): the bracket
%       alone gave 81.3 / 80.3 / 59.4 against the plain sum's 52.7 / 74.8 /
%       53.6; re-weighting gave 81.3 / 76.5 / 59.5 -- the sum over fields
%       spanning 0..theta is already dominated by its largest residual, and
%       the extra weight sent one solve wandering (every field over a wave).
%       Off by default; the switch stays for the A/B.
%   A step starts from the base camp: the last lens that HELD its design
%   field (CALIB is local; the lens that held theta_k is the right start
%   for theta_{k+1}), never from a step that did not -- a solve asked for a
%   field beyond reach re-arranges the lens badly (the first A/B: a seed of
%   merit 81 asked for 1.3 x its field came back with 0.15 waves ON AXIS,
%   and every later step inherited the wreck).  The lens returned is the
%   best-merit lens over every step AND the start, so the climb never
%   returns worse than it was given (the start is scored first,
%   info.merit_start).  Distortion and vignetting end the field in the score
%   as they do in the metric; the solve does not target them, so a field
%   they end lowers the bracket like any other failure, info.rows.limit
%   says which it was, and a seed whose field THEY end gains nothing from
%   a wavefront objective (the 81 seed is one: distortion at 13.5 deg).
%
%   D is held through iodc2010_solve's D_target (the stop re-sized every
%   round, the final scale to |EFL| = 100 landing on D); the start's stop is
%   re-sized to D here.
%
%   info: .rows (table per step: step, theta_design, weights, solve_stop,
%   rounds, wfe_fields (CALIB's own, waves), D, theta, merit, limit, pass),
%   .theta_start / .merit_start (the start as given, scored), .best_step
%   (0 = the start), .merit (the best, on the climb's own grid -- the
%   caller re-scores on the metric's), .stop ('bracketed' | 'steps' |
%   'no-solve'), .lo / .hi (the bracket), .deck (the best step's CALIB deck).
arguments
    L struct
    p struct
    opts.D_target (1,1) double = NaN      % mm at EFL 100; NaN = the lens's own
    opts.theta0 (1,1) double = NaN        % the first design field; NaN = opts.grow x the start's own limit
    opts.theta_min (1,1) double = 1       % the first design field's floor (deg)
    opts.n_fields (1,1) double = 4        % design fields 0 .. theta at equal spacings
    opts.steps (1,1) double = 8
    opts.grow (1,1) double = 1.3
    opts.tol (1,1) double = 0.03          % the bracket's relative width at close; a step passes at (1 - tol) x theta
    opts.reweight (1,1) logical = false
    opts.w_floor (1,1) double = 0.1
    opts.solve (1,1) struct = struct()    % iodc2010_solve overrides (rounds, iters_per_round, floors, ray_grid, var_stop)
    opts.score_grid (1,1) double = 21     % rays across in the climb's own scoring
    opts.score_tol_deg (1,1) double = 0.05
    opts.deck_base (1,:) char = ''        % decks and reports go to <deck_base>_c<k>[_calib].in
    opts.quiet (1,1) logical = true
end
say = @(varargin) iif_(~opts.quiet, varargin{:});
if isempty(opts.deck_base), opts.deck_base = fullfile(p.outdir, L.name); end
[dk_dir, dk_name] = fileparts(opts.deck_base);
if ~exist(dk_dir, 'dir'), mkdir(dk_dir); end
pq = p;  pq.nGridpts = opts.score_grid;  pq.size_apertures = false;  pq.theta_tol_deg = opts.score_tol_deg;
pq.outdir = dk_dir;
solve_args = namedargs2cell(opts.solve);

% ---- the start: the stop at D, scored as given
L0 = L;  L0.semidia = [];
P0 = iodc2010_paraxial(L0, p.lambda_nm);
if isfinite(opts.D_target) && isfinite(P0.D) && P0.D > 0, L0.stop(2) = L0.stop(2) * opts.D_target / P0.D; end
L0.name = [dk_name '_c0'];
S0 = iodc2010_score(L0, pq, 'quiet', true, 'deck', fullfile(dk_dir, [L0.name '.in']));
info.theta_start = S0.theta_lim_deg;  info.merit_start = S0.merit;  info.limit_start = S0.limit_reason;
best = struct('L', L0, 'S', S0, 'step', 0, 'deck', '');
say('  climb %s: start D %.3f x %.2f deg = %.1f (%s)\n', dk_name, S0.D_mm, S0.theta_lim_deg, S0.merit, S0.limit_reason);

th = opts.theta0;
if ~isfinite(th), th = max(S0.theta_lim_deg * opts.grow, opts.theta_min); end
lo = 0;  hi = inf;                           % design fields: the largest held, the smallest not held
w = ones(1, opts.n_fields);
frac = linspace(0, 1, opts.n_fields);
Lc = L0;  rows = {};  info.stop = 'steps';  n_solved = 0;
for k = 1:opts.steps
    fields = th * frac;
    deck_c = fullfile(dk_dir, sprintf('%s_c%d_calib.in', dk_name, k));
    Lc.name = sprintf('%s_c%d', dk_name, k);
    [Ls, si] = iodc2010_solve(Lc, p, 'fields_deg', fields, 'fov_weights', w, 'D_target', opts.D_target, ...
        'deck', deck_c, 'quiet', true, solve_args{:});
    row = struct('step', k, 'theta_design', th, 'weights', {{mat2str(w, 3)}}, 'solve_stop', {{si.stop}}, 'rounds', si.n_rounds, ...
        'wfe_fields', {{mat2str(si.wfe_fields, 3)}}, 'D', NaN, 'theta', NaN, 'merit', 0, 'limit', {{'not solved'}}, 'pass', false);
    if ~si.ok
        say('    step %d: design field %.2f deg -> solve stopped (%s)\n', k, th, si.stop);
        hi = min(hi, th);
        th = tern_(lo > 0, sqrt(lo * hi), th / opts.grow);
    else
        n_solved = n_solved + 1;
        Ls.name = sprintf('%s_c%d', dk_name, k);
        S = iodc2010_score(Ls, pq, 'quiet', true, 'deck', fullfile(dk_dir, [Ls.name '.in']));
        row.D = S.D_mm;  row.theta = S.theta_lim_deg;  row.merit = S.merit;  row.limit = {S.limit_reason};
        row.pass = S.theta_lim_deg >= th * (1 - opts.tol);
        say('    step %d: design field %.2f deg, weights %s -> solve %s, D %.3f x %.2f deg = %.1f (%s)%s\n', k, th, ...
            mat2str(w, 2), si.stop, S.D_mm, S.theta_lim_deg, S.merit, S.limit_reason, tern_(row.pass, ' HELD', ''));
        if S.merit > best.S.merit, best = struct('L', Ls, 'S', S, 'step', k, 'deck', deck_c); end
        if row.pass, Lc = Ls; else, Lc = best.L; end     % the base camp moves only to a lens that held its field
        Lc.semidia = [];
        if opts.reweight && all(isfinite(si.wfe_fields)) && numel(si.wfe_fields) == numel(w)
            w = w .* max(si.wfe_fields / p.rms_limit, opts.w_floor).^2;
            w = w / mean(w);  w = min(max(w, 1e-3), 1e3);
        end
        if row.pass
            lo = max(lo, th);
            th = tern_(isfinite(hi), sqrt(lo * hi), th * opts.grow);
        else
            hi = min(hi, th);
            th = tern_(lo > 0, sqrt(lo * hi), max(S.theta_lim_deg, th / opts.grow));
        end
    end
    rows{end+1} = row; %#ok<AGROW>
    if lo > 0 && isfinite(hi) && hi / lo - 1 < opts.tol, info.stop = 'bracketed';  break; end
end
if n_solved == 0, info.stop = 'no-solve'; end
Lb = best.L;  Lb.name = L.name;
info.rows = struct2table([rows{:}], 'AsArray', true);
info.best_step = best.step;  info.merit = best.S.merit;  info.theta = best.S.theta_lim_deg;  info.D = best.S.D_mm;
info.limit = best.S.limit_reason;  info.deck = best.deck;  info.lo = lo;  info.hi = hi;  info.n_steps = numel(rows);
say('  climb %s: best at step %d, D %.3f x %.2f deg = %.1f (%s); bracket [%.2f, %.2f] deg, %s\n', dk_name, ...
    best.step, info.D, info.theta, info.merit, info.limit, lo, hi, info.stop);
end

function out = tern_(c, a, b)
if c, out = a; else, out = b; end
end

function iif_(c, varargin)
if c, fprintf(varargin{:}); end
end
