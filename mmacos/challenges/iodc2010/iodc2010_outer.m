function B = iodc2010_outer(p, opts)
%IODC2010_OUTER  The outer design loop: forms -> pre-screen -> CALIB solve -> score -> ladder -> map.
%   B = iodc2010_outer(p) runs brief step 4 around the inner solve:
%
%     1. FORMS      opts.families: 'forms' = iodc2010_forms for each element
%                   count in opts.n_elts (the Petzval-balanced sign patterns x
%                   seed air spaces x stop places); 'groups' = iodc2010_groups
%                   (converging / diverging / converging groups of equal
%                   elements, power from the separations, the intermediate-
%                   image form among them).  Each seed is scaled to
%                   |EFL| = 100 with the stop sized for opts.D_target.
%     2. PRE-SCREEN every seed's stop is first shrunk from opts.D_target
%                   until the seed passes ON AXIS (its spherical aberration
%                   and its pupil's own clipping at the stop set that D), then
%                   the seed is SCORED as it stands (no solve) by the metric
%                   itself on a coarse ray grid (opts.prescreen_grid across,
%                   the semi-field to 0.1 deg, apertures open): its own
%                   D x theta.  opts.top_k per element count go on, ranked
%                   by that merit; a seed with merit 0 is dropped.
%     3. SOLVE      iodc2010_solve (CALIB on the spacings in rounds, the
%                   thickness floor, re-emitted per round, the seed's own D
%                   held) at the design fields [0 theta_d/2 theta_d] with
%                   theta_d = the seed's own semi-field (opts.design_field_deg
%                   the floor), then iodc2010_score: the metric of the README,
%                   apertures sized at the limit.  opts.top_k per element
%                   count (of whatever family) go through it.
%     4. LADDER     the opts.n_ladder best solved lenses re-solved at every
%                   (D, theta) in opts.ladder_D x opts.ladder_theta (the stop
%                   re-sized for D, the design fields for theta) and scored:
%                   the lens's own frontier on the (D, theta) plane.
%     5. MAP        every scored point against the published top ten on the
%                   (D, theta) plane with lines of constant merit (the paper's
%                   Fig. 20) -> <outdir>/dtheta_map.png.
%
%   Stages (opts.stage): 'all' (default), 'prescreen' (forms + pre-screen,
%   the CSV written, return), 'map' (redraw the map from the tables on
%   disk), 'solve' (resume from opts.prescreen_csv: the
%   seeds are rebuilt deterministically, each stop re-sized to the CSV's D,
%   then solve + ladder + map).  The split lets the 14-minute pre-screen be
%   kept while the solve stage is re-run (the 2026-10-30 production run
%   needed exactly that: its first solve stage had run on the pre-fix shared
%   mex through the runner's path guard, see iodc2010_run).
%
%   Artifacts in p.outdir: outer_prescreen.csv, outer_solved.csv,
%   outer_ladder.csv, every solved lens's deck / report / .seq, the map.
%   B: .seeds (per element count), .prescreen, .solved, .ladder (tables),
%   .best (lens + score), .timing.
arguments
    p struct
    opts.families (1,:) cell = {'forms', 'groups'}
    opts.n_elts (1,:) double = [2 3]
    opts.groups (1,1) struct = struct()    % name/value overrides for iodc2010_groups
    opts.D_target (1,1) double = 10
    opts.design_field_deg (1,1) double = 1      % the design field's floor (1.5 x the seed's own limit above it)
    opts.top_k (1,1) double = 10
    opts.prescreen_grid (1,1) double = 21
    opts.air_grid (1,:) double = [0.1 0.25 0.5 1 2]
    opts.glass_frac (1,1) double = 0.05
    opts.ladder_D (1,:) double = [10 15 20 30]
    opts.ladder_theta (1,:) double = [3 5 8 12]
    opts.n_ladder (1,1) double = 3
    opts.solve (1,1) struct = struct()     % name/value overrides for iodc2010_solve
    opts.stage (1,:) char = 'all'
    opts.prescreen_csv (1,:) char = ''
    opts.quiet (1,1) logical = false
end
B.ladder = table();  B.solved = table();  B.best = [];  B.best_merit = 0;
say = @(varargin) iif_(~opts.quiet, varargin{:});
if ~exist(p.outdir, 'dir'), mkdir(p.outdir); end
t_all = tic;
macos.init(p.model_size);

% ---- 1. forms
B.seeds = {};
if any(strcmp(opts.families, 'forms'))
    for ne = opts.n_elts
        [C, fi] = iodc2010_forms(ne, p, 'air_grid', opts.air_grid, 'glass_frac', opts.glass_frac, 'D_target', opts.D_target);
        say('forms N=%d: %d patterns, %d Petzval-balanced, %d tried, %d paraxially viable seeds\n', ...
            ne, fi.n_patterns_all, fi.n_patterns_balanced, fi.n_tried, fi.n_seeds);
        B.seeds{end+1} = struct('family', 'forms', 'nElt', ne, 'C', C, 'info', fi);
    end
end
if any(strcmp(opts.families, 'groups'))
    ga = namedargs2cell(opts.groups);
    [C, gi] = iodc2010_groups(p, 'D_target', opts.D_target, ga{:});
    say('groups: %d tried, %d paraxially viable seeds (elements %s)\n', gi.n_tried, gi.n_seeds, ...
        mat2str(unique(arrayfun(@(c) c.L.N / 2, C))));
    B.seeds{end+1} = struct('family', 'groups', 'nElt', NaN, 'C', C, 'info', gi);
end

% ---- 2. pre-screen: the seed's own merit on a coarse grid
pq = p;  pq.nGridpts = opts.prescreen_grid;  pq.size_apertures = false;  pq.theta_tol_deg = 0.1;
pq.outdir = fullfile(p.outdir, 'prescreen_tmp');
if ~exist(pq.outdir, 'dir'), mkdir(pq.outdir); end
rows = {};  t0 = tic;  n_eval = 0;
if strcmp(opts.stage, 'solve')
    % resume: the CSV's rows, the seeds' stops re-sized to the CSV's D
    if isempty(opts.prescreen_csv), opts.prescreen_csv = fullfile(p.outdir, 'outer_prescreen.csv'); end
    T = readtable(opts.prescreen_csv, 'TextType', 'char');
    for s = 1:numel(B.seeds)
        C = B.seeds{s}.C;
        for i = 1:numel(C)
            j = find(strcmp(T.name, C(i).L.name), 1);
            if isempty(j), continue; end
            Pi = iodc2010_paraxial(C(i).L, p.lambda_nm);
            C(i).L.stop(2) = C(i).L.stop(2) * T.D(j) / Pi.D;  C(i).D = T.D(j);
        end
        B.seeds{s}.C = C;
    end
    B.prescreen = T;
    say('resumed from %s: %d seeds\n', opts.prescreen_csv, height(T));
end
for s = 1:numel(B.seeds) * ~strcmp(opts.stage, 'solve')
    C = B.seeds{s}.C;
    for i = 1:numel(C)
        L = C(i).L;
        try
            L = fit_axis_(L, pq, fullfile(pq.outdir, 'tmp.in'));
            C(i).L = L;  C(i).D = iodc2010_paraxial(L, p.lambda_nm).D;
            Sq = iodc2010_score(L, pq, 'quiet', true, 'deck', fullfile(pq.outdir, 'tmp.in'));
            merit = Sq.merit;  th = Sq.theta_lim_deg;  lim = Sq.limit_reason;  rms0 = Sq.fields.rms_waves(1);
        catch err
            merit = 0;  th = 0;  lim = ['error: ' err.message];  rms0 = NaN;
        end
        n_eval = n_eval + 1;
        rows{end+1} = struct('name', L.name, 'family', B.seeds{s}.family, 'nElt', L.N / 2, 'pattern', mat2str(C(i).pattern), ...
            'gaps_R', mat2str(C(i).gaps_R), 'stop_place', C(i).stop_place, 'R', C(i).R, 'oal', C(i).oal, ...
            'efl', C(i).efl, 'D', C(i).D, 'intermediate', C(i).intermediate, 'rms0', rms0, ...
            'theta_seed', th, 'limit_seed', lim, 'merit_seed', merit); %#ok<AGROW>
    end
    B.seeds{s}.C = C;                    % the seeds with their fitted stops
end
if ~strcmp(opts.stage, 'solve')
    B.prescreen = struct2table([rows{:}], 'AsArray', true);
    B.timing.prescreen_s = toc(t0);  B.timing.prescreen_n = n_eval;
    say('pre-screen: %d seeds scored in %.0f s (%.2f s each)\n', n_eval, B.timing.prescreen_s, B.timing.prescreen_s / max(n_eval, 1));
    writetable(B.prescreen, fullfile(p.outdir, 'outer_prescreen.csv'));
end
if exist(pq.outdir, 'dir'), rmdir(pq.outdir, 's'); end
if strcmp(opts.stage, 'prescreen'), B.timing.total_s = toc(t_all);  return; end
if strcmp(opts.stage, 'map')                     % redraw from the tables on disk
    B.solved = readtable(fullfile(p.outdir, 'outer_solved.csv'), 'TextType', 'char');
    B.ladder = readtable(fullfile(p.outdir, 'outer_ladder.csv'), 'TextType', 'char');
    B.best = struct();  B.map = fullfile(p.outdir, 'dtheta_map.png');
    map_(B, p, opts);  B.timing.total_s = toc(t_all);  return
end

% the top_k per element count (any family) by the seed's own merit
keep = [];
for ne = unique(B.prescreen.nElt).'
    T = B.prescreen(B.prescreen.nElt == ne & B.prescreen.merit_seed > 0, :);
    T = sortrows(T, 'merit_seed', 'descend');
    keep = [keep; T(1:min(opts.top_k, height(T)), :)]; %#ok<AGROW>
end
say('pre-screen keeps %d seeds for the solve\n', height(keep));

% ---- 3. solve + score
seedmap = containers.Map();
for s = 1:numel(B.seeds), for i = 1:numel(B.seeds{s}.C), seedmap(B.seeds{s}.C(i).L.name) = B.seeds{s}.C(i).L; end, end
solve_args = namedargs2cell(opts.solve);
rows = {};  t0 = tic;
for i = 1:height(keep)
    L = seedmap(char(keep.name(i)));
    thd = max(keep.theta_seed(i), opts.design_field_deg);
    say('solve %2d/%d %s (R %.1f, OAL %.1f, seed D %.2f x %.2f deg = %.1f; design field %.2f deg)\n', i, height(keep), L.name, ...
        keep.R(i), keep.oal(i), keep.D(i), keep.theta_seed(i), keep.merit_seed(i), thd);
    [Ls, si] = iodc2010_solve(L, p, 'fields_deg', [0 thd/2 thd], 'deck', fullfile(p.outdir, [L.name '_calib.in']), ...
        'D_target', keep.D(i), 'quiet', opts.quiet, solve_args{:});
    row = struct('name', L.name, 'nElt', L.N / 2, 'pattern', char(keep.pattern(i)), 'stop_place', char(keep.stop_place(i)), ...
        'merit_seed', keep.merit_seed(i), 'theta_design', thd, ...
        'solve_stop', si.stop, 'rounds', si.n_rounds, 'clamped', nnz(si.clamped), 'R', Ls.R, 'oal', Ls.z(end), ...
        'efl', NaN, 'D', NaN, 'theta', NaN, 'merit', 0, 'limit', 'not solved', 'rules_ok', false, 'ok', false);
    if si.ok
        S = iodc2010_score(Ls, p, 'quiet', true, 'deck', fullfile(p.outdir, [L.name '.in']));
        row.efl = S.efl_mm;  row.D = S.D_mm;  row.theta = S.theta_lim_deg;  row.merit = S.merit;
        row.limit = S.limit_reason;  row.rules_ok = S.rules.ok;
        row.ok = S.efl_ok && S.rules.ok && S.image_real;
        Lq = S.L;  Lq.semidia = S.semidia;  thl = S.theta_lim_deg;
        iodc2010_seq(Lq, S.P, p, unique(round([0 thl/2 thl 1.1*thl 1.5*thl], 3)), fullfile(p.outdir, [L.name '.seq']));
        say('   -> D %.3f x %.2f deg = %.1f (%s)%s\n', row.D, row.theta, row.merit, row.limit, tern_(row.ok, '', ' REJECTED'));
    else
        say('   -> solve stopped: %s\n', si.stop);
    end
    rows{end+1} = row; %#ok<AGROW>
end
if isempty(rows)
    B.solved = table('Size', [0 2], 'VariableTypes', {'double', 'logical'}, 'VariableNames', {'merit', 'ok'});
else
    B.solved = struct2table([rows{:}], 'AsArray', true);
    writetable(B.solved, fullfile(p.outdir, 'outer_solved.csv'));
end
B.timing.solve_s = toc(t0);

% ---- 4. the ladder on the best solved lenses
ok = B.solved(B.solved.ok, :);
ok = sortrows(ok, 'merit', 'descend');
rows = {};  t0 = tic;  best = [];  bestMerit = -inf;
for j = 1:min(opts.n_ladder, height(ok))
    name0 = char(ok.name(j));
    Lb = read_lens_(fullfile(p.outdir, [name0 '_report.txt']), seedmap(name0));
    P0 = iodc2010_paraxial(Lb, p.lambda_nm);
    for Dl = opts.ladder_D
        for thl = opts.ladder_theta
            L = Lb;  L.stop(2) = Lb.stop(2) * Dl / P0.D;  L.semidia = [];
            L.name = sprintf('%s_D%g_t%g', name0, Dl, thl);
            say('ladder %s: D %g, design field %g deg\n', L.name, Dl, thl);
            [Ls, si] = iodc2010_solve(L, p, 'fields_deg', [0 thl/2 thl], 'deck', fullfile(p.outdir, [L.name '_calib.in']), ...
                'D_target', Dl, 'quiet', true, solve_args{:});
            row = struct('base', name0, 'name', L.name, 'D_target', Dl, 'theta_design', thl, 'solve_stop', si.stop, ...
                'rounds', si.n_rounds, 'R', Ls.R, 'oal', Ls.z(end), 'D', NaN, 'theta', NaN, 'merit', 0, ...
                'limit', 'not solved', 'ok', false);
            if si.ok
                S = iodc2010_score(Ls, p, 'quiet', true, 'deck', fullfile(p.outdir, [L.name '.in']));
                row.D = S.D_mm;  row.theta = S.theta_lim_deg;  row.merit = S.merit;  row.limit = S.limit_reason;
                row.ok = S.efl_ok && S.rules.ok && S.image_real;
                say('   -> D %.3f x %.2f deg = %.1f (%s)%s\n', row.D, row.theta, row.merit, row.limit, tern_(row.ok, '', ' REJECTED'));
                if row.ok && S.merit > bestMerit
                    bestMerit = S.merit;  best = struct('lens', Ls, 'score', S);
                    Lq = S.L;  Lq.semidia = S.semidia;  tl = S.theta_lim_deg;
                    iodc2010_seq(Lq, S.P, p, unique(round([0 tl/2 tl 1.1*tl 1.5*tl], 3)), fullfile(p.outdir, [L.name '.seq']));
                end
            end
            rows{end+1} = row; %#ok<AGROW>
        end
    end
end
if isempty(rows), B.ladder = table(); else, B.ladder = struct2table([rows{:}], 'AsArray', true); end
B.timing.ladder_s = toc(t0);
if ~isempty(rows), writetable(B.ladder, fullfile(p.outdir, 'outer_ladder.csv')); end
% the best over the solve stage too
for j = 1:height(ok)
    if ok.merit(j) > bestMerit
        bestMerit = ok.merit(j);
        best = struct('lens', read_lens_(fullfile(p.outdir, [char(ok.name(j)) '_report.txt']), seedmap(char(ok.name(j)))), 'score', []);
        best.name = char(ok.name(j));
    end
end
B.best = best;  B.best_merit = bestMerit;

% ---- 5. the (D, theta) map
B.map = fullfile(p.outdir, 'dtheta_map.png');
map_(B, p, opts);
B.timing.total_s = toc(t_all);
say('outer loop done in %.0f s; best merit %.1f; map %s\n', B.timing.total_s, bestMerit, B.map);
end

% -------------------------------------------------------------------------
function L = fit_axis_(L, pq, deck)
% Shrink the stop until the seed passes on axis: a lost ray is the pupil's
% own aberration carrying the paraxial pupil's edge outside the stop (an
% area, so the square root of the kept fraction); an RMS over the limit is
% taken as spherical aberration (W ~ D^4, a conservative 0.9 on top).
for k = 1:8
    P = iodc2010_paraxial(L, pq.lambda_nm);
    Lw = L;  Lw.semidia = inf(1, L.N);
    E = iodc2010_emit(Lw, P, pq, deck);
    macos.load_rx(E.path);
    F0 = iodc2010_eval(E, P, pq, 0);
    macos.modify();
    if F0.nLost > 0
        L.stop(2) = L.stop(2) * 0.995 * sqrt(1 - F0.nLost / F0.nLaunched);
    elseif F0.rms_waves > pq.rms_limit
        L.stop(2) = L.stop(2) * 0.9 * (pq.rms_limit / F0.rms_waves)^0.25;
    else
        return
    end
end
end

function L = read_lens_(report, L0)
% The solved prescription from its own report (the record the user sees),
% so the ladder starts from exactly what was scored.
txt = fileread(report);
tok = regexp(txt, 'PRESCRIPTION: R = ([\d.eE+-]+) mm, signs \[([^\]]*)\], t = \[([^\]]*)\] mm, glass \{([^}]*)\}, stop z = ([\d.eE+-]+) mm a = ([\d.eE+-]+) mm', 'tokens', 'once');
assert(~isempty(tok), 'iodc2010_outer: cannot read the prescription from %s', report);
L = L0;
L.R = str2double(tok{1});
L.sgn = sscanf(tok{2}, '%d').';
L.t = sscanf(tok{3}, '%f').';
L.glass = strsplit(tok{4}, ',');
L.stop = [str2double(tok{5}), str2double(tok{6})];
L.z = [0, cumsum(L.t(1:end-1))];
L.semidia = [];
end

function map_(B, p, opts)
top = [81.30 43.50; 94.00 37.25; 110.65 19.60; 100.36 19.60; 55.30 30.00; 56.00 29.00; ...
       81.97 15.85; 87.50 14.00; 63.20 17.64; 44.30 24.70];
f = figure('Visible', 'off', 'Position', [100 100 760 560], 'Color', 'w');
ax = axes(f);  hold(ax, 'on');
Dg = linspace(1, 120, 400);
for m = [30 100 300 1000 3500]
    th = m ./ Dg;  th(th > 50 | th < 0.5) = NaN;
    plot(ax, Dg, th, ':', 'Color', [0.6 0.6 0.6]);
    Dl = min(110, m / 0.6);                        % label near the lower-right end of each line
    text(ax, Dl, m / Dl * 1.15, sprintf('%d', m), 'Color', [0.5 0.5 0.5], 'FontSize', 8, 'HorizontalAlignment', 'right');
end
plot(ax, top(:, 1), top(:, 2), 's', 'MarkerSize', 8, 'MarkerFaceColor', [0.85 0.85 0.85], 'Color', 'k');
pts = [];
for T = {B.solved, B.ladder}
    T = T{1};
    if isempty(T) || ~ismember('ok', T.Properties.VariableNames), continue; end
    ok = T.ok;  if ~islogical(ok), ok = logical(str2double(string(ok))); end   % readtable may give it as text
    pts = [pts; T.D(ok), T.theta(ok)]; %#ok<AGROW>
end
if ~isempty(pts), plot(ax, pts(:, 1), pts(:, 2), 'o', 'MarkerSize', 5, 'Color', [0 0.35 0.75], 'MarkerFaceColor', [0.3 0.6 0.95]); end
if ~isempty(B.best) && ~isempty(pts)
    [~, ib] = max(pts(:, 1) .* pts(:, 2));
    plot(ax, pts(ib, 1), pts(ib, 2), 'p', 'MarkerSize', 14, 'Color', 'k', 'MarkerFaceColor', [1 0.8 0.1]);
    text(ax, pts(ib, 1) + 2, pts(ib, 2) + 1, sprintf('%.0f', pts(ib, 1) * pts(ib, 2)), 'FontSize', 9);
end
set(ax, 'XScale', 'log', 'YScale', 'log');
xlim(ax, [1 130]);  ylim(ax, [0.5 50]);
xlabel(ax, 'entrance pupil diameter D (mm)');  ylabel(ax, 'semi-field at RMS 0.070 wave (deg)');
title(ax, sprintf('IODC 2010: our solved lenses (circles, N = %s) and the published top ten (squares); dotted lines = constant D x theta', ...
    mat2str(opts.n_elts)), 'FontSize', 9);
grid(ax, 'on');
print(f, B.map, '-dpng', '-r110');
close(f);
end

function out = tern_(c, a, b)
if c, out = a; else, out = b; end
end

function iif_(c, varargin)
if c, fprintf(varargin{:}); end
end
