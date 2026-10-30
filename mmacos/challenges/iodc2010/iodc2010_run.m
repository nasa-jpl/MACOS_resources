function R = iodc2010_run(varargin)
%IODC2010_RUN  The IODC 2010 "green lens" challenge runner.
%   R = iodc2010_run()                       scores the bundled lenses
%   R = iodc2010_run('lens', L)              scores one iodc2010_lens
%   R = iodc2010_run('stage', 'gate')        the evaluator gate: engine vs the
%                                            independent tracer on the gate lens
%   R = iodc2010_run('nGridpts', 45, ...)    any iodc2010_params field
%
%   Stages ('stage' = 'evaluate' | 'gate' | 'search' | 'outer' | 'all'):
%     evaluate  iodc2010_score on each lens: deck, paraxial EFL check, aperture
%               sizing, field scan, bisection, report (<name>_report.txt), the
%               CODE V .seq beside it
%     gate      the metric argued: the engine's OPD on FEX's sphere against
%               iodc2010_reftrace on the SAME rays (max |dW|, RMS difference),
%               the chief through the stop centre, the EFL, the image height,
%               and the sampling convergence of the RMS
%     search    iodc2010_search: CALIB solves the spacings of the symmetric pair
%               over a radius x stop scan, each candidate scaled to EFL 100 and
%               scored by the metric; the best is reported
%     outer     iodc2010_outer: the outer design loop -- Petzval-balanced sign
%               patterns x seed spacings x stop places for p.outer.n_elts
%               elements, pre-screened by the evaluator, solved by CALIB in
%               rounds with a thickness floor, scored, the best re-solved over
%               a (D, theta) ladder, and every point mapped against the
%               published top ten (search/outer/)
%
%   A user edits iodc2010_params.m, or the lens list in bundled_lenses(),
%   and re-runs.  Nothing ends in exit(0): run it interactively or wrap it.
%
%   See also iodc2010_params, iodc2010_lens, iodc2010_score, iodc2010_eval.
ip = inputParser;  ip.KeepUnmatched = true;
ip.addParameter('stage', 'evaluate', @ischar);
ip.addParameter('lens', [], @(x) isempty(x) || isstruct(x));
ip.parse(varargin{:});
p = iodc2010_params();
fn = fieldnames(ip.Unmatched);
for i = 1:numel(fn)
    assert(isfield(p, fn{i}), 'iodc2010_run: unknown parameter %s', fn{i});
    p.(fn{i}) = ip.Unmatched.(fn{i});
end
here = fileparts(mfilename('fullpath'));
addpath(here);
if isempty(which('macos.init'))          % exist() is 0 for a package function; which() finds it
    addpath(fullfile(fileparts(fileparts(here)), 'src'));
end
if ~exist(p.outdir, 'dir'), mkdir(p.outdir); end

if isempty(ip.Results.lens), lenses = bundled_lenses(p); else, lenses = {ip.Results.lens}; end
R = struct('params', p, 'scores', {{}}, 'gate', []);

if any(strcmp(ip.Results.stage, {'evaluate', 'all'}))
    for i = 1:numel(lenses)
        L = lenses{i};
        fprintf('== %s\n', L.name);
        S = iodc2010_score(L, p);
        Ls = S.L;  Ls.semidia = S.semidia;       % the sized apertures go to CODE V too
        thl = S.theta_lim_deg;
        iodc2010_seq(Ls, S.P, p, unique(round([0 thl/2 thl 1.1*thl 1.5*thl], 3)), ...
            fullfile(p.outdir, [L.name '.seq']));
        R.scores{end+1} = S; %#ok<AGROW>
    end
    summary_(R, p);
end
if any(strcmp(ip.Results.stage, {'gate', 'all'}))
    R.gate = iodc2010_gate(p);
end
if any(strcmp(ip.Results.stage, {'search', 'all'}))
    % the first ladder: the symmetric pair's spacings solved by CALIB over a
    % radius and stop scan (iodc2010_search); the sign pattern and element
    % count are the next loop out
    seed = lenses{end};
    ps = p;  ps.outdir = fullfile(p.outdir, 'search');       % every candidate's deck + report
    if ~exist(ps.outdir, 'dir'), mkdir(ps.outdir); end
    R.search = iodc2010_search(seed, ps, 'R_list', [60 80 100 130], 'a_list', [3.5 5 7], 'design_field_deg', 3);
    writetable(R.search.table(:, {'R', 'a', 'efl_solved', 'n_iter', 'stop_reason', 'D', 'theta', 'merit', 'limit', 'ok'}), ...
        fullfile(ps.outdir, 'search_pair.csv'));
    if ~isempty(R.search.best)
        b = R.search.best;
        fprintf('search best: %s  D %.3f x %.2f deg = %.1f (%s)\n', b.lens.name, b.score.D_mm, ...
            b.score.theta_lim_deg, b.score.merit, b.score.limit_reason);
    end
end
if any(strcmp(ip.Results.stage, {'outer', 'all'}))
    po = p;  po.outdir = fullfile(p.outdir, 'search', 'outer');
    if ~exist(po.outdir, 'dir'), mkdir(po.outdir); end
    oa = namedargs2cell(p.outer);
    R.outer = iodc2010_outer(po, oa{:});
    fprintf('outer loop best merit %.1f; tables and the (D, theta) map in %s\n', R.outer.best_merit, po.outdir);
end
end

% -------------------------------------------------------------------------
function lenses = bundled_lenses(p)
% The bundled prescriptions.  Each is a full green lens (one |R|, signs,
% N-BK7 / air gaps, a stop); the EFL is set by the paraxial solve in the
% constructor helpers below so |EFL| = 100 to the digit.
n = iodc2010_glass(p.glass, p.lambda_nm);
% 1. the plano-convex singlet, stop on the lens: R = (n-1) f exactly, f/14
%    (at f/10 its on-axis spherical aberration alone reads 0.094 waves)
R1 = (n - 1) * p.efl_mm;
lenses{1} = iodc2010_lens('R', R1, 'sgn', [1 0], 't', [6 NaN], 'glass', {'N-BK7', 'air'}, ...
    'stop', [0 3.5], 'name', 'singlet_f14');
% 2. two plano-convex singlets, planos facing a central stop (the symmetric
%    form): thin-lens f1 = R/(n-1); f = f1^2/(2 f1 - d) -> d for f = 100,
%    then the exact thick-lens EFL is trimmed to 100 by a 1-D solve on d.
R2 = 80;
mk2 = @(d) iodc2010_lens('R', R2, 'sgn', [1 0 0 -1], 't', [6 d 6 NaN], ...
    'glass', {'N-BK7', 'air', 'N-BK7', 'air'}, 'stop', [6 + d/2, 3.5], 'name', 'symmetric_pair');
f1 = R2 / (n - 1);  d0 = 2 * f1 - f1^2 / p.efl_mm;
d  = fzero(@(dd) abs(iodc2010_paraxial(mk2(dd), p.lambda_nm).efl) - p.efl_mm, d0);
lenses{2} = mk2(d);
end

function summary_(R, p)
fprintf('\n%-18s %8s %8s %10s %8s  %s\n', 'lens', 'EFL', 'D mm', 'theta deg', 'merit', 'limit');
for i = 1:numel(R.scores)
    S = R.scores{i};
    fprintf('%-18s %8.4f %8.3f %10.2f %8.1f  %s\n', S.L.name, S.efl_mm, S.D_mm, ...
        S.theta_lim_deg, S.merit, S.limit_reason);
end
fprintf('published winner 2010: 81.30 x 43.50 = 3537 (Bates); 2nd 3502; 10th 1094.\n');
fprintf('reports in %s\n', p.outdir);
end
