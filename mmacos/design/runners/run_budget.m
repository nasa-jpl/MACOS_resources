function art = run_budget(sens, varargin)
%RUN_BUDGET  Optical error budget from dw_d* sensitivity Jacobians.
%   ART = run_budget(SENS) builds a covariance + Monte-Carlo error budget
%   from a sensitivities artifact SENS (a <name>_sens.mat path, a struct
%   with ox/oz/og/os, or a single dw_d*_multi output struct), writes a
%   report, a self-contained .xlsx (WFE + Coefficients sheets), a CSV, a
%   coef.* .mat, and a per-group bar figure, and returns ART.
%
%   This is the budget stage of the design pipeline (design ->
%   segmentation -> sensitivities -> MET -> compare -> simulate): a
%   consumer of the sensitivities (and later MET) artifacts.  The math is
%   Redding 2014 (SPIE): a properly-formed RSS budget is a covariance
%   analysis; Monte-Carlo is the identical answer when assumptions match.
%
%   Name/value options:
%     'spec'      group table / struct / path (Excel|CSV|.m).  Omitted =>
%                 macos.budget.suggest_groups proposes a skeleton (EDIT its
%                 placeholder sigmas).
%     'roles'     containers.Map role -> iElt vector (e.g. 'pm_segs' -> ...)
%     'control'   false (default, uncontrolled) | true (project out the
%                 groups tagged Control) | logical Nz control-DOF mask
%     'wfs_sigma' WFS noise per wavefront sample, {sigma, units} or nm
%                 (Eq. 32 sensing-error row; controlled runs only)
%     'act_sigma' actuator noise per control group: table Name/Sigma/Units
%                 or cell {name, sigma, units; ...} (Eq. 31 rows)
%     'ndraw'     Monte-Carlo draws (default 400; 0 skips MC)
%     'mc_tol'    cov-vs-MC relative tolerance to warn at (default 0.10)
%     'metric'    'opd' (default) | 'spot'
%     'name'      artifact base name (default from SENS)
%     'out_dir'   output directory (default beside SENS, else pwd)
%     'visible'/'figures'  figure controls (default false / true)
%
%   ART fields: .M .spec .cov .mc .table .coef .total_nm .mat .report .xlsx
%   .csv .figs .control.
%
%   See also macos.budget.assemble, macos.budget.cov_budget,
%   macos.budget.mc_budget, run_sensitivities, run_met.

p = inputParser;
p.addParameter('spec', [], @(x) true);
p.addParameter('roles', containers.Map('KeyType','char','ValueType','any'));
p.addParameter('control', false, @(x) islogical(x) || isnumeric(x));
p.addParameter('wfs_sigma', [], @(x) isempty(x) || isnumeric(x) || iscell(x));
p.addParameter('act_sigma', [], @(x) isempty(x) || istable(x) || iscell(x));
p.addParameter('ndraw', 400, @(x) isscalar(x) && x >= 0);
p.addParameter('mc_tol', 0.10, @isscalar);
p.addParameter('metric', 'opd', @(s) any(strcmpi(s, {'opd','spot'})));
p.addParameter('name', '', @(x) ischar(x) || isstring(x));
p.addParameter('out_dir', '', @(x) ischar(x) || isstring(x));
p.addParameter('trans_per_metre', true, @islogical);
p.addParameter('visible', false, @islogical);
p.addParameter('figures', true, @islogical);
p.parse(varargin{:});
o = p.Results;

[name, od] = names_(sens, o);
say = @(varargin) fprintf(varargin{:});
pth = @(suf) fullfile(od, [name suf]);

%% -- [1] assemble ----------------------------------------------------
M = macos.budget.assemble(sens, 'metric', o.metric, ...
    'trans_per_metre', o.trans_per_metre);
say('\n[1] assembled %s: J %dx%d, nw=%d, channels={%s}, metric=%s\n', ...
    name, size(M.J,1), size(M.J,2), M.nw, strjoin(unique(M.meta.channel),','), ...
    M.units.metric);

%% -- [2] group spec --------------------------------------------------
if isempty(o.spec)
    T = macos.budget.suggest_groups(M, 'roles', o.roles);
    say('[2] no spec given -- suggest_groups proposed %d groups ', height(T));
    say('(PLACEHOLDER sigmas = 1; edit them)\n');
else
    T = macos.budget.read_spec(o.spec);
    say('[2] spec: %d group rows\n', height(T));
end
spec = macos.budget.spec_from_table(T, M, 'roles', o.roles, ...
    'wfs_sigma', o.wfs_sigma, 'act_sigma', o.act_sigma);

%% -- [3] control mask ------------------------------------------------
ctrl = control_mask_(o.control, spec, M);
if ~isempty(ctrl) && any(ctrl)
    say('[3] control: %d actuator DOFs projected out (post-WFC)\n', nnz(ctrl));
else
    ctrl = [];
    say('[3] control: none (uncontrolled budget)\n');
end

%% -- [4] covariance + Monte-Carlo ------------------------------------
rc = macos.budget.cov_budget(M, spec, 'control', ctrl);
rm = [];
if o.ndraw > 0
    rm = macos.budget.mc_budget(M, spec, 'ndraw', o.ndraw, 'control', ctrl);
    rel = abs(rm.total - rc.total) / max(rc.total, eps);
    say('[4] covariance total %.4g nm;  MC (%d draws) %.4g nm  [%.1f%%]\n', ...
        rc.total*M.cbm*1e9, o.ndraw, rm.total*M.cbm*1e9, 100*rel);
    if rel > o.mc_tol
        warning('budget:mc', ...
            'cov vs MC total disagree by %.1f%% (> %.0f%%)', 100*rel, 100*o.mc_tol);
    end
    if rc.controlled
        say('    uncontrolled (pre-WFC) total %.4g nm;  %d noise row(s)\n', ...
            rc.total_pre*M.cbm*1e9, numel(rc.extra));
    end
else
    say('[4] covariance total %.4g nm (MC skipped)\n', rc.total*M.cbm*1e9);
end

%% -- [5] roll-up + outputs -------------------------------------------
T_out = macos.budget.rollup(rc, 'mc', rm, 'cbm', M.cbm);
coef  = macos.budget.to_coef(rc, M, spec, 'cbm', M.cbm);
art = struct('M', M, 'spec', spec, 'cov', rc, 'mc', rm, 'table', T_out, ...
    'coef', coef, 'total_nm', coef.total, 'control', ctrl);
art.xlsx = pth('_budget.xlsx');
art.csv  = pth('_budget.csv');
art.mat  = pth('_budget.mat');
art.report = pth('_budget_report.txt');
macos.budget.write_xlsx(T_out, coef, art.xlsx, 'title', name);
macos.budget.write_csv(T_out, art.csv);
write_report_(art.report, name, M, rc, rm, ctrl);
save(art.mat, '-struct', 'coef');

%% -- [6] figure ------------------------------------------------------
art.figs = {};
if o.figures
    f = figure('Visible', tf_(o.visible));
    bar(rc.wfe * M.cbm * 1e9);
    set(gca, 'XTick', 1:numel(rc.names), 'XTickLabel', rc.names, ...
        'XTickLabelRotation', 40, 'YScale', 'log');
    ylabel('RMS WFE contribution (nm)'); grid on;
    ctxt = 'uncontrolled'; if ~isempty(ctrl), ctxt = 'post-WFC'; end
    title(sprintf('%s error budget (%s): total %.3g nm', name, ctxt, coef.total), ...
        'Interpreter', 'none');
    fig = pth('_budget.png');
    print(f, fig, '-dpng', '-r120'); close(f);
    art.figs{end+1} = fig;
end
say('\nartifacts: %s_budget.{xlsx,csv,mat}, report%s\n', name, ...
    tern_(isempty(art.figs), '', ', _budget.png'));
end

% =========================================================================
function [name, od] = names_(sens, o)
if ~isempty(o.name), name = char(o.name); else, name = ''; end
od = char(o.out_dir);
if (ischar(sens) || isstring(sens)) && isfile(char(sens))
    [d, b] = fileparts(char(sens));
    if isempty(name), name = regexprep(b, '_sens$', ''); end
    if isempty(od), od = d; end
end
if isempty(name), name = 'budget'; end
if isempty(od), od = pwd; end
end

% -------------------------------------------------------------------------
function ctrl = control_mask_(c, spec, M)
nz = numel(M.meta.channel);
if islogical(c) && isscalar(c)
    if ~c, ctrl = []; return; end
    ctrl = false(nz, 1);                 % union of groups tagged Control
    for k = 1:numel(spec.groups)
        if spec.groups(k).is_control, ctrl = ctrl | spec.groups(k).cols; end
    end
elseif islogical(c) || (isnumeric(c) && ~isscalar(c))
    ctrl = logical(c(:));
else
    ctrl = [];
end
end

% -------------------------------------------------------------------------
function write_report_(path, name, M, rc, rm, ctrl)
fid = fopen(path, 'w'); c = onCleanup(@() fclose(fid));
tonm = M.cbm * 1e9;
fprintf(fid, 'MACOS error budget: %s\n', name);
fprintf(fid, 'metric=%s  base_units=%s  cbm=%g  nw=%d  Nz=%d\n', ...
    M.units.metric, M.base_units, M.cbm, M.nw, numel(M.meta.channel));
if isempty(ctrl), fprintf(fid, 'control: none (uncontrolled)\n');
else, fprintf(fid, 'control: %d DOFs projected (post-WFC)\n', nnz(ctrl)); end
fprintf(fid, '\n%-24s %-12s %12s %12s %12s\n', 'group', 'class', ...
    'sigma', 'alpha', 'WFE(nm)');
for k = 1:numel(rc.names)
    fprintf(fid, '%-24s %-12s %12.4g %12.4g %12.4g\n', rc.names{k}, ...
        rc.class{k}, rc.sigma(k), rc.alpha(k), rc.wfe(k)*tonm);
end
for k = 1:numel(rc.extra)
    e = rc.extra(k);
    fprintf(fid, '%-24s %-12s %12.4g %12.4g %12.4g\n', e.name, e.kind, ...
        e.sigma, e.alpha, e.wfe*tonm);
end
if rc.controlled
    fprintf(fid, '%-24s %-12s %12s %12s %12.4g\n', 'pre-WFC TOTAL', '', ...
        '', '', rc.total_pre*tonm);
end
fprintf(fid, '%-24s %-12s %12s %12s %12.4g\n', 'TOTAL (RSS)', '', '', '', ...
    rc.total*tonm);
if ~isempty(rm)
    fprintf(fid, '\nMonte-Carlo (%d draws) total: %.4g nm  (cov %.4g nm)\n', ...
        rm.ndraw, rm.total*tonm, rc.total*tonm);
end
fprintf(fid, '\nmethod: Redding 2014, linearized ray-trace covariance budget.\n');
fprintf('wrote %s\n', path);
end

function s = tf_(v), if v, s = 'on'; else, s = 'off'; end, end
function s = tern_(c, a, b), if c, s = a; else, s = b; end, end
