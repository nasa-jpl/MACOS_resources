function rc = cov_budget(M, spec, varargin)
%COV_BUDGET  Covariance error budget from a linearized ray-trace model.
%   RC = macos.budget.cov_budget(M, SPEC) computes, for each error group,
%   the WFE (or any metric the assembled model M carries) contribution
%
%       var_g = trace(Pu * D_g * Sigma_g * D_g' * Pu') / nw
%             = trace(Sigma_g * (Pu*D_g)'(Pu*D_g)) / nw          (Gram form)
%
%   and the RSS roll-up WFE = sqrt(sum_g var_g).  This is Redding 2014
%   (SPIE) "Optical error budgeting using linearized ray-trace models":
%   a properly-formed RSS budget IS a covariance analysis.  The Gram trick
%   (G = Dk'*Dk, trace(P*G)) never forms the dense Nw-by-Nw matrix -- same
%   kernel as mmacos/design/runners/run_met.m.
%
%   M     from macos.budget.assemble (fields G, nw; and Jk when controls
%         are active).
%   SPEC  from macos.budget.spec_from_table (groups with .cols, .cm).
%
%   Options:
%     'control'  logical Nz column mask of control (actuator) DOFs; the
%                uncontrolled residual uses Pu = I - Ju*pinv(Ju) via the
%                projected Gram.  Default [] (uncontrolled, Pu = I).
%
%   RC fields (per group, Ng-by-1): .names .class .sigma .alpha .var .wfe
%   and scalars .total, .controlled.  ALPHA is the sigma-free coefficient
%   (WFE_g = sigma_g * alpha_g) for iid groups; for 'full' groups the
%   covariance is folded in and alpha_g == wfe_g.  Under control ALPHA /
%   VAR / WFE are the POST-WFC values (Eq. 30), as in v1.
%
%   Controls phase (2026-10-15), always present:
%     .alpha_pre .var_pre .wfe_pre (Ng-by-1)  uncontrolled (Pu = I)
%     .total_pre                              uncontrolled RSS
%   and, under control, one row per noise term in .extra (struct array:
%   name kind sigma alpha var wfe), included in .total:
%     kind 'wfs'  sensing error that leaks through the controller, Eq. 32:
%                 alpha^2 = trace(Ju*G*G'*Ju')/nw = trace(Gu*pinv(Gu))/nw
%                 = rank(Ju)/nw, sigma = SPEC.wfs.sigma (per sample)
%     kind 'act'  actuator error per control group, Eq. 31 (NOT projected:
%                 Eq. 22's (dw/du)*du term): alpha^2 = trace(G_aa)/nw
%   Every row (group or noise) also carries its engineering-unit view for
%   the workbook: .section .units .sigma_eng .per_eng (groups: Ng-by-1
%   cells / vectors; noise rows: fields of .extra), with sigma_native =
%   sigma_eng * per_eng.
%
%   SPEC.wfs / SPEC.act come from spec_from_table's 'wfs_sigma' /
%   'act_sigma'; absent, there are no noise rows.
%
%     'met'  struct(drift, H, R[, hcols][, act]): the metrology / drift
%            stage (iSeq = 2) through macos.budget.project_met with the
%            same control mask -> RC.met (its .total_pre / .total_post,
%            per-drift-group .alpha_met / .alpha_met_post, the gauge row)
%            and RC.total_all = sqrt(total^2 + met.total_post^2), the
%            HabEx 'Total WFE' (post-WFC static + post-metrology drift).
%            Actuator rows for the stage: met.act, else SPEC.act.
%
%   See also macos.budget.mc_budget, macos.budget.project_control,
%   macos.budget.assemble, macos.budget.rollup.

p = inputParser;
p.addParameter('control', [], @(x) islogical(x) || isempty(x));
p.addParameter('met', [], @(x) isempty(x) || isstruct(x));
p.parse(varargin{:});
ctrl = p.Results.control;

nw = M.nw;
G  = M.G;
ctl = [];
if ~isempty(ctrl) && any(ctrl)
    [G, ctl] = macos.budget.project_control(M, ctrl); % projected Gram, small
    rc.controlled = true;
else
    rc.controlled = false;
end

ng = numel(spec.groups);
rc.names = {spec.groups.name}';
rc.class = cell(ng, 1);
rc.sigma = nan(ng, 1);
rc.alpha = zeros(ng, 1);
rc.var   = zeros(ng, 1);
rc.wfe   = zeros(ng, 1);
rc.alpha_pre = zeros(ng, 1);
rc.var_pre   = zeros(ng, 1);
rc.section   = repmat({'Optics'}, ng, 1);
rc.type      = repmat({'iid'}, ng, 1);
rc.units     = repmat({''}, ng, 1);
rc.sigma_eng = nan(ng, 1);
rc.per_eng   = ones(ng, 1);

for k = 1:ng
    g = spec.groups(k);
    cm = g.cm;
    rc.class{k} = cm.class;
    rc.sigma(k) = cm.sigma_native;
    [rc.alpha(k), rc.var(k)] = group_term_(cm, G(g.cols, g.cols), nw);
    [rc.alpha_pre(k), rc.var_pre(k)] = group_term_(cm, M.G(g.cols, g.cols), nw);
    rc.wfe(k) = sqrt(rc.var(k));
    rc.type{k} = cm.type;
    if isfield(g, 'section') && ~isempty(g.section), rc.section{k} = g.section; end
    if isfield(cm, 'per_eng')
        rc.units{k} = cm.units;  rc.per_eng(k) = cm.per_eng;
        rc.sigma_eng(k) = cm.sigma_eng;
    else
        rc.sigma_eng(k) = cm.sigma_native;
    end
end
rc.wfe_pre = sqrt(rc.var_pre);
rc.total_pre = sqrt(sum(rc.var_pre));

% ---- WFC noise rows (Eq. 31, 32) -----------------------------------------
rc.extra = struct('name', {}, 'kind', {}, 'sigma', {}, 'alpha', {}, ...
    'var', {}, 'wfe', {}, 'units', {}, 'sigma_eng', {}, 'per_eng', {});
if rc.controlled
    if isfield(spec, 'wfs') && ~isempty(spec.wfs)
        a2 = trace(ctl.Gu * pinv(ctl.Gu)) / nw;   % = rank(Ju)/nw
        rc.extra(end+1) = row_('WFS Sensing Error', 'wfs', ...
            spec.wfs.sigma, a2, spec.wfs.sigma^2 * a2, spec.wfs.units, ...
            spec.wfs.sigma / max(spec.wfs.sigma_eng, realmin));
    end
    if isfield(spec, 'act')
        for q = 1:numel(spec.act)
            ac = spec.act(q);
            [a, v] = group_term_(ac.cm, M.G(ac.cols, ac.cols), nw);
            [u, f] = eng_(ac.cm);
            rc.extra(end+1) = row_(ac.name, 'act', ac.cm.sigma_native, a^2, v, u, f);
        end
    end
end
rc.total = sqrt(sum(rc.var) + sum([rc.extra.var]));

% ---- metrology / drift stage (iSeq = 2) ----------------------------------
mt = p.Results.met;
rc.met = [];
if ~isempty(mt)
    act = [];
    if isfield(mt, 'act'), act = mt.act;
    elseif isfield(spec, 'act'), act = spec.act; end
    hc = [];  if isfield(mt, 'hcols'), hc = mt.hcols; end
    ga = {};
    if isfield(mt, 'gauge_units'), ga = [ga, {'gauge_units', mt.gauge_units}]; end
    if isfield(mt, 'gauge_per_eng'), ga = [ga, {'gauge_per_eng', mt.gauge_per_eng}]; end
    rc.met = macos.budget.project_met(M, mt.drift, mt.H, mt.R, ...
        'control', ctrl, 'hcols', hc, 'act', act, ga{:});
    rc.total_all = sqrt(rc.total^2 + rc.met.total_post^2);
end
end

% =========================================================================
function [alpha, v] = group_term_(cm, Ggg, nw)
% One group's coefficient and variance from its Gram block.
if strcmp(cm.type, 'iid')
    a2 = sum(diag(Ggg)) / nw;             % sigma-free coefficient^2
    alpha = sqrt(max(a2, 0));
    v = cm.var * a2;                      % = sigma_native^2 * alpha^2
else
    v = max(sum(sum(cm.Sigma .* Ggg.')) / nw, 0);   % trace(Sigma*Ggg)/nw
    alpha = sqrt(v);
end
end

function r = row_(name, kind, sigma, a2, v, units, per_eng)
if nargin < 6, units = ''; end
if nargin < 7, per_eng = 1; end
r = struct('name', name, 'kind', kind, 'sigma', sigma, ...
    'alpha', sqrt(max(a2, 0)), 'var', v, 'wfe', sqrt(max(v, 0)), ...
    'units', units, 'sigma_eng', sigma / per_eng, 'per_eng', per_eng);
end

function [u, f] = eng_(cm)
% (units, per_eng) of a covariance descriptor, tolerant of v1-shaped ones
u = '';  f = 1;
if isfield(cm, 'per_eng'), u = cm.units;  f = cm.per_eng; end
end
