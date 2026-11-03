function rt = project_met(M, dspec, H, R, varargin)
%PROJECT_MET  Metrology / drift stage (iSeq = 2) of a covariance error budget.
%   RT = macos.budget.project_met(M, DSPEC, H, R) budgets the DRIFTS after
%   the last WFC update, held by a metrology-driven estimator and the WFC
%   controller (the HabEx "Metrology Control" columns):
%
%     x       drift state (Nz), covariance X from DSPEC's groups (the
%             "Drift Inputs" sigma; the group-table schema, built by
%             macos.budget.spec_from_table on the drift table)
%     y       = H*x + n,  n ~ R          the metrology (gauge) readings
%     x_est   = K*y,  K = X*H'*(H*X*H' + R)^-1      MMSE (run_met.m's K)
%     u       = -G*Jk*x_est + du,  G = pinv(Ju)     the WFC controller
%                                                   acting on the ESTIMATE
%     w       = Jk*x + Ju*u
%             = Jk*(I - E*T*K*H)*x - Jk*E*T*K*n + Ju*du
%
%   with T = G*Jk and E the control-column embedding.  So
%     pre-metrology   per group: trace(X_g * G)/nw          (no correction)
%     post-metrology  per group: trace(X_g * Q'*G*Q)/nw,  Q = I - E*T*K*H
%     gauge noise row (kind 'met'):  trace(R * B'*G*B)/nw,  B = E*T*K
%     actuator rows   (kind 'act'):  as the WFC stage (Eq. 31), when given
%   all at Nz size (Gram discipline; Nw x Nw is never formed).  Two limits
%   fix the reading: a drift the metrology does not see (H*e_j = 0) has
%   K*H*e_j = 0, so it PASSES THROUGH UNCORRECTED (post = pre: the WFC is
%   not re-run between updates; only the metrology drives the controller);
%   and with every state actuated (Jk full column rank) T = I and the sum of
%   the post rows is run_met.m's merit sqrt(trace(post*G)/nw), post = X -
%   X*H'*((H*X*H' + R)\(H*X)) (the Joseph identity (I-KH)X(I-KH)' + KRK' =
%   post for the MMSE K).
%
%   M      macos.budget.assemble model (G, Jk, nw, cbm)
%   DSPEC  drift spec (spec_from_table output, or a struct with .groups
%          carrying .name .cols .cm, as cov_budget takes)
%   H      metrology Jacobian, nm x Nz -- or nm x numel(HCOLS) with the
%          'hcols' option (e.g. run_met's dmet_dx over its body columns)
%   R      gauge-noise covariance, nm x nm (a scalar s^2 or a vector of
%          variances is expanded)
%
%   Options:
%     'control'  logical Nz control mask (the WFC actuators).  [] = no
%                actuators: the estimate corrects nothing (post == pre).
%     'hcols'    column indices of M that H's columns map to (default 1:Nz)
%     'act'      actuator noise records (spec.act from spec_from_table)
%                for this stage's actuator rows (default none)
%     'gauge_units', 'gauge_per_eng'  the engineering view of the gauge
%                row's sigma (default: native units, per_eng 1)
%
%   RT fields: .names (drift groups), .alpha_met .alpha_met_post (sigma-free,
%   iid groups; = wfe for 'full' groups), .var_pre .var_post .wfe_pre
%   .wfe_post, .extra (struct array name kind sigma alpha var wfe: the gauge
%   row 'met' + 'act' rows), .total_pre .total_post, .K (Nz x nm), .X,
%   .post (run_met's posterior), .Q.
%
%   See also macos.budget.cov_budget, macos.budget.mc_met,
%   macos.budget.project_control, run_met.

p = inputParser;
p.addParameter('control', [], @(x) islogical(x) || isempty(x));
p.addParameter('hcols', [], @isnumeric);
p.addParameter('act', [], @(x) isempty(x) || isstruct(x));
p.addParameter('gauge_units', 'native', @(x) ischar(x) || isstring(x));
p.addParameter('gauge_per_eng', 1, @isscalar);
p.parse(varargin{:});
ctrl = p.Results.control;

Jk = M.Jk;  nw = M.nw;  G = M.G;
nz = size(Jk, 2);
[H, R] = met_inputs_(H, R, p.Results.hcols, nz);
X = drift_cov_(dspec, nz);

% ---- run_met.m's estimator, verbatim ------------------------------------
post = X - X*H'*((H*X*H' + R) \ (H*X));
K = (X*H') / (H*X*H' + R);

% ---- the controller acting on the estimate ------------------------------
ETK = zeros(nz, size(H, 1));              % E*T*K: state -> control action
if ~isempty(ctrl) && any(ctrl)
    [~, ctl] = macos.budget.project_control(M, ctrl);
    ETK(ctl.idx, :) = ctl.T * K;
end
Q = eye(nz) - ETK * H;
A = Q' * G * Q;                           % post-metrology Gram, Nz x Nz
B = ETK' * G * ETK;                       % gauge-noise Gram,   nm x nm

ng = numel(dspec.groups);
rt.names = {dspec.groups.name}';
rt.sigma = nan(ng, 1);
rt.alpha_met = zeros(ng, 1);  rt.alpha_met_post = zeros(ng, 1);
rt.var_pre = zeros(ng, 1);    rt.var_post = zeros(ng, 1);
for k = 1:ng
    g = dspec.groups(k);  c = g.cols;
    rt.sigma(k) = g.cm.sigma_native;
    [rt.alpha_met(k), rt.var_pre(k)] = term_(g.cm, G(c, c), nw);
    [rt.alpha_met_post(k), rt.var_post(k)] = term_(g.cm, A(c, c), nw);
end
rt.wfe_pre = sqrt(rt.var_pre);
rt.wfe_post = sqrt(rt.var_post);
rt.units = repmat({''}, ng, 1);  rt.per_eng = ones(ng, 1);  rt.sigma_eng = rt.sigma;
rt.section = repmat({'Metrology'}, ng, 1);
for k = 1:ng
    if isfield(dspec.groups(k), 'section') && ~isempty(dspec.groups(k).section)
        rt.section{k} = dspec.groups(k).section;
    end
    cm = dspec.groups(k).cm;
    if isfield(cm, 'per_eng')
        rt.units{k} = cm.units;  rt.per_eng(k) = cm.per_eng;  rt.sigma_eng(k) = cm.sigma_eng;
    end
end

rt.extra = struct('name', {}, 'kind', {}, 'sigma', {}, 'alpha', {}, ...
    'var', {}, 'wfe', {}, 'units', {}, 'sigma_eng', {}, 'per_eng', {});
vg = max(sum(sum(R .* B.')) / nw, 0);     % trace(R*B)/nw
sg = sqrt(mean(diag(R)));
fg = p.Results.gauge_per_eng;
rt.extra(end+1) = struct('name', 'Metrology Gauge Noise', 'kind', 'met', ...
    'sigma', sg, 'alpha', sqrt(vg) / max(sg, realmin), 'var', vg, 'wfe', sqrt(vg), ...
    'units', char(p.Results.gauge_units), 'sigma_eng', sg / fg, 'per_eng', fg);
act = p.Results.act;
for q = 1:numel(act)
    [a, v] = term_(act(q).cm, G(act(q).cols, act(q).cols), nw);
    u = '';  f = 1;
    if isfield(act(q).cm, 'per_eng'), u = act(q).cm.units;  f = act(q).cm.per_eng; end
    rt.extra(end+1) = struct('name', act(q).name, 'kind', 'act', ...
        'sigma', act(q).cm.sigma_native, 'alpha', a, 'var', v, 'wfe', sqrt(v), ...
        'units', u, 'sigma_eng', act(q).cm.sigma_native / f, 'per_eng', f);
end
rt.total_pre = sqrt(sum(rt.var_pre));
rt.total_post = sqrt(sum(rt.var_post) + sum([rt.extra.var]));
rt.K = K;  rt.X = X;  rt.post = post;  rt.Q = Q;  rt.H = H;  rt.R = R;
rt.controlled = ~isempty(ctrl) && any(ctrl);
end

% =========================================================================
function [H, R] = met_inputs_(H, R, hcols, nz)
if ~isempty(hcols)
    assert(numel(hcols) == size(H, 2), 'budget:met', ...
        'hcols has %d entries, H has %d columns', numel(hcols), size(H, 2));
    Hf = zeros(size(H, 1), nz);
    Hf(:, hcols) = H;
    H = Hf;
end
assert(size(H, 2) == nz, 'budget:met', ...
    'H has %d columns, the model %d (pass ''hcols'')', size(H, 2), nz);
nm = size(H, 1);
if isscalar(R), R = R * eye(nm);
elseif isvector(R), R = diag(R(:));
end
assert(isequal(size(R), [nm nm]), 'budget:met', 'R must be %dx%d', nm, nm);
end

function X = drift_cov_(dspec, nz)
% Drift covariance over the model's columns, from the drift groups.
X = zeros(nz);
for k = 1:numel(dspec.groups)
    g = dspec.groups(k);  c = find(g.cols);
    if strcmp(g.cm.type, 'iid')
        X(c, c) = X(c, c) + g.cm.var * eye(numel(c));
    else
        X(c, c) = X(c, c) + g.cm.Sigma;
    end
end
end

function [alpha, v] = term_(cm, Agg, nw)
if strcmp(cm.type, 'iid')
    a2 = sum(diag(Agg)) / nw;
    alpha = sqrt(max(a2, 0));
    v = cm.var * a2;
else
    v = max(sum(sum(cm.Sigma .* Agg.')) / nw, 0);
    alpha = sqrt(v);
end
end
