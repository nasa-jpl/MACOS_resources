function rm = mc_met(M, dspec, H, R, varargin)
%MC_MET  Monte-Carlo of the metrology / drift stage, estimator in the loop.
%   RM = macos.budget.mc_met(M, DSPEC, H, R, ...) draws the drift state x
%   from DSPEC's groups and the gauge noise n ~ R, and runs the loop that
%   macos.budget.project_met evaluates in closed form:
%       y = H*x + n,  x_est = K*y (MMSE),  u = -G*Jk*x_est + du,
%       w = Jk*x + Ju*u.
%   Same options as project_met ('control', 'hcols', 'act') plus 'ndraw'
%   (default 400) and 'seed' (default 1).
%
%   RM fields: .names, .wfe_pre (group alone, uncorrected), .wfe_post
%   (group alone through estimator + controller, noise-free), .extra
%   (struct array name kind wfe: the gauge noise alone, then each actuator
%   group alone), .total_pre (all drifts, uncorrected), .total_post
%   (everything), .ndraw.
%
%   See also macos.budget.project_met, macos.budget.mc_budget.

p = inputParser;
p.addParameter('control', [], @(x) islogical(x) || isempty(x));
p.addParameter('hcols', [], @isnumeric);
p.addParameter('act', [], @(x) isempty(x) || isstruct(x));
p.addParameter('ndraw', 400, @(x) isscalar(x) && x >= 1);
p.addParameter('seed', 1);
p.parse(varargin{:});
nd = round(p.Results.ndraw);
ctrl = p.Results.control;

% the estimator / controller pieces, from the closed form (one source)
rt = macos.budget.project_met(M, dspec, H, R, 'control', ctrl, ...
    'hcols', p.Results.hcols);
K = rt.K;  Hf = rt.H;  Rf = rt.R;
Jk = M.Jk;  nz = size(Jk, 2);  ng = numel(dspec.groups);
hasc = ~isempty(ctrl) && any(ctrl);
if hasc
    cidx = find(ctrl);  Ju = Jk(:, ctrl);  Gpi = pinv(Ju);
else
    cidx = [];  Ju = zeros(size(Jk, 1), 0);  Gpi = zeros(0, size(Jk, 1));
end
act = p.Results.act;  if isempty(act), act = struct('cols', {}, 'cm', {}, 'name', {}); end
apos = cell(numel(act), 1);
for a = 1:numel(act)
    [~, apos{a}] = ismember(find(act(a).cols), cidx);
    assert(all(apos{a} > 0), 'budget:act', ...
        'actuator group "%s" has columns outside the control set', act(a).name);
end
Lr = chol(Rf + 0 * eye(size(Rf, 1)), 'lower');
ctrl_of = @(xe) -Gpi * (Jk * xe);         % u for an estimated state

if ~isempty(p.Results.seed), rng(p.Results.seed); end
accp = zeros(ng, 1);  accq = zeros(ng, 1);  accx = zeros(1 + numel(act), 1);
acc0 = 0;  acct = 0;
for d = 1:nd
    x = zeros(nz, 1);
    for k = 1:ng
        c = find(dspec.groups(k).cols);
        xk = dspec.groups(k).cm.sampler();
        xg = zeros(nz, 1);  xg(c) = xk;
        w0 = Jk * xg;
        accp(k) = accp(k) + mean(w0.^2);
        wq = w0 + Ju * ctrl_of(K * (Hf * xg));
        accq(k) = accq(k) + mean(wq.^2);
        x = x + xg;
    end
    n = Lr * randn(size(Rf, 1), 1);
    wn = Ju * ctrl_of(K * n);
    accx(1) = accx(1) + mean(wn.^2);
    du = zeros(numel(cidx), 1);
    for a = 1:numel(act)
        dua = act(a).cm.sampler();
        du(apos{a}) = du(apos{a}) + dua;
        accx(1 + a) = accx(1 + a) + mean((Ju(:, apos{a}) * dua).^2);
    end
    w0 = Jk * x;
    acc0 = acc0 + mean(w0.^2);
    w = w0 + Ju * (ctrl_of(K * (Hf * x + n)) + du);
    acct = acct + mean(w.^2);
end
rm.names = {dspec.groups.name}';
rm.wfe_pre = sqrt(accp / nd);
rm.wfe_post = sqrt(accq / nd);
rm.extra = struct('name', 'Metrology Gauge Noise', 'kind', 'met', ...
    'wfe', sqrt(accx(1) / nd));
for a = 1:numel(act)
    rm.extra(end+1) = struct('name', act(a).name, 'kind', 'act', ...
        'wfe', sqrt(accx(1 + a) / nd));
end
rm.total_pre = sqrt(acc0 / nd);
rm.total_post = sqrt(acct / nd);
rm.ndraw = nd;
end
