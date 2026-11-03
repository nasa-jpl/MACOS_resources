function rm = mc_budget(M, spec, varargin)
%MC_BUDGET  Monte-Carlo error budget from a linearized ray-trace model.
%   RM = macos.budget.mc_budget(M, SPEC) draws many realizations of the
%   component errors from each group's distribution, propagates them
%   through the linear model w = Jk*x, and reports the RMS metric per
%   group (that group alone active) and for all groups together.  When
%   the assumptions match, this equals macos.budget.cov_budget to within
%   the Monte-Carlo scatter -- Redding 2014, Fig. 6.  Monte-Carlo also
%   admits non-normal / correlated draws (each group's cm.sampler) and
%   nonlinear controls.
%
%   M     from macos.budget.assemble (needs Jk = J(keep,:)).
%   SPEC  from macos.budget.spec_from_table (groups with .cols, .cm).
%
%   Options:
%     'ndraw'    draws (default 400)
%     'seed'     rng seed (default 1; set [] to leave rng untouched)
%     'control'  logical Nz control-DOF mask: the drawn state is corrected
%                by u = -pinv(Ju)*w before the residual (default [], none)
%
%   RM fields: .names, .wfe (Ng-by-1 per-group), .total, .ndraw.
%
%   Controls phase (2026-10-15): under control the loop is Redding 2014
%   Eq. 22 -- the WFS estimate w_est = w + dw (dw ~ SPEC.wfs.sigma per
%   sample), the pseudo-inverse controller u = -G*w_est + du (du drawn per
%   SPEC.act group), the residual w1 = w + Ju*u -- and:
%     .wfe     per group ALONE through the noise-free controller (post-WFC;
%              v1 reported the uncontrolled group-alone value here)
%     .wfe_pre per group alone, uncontrolled (always)
%     .extra   struct array (name kind wfe), one per cov_budget noise row:
%              'wfs' = the sensing draw alone through -Ju*G, 'act' = the
%              group's actuator draw alone through Ju (Eq. 31, unprojected)
%     .total   every group + every noise draw together
%   Noise draws come AFTER the group draws in each realization, so an
%   uncontrolled run consumes the same random stream as v1.
%
%   See also macos.budget.cov_budget, macos.budget.project_control.

p = inputParser;
p.addParameter('ndraw', 400, @(x) isscalar(x) && x >= 1);
p.addParameter('seed', 1);
p.addParameter('control', [], @(x) islogical(x) || isempty(x));
p.parse(varargin{:});
nd = round(p.Results.ndraw);
ctrl = p.Results.control;

assert(isfield(M, 'Jk') && ~isempty(M.Jk), ...
    'budget:mc', 'mc_budget needs M.Jk (J restricted to kept rows)');
Jk = M.Jk;
nz = size(Jk, 2);
ng = numel(spec.groups);

Gpi = [];
cidx = [];
if ~isempty(ctrl) && any(ctrl)
    cidx = find(ctrl);
    Ju = Jk(:, ctrl);
    Gpi = pinv(Ju);        % control solution u = -Gpi*w
end
% noise terms (controlled runs only): sensing per sample, actuator per group
sw = 0;  act = struct('name', {}, 'pos', {}, 'cm', {});
if ~isempty(Gpi)
    if isfield(spec, 'wfs') && ~isempty(spec.wfs), sw = spec.wfs.sigma; end
    if isfield(spec, 'act')
        for q = 1:numel(spec.act)
            [~, pos] = ismember(find(spec.act(q).cols), cidx);
            assert(all(pos > 0), 'budget:act', ...
                'actuator group "%s" has columns outside the control set', ...
                spec.act(q).name);
            act(end+1) = struct('name', spec.act(q).name, 'pos', pos, ...
                'cm', spec.act(q).cm); %#ok<AGROW>
        end
    end
end
haswfs = ~isempty(Gpi) && isfield(spec, 'wfs') && ~isempty(spec.wfs);
nx = haswfs + numel(act);
nwr = size(Jk, 1);

if ~isempty(p.Results.seed), rng(p.Results.seed); end

acc  = zeros(ng, 1);
accp = zeros(ng, 1);
accx = zeros(nx, 1);
acct = 0;
for d = 1:nd
    xfull = zeros(nz, 1);
    for k = 1:ng
        cols = find(spec.groups(k).cols);
        xk = spec.groups(k).cm.sampler();
        wk = Jk(:, cols) * xk;
        acc(k) = acc(k) + mean(wk.^2);     % group alone (uncontrolled)
        if ~isempty(Gpi)
            wk = wk - Ju * (Gpi * wk);     % group alone, post-WFC
            accp(k) = accp(k) + mean(wk.^2);
        end
        xfull(cols) = xfull(cols) + xk;
    end
    w = Jk * xfull;
    if ~isempty(Gpi)
        west = w;
        q = 0;
        if haswfs
            dw = sw * randn(nwr, 1);       % WFS noise, per sample
            west = west + dw;
            q = q + 1;
            accx(q) = accx(q) + mean((Ju * (Gpi * dw)).^2);
        end
        du = zeros(numel(cidx), 1);
        for a = 1:numel(act)
            dua = act(a).cm.sampler();
            du(act(a).pos) = du(act(a).pos) + dua;
            q = q + 1;
            accx(q) = accx(q) + mean((Ju(:, act(a).pos) * dua).^2);
        end
        u = -Gpi * west + du;              % image-based WFC least-squares
        w = w + Ju * u;                    % residual after control
    end
    acct = acct + mean(w.^2);
end

rm.names = {spec.groups.name}';
rm.wfe_pre = sqrt(acc / nd);
if isempty(Gpi)
    rm.wfe = rm.wfe_pre;
else
    rm.wfe = sqrt(accp / nd);
end
xn = {};  xk = {};
if haswfs, xn{end+1} = 'WFS Sensing Error'; xk{end+1} = 'wfs'; end
for a = 1:numel(act), xn{end+1} = act(a).name; xk{end+1} = 'act'; end %#ok<AGROW>
rm.extra = struct('name', {}, 'kind', {}, 'wfe', {});
for q = 1:nx
    rm.extra(q) = struct('name', xn{q}, 'kind', xk{q}, 'wfe', sqrt(accx(q) / nd));
end
rm.total = sqrt(acct / nd);
rm.ndraw = nd;
end
