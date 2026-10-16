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
if ~isempty(ctrl) && any(ctrl)
    Ju = Jk(:, ctrl);
    Gpi = pinv(Ju);        % control solution u = -Gpi*w
end

if ~isempty(p.Results.seed), rng(p.Results.seed); end

acc  = zeros(ng, 1);
acct = 0;
for d = 1:nd
    xfull = zeros(nz, 1);
    for k = 1:ng
        cols = find(spec.groups(k).cols);
        xk = spec.groups(k).cm.sampler();
        wk = Jk(:, cols) * xk;
        acc(k) = acc(k) + mean(wk.^2);     % group alone (uncontrolled)
        xfull(cols) = xfull(cols) + xk;
    end
    w = Jk * xfull;
    if ~isempty(Gpi)
        u = -Gpi * w;                      % image-based WFC least-squares
        w = w + Ju * u;                    % residual after control
    end
    acct = acct + mean(w.^2);
end

rm.names = {spec.groups.name}';
rm.wfe   = sqrt(acc / nd);
rm.total = sqrt(acct / nd);
rm.ndraw = nd;
end
