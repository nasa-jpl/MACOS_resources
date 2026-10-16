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
%   covariance is folded in and alpha_g == wfe_g.
%
%   See also macos.budget.mc_budget, macos.budget.project_control,
%   macos.budget.assemble, macos.budget.rollup.

p = inputParser;
p.addParameter('control', [], @(x) islogical(x) || isempty(x));
p.parse(varargin{:});
ctrl = p.Results.control;

nw = M.nw;
G  = M.G;
if ~isempty(ctrl) && any(ctrl)
    G = macos.budget.project_control(M, ctrl);   % projected Gram, small-matrix
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

for k = 1:ng
    g = spec.groups(k);
    cols = g.cols;
    cm = g.cm;
    rc.class{k} = cm.class;
    Ggg = G(cols, cols);
    if strcmp(cm.type, 'iid')
        a2 = sum(diag(Ggg)) / nw;             % sigma-free coefficient^2
        rc.alpha(k) = sqrt(max(a2, 0));
        rc.sigma(k) = cm.sigma_native;
        rc.var(k)   = cm.var * a2;            % = sigma_native^2 * alpha^2
    else
        v = sum(sum(cm.Sigma .* Ggg.')) / nw; % trace(Sigma*Ggg)/nw
        rc.var(k)   = max(v, 0);
        rc.alpha(k) = sqrt(rc.var(k));
        rc.sigma(k) = cm.sigma_native;
    end
    rc.wfe(k) = sqrt(rc.var(k));
end
rc.total = sqrt(sum(rc.var));
end
