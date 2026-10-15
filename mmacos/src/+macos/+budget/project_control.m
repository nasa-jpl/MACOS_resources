function Gp = project_control(M, ctrl)
%PROJECT_CONTROL  Control-projected Gram for a post-WFC error budget.
%   GP = macos.budget.project_control(M, CTRL) returns the Gram of the
%   control-projected Jacobian, Gp = (Pu*Jk)'*(Pu*Jk), where CTRL is a
%   logical Nz mask of the control (actuator) DOFs, Ju = Jk(:,CTRL), and
%   Pu = I - Ju*pinv(Ju) is the control projection matrix (Redding 2014,
%   Eq. 23 -- what leaks through the pseudo-inverse controller G =
%   pinv(Ju) is the fitting-error residual).  Pu is never materialized
%   Nw-by-Nw: PuJk = Jk - Ju*(pinv(Ju)*Jk) is formed at Jk's size.
%
%   Feeding GP in place of M.G into macos.budget.cov_budget yields the
%   post-control (uncorrectable) WFE contribution of each error group.
%
%   This is the v1 single-stage WFC projection; the second metrology/drift
%   MMSE stage (run_met.m's post = X - X*H'*((H*X*H'+R)\(H*X))) layers on
%   top in the controls phase.
%
%   See also macos.budget.cov_budget, macos.budget.mc_budget.

assert(isfield(M, 'Jk') && ~isempty(M.Jk), ...
    'budget:control', 'project_control needs M.Jk');
ctrl = logical(ctrl(:));
Jk = M.Jk;
assert(numel(ctrl) == size(Jk, 2), 'budget:control', ...
    'control mask length %d ~= Nz %d', numel(ctrl), size(Jk, 2));
if ~any(ctrl)
    Gp = M.G; return;
end
Ju  = Jk(:, ctrl);
PuJk = Jk - Ju * (pinv(Ju) * Jk);    % (I - Ju*pinv(Ju)) * Jk, Jk-sized
Gp = PuJk' * PuJk;
end
