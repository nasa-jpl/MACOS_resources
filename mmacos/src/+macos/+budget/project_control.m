function [Gp, ctl] = project_control(M, ctrl)
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
%   [GP, CTL] = ... also returns the controller pieces the WFC stage's
%   noise rows and the metrology stage reuse (all at Jk's size or smaller):
%       CTL.idx   control column indices (find(CTRL))
%       CTL.Ju    Jk(:,CTRL), nw x nu
%       CTL.Gpi   pinv(Ju) = the pseudo-inverse controller G (Eq. 20), nu x nw
%       CTL.Gu    Ju'*Ju, nu x nu (the actuator Gram)
%       CTL.rank  rank of Ju (the sensing-error trace, Eq. 32, is rank/nw)
%       CTL.T     Gpi*Jk, nu x Nz (the controller's response to each DOF)
%       CTL.apply @(A) A - Ju*(Gpi*A) = Pu*A without forming Pu
%   (added 2026-10-15, controls phase; the one-output call is unchanged).
%
%   The second (metrology/drift) stage is macos.budget.project_met.
%
%   See also macos.budget.cov_budget, macos.budget.mc_budget.

assert(isfield(M, 'Jk') && ~isempty(M.Jk), ...
    'budget:control', 'project_control needs M.Jk');
ctrl = logical(ctrl(:));
Jk = M.Jk;
assert(numel(ctrl) == size(Jk, 2), 'budget:control', ...
    'control mask length %d ~= Nz %d', numel(ctrl), size(Jk, 2));
if ~any(ctrl)
    Gp = M.G;  ctl = [];  return;
end
Ju  = Jk(:, ctrl);
Gpi = pinv(Ju);
T   = Gpi * Jk;
PuJk = Jk - Ju * T;                  % (I - Ju*pinv(Ju)) * Jk, Jk-sized
Gp = PuJk' * PuJk;
if nargout > 1
    ctl = struct('idx', find(ctrl), 'Ju', Ju, 'Gpi', Gpi, 'Gu', Ju' * Ju, ...
        'rank', rank(Ju), 'T', T, 'apply', @(A) A - Ju * (Gpi * A));
end
end
