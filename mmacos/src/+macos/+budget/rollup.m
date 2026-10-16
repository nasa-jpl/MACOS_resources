function T = rollup(rc, varargin)
%ROLLUP  Assemble a budget table from the covariance (and MC) results.
%   T = macos.budget.rollup(RC) returns a MATLAB table, one row per group
%   plus a final TOTAL row, in the metric's nanometre units.
%   T = macos.budget.rollup(RC, 'mc', RM, 'cbm', CBM) adds the Monte-Carlo
%   WFE column and converts BaseUnit metric to nm via CBM*1e9 (CBM =
%   BaseUnit->metre; default 1 => metric already in metres).
%
%   See also macos.budget.cov_budget, macos.budget.mc_budget,
%   macos.budget.write_csv, macos.budget.write_xlsx.

p = inputParser;
p.addParameter('mc', [], @(x) isempty(x) || isstruct(x));
p.addParameter('cbm', 1, @isscalar);
p.parse(varargin{:});
rm = p.Results.mc;
tonm = p.Results.cbm * 1e9;     % BaseUnit -> nm

ng = numel(rc.names);
ex = struct('name', {}, 'kind', {}, 'sigma', {}, 'alpha', {}, 'var', {}, 'wfe', {});
if isfield(rc, 'extra'), ex = rc.extra; end
Name   = [rc.names(:); {ex.name}'; {'TOTAL'}];
Class  = [rc.class(:); {ex.kind}'; {''}];
Sigma  = [rc.sigma(:); [ex.sigma]'; NaN];
Alpha  = [rc.alpha(:); [ex.alpha]'; NaN];
WFE_nm = [rc.wfe(:); [ex.wfe]'; rc.total] * tonm;
ctl = isfield(rc, 'controlled') && rc.controlled;
if ctl
    % controlled: the uncontrolled column alongside (noise rows have none)
    WFE_pre_nm = [rc.wfe_pre(:); nan(numel(ex), 1); rc.total_pre] * tonm;
end

if ~isempty(rm)
    exm = [];
    if isfield(rm, 'extra') && ~isempty(rm.extra), exm = [rm.extra.wfe]'; end
    if numel(exm) ~= numel(ex), exm = nan(numel(ex), 1); end
    WFE_mc_nm = [rm.wfe(:); exm; rm.total] * tonm;
    if ctl
        T = table(Name, Class, Sigma, Alpha, WFE_pre_nm, WFE_nm, WFE_mc_nm);
    else
        T = table(Name, Class, Sigma, Alpha, WFE_nm, WFE_mc_nm);
    end
elseif ctl
    T = table(Name, Class, Sigma, Alpha, WFE_pre_nm, WFE_nm);
else
    T = table(Name, Class, Sigma, Alpha, WFE_nm);
end
T.Properties.VariableUnits = repmat({''}, 1, width(T));
T.Properties.Description = sprintf('error budget: %d groups, total %.4g nm', ...
    ng, rc.total * tonm);
end
