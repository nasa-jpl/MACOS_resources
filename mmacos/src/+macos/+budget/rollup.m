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
Name   = [rc.names(:); {'TOTAL'}];
Class  = [rc.class(:); {''}];
Sigma  = [rc.sigma(:); NaN];
Alpha  = [rc.alpha(:); NaN];
WFE_nm = [rc.wfe(:); rc.total] * tonm;

if ~isempty(rm)
    WFE_mc_nm = [rm.wfe(:); rm.total] * tonm;
    T = table(Name, Class, Sigma, Alpha, WFE_nm, WFE_mc_nm);
else
    T = table(Name, Class, Sigma, Alpha, WFE_nm);
end
T.Properties.VariableUnits = repmat({''}, 1, width(T));
T.Properties.Description = sprintf('error budget: %d groups, total %.4g nm', ...
    ng, rc.total * tonm);
end
