function T = budget_spec_e5hex1()
%BUDGET_SPEC_E5HEX1  Example error-budget group table for e5hex1.
%   One row per budget line, grouped by rigid-body DOF type across all
%   optics -- the canonical macos.budget schema (authorable equally as
%   this .m, a CSV, or an Excel sheet; see macos.budget.read_spec).
%   Sigmas here are illustrative alignment-class values; edit for a real
%   budget.  Consumed by run_budget_e5hex1.m / run_budget.
%
%   Columns: Name Elements DOFs Modes Param Channel Dist Sigma Units
%            Min Max Cov Control.

rows = { ...
  'Optics Tilt',     'all', 'tilt',     '', '', 'dwdx', 'normal',   50, 'nrad', NaN, NaN, 'iid', ''; ...
  'Optics Clocking', 'all', 'clock',    '', '', 'dwdx', 'normal',  100, 'nrad', NaN, NaN, 'iid', ''; ...
  'Optics Decenter', 'all', 'decenter', '', '', 'dwdx', 'normal',  100, 'nm',   NaN, NaN, 'iid', ''; ...
  'Optics Despace',  'all', 'despace',  '', '', 'dwdx', 'uniform', NaN, 'nm',  -200, 200, 'iid', ''};

T = cell2table(rows, 'VariableNames', {'Name','Elements','DOFs','Modes', ...
    'Param','Channel','Dist','Sigma','Units','Min','Max','Cov','Control'});
end
