% run_budget_e5hex1.m -- worked example: an optical error budget from a
% dw/dx sensitivity harvest on the public 7-segment hex deck e5hex1.in.
%
% Pipeline: harvest the rigid-body Jacobian (macos.dw_dx_multi, 5 fields,
% model 128), declare the error groups (budget_spec_e5hex1.m), then
% run_budget -> covariance + Monte-Carlo budget, report, a self-contained
% .xlsx (WFE + Coefficients), a CSV, a coef .mat, and a per-group figure.
%
% This reproduces, on a public deck, the method of Redding 2014 (SPIE)
% "Optical error budgeting using linearized ray-trace models": a properly
% formed RSS budget is a covariance analysis, and Monte-Carlo gives the
% same answer.  Not a -batch test -- it never calls exit().

here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
root = fullfile(here, '..', '..', '..');
addpath(fullfile(root, 'src'));
addpath(fullfile(root, 'design', 'runners'));
addpath(here);                                    % budget_spec_e5hex1 on path

rx = fullfile(here, '..', 'e5hex1', 'e5hex1.in');

fprintf('=== e5hex1: harvest dw/dx (5 fields, 6 DOF/elt, model 128) ===\n');
m = macos.Session(128);
ox = macos.dw_dx_multi(m, rx, ...
    'field_x_rad', 1e-4, 'field_y_rad', 1e-4, 'delta', 1e-8);

fprintf('=== error budget ===\n');
art = run_budget(ox, 'spec', budget_spec_e5hex1(), ...
    'name', 'e5hex1', 'out_dir', here, 'ndraw', 1000, 'visible', false);

fprintf('\ntotal WFE %.4g nm (cov) / %.4g nm (MC %d draws)\n', ...
    art.cov.total * art.M.cbm * 1e9, art.mc.total * art.M.cbm * 1e9, art.mc.ndraw);
fprintf('artifacts in %s\n', here);
