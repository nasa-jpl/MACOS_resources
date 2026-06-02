% run_dwdx.m -- multi-field dw/dx (rigid-body) on e5hex1.in via mmacos.
%
% Mirrors pymacos/tests/sensitivities/e5hex1/run_dwdx.sh.  Computes
% the full 6-DOF Jacobian (Rx, Ry, Rz, Tx, Ty, Tz) for every actual
% optic at 5 field points (center + 4 corners at +/- 100 urad).
%
% e5hex1.in declares ApStop= 0 0 0 in its header, so no explicit
% STOP needed.  fp_mode='track' (default) drags the EP rigidly with
% the FP so FP perturbations produce non-zero sensitivities.

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..'));

here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
rx = fullfile(here, 'e5hex1.in');

fprintf('=== e5hex1: dw/dx 5-field default, 6 DOFs per element ===\n');
m = macos.Session(128);
out = macos.dw_dx_multi(m, rx, ...
    'field_x_rad', 1e-4, ...
    'field_y_rad', 1e-4, ...
    'delta', 1e-8);

field_table  = out.field_table;
field_names  = out.field_names;
chfraydir_nom = out.chfraydir_nom;
dwdxall      = out.dwdxall;
w0_stacked   = out.w0_stacked;
indxall      = out.indxall;
channel_names = out.channel_names;
delta        = out.delta;
method       = out.method;
wf_elt       = out.wf_elt;
rx           = out.rx_path;
opdall_shape = size(out.OPDall);
model_size   = 128;

out_path = fullfile(here, 'dwdxall_e5hex1_rigidbody.mat');
save(out_path, ...
    'dwdxall', 'w0_stacked', 'indxall', 'channel_names', ...
    'field_table', 'field_names', 'chfraydir_nom', 'delta', 'method', ...
    'wf_elt', 'rx', 'opdall_shape', 'model_size');
fprintf('wrote %s\n', out_path);
