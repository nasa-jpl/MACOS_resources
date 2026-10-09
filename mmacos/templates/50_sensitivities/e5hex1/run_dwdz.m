% run_dwdz.m -- multi-field dw/dz_Zernike on e5hex1.in via mmacos.
%
% Mirrors pymacos/tests/sensitivities/e5hex1/run_dwdz.sh.  Default
% demo: 12 Zernike modes (Z4..Z15) per Z-eligible element at 5 field
% points (center + 4 corners at +/- 100 urad) -- runs in ~80 s on a
% laptop.  To match pymacos's full Z4..Z45 sweep (~9 min), edit
% n_zcoef below to 45.  Saves dwdxall_e5hex1.mat with the canonical
% state-vector layout:
%
%     wall = dwdxall * x + w0_stacked
%
% Inputs the user must set per Rx:
%   field_x_rad / field_y_rad  direction-cosine offsets for corner FPs
%                              (additive on top of the Rx's nominal
%                              ChfRayDir; per-axis independent).
%
% Outputs land in this directory:
%   dwdxall_e5hex1.mat              dwdxall + w0_stacked + indxall +
%                                   channel_names + field_table + ...
%   verify_grid_dwdxall_e5hex1*.png after running verifyall.m

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'src'));

here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
rx = fullfile(here, 'e5hex1.in');

fprintf('=== e5hex1: dw/dz 5-field default, Z4..Z15 (12 modes) per element ===\n');
m = macos.Session(128);
out = macos.dw_dz_zernike_multi(m, rx, ...
    'field_x_rad', 1e-4, ...
    'field_y_rad', 1e-4, ...
    'zmode_start', 4, ...
    'n_zcoef', 15, ...
    'delta', 1e-6);

% Save in the same .mat layout as pymacos so the shared verifyall.m
% lifecycle (m2v / v2m / indxall round-trip) just works.
field_table  = out.field_table;
field_names  = out.field_names;
chfraydir_nom = out.chfraydir_nom;
dwdxall      = out.dwdxall;
dwdzall      = out.dwdzall;
w0_stacked   = out.w0_stacked;
indxall      = out.indxall;
channel_names = out.channel_names;
delta        = out.delta;
method       = out.method;
wf_elt       = out.wf_elt;
rx           = out.rx_path;     % pymacos verifyall.m expects 'rx'
opdall_shape = size(out.OPDall);
model_size   = 128;             % pymacos verifyall.m prints model_size

out_path = fullfile(here, 'dwdxall_e5hex1.mat');
save(out_path, ...
    'dwdxall', 'dwdzall', 'w0_stacked', 'indxall', 'channel_names', ...
    'field_table', 'field_names', 'chfraydir_nom', 'delta', 'method', ...
    'wf_elt', 'rx', 'opdall_shape', 'model_size');
fprintf('wrote %s\n', out_path);
