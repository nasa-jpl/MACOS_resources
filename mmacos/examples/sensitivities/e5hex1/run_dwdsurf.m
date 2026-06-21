% run_dwdsurf.m -- multi-field dw/dKr + dw/dKc (powered-surface) on
% e5hex1.in via mmacos.  Parallels run_dwdx.m (rigid-body) and
% run_dwdz.m (Zernike), for the radius + conic DOFs.
%
% Builds the (Kr, Kc) Jacobian of every POWERED optic (Element=
% Reflector / Refractor with |Kr| << the flat sentinel) at 5 field
% points (center + 4 corners at +/- 100 urad).  On e5hex1 the powered
% set is the secondary (Reflector) + the corrector lens surfaces
% (Refractor); the segmented primary is Element= Segment, outside the
% Reflector/Refractor filter.
%
% e5hex1.in declares ApStop= 0 0 0 in its header, so no explicit STOP
% is needed.  delta = 1e-6 (Kr in BaseUnits, Kc dimensionless).

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..'));

here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
rx = fullfile(here, 'e5hex1.in');

fprintf('=== e5hex1: dw/dKr + dw/dKc 5-field default, powered optics ===\n');
m = macos.Session(128);
out = macos.dw_dsurf_multi(m, rx, ...
    'field_x_rad', 1e-4, ...
    'field_y_rad', 1e-4, ...
    'delta', 1e-6);

field_table   = out.field_table;
field_names   = out.field_names;
chfraydir_nom = out.chfraydir_nom;
dwdsall       = out.dwdsall;
w0_stacked    = out.w0_stacked;
indxall       = out.indxall;
channel_names = out.channel_names;
iElt          = out.iElt;
param         = out.param;
delta         = out.delta;
method        = out.method;
wf_elt        = out.wf_elt;
rx            = out.rx_path;
opdall_shape  = size(out.OPDall);
model_size    = 128;

out_path = fullfile(here, 'dwdsall_e5hex1.mat');
save(out_path, ...
    'dwdsall', 'w0_stacked', 'indxall', 'channel_names', 'iElt', 'param', ...
    'field_table', 'field_names', 'chfraydir_nom', 'delta', 'method', ...
    'wf_elt', 'rx', 'opdall_shape', 'model_size');
fprintf('wrote %s\n', out_path);
