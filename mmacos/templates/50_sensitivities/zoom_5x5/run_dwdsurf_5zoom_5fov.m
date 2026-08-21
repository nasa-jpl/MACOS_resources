% run_dwdsurf_5zoom_5fov.m -- multi-CONFIGURATION x multi-FIELD dw/dsurf.
% =====================================================================
%  The PRESCRIPTION rung of the configuration-axis family: per-element
%  radius (Kr) and conic (Kc) sensitivities, harvested per (zoom
%  position, field point).  Same axis, fixture and 5 x 5 grid as
%  run_dwdx_5zoom_5fov.m -- read that driver's header first for the
%  fixture provenance and the LOAD-CASE warning.
%
%  This is the cheapest rung: two parameters per optic, so the full
%  25-block harvest over the scoped elements runs in one sitting.
%
%  SETUP: run `mmacos_setup` once per MATLAB session first.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX       = fullfile(here, 'jwst_ote_designc.in');
MODEL    = 512;
NGRIDPTS = 63;
STOP_ELT = 25;
CFG_ELT  = 25;
FOV      = 2.90888e-4;
TILT     = 1.45444e-4;
PARAMS   = {'Kr', 'Kc'};
% =====================================================================

sched = table( ...
    ["z0"; "zUL";  "zUR";  "zLL";  "zLR"], ...
    [   0;  -TILT;  +TILT;  -TILT;  +TILT], ...
    [   0;  +TILT;  +TILT;  -TILT;  -TILT], ...
    'VariableNames', {'name', sprintf('%d.Rx', CFG_ELT), ...
                              sprintf('%d.Ry', CFG_ELT)});
cfgs = macos.design.configs_from_table(sched);

[~, rxstem] = fileparts(RX);
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdsurf", ...
    'configs', cfgs, 'resume_dir', string(fullfile(here, '_resume_dwdsurf')), ...
    'stop_elt', STOP_ELT, 'ngridpts', NGRIDPTS, 'model_size', MODEL, ...
    'surf_params', PARAMS, 'out_dir', here, ...
    'name', ['dwdsurf_5zoom_5fov_' rxstem]);

% ---- per-configuration summary --------------------------------------
% The blocks are contiguous and indxall.config carries the index per
% row -- that is the supported way to address one configuration's block.
A  = art.os.dwdsall;
nc = numel(cfgs);
nf = size(art.os.field_table, 1);
fprintf('=== dw/dsurf: %d configurations x %d fields = %d blocks ===\n', ...
    nc, nf, nc * nf);
fprintf('    stacked Jacobian %d x %d over %d channels\n', ...
    size(A, 1), size(A, 2), numel(art.os.channel_names));
for c = 1:nc
    r = (art.os.indxall.config == c);
    fprintf('    config %-5s: %6d rows, |J| max %.3e\n', ...
        art.os.config_names{c}, nnz(r), max(abs(A(r, :)), [], 'all'));
end
