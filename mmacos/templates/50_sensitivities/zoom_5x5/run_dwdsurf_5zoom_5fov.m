% run_dwdsurf_5zoom_5fov.m -- multi-CONFIGURATION x multi-FIELD dw/dsurf.
% =====================================================================
%  The PRESCRIPTION rung of the configuration-axis family: per-element
%  radius (Kr) and conic (Kc) sensitivities, harvested per (zoom
%  position, field point).  Same axis, fixture and 5 x 5 grid as
%  run_dwdx_5zoom_5fov.m -- read that driver's header first for the
%  fixture provenance and the LOAD-CASE warning.
%
%  This is the cheapest rung: two parameters per optic, so the full
%  25-block harvest over every powered optic runs in one sitting.  On this
%  deck the powered optics are the SM (M2, elt 23) and TM (M3, elt 24),
%  each varied in Kr and Kc separately -> 4 channels.  (The segments carry
%  a finite conic too but are Element= Segment, outside dw/dsurf's
%  powered-Reflector/Refractor target set -- per-segment Kr/Kc is a later
%  extension.)
%
%  PISTON + TIP + TILT ARE REMOVED.  A radius (Kr) or conic (Kc) error
%  re-focuses and re-points the beam, and that global piston + pointing is
%  normally ALIGNED OUT during assembly -- so this driver passes
%  'surf_remove_ptt', true and each Kr/Kc response has its piston + two
%  tilts projected out (over the optic's own exit-pupil footprint),
%  leaving the surviving higher-order figure that a sensitivity budget
%  actually cares about.
%
%  DEAD OPTICS ARE DROPPED, NUMBER-FREE.  dw/dsurf builds a channel for
%  every powered optic, which includes element 4 (CenterSegment) -- a
%  VIRTUAL, almost-entirely-obscured element whose sensitivity is ~zero.
%  Rather than hard-code "skip 4", the harvest is flagged after the fact
%  by flag_zero_norm_channels (design/src) -- which keys on the RESPONSE,
%  not an id -- and the flagged element's channels are dropped from the
%  saved Jacobian.  Set EXCLUDE below to force-drop specific ids instead.
%
%  Outputs: <name>.mat is FLAT -- dwdsurf / indxall / w0_stacked /
%  channel_names / config_* at the TOP LEVEL (not in an 'os' struct).
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
EXCLUDE  = [];          % element ids to force-drop ([] = drop whatever the
                        % zero-norm flag reports dead, e.g. the obscured elt 4)
% =====================================================================

sched = table( ...
    ["z0"; "zUL";  "zUR";  "zLL";  "zLR"], ...
    [   0;  -TILT;  +TILT;  -TILT;  +TILT], ...
    [   0;  +TILT;  +TILT;  -TILT;  -TILT], ...
    'VariableNames', {'name', sprintf('%d.Rx', CFG_ELT), ...
                              sprintf('%d.Ry', CFG_ELT)});
cfgs = macos.design.configs_from_table(sched);

[~, rxstem] = fileparts(RX);
name = ['dwdsurf_5zoom_5fov_' rxstem];
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdsurf", ...
    'configs', cfgs, 'resume_dir', string(fullfile(here, '_resume_dwdsurf')), ...
    'stop_elt', STOP_ELT, 'ngridpts', NGRIDPTS, 'model_size', MODEL, ...
    'surf_params', PARAMS, 'surf_remove_ptt', true, ...
    'out_dir', here, 'name', name);

% ---- drop dead (obscured) optics, number-free -----------------------
% Flag any all-zero channel group (the virtual centre segment), then drop
% those channels -- or the explicit EXCLUDE set if given.
dead = flag_zero_norm_channels(art.os);
drop = EXCLUDE;  if isempty(drop), drop = dead; end
art.os = drop_channels(art.os, drop);

% ---- flat, channel-named .mat (dwdsurf at top level) ----------------
save_dw_flat(art.os, fullfile(here, [name '.mat']), ...
    'name', 'dwdsurf', 'model_size', MODEL);

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
