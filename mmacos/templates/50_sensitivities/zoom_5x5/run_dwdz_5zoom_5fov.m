% run_dwdz_5zoom_5fov.m -- multi-CONFIGURATION x multi-FIELD dw/dz (example).
% =====================================================================
%  The MonZernike FIGURE rung of the configuration-axis family:
%  segment-LOCAL MonZernike figure modes, harvested per (zoom position,
%  field point) -- 5 x 5 = 25 blocks -- from ONE call, in the canonical
%  state-vector form  wall = dwdzall * z + w0.  Same axis, same fixture
%  and the same 5 x 5 grid as run_dwdx_5zoom_5fov.m; read that driver's
%  header FIRST -- it carries the fixture provenance, the tiled-canvas
%  explanation, and the LOAD-CASE warning, all of which apply here too.
%
%  THE FIXTURE CARRIES ZERO-AMPLITUDE FIGURE CHANNELS.  The 19 segments
%  of jwst_ote_designc.in are Surface= FreeForm with a MonZernike channel
%  whose coefficients are zero -- an optically inert promotion of the
%  original Surface= Conic (deck header; PLAN_CONFIGURATIONS.md departure
%  #6; gated in tRunSensitivities).  So there is a MonZernike channel to
%  differentiate, and the nominal wavefront is unchanged from the conic
%  deck.  The DOFs have NO design authority; this is machinery, not optics.
%
%  *** THE NUMBERS HERE ARE A LOAD CASE, NOT A DESIGN. ***  At +-1 arcmin
%  the wavefront error is ~0.64 mm (~278 waves at 2.3 um).  The fixture's
%  worth is that it traces cleanly and responds by orders of magnitude on
%  both axes, giving the finite-difference machinery something to
%  differentiate -- not that the WFE means anything.  See the deck header.
%
%  SCOPE.  This harvests the MonZernike channel on ALL 19 segments (the
%  dw/dz supervisor contract sweeps every FreeForm optic; on this deck the
%  FreeForm set IS the 19 segments), over the mode range MODES.  MODES is
%  the scope knob: the shipped 4:6 (three modes -> 57 channels) is a
%  DEMONSTRATION; widen it toward the supervisor default 4:11 for a fuller
%  figure basis, at proportionally longer runtime.  The resume directory
%  carries a long run across restarts.
%
%  Note on 'elts': it is NOT a segment filter here -- dw/dz still lists
%  every FreeForm segment, and 'elts' only picks which of them get the
%  full MODES range (the rest fall back to their Rx-default single mode).
%  So this driver leaves it unset and harvests all 19 uniformly, matching
%  run_dwdx_5zoom_5fov.m, which harvests every element.
%
%  RESUMABLE: each configuration's block is checkpointed into _resume_dwdz/
%  as it completes and reloaded rather than recomputed on a restart; the
%  directory is pruned on success.  Delete it by hand to force a cold run.
%
%  Outputs (this directory): <name>_sens_report.txt + _sens.mat +
%  _opdall / _svspec / _svspec_configs / _dwdz_channels.png + per-element
%  pages.
%
%  SETUP: run `mmacos_setup` once per MATLAB session first.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX       = fullfile(here, 'jwst_ote_designc.in');
MODEL    = 512;         % engine model size
NGRIDPTS = 63;          % ray-grid override (the deck declares 1024)
STOP_ELT = 25;          % the FSM IS the pupil; the deck carries no ApStop=
CFG_ELT  = 25;          % the element the configuration axis steers
FOV      = 2.90888e-4;  % half-field (rad) = 1 arcmin, 5-field set
TILT     = 1.45444e-4;  % configuration tilt (rad) = 0.5 arcmin
MODES    = 4:6;         % MonZernike modes on EVERY segment (scope knob;
                        % widen toward the supervisor default 4:11)
% =====================================================================

% ---- the configuration schedule (see run_dwdx_5zoom_5fov.m) ---------
sched = table( ...
    ["z0"; "zUL";  "zUR";  "zLL";  "zLR"], ...
    [   0;  -TILT;  +TILT;  -TILT;  +TILT], ...   % Rx
    [   0;  +TILT;  +TILT;  -TILT;  -TILT], ...   % Ry
    'VariableNames', {'name', sprintf('%d.Rx', CFG_ELT), ...
                              sprintf('%d.Ry', CFG_ELT)});
cfgs = macos.design.configs_from_table(sched);

[~, rxstem] = fileparts(RX);
name = ['dwdz_5zoom_5fov_' rxstem];
art  = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdz", ...
    'configs', cfgs, 'resume_dir', string(fullfile(here, '_resume_dwdz')), ...
    'stop_elt', STOP_ELT, 'ngridpts', NGRIDPTS, 'model_size', MODEL, ...
    'zmodes_fig', MODES, 'per_element', "center", ...
    'out_dir', here, 'name', name);

% ---- per-configuration summary --------------------------------------
% The blocks are contiguous and indxall.config carries the index per row
% -- that is the supported way to address one configuration's block.
A  = art.oz.dwdzall;
nc = numel(cfgs);
nf = size(art.oz.field_table, 1);
fprintf('=== dw/dz: %d configurations x %d fields = %d blocks ===\n', ...
    nc, nf, nc * nf);
fprintf('    stacked Jacobian %d x %d over %d channels\n', ...
    size(A, 1), size(A, 2), numel(art.oz.channel_names));
for c = 1:nc
    r = (art.oz.indxall.config == c);
    fprintf('    config %-5s: %6d rows, |dwdz| max %.3e\n', ...
        art.oz.config_names{c}, nnz(r), max(abs(A(r, :)), [], 'all'));
end
