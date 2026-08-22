% run_dwdx_5zoom_5fov.m -- multi-CONFIGURATION x multi-FIELD dw/dx (example).
% =====================================================================
%  Thin driver over the general sensitivity stage runner
%  design/runners/run_sensitivities.m ('dwdx' channel only), exercising
%  the CONFIGURATION axis: the rigid-body wavefront Jacobian evaluated
%  per (zoom position, field point) -- 5 x 5 = 25 blocks -- from ONE
%  call, in the canonical state-vector form  wall = dwdxall * x + w0.
%
%  A CONFIGURATION is a named set of element setting overrides.  Here it
%  is a COMPENSATION state, which is what the axis is usually for in our
%  systems: element 25 is a flat FINE STEERING MIRROR at a pupil, and the
%  five configurations point it to the centre and to the four corners of
%  a 0.5 arcmin square -- the states a pointing-drift compensator would
%  actually sit in.  Element 25 is ALSO one of the elements whose 6 DOFs
%  are Jacobian channels, so a configuration moves an optic that is
%  itself a variable and its columns are evaluated about a SHIFTED
%  operating point.  That is not a conflict to design around; it is what
%  a zoom-dependent sensitivity IS.
%
%  Configurations stack as extra ROWS of the Jacobian -- a configuration
%  adds observations of the SAME state vector x, exactly as a field point
%  does -- so run_compare, the MET optimiser and the simulator consume
%  the result unchanged.  w for one zoom stacks its FIELDS and w for the
%  run stacks the ZOOMS, so each zoom owns a contiguous block of rows;
%  address one with out.indxall.config == c.
%
%  The CANVAS is tiled the way the field set is, which is a different
%  walk from the row order and deliberately so: each zoom state sits at
%  its own position on an outer 3x3 grid (four corners and the centre),
%  each cell holding that state's whole five-field canvas.  So
%  _opdall.png is a quincunx of quincunxes and position on the page means
%  (zoom state, field point).  See macos.config_canvas.
%
%  WHY THE FIVE BLOCKS LOOK ALIKE, AND WHY THAT IS RIGHT.  The
%  supervisor re-finds the exit pupil PER FIELD (reset_xp, default true),
%  and a tilt of a FLAT mirror AT A PUPIL is, to first order, exactly a
%  wavefront tilt -- which that re-reference removes.  So the
%  configuration's effect on the nominal wavefront collapses from
%  2.7e-02 mm (measured with the pupil frozen) to 2.3e-07 mm, and its
%  effect on the Jacobian is the SECOND-ORDER residual: 1.7e-05 relative
%  (2.4e-05 frozen).  That residual is the quantity a compensation-state
%  sensitivity study actually wants -- the first-order term is what the
%  compensator is FOR.  Pass 'reset_xp', false through if you want the
%  frozen-pupil view instead; both are gated, neither is a bug.
%
%  *** THE NUMBERS HERE ARE A LOAD CASE, NOT A DESIGN. ***
%  jwst_ote_designc.in is an early 18-segment JWST OTE design study, NOT
%  the flight prescription, and this driver drives it far outside its
%  field: at +-1 arcmin the wavefront error is about 0.64 mm, roughly 278
%  waves at the deck's 2.3 um.  The point of the fixture is that it
%  traces cleanly (no ray loss) and responds by orders of magnitude on
%  BOTH axes, so the finite-difference machinery has something to
%  differentiate.  Do not read the WFE numbers as a result.  Provenance
%  and the published-prescription comparison: the deck header and
%  ../../../design/PLAN_CONFIGURATIONS.md section 6.
%
%  SETUP: run `mmacos_setup` once per MATLAB session first.
%  Self-contained: ships jwst_ote_designc.in beside the script.  TO RUN
%  ON YOUR OWN SYSTEM, point RX at your .in, set STOP_ELT to your stop
%  and CFG_ELT to the element your configurations move.
%
%  RESUMABLE: a 25-block harvest at model 512 is a multi-hour run, so
%  each configuration's block is checkpointed into _resume/ as it
%  completes and reloaded rather than recomputed if the run is killed
%  and restarted.  The directory is pruned automatically on success.
%  Delete it by hand to force a cold recompute.
%
%  DEAD OPTICS DROPPED, NUMBER-FREE.  dw/dx builds 6 DOFs for every actual
%  optic, including element 4 (CenterSegment) -- a VIRTUAL, almost-entirely
%  -obscured element (it passes only the chief-ray sliver) whose rigid-body
%  sensitivity is ~zero.  It is flagged after the harvest by its RESPONSE
%  (flag_zero_norm_channels, design/src -- not a hard-coded id) and its
%  channels are dropped from the saved Jacobian.  SM (23) and TM (24) ARE
%  included (full beam).
%
%  Outputs (this directory): <name>.mat is FLAT -- dwdx / indxall /
%  w0_stacked / channel_names / config_* at the TOP LEVEL, the channel's
%  own name, no empty wrapper structs -- plus <name>_sens_report.txt +
%  _opdall/_svspec/_svspec_configs/_dwdx_channels.png + per-element pages.
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
DELTA    = 1e-8;        % finite-difference step (rigid-body)
EXCLUDE  = [];          % element ids to force-drop ([] = drop whatever the
                        % zero-norm flag reports dead, e.g. the obscured elt 4)
% =====================================================================

% ---- the configuration schedule -------------------------------------
% Centred, then the FSM tilted to each corner of a square, as a LOCAL
% -frame rotation.  Written as the table a zoom schedule arrives in
% (a spreadsheet); configs_from_table emits one rotation-only perturb
% per row, which is also what keeps the restore exact.
sched = table( ...
    ["z0"; "zUL";  "zUR";  "zLL";  "zLR"], ...
    [   0;  -TILT;  +TILT;  -TILT;  +TILT], ...   % Rx
    [   0;  +TILT;  +TILT;  -TILT;  -TILT], ...   % Ry
    'VariableNames', {'name', sprintf('%d.Rx', CFG_ELT), ...
                              sprintf('%d.Ry', CFG_ELT)});
cfgs = macos.design.configs_from_table(sched);

[~, rxstem] = fileparts(RX);
name = ['dwdx_5zoom_5fov_' rxstem];
art  = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdx", ...
    'configs', cfgs, 'resume_dir', string(fullfile(here, '_resume')), ...
    'stop_elt', STOP_ELT, 'ngridpts', NGRIDPTS, 'model_size', MODEL, ...
    'delta_x', DELTA, 'per_element', "center", ...
    'out_dir', here, 'name', name);

% ---- drop dead (obscured) optics, number-free -----------------------
% dw/dx builds 6 DOFs for every actual optic, including element 4
% (CenterSegment) -- a VIRTUAL, almost-entirely-obscured element whose
% rigid-body sensitivity is ~zero (column norms ~1e-7 vs ~0.2 for a real
% segment).  Flag it by RESPONSE (flag_zero_norm_channels, not a hard-coded
% id) and drop its channels.  Set EXCLUDE to force-drop specific ids.
dead = flag_zero_norm_channels(art.ox);
drop = EXCLUDE;  if isempty(drop), drop = dead; end
art.ox = drop_channels(art.ox, drop);

% ---- flat, channel-named .mat (dwdx / indxall / w0_stacked at top
%      level -- not in an 'ox' struct) ---------------------------------
save_dw_flat(art.ox, fullfile(here, [name '.mat']), ...
    'name', 'dwdx', 'model_size', MODEL);

nc = numel(cfgs);
nf = size(art.ox.field_table, 1);
fprintf('=== dw/dx %d configurations x %d fields = %d blocks ===\n', ...
    nc, nf, nc * nf);
fprintf('    stacked Jacobian %d x %d over %d channels\n', ...
    size(art.ox.dwdxall, 1), size(art.ox.dwdxall, 2), ...
    numel(art.ox.channel_names));
for c = 1:nc
    r = (art.ox.indxall.config == c);
    fprintf('    config %-5s: %6d rows, |dwdx| max %.3e\n', ...
        art.ox.config_names{c}, nnz(r), ...
        max(abs(art.ox.dwdxall(r, :)), [], 'all'));
end
