% run_dwdx_multi_zoom.m -- multi-CONFIGURATION x multi-field dw/dx (GENERIC).
% =====================================================================
%  The configuration-axis sibling of run_dwdx_multi.m.  Same rigid-body
%  6-DOF wavefront Jacobian, same canonical state-vector form
%
%      wall = dwdxall * x + w0_stacked
%
%  but evaluated per (CONFIGURATION, field point) instead of per field
%  alone.  Both scripts are thin CONFIG wrappers over the sensitivity
%  stage runner design/runners/run_sensitivities.m.
%
%  TO RUN ON YOUR OWN SYSTEM: edit the CONFIG block -- RX, STOP_ELT, and
%  the SCHEDULE.  Everything below the CONFIG block is generic.
%
%  WHAT A CONFIGURATION IS.  A named set of element setting overrides:
%  a zoom position, or -- more often in our systems -- a COMPENSATION
%  state, e.g. a steering mirror at a pupil re-pointed to cancel
%  pointing drift.  You write the set as a SCHEDULE, the shape it
%  arrives in (a spreadsheet: one row per configuration, one column per
%  <elt>.<DOF>), and macos.design.configs_from_table turns it into the
%  'configs' array the supervisors take.  A row of zeros is a legal
%  configuration -- the nominal state, and a useful first row.
%
%  CONFIGURATIONS STACK AS ROWS, not as a third array dimension.  A
%  configuration adds OBSERVATIONS of the same state vector x, exactly
%  as a field point does, so every downstream consumer (run_compare, the
%  MET optimiser, the simulator) reads the result unchanged.  w for one
%  configuration stacks its FIELDS; w for the run stacks the
%  CONFIGURATIONS, so each owns a contiguous block of rows -- address
%  one with out.indxall.config == c.
%
%  The CANVAS is tiled the way the field set is, which is a DIFFERENT
%  walk from the row order and deliberately so: each configuration sits
%  at its own position on an outer grid, each cell holding that
%  configuration's whole field canvas.  So _opdall.png is a grid of
%  grids and position on the page means (configuration, field point).
%  See macos.config_canvas.
%
%  ELEMENT GROUPS.  GROUPS declares RIGID-BODY groups -- sets of
%  elements perturbed as ONE body by the engine's GPERTURB -- and each
%  contributes 6 more columns per (configuration, field) block, APPENDED
%  AFTER the per-element block.  That is the sensitivity an assembly
%  actually has: the members' responses partly CANCEL when they move
%  together, which summing their individual columns cannot reproduce.
%  It is [] here because a group is SYSTEM-SPECIFIC; the commented line
%  names the bundled demo deck's primary.  Group channels carry NO
%  element id (out.iElt is 0, as a source channel does) and out.kind is
%  'Group' -- section on kind, not on iElt.  Units are the same on both
%  sides: OPD-per-metre for translations, OPD-per-rad for rotations.
%
%  WHY THE CONFIGURATION BLOCKS CAN LOOK ALIKE, AND WHY THAT IS RIGHT.
%  The supervisor re-finds the exit pupil PER FIELD (reset_xp, default
%  true), and a tilt of a FLAT mirror AT A PUPIL is, to first order,
%  exactly a wavefront tilt -- which that re-reference removes.  So a
%  pupil-fold configuration's effect on the nominal wavefront collapses
%  by orders of magnitude and what remains in the Jacobian is the
%  SECOND-ORDER residual.  That residual is the quantity a
%  compensation-state study wants: the first-order term is what the
%  compensator is FOR.  Measured on the bundled deck, both legs with the
%  same statistic: nominal effect 3.033e-02 mm with the pupil frozen ->
%  4.043e-06 mm with the reset; Jacobian effect 1.886e-04 -> 5.308e-06
%  relative.  Neither is a bug.
%
%  RESUMABLE.  Each configuration's block is checkpointed into
%  RESUME_DIR as it completes and reloaded rather than recomputed if the
%  run is killed and restarted; the directory is pruned on success.
%  Set RESUME_DIR = "" to disable.  On the bundled deck the whole
%  5x5-block harvest is about 165 s, so the checkpoints are insurance
%  rather than a necessity -- they earn their keep when NGRIDPTS or
%  MODEL goes up, which is where the cost lives.
%
%  For worked, COMMITTED examples with the numbers see
%  templates/50_sensitivities/zoom_5x5/ (this deck, 5 zoom states x 5
%  fields, PM group ON) and .../run_dwdx_multi/ (single-configuration,
%  lens cell).
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'design', 'runners'));
addpath(fullfile(here, '..', 'design', 'src'));   % flag_zero_norm_channels
addpath(here);   % plot_* / group_exhibit / save_dw_flat live beside this

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX       = '';          % <-- YOUR .in FILE GOES HERE (absolute path)
MODEL    = 512;         % model size (>= your aperture grid sampling)
NGRIDPTS = 63;          % ray-grid sampling override ([] = keep the .in value)
FOV      = 2.90888e-4;  % half-field (rad); 5-field set = centre + 4 corners
STOP_ELT = 25;          % set the stop AT this element ([] = the deck's
                        % own ApStop=).  The exit-pupil machinery needs a
                        % stop; the demo deck carries none and its pupil
                        % IS an element (the fine steering mirror).
CFG_ELT  = 25;          % the element the configuration axis moves
TILT     = 1.45444e-4;  % configuration tilt (rad) = 0.5 arcmin
% Finite-difference step, scalar or (1,6) = [Rx Ry Rz Tx Ty Tz].
% Rotations rad, translations SI metres.  See run_dwdx_multi.m for why
% the vector form is the safer default (translation convergence).
DELTA    = [1e-8 1e-8 1e-8 1e-6 1e-6 1e-6];
% Rigid-body element GROUPS: containers.Map name -> member element ids.
% [] = none.  For the bundled demo deck the physically motivated group
% is the primary-mirror backplane, its 18 real segments as one body:
%   GROUPS = containers.Map('KeyType','char','ValueType','any');
%   GROUPS('PM') = (5:22).';
GROUPS   = [];
EXCLUDE  = [];          % element ids to force-drop ([] = drop whatever the
                        % zero-norm flag reports dead)
RESUME_DIR = string(fullfile(here, '_resume_dwdx_zoom'));   % "" = off
%
%  Bundled demo deck, used when RX is empty.  EXPLICIT path -- a runner
%  that reaches for examples/<its own name>/ breaks silently when the
%  asset directory moves.  It is one CONFIG line here.
DEMO_RX = fullfile(here, '..', 'templates', '50_sensitivities', 'zoom_5x5', ...
                   'jwst_ote_designc.in');
% =====================================================================

% ---- the configuration schedule -------------------------------------
% Nominal, then CFG_ELT tilted to each corner of a square, as a LOCAL
% -frame rotation.  configs_from_table emits one rotation-only perturb
% per row, which is also what keeps the restore exact.  Replace this
% table with your own -- or with readtable('schedule.csv',
% 'VariableNamingRule', 'preserve').
SCHEDULE = table( ...
    ["z0"; "zUL";  "zUR";  "zLL";  "zLR"], ...
    [   0;  -TILT;  +TILT;  -TILT;  +TILT], ...   % Rx
    [   0;  +TILT;  +TILT;  -TILT;  -TILT], ...   % Ry
    'VariableNames', {'name', sprintf('%d.Rx', CFG_ELT), ...
                              sprintf('%d.Ry', CFG_ELT)});
% =====================================================================

if isempty(RX)
    RX = DEMO_RX;
    fprintf('[demo] RX not set -- using bundled example: %s\n', RX);
end
assert(isfile(RX), 'run_dwd:noDeck', ...
    'prescription not found: %s\n(set RX, or fix DEMO_RX in the CONFIG block)', RX);

cfgs = macos.design.configs_from_table(SCHEDULE);
[~, rxstem] = fileparts(RX);
name = ['dwdx_multi_zoom_' rxstem];

extra = {};
if ~isempty(STOP_ELT), extra = {'stop_elt', STOP_ELT}; end
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdx", ...
    'configs', cfgs, 'resume_dir', RESUME_DIR, extra{:}, ...
    'ngridpts', NGRIDPTS, 'model_size', MODEL, 'delta_x', DELTA, ...
    'groups', GROUPS, 'per_element', "center", ...
    'out_dir', here, 'name', name);

% Group-vs-member table appended to <name>_sens_report.txt (no-op when
% GROUPS is empty).  Tabulate one representative member -- an
% 18-segment group does not need 18 rows.
gm = [];
if isa(GROUPS, 'containers.Map') && GROUPS.Count > 0
    k = keys(GROUPS);  v = double(GROUPS(k{1}));  gm = v(1);
end
group_exhibit(art.ox, GROUPS, fullfile(here, [name '_sens_report.txt']), ...
    'members', gm);

% ---- drop dead (obscured) optics, number-free -----------------------
% dw/dx builds 6 DOFs for every actual optic, including any virtual or
% fully-obscured element whose rigid-body sensitivity is ~zero.  Flag by
% RESPONSE (flag_zero_norm_channels), not by a hard-coded id.  Group
% channels are named Grp[...] and carry no 'Elt N' tag, so a drop never
% touches them -- a group column is a distinct rigid-body motion, not a
% sum of its members' columns.
dead = flag_zero_norm_channels(art.ox);
drop = EXCLUDE;  if isempty(drop), drop = dead; end
art.ox = drop_channels(art.ox, drop);

% ---- flat, channel-named .mat (dwdx / indxall / w0_stacked at top
%      level -- not inside an 'ox' struct) -----------------------------
save_dw_flat(art.ox, fullfile(here, [name '.mat']), ...
    'name', 'dwdx', 'model_size', MODEL);

nc = numel(cfgs);
nf = size(art.ox.field_table, 1);
fprintf('=== dw/dx %d configurations x %d fields = %d blocks ===\n', ...
    nc, nf, nc * nf);
fprintf('    stacked Jacobian %d x %d over %d channels\n', ...
    size(art.ox.dwdxall, 1), size(art.ox.dwdxall, 2), ...
    numel(art.ox.channel_names));
if isfield(art.ox, 'kind') && any(strcmp(art.ox.kind, 'Group'))
    fprintf('    incl. %d group channel(s)\n', nnz(strcmp(art.ox.kind, 'Group')));
end
for c = 1:nc
    r = (art.ox.indxall.config == c);
    fprintf('    config %-5s: %6d rows, |dwdx| max %.3e\n', ...
        art.ox.config_names{c}, nnz(r), ...
        max(abs(art.ox.dwdxall(r, :)), [], 'all'));
end
