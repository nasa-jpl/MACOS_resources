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
%  3.033e-02 mm with the pupil FROZEN to 4.043e-06 mm with the per-field
%  reset -- a factor 7500 -- and its effect on the Jacobian drops to the
%  SECOND-ORDER residual, 5.308e-06 relative against 1.886e-04 frozen.
%  Measured on THIS configuration (PM group on, the (1,6) DELTA below),
%  both legs from one script so the statistic is the same on each side:
%  nominal effect = max over fields and configurations of |W(cfg)-W(z0)|
%  at pixels valid in both; Jacobian effect = the same max, relative,
%  over the PER-ELEMENT columns.  (Earlier revisions of this header
%  quoted 2.7e-02 / 2.3e-07 / 1.7e-05 / 2.4e-05 from a statistic that
%  was not recorded; the definition above is stated so the numbers can
%  be reproduced.)  That residual is the quantity a compensation-state
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
%  RESUMABLE: each configuration's block is checkpointed into _resume/
%  as it completes and reloaded rather than recomputed if the run is
%  killed and restarted.  The directory is pruned automatically on
%  success; delete it by hand to force a cold recompute.  As shipped the
%  whole 25-block harvest takes about 165 s (measured 164 s, Linux, gfortran-built
%  engine, model 512 / NGRIDPTS 63 / 138 channels), so the checkpoints
%  are cheap insurance rather than a necessity -- they earn their keep
%  when you raise NGRIDPTS or MODEL, which is where the cost lives.
%  (This header used to call it "a multi-hour run"; on the shipped
%  settings it is not.)
%
%  DEAD OPTICS DROPPED, NUMBER-FREE.  dw/dx builds 6 DOFs for every actual
%  optic, including element 4 (CenterSegment) -- a VIRTUAL, almost-entirely
%  -obscured element (it passes only the chief-ray sliver) whose rigid-body
%  sensitivity is ~zero.  It is flagged after the harvest by its RESPONSE
%  (flag_zero_norm_channels, design/src -- not a hard-coded id) and its
%  channels are dropped from the saved Jacobian.  SM (23) and TM (24) ARE
%  included (full beam).
%
%  THE PM AS ONE RIGID BODY -- an ELEMENT GROUP, ON here.  The deck's
%  18 real segments (elts 5-22) are declared as the group 'PM', so the
%  harvest carries, alongside each segment's own 6 DOFs, the six columns
%  of the primary-mirror BACKPLANE moving as a single body.  Those six
%  are what a pointing/alignment budget actually spends: a backplane
%  thermal tilt is one rigid motion of the whole PM, not 18 independent
%  segment motions that happen to agree.  The engine perturbs the
%  members together via GPERTURB, so the column is the true rigid-body
%  response rather than a MATLAB sum of 18 individually-large columns.
%
%  WHAT THE PM COLUMNS SAY.  The driver appends a "[PM exhibit]" table
%  to <name>_sens_report.txt -- the group's six column norms beside one
%  segment's, both per rad / per METRE.  Group/segment comes out at
%  18.6667 (Rx), 18.9996 (Ry), 19.0411 (Tx), 18.5822 (Ty): a rigid
%  motion of N=18 alike members is N times one member, as it must be.
%  PISTON is the one worth the harvest -- 0.0428, i.e. the whole PM is
%  23x LESS piston-sensitive than a single segment.  A segment pistoning
%  puts a STEP into the wavefront; the whole PM pistoning is a global
%  despace the exit-pupil reference largely absorbs.  That cancellation
%  is what a per-element budget cannot see: summing 18 large per-segment
%  piston columns does not reproduce it.  (Rz reads 11515.6161 for the
%  opposite reason -- clocking one near-symmetric segment is nearly
%  inert, clocking the whole PM about the axis is not.)
%
%  Groups append 6 columns AFTER the per-element block, in EVERY
%  (configuration, field) block, so the stacked column order stays
%  [per-element] [group] and the supervisor's channel-identity assertion
%  covers them.  Group channels carry no element id (iElt 0, like a
%  source channel) -- they are labelled Grp[<name>] and out.kind is
%  'Group', which is what the per-element pages section on to give the
%  group its own page.  Groups are RIGID-BODY groups and reach the dwdx
%  rung ONLY; the figure / surface rungs have no group analogue in the
%  engine.  'groups_auto' would read EltGrp= declarations straight out
%  of the deck; this one carries none, so the map below is explicit.
%
%  UNITS: group and per-element columns share one convention --
%  OPD-per-metre for translations, OPD-per-rad for rotations -- so the
%  PM columns and a segment's are directly comparable and one numeric
%  DELTA is one physical poke for either.  (GroupedRigidBodyChannel
%  converts SI metres to the BaseUnits prb_grp wants, exactly as
%  macos.perturb does for the per-element channel.  It did not always;
%  see that channel's do_perturb comment and tDwDxGroups/
%  test_scalar_delta_matches_the_split_step for what that cost.)
%
%  WHY DELTA IS A (1,6) VECTOR -- convergence, not units.  Rotations sit
%  at 1e-8 rad; translations at 1e-6 (1 um) because the per-element
%  translation columns are still 1.9e-04 away from their converged
%  values at a 1e-8 m poke and 1.8e-06 away at 1e-6.  Rotations show no
%  such drift, so only the translation entries move.
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
% Finite-difference step, (1,6) = [Rx Ry Rz Tx Ty Tz].  Rotations in
% rad, translations in SI metres, for BOTH the per-element and the group
% channels -- see "WHY DELTA IS A (1,6) VECTOR" above for why the
% translation entries are 1e-6 and not the 1e-8 the rotations use.
DELTA    = [1e-8 1e-8 1e-8 1e-6 1e-6 1e-6];
EXCLUDE  = [];          % element ids to force-drop ([] = drop whatever the
                        % zero-norm flag reports dead, e.g. the obscured elt 4)
% Rigid-body element GROUPS: name -> column vector of member element
% ids ([] = none).  Here: the 18 REAL segments as one PM backplane.
% Element 4 (CenterSegment) is deliberately NOT a member -- it is a
% virtual, almost-entirely-obscured element (see DEAD OPTICS below), so
% including it would add nothing and muddy the story.
GROUPS   = containers.Map('KeyType', 'char', 'ValueType', 'any');
GROUPS('PM') = (5:22).';
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
    'delta_x', DELTA, 'per_element', "center", 'groups', GROUPS, ...
    'out_dir', here, 'name', name);

% ---- the PM-group exhibit -------------------------------------------
% The group's six columns beside one segment's, appended to the report
% so the committed artifact carries the numbers the README quotes.  One
% representative member is tabulated -- 18 alike segments do not need 18
% rows.  The helper divides the group TRANSLATION columns by CBM so both
% sides are per-metre; see its header for the units argument.
group_exhibit(art.ox, GROUPS, ...
    fullfile(here, [name '_sens_report.txt']), 'members', 5);

% ---- drop dead (obscured) optics, number-free -----------------------
% (group channels are named Grp[...] and carry no 'Elt N' tag, so an
% element drop never touches them -- a group column is a distinct rigid
% -body motion, not a sum of its members' columns)
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
if isfield(art.ox, 'kind') && any(strcmp(art.ox.kind, 'Group'))
    gk = find(strcmp(art.ox.kind, 'Group'));
    fprintf('    incl. %d group channel(s): %s\n', numel(gk), ...
        strjoin(unique(regexp(strjoin(art.ox.channel_names(gk).', ' '), ...
            'Grp\[[^\]]*\]', 'match'), 'stable'), ' '));
end
for c = 1:nc
    r = (art.ox.indxall.config == c);
    fprintf('    config %-5s: %6d rows, |dwdx| max %.3e\n', ...
        art.ox.config_names{c}, nnz(r), ...
        max(abs(art.ox.dwdxall(r, :)), [], 'all'));
end
