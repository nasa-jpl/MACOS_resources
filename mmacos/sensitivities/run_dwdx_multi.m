% run_dwdx_multi.m -- multi-field dw/dx RIGID-BODY sensitivity (GENERIC).
% =====================================================================
%  Multi-field rigid-body (6-DOF: Rx Ry Rz Tx Ty Tz) wavefront-sensitivity
%  Jacobian for every actual optic, in canonical state-vector form:
%
%      wall = dwdxall * x + w0_stacked
%
%  TO RUN ON YOUR OWN SYSTEM: edit the CONFIG block ("YOUR .in FILE GOES
%  HERE") -- everything below it is generic.
%
%  NOTE (2026-07-19): this script is now a thin wrapper over the
%  sensitivity STAGE RUNNER design/runners/run_sensitivities.m (single
%  algorithm source; per-element pages land in <name>_pages/, plots
%  piston-removed).  The CONFIG interface is unchanged.
%
%  ELEMENT GROUPS (2026-08-23).  GROUPS declares RIGID-BODY groups --
%  sets of elements perturbed as ONE body by the engine's GPERTURB --
%  and each contributes 6 more columns, APPENDED AFTER the per-element
%  block in every field's block.  That is the sensitivity an assembly
%  actually has: a bonded lens cell, a mirror backplane, a camera.  The
%  members' responses partly CANCEL when they move together, which
%  summing their individual columns cannot reproduce -- so a per-element
%  budget can overstate an assembly badly (7.5x on the bundled demo
%  deck's lens cell in tilt; 23x on an 18-segment PM in piston).
%
%  It is [] here because a group is SYSTEM-SPECIFIC -- there is no
%  sensible default for an .in file this script has not seen.  The
%  commented line names the bundled demo deck's cell.  For a worked,
%  committed example with the numbers see
%  templates/50_sensitivities/run_dwdx_multi/ (lens cell) and
%  .../zoom_5x5/ (PM backplane), and run_dwdx_multi_zoom.m beside this
%  script for the multi-CONFIGURATION form.
%
%  Group channels carry NO element id -- out.iElt is 0, the value a
%  source channel also carries -- and out.kind is 'Group'.  Section on
%  kind, not on iElt.  Units are the same on both sides: OPD-per-metre
%  for translations, OPD-per-rad for rotations.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'design', 'runners'));
addpath(here);   % plot_* / group_exhibit live beside this script

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX     = '';            % <-- YOUR .in FILE GOES HERE (absolute path)
MODEL  = 128;           % model size (>= your aperture grid sampling)
NGRIDPTS = 63;          % ray-grid sampling override ([] = keep the .in value)
FOV    = 1e-4;          % half-field (rad) for the 4 corner field points
% Finite-difference step, scalar or (1,6) = [Rx Ry Rz Tx Ty Tz].
% Rotations rad, translations SI metres.  The vector form is the safer
% default: on both decks in this repo a 1e-8 m translation poke is too
% small to be converged (per-element translation columns land 2.5e-3
% away on e5hex1, 1.9e-04 on the zoom deck; 1e-6 and 1e-5 agree with
% each other to ~1e-5).  Rotations show no such drift.  Worth re-checking
% on YOUR deck -- the floor scales with your coordinate magnitudes.
DELTA  = [1e-8 1e-8 1e-8 1e-6 1e-6 1e-6];
DOFS   = (0:5).';       % 0=Rx 1=Ry 2=Rz 3=Tx 4=Ty 5=Tz  (subset allowed)
% Rigid-body element GROUPS: containers.Map name -> column vector of
% member element ids.  [] = none.  For the bundled demo deck the
% physically motivated group is the lens cell behind M2:
%   GROUPS = containers.Map('KeyType','char','ValueType','any');
%   GROUPS('LensCell') = [9; 10];
% 'groups_auto' (below) instead parses EltGrp= declarations out of the
% Rx itself, which is the right switch for a deck that carries them.
GROUPS = [];
GROUPS_AUTO = false;
%
%  Bundled demo deck, used when RX is empty.  EXPLICIT path -- the
%  runner used to reach for examples/<its own name>/, so moving the
%  asset directory broke it silently.  It is one CONFIG line now.
DEMO_RX = fullfile(here, '..', 'templates', '50_sensitivities', 'run_dwdx_multi', ...
                   'e5hex1.in');
% =====================================================================

if isempty(RX)
    RX = DEMO_RX;
    fprintf('[demo] RX not set -- using bundled example: %s\n', RX);
end
assert(isfile(RX), 'run_dwd:noDeck', ...
    'prescription not found: %s\n(set RX, or fix DEMO_RX in the CONFIG block)', RX);
[~, rxstem] = fileparts(RX);
name = ['dwdx_multi_' rxstem];
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdx", ...
    'ngridpts', NGRIDPTS, 'model_size', MODEL, 'delta_x', DELTA, ...
    'dofs', DOFS, 'groups', GROUPS, 'groups_auto', GROUPS_AUTO, ...
    'out_dir', here, 'name', name);
fprintf('=== dw/dx multi: %d channels x %d fields ===\n', ...
    numel(art.ox.channel_names), size(art.ox.field_table, 1));
if isfield(art.ox, 'kind') && any(strcmp(art.ox.kind, 'Group'))
    fprintf('    incl. %d group channel(s)\n', nnz(strcmp(art.ox.kind, 'Group')));
end

% Group-vs-member table appended to <name>_sens_report.txt (no-op when
% GROUPS is empty).  It is what turns "there are six more columns" into
% "the assembly is 7.5x less tilt-sensitive than its front surface".
group_exhibit(art.ox, GROUPS, fullfile(here, [name '_sens_report.txt']));
