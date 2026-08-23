% run_dwdx_multi.m -- multi-field dw/dx RIGID-BODY sensitivity (example).
% =====================================================================
%  Thin driver over the general sensitivity stage runner
%  design/runners/run_sensitivities.m ('dwdx' channel only): rigid-body
%  6-DOF (Rx Ry Rz Tx Ty Tz) wavefront Jacobian for every optic, in
%  canonical state-vector form  wall = dwdxall * x + w0_stacked.
%  (Single source of truth -- the per-example runner copies retired
%  2026-07-19 per the runners doctrine.)
%
%  THE LENS CELL -- an ELEMENT GROUP, on in this example.
%  e5hex1.in's elements 9 and 10 are the two Refractor SURFACES of the
%  doublet behind M2.  A doublet is not mounted one surface at a time:
%  it is bonded in a CELL and the cell is what an assembly aligns, so
%  the six rigid-body columns a tolerancing engineer actually assigns
%  belong to the cell, not to either surface.  `GROUPS` below declares
%  it -- `'LensCell' -> [9; 10]` -- and run_sensitivities appends its 6
%  columns AFTER the per-element block, in every field's block (the
%  engine perturbs the members as one rigid body via GPERTURB, so the
%  intra-cell cancellation is captured directly rather than synthesized
%  from two individually-large per-surface columns).
%
%  WHY IT MATTERS, and the numbers this example exists to show: the two
%  surfaces partially COMPENSATE inside the cell, so a per-surface
%  column OVERSTATES the cell's alignment sensitivity.  The driver
%  measures cell-vs-surface for all six DOFs at the end of the run and
%  appends the table to <name>_sens_report.txt under "LensCell
%  exhibit"; read the committed report for the current numbers rather
%  than trusting a comment.  Every figure quoted below is FROM that
%  committed report (grep it); the step-size numbers further down are a
%  separate convergence diagnostic measured on this deck.  On the
%  shipped configuration:
%
%    TILT is where the compensation shows.  Cell Rx 7.2748e-05 against
%    surface 9's 5.4550e-04 -- the cell is 7.5x LESS tilt-sensitive
%    than its own front surface, because surface 9 (5.4550e-04) and
%    surface 10 (4.7309e-04) respond comparably and largely cancel when
%    the two tilt together.  Ry is the same story, ratio 0.1293.
%
%    DECENTER does NOT compensate on THIS deck, and the reason is worth
%    knowing before you generalize: element 10 is Surface= Conic with
%    KrElt = -1E+18, i.e. FLAT.  A flat refracting surface has no
%    lateral response at all -- its Tx column is 1.6890e-09, five
%    decades under surface 9's, i.e. numerically zero -- so e5hex1's
%    "doublet" is optically a plano-convex SINGLET and there is nothing
%    for the cell decenter to cancel against.  Cell Tx 4.6007e-04 vs
%    surface 9's 4.6007e-04, ratio 1.0000.  That agreement to five
%    digits is not a null result: it is the check that the group
%    channel is a genuine rigid-body motion of both members (a rigid
%    translation must equal the member sum, and the member sum here IS
%    surface 9).  The classic intra-cell decenter compensation needs
%    two POWERED surfaces; put one in your own Rx and the ratio drops.
%
%  UNITS: group and per-element columns share one convention --
%  OPD-per-metre for translations, OPD-per-rad for rotations -- so they
%  are directly comparable and one numeric DELTA is one physical poke
%  for either.  (GroupedRigidBodyChannel converts SI metres to the
%  BaseUnits prb_grp wants, exactly as macos.perturb does for the
%  per-element channel.  It did not always; see the channel's
%  do_perturb comment and tDwDxGroups/
%  test_scalar_delta_matches_the_split_step for what that cost.)
%
%  WHY DELTA IS A (1,6) VECTOR -- convergence, not units.  Rotations sit
%  at 1e-8 rad; translations at 1e-6 (1 um) because a 1e-8 m poke is
%  itself too small on this deck: the per-element translation columns
%  come out 2.5e-3 away from their converged values at 1e-8, while 1e-6
%  and 1e-5 agree with each other to ~1e-5.  Rotations show no such
%  drift, so only the translation entries move.
%
%  Group channels carry NO element id -- out.iElt is 0, the value a
%  source channel also carries -- and out.kind is 'Group'.  Section on
%  kind, not on iElt.  The per-element pages do that already and give
%  the group its own page, <name>_dwdx_grpLensCell_center.png.
%
%  SETUP: run `mmacos_setup` once per MATLAB session first.
%  Self-contained: ships e5hex1.in beside the script.  TO RUN ON YOUR
%  OWN SYSTEM, point RX at your .in and set GROUPS to your own cells
%  (or [] for none) -- everything else is generic.
%
%  Outputs (this directory): <name>_sens_report.txt + _sens.mat +
%  _opdall/_svspec/_dwdx_channels.png + per-element pages.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX     = fullfile(here, 'e5hex1.in');  % your .in goes here
MODEL  = 128;           % model size (>= your aperture grid sampling)
NGRIDPTS = 63;          % ray-grid override ([] = keep the .in value)
FOV    = 1e-4;          % half-field (rad) for the 4 corner field points
% Finite-difference step, (1,6) = [Rx Ry Rz Tx Ty Tz].  Rotations in
% rad, translations in SI metres, for BOTH the per-element and the
% group channels -- see "WHY DELTA IS A (1,6) VECTOR" above for why the
% translation entries are 1e-6 and not the 1e-8 the rotations use.
DELTA  = [1e-8 1e-8 1e-8 1e-6 1e-6 1e-6];
% Rigid-body element GROUPS: name -> column vector of member element
% ids.  [] = none.  Here: the two Refractor surfaces of the doublet
% behind M2, mounted and aligned as ONE cell.
GROUPS = containers.Map('KeyType', 'char', 'ValueType', 'any');
GROUPS('LensCell') = [9; 10];
% =====================================================================

[~, rxstem] = fileparts(RX);
name = ['dwdx_multi_' rxstem];
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdx", ...
    'ngridpts', NGRIDPTS, 'model_size', MODEL, 'delta_x', DELTA, ...
    'groups', GROUPS, 'out_dir', here, 'name', name);
fprintf('=== dw/dx multi: %d channels x %d fields ===\n', ...
    numel(art.ox.channel_names), size(art.ox.field_table, 1));

% ---- the LensCell exhibit -------------------------------------------
% Cell vs member surfaces, all six DOFs, appended to the report so the
% committed artifact carries the numbers this example is about.  The
% helper divides the group TRANSLATION columns by CBM so both sides are
% per-metre -- see its header for the units argument.
group_exhibit(art.ox, GROUPS, ...
    fullfile(here, [name '_sens_report.txt']));
