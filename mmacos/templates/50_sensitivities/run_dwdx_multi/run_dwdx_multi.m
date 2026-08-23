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
%  committed report (grep it) except the step-size sweep in the UNITS
%  TRAP, which is a separate diagnostic measured on this deck.  On the
%  shipped configuration:
%
%    TILT is where the compensation shows.  Cell Rx 7.2745e-05 against
%    surface 9's 5.4550e-04 -- the cell is 7.5x LESS tilt-sensitive
%    than its own front surface, because surface 9 (5.4550e-04) and
%    surface 10 (4.7310e-04) respond comparably and largely cancel when
%    the two tilt together.  Ry is the same story, ratio 0.1293.
%
%    DECENTER does NOT compensate on THIS deck, and the reason is worth
%    knowing before you generalize: element 10 is Surface= Conic with
%    KrElt = -1E+18, i.e. FLAT.  A flat refracting surface has no
%    lateral response at all -- its Tx column is 1.6562e-09, five
%    decades under surface 9's, i.e. numerically zero -- so e5hex1's
%    "doublet" is optically a plano-convex SINGLET and there is nothing
%    for the cell decenter to cancel against.  Cell Tx 4.6010e-04 vs
%    surface 9's 4.6007e-04, ratio 1.0001.  That agreement to four
%    digits is not a null result: it is the check that the group
%    channel is a genuine rigid-body motion of both members (a rigid
%    translation must equal the member sum, and the member sum here IS
%    surface 9).  The classic intra-cell decenter compensation needs
%    two POWERED surfaces; put one in your own Rx and the ratio drops.
%
%  UNITS TRAP -- read before comparing any group column with a
%  per-element one, and before choosing DELTA.  A group TRANSLATION
%  column is OPD per BASE UNIT (the engine's prb_grp takes BaseUnits),
%  while a per-element translation column is OPD per METRE.  On this
%  millimetre deck they are 1000x apart.  Two consequences:
%    (1) The exhibit divides the group columns by CBM so both sides are
%        per-metre.  Do the same in your own post-processing
%        (macos.dw_dx_multi's help documents the convention).
%    (2) A SCALAR 'delta' pokes the group 1/CBM times SMALLER than the
%        elements -- 10 pm here against 10 nm -- and at that amplitude
%        the group columns are finite-difference NOISE, not signal.
%        Measured on this deck, group column / frame-resolved member sum
%        went 1.0000 (delta 1e-5) -> 1.0005 (1e-6) -> 1.012 (1e-7) ->
%        1.657 (1e-8): the error grows as the step shrinks, which is the
%        signature of a step that is too small, not of physics.  So
%        DELTA below is the (1,6) form -- rotations 1e-8 rad (rad on
%        BOTH sides, no mismatch), translations 1e-6, giving 1 um at the
%        elements and 1 nm at the cell.  Both are in the converged
%        regime: the per-element translation columns agree with a 1e-5
%        step to ~1e-5, while the old scalar-1e-8 element step is itself
%        the outlier by 2.5e-3.
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
% Finite-difference step, (1,6) = [Rx Ry Rz Tx Ty Tz].  Rotations rad
% on both sides; translations SI metres for the per-element channels
% and BASE UNITS for the group channels -- see the UNITS TRAP above for
% why this is the vector form and not a scalar.
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
% Cell vs single surface, all six DOFs, with the group TRANSLATION
% columns divided by CBM so both sides are per-metre and the
% BaseUnit/metre convention difference cannot flatter either one.
% Written into the report so the committed artifact carries the numbers
% this example is about.
lens_cell_exhibit(art.ox, GROUPS, fullfile(here, [name '_sens_report.txt']));


function lens_cell_exhibit(ox, groups, report_path)
if ~isa(groups, 'containers.Map') || groups.Count == 0, return; end
if ~isfield(ox, 'kind') || ~any(strcmp(ox.kind, 'Group')), return; end
cbm = 1;
if isfield(ox, 'cbm') && ox.cbm > 0, cbm = ox.cbm; end
LAB = {'Rx','Ry','Rz','Tx','Ty','Tz'};
% column RMS over the rows every channel reached
rmsn = @(A) sqrt(mean(A(all(isfinite(A), 2), :).^2, 1));

fid = fopen(report_path, 'a');
if fid < 0, return; end
closer = onCleanup(@() fclose(fid));
say = @(varargin) fprintf(1, varargin{:}) + fprintf(fid, varargin{:});

gnames = keys(groups);
for gi = 1:numel(gnames)
    nm  = gnames{gi};
    mem = double(groups(nm));  mem = mem(:);
    tag = sprintf('Grp[%s]', nm);
    gc  = find(strncmp(ox.channel_names, tag, numel(tag)) ...
               & strcmp(ox.kind(:), 'Group'));
    ec  = cell(numel(mem), 1);
    for q = 1:numel(mem)
        ec{q} = find(startsWith(ox.channel_names, ...
                     sprintf('Elt %d ', mem(q))));
    end
    if numel(gc) < 6 || any(cellfun(@(v) numel(v) < 6, ec)), continue; end

    say('\n[%s exhibit] the CELL against its member SURFACES\n', nm);
    say(['    column RMS of dW/d(DOF): rotations in OPD-metres per rad, ' ...
         'translations\n    in OPD-metres per METRE.  The group ' ...
         'translation columns are divided by\n    CBM = %.6g -- ' ...
         'prb_grp reads BaseUnits where macos.perturb reads SI\n' ...
         '    metres, so a raw comparison would be %.0fx out.\n'], ...
        cbm, 1/cbm);
    hdr = sprintf('%-22s', 'channel');
    for d = 1:6, hdr = [hdr sprintf('%12s', LAB{d})]; end %#ok<AGROW>
    say('    %s\n', hdr);

    gv = zeros(1, 6);
    for d = 1:6
        sc = 1;  if d > 3, sc = 1/cbm; end     % dof 3..5 are Tx Ty Tz
        gv(d) = rmsn(ox.dwdxall(:, gc(d))) * sc;
    end
    say('    %-22s%s\n', [tag ' (cell)'], sprintf('%12.4e', gv));
    for q = 1:numel(mem)
        ev = rmsn(ox.dwdxall(:, ec{q}));
        say('    %-22s%s\n', sprintf('Elt %d (surface)', mem(q)), ...
            sprintf('%12.4e', ev));
        if q == 1
            rat = gv ./ max(ev, realmin);
            say('    %-22s%s\n', '  cell / this surface', ...
                sprintf('%12.4f', rat));
        end
    end
    say(['    (a ratio below 1 is intra-cell COMPENSATION -- the ' ...
         'members'' responses\n     partly cancel when they move as ' ...
         'one body; a ratio at 1 means the other\n     member ' ...
         'contributes nothing to that DOF)\n']);
end
end
