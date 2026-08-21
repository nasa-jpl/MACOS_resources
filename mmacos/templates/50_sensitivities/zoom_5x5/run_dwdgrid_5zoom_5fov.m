% run_dwdgrid_5zoom_5fov.m -- multi-CONFIGURATION x multi-FIELD dw/dgrid.
% =====================================================================
%  The SEGMENT-GRID rung of the configuration-axis family, and the one
%  with a step of its own: before harvesting, the prescription is
%  GRID-AUGMENTED in each segment's clocked Mon frame
%  (macos.design.grid_augment_rx -- it writes the grid channel nGridMat /
%  GridFile / GridSrfdx / pData..zData into each Segment block), and then
%  poked with a per-segment Gram-Schmidt Zernike influence basis
%  (macos.segment_grid_basis).  run_sensitivities('channels','dwdgrid')
%  does both.  The result is the wavefront Jacobian dW/d(grid-mode
%  amplitude) per (zoom position, field point) -- 5 x 5 = 25 blocks.
%
%  Same axis, fixture and 5 x 5 grid as run_dwdx_5zoom_5fov.m -- read that
%  driver's header first for the fixture provenance, the tiled canvas, and
%  the LOAD-CASE warning (the +-1' fields are ~278 waves; the numbers are
%  for the machinery, not a design result).
%
%  THE FIXTURE CARRIES ZERO-AMPLITUDE FIGURE CHANNELS.  jwst_ote_designc's
%  19 segments are Surface= FreeForm (an optically inert promotion of the
%  original Conic; deck header, PLAN_CONFIGURATIONS.md departure #6, gated
%  in tRunSensitivities).  FreeForm is a grid-bearing surface type, so
%  after augmentation the segments carry a live grid the basis can poke.
%
%  TWO THINGS THIS DRIVER MUST GET RIGHT ON THIS DECK:
%    * PM_REF_ELT is a SEGMENT (element 4), not the default near-pupil
%      Reference.  This deck's primary IS the segmented set (no dedicated
%      Reference plane), so segment_grid_basis takes its ray-history-union
%      footprint path -- which needs a segment as the trace anchor.  With
%      the default (element 1, a Spider) the per-segment footprints come
%      out degenerate (measured mask ~8 px vs ~130) and the pokes do not
%      localize.
%    * the grid FRAME is the clocked Mon frame (grid_augment_rx copies
%      pMon->pData), and the grid SPAN is (nGridMat-1)*GridSrfdx = the
%      Aperture (span_frac 1.0); a too-small span figures only near the
%      segment centre.  Both are handled by the augmenter here.
%
%  CONFIGURATIONS STAY POSE-ONLY.  A configuration may carry only the v1
%  pose setters (perturb / set_elt_vpt / psi / rpt / csys); the grid STATE
%  this rung pokes is not in the pose snapshot, so a configuration that
%  wrote a grid would restore silently wrong.  The supervisor rejects one
%  loudly at validation time.
%
%  SCOPE / COST.  This harvests MODES grid modes on ALL 19 segments -- at
%  the shipped 4:6 that is 57 grid channels, 25 blocks, the same size as
%  the dw/dz run.  Grid modes are the scope knob (the supervisor default
%  is 4:9); a wide modal basis over 19 segments is the largest of the four
%  rungs, so widen MODES deliberately and let the resume directory carry
%  the run.  To harvest only some segments, pass 'elts' (it scopes the
%  dw/dgrid channel set; the basis is still built for every segment).
%
%  RESUMABLE: per-configuration checkpoints in _resume_dwdgrid/, pruned on
%  success.  Delete it by hand to force a cold run.
%
%  Outputs (this directory): <name>_sens_report.txt + _sens.mat +
%  <name>_grid.in (the augmented Rx + flat grid file) + _opdall /
%  _svspec / _svspec_configs / _dwdgrid_channels.png + per-element pages.
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
PM_REF   = 4;           % footprint trace anchor: a SEGMENT (see header)
FOV      = 2.90888e-4;  % half-field (rad) = 1 arcmin, 5-field set
TILT     = 1.45444e-4;  % configuration tilt (rad) = 0.5 arcmin
MODES    = 4:6;         % grid influence-basis modes per segment (scope knob)
NG       = 64;          % augmented grid size (nGridMat)
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
name = ['dwdgrid_5zoom_5fov_' rxstem];
art  = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdgrid", ...
    'configs', cfgs, 'resume_dir', string(fullfile(here, '_resume_dwdgrid')), ...
    'stop_elt', STOP_ELT, 'ngridpts', NGRIDPTS, 'model_size', MODEL, ...
    'zmodes_grid', MODES, 'ng', NG, 'pm_ref_elt', PM_REF, ...
    'per_element', "center", 'out_dir', here, 'name', name);

% ---- per-configuration summary --------------------------------------
% The blocks are contiguous and indxall.config carries the index per row
% -- that is the supported way to address one configuration's block.
A  = art.og.dwdgall;
nc = numel(cfgs);
nf = size(art.og.field_table, 1);
fprintf('=== dw/dgrid: %d configurations x %d fields = %d blocks ===\n', ...
    nc, nf, nc * nf);
fprintf('    stacked Jacobian %d x %d over %d channels\n', ...
    size(A, 1), size(A, 2), numel(art.og.channel_names));
for c = 1:nc
    r = (art.og.indxall.config == c);
    fprintf('    config %-5s: %6d rows, |dwdg| max %.3e\n', ...
        art.og.config_names{c}, nnz(r), max(abs(A(r, :)), [], 'all'));
end
