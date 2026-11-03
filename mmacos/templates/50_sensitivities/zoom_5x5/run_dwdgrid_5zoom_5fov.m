% run_dwdgrid_5zoom_5fov.m -- multi-CONFIGURATION x multi-FIELD dw/dgrid.
% =====================================================================
%  The SEGMENT-GRID rung of the configuration-axis family, and the one
%  with a step of its own: the prescription is GRID-AUGMENTED in each
%  segment's clocked Mon frame (macos.design.grid_augment_rx), then poked
%  with an influence basis, per (zoom position, field point) -- 5 x 5 = 25
%  blocks of dW/d(grid-mode amplitude).
%
%  SEPARATE BASES PER OPTIC (Dave 2026-08-21).  The 18 real segments share
%  ONE bespoke per-segment Gram-Schmidt Zernike basis
%  (macos.segment_grid_basis), while the SM (elt 23) and TM (elt 24) --
%  monolithic mirrors, not part of the segment tiling -- each get their
%  OWN full-aperture Zernike basis (macos.zernike_grid_basis on their own
%  grid).  grid_channels keys a per-element basis by iElt, so the two are
%  merged into ONE influence struct and handed to run_sensitivities via
%  'influence' (the verbatim path).  This is the mixed-basis pattern of
%  run_dwdgrid_multi_multisegbasis, extended to non-segment optics.
%
%  Same axis, fixture and 5 x 5 grid as run_dwdx_5zoom_5fov.m -- read that
%  driver's header first for the fixture provenance, the tiled canvas, and
%  the LOAD-CASE warning (the +-1' fields are ~278 waves; the numbers are
%  for the machinery, not a design result).
%
%  THE FIXTURE CARRIES ZERO-AMPLITUDE FIGURE CHANNELS.  Segments 5-22 and
%  SM/TM are Surface= FreeForm (an optically inert promotion of the
%  original Conic; deck header, PLAN_CONFIGURATIONS.md departure #6, gated
%  in tRunSensitivities).  FreeForm is a grid-bearing type; the SM/TM
%  grids are written into the deck (footprint-sized), the segment grids are
%  added here by grid_augment_rx.  Element 4 (CenterSegment) stays Conic
%  (virtual, obscured) so it carries no grid and is not poked.
%
%  TWO THINGS THIS DRIVER MUST GET RIGHT ON THIS DECK:
%    * PM_REF is a SEGMENT (element 5), not the default near-pupil
%      Reference.  This deck's primary IS the segmented set (no dedicated
%      Reference plane), so segment_grid_basis takes its ray-history-union
%      footprint path -- which needs a segment as the trace anchor.  With
%      the default (element 1, a Spider) the per-segment footprints come
%      out degenerate (measured mask ~8 px vs ~130) and the pokes do not
%      localize.
%    * the grid FRAME is the clocked Mon frame (grid_augment_rx copies
%      pMon->pData for segments; the promoter did the same for SM/TM), and
%      the grid SPAN is (nGridMat-1)*GridSrfdx; a too-small span figures
%      only near the optic centre.
%
%  CONFIGURATIONS STAY POSE-ONLY.  A configuration may carry only the v1
%  pose setters (perturb / set_elt_vpt / psi / rpt / csys); the grid STATE
%  this rung pokes is not in the pose snapshot, so a configuration that
%  wrote a grid would restore silently wrong.  The supervisor rejects one
%  loudly at validation time.
%
%  SCOPE / COST.  MODES grid modes on the 18 segments + SM + TM (20
%  optics) -- at the shipped 4:6 that is 60 grid channels, 25 blocks.
%  Grid modes are the scope knob (the supervisor default is 4:9); widen
%  MODES deliberately and let the resume directory carry a long run.  A
%  dead/obscured optic is dropped number-free after the harvest by
%  flag_zero_norm_channels.
%
%  RESUMABLE: per-configuration checkpoints in _resume_dwdgrid/, pruned on
%  success.  Delete it by hand to force a cold run.
%
%  Outputs (this directory): <name>.mat is FLAT (dwdgrid / indxall /
%  w0_stacked / channel_names / config_* / sgb at the TOP LEVEL, not in an
%  'og' struct) + <name>_grid.in (the augmented Rx + flat grid file) +
%  <name>_sens_report.txt + _opdall / _svspec / _svspec_configs /
%  _dwdgrid_channels.png + per-element pages.
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
PM_REF   = 5;           % footprint trace anchor: a real SEGMENT (elt 4 is
                        % the virtual Conic centre; 5 = first real segment)
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

% ---- build the MIXED influence basis (Dave: separate bases per optic) ---
% One shape SHARED by the 18 real segments, and a SEPARATE full-aperture
% basis for the SM and the TM (they are monolithic mirrors, not part of
% the segment Voronoi tiling).  macos.channels.grid_channels keys a
% per-element basis by iElt, so a single struct whose .seg array covers
% segments + SM + TM is consumed directly.  Assembled here rather than in
% run_sensitivities (whose internal segment_grid_basis covers only the
% segments), then handed to it via 'influence' (the verbatim path).
m = macos.Session(MODEL);
% (a) grid-augment the SEGMENTS in their clocked Mon frames (the deck's
%     SM/TM already carry grids; grid_augment_rx touches only Segments).
rxg = fullfile(here, [name '_grid.in']);
macos.design.grid_augment_rx(RX, rxg, 'ng', NG, 'span_frac', 1.0);
% (b) the shared per-segment basis.  segment_grid_basis defaults its
%     seg_elts to the grid-bearing Voronoi centres = the Element= Segment
%     blocks; SM/TM are Reflectors, so they are NOT swept here.
sgb = macos.segment_grid_basis(m, rxg, 'pm_ref_elt', PM_REF, ...
    'modes', MODES, 'orthogonalize', true);
% (c) a full-aperture basis for SM (23) and TM (24), each on its OWN grid,
%     appended as .seg entries (grid_channels keys a per-element basis by
%     iElt -- see run_dwdgrid_multi_multisegbasis).  Match the seg-entry
%     field set exactly so the struct array concatenates.
fn = fieldnames(sgb.seg);
for e = [23 24]
    ns = double(macos.get_elt_grid_size(e));           % SM/TM grid size (NG)
    B  = macos.zernike_grid_basis(ns, MODES);          % full-aperture modes
    s  = sgb.seg(1);                                    % clone the field layout
    for q = 1:numel(fn), s.(fn{q}) = []; end
    s.iElt = e;  s.B = B;  s.mask = true(ns);  s.mask_px = ns*ns;
    if isfield(s,'R_seg'), s.R_seg = (ns-1)/2; end
    sgb.seg(end+1) = s;
end

art = run_sensitivities(rxg, 'fov_rad', FOV, 'channels', "dwdgrid", ...
    'configs', cfgs, 'resume_dir', string(fullfile(here, '_resume_dwdgrid')), ...
    'stop_elt', STOP_ELT, 'ngridpts', NGRIDPTS, 'model_size', MODEL, ...
    'influence', sgb, 'per_element', "center", 'out_dir', here, 'name', name);

% ---- drop dead (obscured) grid optics, number-free ------------------
dead = flag_zero_norm_channels(art.og);
art.og = drop_channels(art.og, dead);

% ---- flat, channel-named .mat (dwdgrid at top level; keep sgb) ------
save_dw_flat(art.og, fullfile(here, [name '.mat']), ...
    'name', 'dwdgrid', 'model_size', MODEL, ...
    'extra', struct('sgb', art.og.sgb));

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
