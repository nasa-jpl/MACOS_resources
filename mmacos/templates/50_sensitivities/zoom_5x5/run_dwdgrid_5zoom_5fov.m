% run_dwdgrid_5zoom_5fov.m -- multi-CONFIGURATION x multi-FIELD dw/dgrid.
% =====================================================================
%  The SEGMENT-GRID rung of the configuration-axis family, and the one
%  with a step of its own: the prescription is first GRID-AUGMENTED in
%  each segment's clocked Mon frame (macos.design.grid_augment_rx, which
%  REPLACES any stale parent-frame grid lines -- the e5-corpus trap),
%  then poked with a per-segment Gram-Schmidt Zernike influence basis.
%  The deck carries 22 Segment blocks, so this rung is by far the
%  largest of the four.
%
%  Same axis, fixture and 5 x 5 grid as run_dwdx_5zoom_5fov.m -- read
%  that driver's header first for the fixture provenance and the
%  LOAD-CASE warning.
%
%  The CONFIGURATIONS STAY POSE-ONLY here.  A configuration may only
%  carry the v1 pose setters (perturb / set_elt_vpt / psi / rpt / csys);
%  the grid STATE this rung pokes is not part of the pose snapshot, so a
%  configuration that wrote a grid would restore silently wrong.  The
%  supervisor rejects one loudly at validation time.
%
%  SCOPE.  22 segments x a full modal basis x 25 blocks is a multi-day
%  harvest.  The shipped settings scope it to MODES over SEGS as a
%  DEMONSTRATION; widen for a real budget and let the resume directory
%  carry the run across restarts.
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
MODES    = 4:6;         % influence-basis modes per segment (scope knob)
NG       = 64;          % augmented grid size
SEGS     = 1:4;         % segment subset ([] = every segment)
% =====================================================================


% ---- preflight: is this deck GRID-BEARING? --------------------------
% MEASURED on the shipped fixture, 2026-08-20: after
% macos.design.grid_augment_rx the deck still reports NO grid surfaces.
% The augmenter writes the grid CHANNEL (nGridMat / GridFile /
% GridSrfdx / pData..zData) into each Segment block, but it does not
% promote `Surface=`, and the shipped fixture's 19 segments are
% `Surface= Conic` -- a type whose intersection routine never consumes a
% grid.  find_grid_elts is empty before and after augmentation, so this
% rung harvests ZERO channels on it.  Making the fixture carry a grid
% figure means promoting the segment surface type (Conic -> FreeForm,
% which would also unlock the dwdz rung) -- a fixture change, not a
% driver one.
%
% TO RUN THIS RUNG TODAY: point RX at a deck whose segments are figured,
% e.g. ../run_dwdx_multi/e5hex1.in (FreeForm segments).  The
% configuration axis itself is gated on exactly that deck.
macos.init(MODEL);
m_ = macos.Session(MODEL);
m_.load_rx(RX);
if isempty(macos.find_grid_elts()) && isempty(m_.find_freeform_elts())
    error(['run_dwdgrid_5zoom_5fov: %s has no grid-bearing surface ' ...
        'type, so grid augmentation cannot give it a figure channel.  ' ...
        'Point RX at a figured deck (see the note above this check).'], RX);
end

sched = table( ...
    ["z0"; "zUL";  "zUR";  "zLL";  "zLR"], ...
    [   0;  -TILT;  +TILT;  -TILT;  +TILT], ...
    [   0;  +TILT;  +TILT;  -TILT;  -TILT], ...
    'VariableNames', {'name', sprintf('%d.Rx', CFG_ELT), ...
                              sprintf('%d.Ry', CFG_ELT)});
cfgs = macos.design.configs_from_table(sched);

[~, rxstem] = fileparts(RX);
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdgrid", ...
    'configs', cfgs, 'resume_dir', string(fullfile(here, '_resume_dwdgrid')), ...
    'stop_elt', STOP_ELT, 'ngridpts', NGRIDPTS, 'model_size', MODEL, ...
    'zmodes_grid', MODES, 'ng', NG, 'elts', SEGS, 'out_dir', here, ...
    'name', ['dwdgrid_5zoom_5fov_' rxstem]);

% ---- per-configuration summary --------------------------------------
% The blocks are contiguous and indxall.config carries the index per
% row -- that is the supported way to address one configuration's block.
A  = art.og.dwdgall;
nc = numel(cfgs);
nf = size(art.og.field_table, 1);
fprintf('=== dw/dgrid: %d configurations x %d fields = %d blocks ===\n', ...
    nc, nf, nc * nf);
fprintf('    stacked Jacobian %d x %d over %d channels\n', ...
    size(A, 1), size(A, 2), numel(art.og.channel_names));
for c = 1:nc
    r = (art.og.indxall.config == c);
    fprintf('    config %-5s: %6d rows, |J| max %.3e\n', ...
        art.og.config_names{c}, nnz(r), max(abs(A(r, :)), [], 'all'));
end
