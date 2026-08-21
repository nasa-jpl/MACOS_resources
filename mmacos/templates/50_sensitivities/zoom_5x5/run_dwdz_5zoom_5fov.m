% run_dwdz_5zoom_5fov.m -- multi-CONFIGURATION x multi-FIELD dw/dz (example).
% =====================================================================
%  The FIGURE rung of the configuration-axis family: segment-LOCAL
%  MonZernike figure modes, harvested per (zoom position, field point).
%  Same axis, same fixture and the same 5 x 5 grid as
%  run_dwdx_5zoom_5fov.m -- read that driver's header first; it carries
%  the fixture provenance and the LOAD-CASE warning that applies here
%  too (the +-1 arcmin wavefront is ~278 waves; these numbers are not an
%  optical result).
%
%  SCOPE.  A full-scope figure harvest on this deck -- every lMon-bearing
%  optic, eight modes, 25 blocks -- is a multi-day run.  The shipped
%  settings are a scoped DEMONSTRATION of the axis: three modes over the
%  elements named in ELTS.  Widen MODES / ELTS for a real budget, and
%  leave the resume directory in place so a long run survives a restart.
%
%  SETUP: run `mmacos_setup` once per MATLAB session first.
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end

% ===================  CONFIG -- EDIT FOR YOUR SYSTEM  ================
RX       = fullfile(here, 'jwst_ote_designc.in');
MODEL    = 512;
NGRIDPTS = 63;
STOP_ELT = 25;          % the FSM IS the pupil; the deck carries no ApStop=
CFG_ELT  = 25;          % the element the configuration axis steers
FOV      = 2.90888e-4;  % 1 arcmin half-field, 5-field set
TILT     = 1.45444e-4;  % 0.5 arcmin configuration tilt
MODES    = 4:6;         % MonZernike modes (scope knob)
ELTS     = [23 24 25];  % SM / TM / FSM ([] = every eligible optic)
% =====================================================================


% ---- preflight: is this deck FIGURABLE? -----------------------------
% MEASURED on the shipped fixture, 2026-08-20: jwst_ote_designc.in has
% NO FreeForm elements.  Its 19 segments are `Surface= Conic`, and the
% MonZernike channel builder targets FreeForm-typed elements
% (macos.channels.freeform_monzern_channels -> find_freeform_elts), so
% this rung harvests ZERO channels on it.  The deck is a rigid-body and
% prescription-parameter fixture (see run_dwdx_5zoom_5fov.m and
% run_dwdsurf_5zoom_5fov.m, both of which run on it in full); making it
% carry a figure means promoting the segment surface type -- a fixture
% change, not a driver one.
%
% TO RUN THIS RUNG TODAY: point RX at a deck whose segments are figured,
% e.g. ../run_dwdx_multi/e5hex1.in (FreeForm segments), and set CFG_ELT
% to an optic that deck actually has.  The configuration axis itself is
% gated on exactly that deck.
macos.init(MODEL);
m_ = macos.Session(MODEL);
m_.load_rx(RX);
ff = m_.find_freeform_elts();
if isempty(ff)
    error(['run_dwdz_5zoom_5fov: %s declares no FreeForm elements, so ' ...
        'the MonZernike figure channel has nothing to build.  Point RX ' ...
        'at a figured deck (see the note above this check).'], RX);
end

sched = table( ...
    ["z0"; "zUL";  "zUR";  "zLL";  "zLR"], ...
    [   0;  -TILT;  +TILT;  -TILT;  +TILT], ...
    [   0;  +TILT;  +TILT;  -TILT;  -TILT], ...
    'VariableNames', {'name', sprintf('%d.Rx', CFG_ELT), ...
                              sprintf('%d.Ry', CFG_ELT)});
cfgs = macos.design.configs_from_table(sched);

[~, rxstem] = fileparts(RX);
art = run_sensitivities(RX, 'fov_rad', FOV, 'channels', "dwdz", ...
    'configs', cfgs, 'resume_dir', string(fullfile(here, '_resume_dwdz')), ...
    'stop_elt', STOP_ELT, 'ngridpts', NGRIDPTS, 'model_size', MODEL, ...
    'zmodes_fig', MODES, 'elts', ELTS, 'out_dir', here, ...
    'name', ['dwdz_5zoom_5fov_' rxstem]);

% ---- per-configuration summary --------------------------------------
% The blocks are contiguous and indxall.config carries the index per
% row -- that is the supported way to address one configuration's block.
A  = art.oz.dwdzall;
nc = numel(cfgs);
nf = size(art.oz.field_table, 1);
fprintf('=== dw/dz: %d configurations x %d fields = %d blocks ===\n', ...
    nc, nf, nc * nf);
fprintf('    stacked Jacobian %d x %d over %d channels\n', ...
    size(A, 1), size(A, 2), numel(art.oz.channel_names));
for c = 1:nc
    r = (art.oz.indxall.config == c);
    fprintf('    config %-5s: %6d rows, |J| max %.3e\n', ...
        art.oz.config_names{c}, nnz(r), max(abs(A(r, :)), [], 'all'));
end
