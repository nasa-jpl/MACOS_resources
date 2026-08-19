% run_dwdgrid_multi.m -- multi-field dw/d(grid-data) on e5hex1_grid.in via mmacos.
%
% Multi-field companion to run_dwdgrid.m (single field).  Mirrors
% run_dwdz.m's structure, but on the GRID-DATA channel (the GMI pgrid
% leg): each grid "poke" -- an influence-function map ADDED to a
% surface's grid data IN PLACE -- has a wavefront sensitivity, and
% dw_dgrid_multi tiles that sensitivity over a field set into the
% canonical state-vector Jacobian:
%
%     wall = dwdgall * x + w0_stacked
%
% Default demo: 6 Zernike-shaped pokes per hex segment at 5 field points
% (center + 4 corners at +/- 100 urad).  Uses e5hex1_grid.in (per-segment
% grid frames) and the same aperture-confined influence basis as the
% single-field run_dwdgrid.m, so each poke is a clean segment Zernike.
%
% Inputs the user must set per Rx:
%   field_x_rad / field_y_rad  direction-cosine offsets for corner FPs
%                              (additive on the Rx's nominal ChfRayDir).
%
% Outputs land in this directory:
%   dwdgall_e5hex1.mat   dwdgall + w0_stacked + indxall + channel_names +
%                        field_table + iElt + map_idx + ...
%   dwdgrid_multi_OPDall_e5hex1.png   tiled nominal-OPD field canvas

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
addpath(fullfile(here, '..', '..', '..', 'src'));   % +macos
rx = fullfile(here, 'e5hex1_grid.in');   % pData (grid frame) = RptElt per segment

% ====================  KNOBS  ========================================
MODES = [4 5 6 7 8 9];  % poke modes = MACOS ANSI indices (= MonZernModes in
                        % the .in): 4=astig45 5=defocus 6=astig0 7=trefoil-y
                        % 8=coma-y 9=coma-x.
DELTA = 1e-6;           % FD step (grid-map amplitude, BaseUnits)
MODEL = 256;            % >= e5hex1_grid's nGridMat(256) so mGridMat=256 (else the
                        % grid-add overflows the slot -- cross-segment corruption)
FOV   = 1e-4;           % half-FoV (rad) for the 4 corner field points
% =====================================================================

fprintf('=== e5hex1: multi-field dw/d(grid), 6 pokes/segment, 5 fields ===\n');
m = macos.Session(MODEL);  m.load_rx(rx);

% Aperture-confined influence basis (same recipe as run_dwdgrid.m): each
% Zernike lives inside the segment aperture (lMon), not the oversized grid.
g    = macos.find_grid_elts();
N    = double(mmacos('elt_srf_grid_size', g(1), 1));
txt  = fileread(rx);
% str2num (not str2double) so Fortran D-exponents parse, e.g. 1.0D-02
lMon = str2num(regexp(txt, '(?<=lMon=)\s*[\d.eEdD+-]+',      'match', 'once'));  %#ok<ST2NM>
gdx  = str2num(regexp(txt, '(?<=GridSrfdx=)\s*[\d.eEdD+-]+', 'match', 'once'));  %#ok<ST2NM>
ap_frac = lMon / (((N-1)/2) * gdx);
fprintf('grid-bearing elements: %s\n', mat2str(g(:).'));
fprintf('lMon=%.1f  GridSrfdx=%.3f  -> aperture = %.0f%% of grid half-width\n', ...
        lMon, gdx, 100*ap_frac);
infl = macos.zernike_grid_basis(N, MODES, ap_frac);

out = macos.dw_dgrid_multi(m, rx, ...
    'field_x_rad', FOV, ...
    'field_y_rad', FOV, ...
    'influence', infl, ...
    'delta', DELTA);

%% -- per-field nominal WFE (proof the per-field FEX removed field tilt) --
%  Each field's nominal (unpoked) wavefront is now referenced to its OWN
%  chief ray (FEX inside dw_dgrid_multi), so the gross field tilt is gone.
%  Without the reset the corners ran ~265 waves of uncompensated tilt.
LAM_MM = 1.0e-3;   % Wavelen=1.0E-03 mm = 1 um
fprintf('\nper-field nominal RMS WFE (exit-pupil-referenced):\n');
for k = 1:numel(out.field_names)
    Wk = out.per_field_w_nom_2d{k};  wk = Wk(Wk~=0);
    fprintf('  %-4s  RMS = %.4e mm = %8.3f waves\n', ...
        char(out.field_names{k}), std(wk), std(wk)/LAM_MM);
end

%% -- VERIFY: FD-consistency at the center field ---------------------
%  A manual central difference of one poke must reproduce its center-field
%  Jacobian column exactly (convention-free).  The center field (k=1) starts
%  from the freshly-loaded nominal state, then per-field FEX; to match it we
%  reload, FEX the nominal chief ray (mode 1) ONCE, then poke WITHOUT re-FEX
%  (the reference is fixed, so the poke's own tilt is retained -- exactly what
%  the supervisor did).  FEX is mildly state-dependent (it adjusts nGridPts),
%  so the fresh reload is what makes this reproduce the center block exactly.
ctr = find(out.field_table(:,1)==0 & out.field_table(:,2)==0, 1);
e1  = g(1);  M1 = infl(:,:,1);  wf = out.wf_elt;
[~, cindx] = macos.m2v(out.per_field_w_nom_2d{ctr});
col1 = find(out.iElt==e1 & out.map_idx==1, 1);
m.load_rx(rx);  macos.fex(1);        m.trace(wf);  W0 = m.opd(); %#ok<NASGU>
macos.elt_grid_add(e1, +DELTA*M1);   m.modify();  m.trace(wf);  Wp = m.opd();
macos.elt_grid_add(e1, -2*DELTA*M1); m.modify();  m.trace(wf);  Wm = m.opd();
macos.elt_grid_add(e1, +DELTA*M1);   m.modify();                    % restore
dW_manual = macos.m2v((Wp - Wm)/(2*DELTA), cindx);
col_ctr   = out.per_field_dwdg{ctr}(:, col1);
rel = norm(dW_manual - col_ctr) / max(norm(col_ctr), eps);
fprintf('FD-consistency (manual vs center-field Jacobian, Elt %d poke 1): rel = %.3e  [%s]\n', ...
        e1, rel, tern_(rel < 1e-9, 'OK', '** FAIL'));

%% -- FIGURE: tiled nominal-OPD field canvas -------------------------
f1 = figure('Name','dw/d(grid) multi -- OPDall', 'Position',[40 40 700 700]);
imagesc(out.OPDall);  axis image off;  colormap(parula);  colorbar;
title(sprintf('e5hex1 OPDall: %d fields, %d grid channels', ...
      numel(out.field_names), numel(out.channel_names)), 'Interpreter','none');
print(f1, fullfile(here,'dwdgrid_multi_OPDall_e5hex1.png'), '-dpng','-r140');

%% -- SAVE: canonical state-vector layout (matches run_dwdz.m) -------
field_table   = out.field_table;
field_names   = out.field_names;
chfraydir_nom = out.chfraydir_nom;
dwdgall       = out.dwdgall;
dwdxall       = out.dwdxall;
w0_stacked    = out.w0_stacked;
indxall       = out.indxall;
channel_names = out.channel_names;
iElt          = out.iElt;
map_idx       = out.map_idx;
delta         = out.delta;
method        = out.method;
wf_elt        = out.wf_elt;
rx            = out.rx_path;
opdall_shape  = size(out.OPDall);
model_size    = MODEL;

out_path = fullfile(here, 'dwdgall_e5hex1.mat');
save(out_path, ...
    'dwdgall', 'dwdxall', 'w0_stacked', 'indxall', 'channel_names', ...
    'iElt', 'map_idx', 'field_table', 'field_names', 'chfraydir_nom', ...
    'delta', 'method', 'wf_elt', 'rx', 'opdall_shape', 'model_size');
fprintf('wrote %s\n', out_path);
fprintf('\n=== dw_dgrid_multi: %d channels x %d fields; FD-consistency %s ===\n', ...
        numel(out.channel_names), numel(out.field_names), ...
        tern_(rel<1e-9,'PASS','FAIL'));

% ---- helper --------------------------------------------------------
function s = tern_(c,a,b), if c, s=a; else, s=b; end, end
