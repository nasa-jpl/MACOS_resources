function out = pdi_layout_fig(varargin)
%PDI_LAYOUT_FIG  The point-diffraction (stepped-pinhole) gauge's layout, in
%   the deck-figure recipe (Dave 2026-09-12; pdi_vfig_util): drawn by the
%   ENGINE from the emitted deck with macos.view_rx, the fold plane seen
%   from above, passive bookkeeping planes hidden, elements named with
%   leader lines off the beam, the crowded node -- the mask seat inside its
%   sphere bracket -- as a cropped panel at full width.
%
%   The hardware is the TG96 test arm exactly as the Zernike sensor uses it
%   (source, collimator L1, the 7 deg beamsplitter, the 700 mm leg to the
%   96 mm DM, L2, the seat at the internal focus, field lens, camera); the
%   ONE difference is what sits in the seat -- a PINHOLE substrate with an
%   attenuating surround and a stepped phase, in place of the etched
%   dimple.  That is the whole point of the common-path form, so the figure
%   says it in the caption rather than drawing a different bench.
%
%   out = pdi_layout_fig()            writes pdi_layout.png (1800 px)
%   pdi_layout_fig('MODEL', 1024)     any pdi_params override (dev-res default)
%   Run from this dir:  matlab -batch "pdi_layout_fig; exit(0)"
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init')), run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m')); end
addpath(fullfile(exdir, '..', 'zwfs_dm96'));                % zwfs_params, zwfs_mask
cd(exdir);
P = pdi_params();  P.MODEL = 512;  P.NGRID = 65;  P.grid.N_G = 256;  P.grid.DX_G = 0.42;
for i = 1:2:numel(varargin), parts = strsplit(varargin{i}, '.');  P = setfield(P, parts{:}, varargin{i+1}); end %#ok<SFLD>
macos.init(P.MODEL);
macos.write_grid_file(P.grid.flat_file, zeros(P.grid.N_G));
bf = fieldnames(P.bench);  bargs = cell(1, 2*numel(bf));
for i = 1:numel(bf), bargs{2*i-1} = bf{i};  bargs{2*i} = P.bench.(bf{i}); end
G = macos.design.twyman_green(bargs{:}, 'ngridpts', P.NGRID, ...
    'to_grid_file', P.grid.flat_file, 'to_grid_n', P.grid.N_G, 'to_grid_dx', P.grid.DX_G);
G.bt.wavelen = P.LAM;
deck = 'pdi_layout.in';  G.bt.emit(deck);
macos.load_rx(deck);
E = G.bt.E;  nm = {E.name};
idx = @(s) find(strcmp(nm, s), 1);
passive = find(strcmp({E.element}, 'Reference'));           % seat + sphere bracket + Recomb
green = [0 0.62 0.10];

f = figure('Color', 'w', 'Position', [40 40 1800 1010], 'Visible', 'off');
tl = tiledlayout(f, 5, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
ax1 = nexttile(tl, [2 1]);  ax2 = nexttile(tl, [3 1]);
for ax = [ax1 ax2]
    macos.view_rx('ax', ax, 'ray_color', green, 'title', '', 'labels', false, 'hide', passive);
end
% ---- top: the whole train, from above the fold plane ----------------------
pdi_vfig_util('flat', ax1, ...
    'The point-diffraction gauge: the TG96 test arm, with a stepped PINHOLE substrate in the seat at the internal focus', 15);
axis(ax1, 'off');
L1 = E(idx('L1pow')).vpt;  BS = E(idx('BSrefl')).vpt;  TO = E(idx('TestOptic')).vpt;
CM = E(idx('Comptxfd')).vpt;  SEAT = E(G.T.iMASK).vpt;  CAM = E(G.T.iDET).vpt;
L2 = E(idx('L2pow')).vpt;  FL = E(idx('FLpow')).vpt;
pdi_vfig_util('label', ax1, { ...
    L1,   [0   130], 'collimator L1'; ...
    BS,   [90 -150], 'beamsplitter, 7 deg'; ...
    CM,   [-60 150], 'compensator'; ...
    TO,   [0  -170], '96 mm deformable mirror (retro)'; ...
    L2,   [-40 170], 'L2'; ...
    SEAT, [180 -130], 'pinhole substrate at the internal focus'}, 15);
% ---- bottom: the crowded node, cropped -----------------------------------
pdi_vfig_util('flat', ax2, sprintf(['The last %.0f mm: the seat between the near-field entrance and exit spheres, then the field lens and the camera at the pupil image.  ' ...
    'The pinhole is %.2f lam/D across (%.1f um); the surround transmits %s of the amplitude, and the substrate steps the pinhole''s phase'], ...
    norm(CAM - L2), P.pdi.DIA_LAMD, P.pdi.DIA_LAMD*P.LAM*P.bench.F2/(2*P.bench.R_TO_AP)*1e3, ...
    'a matched fraction'), 15);
pdi_vfig_util('frame', ax2, [SEAT CAM FL L2], [30 25 22 22]);
pdi_vfig_util('label', ax2, { ...
    SEAT, [-16 -13], 'pinhole substrate (the seat)'; ...
    FL,   [-6   12], 'field lens'; ...
    CAM,  [10    7], 'camera at the pupil image'}, 16);
print(f, 'pdi_layout.png', '-dpng', '-r130');  close(f);
fprintf('wrote pdi_layout.png\n');
out = struct('bench', G.bt, 'deck', deck, 'iMASK', G.T.iMASK, 'iDET', G.T.iDET);
macos.unload();
end
