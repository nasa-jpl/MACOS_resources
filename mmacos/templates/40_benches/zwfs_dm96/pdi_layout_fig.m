function pdi_layout_fig(varargin)
%PDI_LAYOUT_FIG  The bench sketch for the point-diffraction deck (deck-grade).
%   pdi_layout_fig()            builds the bench of pdi_params (the ZWFS test
%                               arm: same deck, the mask seat holds the pinhole
%                               plate) and writes pdi_layout.png at 260 dpi
%   pdi_layout_fig('MODEL',512) any zwfs_params override (dev-res default here)
%   The sketch is the Bench builder's own (G.bt.sketch), as zwfs_s1_figs draws
%   it; only the title, size and resolution differ.  Run from this dir.
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init')), run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m')); end
cd(exdir);
P = pdi_params();  P.MODEL = 512;  P.NGRID = 65;  P.grid.N_G = 256;  P.grid.DX_G = 0.42;
for i = 1:2:numel(varargin), parts = strsplit(varargin{i}, '.');  P = setfield(P, parts{:}, varargin{i+1}); end %#ok<SFLD>
macos.init(P.MODEL);
macos.write_grid_file(P.grid.flat_file, zeros(P.grid.N_G));
bf = fieldnames(P.bench);  bargs = cell(1, 2*numel(bf));
for i = 1:numel(bf), bargs{2*i-1} = bf{i};  bargs{2*i} = P.bench.(bf{i}); end
G = macos.design.twyman_green(bargs{:}, 'ngridpts', P.NGRID, ...
    'to_grid_file', P.grid.flat_file, 'to_grid_n', P.grid.N_G, 'to_grid_dx', P.grid.DX_G);
fs = G.bt.sketch('title', ['Point-diffraction train: source - splitter - 96 mm DM - focus (pinhole plate, seat 16)' ...
    ' - pupil image; the P/SRI reference arm is synthesized from the field at the pupil']);
set(fs, 'Position', [50 50 1600 640]);
set(findobj(fs, 'Type', 'axes'), 'XLim', [-20 1900], 'YLim', [-330 320], 'FontSize', 12);
exportgraphics(fs, 'pdi_layout.png', 'Resolution', 260);
close(fs);
fprintf('wrote %s\n', fullfile(exdir, 'pdi_layout.png'));
% the tail, zoomed: focusing lens to camera (the mask seat inside the sphere bracket, the relay lens)
ft = G.bt.sketch('title', 'The last 130 mm: mask seat (16) between the entrance (15) and exit (17) spheres, relay lens (18, 19), camera (20)');
set(ft, 'Position', [50 50 1600 640]);
xt = G.bt.E(G.T.iMASK).s;                          % station of the mask seat along the chief ray
axt = findobj(ft, 'Type', 'axes');
set(axt, 'XLim', [1700 1830], 'YLim', [115 200], 'FontSize', 12);
% the sketch places its labels for the full view: drop the text that now falls outside the window
for h = findobj(ft, 'Type', 'text').'
    q = get(h, 'Position');
    if q(1) < 1700 || q(1) > 1830 || q(2) < 115 || q(2) > 200, delete(h); else, set(h, 'FontSize', 18); end
end
set(axt, 'FontSize', 15);  set(findobj(ft, 'Type', 'line'), 'LineWidth', 2.5);
tt = get(axt, 'Title');  set(tt, 'FontSize', 16);
exportgraphics(ft, 'pdi_layout_tail.png', 'Resolution', 260);
close(ft);
fprintf('wrote %s (mask seat at s = %.2f mm)\n', fullfile(exdir, 'pdi_layout_tail.png'), xt);
end
