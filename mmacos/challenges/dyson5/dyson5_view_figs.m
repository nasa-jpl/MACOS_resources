function dyson5_view_figs(decks, outdir)
%DYSON5_VIEW_FIGS  Engine renders of the dyson5 decks for the record and the deck.
%   dyson5_view_figs()                      renders the R3 Dyson and the Offner
%   dyson5_view_figs({'dyson5_s3_r3'}, dir) a chosen set into dir
%   Each deck -> <tag>_view3d.png (macos.view_rx default view) and
%   <tag>_viewyz.png (the dispersion plane, az 90 el 0).  The bodies, rays
%   and labels are read back from the engine (macos.view_rx), never drawn
%   by hand; deck_dyson tiles these two per deck (demo_session/tile_views.py).
%   Model 128 is enough: the viewer cuts a sparse ring-and-spoke bundle.
here = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(decks), decks = {'dyson5_s3_r3', 'dyson5_s1_offner'}; end
if nargin < 2 || isempty(outdir), outdir = here; end
for k = 1:numel(decks)
    macos.init(128);
    macos.load_rx(fullfile(here, [decks{k} '.in']));
    macos.view_rx('save', fullfile(outdir, [decks{k} '_view3d.png']), 'visible', false, ...
        'nrings', 2, 'nspokes', 6, 'title', sprintf('%s: as traced by the engine', strrep(decks{k}, '_', ' ')));
    macos.view_rx('save', fullfile(outdir, [decks{k} '_viewyz.png']), 'visible', false, ...
        'nrings', 2, 'nspokes', 6, 'view', [90 0], 'title', sprintf('%s: dispersion plane (Y-Z)', strrep(decks{k}, '_', ' ')));
    fprintf('wrote %s_view3d.png, %s_viewyz.png\n', decks{k}, decks{k});
end
end
