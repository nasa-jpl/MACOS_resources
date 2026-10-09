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
    % the axes are the viewer's (bodies, rays, labels from the engine); only the LIMITS are tightened to the
    % drawn data -- 'axis equal' alone pads the short axes to the figure's aspect (a 2-m box around a 0.6-m train)
    views = {'view3d', [-35 18], 'as traced by the engine'; 'viewyz', [90 0], 'dispersion plane (Y-Z)'};
    for v = 1:size(views, 1)
        fig = figure('Visible', 'off', 'Position', [50 50 980 640]);  ax = axes('Parent', fig);
        macos.view_rx('ax', ax, 'nrings', 2, 'nspokes', 6, 'view', views{v, 2}, ...
            'title', sprintf('%s: %s', strrep(decks{k}, '_', ' '), views{v, 3}));
        axis(ax, 'tight');
        print(fig, fullfile(outdir, [decks{k} '_' views{v, 1} '.png']), '-dpng', '-r150');  close(fig);
    end
    fprintf('wrote %s_view3d.png, %s_viewyz.png\n', decks{k}, decks{k});
end
end
