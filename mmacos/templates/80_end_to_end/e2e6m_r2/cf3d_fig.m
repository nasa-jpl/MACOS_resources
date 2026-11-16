function cf3d_fig()
%CF3D_FIG  The restart-ladder figure: dark-zone floor vs round, with the
%   linear-achievable trajectory.  Reads cf3d_run.mat; writes cf3d_dig.png.
    here = fileparts(mfilename('fullpath'));
    C = load(fullfile(here, 'cf3d_run.mat'));
    h = C.hist;
    r  = [h.round];  c = [h.c_end];  la = [h.la_floor];  st = [h.stroke_nm];
    fig = figure('Position',[60 60 900 560], 'Color','w', 'Visible','off');
    ax = axes(fig);  hold(ax,'on');  grid(ax,'on');
    semilogy(ax, [0 r], [h(1).c_start c], 'o-', 'Color',[0.12 0.31 0.47], ...
             'LineWidth', 2, 'MarkerFaceColor',[0.12 0.31 0.47], ...
             'DisplayName','achieved floor (measured contrast)');
    semilogy(ax, r, la, 's--', 'Color',[0.75 0.45 0.10], 'LineWidth',1.4, ...
             'DisplayName','linear-achievable at 50 nm strokes');
    yline(ax, 5e-11, ':', 'HWO-class 5\times10^{-11}', 'Color',[0.4 0.4 0.4], ...
          'HandleVisibility','off');
    xline(ax, 8.5, ':', 'extension', 'Color',[0.6 0.6 0.6], ...
          'LabelVerticalAlignment','bottom', 'HandleVisibility','off');
    set(ax, 'YScale','log');
    xlabel(ax, 'restart round (each = relinearize about the dug state, then EFC)');
    ylabel(ax, 'dark-zone mean contrast (3–15 \lambda/D)');
    title(ax, sprintf(['CF3d: the EFC restart ladder, d = 1.10 m apodized-Lyot ' ...
        '-- %.2e \\rightarrow %.2e in %d rounds'], h(1).c_start, c(end), r(end)));
    legend(ax, 'Location','northeast');
    text(ax, r(end), c(end)*2.2, sprintf('%.2e @ %.0f nm strokes  ', ...
         c(end), st(end)), 'FontSize', 9, 'HorizontalAlignment','right');
    exportgraphics(fig, fullfile(here, 'cf3d_dig.png'), 'Resolution', 150);
    fprintf('wrote cf3d_dig.png\n');
end
