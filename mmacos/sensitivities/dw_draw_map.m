function him = dw_draw_map(ax, M, ttl, cbrect, fs)
%DW_DRAW_MAP  One dW map in one panel, in the pinned display convention.
%   Exact zeros are NOT drawn (white -- outside the pupils), piston is
%   removed, the panel is autoscaled, colormap jet, no colour limits
%   (Dave 2026-09-09/10: the deck shows the tool's own output, rendered
%   the way the user renders it).  Do not add clim or a bespoke mask here.
%
%   CBRECT (from DW_PAGE_AXES) places a colorbar in the reserved gutter
%   without letting it shrink the map; [] = no colorbar.
%
%   See also: dw_page_axes, plot_dw_channels, plot_dw_per_element.

arguments
    ax (1,1) matlab.graphics.axis.Axes
    M double
    ttl (1,:) char = ''
    cbrect double = []
    fs (1,1) double = 9
end
M(M == 0) = NaN;                       % mask outside the pupils
M = M - mean(M(:), 'omitnan');         % piston removed (display
                                       % convention, Dave 2026-07-19)
him = imagesc(ax, M);
set(him, 'AlphaData', ~isnan(M));
axis(ax, 'image');  axis(ax, 'off');  set(ax, 'Color', 'w');
colormap(ax, jet);
if ~isempty(ttl)
    title(ax, ttl, 'FontSize', fs, 'Interpreter', 'none');
end
if ~isempty(cbrect)
    pos = get(ax, 'Position');
    cb  = colorbar(ax);
    set(ax, 'Position', pos);          % colorbar() shrinks its axes
    set(cb, 'Units', 'normalized', 'Position', cbrect, 'FontSize', fs-1);
end
end
