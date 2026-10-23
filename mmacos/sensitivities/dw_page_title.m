function h = dw_page_title(fig_in, lines, fs)
%DW_PAGE_TITLE  Page header at a FIXED distance from the top of the page.
%   h = DW_PAGE_TITLE([W H], LINES) writes LINES (char or cellstr) across
%   the top of the current figure, whose printed size is [W H] inches.
%
%   Why not SGTITLE: it places its text at a FRACTION of the figure
%   height, so on a tall page it drifts DOWN into the first row of
%   panels -- measured on a 5.7 x 20 in index sheet, the title landed
%   1.2 in from the top over thumbnails that start at 1.0 in, and on a
%   22 x 12 in channels page it sat at the same height as the panel
%   titles.  The layout reserves a fixed header band (pad.sg), so the
%   title has to be placed in inches, not in per cent.
%
%   See also: dw_page_fig, dw_page_layout, dw_page_axes.

arguments
    fig_in (1,2) double {mustBePositive}
    lines
    fs (1,1) double = 11
end
if ischar(lines) || isstring(lines), lines = cellstr(lines); end
top = 0.10;                                   % inches of clear margin
band = min(0.34 + 0.22*numel(lines), fig_in(2)/3);
h = annotation('textbox', ...
    [0.01, 1 - (top + band)/fig_in(2), 0.98, band/fig_in(2)], ...
    'String', lines, 'Interpreter', 'none', 'FontSize', fs, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
    'EdgeColor', 'none', 'FitBoxToText', 'off');
end
