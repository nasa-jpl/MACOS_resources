function f = dw_page_fig(fig_in, ttl)
%DW_PAGE_FIG  An invisible figure whose PRINTED size is FIG_IN inches.
%   f = DW_PAGE_FIG([W H], TTL) returns a hidden figure set up so that
%   print(f, ..., '-r140') writes exactly [W*140 H*140] pixels.
%
%   Why inches and a MANUAL PaperPosition: with the default
%   PaperPositionMode 'auto' the printed size is the figure's PIXEL size
%   divided by get(0,'ScreenPixelsPerInch') -- so the same script prints
%   a different page on a different screen, and a tall figure is silently
%   CLAMPED to the screen height first (measured on this box: a 1400x950
%   figure prints 2042x1386 at -r140, i.e. 14.58 x 9.90 in).  Sizing in
%   inches with PaperPositionMode 'manual' makes the page a stated
%   physical size on every platform, which is what a minimum panel size
%   in inches has to rest on.
%
%   See also: dw_page_layout, dw_page_axes.

arguments
    fig_in (1,2) double {mustBePositive}
    ttl (1,:) char = ''
end
f = figure('Visible', 'off', 'Units', 'inches', ...
           'Position', [0.5 0.5 fig_in(1) fig_in(2)], 'Color', 'w');
if ~isempty(ttl), set(f, 'Name', ttl); end
set(f, 'PaperUnits', 'inches', 'PaperPosition', [0 0 fig_in], ...
       'PaperSize', fig_in, 'PaperPositionMode', 'manual');
end
