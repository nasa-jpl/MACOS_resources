function [ax, cbrect] = dw_page_axes(p, slot)
%DW_PAGE_AXES  Axes for one panel slot of a DW_PAGE_LAYOUT page.
%   [ax, cbrect] = DW_PAGE_AXES(P, SLOT) creates the axes for the SLOT-th panel
%   (1-based, row-major from the top left) of page plan P in the current
%   figure, positioned so the MAP BOX is exactly P.panel_in inches --
%   room for the panel title above it and for a colorbar to its right is
%   reserved by the plan, not taken out of the map.  CBRECT is the
%   normalized rectangle reserved for that panel's colorbar (empty when
%   the plan reserved none): a bare COLORBAR call SHRINKS its axes, which
%   would silently undo the sizing this whole file exists to guarantee,
%   so the caller restores the axes Position and places the bar here.
%
%   See also: dw_page_layout, dw_page_fig.

arguments
    p (1,1) struct
    slot (1,1) double {mustBeInteger, mustBePositive}
end
r = ceil(slot / p.nc);
c = slot - (r-1)*p.nc;
W = p.fig_in(1);  H = p.fig_in(2);
x0 = p.pad.marg + (c-1)*p.cell_in(1);
ytop = H - p.pad.marg - p.pad.sg - (r-1)*p.cell_in(2);
y0 = ytop - p.pad.title - p.panel_in(2);
ax = axes('Units', 'normalized', 'Position', ...
    [x0/W, y0/H, p.panel_in(1)/W, p.panel_in(2)/H]);
if p.pad.cbar > 0
    cbrect = [(x0 + p.panel_in(1) + 0.10)/W, y0/H, 0.16/W, p.panel_in(2)/H];
else
    cbrect = [];
end
end
