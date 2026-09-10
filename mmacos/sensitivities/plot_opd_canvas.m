function fig = plot_opd_canvas(out, ttl, here, pngname, opts)
%PLOT_OPD_CANVAS  Field-tiled nominal OPD canvas (OUT.OPDall), pupils masked.
%   Each tile is one field point's nominal (unpoked) wavefront -- and with
%   a CONFIGURATION axis, one outer tile per configuration holding that
%   configuration's whole field canvas.  With the per-field exit-pupil
%   reset (reset_xp=true, the default for dw_dz/dw_dsurf/dw_dgrid), the
%   gross field tilt is removed so the tiles show the real residual
%   aberration rather than a giant tilt ramp.
%
%   SIZE (2026-09-10).  The canvas is drawn at least 'tile_in' inches per
%   FIELD TILE, capped by 'page_max_in' -- the historical 760 px box is
%   the floor, and it was unbounded the other way: a big 'grid','NxM'
%   field set (or 5 configurations x 5 fields, a 9 x 9 tile canvas) drew
%   each field smaller and smaller in the same box.  Sized in inches with
%   a manual PaperPosition (dw_page_fig), so the printed page no longer
%   depends on the screen: the same 760 px figure printed 1109 px here
%   (96 px/in) and 1478 px on the box the committed baselines came from
%   (72 px/in).
%
%   See also: plot_dw_channels, plot_dw_per_element, dw_page_layout,
%             macos.dw_dgrid_multi.

arguments
    out (1,1) struct
    ttl (1,:) char
    here (1,:) char = ''
    pngname (1,:) char = ''
    opts.tile_in (1,1) double {mustBePositive} = 1.2
    opts.page_max_in (1,2) double {mustBePositive} = [32 20]
end
C = out.OPDall;
C(C == 0) = NaN;                       % mask outside the pupils
% Follow the canvas aspect when it is WIDE -- a multi-configuration
% harvest lays the configurations out along columns, and a fixed square
% figure would strand a 5x-wide strip in a sea of white.  A square or
% tall canvas keeps the historical 760 px (= 7.92 in) box exactly.
ar    = size(C, 2) / size(C, 1);
tiles = dw_canvas_tiles(out);
h = max(760/96, tiles(1) * opts.tile_in);
w = h * ar;                            % the DRAWN aspect, exactly -- the
                                       % legacy box capped width at 1900 px
                                       % and kept the height, letterboxing
                                       % a wide canvas inside white
if w > opts.page_max_in(1), w = opts.page_max_in(1);  h = w / ar;  end
if h > opts.page_max_in(2), h = opts.page_max_in(2);  w = h * ar;  end
fig = dw_page_fig([w h], ttl);
ax  = axes('Units', 'normalized', 'Position', [0.03 0.03 0.94 0.90]);
hh  = imagesc(ax, C);  set(hh, 'AlphaData', ~isnan(C));
axis(ax, 'image');  axis(ax, 'off');  set(ax, 'Color', 'w');
colormap(ax, jet);  colorbar(ax);
title(ax, ttl, 'Interpreter', 'none');
if ~isempty(pngname)
    if isempty(here), here = pwd; end
    print(fig, fullfile(here, pngname), '-dpng', '-r140');
    close(fig);
    fprintf('wrote %s\n', fullfile(here, pngname));
end
end
