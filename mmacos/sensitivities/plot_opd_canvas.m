function fig = plot_opd_canvas(out, ttl, here, pngname)
%PLOT_OPD_CANVAS  Field-tiled nominal OPD canvas (OUT.OPDall), pupils masked.
%   Each tile is one field point's nominal (unpoked) wavefront.  With the
%   per-field exit-pupil reset (reset_xp=true, the default for
%   dw_dz/dw_dsurf/dw_dgrid), the gross field tilt is removed so the tiles
%   show the real residual aberration rather than a giant tilt ramp.
%
%   See also: plot_dw_channels, macos.dw_dgrid_multi.
C = out.OPDall;
C(C == 0) = NaN;                       % mask outside the pupils
fig = figure('Name', ttl, 'Position', [40 40 760 760]);
h = imagesc(C);  set(h, 'AlphaData', ~isnan(C));
axis image off;  set(gca, 'Color', 'w');
colormap(parula);  colorbar;
title(ttl, 'Interpreter', 'none');
if nargin >= 4 && ~isempty(pngname)
    if nargin < 3 || isempty(here), here = pwd; end
    print(fig, fullfile(here, pngname), '-dpng', '-r140');
    fprintf('wrote %s\n', fullfile(here, pngname));
end
end
