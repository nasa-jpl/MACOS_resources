function [OPDall, indxall] = config_canvas(canvases, tile_rc)
%MACOS.CONFIG_CANVAS  Lay per-configuration canvases out, and index them.
%   [OPDall, indxall] = macos.config_canvas(CANVASES) concatenates the
%   per-configuration field canvases in CANVASES (a 1xNc cell of
%   equal-size 2-D maps, each one configuration's own field-tiled
%   canvas) left to right, and returns the combined canvas with the
%   m2v-style index struct that addresses it.
%
%   [...] = macos.config_canvas(CANVASES, TILE_RC) instead places
%   configuration c at tile position TILE_RC(c,:) = [row col] (0-based)
%   of an outer grid, so the layout MIRRORS the field set's own tiling:
%   a five-state zoom schedule reads as its four corners and its centre,
%   each cell holding that state's whole field canvas, exactly as the
%   five field points read within one cell.  Empty outer cells stay
%   zero and cost no rows.
%
%   TWO ORDERINGS, DELIBERATELY DIFFERENT
%   -------------------------------------
%   The canvas is for READING; the row order is for the state-vector
%   form  wall = J*x + w0.  They are not the same walk, and this is the
%   function that keeps them apart:
%
%     * OPDall places each configuration where its GEOMETRY says.
%     * indxall.i/.j are built CONFIGURATION-MAJOR -- all of
%       configuration 1's rows (its fields stacked in the usual
%       per-canvas m2v order), then all of configuration 2's, and so on.
%
%   Vectorising the outer canvas directly would NOT do this: m2v walks
%   column-major, so any outer layout that varies down a column
%   interleaves the configurations' rows.  Building the index instead of
%   deriving it is what lets a configuration keep a contiguous block of
%   rows while its tile sits where the reader expects.
%
%   `macos.v2m` scatters by sub2ind and so is indifferent to row ORDER;
%   the round trip through this index is exact.  Pair with
%   `macos.m2v(OPDall, indxall)` for the values.
%
%   indxall carries the usual .i / .j / .size, plus
%     .config   Nrows x 1 configuration index per row (stored as
%               VALUES, so it survives the 'orient','xy' transpose)
%
%   With ONE canvas and no TILE_RC this is exactly
%   `macos.m2v(canvases{1})` -- which is what keeps the no-configurations
%   call byte-identical to the pre-configuration-axis one.
%
%   See also: macos.m2v, macos.v2m, macos.dw_dx_multi,
%             macos.design.configs_from_table.

arguments
    canvases cell {mustBeNonempty}
    tile_rc  double = []
end

nc = numel(canvases);
[nr, ncol] = size(canvases{1});
for c = 2:nc
    assert(isequal(size(canvases{c}), [nr, ncol]), ...
        'macos:config_canvas:size', ...
        'canvas %d is %s, expected %s', c, ...
        mat2str(size(canvases{c})), mat2str([nr ncol]));
end

if isempty(tile_rc)
    tile_rc = [zeros(nc, 1), (0:nc-1).'];      % one row, left to right
else
    assert(isequal(size(tile_rc), [nc 2]), 'macos:config_canvas:tiles', ...
        'tile_rc must be %d x 2 (0-based [row col] per configuration)', nc);
    assert(all(tile_rc(:) >= 0 & tile_rc(:) == fix(tile_rc(:))), ...
        'macos:config_canvas:tiles', ...
        'tile_rc entries must be non-negative integers');
end

grid_r = max(tile_rc(:, 1)) + 1;
grid_c = max(tile_rc(:, 2)) + 1;
OPDall = zeros(grid_r * nr, grid_c * ncol);
for c = 1:nc
    r0 = tile_rc(c, 1) * nr;
    c0 = tile_rc(c, 2) * ncol;
    OPDall(r0+1:r0+nr, c0+1:c0+ncol) = canvases{c};
end

% CONFIGURATION-MAJOR index: each configuration's own canvas is
% vectorised in the ordinary way, then shifted into place.  Order is by
% configuration, NOT by a walk of the assembled canvas.
ii = cell(1, nc);  jj = cell(1, nc);  cc = cell(1, nc);
for c = 1:nc
    [~, ix] = macos.m2v(canvases{c});
    ii{c} = ix.i(:) + tile_rc(c, 1) * nr;
    jj{c} = ix.j(:) + tile_rc(c, 2) * ncol;
    cc{c} = repmat(c, numel(ix.i), 1);
end
indxall = struct();
indxall.i      = vertcat(ii{:});
indxall.j      = vertcat(jj{:});
indxall.size   = size(OPDall);
indxall.config = vertcat(cc{:});
end
