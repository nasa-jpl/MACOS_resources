function pg = dw_page_layout(block, map_rc, tile_rc, opts)
%DW_PAGE_LAYOUT  Size-FIRST page plan for the dw_d* channel plotters.
%   PG = DW_PAGE_LAYOUT(BLOCK, MAP_RC, TILE_RC) returns the pages needed to
%   draw NUMEL(BLOCK) OPD panels at or above a MINIMUM PANEL SIZE -- the
%   size is fixed first and the panels-per-page follow from it, never the
%   other way round (Dave 2026-09-10: "make the plots large enough to be
%   interpretable, even with large numbers of segments -- this will mean
%   many more plots and pages").
%
%   BLOCK    n x 1 block id per panel (numeric, string or cellstr): the
%            ELEMENT / group / source the panel belongs to.  Pagination
%            never splits a block across pages unless the block alone
%            exceeds one page -- an element's DOF / mode / parameter set
%            side by side is the readable unit.
%   MAP_RC   [rows cols] of the drawn map, in pixels: sets the panel ASPECT.
%   TILE_RC  [tiles_r tiles_c] of the FIELD CANVAS inside one panel
%            (default [1 1] = a single-field map).
%
%   NAME-VALUE
%     'panel_in'     minimum size (inches at print resolution) of the
%                    SMALLER dimension of one panel, i.e. of ONE OPD map
%                    (default 3.5, Dave's number).  It is a FLOOR: a page
%                    with room to spare grows its panels to fill the
%                    envelope, so a two-panel page is not drawn small.
%     'tile_in'      minimum size of ONE FIELD TILE inside a multi-field
%                    canvas panel (default 1.2).  A canvas panel is
%                    TILE_RC * tile_in, grown if that leaves it under
%                    'panel_in' -- so a single-field map (TILE_RC [1 1])
%                    is exactly 'panel_in' and the two rules are one rule.
%     'page_in'      page envelope, inches (default [16 9], 16:9) -- the
%                    size a page is built to when it can be.
%     'page_max_in'  how far a page may GROW to keep one block whole
%                    (default [32 20]).  A block that does not fit the
%                    envelope gets its own, larger page rather than being
%                    split: an element's Kr and Kc, or its six DOFs, are
%                    read against each other.  Only a block too big for
%                    even this is split, onto _p02 ...  A panel is never
%                    shrunk below its floor to make something fit.
%     'max_per_page' hard cap on panels per page (default 24) on top of
%                    the size-derived grid.
%     'cbar'         reserve room for a per-panel colorbar (default true).
%     'row_per_block' each block starts a new ROW (default false).  The
%                    all-channels overview uses this to keep today's
%                    "rows = element, cols = that element's channels"
%                    reading; a block wider than the page wraps WITHIN
%                    its own rows rather than sharing a row with the
%                    next element.
%
%   PG(k) fields
%     idx        panel indices on page k, in order
%     slot       the grid slot (row-major, 1-based) each one is drawn in:
%                with 'row_per_block' a block starts a new row, so the
%                slots are not simply 1..n
%     nr, nc     the tile grid drawn on the page
%     fig_in     [W H] figure/paper size, inches
%     panel_in   [w h] the drawn map box, inches (>= the floors)
%     cell_in    [w h] one grid cell (map + title + colorbar + gutter)
%     pad        struct with .title .cbar .gap .marg .sg (inches)
%     blocks     the block ids appearing on the page
%     part, nparts  1/1 unless one block had to be split over pages
%
%   Pair with DW_PAGE_AXES, which turns PG(k) plus a panel slot into an
%   axes Position, so the map really is 'panel_in' on the printed page --
%   MATLAB's default SUBPLOT margins give away about 30% of every cell,
%   which is how a 2042x1386 page ended up drawing its maps at 4.3 in.
%
%   See also: plot_dw_channels, plot_dw_per_element, plot_dw_index.

arguments
    block
    map_rc  (1,2) double {mustBePositive}
    tile_rc (1,2) double {mustBePositive} = [1 1]
    opts.panel_in     (1,1) double {mustBePositive} = 3.5
    opts.tile_in      (1,1) double {mustBePositive} = 1.2
    opts.page_in      (1,2) double {mustBePositive} = [16 9]
    opts.page_max_in  (1,2) double {mustBePositive} = [32 20]
    opts.max_per_page (1,1) double {mustBePositive} = 24
    opts.cbar         (1,1) logical = true
    opts.row_per_block (1,1) logical = false
end

% ---- 1. the panel: tiles x tile_in, grown to the single-map floor -----
h = opts.tile_in * tile_rc(1);
w = h * (map_rc(2) / map_rc(1));          % true drawn aspect, not the
                                          % tile count (they agree for
                                          % square tiles; the map wins)
s = opts.panel_in / min(w, h);
if s > 1, w = w * s;  h = h * s;  end

pad = struct('title', 0.34, 'cbar', 0.62 * double(opts.cbar), ...
             'gap', 0.20, 'marg', 0.18, 'sg', 0.52);
cellsz = [w + pad.cbar + pad.gap, h + pad.title + pad.gap];

% ---- 2. how many fit the envelope, and how many if the page grows -----
[nc_env, nr_env] = local_fit(opts.page_in,     cellsz, pad);
[nc_grw, nr_grw] = local_fit(opts.page_max_in, cellsz, pad);
cap_env = min(opts.max_per_page, nc_env * nr_env);
cap_grw = min(opts.max_per_page, nc_grw * nr_grw);

% ---- 3. pack whole blocks; grow before splitting ---------------------
key = local_keys(block);
[ub, ~, bi] = unique(key, 'stable');
P = struct('idx', {}, 'slot', {}, 'nc', {}, 'blocks', {}, ...
           'part', {}, 'nparts', {});
cur = local_new();
for b = 1:numel(ub)
    cols = find(bi == b);
    nb   = numel(cols);
    if nb <= cap_env
        rows_b = ceil(nb / nc_env);
        if opts.row_per_block
            fits = (cur.rows + rows_b <= nr_env) && ...
                   (numel(cur.idx) + nb <= opts.max_per_page);
        else
            fits = (numel(cur.idx) + nb <= cap_env);
        end
        if ~isempty(cur.idx) && ~fits
            P = local_flush(P, cur, opts.row_per_block);
            cur = local_new();
        end
        % under 'row_per_block' a block starts a NEW ROW (rows = element,
        % cols = that element's channels, the reading the overview has
        % always had); otherwise panels simply flow on
        cur.idx  = [cur.idx,  cols(:).'];
        cur.row  = [cur.row,  cur.rows + ceil((1:nb) / nc_env)];
        cur.col  = [cur.col,  mod((1:nb) - 1, nc_env) + 1];
        cur.rows = cur.rows + rows_b;
        cur.wide = max(cur.wide, min(nb, nc_env));
        cur.blocks = [cur.blocks, ub(b)];
        continue
    end
    % the block does not fit the envelope: flush, then give it its OWN
    % page(s) -- grown to page_max_in before any split
    P = local_flush(P, cur, opts.row_per_block);
    cur = local_new();
    if nb <= cap_grw
        nc = local_pick(nb, nc_grw, nr_grw, cellsz);
        P = local_flush(P, local_solo(cols, nc, ub(b)), false);
    else
        nc = local_pick(min(nb, cap_grw), nc_grw, nr_grw, cellsz);
        np = ceil(nb / cap_grw);
        for q = 1:np
            sl = cols((q-1)*cap_grw + 1 : min(q*cap_grw, nb));
            P = local_flush(P, local_solo(sl, nc, ub(b)), false);
            P(end).part = q;  P(end).nparts = np;
        end
    end
end
P = local_flush(P, cur, opts.row_per_block);

% ---- 4. GROW to fill the envelope, then materialise ------------------
% The floor is a MINIMUM, not the size to draw at: a page carrying two
% panels on a 16 x 9 sheet should use the sheet.  Scale the map box (the
% title / colorbar gutters are fixed text, so they do not scale) by the
% same factor on EVERY page of the set -- taken from the fullest page --
% so panels are one size throughout and no page passes the envelope.
% Grow only: a page that already exceeds the envelope to keep a block
% whole is left alone.
gk = Inf;
for k = 1:numel(P)
    nc = max(1, P(k).nc);
    nr = max(1, ceil(max(P(k).slot) / nc));
    usable = opts.page_in - 2*pad.marg - [0 pad.sg];
    gk = min(gk, min((usable(1) - nc*(pad.cbar + pad.gap)) / (nc*w), ...
                     (usable(2) - nr*(pad.title + pad.gap)) / (nr*h)));
end
if isfinite(gk) && gk > 1
    w = w * gk;  h = h * gk;
    cellsz = [w + pad.cbar + pad.gap, h + pad.title + pad.gap];
end

pg = struct('idx', {}, 'slot', {}, 'nr', {}, 'nc', {}, 'fig_in', {}, ...
            'panel_in', {}, 'cell_in', {}, 'pad', {}, 'blocks', {}, ...
            'part', {}, 'nparts', {});
for k = 1:numel(P)
    nc = max(1, P(k).nc);
    nr = max(1, ceil(max(P(k).slot) / nc));
    fig = [nc*cellsz(1) + 2*pad.marg, nr*cellsz(2) + 2*pad.marg + pad.sg];
    pg(end+1) = struct('idx', P(k).idx, 'slot', P(k).slot, 'nr', nr, ...
        'nc', nc, 'fig_in', fig, 'panel_in', [w h], 'cell_in', cellsz, ...
        'pad', pad, 'blocks', {P(k).blocks}, 'part', P(k).part, ...
        'nparts', P(k).nparts);  %#ok<AGROW>
end
end

% ---------------------------------------------------------------------
function [nc, nr] = local_fit(page, cellsz, pad)
usable = page - 2*pad.marg - [0 pad.sg];
nc = max(1, floor(usable(1) / cellsz(1)));
nr = max(1, floor(usable(2) / cellsz(2)));
end

% ---------------------------------------------------------------------
function nc = local_pick(nb, nc_max, nr_max, cellsz)
%LOCAL_PICK  Column count for a block on its own page: the grid whose
%   page aspect is closest to the 16:9 the envelope is written in.
best = Inf;  nc = min(nb, nc_max);
for c = 1:min(nb, nc_max)
    r = ceil(nb / c);
    if r > nr_max, continue; end
    ar = (c*cellsz(1)) / (r*cellsz(2));
    d  = abs(log(ar / (16/9)));
    if d < best, best = d;  nc = c;  end
end
end

% ---------------------------------------------------------------------
function cur = local_new()
cur = struct('idx', [], 'row', [], 'col', [], 'rows', 0, 'wide', 0, ...
             'blocks', {{}});
end

% ---------------------------------------------------------------------
function cur = local_solo(cols, nc, blk)
%LOCAL_SOLO  One block alone on its own (grown) page: a dense nc grid.
n = numel(cols);
cur = struct('idx', cols(:).', 'row', ceil((1:n)/nc), ...
             'col', mod((1:n)-1, nc) + 1, 'rows', ceil(n/nc), ...
             'wide', min(n, nc), 'blocks', {blk});
end

% ---------------------------------------------------------------------
function P = local_flush(P, cur, row_per_block)
%LOCAL_FLUSH  Freeze one page.  Each panel's SLOT is only knowable here:
%   the column count is the widest block actually ON the page, not the
%   envelope's, and a slot computed against the wrong width lands the
%   panel in the wrong row.
if isempty(cur.idx), return; end
if row_per_block
    nc   = max(1, cur.wide);
    slot = (cur.row - 1)*nc + cur.col;
else
    nc   = max(1, min(cur.wide, numel(cur.idx)));
    slot = 1:numel(cur.idx);          % dense: panels flow
end
P(end+1) = struct('idx', cur.idx, 'slot', slot, 'nc', nc, ...
    'blocks', {cur.blocks}, 'part', 1, 'nparts', 1);
end

% ---------------------------------------------------------------------
function key = local_keys(block)
if isnumeric(block)
    key = arrayfun(@(v) sprintf('%g', v), block(:), 'UniformOutput', false);
elseif isstring(block)
    key = cellstr(block(:));
elseif ischar(block)
    key = {block};
else
    key = cellfun(@(v) char(string(v)), block(:), 'UniformOutput', false);
end
end
