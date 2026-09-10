function pages = plot_dw_channels(out, ttl, here, pngname, opts)
%PLOT_DW_CHANNELS  Every channel's MULTI-FIELD dW, at a readable size.
%   PAGES = PLOT_DW_CHANNELS(OUT, TTL, HERE, PNGNAME) renders, for every
%   column of the multi-field Jacobian OUT.dwdxall, that channel's
%   wavefront sensitivity reconstructed onto the tiled field canvas
%   (macos.v2m on OUT.indxall) -- so each panel shows the sensitivity at
%   ALL field points at once -- and PAGINATES so that no panel is ever
%   smaller than the floor in 'panel_in' / 'tile_in'.
%
%   SIZE FIRST, COUNT SECOND (Dave 2026-09-10).  One sheet of every
%   channel is unreadable past a handful of optics: the jwst zoom deck
%   gives 42 dwdsurf channels (21 optics x Kr/Kc) and 138 dwdx channels,
%   which used to print as ~30 px specks.  The panel size is now fixed
%   first and the panels per page follow (dw_page_layout).
%
%   PAGINATION IS ON ELEMENT BOUNDARIES.  An element's DOF / mode /
%   parameter block is the readable unit and is never split across pages
%   unless the block alone exceeds one page (then it continues onto
%   _p02).  Blocks are sectioned by KIND -- element, group, source --
%   through dw_block_keys, so a group's columns get their own block
%   rather than piling onto the source row.
%
%   FILES.  PNGNAME in HERE is ALWAYS written, so no README, script or
%   committed artifact ever loses its reference:
%     one page  -> it IS that page, exactly as before;
%     paginated -> it is the INDEX contact sheet (the dense one-sheet
%                  view this function used to produce), each thumbnail
%                  labelled with the page it is drawn on, and the
%                  full-size pages are <stem>_p01.png, _p02.png ... in
%                  'page_dir' (default HERE; the runner points it at
%                  <name>_pages/, where the numerous per-element pages
%                  already live).
%
%   Generic across every dw_d*_multi supervisor: dw_dx_multi,
%   dw_dz_zernike_multi, dw_dsurf_multi and dw_dgrid_multi all expose the
%   same canonical fields (dwdxall / indxall / channel_names).
%
%   NAME-VALUE
%     'panel_in','tile_in','page_in','page_max_in','max_per_page'
%                  see DW_PAGE_LAYOUT
%     'page_dir'   where the _pNN pages go (default HERE)
%     'index'      write the index contact sheet (default true when the
%                  set paginates; never for a single page)
%     'index_only' compute the page plan and write ONLY the index sheet
%                  (default false).  The index depends on the plan, not
%                  on the drawing, so this rebuilds it in seconds after a
%                  cosmetic change instead of redrawing every page.
%     'fontsize'   panel title font (default 9)
%
%   PAGES is the manifest of what was written (file / kind / mode /
%   blocks / channels / page index), which the runner collects into the
%   harvest-level index.  HERE defaults to pwd and PNGNAME to
%   dw_channels.png -- this function always WRITES; there is no
%   draw-but-do-not-print mode (the pages are printed and closed one at a
%   time, which is what lets a 138-channel harvest run in batch).
%
%   See also: plot_dw_per_element, plot_dw_index, dw_page_layout,
%             macos.dw_dgrid_multi, macos.v2m.

arguments
    out (1,1) struct
    ttl (1,:) char
    here (1,:) char = ''
    pngname (1,:) char = 'dw_channels.png'
    opts.panel_in (1,1) double = 3.5
    opts.tile_in (1,1) double = 1.2
    opts.page_in (1,2) double = [16 9]
    opts.page_max_in (1,2) double = [32 20]
    opts.max_per_page (1,1) double = 24
    opts.page_dir (1,:) char = ''
    opts.index = []
    opts.index_only (1,1) logical = false
    opts.fontsize (1,1) double = 9
end
if isempty(here), here = pwd; end
if isempty(pngname), pngname = 'dw_channels.png'; end
pdir = opts.page_dir;  if isempty(pdir), pdir = here; end

J     = out.dwdxall;          % canonical multi-field Jacobian (alias in all 4)
indx  = out.indxall;
names = out.channel_names;
nchan = size(J, 2);
[key, ~, lbl] = dw_block_keys(out);

plan = dw_page_layout(key, indx.size, dw_canvas_tiles(out), ...
    'panel_in', opts.panel_in, 'tile_in', opts.tile_in, ...
    'page_in', opts.page_in, 'page_max_in', opts.page_max_in, ...
    'max_per_page', opts.max_per_page, ...
    'cbar', false, 'row_per_block', true);
np = numel(plan);

[stem, ext] = local_stem(pngname);
pageno = zeros(nchan, 1);
pages  = local_empty_manifest();
for ip = 1:np
    p = plan(ip);
    pageno(p.idx) = ip;
    if opts.index_only, continue; end    % plan only: the index is a
                                         % function of the page PLAN, so
                                         % it can be rebuilt without
                                         % redrawing 69 pages
    f = dw_page_fig(p.fig_in, ttl);
    for s = 1:numel(p.idx)
        c = p.idx(s);
        [ax, ~] = dw_page_axes(p, p.slot(s));
        dw_draw_map(ax, macos.v2m(J(:, c), indx), ...
            dw_panel_label(names{c}), [], opts.fontsize);
    end
    rng = local_range(lbl(p.idx));
    if p.nparts > 1                      % one block over several pages
        rng = sprintf('%s [%d of %d]', rng, p.part, p.nparts);
    end
    if np == 1
        head = ttl;
        file = fullfile(here, pngname);
    else
        head = sprintf('%s -- %s (p%d/%d)', ttl, rng, ip, np);
        if ~isfolder(pdir), mkdir(pdir); end
        file = fullfile(pdir, sprintf('%s_p%02d%s', stem, ip, ext));
    end
    sgtitle(head, 'Interpreter', 'none', 'FontSize', opts.fontsize + 2);
    print(f, file, '-dpng', '-r140');
    close(f);
    fprintf('wrote %s\n', file);
    pages(end+1) = struct('file', string(file), 'kind', "channels", ...
        'mode', "multi", 'block', string(rng), ...
        'channels', {names(p.idx)}, 'n', numel(p.idx), ...
        'page', ip, 'npages', np);  %#ok<AGROW>
end

want_index = opts.index;
if isempty(want_index), want_index = (np > 1) || opts.index_only; end
if want_index && np > 1
    % the historical name keeps its place and becomes the index: nothing
    % that referenced <name>_<ch>_channels.png stops resolving, and the
    % one-sheet view of the whole Jacobian survives as the way to FIND a
    % channel rather than to read it
    ifile = fullfile(here, pngname);
    [~, pdirname] = fileparts(pdir);
    plot_dw_index(out, pageno, sprintf( ...
        '%s -- index of %d full-size pages in %s/', ttl, np, pdirname), ...
        ifile);
    pages(end+1) = struct('file', string(ifile), 'kind', "index", ...
        'mode', "channels", 'block', "all", 'channels', {names(:)}, ...
        'n', nchan, 'page', 0, 'npages', np);
end
end

% ---------------------------------------------------------------------
function s = local_range(lbls)
%LOCAL_RANGE  "element 5" | "elements 5-8" | "element 5 + group PM"
u = unique(lbls, 'stable');
if numel(u) == 1, s = u{1}; return; end
ie = regexp(u, '^element (\d+)$', 'tokens', 'once');
ok = ~cellfun(@isempty, ie);
if all(ok)
    v = cellfun(@(t) str2double(t{1}), ie);
    s = sprintf('elements %d-%d', min(v), max(v));
else
    s = strjoin(u, ' + ');
end
end

function [stem, ext] = local_stem(pngname)
[~, stem, ext] = fileparts(pngname);
if isempty(ext), ext = '.png'; end
end

function m = local_empty_manifest()
m = struct('file', {}, 'kind', {}, 'mode', {}, 'block', {}, ...
           'channels', {}, 'n', {}, 'page', {}, 'npages', {});
end
