function pages = plot_dw_per_element(out, fieldmode, here, prefix, opts)
%PLOT_DW_PER_ELEMENT  One page per element: that element's dW channels,
%   drawn at a size you can read at any segment count.
%
%   PAGES = PLOT_DW_PER_ELEMENT(OUT, FIELDMODE, HERE, PREFIX) writes one
%   PNG per element (per group, per source block -- see DW_BLOCK_KEYS),
%   one panel per channel (DOF / mode / parameter) on that element.  The
%   panel size is fixed FIRST from 'panel_in' / 'tile_in' and the panels
%   per page follow (DW_PAGE_LAYOUT); an element whose channels do not
%   fit one page continues onto _p02 rather than shrinking.  Generic
%   across every dw_d*_multi supervisor:
%     dw_dx_multi    -> per optic   x {Rx Ry Rz Tx Ty Tz}
%     dw_dz_*_multi  -> per optic   x Zernike modes
%     dw_dsurf_multi -> per optic   x {Kr, Kc}
%     dw_dgrid_multi -> per segment x grid-influence modes
%
%   FIELDMODE = 'center' : the CENTRE-field dW (single-field map).
%             = 'multi'  : the MULTI-field tiled canvas.
%             = 'field'  : ONE PAGE PER (configuration, field) -- every
%                          field's own single-field maps at full size.
%                          The mode for the segment-count regime: a
%                          5-config x 5-field canvas draws each field at
%                          a ninth of a map, and this is how you get the
%                          maps themselves.  Opt-in (it multiplies the
%                          page count by the block count) and it warns
%                          once with the page count before drawing.
%
%   Display: jet (Dave 2026-09-10), exact zeros not drawn (white),
%   piston removed, each panel autoscaled -- no clim, no bespoke masks
%   (DW_DRAW_MAP owns the convention).
%
%   OUT     struct from any macos.dw_d*_multi call.  Uses the canonical
%           fields dwdxall / indxall / iElt / channel_names /
%           field_names, plus (for 'center' and 'field') the per-field
%           cell per_field_dwd{x,z,s,g} (auto-detected) and
%           per_field_w_nom_2d.
%   PREFIX  filename stem; writes <PREFIX>_<tag>_<mode>.png per element,
%           where <tag> is elt<N> / grp<NAME> / src.  A paginated element
%           adds _p01, _p02 ...; 'field' adds the field (and, with a
%           configuration axis, the configuration) to the name.
%
%   NAME-VALUE
%     'panel_in','tile_in','page_in','page_max_in','max_per_page'
%                 see DW_PAGE_LAYOUT
%     'fontsize'  panel title font (default 9)
%     'fields'    'field' mode: which field indices (default all)
%     'configs'   'field' mode: which configuration indices (default all)
%
%   PAGES is the manifest of what was written, for the harvest index.
%
%   See also: plot_dw_channels, plot_dw_index, plot_opd_canvas,
%             per_field_indx, dw_page_layout.

arguments
    out (1,1) struct
    fieldmode (1,:) char
    here (1,:) char
    prefix (1,:) char
    opts.panel_in (1,1) double = 3.5
    opts.tile_in (1,1) double = 1.2
    opts.page_in (1,2) double = [16 9]
    opts.page_max_in (1,2) double = [32 20]
    opts.max_per_page (1,1) double = 24
    opts.fontsize (1,1) double = 9
    opts.fields double = []
    opts.configs double = []
end

[key, tag, lbl] = dw_block_keys(out);
[ub, ~, bi] = unique(key, 'stable');   % 'stable' keeps the historical
                                       % ascending element page order
pages = local_empty_manifest();

switch lower(fieldmode)
case 'multi'
    blocks = {struct('src', out.dwdxall, 'indx', out.indxall, ...
        'tiles', dw_canvas_tiles(out), 'suffix', 'multi', ...
        'tag', sprintf('%d fields', numel(out.field_names)))};
case 'center'
    ctr = find(strcmp(out.field_names, 'C'), 1);
    if isempty(ctr), ctr = 1; end
    ctag = 'centre field';
    if isfield(out, 'config_names') && numel(out.config_names) > 1
        % with a configuration axis this page is ONE configuration's
        % centre field -- say which, rather than let it read as all of
        % them ('field' mode is how you get the rest)
        ctag = sprintf('centre field, config %s', char(out.config_names{1}));
    end
    blocks = {local_block(out, 1, ctr, 'center', ctag)};
case 'field'
    [nc, nf] = local_block_shape(out);
    ics = opts.configs;  if isempty(ics), ics = 1:nc; end
    kfs = opts.fields;   if isempty(kfs), kfs = 1:nf; end
    blocks = cell(1, numel(ics)*numel(kfs));
    q = 0;
    for ic = ics(:).'
        for k = kfs(:).'
            q = q + 1;
            fn = char(out.field_names{k});
            if nc > 1 && isfield(out, 'config_names')
                cn = char(out.config_names{ic});
                blocks{q} = local_block(out, ic, k, ...
                    sprintf('%s_field%s', cn, fn), ...
                    sprintf('config %s, field %s', cn, fn));
            else
                blocks{q} = local_block(out, ic, k, ...
                    sprintf('field%s', fn), sprintf('field %s', fn));
            end
        end
    end
    local_warn_pages(numel(blocks) * numel(ub));
otherwise
    error('plot_dw_per_element:mode', ...
        'FIELDMODE must be ''center'', ''multi'' or ''field''');
end

for ib = 1:numel(blocks)
    B = blocks{ib};
    for ip = 1:numel(ub)
        cols = find(bi == ip);
        plan = dw_page_layout(key(cols), B.indx.size, B.tiles, ...
            'panel_in', opts.panel_in, 'tile_in', opts.tile_in, ...
            'page_in', opts.page_in, 'page_max_in', opts.page_max_in, ...
            'max_per_page', opts.max_per_page, 'cbar', true);
        np = numel(plan);
        for q = 1:np
            p = plan(q);
            f = dw_page_fig(p.fig_in, prefix);
            for s = 1:numel(p.idx)
                c = cols(p.idx(s));
                [ax, cbr] = dw_page_axes(p, p.slot(s));
                dw_draw_map(ax, macos.v2m(B.src(:, c), B.indx), ...
                    strtrim(char(out.channel_names{c})), cbr, opts.fontsize);
            end
            head = sprintf('%s -- dW, %s, %s (piston removed)', ...
                prefix, lbl{cols(1)}, B.tag);
            if np > 1
                head = sprintf('%s  [pg %d of %d]', head, q, np);
                png = sprintf('%s_%s_%s_p%02d.png', prefix, ...
                    tag{cols(1)}, B.suffix, q);
            else
                png = sprintf('%s_%s_%s.png', prefix, tag{cols(1)}, B.suffix);
            end
            dw_page_title(p.fig_in, head, opts.fontsize + 2);
            if ~isfolder(here), mkdir(here); end
            print(f, fullfile(here, png), '-dpng', '-r140');
            close(f);
            fprintf('wrote %s\n', png);
            pages(end+1) = struct('file', string(fullfile(here, png)), ...
                'kind', "per_element", 'mode', string(B.suffix), ...
                'block', string(lbl{cols(1)}), ...
                'channels', {out.channel_names(cols(p.idx))}, ...
                'n', numel(p.idx), 'page', q, 'npages', np);  %#ok<AGROW>
        end
    end
end
end

% ---------------------------------------------------------------------
function B = local_block(out, ic, k, suffix, tag)
%LOCAL_BLOCK  One (configuration, field) single-field block + its index.
B = struct('src', local_per_field(out, ic, k), ...
           'indx', per_field_indx(out, ic, k), ...
           'tiles', [1 1], 'suffix', suffix, 'tag', tag);
end

function [nc, nf] = local_block_shape(out)
%LOCAL_BLOCK_SHAPE  [n_configs n_fields] of the per-field cell.
cand = {'per_field_dwdx', 'per_field_dwdz', 'per_field_dwds', 'per_field_dwdg'};
for i = 1:numel(cand)
    if isfield(out, cand{i})
        s = size(out.(cand{i}));
        if isvector(out.(cand{i})), nc = 1;  nf = numel(out.(cand{i}));
        else,                       nc = s(1);  nf = s(2);
        end
        return
    end
end
error('plot_dw_per_element:nofield', ...
      'no per_field_dwd{x,z,s,g} cell in OUT -- cannot build a field view');
end

function pf = local_per_field(out, ic, k)
%LOCAL_PER_FIELD  The (ic, k) single-field dW block.  The cell field name
%   is supervisor-specific (per_field_dwdx/dwdz/dwds/dwdg), so detect it.
%   The cell is Nc x Nf WITH a configuration axis and Nf x 1 without --
%   index it 2-D, never linearly: {k} on a 5x5 cell is (config k, field
%   1), which is the centre field only by the accident that 'C' is
%   listed first.
cand = {'per_field_dwdx', 'per_field_dwdz', 'per_field_dwds', 'per_field_dwdg'};
for i = 1:numel(cand)
    if isfield(out, cand{i})
        C = out.(cand{i});
        if isvector(C), pf = C{k};  else, pf = C{ic, k};  end
        return
    end
end
error('plot_dw_per_element:nofield', ...
      'no per_field_dwd{x,z,s,g} cell in OUT -- cannot build a field view');
end

function local_warn_pages(n)
persistent warned
if isempty(warned), warned = false; end
if n > 200 && ~warned
    warned = true;
    warning('plot_dw_per_element:manyPages', ...
        ['''field'' mode will write %d pages (one per element per ' ...
         '(configuration, field)).  Narrow it with ''fields'' / ' ...
         '''configs'', or use ''center''.'], n);
end
end

function m = local_empty_manifest()
m = struct('file', {}, 'kind', {}, 'mode', {}, 'block', {}, ...
           'channels', {}, 'n', {}, 'page', {}, 'npages', {});
end
