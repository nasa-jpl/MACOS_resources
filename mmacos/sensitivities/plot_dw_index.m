function fig = plot_dw_index(out, pageno, ttl, file)
%PLOT_DW_INDEX  Contact sheet: every channel as a thumbnail, page-labelled.
%   PLOT_DW_INDEX(OUT, PAGENO, TTL, FILE) draws one small panel per
%   channel -- rows = element / group / source block, cols = that block's
%   channels -- with each panel titled "<channel>  pNN", NN being the
%   page of the full-size set that channel is drawn on (PAGENO(k), 0 =
%   not paginated).
%
%   This is deliberately the DENSE sheet the all-channels overview used
%   to be: once the readable pages are paginated, the one-sheet view
%   stops being the way to read a channel and becomes the way to FIND
%   one.  Keeping it means a harvest of hundreds of pages still opens
%   with a single picture of the whole Jacobian.
%
%   See also: plot_dw_channels, dw_block_keys.

arguments
    out (1,1) struct
    pageno (:,1) double
    ttl (1,:) char
    file (1,:) char
end
J     = out.dwdxall;
indx  = out.indxall;
names = out.channel_names;
[key, ~, ~] = dw_block_keys(out);
[ub, ~, bi] = unique(key, 'stable');
nrow = numel(ub);
ncol = max(accumarray(bi(:), 1));

% Geometry: a thumbnail size first (bounded, so a 138-channel sheet stays
% one page), then the sheet around it.  Sized in inches like every other
% dW page -- the legacy pixel formula printed a different sheet on every
% screen, and its title ran off both edges.
% Two passes, because the header reserve depends on how many lines the
% title wraps to and that depends on the width, which depends on the
% thumbnail: guess a header, size the sheet, then re-reserve for the
% title the sheet actually needs.  (One pass with a fixed reserve put a
% three-line title straight through the first row of thumbnails.)
head = 1.0;
for pass = 1:2
    thumb = min(0.70, max(0.28, (20 - head - 0.2)/nrow - 0.22));
    cw = thumb + 0.30;  ch = thumb + 0.22;
    W  = max(3.0, 0.3 + ncol*cw);        % 3 in is the title's floor
    tlines = local_wrap(ttl, max(24, floor(W/0.085)), 4);
    head = 0.34 + 0.24*numel(tlines);
end
H  = head + 0.2 + nrow*ch;
xoff = (W - ncol*cw) / 2;               % centre the columns in it
fig = dw_page_fig([W H], ttl);
for b = 1:nrow
    ks = find(bi == b);
    for c = 1:numel(ks)
        x0 = xoff + (c-1)*cw + (cw - thumb)/2;
        y0 = (H - head) - (b-1)*ch - 0.20 - thumb;
        ax = axes('Units', 'normalized', ...
            'Position', [x0/W, y0/H, thumb/W, thumb/H]);
        if pageno(ks(c)) > 0
            lab = sprintf('%s  p%02d', dw_panel_label(names{ks(c)}), ...
                pageno(ks(c)));
        else
            lab = dw_panel_label(names{ks(c)});
        end
        dw_draw_map(ax, macos.v2m(J(:, ks(c)), indx), lab, [], 7);
    end
end
dw_page_title([W H], tlines, 9);
print(fig, file, '-dpng', '-r140');
close(fig);
fprintf('wrote %s\n', file);
end

% ---------------------------------------------------------------------
function c = local_wrap(t, ncmax, nlmax)
%LOCAL_WRAP  Break a title onto at most NLMAX lines of NCMAX characters.
%   The index sheet is as narrow as its channel count lets it be, and a
%   one-line title simply ran off both edges of it.
w = strsplit(strtrim(t), ' ');
c = {''};
for k = 1:numel(w)
    if isempty(c{end})
        c{end} = w{k};
    elseif numel(c{end}) + 1 + numel(w{k}) <= ncmax
        c{end} = [c{end} ' ' w{k}];
    elseif numel(c) < nlmax
        c{end+1} = w{k};  %#ok<AGROW>
    else
        c{end} = [c{end} ' ...'];
        break
    end
end
end
