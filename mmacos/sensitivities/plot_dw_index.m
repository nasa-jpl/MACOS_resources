function fig = plot_dw_index(out, pageno, ttl, file)
%PLOT_DW_INDEX  Contact sheet: every channel as a thumbnail, page-labelled.
%   PLOT_DW_INDEX(OUT, PAGENO, TTL, FILE) draws one small panel per
%   channel -- rows = element / group / source block, cols = that block's
%   channels.  Each row is named ONCE, in a rotated label on its first
%   thumbnail; each thumbnail is titled with its channel and the page of
%   the full-size set it is drawn on ("Rx   pg 4"), and carries its own
%   rms underneath.
%
%   WHY THE RMS IS THERE (Dave 2026-09-10).  These thumbnails are
%   autoscaled and carry no colorbar, so a channel whose entire content
%   is round-off gets painted across the full jet range and reads as
%   structure -- the jwst zoom deck's virtual centre segment (element 4)
%   did exactly that at 1e-16 mm.  The rms under each panel is the scale
%   the colours are missing: a dead channel now announces itself.
%   It is the rms of the map AS DRAWN (piston removed, valid pixels
%   only), so it describes the picture above it.
%
%   This is deliberately the DENSE sheet the all-channels overview used
%   to be: once the readable pages are paginated, the one-sheet view
%   stops being the way to read a channel and becomes the way to FIND
%   one.  Keeping it means a harvest of hundreds of pages still opens
%   with a single picture of the whole Jacobian.
%
%   See also: plot_dw_channels, dw_block_keys, dw_page_title.

arguments
    out (1,1) struct
    pageno (:,1) double
    ttl (1,:) char
    file (1,:) char
end
J     = out.dwdxall;
indx  = out.indxall;
names = out.channel_names;
[key, tag, ~] = dw_block_keys(out);
[ub, ~, bi] = unique(key, 'stable');
nrow = numel(ub);
ncol = max(accumarray(bi(:), 1));

% Geometry: a thumbnail size first, then the sheet around it.  Each cell
% carries a title above and an rms below, so the rows are spread by that
% much -- crowding them is what made the old sheet unreadable.  Sized in
% inches like every other dW page (see dw_page_fig): the legacy pixel
% formula printed a different sheet on every screen.
lab  = 0.20;                                     % title band
rms_ = 0.22;                                     % rms band
side = 0.42;                                     % room for the row label
head = 1.0;
for pass = 1:2
    thumb = min(0.70, max(0.42, (28 - head - 0.2)/nrow - (lab + rms_ + 0.06)));
    cw = thumb + 0.34;  ch = thumb + lab + rms_ + 0.06;
    W  = max(3.4, side + 0.3 + ncol*cw);
    tlines = local_wrap(ttl, max(24, floor(W/0.085)), 4);
    head = 0.34 + 0.24*numel(tlines);
end
H  = head + 0.2 + nrow*ch;
xoff = side + (W - side - ncol*cw) / 2;          % centre the columns
fig = dw_page_fig([W H], ttl);
for b = 1:nrow
    ks = find(bi == b);
    for c = 1:numel(ks)
        k  = ks(c);
        x0 = xoff + (c-1)*cw + (cw - thumb)/2;
        y0 = (H - head) - (b-1)*ch - lab - thumb;
        ax = axes('Units', 'normalized', ...
            'Position', [x0/W, y0/H, thumb/W, thumb/H]);
        M = macos.v2m(J(:, k), indx);
        dw_draw_map(ax, M, local_label(names{k}, pageno(k)), [], 6);
        text(ax, 0.5, -0.05, local_rms(M), 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
            'FontSize', 6, 'Color', [0.25 0.25 0.25]);
        if c == 1                                % name the row ONCE
            text(ax, -0.10, 0.5, local_rowname(tag{k}), ...
                'Units', 'normalized', 'Rotation', 90, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', 'FontSize', 7, ...
                'FontWeight', 'bold', 'Color', [0.15 0.15 0.15]);
        end
    end
end
dw_page_title([W H], tlines, 9);
print(fig, file, '-dpng', '-r140');
close(fig);
fprintf('wrote %s\n', file);
end

% ---------------------------------------------------------------------
function s = local_label(name, pg)
%LOCAL_LABEL  "Rx   pg 4" -- the element is the ROW's label, not each
%   panel's, so only the channel suffix and the page appear here.
t = regexp(strtrim(char(name)), '^Elt\s+\d+\s+(.*)$', 'tokens', 'once');
if isempty(t), s = strtrim(char(name)); else, s = strtrim(t{1}); end
if pg > 0, s = sprintf('%s   pg %d', s, pg); end
end

% ---------------------------------------------------------------------
function s = local_rowname(tg)
%LOCAL_ROWNAME  'elt7' -> 'Elt 7', 'grpPM' -> 'Grp PM', 'src' -> 'Src'.
if strcmp(tg, 'src')
    s = 'Src';
elseif startsWith(tg, 'grp')
    s = ['Grp ' extractAfter(tg, 'grp')];
else
    s = ['Elt ' extractAfter(tg, 'elt')];
end
end

% ---------------------------------------------------------------------
function s = local_rms(M)
%LOCAL_RMS  The rms of the map AS DRAWN: exact zeros are outside the
%   pupils and are not drawn, and the panel has its piston removed, so
%   the number describes the picture.  A dead channel reads 0.
v = M(M ~= 0);
if isempty(v), s = 'rms 0';  return;  end
v = v - mean(v);
r = sqrt(mean(v.^2));
if r == 0, s = 'rms 0'; else, s = sprintf('rms %.1e', r); end
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
