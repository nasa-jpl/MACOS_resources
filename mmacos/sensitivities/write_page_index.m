function write_page_index(man, file, name)
%WRITE_PAGE_INDEX  The harvest's page index: one line per figure written.
%   WRITE_PAGE_INDEX(MAN, FILE, NAME) writes the manifest returned by
%   PLOT_DW_CHANNELS / PLOT_DW_PER_ELEMENT as a plain-text table -- kind,
%   mode, block (element / group / source), panel count, page number, and
%   the file -- so a harvest of hundreds of pages is navigable and
%   greppable ("which page is element 17's Kc on?").  The per-channel-kind
%   contact sheets -- <name>_<ch>_channels.png, which keeps its historical
%   name and becomes the INDEX once the set paginates -- are listed in it
%   too; they are the picture version of the same question.
%
%   See also: plot_dw_channels, plot_dw_per_element, plot_dw_index.

arguments
    man struct
    file (1,:) char
    name (1,:) char = ''
end
fid = fopen(file, 'w');
if fid < 0
    warning('write_page_index:open', 'cannot write %s', file);
    return
end
closer = onCleanup(@() fclose(fid));
fprintf(fid, '# %s -- sensitivity figure index (%d files)\n', name, numel(man));
fprintf(fid, '# every dW page written by this harvest, in the order it was drawn.\n');
fprintf(fid, '# %-11s %-14s %-22s %5s %8s  %s\n', ...
    'kind', 'mode', 'block', 'panels', 'page', 'file');
for k = 1:numel(man)
    if man(k).npages > 1 && man(k).page > 0
        pg = sprintf('%d/%d', man(k).page, man(k).npages);
    else
        pg = '-';
    end
    fprintf(fid, '  %-11s %-14s %-22s %5d %8s  %s\n', ...
        char(man(k).kind), char(man(k).mode), char(man(k).block), ...
        man(k).n, pg, local_short(man(k).file));
end
fprintf(fid, '\n# channels per page\n');
for k = 1:numel(man)
    fprintf(fid, '  %s: %s\n', local_short(man(k).file), ...
        strjoin(cellfun(@(c) strtrim(char(c)), man(k).channels(:).', ...
        'UniformOutput', false), ', '));
end
fprintf('wrote %s\n', file);
end

function s = local_short(f)
%LOCAL_SHORT  <parent>/<file> -- enough to find it, short enough to read.
[d, b, e] = fileparts(char(f));
[~, dd] = fileparts(d);
s = fullfile(dd, [b e]);
end
