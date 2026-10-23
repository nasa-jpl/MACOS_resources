function [key, tag, lbl] = dw_block_keys(out)
%DW_BLOCK_KEYS  Per-channel PAGE BLOCK: which element / group / source a
%   channel belongs to.  Shared by every dw_d* plotter so the overview,
%   the per-element pages and the index all section the SAME way.
%
%   [KEY, TAG, LBL] = DW_BLOCK_KEYS(OUT) returns, per channel:
%     KEY  'elt:<id>' | 'grp:<name>' | 'elt:0' (source)   -- the block id
%     TAG  'elt<id>'  | 'grp<name>'  | 'src'              -- for filenames
%     LBL  'element <id>' | 'group <name>' | 'source'     -- for titles
%
%   GROUP channels (macos.dw_dx / dw_dx_multi 'groups') carry iElt = 0,
%   the same value SOURCE channels carry, so they are sectioned by KIND:
%   OUT.kind == 'Group' when the supervisor reports it, else the 'Grp[..]'
%   name prefix macos.channels.GroupedRigidBodyChannel writes.  A harvest
%   with neither (an older one, or one with no iElt at all) falls back to
%   the "Elt N <suffix>" channel-name parse.
%
%   See also: plot_dw_channels, plot_dw_per_element, dw_page_layout.

names = out.channel_names;
n     = numel(names);
key   = cell(n, 1);  tag = cell(n, 1);  lbl = cell(n, 1);
has_elt  = isfield(out, 'iElt') && numel(out.iElt) == n;
has_kind = isfield(out, 'kind') && numel(out.kind) == n;
for k = 1:n
    nm = strtrim(char(names{k}));
    isgrp = startsWith(nm, 'Grp[');
    if has_kind, isgrp = isgrp || strcmp(out.kind{k}, 'Group'); end
    if isgrp
        g = regexp(nm, '^Grp\[(.*?)\]', 'tokens', 'once');
        if isempty(g), gname = 'group'; else, gname = g{1}; end
        key{k} = ['grp:' gname];
        tag{k} = ['grp' regexprep(gname, '[^A-Za-z0-9]', '_')];
        lbl{k} = sprintf('group %s', gname);
        continue
    end
    if has_elt
        ie = out.iElt(k);
    else
        t = regexp(nm, '^Elt\s+(\d+)\s', 'tokens', 'once');
        if isempty(t), ie = 0; else, ie = str2double(t{1}); end
    end
    key{k} = sprintf('elt:%d', ie);
    if ie == 0
        tag{k} = 'src';   lbl{k} = 'source';
    else
        tag{k} = sprintf('elt%d', ie);  lbl{k} = sprintf('element %d', ie);
    end
end
end
