function s = dw_panel_label(name)
%DW_PANEL_LABEL  Short panel title from a channel name.
%   "Elt 12 MonZern5" -> "E12 MonZern5"; a source or group channel keeps
%   its own name.  (The overview has always shortened the element prefix
%   this way; the per-element pages print the full channel name, where
%   the element is in the page title instead.)
t = regexp(strtrim(char(name)), '^Elt\s+(\d+)\s+(.*)$', 'tokens', 'once');
if isempty(t)
    s = strtrim(char(name));
else
    s = sprintf('E%s %s', t{1}, strtrim(t{2}));
end
end
