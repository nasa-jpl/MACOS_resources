function groups = parse_rx_groups(rx_path)
%MACOS.CHANNELS.PARSE_RX_GROUPS  Parse EltGrp= declarations from an Rx.
%   groups = macos.channels.parse_rx_groups(RX_PATH) scans the .in
%   file for EltGrp= lines and returns a containers.Map (char -> column
%   vector of double) mapping group names ('min-max' of member ids)
%   to their member lists.
%
%   macos's .in convention has every member of a group repeat the
%   same "EltGrp= N m1 m2 ... mN" declaration in its own per-element
%   block.  This parser dedups by the sorted member tuple and emits
%   one entry per unique group.
%
%   Only the positive-N (explicit list) form is parsed.  Negative-N
%   (range form) and MrEltGrp (multi-range) are deferred.

arguments
    rx_path (1,:) char
end

groups = containers.Map('KeyType', 'char', 'ValueType', 'any');
seen   = containers.Map('KeyType', 'char', 'ValueType', 'any');

fid = fopen(rx_path, 'r');
if fid < 0
    error('macos:channels:parse_rx_groups:open', ...
        'cannot open Rx file: %s', rx_path);
end
c = onCleanup(@() fclose(fid));

cur_elt = [];
while true
    ln = fgetl(fid);
    if ~ischar(ln); break; end
    s = strtrim(ln);
    if startsWith(s, 'iElt=')
        rest = strtrim(extractAfter(s, '='));
        v = sscanf(rest, '%d', 1);
        if isempty(v), cur_elt = []; else, cur_elt = v; end
    elseif startsWith(s, 'EltGrp=') && ~isempty(cur_elt)
        payload = extractAfter(s, '=');
        toks = regexp(strrep(payload, ',', ' '), '\s+', 'split');
        toks = toks(~cellfun(@isempty, toks));
        ints = zeros(0, 1);
        for tk = toks
            v = sscanf(tk{1}, '%d', 1);
            if isempty(v); break; end
            ints(end+1, 1) = v; %#ok<AGROW>
        end
        if isempty(ints) || ints(1) <= 0
            continue;
        end
        n = ints(1);
        if numel(ints) < n + 1
            continue;
        end
        members = sort(ints(2:n+1));
        key = mat2str(members(:).');
        if ~isKey(seen, key)
            seen(key) = true;
            nm = sprintf('%d-%d', members(1), members(end));
            groups(nm) = members(:);
        end
    end
end
end
