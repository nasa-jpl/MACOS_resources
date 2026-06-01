function out = parse_rx_modes_(rx_path, n_key, modes_key)
%PARSE_RX_MODES_  Scan an Rx file for per-element Zernike-mode declarations.
%   out = parse_rx_modes_(RX_PATH, N_KEY, MODES_KEY) returns a
%   containers.Map (int->vec) mapping element id to the active list
%   of mode indices.
%
%   Semantics mirror msmacosio.inc:
%     - If MODES_KEY=... is present, use the explicit list.
%     - Else if N_KEY=K is present, default to [1..K].
%     - Else the element is omitted.
%   Continuation lines after MODES_KEY=... (bare ints, msmacosio.inc
%   Grp parameter) are honoured up to N_KEY items total.

out = containers.Map('KeyType','int32','ValueType','any');

fid = fopen(rx_path, 'r');
if fid < 0
    error('parse_rx_modes_:open', 'cannot open Rx file: %s', rx_path);
end
cleanup_obj = onCleanup(@() fclose(fid));

lines = {};
while true
    ln = fgetl(fid);
    if ~ischar(ln); break; end
    lines{end+1} = ln; %#ok<AGROW>
end

cur_elt = [];
n_active = [];
explicit_modes = [];
pending_cont = 0;

flush = @(elt, n, modes) flush_(out, elt, n, modes);

n_lines = numel(lines);
i = 1;
while i <= n_lines
    ln = lines{i};
    s = strtrim(ln);

    if pending_cont > 0
        toks = regexp(strrep(ln, ',', ' '), '\s+', 'split');
        toks = toks(~cellfun(@isempty, toks));
        ints = zeros(0,1);
        for t = toks
            v = sscanf(t{1}, '%g', 1);
            if isempty(v) || isnan(v)
                break;
            end
            ints(end+1, 1) = round(v); %#ok<AGROW>
        end
        if ~isempty(ints)
            take = min(numel(ints), pending_cont);
            explicit_modes = [explicit_modes; ints(1:take)]; %#ok<AGROW>
            pending_cont = pending_cont - take;
        end
        i = i + 1;
        continue;
    end

    if startsWith(s, 'iElt=')
        flush(cur_elt, n_active, explicit_modes);
        rest = strtrim(extractAfter(s, '='));
        v = sscanf(rest, '%d', 1);
        if isempty(v), cur_elt = []; else, cur_elt = v; end
        n_active = [];
        explicit_modes = [];
    elseif startsWith(s, [n_key '='])
        rest = strtrim(extractAfter(s, '='));
        v = sscanf(rest, '%d', 1);
        if isempty(v), n_active = []; else, n_active = v; end
        explicit_modes = [];
    elseif startsWith(s, [modes_key '='])
        if isempty(n_active)
            i = i + 1;
            continue;
        end
        payload = extractAfter(s, '=');
        toks = regexp(strrep(payload, ',', ' '), '\s+', 'split');
        toks = toks(~cellfun(@isempty, toks));
        explicit_modes = zeros(0,1);
        for t = toks
            v = sscanf(t{1}, '%g', 1);
            if isempty(v) || isnan(v)
                break;
            end
            explicit_modes(end+1, 1) = round(v); %#ok<AGROW>
        end
        remaining = n_active - numel(explicit_modes);
        if remaining > 0
            pending_cont = remaining;
        end
    end
    i = i + 1;
end
flush(cur_elt, n_active, explicit_modes);
end


function flush_(out, cur_elt, n_active, explicit_modes)
if isempty(cur_elt) || isempty(n_active)
    return;
end
if ~isempty(explicit_modes)
    take = min(numel(explicit_modes), n_active);
    out(int32(cur_elt)) = explicit_modes(1:take);
else
    out(int32(cur_elt)) = (1:n_active).';
end
end
