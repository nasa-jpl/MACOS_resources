function out = parse_rx_actual_optic_elts_(rx_path, include_non_optics)
%PARSE_RX_ACTUAL_OPTIC_ELTS_  Scan an Rx for actual-optic elements.
%   out = parse_rx_actual_optic_elts_(RX_PATH) returns a
%   containers.Map (int->char) mapping 1-based element id to the
%   Element= kind label (Reflector / Refractor / Segment / FocalPlane
%   / HOE / Grating / ...) for every element whose Element= line is
%   NOT 'Reference' or 'Return' (the bookkeeping-only kinds).
%
%   parse_rx_actual_optic_elts_(RX_PATH, true) retains Reference /
%   Return too -- needed by predict_global_rigid_response-style
%   workflows that combine all per-element columns.

arguments
    rx_path             (1,:) char
    include_non_optics  (1,1) logical = false
end

non_optic = {'Reference','Return'};
out = containers.Map('KeyType','int32','ValueType','char');

fid = fopen(rx_path, 'r');
if fid < 0
    error('parse_rx_actual_optic_elts_:open', ...
        'cannot open Rx file: %s', rx_path);
end
c = onCleanup(@() fclose(fid));

cur_elt = [];
cur_kind = '';

flush = @(elt, kind) flush_(out, elt, kind, non_optic, include_non_optics);

while true
    ln = fgetl(fid);
    if ~ischar(ln); break; end
    s = strtrim(ln);
    if startsWith(s, 'iElt=')
        flush(cur_elt, cur_kind);
        rest = strtrim(extractAfter(s, '='));
        v = sscanf(rest, '%d', 1);
        if isempty(v), cur_elt = []; else, cur_elt = v; end
        cur_kind = '';
    elseif startsWith(s, 'Element=') && ~isempty(cur_elt)
        rest = strtrim(extractAfter(s, '='));
        toks = regexp(rest, '\s+', 'split');
        if ~isempty(toks)
            cur_kind = toks{1};
        end
    end
end
flush(cur_elt, cur_kind);
end


function flush_(out, cur_elt, cur_kind, non_optic, include_non_optics)
if isempty(cur_elt) || isempty(cur_kind)
    return;
end
if ~include_non_optics && any(strcmp(cur_kind, non_optic))
    return;
end
out(int32(cur_elt)) = cur_kind;
end
