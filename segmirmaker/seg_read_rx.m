function D = seg_read_rx(rx)
%SEG_READ_RX  Minimal MACOS .in reader: header keys + per-element blocks.
%   D = SEG_READ_RX(RX) parses the prescription as TEXT (not via engine
%   getters) on purpose: a routing audit must see what the DECK says,
%   including keys the engine silently defaults.
%
%   Returns:
%     D.hdr      header keys, numeric where parseable (GridType kept as text)
%     D.hdrRaw   header keys, raw strings
%     D.eltType  1xN cellstr of Element= values
%     D.vecs.<k> 3xN array for each of the standard 3-vector keys
%     D.raw      1xN cell of containers.Map, the raw per-element key/values
%
%   Shared by seg_audit.m and seg_route.m.

% Normalise line terminators first.  Some legacy decks (e.g.
% MACOS_sandbox/old_Rx/btc3.in) are classic-Mac CR-only, which
% strsplit(...,newline) reads as ONE line -- the whole deck then parses as
% a single key and no elements are found.
raw = fileread(rx);
raw = strrep(strrep(raw, sprintf('\r\n'), newline), sprintf('\r'), newline);
txt = strsplit(raw, newline);
D.hdr = struct(); D.hdrRaw = struct(); D.eltType = {};
inElt = false; lastKey = '';
raw = containers.Map('KeyType','char','ValueType','char');
rawElt = {};
for n = 1:numel(txt)
    L = txt{n};
    p = strfind(L, '%'); if ~isempty(p), L = L(1:p(1)-1); end
    if isempty(strtrim(L)), continue, end
    tok = regexp(L, '^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*)$', 'tokens', 'once');
    if ~isempty(tok)
        k = tok{1}; v = strtrim(tok{2});
        if strcmp(k, 'iElt')
            if inElt, rawElt{end+1} = raw; end %#ok<AGROW>
            raw = containers.Map('KeyType','char','ValueType','char');
            inElt = true;
        end
        raw(k) = v; lastKey = k;
        if ~inElt, D.hdrRaw.(k) = v; end
    elseif ~isempty(lastKey)
        raw(lastKey) = [raw(lastKey) ' ' strtrim(L)];
        if ~inElt, D.hdrRaw.(lastKey) = raw(lastKey); end
    end
end
if inElt, rawElt{end+1} = raw; end

for f = fieldnames(D.hdrRaw).'
    D.hdr.(f{1}) = str2vec(D.hdrRaw.(f{1}));
end
if isfield(D.hdrRaw, 'GridType')
    D.hdr.GridType = strtrim(D.hdrRaw.GridType);
else
    D.hdr.GridType = '';
end

nE = numel(rawElt);
D.eltType = repmat({''}, 1, nE);
D.apType  = repmat({''}, 1, nE);
keys3 = {'psiElt','VptElt','RptElt','pMon','xMon','yMon','zMon', ...
         'pFF','xFF','yFF','zFF','pData','xData','yData','zData'};
for kk = keys3, D.vecs.(kk{1}) = zeros(3, nE); end
for e = 1:nE
    m = rawElt{e};
    if m.isKey('Element'), D.eltType{e} = strtrim(m('Element')); end
    if m.isKey('ApType'),  D.apType{e}  = strtrim(m('ApType'));  end
    for kk = keys3
        if m.isKey(kk{1})
            v = str2vec(m(kk{1}));
            if numel(v) >= 3, D.vecs.(kk{1})(:,e) = v(1:3); end
        end
    end
end
D.raw = rawElt;
end

function v = str2vec(s)
s = strrep(strrep(s, 'D', 'E'), 'd', 'e');
v = sscanf(s, '%f');
if isempty(v), v = NaN; end
end
