function R = rx_blocks(f, who)
%RX_BLOCKS  Split a prescription into header / element blocks / terminator.
%   R = rx_blocks(F, WHO) returns R.units (the BaseUnits= value, '' if none),
%   R.header (the text before the first iElt= line), R.elts (one cell per
%   element block, delimited by the iElt= lines) and R.tail (the `% Output
%   Coordinate System` / nOutCord terminator block the parser needs as the
%   element-list end marker -- without it SMACOS loads nElt = 0).  WHO names
%   the caller in the error identifiers (append_rx, add_exit_pupil).
%
%   Text only, no engine: the caller traces the result it writes.
    txt = fileread(f);
    u = regexp(txt, '(?m)^\s*BaseUnits=\s*(\S+)', 'tokens', 'once');
    R.units = '';  if ~isempty(u), R.units = u{1}; end

    iTail = regexp(txt, '(?m)^%\s*Output Coordinate System', 'once');
    if isempty(iTail)
        iTail = regexp(txt, '(?m)^\s*nOutCord=', 'once');
    end
    if isempty(iTail)
        error(['macos:design:' who ':terminator'], ...
            ['%s: %s has no nOutCord terminator block -- the parser ' ...
             'needs it as the element-list end marker (without it SMACOS ' ...
             'loads nElt = 0).'], who, f);
    end
    body = txt(1:iTail-1);
    R.tail = txt(iTail:end);

    st = regexp(body, '(?m)^\s*iElt=', 'start');
    if isempty(st)
        error(['macos:design:' who ':noelts'], '%s: %s declares no elements', who, f);
    end
    R.header = regexprep(body(1:st(1)-1), '\s+$', '');
    R.elts = cell(1, numel(st));
    for k = 1:numel(st)
        if k < numel(st), e = body(st(k):st(k+1)-1); else, e = body(st(k):end); end
        R.elts{k} = regexprep(e, '\s+$', '');
    end
end
