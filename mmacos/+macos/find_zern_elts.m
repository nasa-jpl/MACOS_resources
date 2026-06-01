function ze = find_zern_elts(rx_path)
%MACOS.FIND_ZERN_ELTS  Indices of Zernike-typed elements (SrfType=8/13).
%   ze = macos.find_zern_elts(RX_PATH) returns a column vector of 1-based
%   element ids for every element declared as Surface=Zernike or
%   Surface=ZrnGridData / ZrnGrData in the Rx text file.  Empty array
%   if no such surfaces are present.
%
%   No mex query for this exists yet (per pymacos's _parse_rx_zern_elts);
%   the Rx text is the source of truth.  Used by zernike_channels() to
%   build the eligibility set for the Zern channel kind on top of any
%   Conic / sphere base.
%
%   See also: macos.find_freeform_elts.
arguments
    rx_path (1,:) char {mustBeNonempty}
end
ze = zeros(0,1);
cur_elt = NaN;
fid = fopen(rx_path, 'r');
if fid < 0
    error('macos:find_zern_elts:open', ...
        'cannot open Rx file: %s', rx_path);
end
cleanup_obj = onCleanup(@() fclose(fid));
zern_surface_names = {'Zernike','ZrnGridData','ZrnGrData'};
while true
    ln = fgetl(fid);
    if ~ischar(ln); break; end
    s = strtrim(ln);
    if startsWith(s, 'iElt=')
        rest = strtrim(extractAfter(s, '='));
        v = sscanf(rest, '%d', 1);
        if ~isempty(v)
            cur_elt = v;
        else
            cur_elt = NaN;
        end
    elseif startsWith(s, 'Surface=') && ~isnan(cur_elt)
        rest = strtrim(extractAfter(s, '='));
        toks = regexp(rest, '\s+', 'split');
        if ~isempty(toks) && any(strcmp(toks{1}, zern_surface_names))
            ze(end+1, 1) = cur_elt; %#ok<AGROW>
        end
    end
end
ze = unique(ze);
end
