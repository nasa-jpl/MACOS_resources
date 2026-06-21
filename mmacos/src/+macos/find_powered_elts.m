function pe = find_powered_elts(session, rx_path, opts)
%MACOS.FIND_POWERED_ELTS  Indices of POWERED optics in the beam train.
%   pe = macos.find_powered_elts(SESSION, RX_PATH) returns a column vector
%   of 1-based element ids for every POWERED optic in RX_PATH: an
%   Element= Reflector or Refractor whose base radius is real
%   (|Kr| << the flat sentinel 1e22).  Flats (fold mirrors, FocalPlane,
%   Reference, Return) carry the ~1e22 sentinel and are excluded.
%
%   The Element= type is parsed from the Rx text (no mex query exists, as
%   in find_zern_elts); the powered (finite-Kr) filter is applied via the
%   engine (RX_PATH must already be loaded on SESSION).  Used by
%   surf_channels() to build the dw/dKr, dw/dKc eligibility set.
%
%   Name-value:
%     'kr_max'  |Kr| below this counts as powered.  Default 1e21.
%
%   See also: macos.find_zern_elts, macos.channels.surf_channels.
    arguments
        session
        rx_path (1,:) char {mustBeNonempty}
        opts.kr_max (1,1) double = 1e21
    end

    % --- parse Element= Reflector / Refractor (the powered-capable optics)
    optic = zeros(0,1);  cur = NaN;
    fid = fopen(rx_path, 'r');
    if fid < 0
        error('macos:find_powered_elts:open', 'cannot open Rx file: %s', rx_path);
    end
    cleanup_obj = onCleanup(@() fclose(fid));
    while true
        ln = fgetl(fid);
        if ~ischar(ln); break; end
        s = strtrim(ln);
        if startsWith(s, 'iElt=')
            v = sscanf(strtrim(extractAfter(s, '=')), '%d', 1);
            if ~isempty(v), cur = v; else, cur = NaN; end
        elseif startsWith(s, 'Element=') && ~isnan(cur)
            toks = regexp(strtrim(extractAfter(s, '=')), '\s+', 'split');
            if ~isempty(toks) && any(strcmpi(toks{1}, {'Reflector','Refractor'}))
                optic(end+1, 1) = cur; %#ok<AGROW>
            end
        end
    end
    optic = unique(optic);

    % --- keep only the POWERED ones (finite curvature), via the engine
    pe = zeros(0,1);
    for k = 1:numel(optic)
        if abs(macos.get_elt_kr(optic(k))) < opts.kr_max
            pe(end+1, 1) = optic(k); %#ok<AGROW>
        end
    end
end
