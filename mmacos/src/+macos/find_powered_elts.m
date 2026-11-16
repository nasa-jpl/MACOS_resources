function [pe, kinds] = find_powered_elts(session, rx_path, opts)
%MACOS.FIND_POWERED_ELTS  Indices of POWERED optics in the beam train.
%   pe = macos.find_powered_elts(SESSION, RX_PATH) returns a column vector
%   of 1-based element ids for every POWERED optic in the LOADED Rx: an
%   element of a powered-capable Element= kind whose base radius is real
%   (|Kr| << the flat sentinel 1e22).
%
%   Powered-capable kinds: Reflector, Refractor, NSReflector, NSRefractor,
%   Segment.  Reference/Return/FocalPlane/Obscuring are excluded even with
%   finite Kr (a powered exit-pupil Return sphere is a reference, not an
%   optic to perturb).  Gratings/HOE/TrGrating also carry base conics but
%   are EXCLUDED for now (Dave's ruling 2026-09-05, BRIEF_luis_round3 --
%   revisit when a grating sensitivity user appears).
%
%   Element kinds are read from the ENGINE (macos.get_elt_info /
%   elt_info_get), not by parsing the .in text -- text parsing silently
%   dropped NSReflector (Luis 2026-09-05), and decks whose declared nElt
%   disagrees with their Element-block count mis-index a text parse (the
%   FEX blast-radius lesson).  RX_PATH is kept for signature compatibility
%   and error context only; it must already be loaded on SESSION.
%
%   [pe, kinds] = ... also returns the powered-capable kind list (cellstr)
%   so callers can compose eligibility messages from the same authority.
%
%   Name-value:
%     'kr_max'  |Kr| below this counts as powered.  Default 1e21.
%
%   See also: macos.get_elt_info, macos.find_zern_elts,
%             macos.channels.surf_channels.
    arguments
        session
        rx_path (1,:) char {mustBeNonempty} %#ok<INUSA> engine state is the authority
        opts.kr_max (1,1) double = 1e21
    end

    kinds = {'Reflector','Refractor','NSReflector','NSRefractor','Segment'};
    pe = zeros(0,1);
    for k = 1:session.num_elt()
        info = macos.get_elt_info(k);
        if any(strcmp(info.type, kinds)) && ...
                abs(macos.get_elt_kr(k)) < opts.kr_max
            pe(end+1, 1) = k; %#ok<AGROW>
        end
    end
end
