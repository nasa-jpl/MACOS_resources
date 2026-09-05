function chans = surf_channels(session, rx_path, opts)
%MACOS.CHANNELS.SURF_CHANNELS  Build Kr/Kc channels for powered optics.
%   chans = macos.channels.surf_channels(SESSION, RX_PATH) returns a cell
%   array of SurfChannel handles -- one per (powered optic, param) pair,
%   with param in {Kr, Kc} -- for every POWERED optic (Reflector /
%   Refractor with |Kr| << 1e22) in the loaded Rx.  Channels are emitted
%   element-major, param-minor (Kr then Kc per optic) so the natural
%   per-optic block layout is the output order.
%
%   Name-value:
%     'params'  cellstr subset of {'Kr','Kc'}.  Default {'Kr','Kc'}.
%     'elts'    restrict to these element ids.  An explicitly requested id
%               that is not powered-capable ERRORS with a named reason
%               (macos:channels:eltNotEligible) -- it never vanishes
%               silently (BRIEF_luis_round3).
%
%   See also: macos.channels.SurfChannel, macos.find_powered_elts,
%             macos.dw_dsurf.
    arguments
        session
        rx_path (1,:) char {mustBeNonempty}
        opts.params cell = {'Kr','Kc'}
        opts.elts  (:,1) double = []
    end
    [pe, kinds] = macos.find_powered_elts(session, rx_path);
    pe = require_elts(pe, opts.elts, 'surf', ...
        @(id) surf_reason(session, id, kinds));
    chans = {};
    for k = 1:numel(pe)
        for p = 1:numel(opts.params)
            chans{end+1, 1} = macos.channels.SurfChannel( ...
                session, pe(k), opts.params{p}); %#ok<AGROW>
        end
    end
end

function r = surf_reason(session, id, kinds)
% Why a requested element is not in the powered set -- engine-truth.
    n = session.num_elt();
    if id < 1 || id ~= round(id) || id > n
        r = sprintf('element id out of range (nElt = %d)', n);
        return;
    end
    info = macos.get_elt_info(id);
    if ~any(strcmp(info.type, kinds))
        r = sprintf(['Element= %s is not powered-capable ' ...
                     '(family: %s)'], info.type, strjoin(kinds, '/'));
    elseif abs(macos.get_elt_kr(id)) >= 1e21
        r = sprintf('|Kr| = %.3g is the flat sentinel -- no radius to perturb', ...
                    macos.get_elt_kr(id));
    else
        r = 'excluded for an unrecognized reason -- report this';
    end
end
