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
%     'elts'    restrict to these element ids (intersected with the
%               discovered powered set).
%
%   See also: macos.channels.SurfChannel, macos.find_powered_elts,
%             macos.dw_dsurf.
    arguments
        session
        rx_path (1,:) char {mustBeNonempty}
        opts.params cell = {'Kr','Kc'}
        opts.elts  (:,1) double = []
    end
    pe = macos.find_powered_elts(session, rx_path);
    if ~isempty(opts.elts)
        pe = intersect(pe, opts.elts(:));
    end
    chans = {};
    for k = 1:numel(pe)
        for p = 1:numel(opts.params)
            chans{end+1, 1} = macos.channels.SurfChannel( ...
                session, pe(k), opts.params{p}); %#ok<AGROW>
        end
    end
end
