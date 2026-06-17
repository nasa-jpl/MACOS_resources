function chans = freeform_monzern_channels(session, rx_path, opts)
%MACOS.CHANNELS.FREEFORM_MONZERN_CHANNELS  Build MonZern channels per FF element.
%   chans = macos.channels.freeform_monzern_channels(SESSION, RX_PATH)
%   returns a cell array of ZernikeCoefChannel handles, one per
%   (FreeForm element, MonZern mode) pair declared in the Rx.
%
%   Name-value pairs:
%     'modes_per_elt'  containers.Map (int->vec) overriding the
%                      MonZernModes/MonZernCoef lists from the Rx.
%                      Keys are 1-based element ids; values are
%                      column vectors of 1-based mode indices.
%     'elts'           Vector of element ids to restrict the channel
%                      set to.  Default: all FreeForm-typed elts.
%
%   See also: macos.channels.MonZernChannel,
%             macos.channels.freeform_ffzern_channels.
arguments
    session
    rx_path (1,:) char {mustBeNonempty}
    opts.modes_per_elt = []   % containers.Map or []
    opts.elts          (:,1) double = []
end
ff = session.find_freeform_elts();
targets = ff(:);
if ~isempty(opts.elts)
    targets = intersect(targets, opts.elts(:));
end
rx_modes = parse_rx_modes_(rx_path, 'nMonZernCoef', 'MonZernModes');
chans = build_channels_(session, targets, rx_modes, opts.modes_per_elt, ...
    @(s, ie, m) macos.channels.MonZernChannel(s, ie, m));
end
