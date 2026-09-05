function chans = freeform_ffzern_channels(session, rx_path, opts)
%MACOS.CHANNELS.FREEFORM_FFZERN_CHANNELS  Build FFZern channels per FF element.
%   chans = macos.channels.freeform_ffzern_channels(SESSION, RX_PATH)
%   returns a cell array of ZernikeCoefChannel handles, one per
%   (FreeForm element, FFZern mode) pair declared in the Rx.
%
%   Name-value pairs: see freeform_monzern_channels (same semantics,
%   the Rx parser looks at nFFZernCoef / FFZernModes instead).
%
%   See also: macos.channels.FFZernChannel,
%             macos.channels.freeform_monzern_channels.
arguments
    session
    rx_path (1,:) char {mustBeNonempty}
    opts.modes_per_elt = []
    opts.elts          (:,1) double = []
end
ff = session.find_freeform_elts();
targets = require_elts(ff(:), opts.elts, 'freeform-ffzern', ...
    @(id) 'not a FreeForm-typed element per the Rx (find_freeform_elts)');
rx_modes = parse_rx_modes_(rx_path, 'nFFZernCoef', 'FFZernModes');
chans = build_channels_(session, targets, rx_modes, opts.modes_per_elt, ...
    @(s, ie, m) macos.channels.FFZernChannel(s, ie, m));
end
