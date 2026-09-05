function chans = zernike_channels(session, rx_path, opts)
%MACOS.CHANNELS.ZERNIKE_CHANNELS  Build Zern channels per Zern-typed element.
%   chans = macos.channels.zernike_channels(SESSION, RX_PATH) returns
%   a cell array of ZernikeCoefChannel handles, one per
%   (Zern-typed element, Zern mode) pair declared in the Rx.
%
%   Eligibility set is parsed from the Rx text via find_zern_elts
%   (no mex query for "find Zernike-typed elements" exists yet).
%
%   Name-value pairs: see freeform_monzern_channels (Rx parser looks
%   at nZernCoef / ZernModes here).
%
%   See also: macos.channels.ZernChannel,
%             macos.channels.freeform_monzern_channels.
arguments
    session
    rx_path (1,:) char {mustBeNonempty}
    opts.modes_per_elt = []
    opts.elts          (:,1) double = []
end
ze = session.find_zern_elts(rx_path);
targets = require_elts(ze(:), opts.elts, 'zernike', ...
    @(id) 'no Zernike surface/modes on this element per the Rx (find_zern_elts)');
rx_modes = parse_rx_modes_(rx_path, 'nZernCoef', 'ZernModes');
chans = build_channels_(session, targets, rx_modes, opts.modes_per_elt, ...
    @(s, ie, m) macos.channels.ZernChannel(s, ie, m));
end
