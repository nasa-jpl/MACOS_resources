function ch = FFZernChannel(session, iElt, mode)
%MACOS.CHANNELS.FFZERNCHANNEL  Factory for an FFZern coefficient channel.
%   ch = macos.channels.FFZernChannel(SESSION, IELT, MODE) constructs a
%   ZernikeCoefChannel rooted in the FFZernCoef array of a FreeForm
%   element.
%
%   See also: macos.channels.ZernikeCoefChannel,
%             macos.channels.freeform_ffzern_channels.
ch = macos.channels.ZernikeCoefChannel(session, iElt, mode, 'FFZern');
end
