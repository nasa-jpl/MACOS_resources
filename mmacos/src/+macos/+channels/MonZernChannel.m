function ch = MonZernChannel(session, iElt, mode)
%MACOS.CHANNELS.MONZERNCHANNEL  Factory for a MonZern coefficient channel.
%   ch = macos.channels.MonZernChannel(SESSION, IELT, MODE) constructs a
%   ZernikeCoefChannel rooted in the MonZernCoef array of a FreeForm
%   element.  Thin convenience wrapper over the underlying classdef
%   constructor.
%
%   See also: macos.channels.ZernikeCoefChannel,
%             macos.channels.freeform_monzern_channels.
ch = macos.channels.ZernikeCoefChannel(session, iElt, mode, 'MonZern');
end
