function ch = ZernChannel(session, iElt, mode)
%MACOS.CHANNELS.ZERNCHANNEL  Factory for a Zern coefficient channel.
%   ch = macos.channels.ZernChannel(SESSION, IELT, MODE) constructs a
%   ZernikeCoefChannel rooted in the ZernCoef array of a Zern-typed
%   element (Surface=Zernike or Surface=ZrnGridData).
%
%   See also: macos.channels.ZernikeCoefChannel,
%             macos.channels.zernike_channels.
ch = macos.channels.ZernikeCoefChannel(session, iElt, mode, 'Zern');
end
