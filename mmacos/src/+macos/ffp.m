function ffp(place_elt, offset)
%MACOS.FFP  Place an off-axis field point by DIRECTION COSINES (angle on sky).
%   macos.ffp(PLACE_ELT, OFFSET) tilts the source so the image at element
%   PLACE_ELT lands at the off-axis field point OFFSET = [dx dy], given as
%   DIRECTION COSINES (normalized; ~= field angle in radians for small
%   angles).  This is the "angle on the sky" way to place a planet --
%   contrast with macos.pfp, which positions in focal-plane PIXELS.  The
%   two inputs differ by the plate scale (see first_order_properties:
%   plate_px_rad, rad/pixel):
%       pixels = direction_cosine / plate_px_rad
%
%   To place a source N lambda/D off-axis, use the SYSPROP tilt directly:
%       p = macos.first_order_properties(place_elt);
%       macos.ffp(place_elt, N * p.lamD_rad * [1 0]);   % +x by N lambda/D
%
%   Requires the system stop to be set first (macos.stop / stop_info_set).
%   FFP changes the source pointing, so reset Return / exit-pupil
%   reference surfaces afterwards (macos.ors / macos.fex), as the
%   CoroExample recipe does.  Enable macos.window(...) beforehand so the
%   off-axis PSF lands at its true offset on a COMPOSE grid rather than
%   re-centred.
%
%   See also: macos.pfp, macos.window, macos.compose, macos.first_order_properties.
arguments
    place_elt (1,1) double {mustBePositive, mustBeInteger}
    offset    (1,2) double
end
mmacos('ffp', place_elt, offset(1), offset(2));
end
