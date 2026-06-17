function pfp(place_elt, pix_size, offset)
%MACOS.PFP  Place an off-axis field point in focal-plane PIXELS.
%   macos.pfp(PLACE_ELT, PIX_SIZE, OFFSET) tilts the source so the image at
%   element PLACE_ELT lands at pixel position OFFSET = [dx dy] on a grid of
%   pitch PIX_SIZE (BaseUnits).  This is the focal-plane-pixel sibling of
%   macos.ffp (which places by direction cosines / sky angle); the two
%   inputs differ by the plate scale (first_order_properties.plate_px_rad,
%   rad/pixel):
%       direction_cosine = pixels * plate_px_rad
%
%   Match PIX_SIZE to the COMPOSE pixel pitch (and macos.window sizPix) so
%   the placement lands on the same grid the composite is assembled on.
%
%   Requires the system stop to be set first (macos.stop / stop_info_set).
%   PFP changes the source pointing, so reset Return / exit-pupil
%   reference surfaces afterwards (macos.ors / macos.fex), as the
%   CoroExample recipe does.  Enable macos.window(...) beforehand so the
%   off-axis PSF lands at its true offset on a COMPOSE grid.
%
%   See also: macos.ffp, macos.window, macos.compose, macos.first_order_properties.
arguments
    place_elt (1,1) double {mustBePositive, mustBeInteger}
    pix_size  (1,1) double {mustBePositive}
    offset    (1,2) double
end
mmacos('pfp', place_elt, pix_size, offset(1), offset(2));
end
