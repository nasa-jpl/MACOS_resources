function set_src_size(aperture, obscuration)
%MACOS.SET_SRC_SIZE  Set source aperture and obscuration.
%   macos.set_src_size(APERTURE, OBSCURATION) sets the beam N.A. (point
%   source) or diameter (collimated) and the central obscuration.  Requires
%   APERTURE>0, APERTURE>OBSCURATION, OBSCURATION>=0.  An omitted OBSCURATION
%   keeps the current one (macos.get_src_size; pymacos src_size(ape) does the
%   same) -- it was 0, which cleared the deck's Obscuration= (veneer-default
%   audit, 2026-10-23).  See also: macos.get_src_size.
arguments
    aperture    (1,1) double {mustBePositive}
    obscuration (1,1) double {mustBeNonnegative} = macos.get_src_size().obscuration
end
mmacos('set_src_size', aperture, obscuration);
end
