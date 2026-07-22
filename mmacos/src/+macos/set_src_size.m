function set_src_size(aperture, obscuration)
%MACOS.SET_SRC_SIZE  Set source aperture and obscuration.
%   macos.set_src_size(APERTURE, OBSCURATION) sets the beam N.A. (point
%   source) or diameter (collimated) and the central obscuration.  Requires
%   APERTURE>0, APERTURE>OBSCURATION, OBSCURATION>=0.  OBSCURATION defaults
%   to 0.  See also: macos.get_src_size.
arguments
    aperture    (1,1) double {mustBePositive}
    obscuration (1,1) double {mustBeNonnegative} = 0
end
mmacos('set_src_size', aperture, obscuration);
end
