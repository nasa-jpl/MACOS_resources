function s = get_src_size()
%MACOS.GET_SRC_SIZE  Source aperture and obscuration.
%   s = macos.get_src_size() returns a struct:
%       .aperture     scalar   beam N.A. (point source) or diameter (collimated)
%       .obscuration  scalar   central obscuration, same units as .aperture
%
%   Collimated diameters are in BaseUnits; point-source values are numerical
%   apertures.  See also: macos.set_src_size, macos.is_point_source.
[ape, obs] = mmacos('get_src_size');
s.aperture    = ape;
s.obscuration = obs;
end
