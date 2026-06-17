function set_src_flux(flux)
%MACOS.SET_SRC_FLUX  Set the source flux.
%   Propagated intensity scales linearly with the source flux (each ray
%   amplitude is seeded as sqrt(flux)).  Set a small flux on an off-axis
%   source to inject a faint "planet" alongside the on-axis star, then
%   macos.compose / the COMPOSE primitives add the two scenes onto one
%   detector image.  Takes effect on the next trace/propagation.
arguments
    flux (1,1) double {mustBeNonnegative}
end
mmacos('src_flux', flux, 1);
end
