function scale_elt_grid(srf, factor)
%MACOS.SCALE_ELT_GRID  Scale the grid-data figure at element SRF in place.
%   macos.scale_elt_grid(SRF, FACTOR) multiplies every grid node value by the
%   scalar FACTOR and invalidates the cached trace.  Errors if SRF is not a
%   grid surface.  See also: macos.get_elt_grid, macos.set_elt_grid.
arguments
    srf    (1,1) double {mustBeInteger, mustBePositive}
    factor (1,1) double
end
mmacos('elt_srf_grid_data_scale', srf, factor, 1);
end
