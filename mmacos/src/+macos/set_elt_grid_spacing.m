function set_elt_grid_spacing(srf, dx)
%MACOS.SET_ELT_GRID_SPACING  Set grid node spacing GridSrfdx (dx==dy) at SRF.
%   macos.set_elt_grid_spacing(SRF, DX) sets the grid sampling spacing (BaseUnits)
%   and invalidates the cached trace.  Errors if SRF is not a grid surface.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
    dx  (1,1) double {mustBePositive}
end
mmacos('elt_srf_grid_spacing', srf, dx, 1, 1);
end
