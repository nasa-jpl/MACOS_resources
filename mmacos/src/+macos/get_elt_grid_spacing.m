function dx = get_elt_grid_spacing(srf)
%MACOS.GET_ELT_GRID_SPACING  Grid node spacing GridSrfdx (dx==dy) at elt SRF.
%   dx = macos.get_elt_grid_spacing(SRF) returns the grid sampling spacing in
%   BaseUnits.  The grid spans (nGridMat-1)*dx centred on the surface.  Errors
%   if SRF is not a grid surface.  See also: macos.set_elt_grid_spacing.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
dx = mmacos('elt_srf_grid_spacing', srf, 0.0, 0, 1);
end
