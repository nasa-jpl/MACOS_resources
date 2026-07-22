function n = get_elt_grid_size(srf)
%MACOS.GET_ELT_GRID_SIZE  Grid sampling (nGridMat) at element SRF.
%   n = macos.get_elt_grid_size(SRF) returns the square grid dimension for a
%   grid-bearing surface, or -1 if SRF carries no grid.  See also:
%   macos.get_elt_grid, macos.find_grid_elts.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
n = double(mmacos('elt_srf_grid_size', srf, 1));
end
