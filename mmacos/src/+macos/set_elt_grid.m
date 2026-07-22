function set_elt_grid(srf, dx, mat)
%MACOS.SET_ELT_GRID  Replace the grid-data figure at element SRF.
%   macos.set_elt_grid(SRF, DX, MAT) sets the node spacing GridSrfdx=DX and
%   the square MAT (size×size) of displacements from the nominal shape, and
%   invalidates the cached trace.  MAT must be square with size in [3,
%   macos.grid_size_max()].  Orientation matches macos.get_elt_grid /
%   elt_grid_add (first index along +x, ndgrid convention).
%
%   This REPLACES the surface's grid; use macos.elt_grid_add to accumulate.
%   Errors if SRF is not a grid surface.  See also: macos.get_elt_grid.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
    dx  (1,1) double {mustBePositive}
    mat (:,:) double
end
[ny, nx] = size(mat);
if nx ~= ny
    error('macos:set_elt_grid:notSquare', ...
          'Grid must be square; got %d x %d.', ny, nx);
end
mmacos('elt_srf_grid_data', srf, dx, mat, 1, nx, ny);
end
