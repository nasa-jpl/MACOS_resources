function s = get_elt_grid(srf)
%MACOS.GET_ELT_GRID  Read the grid-data surface figure at element SRF.
%   s = macos.get_elt_grid(SRF) returns a struct:
%       .size  scalar   grid dimension nGridMat (square)
%       .dx    scalar   node spacing GridSrfdx (BaseUnits, dx==dy)
%       .mat   size×size grid displacements from the nominal shape (BaseUnits)
%
%   Orientation matches macos.elt_grid_add / write_grid_file: the FIRST
%   (row) index runs along +x (ndgrid convention, NOT meshgrid).  The grid
%   spans (size-1)*dx centred on the surface vertex.
%
%   Errors if SRF is not a grid-bearing surface.  See also:
%   macos.set_elt_grid, macos.elt_grid_add, macos.get_elt_grid_spacing.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
sz = double(mmacos('elt_srf_grid_size', srf, 1));
if sz <= 0
    error('macos:get_elt_grid:notGrid', ...
          'Element %d carries no grid surface (grid size %d).', srf, sz);
end
[dx, mat] = mmacos('elt_srf_grid_data', srf, 0.0, zeros(sz, sz), 0, sz, sz);
s.size = sz;
s.dx   = dx;
s.mat  = mat;
end
