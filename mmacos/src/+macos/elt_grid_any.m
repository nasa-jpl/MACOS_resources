function tf = elt_grid_any()
%MACOS.ELT_GRID_ANY  True iff the loaded Rx has any grid-data surfaces.
%   Covers every GridData-bearing SrfType (GridData(9), AsGrData(11),
%   MonGrData(12), ZrnGridData(13), FreeForm(14) -- the engine's
%   GridTypeAll set).  See also: macos.find_grid_elts, macos.get_elt_grid.
tf = logical(mmacos('elt_srf_grid_any'));
end
