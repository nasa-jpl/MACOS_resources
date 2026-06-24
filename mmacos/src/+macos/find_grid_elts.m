function g = find_grid_elts()
%MACOS.FIND_GRID_ELTS  Indices of grid-bearing elements (ANY GridData type).
%   g = macos.find_grid_elts() returns a column vector of 1-based element ids
%   for every surface that carries a grid-data component (grid size > 0) in the
%   loaded prescription, in element order.  Empty if none.
%
%   Eligibility is discovered by grid SIZE (nGridMat>0), NOT by SrfType, so
%   every GridData-enabled type is included -- GridData(9), AsGrData(11),
%   MonGrData(12), ZrnGridData(13), FreeForm(14) (the engine's GridTypeAll
%   set).  This is the eligibility set for grid-data sensitivity channels
%   (dw_dgrid); a grid perturbation is ADDED to GridMat in place, so the
%   element keeps its original SrfType (and any conic / Zernike / monomial
%   components of a composite surface).
%
%   See also: macos.find_freeform_elts, macos.dw_dgrid,
%             macos.channels.grid_channels.
n = macos.num_elt();
if n <= 0
    g = zeros(0,1);
    return
end
g = zeros(0,1);
for e = 1:n
    if double(mmacos('elt_srf_grid_size', e, 1)) > 0
        g(end+1,1) = e; %#ok<AGROW>
    end
end
end
