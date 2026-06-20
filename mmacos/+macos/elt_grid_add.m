function elt_grid_add(srf, grid_dz)
%MACOS.ELT_GRID_ADD  Add a z-displacement grid to a grid surface.
%   macos.elt_grid_add(SRF, GRID_DZ) ADDS the [N x N] displacement matrix
%   GRID_DZ (along the surface's local z-axis) to the existing grid-surface
%   data on element SRF.  Use it to perturb an optic through its grid-data
%   description (e.g. a measured/figure-error map or a DM influence sum).
%
%   N must equal the element's current grid sampling size and SRF must
%   already carry a grid surface (SrfType Grid / MonGrData / FreeForm with
%   a grid term).  GRID_DZ is indexed (y-row, x-col), running -Y..+Y by
%   -X..+X -- the same orientation as macos.zrn_freeform's grid.mat, and
%   no transpose is needed (MATLAB and the Fortran grid are both
%   column-major).
%
%   Mirrors pymacos's elt_grid_add (Luis, 2026-06-20).
%
%   See also: macos.zrn_freeform.
    arguments
        srf     (1,1) double {mustBeInteger, mustBePositive}
        grid_dz (:,:) double {mustBeReal, mustBeFinite}
    end

    % current grid sampling on this surface (<= 0 when no grid is defined)
    n = double(mmacos('elt_srf_grid_size', srf, 1));
    if n < 3
        error('macos:elt_grid_add:noGrid', ...
            'element %d has no grid surface defined (grid size = %d).', srf, n);
    end
    if size(grid_dz,1) ~= size(grid_dz,2)
        error('macos:elt_grid_add:notSquare', ...
            'grid_dz must be a square [N x N] matrix (got [%d x %d]).', ...
            size(grid_dz,1), size(grid_dz,2));
    end
    if size(grid_dz,1) ~= n
        error('macos:elt_grid_add:sizeMismatch', ...
            'grid_dz is [%d x %d] but the surface grid size is %d.', ...
            size(grid_dz,1), size(grid_dz,2), n);
    end

    Nx = size(grid_dz, 2);
    Ny = size(grid_dz, 1);
    % Fortran wrapper raises (mexErrMsgTxt) on failure; no return value.
    mmacos('elt_srf_grid_data_add', srf, grid_dz, Nx, Ny);
end
