function n = grid_size_max()
%MACOS.GRID_SIZE_MAX  Max permitted grid sampling for the current model size.
%   n = macos.grid_size_max() returns mGridMat -- the largest square grid a
%   surface may carry at the initialised model size.  A grid larger than this
%   cannot be loaded/set (see macos.set_elt_grid).
n = double(mmacos('elt_srf_grid_size_max'));
end
