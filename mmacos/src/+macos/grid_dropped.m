function n = grid_dropped()
%MACOS.GRID_DROPPED  Grid files the last load declined (engine 2026-10-10).
%   n = macos.grid_dropped() counts the GridFile= grids larger than this
%   model's mGridMat (macos_param.txt): they cannot be stored, so the load
%   declined them by name and those surfaces trace WITHOUT their figure.
%   Nonzero means a harvest at this model is missing figure data -- load at
%   a model whose mGridMat fits (e.g. a 256-point grid needs model 256).
%   Reset by every load.
%
%   See also: macos.grid_idx_ovf, macos.load_rx.
n = double(mmacos('grid_dropped_get'));
end
