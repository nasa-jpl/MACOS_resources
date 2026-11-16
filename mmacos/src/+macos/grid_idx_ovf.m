function n = grid_idx_ovf()
%MACOS.GRID_IDX_OVF  Grid-index overflow samples rejected by the last trace.
%   n = macos.grid_idx_ovf() is the engine's nGridIdxOvf (macos db9236b,
%   api grid_idx_ovf_get): the number of grid-term samples whose pixel
%   index xi/yj was non-finite or >= 2e9 and was therefore treated as
%   OFF the grid (fh = 0) instead of indexing GridMat out of bounds --
%   before that guard an INT32 wrap passed the bounds test and killed the
%   host.  Reset at the start of every trace / propagation.  Nonzero means
%   a grid surface's pitch or the ray solve is wrong (a lost NSCount= hit
%   budget is the known cause); the trace completes and the WARN line
%   reports the count.
%
%   See also: macos.trace.
n = double(mmacos('grid_idx_ovf_get'));
end
