function [vec, jndx] = m2v(mat, indx)
%MACOS.M2V  Vectorise a 2D OPD matrix preserving non-zero index map.
%   [vec, indx] = macos.m2v(mat)   first call -- find non-zeros + cache.
%   vec        = macos.m2v(mat, indx)  reuse cached index struct.
%
%   indx is a struct with fields:
%       .i      Nnz×1 row indices of non-zero positions
%       .j      Nnz×1 col indices of non-zero positions
%       .size   1×2 [nrows, ncols] of the original matrix
%
%   For an OPD map, build indx from the VALID-RAY MASK, not the map:
%       [~, indx] = macos.m2v(double(macos.opd_mask()));  w = macos.m2v(W, indx);
%   -- under the chief-ray reference (the default since 2026-10-10) a valid
%   ray at the chief's path reads exactly 0 and the one-argument form on
%   the map would drop it.
%   Pair with macos.v2m to round-trip back to the original NxM canvas.
%   Used by dw_dz_zernike / dw_dx multi-field stack outputs.
if nargin == 1
    [i, j, vec] = find(mat);
    jndx.i = i;
    jndx.j = j;
    jndx.size = size(mat);
elseif nargin == 2
    jndx = indx;
    k = sub2ind(indx.size, indx.i, indx.j);
    vec = mat(k);
end
end
