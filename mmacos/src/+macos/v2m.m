function mat = v2m(vec, indx)
%MACOS.V2M  Reconstruct a 2D matrix from m2v's compressed vec + indx.
%   mat = macos.v2m(vec, indx) un-vectorises the result of macos.m2v.
%   Sparse scatter into a full matrix of size indx.size.
m = indx.size(1);
n = indx.size(2);
mat = full(sparse(indx.i, indx.j, vec, m, n));
end
