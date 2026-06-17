function set_src_sampling(n)
%MACOS.SET_SRC_SAMPLING  Resize the source grid (nGridPts <- N).
arguments
    n (1,1) double {mustBeInteger, mustBePositive}
end
mmacos('set_src_sampling', n);
end
