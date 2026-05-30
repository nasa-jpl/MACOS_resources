function W = opd()
%MACOS.OPD  OPD matrix from the most recent trace (N x N, WaveUnits).
%   Returns the OPDMat (N = source-grid sampling).  Call macos.trace()
%   first.
W = mmacos('opd');
end
