function W = opd(opts)
%MACOS.OPD  OPD matrix from the most recent trace (N x N).
%   W = macos.opd() returns the engine OPDMat (N = source-grid sampling).
%   Call macos.trace() first.
%
%   Conventions (see mmacos/doc/opd_conventions.md for the full story):
%     * raw array is OPD(i,j) with FIRST index i = global X, SECOND
%       index j = global Y -- identical in the CLI, mmacos and pymacos;
%     * sign: a ray LONGER than the reference is POSITIVE (optical path
%       difference).  The reference is the chief ray when it survives
%       the trace, else the bundle mean (mean-removed map).
%
%   Name-value options:
%     'orient'  'raw' (default) | 'xy'
%               'raw': the engine array as stored, (i,j) = (X,Y).
%               'xy' : transposed so ROWS run along Y and COLUMNS along
%               X -- the standard image convention.  Display with
%               imagesc(xv, yv, W); axis xy  for an x-right / y-up view
%               matching the CLI plot.  (Equivalent to W_raw.')
%     'sign'    'opl' (default) | 'wavefront'
%               'opl': engine convention, longer path positive.
%               'wavefront': negated -- the interferometer-style
%               wavefront-error map, and the sign PROPER's
%               prop_add_phase expects.
arguments
    opts.orient (1,:) char {mustBeMember(opts.orient, {'raw','xy'})} = 'raw'
    opts.sign   (1,:) char {mustBeMember(opts.sign, {'opl','wavefront'})} = 'opl'
end
W = mmacos('opd');
if strcmp(opts.orient, 'xy')
    W = W.';
end
if strcmp(opts.sign, 'wavefront')
    W = -W;
end
end
