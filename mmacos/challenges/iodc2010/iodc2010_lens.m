function L = iodc2010_lens(varargin)
%IODC2010_LENS  A "green lens" prescription: one radius magnitude, a sign per surface.
%   L = iodc2010_lens('R', 51.95, 'sgn', [1 0], 't', [8 NaN], 'glass', {'N-BK7','air'}, ...
%                     'stop', [0 10], 'name', 'singlet')
%
%   Every field is in the problem's units (mm, degrees) and in CODE V's sign
%   convention (light travels +z; a radius is POSITIVE when its centre of
%   curvature lies to the +z side of the vertex).  The MACOS deck emitter
%   (iodc2010_emit) translates to KrElt/psiElt.
%
%   Fields
%     R        the common radius magnitude (mm), one number for the whole lens
%     sgn      1xN, per surface: +1 -> radius +R, -1 -> radius -R, 0 -> plano
%     t        1xN, thickness AFTER surface k (mm).  t(N) is the distance from
%              the last surface to the image plane; NaN = the paraxial focus
%              (solved by iodc2010_paraxial).
%     glass    1xN cellstr, the medium AFTER surface k: 'N-BK7' or 'air'.  The
%              last entry must be 'air' (a real image in air).
%     stop     [z_s a_s]: the stop plane's axial position measured from
%              surface 1's vertex (mm, may be negative or beyond the last
%              surface) and its semi-diameter (mm).  The stop may sit anywhere
%              the problem allows (in air or inside an element); it is a
%              passive plane that only clips.
%     semidia  1xN clear semi-diameters (mm); [] = sized from the footprint at
%              the scan's largest field by iodc2010_score ("no mounting material
%              beyond the clear aperture", the loophole the paper names).
%     name     a label for reports and file names.
%
%   Derived (filled here): z (1xN vertex positions, z(1) = 0), N.
%
%   See also iodc2010_paraxial, iodc2010_emit, iodc2010_score.
p = inputParser;
p.addParameter('R',       NaN, @(x) isscalar(x) && x > 0);
p.addParameter('sgn',     [],  @(x) isnumeric(x) && all(ismember(x, [-1 0 1])));
p.addParameter('t',       [],  @isnumeric);
p.addParameter('glass',   {},  @iscellstr);
p.addParameter('stop',    [],  @(x) isnumeric(x) && numel(x) == 2);
p.addParameter('semidia', [],  @isnumeric);
p.addParameter('name',    'lens', @ischar);
p.parse(varargin{:});
L = p.Results;

N = numel(L.sgn);
assert(N >= 1, 'iodc2010_lens: at least one surface');
assert(numel(L.t) == N, 'iodc2010_lens: t must have one entry per surface (the last = image distance)');
assert(numel(L.glass) == N, 'iodc2010_lens: glass must have one entry per surface');
assert(all(L.t(1:N-1) > 0), 'iodc2010_lens: every thickness before the image must be positive');
assert(isnan(L.t(N)) || L.t(N) > 0, 'iodc2010_lens: the image distance must be positive or NaN (paraxial focus)');
assert(strcmpi(L.glass{N}, 'air'), 'iodc2010_lens: the image must be in air');
assert(~isempty(L.stop) && L.stop(2) > 0, 'iodc2010_lens: stop = [z_s a_s] with a_s > 0');
if ~isempty(L.semidia)
    assert(numel(L.semidia) == N && all(L.semidia > 0), 'iodc2010_lens: semidia must be 1xN positive');
end
L.sgn   = L.sgn(:).';
L.t     = L.t(:).';
L.glass = L.glass(:).';
L.N     = N;
L.z     = [0, cumsum(L.t(1:N-1))];
end
