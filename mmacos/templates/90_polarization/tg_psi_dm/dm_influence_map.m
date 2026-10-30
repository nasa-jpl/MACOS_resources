function [M, info] = dm_influence_map(N, dx, opts)
%DM_INFLUENCE_MAP  A deformable-mirror surface from actuator influence functions.
%   M = DM_INFLUENCE_MAP(N, DX) returns an N x N surface-height map (mm) on a
%   square grid of pitch DX (mm), centred on the DM vertex, built as the
%   superposition of per-actuator influence functions
%
%       M(x,y) = sum_k  c_k * f((x-x_k)/w, (y-y_k)/w)
%
%   with f a normalized Gaussian, w the influence width and c_k the actuator
%   commands.  This is the hardware truth for a DM (see the MACOS DM-surface
%   doctrine: actuator commands are the physical DOFs; a Zernike basis is a
%   convenience, not the mechanism), and it is the form MACOS consumes
%   directly as a GridData figure -- write it with macos.write_grid_file and
%   point a Bench add_mirror('grid_file',...) at the result.
%
%   The returned M is in the engine's GridMat convention (first index = +x,
%   second index = +y), ready for macos.write_grid_file / macos.set_elt_grid
%   with NO transpose.
%
%   Name-value options
%     'nact'      16        actuators across (nact x nact square lattice)
%     'pitch'     3.5       actuator pitch, mm
%     'poke'      50e-6     command amplitude, mm (50 nm)
%     'width'     0.85      influence-function 1/e radius, in PITCH units
%     'pattern'   'checker' 'checker' | 'random' | 'single' | 'zero'
%     'seed'      7         RNG seed for 'random'
%     'act'       []        explicit nact x nact command matrix (mm); when
%                           given it OVERRIDES 'pattern'/'poke'
%
%   The second output INFO carries .act (the command matrix, mm), .xact
%   (actuator centres, mm), .pitch, .width_mm and .span_mm.
%
%   WHY A CHECKERBOARD IS THE DEFAULT.  A random command pattern spreads its
%   power over every spatial frequency the DM can make, so a recovery is
%   judged against a target whose highest frequencies the instrument's pupil
%   imaging cannot carry -- the residual then reports the OPTICS, not the
%   gauge.  The staggered +/- checkerboard is a single spatial frequency at
%   the actuator Nyquist: the hardest pattern the DM can produce, and one
%   whose recovery is unambiguous to read.  Use 'random' to score a realistic
%   command set once the checker case closes.
%
%   See also: macos.write_grid_file, macos.set_elt_grid, macos.read_grid_file.

arguments
    N   (1,1) double {mustBeInteger, mustBePositive}
    dx  (1,1) double {mustBePositive}
    opts.nact    (1,1) double {mustBeInteger, mustBePositive} = 16
    opts.pitch   (1,1) double {mustBePositive} = 3.5
    opts.poke    (1,1) double = 50e-6
    opts.width   (1,1) double {mustBePositive} = 0.85
    opts.pattern (1,:) char {mustBeMember(opts.pattern, ...
                    {'checker','random','single','zero'})} = 'checker'
    opts.seed    (1,1) double = 7
    opts.act     double = []
end

na = opts.nact;
if ~isempty(opts.act)
    assert(isequal(size(opts.act), [na na]), ...
        'dm_influence_map: ''act'' must be %d x %d.', na, na);
    C = opts.act;
else
    switch opts.pattern
    case 'checker'
        [ia, ja] = ndgrid(1:na, 1:na);
        C = opts.poke * (-1).^(ia + ja);
    case 'random'
        rs = RandStream('twister', 'Seed', opts.seed);
        C = opts.poke * (2*rand(rs, na, na) - 1);
    case 'single'
        C = zeros(na);  C(ceil(na/2), ceil(na/2)) = opts.poke;
    case 'zero'
        C = zeros(na);
    end
end

% actuator centres and grid coordinates, both centred on the DM vertex
xact = ((1:na) - (na+1)/2) * opts.pitch;
ax   = ((1:N)  - (N+1)/2)  * dx;
w    = opts.width * opts.pitch;

% ndgrid (NOT meshgrid): X varies along the FIRST index, which is the
% engine's +x -- the GridMat convention macos.write_grid_file expects.
[X, Y] = ndgrid(ax, ax);
M = zeros(N, N);
for ia = 1:na
    for ja = 1:na
        if C(ia, ja) == 0, continue; end
        M = M + C(ia, ja) * exp(-(((X - xact(ia)).^2 + (Y - xact(ja)).^2) / w^2));
    end
end

info = struct('act', C, 'xact', xact, 'pitch', opts.pitch, ...
              'width_mm', w, 'span_mm', (N-1)*dx, 'pattern', opts.pattern);
end
