function varargout = dmg_camres(varargin)
%DMG_CAMRES  The camera's PIXELS on the model's detector grid: the resolution sweep's
%   resample (cophi_dm96 ressweep, BRIEF_to_resolution_sweep 2026-10-13).
%
%   The model's detector grid point-samples the pupil image (one ray per model
%   pixel, ~3.28 model px per actuator on the TG96 benches).  A camera with
%   COARSER pixels integrates the intensity over each pixel's area.  This
%   operator does exactly that to every detector FRAME, and maps the camera
%   pixels back onto the model grid by replication, so everything downstream --
%   the mask, the placement, the 64-state matrix, the rows -- runs unchanged
%   THROUGH the resampled reading (blur and registration absorbed by calibration,
%   as on the bench).
%
%     dmg_camres('set', cfg)   configure (persistent; one per MATLAB session)
%         cfg.c       model px per camera px (>= 1 for a coarser camera; 1 = identity)
%         cfg.center  model px (1-based, both axes) at the CENTRE of one camera pixel
%                     [floor(N/2)+1]
%         cfg.offset  [dx dy] model px: the camera grid shifted against the image
%                     (the registration row) -- the replication back onto the model
%                     grid stays on the NOMINAL (offset 0) cells, which is what a
%                     shifted camera read through an unshifted calibration does  [0 0]
%     dmg_camres('clear')      back to the identity (every caller unchanged)
%     cfg = dmg_camres('get')  the configuration ([] when clear)
%     dmg_camres('suspend', tf)  suspend / resume the operator WITHOUT clearing it
%                     (the sweep's hook takes the unbinned frames once and bins them
%                     itself, e.g. at several offsets)
%     I = dmg_camres(I)        apply to an image or a stack (planes along dim 3):
%                     area-weighted integration over each camera pixel (fractional
%                     overlaps for a non-integer c), divided by the pixel's area in
%                     model px (so intensities keep their units and the mask
%                     threshold is unchanged), replicated back.  EXACT identity
%                     (returns I itself) when clear, suspended, or c == 1 with zero
%                     offset.
%     [B, T] = dmg_camres('bin', I [, cfg])   the camera pixels themselves (ncell x
%                     ncell x planes, area-averaged) and the operator struct
%     I = dmg_camres('rep', B, T)            back onto the model grid
%
%   Gate: tests/tResolutionSweep (identity, flux, replication, offset).
persistent CFG SUSP OPS
if isempty(SUSP), SUSP = false; end
if nargin >= 1 && ischar(varargin{1})
    switch varargin{1}
        case 'set'
            cfg = varargin{2};
            if ~isfield(cfg, 'offset') || isempty(cfg.offset), cfg.offset = [0 0]; end
            if ~isfield(cfg, 'center'), cfg.center = []; end
            assert(cfg.c > 0, 'dmg_camres: c must be > 0');
            CFG = cfg;  OPS = [];  SUSP = false;
        case 'clear'
            CFG = [];  OPS = [];  SUSP = false;
        case 'get'
            varargout{1} = CFG;
        case 'suspend'
            SUSP = logical(varargin{2});
        case 'bin'
            I = varargin{2};  cfg = CFG;  if nargin >= 3 && ~isempty(varargin{3}), cfg = varargin{3}; end
            T = ops_(cfg, size(I, 1), size(I, 2));
            varargout{1} = binstack_(I, T);  varargout{2} = T;
        case 'rep'
            B = varargin{2};  T = varargin{3};
            varargout{1} = repstack_(B, T);
        otherwise
            error('dmg_camres: unknown command %s', varargin{1});
    end
    return
end
I = varargin{1};
if isempty(CFG) || SUSP || (CFG.c == 1 && all(CFG.offset == 0))
    varargout{1} = I;  return
end
if isempty(OPS) || OPS.ny ~= size(I, 1) || OPS.nx ~= size(I, 2)
    OPS = ops_(CFG, size(I, 1), size(I, 2));
end
varargout{1} = repstack_(binstack_(I, OPS), OPS);
end

% ---------------------------------------------------------------------
function T = ops_(cfg, ny, nx)
%OPS_  the separable operators: W (ncell x n) = the overlap of each camera pixel
%   with each model pixel (model pixel n spans [n-1/2, n+1/2]), R (n x ncell) =
%   the nominal cell each model pixel's CENTRE falls in.
c = cfg.c;  off = cfg.offset;
ctr = cfg.center;  if isempty(ctr), ctr = [floor(nx/2)+1, floor(ny/2)+1]; end
if isscalar(ctr), ctr = [ctr ctr]; end
[Wx, Rx, mx] = axis_(nx, c, ctr(1), off(1));
[Wy, Ry, my] = axis_(ny, c, ctr(2), off(2));
T = struct('Wx', Wx, 'Rx', Rx, 'Wy', Wy, 'Ry', Ry, 'nx', nx, 'ny', ny, ...
    'ncx', size(Wx, 1), 'ncy', size(Wy, 1), 'mx', mx, 'my', my, 'c', c, 'offset', off, 'center', ctr);
end

function [W, R, m] = axis_(n, c, ctr, off)
% cell j (integer, ..., -1, 0, 1, ...) spans [ctr + off + (j - 1/2) c, ctr + off + (j + 1/2) c]
jlo = floor(((1 - 0.5) - ctr - off)/c + 0.5) - 1;
jhi = ceil(((n + 0.5) - ctr - off)/c - 0.5) + 1;
m = (jlo:jhi).';  nc = numel(m);
lo = ctr + off + (m - 0.5)*c;  hi = lo + c;
ii = [];  jj = [];  vv = [];
for q = 1:nc
    p1 = max(1, floor(lo(q) + 0.5));  p2 = min(n, ceil(hi(q) - 0.5));
    for p = p1:p2
        w = min(hi(q), p + 0.5) - max(lo(q), p - 0.5);
        if w > 1e-12, ii(end+1) = q; jj(end+1) = p; vv(end+1) = w/c; end %#ok<AGROW>
    end
end
W = sparse(ii, jj, vv, nc, n);                 % rows sum to (overlap / c): an area AVERAGE per axis
% nominal cell (offset 0) of each model pixel's centre
j0 = floor(((1:n).' - ctr)/c + 0.5);
[tf, loc] = ismember(j0, m);
assert(all(tf), 'dmg_camres: a model pixel falls outside the cell table');
R = sparse((1:n).', loc, 1, n, nc);
end

function B = binstack_(I, T)
np = size(I, 3);  B = zeros(T.ncy, T.ncx, np);
for k = 1:np
    B(:, :, k) = full(T.Wy * I(:, :, k) * T.Wx.');
end
end

function I = repstack_(B, T)
np = size(B, 3);  I = zeros(T.ny, T.nx, np);
for k = 1:np
    I(:, :, k) = full(T.Ry * B(:, :, k) * T.Rx.');
end
end
