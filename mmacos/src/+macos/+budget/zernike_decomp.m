function [c, B, res] = zernike_decomp(w2d, mask, modes, varargin)
%ZERNIKE_DECOMP  Decompose an OPD map onto Noll Zernike modes over a mask.
%   [C, B, RES] = macos.budget.zernike_decomp(W2D, MASK, MODES) fits the
%   2-D wavefront W2D, over the support MASK, to the Noll Zernike modes
%   MODES (1-based), returning the coefficients C (per mode, same units as
%   W2D), the basis B (Npix x numel(MODES)) and the fit residual RES
%   (Npix x 1).  The basis is built from macos.noll_mode (self-contained;
%   centroid-centred, circumscribing-disk radius), so a per-Zernike-mode
%   error budget can attribute WFE to output modes.
%
%   Options:
%     'conv'   Zernike convention, 'noll' (default; only one supported here
%              -- macos.noll_mode is pure Noll).  'ansi'/'bornwolf' error.
%
%   See also macos.noll_mode, macos.budget.cov_budget.

p = inputParser;
p.addParameter('conv', 'noll', @(s) ischar(s) || isstring(s));
p.parse(varargin{:});
assert(strcmpi(p.Results.conv, 'noll'), 'budget:conv', ...
    'zernike_decomp supports conv=''noll'' only (macos.noll_mode is Noll)');

mask = mask ~= 0;
assert(isequal(size(mask), size(w2d)), 'budget:decomp', ...
    'w2d and mask must be the same size');
idx = find(mask);
B = zeros(numel(idx), numel(modes));
for j = 1:numel(modes)
    Z = macos.noll_mode(mask, modes(j));
    B(:, j) = Z(idx);
end
w = w2d(idx);
c = B \ w;
res = w - B * c;
end
