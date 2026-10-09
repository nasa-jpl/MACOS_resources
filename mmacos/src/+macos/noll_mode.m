function [zern, m, n] = noll_mode(pupil, imode)
%NOLL_MODE  Noll Zernike mode over an arbitrary pupil support (self-contained).
%   [Z, M, N] = NOLL_MODE(PUPIL, IMODE) evaluates Noll mode IMODE (1-based;
%   Noll 1976 ordering: even index -> cosine, odd -> sine) on the support of
%   PUPIL (any nonzero = inside), normalized sqrt(n+1) (x sqrt(2) for m>0)
%   over the unit disk that circumscribes the support.
%
%   Drop-in replacement for the JPL-internal zernike_mode.m so mmacos is
%   self-contained for distribution (Luis, 2026-09-03).  Semantics are
%   replicated EXACTLY, including its conventions: the support center is the
%   MASK CENTROID; the radial unit is the mask's maximum pixel excursion
%   from that centroid; the azimuth is the NEGATED atan2(y, x) of pixel
%   coordinates (rows = y).  Verified against the original at 1e-9 class on
%   circular, annular and off-center supports, modes 1..15.
    mask = pupil ~= 0;
    [rows, cols] = size(mask);
    npx = nnz(mask);
    xc = floor(cols/2) + 1;  yc = floor(rows/2) + 1;
    if npx ~= 0
        xc = (sum(mask, 1) * (1:cols)') / npx;
        yc = (sum(mask, 2)' * (1:rows)') / npx;
    end
    [yy, xx] = find(mask);
    rmax = max(hypot(xx - xc, yy - yc));

    % Noll index -> (n, m):
    n = 0;
    while (n+1)*(n+2)/2 < imode, n = n + 1; end
    k = imode - n*(n+1)/2 - 1;                    % 0-based rank within order
    if mod(n,2) == 0, m = 2*floor((k+1)/2);
    else,             m = 2*floor(k/2) + 1;
    end

    [x, y] = meshgrid(1:cols, 1:rows);
    x = x - xc;  y = y - yc;
    r = hypot(x, y) / rmax;
    a = -atan2(y, x);

    R = zeros(rows, cols);
    for s = 0:((n-m)/2)
        Rk = (-1)^s * factorial(n-s) / factorial(s) ...
             / factorial((n+m)/2 - s) / factorial((n-m)/2 - s);
        R = R + Rk * r.^(n - 2*s);
    end
    zern = sqrt(n+1) * R;
    if m > 0
        if mod(imode,2) == 0, zern = zern .* cos(m*a) * sqrt(2);
        else,                 zern = zern .* sin(m*a) * sqrt(2);
        end
    end
    zern = zern .* mask;
end
