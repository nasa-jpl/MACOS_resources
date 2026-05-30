function out = build_apodised_mask(N, dx_m, aperture_fn, taper_fn, supersample)
%BUILD_APODISED_MASK  N×N amplitude transmission mask with sub-pixel weighting.
%   Mirror of pymacos
%   tests/proper_compare/apodizer.py:build_apodised_mask.
%
%   Output convention: pixel (i, j) is centred at physical coordinate
%   ((j - (N-1)/2)*dx, (i - (N-1)/2)*dx) (x along columns, y along
%   rows — matches macos's and PROPER's FFT-centre layout).
%
%   Args:
%     N            grid size (must match macos's mdttl and PROPER gridsize)
%     dx_m         pixel pitch in metres
%     aperture_fn  function handle f(x, y) returning bool/double
%     taper_fn     []   or function handle f(x, y) returning [0..1]
%     supersample  linear K factor; K=16 → ~0.4% edge accuracy
arguments
    N           (1,1) double {mustBeInteger, mustBePositive}
    dx_m        (1,1) double {mustBePositive}
    aperture_fn (1,1) function_handle
    taper_fn        % may be [] or function_handle
    supersample (1,1) double {mustBeInteger, mustBePositive} = 16
end

K = supersample;
centers     = (0:N-1) - (N - 1) / 2.0;       % (1,N) integer offsets
centers     = centers * dx_m;                 % metres
sub_offsets = ((0:K-1) - (K - 1) / 2.0) * dx_m / K;

out = zeros(N, N);

% Loop rows to keep peak memory at O(K^2 * N) per row.
for i = 1:N
    yc = centers(i);
    sub_y_row = yc + sub_offsets;             % (1,K)
    % Build (K, N, K) — K sub-y × N pixels × K sub-x
    ys = reshape(sub_y_row, [K, 1, 1]);
    xs = reshape(centers, [1, N, 1]) + reshape(sub_offsets, [1, 1, K]);
    ys = repmat(ys, [1, N, K]);
    xs = repmat(xs, [K, 1, 1]);
    mask_hi = double(aperture_fn(xs, ys));
    out(i, :) = mean(mask_hi, [1, 3]);
end

if ~isempty(taper_fn)
    [xc, yc_grid] = meshgrid(centers, centers);  % xy-indexing
    out = out .* double(taper_fn(xc, yc_grid));
end
end
