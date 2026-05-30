function fn = gaussian_edge_taper(r0, sigma)
%GAUSSIAN_EDGE_TAPER  Soft Gaussian roll-off outside radius r0.
%   T(r) = 1                          for r < r0
%   T(r) = exp(-((r-r0)/sigma)^2)     for r >= r0
%
%   Pairs naturally with apodizer.circle(r1) where r1 = r0 + ~4*sigma
%   to truncate the Gaussian tail.
arguments
    r0    (1,1) double {mustBeNonnegative}
    sigma (1,1) double {mustBePositive}
end
fn = @(x, y) compute_taper(x, y, r0, sigma);
end

function out = compute_taper(x, y, r0, sigma)
r   = sqrt(x.*x + y.*y);
out = ones(size(r));
edge = r > r0;
out(edge) = exp(-((r(edge) - r0) / sigma).^2);
end
