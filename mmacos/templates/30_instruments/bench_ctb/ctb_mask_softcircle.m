function M = ctb_mask_softcircle(N, dx, r0_m, sigma_m, K)
%CTB_MASK_SOFTCIRCLE  1 inside r0, Gaussian roll-off outside, truncated at
%   r0+4*sigma.  Amplitude apodizer, centred on the beam pixel floor(N/2)
%   (0-based) to match ctb_mask_disk (centering fix).
    if nargin < 5, K = 8; end
    r1 = r0_m + 4*sigma_m;
    base = ctb_mask_disk(N, dx, r1, K);                % hard truncation
    c = floor(N/2); [xx,yy] = meshgrid((0:N-1)-c, (0:N-1)-c);
    rr = hypot(xx, yy) * dx;
    tap = ones(N); out = rr > r0_m;
    tap(out) = exp(-((rr(out)-r0_m)/sigma_m).^2);
    M = base .* tap;
end
