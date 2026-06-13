function [centers, means, stds, ns] = radial_profile(img, center, max_radius, bin_size)
%RADIAL_PROFILE  Azimuthally-averaged radial profile of a 2D image.
%   [centers, means, stds, ns] = RADIAL_PROFILE(IMG) returns the bin
%   centres, per-bin mean, per-bin std, and per-bin pixel count of the
%   azimuthal average of IMG about its array centre.
%
%   MATLAB port of contrast.py:radial_profile (pymacos
%   tests/proper_compare).  Conventions match the Python exactly so
%   the macos<->PROPER contrast baselines carry over:
%     - center defaults to the (N-1)/2 array centre (FFT-shift
%       convention for even N); pixel coordinates are 0-based.
%     - max_radius defaults to half the shorter image dimension.
%     - empty bins are NaN-filled (clean for log-y plotting and to
%       signal "no data" rather than zero intensity).
%
%   Args:
%     img        : 2D array.
%     center     : [cy cx] in 0-based pixel coords (default array centre).
%     max_radius : max radius to bin out to, in pixels.
%     bin_size   : bin width in pixels (default 1).
    arguments
        img        (:,:) double
        center     double = []
        max_radius double = []
        bin_size   (1,1) double = 1.0
    end

    if isempty(center)
        cy = (size(img,1) - 1) / 2.0;
        cx = (size(img,2) - 1) / 2.0;
    else
        cy = center(1);  cx = center(2);
    end
    if isempty(max_radius)
        max_radius = min(size(img)) / 2;
    end

    [ny, nx] = size(img);
    [xx, yy] = meshgrid(0:nx-1, 0:ny-1);   % 0-based, matches np.indices
    rr = hypot(yy - cy, xx - cx);

    % np.arange(0, max_radius + bin_size, bin_size): half-open, excludes
    % the stop value.  Replicate so bin count matches the Python.
    bins = 0 : bin_size : (max_radius + bin_size);
    bins(bins >= max_radius + bin_size) = [];
    n_bins  = numel(bins) - 1;
    centers = 0.5 * (bins(1:end-1) + bins(2:end));

    means = nan(1, n_bins);
    stds  = nan(1, n_bins);
    ns    = zeros(1, n_bins);
    for i = 1:n_bins
        msk = (rr >= bins(i)) & (rr < bins(i+1));
        if any(msk(:))
            vals     = img(msk);
            means(i) = mean(vals);
            stds(i)  = std(vals, 1);   % population std (numpy ddof=0)
            ns(i)    = numel(vals);
        end
    end
end
