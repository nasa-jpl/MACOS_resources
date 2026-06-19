function r_null = first_airy_null(intensity, search_min_px, search_max_px, ...
                                  bin_size, null_max_fraction_of_peak)
%FIRST_AIRY_NULL  First Airy null radius (pixels) of a centred PSF.
%   R = FIRST_AIRY_NULL(INTENSITY) walks outward from the centre of the
%   radial profile and returns the radius of the first interior local
%   minimum whose value is below NULL_MAX_FRACTION_OF_PEAK times the
%   central peak.  The fractional-depth guard excludes spurious
%   sub-pixel minima on the steep falling slope of the central peak.
%   Returns [] (empty) if no qualifying null is found in
%   [search_min_px, search_max_px].
%
%   MATLAB port of contrast.py:first_airy_null.
    arguments
        intensity                 (:,:) double
        search_min_px             (1,1) double = 3.0
        search_max_px             (1,1) double = 60.0
        bin_size                  (1,1) double = 1.0
        null_max_fraction_of_peak (1,1) double = 0.05
    end

    [r, mean_, ~, ~] = radial_profile(intensity, [], search_max_px, bin_size);
    peak = max(mean_(isfinite(mean_)));
    if isempty(peak) || peak <= 0 || ~isfinite(peak)
        r_null = [];
        return
    end
    threshold = peak * null_max_fraction_of_peak;

    r_null = [];
    for i = 2:numel(mean_)-1
        if ~(isfinite(mean_(i-1)) && isfinite(mean_(i)) && isfinite(mean_(i+1)))
            continue
        end
        if r(i) < search_min_px
            continue
        end
        if mean_(i) < mean_(i-1) && mean_(i) < mean_(i+1) && mean_(i) < threshold
            r_null = r(i);
            return
        end
    end
end
