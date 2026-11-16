function spec = band_limited_circle(r0)
%BAND_LIMITED_CIRCLE  Spec for a band-limited circular aperture.
%   Used as input to apodizer.build_band_limited_mask.
arguments
    r0 (1,1) double {mustBeNonnegative}
end
spec.r0 = r0;
end
