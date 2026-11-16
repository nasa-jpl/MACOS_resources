function lamD = lambda_over_D_pixels(unaberrated_psf, search_min_px, search_max_px)
%MACOS.LAMBDA_OVER_D_PIXELS  lambda/D in pixels, derived from an Airy PSF.
%   LAMD = LAMBDA_OVER_D_PIXELS(PSF) returns lambda/D in pixel units by
%   locating the first Airy null of a centred un-coronagraphed PSF
%   (first null at 1.22 lambda/D for a circular pupil).  This is the
%   EMPIRICAL conversion that avoids having to compute the effective
%   pupil diameter at the science focal plane analytically (it depends
%   on the full prescription's magnification chain).  Works as long as
%   the no-mask PSF is approximately Airy-like.
%
%   MATLAB port of contrast.py:lambda_over_D_pixels.
    arguments
        unaberrated_psf (:,:) double
        search_min_px   (1,1) double = 3.0
        search_max_px   (1,1) double = 40.0
    end

    r_null = macos.first_airy_null(unaberrated_psf, search_min_px, search_max_px);
    if isempty(r_null)
        error('lambda_over_D_pixels:noNull', ...
            ['no Airy null found in radius range [%g, %g] px -- check ', ...
             'that the input is a centred un-coronagraphed PSF'], ...
            search_min_px, search_max_px);
    end
    lamD = r_null / 1.22;
end
