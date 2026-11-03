function [r_lamD, contrast] = radial_contrast(intensity, peak_unaberrated, ...
        lam_over_D_px, max_lambda_over_D, bins_per_lambda_over_D)
%MACOS.RADIAL_CONTRAST  Radially-averaged contrast vs separation in lambda/D.
%   [R_LAMD, C] = RADIAL_CONTRAST(I, PEAK_UNAB, LAMD_PX) returns the
%   azimuthally-averaged contrast curve
%       contrast(r) = mean(intensity in ring at r) / peak_unaberrated
%   with separation R_LAMD expressed in lambda/D.  This is the
%   "dark-zone contrast" form used by the coronagraph literature;
%   normalising by the un-coronagraphed on-axis peak (Strehl-norm)
%   decouples the score from each engine's intensity normalisation.
%
%   MATLAB port of contrast.py:radial_contrast.
%
%   Args:
%     intensity              : (N,N) focal-plane intensity to score.
%     peak_unaberrated       : peak of the no-mask reference PSF.
%     lam_over_D_px          : lambda/D in pixels at this focal plane.
%     max_lambda_over_D      : how far out to score (default 20).
%     bins_per_lambda_over_D : radial bin density (default 4).
    arguments
        intensity              (:,:) double
        peak_unaberrated       (1,1) double
        lam_over_D_px          (1,1) double
        max_lambda_over_D      (1,1) double = 20.0
        bins_per_lambda_over_D (1,1) double = 4
    end

    bin_size_px   = lam_over_D_px / bins_per_lambda_over_D;
    max_radius_px = max_lambda_over_D * lam_over_D_px;
    [r_px, mean_, ~, ~] = macos.radial_profile(intensity, [], max_radius_px, bin_size_px);
    r_lamD   = r_px / lam_over_D_px;
    contrast = mean_ / peak_unaberrated;
end
