function mask = build_band_limited_mask(N, dx, spec)
%BUILD_BAND_LIMITED_MASK  Band-limited apodisation mask via analytic FT.
%   Mirror of pymacos apodizer.build_band_limited_mask.
%
%   The 2D Fourier transform of a unit-amplitude disc of radius r0
%   centred at the origin is the Airy-like:
%       F(k) = r0 * J_1(2*pi*r0*k) / k    for k > 0
%       F(0) = pi * r0^2                   (disc area)
%   Sample F at the FFT grid's k values, ifft to real space → N×N mask
%   whose pixel values are the exact band-limited representation of the
%   continuous disc.
%
%   Args:
%     N     grid size
%     dx    pixel pitch (m)
%     spec  struct with .r0 (m) — currently the only shape supported,
%           created by `apodizer.band_limited_circle(r0)`
arguments
    N    (1,1) double {mustBeInteger, mustBePositive}
    dx   (1,1) double {mustBePositive}
    spec (1,1) struct
end
if ~isfield(spec, 'r0')
    error('build_band_limited_mask:badSpec', ...
        'spec must have a .r0 field (use apodizer.band_limited_circle).');
end
r0 = spec.r0;

% k-axis in cycles per metre, fft-ordered.  MATLAB's fft uses the same
% layout as numpy.fft.fftfreq for ifft purposes when we then fftshift
% the result.
%   k_axis = [0, +1/(N dx), ..., +(N/2-1)/(N dx),
%             -N/(2 N dx), ..., -1/(N dx)]
k_axis = (mod((0:N-1) + floor(N/2), N) - floor(N/2)) / (N * dx);
[kx, ky] = meshgrid(k_axis, k_axis);
k = hypot(kx, ky);

% Analytic FT of a unit-amplitude disc of radius r0.
F = zeros(N, N);
nz = k > 0;
F(nz)  = r0 ./ k(nz) .* besselj(1, 2 * pi * r0 * k(nz));
F(~nz) = pi * r0^2;

% MATLAB's ifft2 has 1/(N*M) normalisation (same as NumPy).  The
% continuous-IFT ↔ discrete-grid relation requires multiplying by
% N^2 and the sampling factor 1/(N*dx)^2; net: divide by dx^2.
mask_fft_ordered = real(ifft2(F)) / (dx * dx);
% Shift array centre from (1, 1) to ((N+1)/2, (N+1)/2).
mask = fftshift(mask_fft_ordered);
end
