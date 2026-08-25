function V = ctb_mask_vortex(N, m, K)
%CTB_MASK_VORTEX  Charge-m scalar vortex, complex-binned (gray core).
%   V = CTB_MASK_VORTEX(N, M, K) returns the N-by-N complex transmission
%   of a charge-M scalar vortex exp(i*M*theta), GENERATED AT K-x SUB-PIXEL
%   RESOLUTION AND COMPLEX-AVERAGED down to the model grid (default K=8).
%   Centred on the beam pixel floor(N/2) (0-based) = the FFT DC pixel,
%   like ctb_mask_disk (centering fix).
%
%   Why binned: a directly-sampled vortex floors the dark zone at the
%   SINGULAR CORE -- near the axis the phase M*theta wraps faster than
%   the grid, and the mis-phased pixels sit exactly on the stellar Airy
%   peak, scattering ~0.2% of the starlight back inside the Lyot.
%   Complex-averaging the K^2 sub-samples makes those phasors cancel:
%   the binned mask has |V|=0 at the core pixel and a smooth ~1-px
%   amplitude taper -- the physically correct pixel-averaged
%   transmittance, with no hard edge of its own.  Measured on the ideal
%   clear pupil (N=1024, Lyot 0.90, 3-15 lambda/D): direct 3.0e-7,
%   4x-binned 3.4e-9, 8x-binned 3.0e-9 (converged); an explicit opaque
%   core dot is WORSE (9.8e-9 -- its own edge diffracts).
%
%   Memory: the K-x grid is never materialized -- the K^2 sub-pixel
%   shifts are accumulated at model resolution (O(N^2), ~2 s at N=1024,
%   K=8).
%
%   K=1 reproduces the legacy direct-sampled mask (singular pixel set to
%   transmission 1) for A/B comparisons.
%
%   See also: ctb_mask_disk, ctb_mask_phase, ctb_vortex, ctb_vortex_matched.
    if nargin < 3, K = 8; end
    c = floor(N/2);                                    % 0-based beam pixel
    [xx, yy] = meshgrid((0:N-1) - c, (0:N-1) - c);
    if K == 1
        V = exp(1i * m * atan2(yy, xx));
        V(c+1, c+1) = 1;                               % singular pixel (1-based)
        return
    end
    off = ((0:K-1) - (K-1)/2) / K;                     % sub-pixel offsets
    V = zeros(N);
    for a = 1:K
        for b = 1:K
            V = V + exp(1i * m * atan2(yy + off(b), xx + off(a)));
        end
    end
    V = V / K^2;
end
