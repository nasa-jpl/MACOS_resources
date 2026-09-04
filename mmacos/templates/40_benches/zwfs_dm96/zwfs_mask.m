function [V, D] = zwfs_mask(N, dx, dia_m, phi_rad, ctr, K)
%ZWFS_MASK  Complex focal-plane Zernike-dimple mask, gray-edge supersampled.
%   [V, D] = ZWFS_MASK(N, DX, DIA_M, PHI_RAD, CTR, K) returns an N-by-N
%   COMPLEX mask (and the underlying real disk D, for reference-wave
%   measurement)
%       V = 1 + (exp(i*PHI_RAD) - 1) * D
%   where D is the KxK-supersampled (area-weighted, gray-edged) disk of
%   diameter DIA_M (same units as DX), centred at CTR = [col row]
%   (1-based, may be fractional; default the FFT DC pixel floor(N/2)+1).
%   CENTER THE MASK ON THE MEASURED SPOT: the builder's nominal axis is
%   a straight chief line, but the engine's rays refract through the BS
%   plate (~0.1 mm lateral walk = tens of lambda/D at focus), so the
%   spot does NOT land on the DC pixel -- exactly the alignment a real
%   bench does by translating the mask substrate.  Unit transmission
%   everywhere; pure phase inside the dimple.  Applied via
%   macos.apodize_complex.
%
%   Disk construction copied from bench_ctb/ctb_mask_disk.m
%   (supersampling); the ZWFS wrapper adds the phase + settable center.
%   Hardware numbers (etch depth -> PHI_RAD, spot diameters in lambda/D)
%   live in 40_benches/vsg_wip/vsg2_params.m section 9.
    if nargin < 6, K = 8; end
    if nargin < 5 || isempty(ctr), ctr = (floor(N/2)+1)*[1 1]; end
    r_m = dia_m / 2;
    cx = ctr(1) - 1;  cy = ctr(2) - 1;                 % 0-based, fractional OK
    off = ((0:K-1) - (K-1)/2) / K;                     % sub-pixel offsets
    [ox, oy] = meshgrid(off, off); ox = ox(:).'; oy = oy(:).';
    D = zeros(N);
    for i = 1:N
        yc = (i-1-cy); xs = ((0:N-1)-cx).'; acc = zeros(N,1);
        for s = 1:numel(ox)
            xx = (xs + ox(s)) * dx; yy = (yc + oy(s)) * dx;
            acc = acc + double(xx.^2 + yy.^2 <= r_m^2);
        end
        D(i,:) = acc.' / numel(ox);
    end
    V = 1 + (exp(1i*phi_rad) - 1) * D;
end
