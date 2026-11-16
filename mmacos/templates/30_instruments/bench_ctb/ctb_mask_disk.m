function M = ctb_mask_disk(N, dx, r_m, K)
%CTB_MASK_DISK  KxK supersampled binary disk, radius r_m (metres), centred
%   on the BEAM pixel floor(N/2) (0-based) = the FFT DC pixel where MACOS's
%   FarField/NF2 focus lands (1-based N/2+1).  The old builders centred on
%   (N-1)/2, half a pixel low -> an off-centre occulter that leaks an
%   asymmetric residual; this is the centering fix.
    if nargin < 4, K = 8; end
    c = floor(N/2);                                    % 0-based beam pixel
    off = ((0:K-1) - (K-1)/2) / K;                     % sub-pixel offsets
    [ox, oy] = meshgrid(off, off); ox = ox(:).'; oy = oy(:).';
    M = zeros(N);
    for i = 1:N
        yc = (i-1-c); xs = ((0:N-1)-c).'; acc = zeros(N,1);
        for s = 1:numel(ox)
            xx = (xs + ox(s)) * dx; yy = (yc + oy(s)) * dx;
            acc = acc + double(xx.^2 + yy.^2 <= r_m^2);
        end
        M(i,:) = acc.' / numel(ox);
    end
end
