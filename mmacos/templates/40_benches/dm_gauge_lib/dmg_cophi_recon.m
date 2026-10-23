function [phi, amp, dc] = dmg_cophi_recon(Fr, alpha)
%DMG_COPHI_RECON  Phase from N heterodyne buckets by least squares.
%   phi = dmg_cophi_recon(Fr, alpha) fits every pixel of the N frames
%   Fr(:,:,k) to the model
%
%       I_k = dc + b cos(alpha_k) + c sin(alpha_k),    phi = atan2(c, b)
%
%   where alpha_k are the bucket-centre phases the frames were taken at
%   (dmg_cophi_frames' second output).  For N = 4 at alpha = D/2 + {0, pi/2,
%   pi, 3pi/2} this is EXACTLY the four-step atan2(I4 - I2, I1 - I3) about the
%   bucket centres (gate tCophi asserts it); for N > 4 it is the N-frame
%   least-squares algorithm Feng's backup slides give (the Zygo four-step is
%   its N = 4 case), the known cure for step error.  N >= 3 is required.
%   amp = sqrt(b^2 + c^2) (the fringe amplitude I0 g sinc(D/2)), dc the mean.
%   Pure MATLAB; the surface is phi * LAM / (4 pi) about a frozen null, as in
%   dmg_ifo_gauge.  (COPHI campaign, 2026-10-11.)
%   alpha = 'iq': a lock-in I/Q sample [dc, I, Q] (dmg_cophi_frames opt.iq) is
%   read directly, phi = atan2(Q, I), amp = hypot(I, Q) = 2|X| (2026-10-12).
if ischar(alpha) && strcmp(alpha, 'iq')
    % a lock-in I/Q sample [dc, I, Q] (dmg_cophi_frames opt.iq): no fit, no bucket
    dc = Fr(:,:,1);  b = Fr(:,:,2);  c = Fr(:,:,3);
    phi = atan2(c, b);  amp = hypot(b, c);
    return
end
N = numel(alpha);
assert(size(Fr, 3) == N && N >= 3, 'dmg_cophi_recon: N frames and N phases, N >= 3');
A = [ones(N,1), cos(alpha(:)), sin(alpha(:))];           % N x 3 design
Pinv = pinv(A);                                           % 3 x N, the same for every pixel
[ny, nx, ~] = size(Fr);
F = reshape(Fr, ny*nx, N).';                              % N x npix
X = Pinv * F;                                             % 3 x npix: [dc; b; c]
dc  = reshape(X(1,:), ny, nx);
b   = reshape(X(2,:), ny, nx);
c   = reshape(X(3,:), ny, nx);
phi = atan2(c, b);
amp = hypot(b, c);
end
