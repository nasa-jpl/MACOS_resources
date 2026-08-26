function [Vc, Vs] = ctb_mask_vvc(N, m, K)
%CTB_MASK_VVC  Vector-vortex mask entries, complex-binned (gray core).
%   [Vc, Vs] = CTB_MASK_VVC(N, M, K) returns the two azimuthal maps a
%   charge-M vector vortex is built from, generated at K-x sub-pixel
%   resolution and averaged to the model grid (default K=8, the
%   ctb_mask_vortex rule):
%       Vc = binned cos(M*theta),   Vs = binned sin(M*theta)
%   centred on the beam pixel floor(N/2) (0-based).
%
%   A vector vortex is a half-wave plate whose fast axis rotates as
%   M*theta/2.  With retardance delta its focal-plane Jones matrix is
%       J(theta) = e^{i delta/2} [ cos(delta/2) I  -  i sin(delta/2) M ]
%       M(theta) = [ cos M*theta   sin M*theta
%                    sin M*theta  -cos M*theta ]
%   so the four entry masks are alpha +/- beta*Vc and beta*Vs with
%   alpha = e^{i delta/2} cos(delta/2), beta = -i e^{i delta/2}
%   sin(delta/2).  delta = pi is the ideal plate (pure vortex term);
%   a zero-order plate has delta(lambda) = pi*lambda0/lambda, and the
%   alpha leakage term -- starlight with NO spiral -- is the classic
%   chromatic limitation this family maps.
%
%   Binning matters at the core for the same reason as the scalar
%   vortex: cos/sin of M*theta wrap faster than the grid near the
%   axis, and the averaged phasors cancel smoothly (|Vc|,|Vs| -> 0 at
%   the core pixel) instead of scattering the stellar peak.
%
%   See also: ctb_mask_vortex, ctb_vvc.
    if nargin < 3, K = 8; end
    c = floor(N/2);
    [xx, yy] = meshgrid((0:N-1) - c, (0:N-1) - c);
    if K == 1
        th = atan2(yy, xx);
        Vc = cos(m*th);  Vs = sin(m*th);
        Vc(c+1, c+1) = 0;  Vs(c+1, c+1) = 0;
        return
    end
    off = ((0:K-1) - (K-1)/2) / K;
    Vc = zeros(N);  Vs = zeros(N);
    for a = 1:K
        for b = 1:K
            th = atan2(yy + off(b), xx + off(a));
            Vc = Vc + cos(m*th);
            Vs = Vs + sin(m*th);
        end
    end
    Vc = Vc / K^2;  Vs = Vs / K^2;
end
