function W = strict_sphere_opl(P, D, L, c, R)
%STRICT_SPHERE_OPL  Exact OPL from the source to a reference sphere.
%   W = STRICT_SPHERE_OPL(P, D, L, C, R) takes, for each ray, its terminal
%   point P (3xN), its direction there D (3xN), and the cumulative OPL L
%   (N) at P, and returns the OPL to the sphere of radius R centred at C
%   (3x1), crossing the ray UPSTREAM of P.
%
%   Rays are straight after the last surface (index 1), so this is exact:
%     v = P - C ;  a = v.d ;  e = |v|^2
%     |v + s d| = R   ->   s = -a - sqrt(R^2 + a^2 - e)
%     W = L + s
%   No paraxial / 2*rho^2 expansion is used.  The additive constant -R is
%   retained (it cancels under the piston-only std, but keeping it makes W
%   directly differenceable between two spheres).
%
%   See also STRICT_REFS, STRICT_RUNGS, STRICT_WFE_DECK.
    v = P - c(:);
    a = sum(v.*D, 1).';
    e = sum(v.^2, 1).';
    W = L(:) - a - sqrt(R^2 + a.^2 - e);
end
