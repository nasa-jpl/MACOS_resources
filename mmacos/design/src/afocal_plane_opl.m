function W = afocal_plane_opl(P, D, L, a, n)
%AFOCAL_PLANE_OPL  Exact OPL from the source to a flat reference.
%   W = AFOCAL_PLANE_OPL(P, D, L, A, N) takes, for each ray, its terminal
%   point P (3xN), its direction there D (3xN), and the cumulative OPL L
%   (N) at P, and returns the OPL to the PLANE through A (3x1) with unit
%   normal N (3x1).
%
%   Rays are straight after the last surface, so this is exact:
%     s = n.(a - P) / (n.d)     W = L + s
%   No paraxial expansion and no small-angle assumption.
%
%   THIS IS STRICT_SPHERE_OPL'S R -> INFINITY LIMIT, and that is not a
%   figure of speech -- it is the gate.  Place the sphere centre DOWNSTREAM
%   at c = a + R*n and keep the same (minus) root:
%
%     v = P - c = w - R n,  w = P - a
%     a_dot = v.d = w.d - R(n.d)
%     e     = |v|^2 = |w|^2 - 2R(w.n) + R^2
%     R^2 + a_dot^2 - e = R^2 + 2R[(w.n) - (w.d)(n.d)] + O(1)
%     s = -a_dot - sqrt(...) -> -(w.n)/(n.d)   as R -> inf
%
%   i.e. exactly the plane distance above.  TAFOCALKERNEL measures that
%   convergence on real rays over a decade sweep in R; it is 1/R, and the
%   sphere form loses conditioning long before the plane form does (the
%   sqrt subtracts two numbers of size R^2), which is why the plane case
%   gets its own exact primitive instead of a large-R sphere call.
%
%   SIGN.  s > 0 for a plane the rays have not yet reached.  The additive
%   constant is retained so W is directly differenceable between two
%   reference planes.
%
%   See also AFOCAL_REFS, AFOCAL_RUNGS, STRICT_SPHERE_OPL.

    n = n(:)/norm(n);
    den = (n.'*D).';
    W = L(:) + ((n.'*(a(:) - P)).') ./ den;
end
