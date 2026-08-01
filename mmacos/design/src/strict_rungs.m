function [v, W] = strict_rungs(P, D, L, p1, d1, Vd, Nd, X)
%STRICT_RUNGS  The four reference-freedom rungs of the strict WFE metric.
%
%   v = STRICT_RUNGS(P, D, L, p1, d1, Vd, Nd, X) returns a 1x4 row of RMS
%   wavefront errors in metres, one per rung, for ONE field.  Arguments are
%   the same ray arrays STRICT_REFS takes: P (3,N) ray positions at the last
%   surface, D (3,N) directions there, L (N) cumulative OPL, p1/d1 the chief
%   ray, (Vd,Nd) the frozen detector plane, X (3,1) the exit pupil.
%
%   [v, W] = ... also returns the per-ray RESIDUAL WAVEFRONTS, W (N,4), one
%   column per rung, each already reduced by that rung's freedoms (piston
%   is left in; every rung's statistic is a std, so piston never enters).
%   Hand a column to a Strehl evaluation -- |mean(exp(i*2*pi*W/lambda))|^2
%   is the exact aperture form and needs the wavefront, not its RMS.
%
%   THE RUNGS, in order of increasing reference freedom:
%     1  strict-chief      sphere centred on the CHIEF-RAY intercept on the
%                          frozen detector, piston-only removal.
%     2  strict-centroid   sphere centred on the SPOT CENTROID on the same
%                          plane.  PRIMARY per Dave's 2026-07-31 ruling.
%     3  + best focus      rung 2 with the sphere centre slid along the chief
%                          to that field's own best focus.  This is the FLOOR
%                          any detector surface can reach, since a detector
%                          can only choose where along each chief ray the
%                          image point sits.
%                          SOLVER FLOOR, measured and left alone: the slide is
%                          a bounded FMINBND over +-50 mm, and u = 0 (i.e.
%                          rung 2) is inside that domain, so rung 3 ought to
%                          be <= rung 2 by construction.  On a field where the
%                          centroid sphere already sits at best focus the
%                          search stops ~1.7e-4 RELATIVE above it instead
%                          (measured on rodgers1_epd4060_rodgersS3, fields 1
%                          and 7 of a uniform 3x3 box).  It is a search
%                          tolerance, not physics.  Clamping it to
%                          min(ff(u), ff(0)) is the obvious repair and would
%                          move every committed rodgers1 rung-3/rung-4 number
%                          in the 4th digit -- a reviewed change, not a
%                          side effect of a file move.  tStrictKernel gates
%                          the ordering at 1e-3 relative so a REAL inversion
%                          still trips.
%     4  + LS tip/tilt     rung 3 with least-squares tip/tilt removed over
%                          the exit-pupil coordinates.  This is the rung
%                          CODE V's field-map RMS is consistent with
%                          (PACKET.md Addendum 8.7 brackets it from both
%                          sides: the next rung, removing astigmatism,
%                          overshoots).
%
%   The rungs are ORDERED and each is more permissive than the last, so a
%   result that lands between two of them is bracketed, not fitted.  Always
%   name the rung a quoted number came from -- reporting conventions differ
%   by 1.3-1.7x on comatic fields and the tilt treatment is the whole game.
%
%   Pupil coordinates for the tilt fit are each ray propagated to the plane
%   through the exit pupil normal to the chief, measured from the pupil.
%
%   Hoisted from the local RUNGS_ that PUPIL_AUDIT and DENSE_FIELD_CHECK
%   each carried (identical copies) -- one kernel so the two cannot drift.
%
%   See also STRICT_REFS, STRICT_SPHERE_OPL, STRICT_LADDER_DECK.

    f = strict_refs(P, D, L, p1, d1, Vd, Nd, X);
    v = nan(1,4);
    v(1) = f.wfe_chief;
    v(2) = f.wfe_centroid;

    % ---- exit-pupil coordinates transverse to the chief -----------------
    e3 = d1(:)/norm(d1(:));
    e1 = [1;0;0] - e3*dot([1;0;0],e3);
    if norm(e1) < 1e-8, e1 = [0;1;0] - e3*dot([0;1;0],e3); end
    e1 = e1/norm(e1);   e2 = cross(e3,e1);
    tp = (e3.'*(X(:)-P))./(e3.'*D);   Q = P + D.*tp;
    px = (e1.'*(Q-X(:))).';   py = (e2.'*(Q-X(:))).';

    % ---- rung 3: per-field best focus, slid along the chief -------------
    c0 = f.c_centroid;   R0 = f.R_centroid;
    ff = @(u) std(strict_sphere_opl(P, D, L, c0+e3*u, R0+u));
    u  = fminbnd(ff, -0.05, 0.05);
    v(3) = ff(u);

    % ---- rung 4: + least-squares tip/tilt over the pupil ----------------
    Wb = strict_sphere_opl(P, D, L, c0+e3*u, R0+u);
    Am = [ones(numel(px),1), px(:), py(:)];
    W4 = Wb(:) - Am*(Am\Wb(:));
    v(4) = std(W4);

    if nargout > 1
        W = [strict_sphere_opl(P, D, L, f.c_chief,    f.R_chief), ...
             strict_sphere_opl(P, D, L, f.c_centroid, f.R_centroid), ...
             Wb(:), W4];
    end
end
