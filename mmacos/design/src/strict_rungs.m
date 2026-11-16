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
%                          u = 0 IS the rung-2 sphere and IS inside the
%                          search domain, so rung 3 can never legitimately
%                          exceed rung 2.  It is now guaranteed not to: the
%                          search takes the better of its own result and
%                          ff(0), and runs at a tolerance scaled to the
%                          system.  See THE FOCUS SEARCH below for why that
%                          was necessary.
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
    %
    % THE FOCUS SEARCH, and why it is written this way.  u is a focus SLIDE
    % in BaseUnits, and FMINBND's default TolX is 1e-4 in those units -- on
    % a metre-based prescription, 0.1 mm.  A 0.1 mm slide on an f/20 beam is
    % u/(8*F^2) ~ 31 nm of wavefront, so on a design already corrected to
    % the nm level the default tolerance cannot resolve the minimum at all:
    % measured on the e2e2 axial TMA (60 m EFL, f/20, sub-nm residual) the
    % search returned 2.386 nm where the rung-2 sphere it started from reads
    % 0.890 nm -- rung 3 reporting 2.7x WORSE than the more restrictive rung
    % below it.  On the rodgers1 deck, 100x more aberrated, the same defect
    % is only 1.7e-4 relative, which is why it survived.
    %
    % Two changes, both load-bearing:
    %   TolX is RELATIVE to the pupil-to-detector distance, so it means the
    %   same physical precision on a metre deck and a millimetre one;
    %   ff(0) -- the rung-2 sphere -- is evaluated explicitly and wins ties,
    %   which makes rung 3 <= rung 2 true BY CONSTRUCTION rather than by
    %   the solver's good behaviour.
    %
    % The +-0.05 BRACKET is still in BaseUnits and is NOT scale-free: on a
    % millimetre prescription it is 50 um and may not contain the optimum.
    % The ff(0) guard is that case's safety net -- the rung degrades to
    % rung 2 rather than returning something worse than it.
    c0 = f.c_centroid;   R0 = f.R_centroid;
    ff = @(u) std(strict_sphere_opl(P, D, L, c0+e3*u, R0+u));
    w0 = ff(0);
    [u, wu] = fminbnd(ff, -0.05, 0.05, ...
                      optimset('TolX', max(realmin, 1e-12*abs(R0))));
    if ~(wu < w0), u = 0;  wu = w0; end
    v(3) = wu;

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
