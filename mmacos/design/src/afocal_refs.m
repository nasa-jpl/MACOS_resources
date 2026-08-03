function r = afocal_refs(P, D, L, p1, d1, a)
%AFOCAL_REFS  The flat wavefront reference for an AFOCAL exit beam, plus
%   the identity that connects its tilt term to the exit ray directions.
%   Shared kernel for AFOCAL_RUNGS and AFOCAL_WFE_DECK, so the two cannot
%   drift.  Plane-reference sibling of STRICT_REFS.
%
%   r = AFOCAL_REFS(P, D, L, p1, d1, a)
%
%   P (3,N) ray positions at the reference surface, D (3,N) directions
%   there, L (N) cumulative OPL, p1/d1 the exit chief ray, a (3,1) the
%   reference ANCHOR (the coldstop / interface-pupil vertex).
%
%   WHY A PLANE.  An afocal system emits a collimated beam; the wavefront
%   its output should be compared against is a PLANE, not a sphere.  There
%   is no exit-pupil radius to anchor and no detector to centre on, so the
%   two degrees of freedom a focal reference spends on WHERE the sphere
%   sits are spent here on WHICH WAY the plane faces.  The reference is
%   therefore the plane through A normal to the exit CHIEF RAY:
%
%       W = OPL to { x : d1.(x - a) = 0 }
%
%   Moving A along d1 adds a constant to every W, so the anchor's position
%   ALONG the beam is pure piston and cannot change any statistic here.
%   Its TRANSVERSE position sets where the pupil coordinates are measured
%   from, which is why it is the coldstop vertex and not an arbitrary point.
%
%   THE IDENTITY.  For a locally plane wavefront of direction D, the OPL on
%   the reference plane parametrised by (u,v) is
%
%       W(u,v) = c + u (D.e1) + v (D.e2)
%
%   -- the least-squares GRADIENT of W over the pupil IS the transverse part
%   of the exit ray direction, with no sign flip and no factor.  So the
%   tilt this kernel removes at rung 2 is a BORESIGHT, and there are two
%   estimators of it:
%     .boresight_ls    from the LS gradient of W  (= what rung 2 removes)
%     .boresight_mean  from the mean of the ray directions  (= where the
%                      far-field centroid of the exit beam actually sits)
%
%   THEY ARE NOT THE SAME NUMBER, AND THE GAP IS THE POINT.  On a wavefront
%   that is tilt plus defocus they agree exactly (rho^2 is even, u and v are
%   odd).  On an ODD term they do not: for a pure coma monomial W = A*u*rho^2
%   over a full disc of radius R the least-squares fit onto [1,u,v] returns
%   2*A*R^2/3 while the mean gradient is A*R^2 -- a fixed factor 2/3.  So
%
%     .bore_split_urad   the angle between the two estimators
%
%   is an ODD-ABERRATION (coma) INDICATOR, not an error bar: it is the
%   afocal analogue of STRICT_REFS' chief-minus-centroid displacement, which
%   on the focal side tracked coma the same way.  TAFOCALKERNEL pins the 2/3
%   relation exactly on a synthetic coma bundle.  On the Rodgers2 decks it
%   runs at ~0.6 of the tilt, i.e. those exit wavefronts are coma-dominated.
%
%   Returns r with
%     .a .n_chief (3,1)        anchor and reference normal (= d1)
%     .W (N,1)                 OPL to the reference plane, metres
%     .wfe_chief               std(W), metres  -- rung 1
%     .e1 .e2 (3,1)            pupil basis in the reference plane
%     .px .py (N,1) .rho_max   pupil coordinates, metres
%     .tilt (1,2) .tilt_urad   LS gradient (rad) and its magnitude (urad)
%     .boresight_ls (3,1) .boresight_mean (3,1) .bore_split_urad
%     .power_coef              defocus coefficient c3, 1/m  (W = ... + c3*rho^2)
%     .power_sag_m             c3*rho_max^2 -- the edge sag it represents
%     .divergence_urad         2*c3*rho_max -- residual marginal-ray angle
%     .R_curv_m                -1/(2*c3) -- signed wavefront radius
%
%   SIGN OF THE POWER TERM, derived and then pinned by test.  A beam
%   converging to a focus R metres downstream carries a wavefront sphere of
%   radius R through the anchor; a point on that sphere at pupil radius rho
%   lies rho^2/(2R) FURTHER ALONG the beam than the anchor plane, so its OPL
%   measured to that plane is SHORTER: W = const - rho^2/(2R).  Hence
%
%       c3 < 0  =>  CONVERGING,  R_curv_m = -1/(2*c3) > 0 = distance to focus
%       c3 > 0  =>  DIVERGING,   R_curv_m < 0,  divergence_urad > 0
%
%   TAFOCALKERNEL pins this against a known converging bundle: the rodgers1
%   deck read at M3, whose focus is its committed M3-to-FP distance.
%
%   See also AFOCAL_RUNGS, AFOCAL_PLANE_OPL, STRICT_REFS.

    d1 = d1(:)/norm(d1);   a = a(:);
    r = struct();
    r.a = a;   r.n_chief = d1;

    % ---- pupil basis in the reference plane ------------------------------
    e3 = d1;
    e1 = [1;0;0] - e3*dot([1;0;0],e3);
    if norm(e1) < 1e-8, e1 = [0;1;0] - e3*dot([0;1;0],e3); end
    e1 = e1/norm(e1);   e2 = cross(e3,e1);
    r.e1 = e1;   r.e2 = e2;

    % ---- pupil coordinates: each ray on the reference plane, measured
    %      from the chief ray's own intercept -----------------------------
    tp = (e3.'*(a - P)) ./ (e3.'*D);
    Q  = P + D .* tp;
    c0 = p1(:) + d1*(dot(d1, a - p1(:)));      % chief intercept on the plane
    px = (e1.'*(Q - c0)).';   py = (e2.'*(Q - c0)).';
    r.px = px(:);   r.py = py(:);
    r.rho_max = max(hypot(px, py));

    % ---- the wavefront ---------------------------------------------------
    W = afocal_plane_opl(P, D, L, a, d1);
    r.W = W(:);
    r.wfe_chief = std(W);

    % ---- rung-2 freedom: the LS tilt = the boresight ---------------------
    A = [ones(numel(px),1), r.px, r.py];
    cf = A \ r.W;
    g  = [cf(2), cf(3)];                       % rad, transverse of the mean dir
    r.tilt = g;
    r.tilt_urad = norm(g)*1e6;
    b = e3*sqrt(max(0,1 - g(1)^2 - g(2)^2)) + e1*g(1) + e2*g(2);
    r.boresight_ls = b/norm(b);
    dm = mean(D, 2);   dm = dm/norm(dm);
    r.boresight_mean = dm;
    r.bore_split_urad = norm(cross(r.boresight_ls, dm))*1e6;

    % ---- rung-3 freedom: the power = residual divergence -----------------
    A2 = [A, r.px.^2 + r.py.^2];
    c2 = A2 \ r.W;
    c3 = c2(4);
    r.power_coef      = c3;
    r.power_sag_m     = c3 * r.rho_max^2;
    r.divergence_urad = 2*c3*r.rho_max*1e6;
    if c3 ~= 0, r.R_curv_m = -1/(2*c3); else, r.R_curv_m = Inf; end
end
