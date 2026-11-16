function r = strict_refs(P, D, L, p1, d1, Vd, Nd, X)
%STRICT_REFS  Both reference-sphere centres for the strict metric, plus the
%   identity that connects them.  Shared kernel for STRICT_WFE (Telescope
%   path) and STRICT_WFE_DECK (deck path), so the two cannot drift.
%
%   r = STRICT_REFS(P, D, L, p1, d1, Vd, Nd, X)
%
%   P (3,N) ray positions at the last surface, D (3,N) directions, L (N)
%   cumulative OPL, p1/d1 the chief ray, (Vd,Nd) the frozen detector plane,
%   X (3,1) the exit pupil.  Rays are straight after the last surface, so
%   everything below is exact algebra on that segment.
%
%   THE TWO REFERENCES (Dave, 2026-07-31 ruling):
%     'strict-centroid'  sphere centred on the SPOT CENTROID on the frozen
%                        detector.  PRIMARY.  It is what the detector
%                        actually integrates, and it is the reference CODE V's
%                        field-map RMS is consistent with.
%     'strict-chief'     sphere centred on the CHIEF-RAY intercept.
%                        SECONDARY, reported as a labelled column.
%   Both are anchored at the exit pupil and use piston-only removal; nothing
%   else about the metric changes.
%
%   THE IDENTITY.  Shifting the sphere centre from c1 to c2 = c1 + delta
%   changes the OPL of the ray through pupil point q by |q-c1| - |q-c2|.  With
%   q = X + p (p transverse to the chief) and c1 = X + R*d1,
%
%       W_chief - W_centroid  ~  -(d1.delta)  +  (p . delta_perp)/R  + O(2)
%                                 \_piston_/     \___pure tip/tilt___/
%
%   so the tilt's gradient times R must reproduce the directly measured
%   centroid-minus-chief displacement:
%     .dcen_err_m   |implied displacement - measured TRANSVERSE displacement|
%
%   THE SECOND-ORDER TERM IS NOT NOISE, and on this design it is the thing
%   worth knowing.  delta lies IN THE DETECTOR PLANE, and the beam meets that
%   plane at ~14 deg AOI, so delta has a component ALONG the chief -- i.e.
%   moving the reference from the chief intercept to the centroid changes
%   FOCUS as well as tilt.  Two residuals are therefore returned:
%     .tilt_resid    (difference - piston/tilt) / std(difference)
%                    ~2e-3 here -- that is the induced defocus, not error
%     .defoc_resid   (difference - piston/tilt/DEFOCUS) / std(difference)
%                    machine-small -- the difference IS tilt + defocus
%   Nothing is fitted to anything; these are the cross-check.
%
%   Returns r with
%     .c_chief .c_centroid (3,1)   sphere centres
%     .wfe_chief .wfe_centroid     std(W), metres
%     .R_chief .R_centroid         sphere radii
%     .dcen (3,1) .dcen_m          centroid - chief, vector and magnitude (m)
%     .dcen_implied (3,1)          the same, implied by the fitted tilt
%     .dcen_perp_m .dcen_long_m    transverse / along-chief parts of dcen
%     .dcen_err_m                  implied-minus-measured transverse displacement
%     .tilt_resid .defoc_resid     the two cross-check residuals
%     .diff_rms                    std(W_chief - W_centroid), metres
%     .centroid_2d (2,1)           centroid in the detector's own 2-D frame,
%                                  origin at the chief intercept (m)
%     .e1 .e2                      that frame's basis (in the detector plane)

    Nd = Nd(:)/norm(Nd);  d1 = d1(:)/norm(d1);  X = X(:);
    den = dot(Nd, d1);

    % ---- chief-ray intercept on the frozen detector --------------------
    c_ch = p1(:) + d1*(dot(Nd, Vd(:) - p1(:))/den);

    % ---- spot centroid on the SAME plane -------------------------------
    % Every ray is propagated to the frozen plane along its own straight
    % final segment; the centroid is the unweighted mean of those intercepts
    % (the pupil is sampled uniformly, so unweighted IS the flux centroid).
    tt = (Nd.'*(Vd(:) - P)) ./ (Nd.'*D);
    Q  = P + D .* tt;                       % (3,N) intercepts on the plane
    c_cn = mean(Q, 2);

    r = struct();
    r.c_chief = c_ch;  r.c_centroid = c_cn;
    r.dcen = c_cn - c_ch;
    r.dcen_m = norm(r.dcen);

    % ---- the two WFEs ---------------------------------------------------
    R_ch = norm(X - c_ch);   R_cn = norm(X - c_cn);
    W_ch = strict_sphere_opl(P, D, L, c_ch, R_ch);
    W_cn = strict_sphere_opl(P, D, L, c_cn, R_cn);
    r.R_chief = R_ch;  r.R_centroid = R_cn;
    r.wfe_chief = std(W_ch);  r.wfe_centroid = std(W_cn);

    % ---- pupil coordinates: each ray propagated to the plane through the
    %      exit pupil, normal to the chief ------------------------------
    e3 = d1;
    e1 = [1;0;0] - e3*dot([1;0;0],e3);
    if norm(e1) < 1e-8, e1 = [0;1;0] - e3*dot([0;1;0],e3); end
    e1 = e1/norm(e1);   e2 = cross(e3,e1);
    tp = (e3.'*(X - P)) ./ (e3.'*D);
    Qp = P + D .* tp;                       % on the exit-pupil plane
    px = (e1.'*(Qp - X)).';   py = (e2.'*(Qp - X)).';

    % ---- the identity ----------------------------------------------------
    Diff = W_ch(:) - W_cn(:);
    A    = [ones(numel(px),1), px(:), py(:)];
    coef = A \ Diff;
    sD   = std(Diff);
    r.diff_rms = sD;
    r.tilt_resid  = std(Diff - A*coef) / max(sD, realmin);
    A2   = [A, px(:).^2 + py(:).^2];          % + defocus
    r.defoc_resid = std(Diff - A2*(A2\Diff)) / max(sD, realmin);
    % gradient = delta_perp / R  ->  delta_perp = R * gradient
    r.dcen_implied = R_ch * (coef(2)*e1 + coef(3)*e2);
    dperp = r.dcen - e3*dot(e3, r.dcen);     % measured, transverse part
    r.dcen_err_m  = norm(r.dcen_implied - dperp);
    r.dcen_perp_m = norm(dperp);
    r.dcen_long_m = dot(e3, r.dcen);         % along-chief part = the defocus

    % ---- centroid in the DETECTOR's own 2-D frame -----------------------
    % f1 is the in-plane direction closest to global +x, f2 completes it;
    % this is the frame the spot diagrams and displacement maps are drawn in.
    f1 = [1;0;0] - Nd*dot([1;0;0],Nd);
    if norm(f1) < 1e-8, f1 = [0;1;0] - Nd*dot([0;1;0],Nd); end
    f1 = f1/norm(f1);  f2 = cross(Nd, f1);
    r.e1 = f1;  r.e2 = f2;
    r.centroid_2d = [dot(f1, r.dcen); dot(f2, r.dcen)];
end
