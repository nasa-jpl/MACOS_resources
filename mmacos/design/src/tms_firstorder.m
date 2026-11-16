function T = tms_firstorder(f, d, R2)
%TMS_FIRSTORDER  General first order of the two-mirror TMS: EFL held by eliminating R1, the stop telecentric.
%   T = tms_firstorder(f, d, R2) for focal length f, spacing d and the
%   concave secondary's |R2| (metres).  Thin mirrors unfolded: phi2 = 2/R2,
%   phi1 from the EFL, 1/f = phi1 + phi2 - d phi1 phi2  ->
%       phi1 = (1/f - phi2)/(1 - d phi2)        (negative: a CONVEX primary)
%   The stop at M2's front focal point (1/phi2 before M2) makes the output
%   TELECENTRIC; its object-space image through M1 (the entrance pupil) is at
%   z_ep = z_s/(1 - z_s phi1), z_s = d - 1/phi2 (signed from M1 along the
%   beam; z_ep < 0 = ahead of M1).  Back focus t2 from the marginal ray.
%   Petzval sum phi1 + phi2 (zero = a flat field; R1 = R2).
%   Returns T: .R [R1 R2] (|R|), .phi [phi1 phi2], .z_ep, .s_stop, .t2,
%   .petzval, .ok (phi1 < 0 and a real focus).  Reduces to TMS_PARAXIAL at
%   R2 = 2 sqrt(f d).
%
%   See also TMS_PARAXIAL, TMS_GEOM.
    phi2 = 2/R2;  phi1 = (1/f - phi2)/(1 - d*phi2);
    z_s = d - 1/phi2;  z_ep = z_s/(1 - z_s*phi1);
    y1 = 1;  u1 = -y1*phi1;  y2 = y1 + d*u1;  u2 = u1 - y2*phi2;
    t2 = -y2/u2;
    T = struct('R', [2/abs(phi1), R2], 'phi', [phi1 phi2], 'z_ep', z_ep, 's_stop', 1/phi2, 't2', t2, ...
               'petzval', phi1 + phi2, 'efl', -y1/u2, 'ok', phi1 < 0 && t2 > 0 && isfinite(t2));
end
