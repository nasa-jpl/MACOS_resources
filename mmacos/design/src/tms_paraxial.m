function T = tms_paraxial(f, D, d, fov)
%TMS_PARAXIAL  Closed-form first order of the two-mirror modified Schwarzschild (flat, telecentric).
%   T = tms_paraxial(f, D, d, fov) for focal length f, entrance-pupil
%   diameter D, mirror spacing d (M1 -> M2) and full cross-track field fov
%   (rad), all in metres / radians.  Thin mirrors unfolded: the CONVEX
%   primary has power -phi, the CONCAVE secondary +phi (equal |R|: the
%   Petzval sum is zero, a FLAT field), so
%       f   = 1/(d phi^2)            ->  phi = 1/sqrt(f d),  R = 2/phi
%   The output is TELECENTRIC when the stop sits at the secondary's front
%   focal point, a distance 1/phi = sqrt(f d) before M2 -- VIRTUAL, behind
%   the primary, whenever d < f (Mouroulis & Green 2018 sec. 5.1).  Its
%   image in object space, the ENTRANCE PUPIL, lies f - sqrt(f d) ahead of
%   M1 (z_ep = sqrt(f d) - f, signed along the incoming beam from M1); the
%   back focus is t2 = f + sqrt(f d) beyond M2 -- the inverted telephoto,
%   "large against f".  The family is ONE-dimensional in d at fixed f.
%
%   Returns T: .f .D .d .phi .R (|R|, both mirrors) .z_ep .s_stop (the
%   stop's distance before M2) .t2 .len (EP -> image along the axis,
%   unfolded: the envelope's scale) .y (marginal heights at M1, M2), .ybar
%   (chief heights at M1, M2 at the field edge fov/2), .Dfp (beam footprint
%   diameter at M1, M2 along the slit: 2(|y| + |ybar|)), .EFL_check (from
%   the trace), .telec_check (the exit chief slope at the field edge; 0 =
%   telecentric).
    phi = 1/sqrt(f*d);  R = 2/phi;
    s_stop = 1/phi;  z_ep = s_stop - f;  t2 = f + s_stop;
    pw = [-phi, +phi];
    % marginal ray from infinity, height D/2 at the entrance pupil (= at M1, collimated)
    [ym, um] = tr_(pw, d, D/2, 0);
    % chief: through the EP centre at the field edge
    th = fov/2;  y1c = -z_ep*th;                          % EP ahead of M1 by -z_ep
    [yc, uc] = tr_(pw, d, y1c, th);
    T = struct('f', f, 'D', D, 'd', d, 'phi', phi, 'R', R, 'z_ep', z_ep, 's_stop', s_stop, 't2', t2, ...
               'len', -z_ep + d + t2, 'y', ym, 'ybar', yc, 'Dfp', 2*(abs(ym) + abs(yc)), ...
               'EFL_check', -(D/2)/um(end), 'telec_check', uc(end));
end

function [y, u] = tr_(pw, d, y1, u0)
%TR_  Unfolded thin-mirror paraxial trace: heights at M1, M2; slopes after each.
    y = zeros(1, 2);  u = zeros(1, 2);
    y(1) = y1;  u(1) = u0 - y(1)*pw(1);
    y(2) = y(1) + d*u(1);  u(2) = u(1) - y(2)*pw(2);
end
