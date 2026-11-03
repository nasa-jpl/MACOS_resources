function FO = tls_first_order(P, stop)
%TLS_FIRST_ORDER  Paraxial first order of the zig-zag long-slit TMA.
%
%   FO = TLS_FIRST_ORDER(P) solves the three mirror powers for the legs
%   P.legs_m = [M1->M2 M2->M3 M3->slit] so that the system has focal length
%   P.f_m, focuses at the slit (back focus = the third leg) and is
%   TELECENTRIC there (exit pupil at infinity) with the stop at P.stop.
%   FO = TLS_FIRST_ORDER(P, STOP) overrides the stop: 'M1', 'M2', or a
%   number s in (0,1) = a stop that fraction of the way from M1 to M2.
%
%   The model is the unfolded thin-mirror chain along the chief ray, solved
%   ONCE: the legs are the same in the tangential (fold-plane) and the
%   sagittal (strip) section, so the three conditions give the SAME unfolded
%   power phi_k in both sections.  A tilted mirror supplies phi_t = 2/(R_t
%   cos i) and phi_s = 2 cos i/R_s (Coddington), so mirror k needs LOCAL
%   radii at the chief hit
%       R_t = 2/(phi cos i),  R_s = 2 cos i/phi,   R_t/R_s = 1/cos^2 i
%   -- a tilted SPHERE cannot meet the first order in both sections at once
%   (it would need i = 0); an off-axis conic section can (TLS_SECTION).
%
%   With the stop at M2 the telecentric condition is closed form: M2 sits
%   at M3's front focus, f3 = M2->M3.  Otherwise a 3x3 Newton solve.
%
%   FO fields: phi, f_k (m), Rt, Rs (signed: + concave), y (marginal height
%   fraction at M1..M3, slit), u (marginal slope after each mirror), the
%   intermediate-focus crossings per leg (m from the leg start; NaN = none
%   inside the leg), chief (height at M1..M3 and slope per rad of field),
%   entrance pupil (m from M1 along the input chief, + = downstream of M1,
%   i.e. virtual), exit slope (0 = telecentric), beam and footprint widths
%   at P.strip_half_deg (m): foot_x (strip direction) and foot_y (fold
%   plane, perpendicular to the chief), cone F/# (paraxial, both sections).
%
%   See also TMA_LONGSLIT_PARAMS, TLS_SECTION.
if nargin < 2 || isempty(stop), stop = P.stop; end
L = P.legs_m(:)';  f = P.f_m;  D = P.D_m;  ci = cosd(P.aoi_deg);
if ischar(stop) || isstring(stop)
    switch upper(char(stop))
        case 'M1', s_stop = 0;
        case 'M2', s_stop = 1;
        otherwise, error('tls_first_order: stop must be ''M1'', ''M2'' or a fraction');
    end
else
    s_stop = stop;
end
res = @(ph) [ (efl_(ph, L) - f)/f; ...
              bfd_(ph, L)/f - L(3)/f; ...
              chief_exit_(ph, L, s_stop) ];
if s_stop == 1
    % closed form: f3 = d2; then EFL and back focus fix M1, M2
    ph3 = 1/L(2);
    u3 = -1/f;  y3 = -L(3)*u3;
    u2 = u3 + y3*ph3;  y2 = y3 - L(2)*u2;
    u1 = -(1 - y2)/L(1);
    ph = [-u1, (u1 - u2)/y2, ph3];
else
    ph = [1/2.0, -1/0.9, 1/L(2)]*1.0;          % seed: the stop-at-M2 family's signs
    for it = 1:60
        r = res(ph);  J = zeros(3);
        for j = 1:3, h = 1e-7*max(abs(ph(j)), 1);  pj = ph;  pj(j) = pj(j) + h;  J(:, j) = (res(pj) - r)/h; end
        dp = -(J\r);  ph = ph + dp(:)';
        if norm(dp) < 1e-14*norm(ph), break, end
    end
end
r = res(ph);
assert(norm(r) < 1e-9, 'tls_first_order: no first-order solution for these legs (residual %.3g)', norm(r));
FO = struct();
FO.stop = stop;  FO.s_stop = s_stop;  FO.legs = L;  FO.phi = ph;  FO.f_k = 1./ph;
FO.Rt = 2./(ph.*ci);  FO.Rs = 2*ci./ph;  FO.R_unfolded = 2./ph;
% marginal ray (unit height at M1)
[y, u] = marg_(ph, L);
FO.y = y;  FO.u = u;
yl = y(1:3);  ul = u(1:3);
FO.int_focus = nan(1, 3);
for k = 1:3
    s = -yl(k)/ul(k);
    if s > 0 && s < L(k), FO.int_focus(k) = s; end
end
% chief ray: crosses the axis at the stop, slope 1 there (unit field at the stop)
[yc, uc] = chief_(ph, L, s_stop);
w = uc(1);                                    % object-space chief slope per unit stop slope
FO.chief_y = yc/w;  FO.chief_u = uc/w;        % per radian of OBJECT field
FO.exit_slope = FO.chief_u(4);
FO.ep_from_m1 = -FO.chief_y(1);              % the input chief (slope 1 per rad) crosses the axis here, m from M1 (+ = downstream: virtual)
% footprints at the strip edge (paraxial)
th = deg2rad(P.strip_half_deg);
FO.beam = D*abs(y(1:3));                      % on-axis beam width at M1..M3
FO.foot_x = FO.beam + 2*abs(FO.chief_y(1:3))*th;   % strip direction (sagittal)
FO.foot_y = FO.beam;                          % fold plane, perpendicular to the chief (no field in-plane)
FO.foot_y_surf = FO.beam./ci;                 % on the mirror surface in the fold plane
FO.efl = efl_(ph, L);  FO.bfd = bfd_(ph, L);
FO.fnum = FO.efl/D;
FO.image_half = FO.efl*tan(th);
FO.slit_cover = 2*FO.image_half;
FO.mag_field_at_m2 = FO.chief_u(2);           % chief slope after M1 per unit object field
end

% ---------------------------------------------------------------------------
function [y, u] = marg_(ph, L)
y = zeros(1, 4);  u = zeros(1, 4);
yy = 1;  uu = 0;
for k = 1:3
    y(k) = yy;  uu = uu - yy*ph(k);  u(k) = uu;
    yy = yy + L(k)*uu;
end
y(4) = yy;  u(4) = uu;
end
function e = efl_(ph, L), [~, u] = marg_(ph, L);  e = -1/u(3); end
function b = bfd_(ph, L), [y, u] = marg_(ph, L);  b = -y(3)/u(3); end
function [yc, uc] = chief_(ph, L, s)
% heights at M1..M3 and the slit, slopes after M1..M3 (uc(1) = object space,
% uc(2..4) = after M1..M3); the chief crosses the axis at the stop with slope 1
yc = zeros(1, 4);  uc = zeros(1, 4);
if s == 0
    yc(1) = 0;  u_after1 = 1;  u_obj = 1;      % stop at M1: object slope = slope after M1 (height 0)
else
    zs = s*L(1);                               % stop between M1 and M2 (or at M2)
    u_after1 = 1;  yc(1) = -zs*u_after1;
    u_obj = u_after1 + yc(1)*ph(1);
end
uc(1) = u_obj;
u = u_after1;  y = yc(1);
for k = 2:3
    y = y + L(k-1)*u;  yc(k) = y;  u = u - y*ph(k);
end
uc(2) = u_after1;  uc(4) = u;
yc(4) = yc(3) + L(3)*u;
uc(3) = u_after1 - yc(2)*ph(2);
end
function e = chief_exit_(ph, L, s), [~, uc] = chief_(ph, L, s);  e = uc(4)/uc(1); end
