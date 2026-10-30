function P = iodc2010_paraxial(L, lambda_nm)
%IODC2010_PARAXIAL  Textbook y-nu paraxial trace of a green-lens prescription.
%   P = iodc2010_paraxial(L, 532) returns the first-order quantities the
%   challenge is scored with, computed from the prescription alone (no
%   engine call), so the engine's numbers are checked against them, never
%   the other way round:
%
%     P.n_after(k)   index after surface k at lambda (iodc2010_glass)
%     P.c(k)         curvature of surface k (1/mm, signed, 0 = plano)
%     P.efl          effective focal length (mm, signed; the problem wants |efl| = 100)
%     P.bfl          back focal length from the last vertex (mm)
%     P.z_focus      paraxial focus position (mm from surface 1)
%     P.z_img        the image plane used (L.t(end), or z_focus when NaN)
%     P.z_ep, P.D    entrance pupil position (mm from surface 1) and DIAMETER
%                    (mm): the stop's paraxial image in object space.  D is
%                    the "entrance pupil diameter" of the merit.
%     P.z_xp, P.Dxp  exit pupil position and diameter (the stop's paraxial
%                    image in image space)
%     P.yP_unit      paraxial chief-ray height at the image plane per unit
%                    tan(theta): yP(theta) = P.yP_unit * tan(theta).  Equals
%                    efl when the image plane is the paraxial focus.
%     P.h_marg(k)    unit marginal-ray height at surface k (y1 = 1)
%     P.fno          |efl| / D
%
%   Recipe (Kingslake / Welford): refraction  n' u' = n u - y c (n' - n),
%   transfer y' = y + t u'.  The pupils come from the paraxial CHIEF ray, the
%   ray through the stop centre: its object-space crossing of the axis is the
%   entrance pupil, its image-space crossing the exit pupil; the pupil
%   diameters scale the stop by the marginal ray's height ratio.  The stop
%   plane may lie in any gap (air or glass); the ray height there is the
%   linear transfer inside that gap.
arguments
    L struct
    lambda_nm (1,1) double {mustBePositive}
end
N = L.N;
n_after = zeros(1, N);
for k = 1:N, n_after(k) = iodc2010_glass(L.glass{k}, lambda_nm); end
c = L.sgn / L.R;

% --- unit marginal ray from infinity: y1 = 1, u0 = 0
[ym, um] = trace_(L, n_after, c, 1, 0);
assert(um(N) ~= 0, 'iodc2010_paraxial: afocal lens (u'' = 0); no focal length');
efl     = -1 / um(N);
bfl     = -ym(N) / um(N);
z_focus = L.z(N) + bfl;
if isnan(L.t(N)), z_img = z_focus; else, z_img = L.z(N) + L.t(N); end

% --- the stop: gap index and the unit marginal height there
z_s = L.stop(1);  a_s = L.stop(2);
h_m = height_at_(L, ym, um, 0, z_s);     % marginal height at the stop (y1 = 1 scale)
assert(h_m ~= 0, 'iodc2010_paraxial: the marginal ray crosses the axis at the stop plane');
D   = 2 * a_s / abs(h_m);                % paraxial entrance-pupil diameter

% --- the chief ray: unit object angle, the height at surface 1 that puts it
%     through the stop centre (heights are affine in y1)
[y0, u0] = trace_(L, n_after, c, 0, 1);  h0 = height_at_(L, y0, u0, 1, z_s);
[y1, u1] = trace_(L, n_after, c, 1, 1);  h1 = height_at_(L, y1, u1, 1, z_s);
assert(h1 ~= h0, 'iodc2010_paraxial: degenerate chief-ray solve');
ybar1 = -h0 / (h1 - h0);
[yc, uc] = trace_(L, n_after, c, ybar1, 1);
z_ep = L.z(1) - ybar1;                   % object-space axis crossing (ubar0 = 1)
if uc(N) ~= 0
    z_xp = L.z(N) - yc(N) / uc(N);       % image-space axis crossing
    % the stop scaled by the marginal ray's height ratio XP : stop
    Dxp  = 2 * a_s * abs(height_at_(L, ym, um, 0, z_xp)) / abs(h_m);
else
    z_xp = Inf;  Dxp = Inf;              % image-space telecentric
end
yP_unit = yc(N) + (z_img - L.z(N)) * uc(N);

P = struct('n_after', n_after, 'c', c, 'efl', efl, 'bfl', bfl, 'z_focus', z_focus, ...
    'z_img', z_img, 'z_ep', z_ep, 'D', D, 'z_xp', z_xp, 'Dxp', Dxp, 'yP_unit', yP_unit, ...
    'h_marg', ym, 'u_marg', um, 'y_chief', yc, 'u_chief', uc, 'fno', abs(efl) / D, ...
    'lambda_nm', lambda_nm, 'z_s', z_s, 'a_s', a_s);
end

% -------------------------------------------------------------------------
function [y, u] = trace_(L, n_after, c, y1, u0)
% y(k): height AT surface k; u(k): angle AFTER surface k (tan, paraxial).
N = L.N;  y = zeros(1, N);  u = zeros(1, N);
yk = y1;  uk = u0;  nk = 1;
for k = 1:N
    y(k) = yk;
    np   = n_after(k);
    uk   = (nk * uk - yk * c(k) * (np - nk)) / np;
    u(k) = uk;  nk = np;
    if k < N, yk = yk + L.t(k) * uk; end
end
end

function h = height_at_(L, y, u, u0, z)
% Height of a traced paraxial ray at axial position z: in the gap that
% contains z (before surface 1 the ray has angle u0 and height y(1) at z(1)).
j = find(L.z <= z, 1, 'last');
if isempty(j)
    h = y(1) + (z - L.z(1)) * u0;
else
    h = y(j) + (z - L.z(j)) * u(j);
end
end
