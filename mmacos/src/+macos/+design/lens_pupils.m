function P = lens_pupils(L, lambda_nm, opts)
%MACOS.DESIGN.LENS_PUPILS  First-order layout of a lens with its stop: pupils, f-number, image.
%   P = macos.design.lens_pupils(L, lambda_nm, 'stop', [z_s a_s])
%   P = macos.design.lens_pupils(L, lambda_nm, 'stop', [z_s NaN], 'epd', D)
%
%   L is a macos.design.Lens.  The STOP is a plane at z_s (measured from
%   surface 1's vertex along +z, in any gap or outside the lens) with
%   semi-diameter a_s; give 'epd' instead of a_s and a_s is solved so the
%   entrance pupil has that diameter.  'object' is the object distance in
%   front of surface 1 (Inf, the default = collimated); 'image' the image
%   plane's z (NaN = the paraxial focus).  Everything is the textbook y-nu
%   trace (Welford): the pupils are the stop's paraxial images, found from
%   the CHIEF ray (through the stop centre); the pupil diameters scale the
%   stop by the MARGINAL ray's height ratio.  Generalized from
%   challenges/iodc2010/iodc2010_paraxial (one glass, one sign pattern) to any
%   Lens and a finite object.
%
%   P fields (lengths in the lens's units; z from surface 1's vertex)
%     .efl .bfl .z_focus     the lens's EFL, back focal length and paraxial
%                            focus (of the OBJECT given: the conjugate image
%                            for a finite object)
%     .z_img                 the image plane used
%     .z_ep .D               entrance pupil position and DIAMETER
%     .z_xp .Dxp             exit pupil position and diameter (Inf: telecentric)
%     .fno                   |efl| / D (infinite object); the working f-number
%                            1/(2 u') for a finite one is .fno_w
%     .u_img                 the marginal ray's image-space slope at the stop edge
%     .yP_unit               chief height at the image plane per unit tan(field)
%                            (infinite object)
%     .z_obj .cone           finite object: its z and the FULL cone angle (rad)
%                            the stop admits (MACOS's point-source Aperture)
%     .n .c .z .z_s .a_s .lambda_nm
arguments
    L (1,1) macos.design.Lens
    lambda_nm (1,1) double {mustBePositive}
    opts.stop (1,2) double
    opts.epd (1,1) double = NaN
    opts.object (1,1) double = Inf
    opts.image (1,1) double = NaN
end
n = [L.index(lambda_nm) 1];
c = L.curv();  z = L.z;  t = L.t;  N = L.N;
z_s = opts.stop(1);
fin = isfinite(opts.object);
assert(~fin || opts.object > 0, 'lens_pupils: the object distance must be positive (in front of surface 1)');
Pe = L.paraxial(lambda_nm);

% ---- the marginal ray: from infinity (y1 = 1, u0 = 0) or the object point (u0 = 1)
if fin
    s = opts.object;  [ym, um] = ynu_(t, n, c, s, 1);  u0m = 1;
else
    [ym, um] = ynu_(t, n, c, 1, 0);  u0m = 0;
end
assert(um(N) ~= 0, 'lens_pupils: the image is at infinity');
z_focus = z(N) - ym(N) / um(N);
if isnan(opts.image), z_img = z_focus; else, z_img = opts.image; end
h_m = height_(z, ym, um, u0m, z_s);
assert(h_m ~= 0, 'lens_pupils: the marginal ray crosses the axis at the stop plane');

% ---- the chief ray: unit object slope, the surface-1 height that puts it through the stop centre
[y0, u0] = ynu_(t, n, c, 0, 1);  h0 = height_(z, y0, u0, 1, z_s);
[y1, u1] = ynu_(t, n, c, 1, 1);  h1 = height_(z, y1, u1, 1, z_s);
assert(h1 ~= h0, 'lens_pupils: degenerate chief-ray solve');
yb = -h0 / (h1 - h0);
[yc, uc] = ynu_(t, n, c, yb, 1);
z_ep = z(1) - yb;                      % object-space axis crossing (slope 1)

% ---- the stop size: given, or solved from the EPD
a_s = opts.stop(2);
if isnan(a_s)
    assert(isfinite(opts.epd) && opts.epd > 0, 'lens_pupils: give the stop semi-diameter or ''epd''');
    if fin
        % marginal slope that fills an EP of diameter epd, then its height at the stop
        u_need = (opts.epd / 2) / (z_ep + s);
        a_s = abs(h_m) * u_need;
    else
        a_s = abs(h_m) * opts.epd / 2;
    end
end
if fin
    u0_edge = a_s / abs(h_m);          % object-space marginal slope at the stop edge
    D = 2 * u0_edge * (z_ep + s);      % the marginal ray's height at the EP, doubled
    cone = 2 * atan(u0_edge);
    u_img = um(N) * u0_edge;
else
    D = 2 * a_s / abs(h_m);
    cone = NaN;
    u_img = um(N) * D / 2;
end
if uc(N) ~= 0
    z_xp = z(N) - yc(N) / uc(N);
    Dxp  = 2 * a_s * abs(height_(z, ym, um, u0m, z_xp)) / abs(h_m);
else
    z_xp = Inf;  Dxp = Inf;
end
yP_unit = yc(N) + (z_img - z(N)) * uc(N);

P = struct('efl', Pe.efl, 'bfl', Pe.bfl, 'z_focus', z_focus, 'z_img', z_img, ...
    'z_ep', z_ep, 'D', D, 'z_xp', z_xp, 'Dxp', Dxp, 'fno', abs(Pe.efl) / D, ...
    'fno_w', 1 / (2 * abs(u_img)), 'u_img', u_img, 'yP_unit', yP_unit, ...
    'z_obj', -opts.object, 'cone', cone, 'object', opts.object, ...
    'n', n(1:N-1), 'c', c, 'z', z, 'z_s', z_s, 'a_s', a_s, 'lambda_nm', lambda_nm);
end

% -------------------------------------------------------------------------
function [y, u] = ynu_(t, n_after, c, y1, u0)
N = numel(c);  y = zeros(1, N);  u = zeros(1, N);
yk = y1;  uk = u0;  nk = 1;
for k = 1:N
    y(k) = yk;  np = n_after(k);
    uk = (nk * uk - yk * c(k) * (np - nk)) / np;
    u(k) = uk;  nk = np;
    if k < N, yk = yk + t(k) * uk; end
end
end

function h = height_(z, y, u, u0, zq)
% height at axial position zq, in the gap that contains it
j = find(z <= zq, 1, 'last');
if isempty(j), h = y(1) + (zq - z(1)) * u0; else, h = y(j) + (zq - z(j)) * u(j); end
end
