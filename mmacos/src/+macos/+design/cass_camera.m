function C = cass_camera(opts)
%MACOS.DESIGN.CASS_CAMERA  A small dedicated imager: a two-mirror Cassegrain with a refractive relay.
%   C = macos.design.cass_camera('family','rc', 'D',350, 'primary_fnum',1.334, ...
%           'telescope_fnum',8.956, 'bfd',47.08, 'relay',{L1, z1; L2, z2}, ...)
%   lays out the camera of templates/36_cameras/small_camera (worked case:
%   the manual's ImageDemo.in): a primary at z = 0, a
%   convex secondary in front of it, and refractive elements (macos.design.Lens:
%   a field corrector, a filter) behind the primary's central hole, ending at
%   one image plane.  The telescope and the instrument are ONE design -- the
%   relay's power is part of the camera's focal length -- which is what sets
%   this family apart from an observatory front end feeding a Bench.
%
%   THE MIRRORS.  From design intent ('family' cassegrain | rc, 'D',
%   'primary_fnum', 'telescope_fnum' = the two-mirror f-number, 'bfd' = the
%   primary vertex to the two-mirror focus, + behind the primary) the layout and
%   the conics are macos.design.Telescope's closed form (Schroeder's (m, beta);
%   one copy of the formulas, read from Telescope(...).spec.derived), or they
%   are given outright with 'mirrors', struct('R1',..,'K1',..,'R2',..,'K2',..,
%   'sep',..) (radii unsigned: R1 the concave primary, R2 the convex secondary;
%   sep the vertex separation) -- the form a found prescription is rebuilt in.
%   Explicit conics may also override the family's: 'K1', 'K2'.
%
%   THE RELAY.  'relay' is an N x 2 cell {Lens, z_front}: each Lens's surface 1
%   sits z_front behind the primary vertex (mm).  Radii are CODE V radii along
%   the light, which behind the primary travels +z, so a Lens is written as
%   macos.design.lens_emit writes it.
%
%   GEOMETRY (the deck camera_emit writes): light enters along +z from the
%   source, passes the secondary's plane (its shadow: an obscuration of radius
%   'obscuration' x D/2 there), reflects at the primary (vertex z = 0, the
%   aperture stop) back toward -z, reflects at the secondary (vertex z = -sep)
%   toward +z, and crosses the primary's plane into the relay.  Units mm.
%
%   PARAXIAL MODEL: one y-nu trace through the folded train in GLOBAL z --
%   signed indices (n < 0 while the light travels -z), global curvatures (centre
%   of curvature at z_vertex + 1/c), thicknesses z_{k+1} - z_k -- so
%   n'u' = nu - y (n' - n) c and y' = y + t u' hold through the mirrors as
%   through the lenses (u = dy/dz).  C.par carries
%     .efl            the camera's focal length (mm)
%     .z_img          the paraxial focus ('image' NaN) or the given plane
%     .bfl            last relay surface to the focus (mm)
%     .fno            efl / D
%     .yP_unit        the paraxial chief height at the image per tan(theta)
%                     (signed; the distortion reference lens_eval uses)
%     .z_xp .Dxp      the exit pupil (the image of the primary) and its diameter
%     .telescope      the two-mirror focal length and focus z, the relay's
%                     magnification efl / f_telescope
%     .hole           the radius the returning beam needs at the primary's
%                     plane (on axis, and at 'field_deg' when given) against the
%                     obscured radius there: a hole that clears the beam and
%                     sits inside the shadow costs no aperture (.clears)
%   at 'lambda_nm'.  The surfaces are C.surf (z, c, K, n_after, kind, name).
%
%   See also macos.design.camera_emit, macos.design.lens_eval,
%   macos.design.Telescope, macos.design.Lens.
arguments
    opts.family (1,:) char = 'rc'
    opts.D (1,1) double {mustBePositive} = 350
    opts.primary_fnum (1,1) double = NaN
    opts.telescope_fnum (1,1) double = NaN
    opts.bfd (1,1) double = NaN
    opts.mirrors struct = struct()
    opts.K1 (1,1) double = NaN
    opts.K2 (1,1) double = NaN
    opts.relay cell = {}
    opts.obscuration (1,1) double {mustBeNonnegative, mustBeLessThan(opts.obscuration, 1)} = 0
    opts.lambda_nm (1,1) double {mustBePositive} = 700
    opts.image (1,1) double = NaN
    opts.field_deg (1,1) double {mustBeNonnegative} = 0
    opts.name (1,:) char = 'camera'
end
D = opts.D;
tel = struct('family', opts.family, 'm', NaN, 'beta', NaN, 'source', '');
if isempty(fieldnames(opts.mirrors))
    assert(all(isfinite([opts.primary_fnum opts.telescope_fnum opts.bfd])), ...
        'cass_camera: give primary_fnum, telescope_fnum and bfd (or ''mirrors'')');
    t = macos.design.Telescope('family', opts.family, 'aperture_diameter_mm', D, ...
        'system_fnum', opts.telescope_fnum, 'primary_fnum', opts.primary_fnum, 'BFD_mm', opts.bfd);
    d = t.spec.derived;                                 % SI metres
    M = struct('R1', abs(d.R1) * 1e3, 'K1', d.K1, 'R2', abs(d.R2) * 1e3, 'K2', d.K2, 'sep', d.sep * 1e3);
    tel.m = d.m;  tel.beta = d.beta;  tel.family = t.spec.family;
    tel.source = 'macos.design.Telescope closed form';
else
    M = opts.mirrors;
    assert(all(isfield(M, {'R1', 'K1', 'R2', 'K2', 'sep'})), ...
        'cass_camera: mirrors needs R1 K1 R2 K2 sep');
    M.R1 = abs(M.R1);  M.R2 = abs(M.R2);
    tel.family = 'given';  tel.source = 'mirrors given';
end
if isfinite(opts.K1), M.K1 = opts.K1; end
if isfinite(opts.K2), M.K2 = opts.K2; end
assert(M.sep > 0 && M.sep < M.R1 / 2, 'cass_camera: the secondary must sit inside the primary focus (0 < sep < R1/2)');

% ---- the surface list in light order, global z
S = struct('name', {}, 'kind', {}, 'z', {}, 'c', {}, 'K', {}, 'n_after', {}, 'lens', {}, 'k', {});
S(1) = struct('name', 'M1', 'kind', 'mirror', 'z', 0, 'c', -1 / M.R1, 'K', M.K1, 'n_after', -1, 'lens', 0, 'k', 0);
S(2) = struct('name', 'M2', 'kind', 'mirror', 'z', -M.sep, 'c', -1 / M.R2, 'K', M.K2, 'n_after', 1, 'lens', 0, 'k', 0);
relay = opts.relay;
assert(isempty(relay) || size(relay, 2) == 2, 'cass_camera: relay is an N x 2 cell {Lens, z_front}');
zlast = -Inf;
for j = 1:size(relay, 1)
    L = relay{j, 1};  z0 = relay{j, 2};
    assert(isa(L, 'macos.design.Lens'), 'cass_camera: relay{%d,1} must be a macos.design.Lens', j);
    assert(z0 > zlast, 'cass_camera: relay element %d starts at z = %g, inside the element before it', j, z0);
    n = [L.index(opts.lambda_nm) 1];
    nm = L.names();
    for k = 1:L.N
        S(end + 1) = struct('name', nm{k}, 'kind', 'lens', 'z', z0 + L.z(k), 'c', 1 / L.R(k), ...
            'K', L.K(k), 'n_after', n(k), 'lens', j, 'k', k); %#ok<AGROW>
    end
    zlast = z0 + L.span;
end
if ~isempty(relay)
    assert(S(3).z > 0, 'cass_camera: the relay must sit behind the primary vertex (z > 0)');
end

% ---- paraxial: the marginal ray (y = D/2 at the primary) and the chief (through its vertex)
[ym, um] = ynu_(S, D / 2, 0);
[yc, uc] = ynu_(S, 0, 1);
efl = -(D / 2) / (S(end).n_after * um(end));
z_par = S(end).z - ym(end) / um(end);
z_img = opts.image;  if isnan(z_img), z_img = z_par; end
% the two-mirror telescope alone
[ym2, um2] = ynu_(S(1:2), D / 2, 0);
f_tel = -(D / 2) / um2(end);
z_tel = S(2).z - ym2(end) / um2(end);
% the chief at the image plane, per tan(theta) (u0 = 1: linear in the slope)
yP_unit = yc(end) + uc(end) * (z_img - S(end).z);
% exit pupil: where the image-space chief crosses the axis
z_xp = S(end).z - yc(end) / uc(end);
Dxp = 2 * abs(ym(end) + um(end) * (z_xp - S(end).z));

% ---- the primary's hole: the returning beam at z = 0 against the shadow there
%      (on the leg M2 -> relay, after surface 2)
r_beam0 = abs(ym(2) + um(2) * (0 - S(2).z));
r_chief = abs(yc(2) + uc(2) * (0 - S(2).z)) * tand(opts.field_deg);
r_need = r_beam0 + r_chief;
r_shadow = opts.obscuration * D / 2;
hole = struct('r_beam_axis', r_beam0, 'r_need', r_need, 'field_deg', opts.field_deg, ...
    'r_shadow', r_shadow, 'clears', r_need <= r_shadow);
if r_shadow > 0 && ~hole.clears
    warning('macos:design:cass_camera:hole', ...
        'cass_camera: the returning beam needs a %.4g mm hole at the primary, larger than the %.4g mm shadow (aperture is lost to the hole)', ...
        r_need, r_shadow);
end

par = struct('efl', efl, 'fno', efl / D, 'z_img', z_img, 'z_paraxial', z_par, ...
    'bfl', z_par - S(end).z, 'yP_unit', yP_unit, 'z_xp', z_xp, 'Dxp', Dxp, ...
    'telescope', struct('efl', f_tel, 'z_focus', z_tel, 'relay_mag', efl / f_tel), ...
    'hole', hole, 'lambda_nm', opts.lambda_nm, 'y', ym, 'u', um, 'yc', yc, 'uc', uc);
C = struct('name', opts.name, 'D', D, 'mirrors', M, 'telescope', tel, 'relay', {relay}, ...
    'obscuration', opts.obscuration, 'surf', S, 'par', par, 'lambda_nm', opts.lambda_nm);
end

function [y, u] = ynu_(S, y1, u0)
% y-nu in global z with signed indices: y(k) at surface k, u(k) after it.
N = numel(S);  y = zeros(1, N);  u = zeros(1, N);
n = 1;  uu = u0;  y(1) = y1;
for k = 1:N
    if k > 1, y(k) = y(k - 1) + u(k - 1) * (S(k).z - S(k - 1).z); end
    np = S(k).n_after;
    uu = (n * uu - y(k) * (np - n) * S(k).c) / np;
    u(k) = uu;  n = np;
end
end
