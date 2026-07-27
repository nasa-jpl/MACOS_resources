function jp = jones_pupil(srf, opts)
%MACOS.JONES_PUPIL  2x2 Jones pupil at an element from two polarized traces.
%   jp = macos.jones_pupil(SRF) traces the loaded prescription twice with
%   orthogonal source polarization states, (Ex0,Ey0) = (1,0) and (0,1),
%   harvests the per-ray vector E-field at element SRF (macos.ray_field),
%   and assembles the 2x2 complex Jones matrix at every ray-grid point:
%
%       [E_out in exit basis] = J * [source state in (xGrid, yGrid)]
%
%   INPUT basis (fixed by the engine): the source launches every ray with
%   E = S*(Ex0*xGrid + Ey0*yGrid) -- the source-frame transverse pair,
%   uniform across the grid (ssrcray.inc:309).  S is the engine's flux
%   normalization (~1/sqrt(nRays)); J carries this common real scalar, so
%   absolute elements are scaled but every polarization metric
%   (diattenuation, retardance, ratios) is unaffected.
%
%   OUTPUT basis ('basis' option):
%     'double-pole'  (default) Chipman double-pole coordinates: the global
%                    reference pair (xref, yref) is rotated from the exit
%                    axis to each ray direction along the great circle
%                    (Rodrigues rotation).  Smooth over any physical pupil;
%                    singular only at the antipode.  Use for budget numbers.
%     'local-sp'     per-ray s/p relative to the exit axis: s = k x a /
%                    |k x a|, p = s x k.  Coordinate-singular on axis --
%                    diagnostic only, never for budgets (the singularity
%                    imprints spurious tilt/astig-like retardance).
%     'global'       project E onto (xref, yref) ignoring ray direction.
%                    Diagnostic for near-collimated exits.
%
%   jp = macos.jones_pupil(SRF, 'basis', B, 'axis', A, 'xref', X) options:
%     'basis'  as above.
%     'axis'   exit-axis 3-vector (default: mean unit ray direction over
%              unvignetted rays -- the chief direction).
%     'xref'   reference x direction (default: global +x projected out of
%              the axis; falls back to global +y when axis ~ +/-x).
%
%   Returns struct:
%     .J        N x N x 2 x 2 complex Jones (NaN at vignetted points --
%               never zero-filled, so statistics are honest)
%     .mask     N x N logical, true where both traces have status 0
%     .basis    the basis used
%     .axis     exit axis actually used (3x1 unit)
%     .xref     reference pair actually used (.xref, .yref, 3x1 each)
%     .kx,.ky,.kz  exit ray direction cosines (from the x-state trace)
%     .leak     max over mask of |E . khat| / |E| (longitudinal residual;
%               paraxial-consistency diagnostic)
%     .singular N x N logical: points where the requested basis is
%               coordinate-singular ('local-sp' on axis); [] otherwise
%
%   The two traces are geometry-identical by construction (polarization
%   does not steer rays); this is asserted bitwise.  The pre-call
%   polarization state is restored on exit.
%
%   See also: macos.pol_maps, macos.ray_field, macos.polarization.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
    opts.basis (1,:) char {mustBeMember(opts.basis, ...
        {'double-pole','local-sp','global'})} = 'double-pole'
    opts.axis (3,1) double = [0;0;0]     % sentinel: derive from rays
    opts.xref (3,1) double = [0;0;0]     % sentinel: derive from axis
end

% -- save pol state to restore afterwards ---------------------------------
s0 = macos.polarization();

% -- trace 1: x-state (1,0) -----------------------------------------------
macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
macos.trace(srf);
rfx = macos.ray_field(srf);

% -- trace 2: y-state (0,1) -----------------------------------------------
macos.polarization('on', 'Ex', [0 0], 'Ey', [1 0]);
macos.trace(srf);
rfy = macos.ray_field(srf);

% -- restore pol state ----------------------------------------------------
if s0.on
    macos.polarization('on', 'Ex', [real(s0.Ex) imag(s0.Ex)], ...
                             'Ey', [real(s0.Ey) imag(s0.Ey)]);
else
    macos.polarization('off');
end

% -- ray correspondence: the traces must be geometry-identical ------------
if ~isequal(rfx.status, rfy.status) || ...
   ~isequal(rfx.kx, rfy.kx) || ~isequal(rfx.ky, rfy.ky) || ...
   ~isequal(rfx.kz, rfy.kz)
    error('macos:jones_pupil:geometry', ...
        ['x-state and y-state traces disagree in ray geometry -- ' ...
         'polarization state is steering rays (engine bug?).']);
end

mask = (rfx.status == 0) & (rfy.status == 0);
if ~any(mask(:))
    error('macos:jones_pupil:noRays', 'no unvignetted rays at element %d', srf);
end
N = size(rfx.Ex, 1);

% unit ray directions (engine directions are unit already; normalize anyway)
kmag = sqrt(rfx.kx.^2 + rfx.ky.^2 + rfx.kz.^2);
kmag(~mask) = 1;
kx = rfx.kx ./ kmag;  ky = rfx.ky ./ kmag;  kz = rfx.kz ./ kmag;

% -- exit axis ------------------------------------------------------------
if any(opts.axis)
    ax = opts.axis / norm(opts.axis);
else
    ax = [mean(kx(mask)); mean(ky(mask)); mean(kz(mask))];
    ax = ax / norm(ax);
end

% -- reference transverse pair (xref ⊥ axis, yref = axis x xref) ----------
if any(opts.xref)
    xr = opts.xref;
else
    xr = [1;0;0];
    if abs(dot(xr, ax)) > 1 - 1e-8, xr = [0;1;0]; end
end
xr = xr - dot(xr, ax)*ax;          % project out the axis
xr = xr / norm(xr);
yr = cross(ax, xr);                % right-handed: xr x yr = ax

% -- per-point transverse basis (e1, e2), each N x N x 3 ------------------
singular = [];
switch opts.basis
    case 'double-pole'
        % Rodrigues rotation carrying ax -> khat along the great circle,
        % applied to (xr, yr).  r = ax x k, sin(th) = |r|, cos(th) = ax.k.
        cth = ax(1)*kx + ax(2)*ky + ax(3)*kz;
        rx_ = ax(2)*kz - ax(3)*ky;
        ry_ = ax(3)*kx - ax(1)*kz;
        rz_ = ax(1)*ky - ax(2)*kx;
        sth = sqrt(rx_.^2 + ry_.^2 + rz_.^2);
        onax = sth < 1e-12;                    % k ~ axis: basis = (xr, yr)
        sthn = sth;  sthn(onax) = 1;           % avoid 0/0
        ux = rx_./sthn;  uy = ry_./sthn;  uz = rz_./sthn;  % rotation axis
        [e1x,e1y,e1z] = rodrigues_(xr, ux,uy,uz, cth, sth, onax);
        [e2x,e2y,e2z] = rodrigues_(yr, ux,uy,uz, cth, sth, onax);
    case 'local-sp'
        % s = k x ax (normalized), p = s x k;  (p, s, k) with p x s = k...
        % convention here: e1 = p (meridional), e2 = s; e1 x e2 = k.
        sx = ky*ax(3) - kz*ax(2);
        sy = kz*ax(1) - kx*ax(3);
        sz = kx*ax(2) - ky*ax(1);
        smag = sqrt(sx.^2 + sy.^2 + sz.^2);
        singular = mask & (smag < 1e-9);       % on-axis: s/p undefined
        smagn = smag;  smagn(smag < 1e-9) = 1;
        sx = sx./smagn;  sy = sy./smagn;  sz = sz./smagn;
        e1x = sy.*kz - sz.*ky;  e1y = sz.*kx - sx.*kz;  e1z = sx.*ky - sy.*kx;
        e2x = sx;  e2y = sy;  e2z = sz;
        % fill singular points with the reference pair so numbers exist
        e1x(singular) = xr(1); e1y(singular) = xr(2); e1z(singular) = xr(3);
        e2x(singular) = yr(1); e2y(singular) = yr(2); e2z(singular) = yr(3);
    case 'global'
        e1x = xr(1)*ones(N); e1y = xr(2)*ones(N); e1z = xr(3)*ones(N);
        e2x = yr(1)*ones(N); e2y = yr(2)*ones(N); e2z = yr(3)*ones(N);
end

% -- assemble J -----------------------------------------------------------
J = nan(N, N, 2, 2);  J = complex(J, J);
J(:,:,1,1) = e1x.*rfx.Ex + e1y.*rfx.Ey + e1z.*rfx.Ez;
J(:,:,2,1) = e2x.*rfx.Ex + e2y.*rfx.Ey + e2z.*rfx.Ez;
J(:,:,1,2) = e1x.*rfy.Ex + e1y.*rfy.Ey + e1z.*rfy.Ez;
J(:,:,2,2) = e2x.*rfy.Ex + e2y.*rfy.Ey + e2z.*rfy.Ez;
nanm = ~mask;
for a = 1:2
    for b = 1:2
        Jab = J(:,:,a,b);  Jab(nanm) = NaN + 1i*NaN;  J(:,:,a,b) = Jab;
    end
end

% -- longitudinal-leak diagnostic -----------------------------------------
lk = zeros(N);
for rf = {rfx, rfy}
    r = rf{1};
    Edotk = r.Ex.*kx + r.Ey.*ky + r.Ez.*kz;
    Emag  = sqrt(abs(r.Ex).^2 + abs(r.Ey).^2 + abs(r.Ez).^2);
    Emag(Emag == 0) = 1;
    lk = max(lk, abs(Edotk)./Emag);
end
leak = max(lk(mask));

jp = struct('J', J, 'mask', mask, 'basis', opts.basis, ...
            'axis', ax, 'xref', xr, 'yref', yr, ...
            'kx', kx, 'ky', ky, 'kz', kz, ...
            'leak', leak, 'singular', singular);
end

function [vx,vy,vz] = rodrigues_(v0, ux,uy,uz, cth, sth, onax)
% Rotate constant vector v0 by angle th about per-point unit axis u:
%   v = v0*cos + (u x v0)*sin + u*(u.v0)*(1-cos)
cx = uy*v0(3) - uz*v0(2);
cy = uz*v0(1) - ux*v0(3);
cz = ux*v0(2) - uy*v0(1);
ud = ux*v0(1) + uy*v0(2) + uz*v0(3);
vx = v0(1)*cth + cx.*sth + ux.*ud.*(1-cth);
vy = v0(2)*cth + cy.*sth + uy.*ud.*(1-cth);
vz = v0(3)*cth + cz.*sth + uz.*ud.*(1-cth);
vx(onax) = v0(1);  vy(onax) = v0(2);  vz(onax) = v0(3);
end
