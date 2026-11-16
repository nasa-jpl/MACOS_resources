function dldx = dldx_analytic(bodies, src_pts, tgt_pts, src_body, tgt_body, unit_to_m)
%DLDX_ANALYTIC  Closed-form MET Jacobian for straight-line gauges.
%
%   dldx = macos.design.dldx_analytic(BODIES, SRC, TGT, SB, TB)
%   returns the nbeam x 6*numel(BODIES) sensitivity of each gauge
%   length l = |s - t| to the rigid DOFs of the bodies carrying its
%   endpoints — the analytic equivalent of the engine-FD
%   macos.design.dmet_dx (same column convention: per body
%   [rot_xyz | trans_xyz] in the body's LOCAL triad; x in SI rad/m,
%   l in SI m; positions in BaseUnits are fine — only unit vectors
%   and metre-converted moment arms enter).
%
%   dldx_analytic(..., UNIT_TO_M) sets the BaseUnits->metres factor
%   for the rotation moment arms (positions and .rpt must share it).
%   Default 1e-3 (mm parents — the e5 heritage); pass 1 for a
%   metres-BaseUnits prescription (e.g. the e2e example), or pass
%   mmacos('base_unit_to_metres') from the loaded Rx.
%
%   BODIES: 1xN struct with .rpt (3x1, BaseUnits) and .T (3x3 triads)
%   SRC/TGT: 3 x nbeam endpoint positions (BaseUnits, global)
%   SB/TB:   1 x nbeam body index (into BODIES) owning each endpoint
%            (0 = fixed to ground, contributes nothing)
%
%   Physics: dl = u.(ds - dt), u = (s-t)/|s-t|; a rigid body's point p
%   moves by T*del + (T*th) x (p - rpt) under local (th, del).  So per
%   body b at endpoint p with sign q (+1 source, -1 target):
%     trans cols:  q * u' * T
%     rot cols:    q * u' * [T(:,d) x (p - rpt)]  (arm mm -> m: *1e-3)
%
%   Validated against dmet_dx engine FD in tMet.

arguments
    bodies (1,:) struct
    src_pts (3,:) double
    tgt_pts (3,:) double
    src_body (1,:) double
    tgt_body (1,:) double
    unit_to_m (1,1) double {mustBePositive} = 1e-3
end
nb = size(src_pts, 2);
dldx = zeros(nb, 6*numel(bodies));
for q = 1:nb
    d = src_pts(:,q) - tgt_pts(:,q);
    u = d / norm(d);
    dldx(q,:) = dldx(q,:) + row_(bodies, src_body(q), src_pts(:,q), u, +1, unit_to_m);
    dldx(q,:) = dldx(q,:) + row_(bodies, tgt_body(q), tgt_pts(:,q), u, -1, unit_to_m);
end
end

function r = row_(bodies, b, p, u, sgn, unit_to_m)
r = zeros(1, 6*numel(bodies));
if b == 0, return; end
T = bodies(b).T;
arm = (p - bodies(b).rpt) * unit_to_m;     % BaseUnits -> m moment arm
c = (b-1)*6;
for d = 1:3
    r(c+d)   = sgn * dot(u, cross(T(:,d), arm));   % rot_d, per rad
end
r(c+(4:6)) = sgn * (u' * T);                        % trans, per m
end
