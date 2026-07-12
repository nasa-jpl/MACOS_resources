function dldx = dldx_analytic(bodies, src_pts, tgt_pts, src_body, tgt_body)
%DLDX_ANALYTIC  Closed-form MET Jacobian for straight-line gauges.
%
%   dldx = macos.design.dldx_analytic(BODIES, SRC, TGT, SB, TB)
%   returns the nbeam x 6*numel(BODIES) sensitivity of each gauge
%   length l = |s - t| to the rigid DOFs of the bodies carrying its
%   endpoints — the analytic equivalent of the engine-FD
%   macos.design.dmet_dx (same column convention: per body
%   [rot_xyz | trans_xyz] in the body's LOCAL triad; x in SI rad/m,
%   l in SI m; positions in BaseUnits mm are fine — only unit vectors
%   and metre-converted moment arms enter).
%
%   BODIES: 1xN struct with .rpt (3x1, mm) and .T (3x3 triad columns)
%   SRC/TGT: 3 x nbeam endpoint positions (mm, global)
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
end
nb = size(src_pts, 2);
dldx = zeros(nb, 6*numel(bodies));
for q = 1:nb
    d = src_pts(:,q) - tgt_pts(:,q);
    u = d / norm(d);
    dldx(q,:) = dldx(q,:) + row_(bodies, src_body(q), src_pts(:,q), u, +1);
    dldx(q,:) = dldx(q,:) + row_(bodies, tgt_body(q), tgt_pts(:,q), u, -1);
end
end

function r = row_(bodies, b, p, u, sgn)
r = zeros(1, 6*numel(bodies));
if b == 0, return; end
T = bodies(b).T;
arm = (p - bodies(b).rpt) * 1e-3;          % mm -> m moment arm
c = (b-1)*6;
for d = 1:3
    r(c+d)   = sgn * dot(u, cross(T(:,d), arm));   % rot_d, per rad
end
r(c+(4:6)) = sgn * (u' * T);                        % trans, per m
end
