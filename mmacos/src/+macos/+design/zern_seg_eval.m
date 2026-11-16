function z = zern_seg_eval(frame, mode, P)
%ZERN_SEG_EVAL  Engine-exact segment MonZernike mode value at world points.
%
%   z = macos.design.zern_seg_eval(FRAME, MODE, P) evaluates the surface
%   sag contribution of a unit MonZernCoef(MODE) poke at the 3 x n world
%   points P on the segment whose face frame is FRAME (a segment_rx /
%   seg_from_rx frames entry: .rpt .xhat .yhat .lmon).  This replicates
%   the engine's Mon channel exactly for `MonZernType= ANSI` segments
%   (the segment_rx emission): local coordinates x = xhat.(p-rpt)/lMon,
%   y = yhat.(p-rpt)/lMon (surfsub.F MonomialEval call site), then the
%   UN-normalized ANSI Zernike radial polynomial x cos/sin -- the
%   NORM_RMS factor applies only to the Norm* ZernTypes (surfsub.F
%   iZernTypeForNorm gating, the E1 fix), so it is NOT applied here.
%
%   Engine-exactness is gated by tRunCompare's grid-vs-MonZern
%   equivalence test: the same mode sampled onto a grid channel and
%   poked via elt_grid_add must reproduce the MonZernCoef poke's OPD.
%
%   Used by macos.design.dmet_dfig to evaluate figure motion at edge-
%   sensor and MET-launcher mount points (dmdz / dmdgrid generation).
%
%   See also: macos.design.dmet_dfig, macos.segment_grid_basis.

arguments
    frame (1,1) struct
    mode (1,1) double {mustBeInteger, mustBePositive}
    P (3,:) double
end
d = P - frame.rpt;
u = (frame.xhat.' * d) / frame.lmon;
v = (frame.yhat.' * d) / frame.lmon;
rho = hypot(u, v);
th = atan2(v, u);

% ANSI (j-1) -> (n, m); un-normalized radial polynomial (ZernType_ANSI)
jj = mode - 1;
n = ceil((-3 + sqrt(9 + 8*jj)) / 2);
m = 2*jj - n*(n + 2);
am = abs(m);
R = zeros(size(rho));
for s = 0:((n - am)/2)
    c = (-1)^s * factorial(n - s) / ...
        (factorial(s) * factorial((n + am)/2 - s) * factorial((n - am)/2 - s));
    R = R + c * rho.^(n - 2*s);
end
if m >= 0, ang = cos(m*th); else, ang = sin(am*th); end
z = R .* ang;
end
