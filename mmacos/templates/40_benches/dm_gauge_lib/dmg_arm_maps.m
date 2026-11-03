function [qL, qR, info] = dmg_arm_maps(iSRF, msk, laser_deg)
%DMG_ARM_MAPS  The test arm's polarization aberration per circular channel.
%   [qL, qR, info] = dmg_arm_maps(iSRF, msk, laser_deg) traces the LOADED
%   bench twice with polarization on and vector diffraction on (source
%   states x and y), assembles the 2x2 Jones matrix J at every pixel of the
%   diffraction grid at element iSRF -- the mask sandwich's ENTRANCE sphere
%   (iMASK-1), where the grid is the detector's (gate G1) and the beam has
%   passed every arm optic but not the mask; the field lens behind the
%   mask acts on the already-masked images (a flat-field factor, not a
%   pupil aberration) and is left out -- in the beam's transverse basis
%   there, strips its common scalar
%   (Jhat = J / sqrt(det J): diattenuation, retardance and rotation stay,
%   the polarized trace's own amplitude and phase go), applies the laser's
%   linear state e_in = (cos, sin) of laser_deg from the source x axis, and
%   returns the two circular components normalized to the ideal 50/50
%   split and to unit mean power over msk for that state:
%       qL = sqrt(2) <L| Jhat e_in> / sqrt(T),   qR = sqrt(2) <R| Jhat e_in> / sqrt(T)
%   (|q| = 1 and qL = qR for a polarization-neutral arm), unit outside msk.
%   A vector Zernike sensor whose metasurface converts L -> R with the
%   +phi dimple and R -> L with -phi sees the pupil field qL.*E in one
%   image and qR.*E in the other: a per-channel aberration the scalar
%   sensor never has (dmg_zwfs_gauge V3, 2026-09-12).
%
%   WHY THE COMMON SCALAR IS STRIPPED (measured on the zwfs_dm96 train):
%   the vector-mode field's phase is the exact CONJUGATE of the scalar
%   trace's (the common phase of J/E0 fits -2.0000 x the pupil phase,
%   residual 4e-9 -- the RayE-vs-CumRayL bookkeeping of a plain
%   trace-to-detector train, macos_f90/CLAUDE.md "PHASE CONVENTION AT THE
%   SEED"), and its amplitude carries the Fresnel losses the scalar trace
%   does not.  Neither is polarization physics; the record's field is the
%   scalar trace's, the maps carry only what polarization adds to it.
%
%   info: .axis/.xref/.yref (the exit basis; the per-pixel pair is xref
%   projected into each pixel's own transverse plane and completed by the
%   ray direction -- the mask's axes as the ray sees them -- the direction
%   taken from the two traces' field vectors), .leak (max
%   |E.k|/|E| on msk: the longitudinal residual, 1e-14 class), .cone_deg
%   (the beam's half-cone at iSRF), .iSRF, .D (per-pixel
%   diattenuation map, NaN off msk), .ret (retardance map, rad), .T (mean
%   transmittance |sqrt(det J)|^2 on msk, the Fresnel budget), .T_state
%   (the laser state's relative transmission through the polarization part,
%   divided out of the maps), .common_slope
%   (the fitted slope of the common phase vs the pupil phase; -2 = the
%   conjugate convention above), .stats (means/rms/max of D, ret; the
%   per-channel phase and amplitude variation; the channel difference),
%   .e_in, .laser_deg.  The pre-call polarization state is restored.
%
%   See also: macos.jones_pupil (per RAY; used here for the exit basis),
%   macos.complex_field 'plane', dmg_zwfs_gauge.
if nargin < 3 || isempty(laser_deg), laser_deg = 45; end
N = size(msk, 1);  np = nnz(msk);
s0 = macos.polarization();
jp = macos.jones_pupil(iSRF, 'basis', 'global');          % the exit axis + reference pair
ax = jp.axis;  xr = jp.xref;  yr = jp.yref;
Jx = cell(1,3);  Jy = cell(1,3);
macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);  macos.vector_diffraction(true);
for k = 1:3, Jx{k} = macos.complex_field(iSRF, 'plane', k, 'reset_trace', k == 1); end
macos.polarization('on', 'Ex', [0 0], 'Ey', [1 0]);
for k = 1:3, Jy{k} = macos.complex_field(iSRF, 'plane', k, 'reset_trace', k == 1); end
macos.vector_diffraction(false);
if s0.on
    macos.polarization('on', 'Ex', [real(s0.Ex) imag(s0.Ex)], 'Ey', [real(s0.Ey) imag(s0.Ey)]);
else
    macos.polarization('off');
end
E0 = macos.complex_field(iSRF);                           % the scalar field again (the record's)
% the per-pixel transverse basis: the entrance sphere sits in L2's
% converging cone, so a single transverse pair mis-projects the edge
% pixels (a 9% longitudinal residual there fakes a 0.4% diattenuation, the
% size of the physics).  Each pixel's ray direction is the normal to the
% plane its two field vectors span, k = unit(E(x) x E(y)).  The pair is
% the MASK's own axes projected into that ray's transverse plane (x' =
% unit(xr - (xr.k) k), y' = k x x'): the basis a thin polarizing element
% acts in at oblique incidence (the engine's settled convention for its
% own polarizers and waveplates -- project the material axis; Korger et
% al. 2013), so the L / R channels are the ones the metasurface converts.
% NOT the double-pole pair: carrying (xr, yr) along the great circle adds
% a rotation (theta^2/4) sin 2 alpha per pixel that the mask never sees --
% 1.6 mrad rms of fake channel difference on this 5-deg cone (measured
% 2026-09-12 with the laser on the tilted faces' s axis, where the
% projected basis gives 2e-5).
kx = real(Jx{2}.*conj(Jy{3}) - Jx{3}.*conj(Jy{2}));
ky = real(Jx{3}.*conj(Jy{1}) - Jx{1}.*conj(Jy{3}));
kz = real(Jx{1}.*conj(Jy{2}) - Jx{2}.*conj(Jy{1}));
kn = sqrt(kx.^2 + ky.^2 + kz.^2);  kn(~msk | kn == 0) = 1;
kx = kx./kn;  ky = ky./kn;  kz = kz./kn;
sg = sign(kx*ax(1) + ky*ax(2) + kz*ax(3));  sg(sg == 0) = 1;
kx = kx.*sg;  ky = ky.*sg;  kz = kz.*sg;
kx(~msk) = ax(1);  ky(~msk) = ax(2);  kz(~msk) = ax(3);
cth = ax(1)*kx + ax(2)*ky + ax(3)*kz;
xk = xr(1)*kx + xr(2)*ky + xr(3)*kz;                     % xr projected out of k, normalized
e1x = xr(1) - xk.*kx;  e1y = xr(2) - xk.*ky;  e1z = xr(3) - xk.*kz;
en = sqrt(e1x.^2 + e1y.^2 + e1z.^2);  e1x = e1x./en;  e1y = e1y./en;  e1z = e1z./en;
e2x = ky.*e1z - kz.*e1y;  e2y = kz.*e1x - kx.*e1z;  e2z = kx.*e1y - ky.*e1x;   % y' = k x x'
proj = @(Ec, vx, vy, vz) Ec{1}.*vx + Ec{2}.*vy + Ec{3}.*vz;
J11 = proj(Jx, e1x, e1y, e1z);  J21 = proj(Jx, e2x, e2y, e2z);
J12 = proj(Jy, e1x, e1y, e1z);  J22 = proj(Jy, e2x, e2y, e2z);
lz = [proj(Jx, kx, ky, kz), proj(Jy, kx, ky, kz)];  ea = [abs(J11).^2 + abs(J21).^2, abs(J12).^2 + abs(J22).^2];
info = struct('axis', ax, 'xref', xr, 'yref', yr, 'laser_deg', laser_deg);
info.leak = max(abs(lz([msk msk]))) / sqrt(max(ea([msk msk])));
info.cone_deg = acosd(min(cth(msk)));                    % the beam's half-cone at iSRF
info.iSRF = iSRF;
% the common scalar per pixel and its phase convention vs the scalar trace
dJ = J11.*J22 - J12.*J21;  sJ = sqrt(dJ);
flip = real(sJ .* conj(J11)) < 0;  sJ(flip) = -sJ(flip);    % the root on J11's side
th0 = angle(E0(msk));  ph = angle(sJ(msk) ./ E0(msk));  ph = ph - mean(ph);
c1 = [ones(np,1) th0] \ ph;
info.common_slope = c1(2);  info.common_resid = std(ph - [ones(np,1) th0]*c1);
info.T = mean(abs(sJ(msk)).^2 ./ abs(E0(msk)).^2);
% the polarization part, det 1
Jh = {J11./sJ, J12./sJ; J21./sJ, J22./sJ};
for a = 1:2, for b = 1:2, t = Jh{a,b};  t(~msk) = double(a == b);  Jh{a,b} = t; end, end
% diattenuation + retardance maps (singular values; unitary part's eigenphases)
D = nan(N);  ret = nan(N);
idx = find(msk);
for i = 1:np
    k = idx(i);  M = [Jh{1,1}(k) Jh{1,2}(k); Jh{2,1}(k) Jh{2,2}(k)];
    [U, S, W] = svd(M);  s = diag(S);
    D(k) = (s(1)^2 - s(2)^2) / (s(1)^2 + s(2)^2);
    ev = eig(U*W');  ret(k) = abs(angle(ev(1)*conj(ev(2))));
end
info.D = D;  info.ret = ret;
% the laser's state through the polarization part, in the circular basis
e = [cosd(laser_deg); sind(laser_deg)];  info.e_in = e;
Ex = Jh{1,1}*e(1) + Jh{1,2}*e(2);  Ey = Jh{2,1}*e(1) + Jh{2,2}*e(2);
qL = (Ex - 1i*Ey);  qR = (Ex + 1i*Ey);                    % sqrt(2) <L|.>, sqrt(2) <R|.>
% unit mean power over the pupil for THIS laser state: the arm's throughput
% and the diattenuation's state-dependent transmission (0.25% along the
% tilted faces' s axis here) are the photon budget's business -- a bench
% takes its reference amplitude from the light it has -- not a pupil
% aberration; what stays is the pupil-VARYING part and the channel split
info.T_state = mean(abs(qL(msk)).^2 + abs(qR(msk)).^2) / 2;
qL = qL / sqrt(info.T_state);  qR = qR / sqrt(info.T_state);
qL(~msk) = 1;  qR(~msk) = 1;
r = qR(msk) ./ qL(msk);  r = r / mean(r);
pL = angle(qL(msk) / mean(qL(msk)));  pR = angle(qR(msk) / mean(qR(msk)));
st = struct('D_mean', mean(D(msk)), 'D_rms', std(D(msk)), 'D_max', max(D(msk)), ...
            'ret_mean', mean(ret(msk)), 'ret_rms', std(ret(msk)), 'ret_max', max(ret(msk)), ...
            'aL_mean', mean(abs(qL(msk))), 'aL_rms', std(abs(qL(msk))), ...
            'aR_mean', mean(abs(qR(msk))), 'aR_rms', std(abs(qR(msk))), ...
            'pL_rms', std(pL), 'pR_rms', std(pR), ...
            'dphase_rms', std(angle(r)), 'dphase_pv', max(angle(r)) - min(angle(r)), ...
            'damp_rms', std(abs(r) - 1), 'damp_pv', max(abs(r)) - min(abs(r)));
info.stats = st;
end
