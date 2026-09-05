function [mag, dxd_mm] = dmg_frame(iTO, iDET)
%DMG_FRAME  Ray-measured DM->detector affine scale (registration DOF 1).
%   [mag, dxd_mm] = dmg_frame(iTO, iDET) traces the loaded bench and fits
%   the affine map detector-xy -> DM-xy over the surviving rays; mag is
%   sqrt(|det|) in DM-mm per detector-mm, dxd_mm the detector pixel pitch.
%   Frame-before-angle: the support-area radius estimate was 25% off on
%   this bench (S1 rounds 8-9); the ray affine is the frame authority.
%   Extracted verbatim from zwfs_s3/tg96_s3 @ 10cf593.
s1t = macos.trace(iTO);   ito  = macos.get_ray_info(s1t.nRays);
s2t = macos.trace(iDET);  idet = macos.get_ray_info(s2t.nRays);
okr = ito.ok_trace(:) & ito.ok_pass(:) & idet.ok_trace(:) & idet.ok_pass(:);
psi1 = macos.get_elt_psi(iTO);  vpt1 = macos.get_elt_vpt(iTO);
u1 = macos.design.Bench.perp(psi1);  v1 = cross(psi1, u1);
xy_to = [u1.'; v1.'] * (ito.pos - vpt1);
psi2 = macos.get_elt_psi(iDET);
u2 = macos.design.Bench.perp(psi2);  v2 = cross(psi2, u2);
xy_d = [u2.'; v2.'] * (idet.pos - idet.pos(:,1));
Aaf = [xy_d(:,okr).' ones(nnz(okr),1)] \ xy_to(:,okr).';
mag = sqrt(abs(det(Aaf(1:2,:).')));
dxd_mm = abs(macos.dx_at(iDET, 'mm'));
end
