function [mag, dxd_mm, frm] = dmg_frame(iTO, iDET)
%DMG_FRAME  Ray-measured DM->detector affine (registration DOF 1).
%   [mag, dxd_mm] = dmg_frame(iTO, iDET) traces the loaded bench and fits
%   the affine map detector-xy -> DM-xy over the surviving rays; mag is
%   sqrt(|det|) in DM-mm per detector-mm, dxd_mm the detector pixel pitch.
%   Frame-before-angle: the support-area radius estimate was 25% off on
%   this bench (S1 rounds 8-9); the ray affine is the frame authority.
%   Extracted verbatim from zwfs_s3/tg96_s3 @ 10cf593.
%
%   [mag, dxd_mm, frm] = dmg_frame(iTO, iDET) additionally returns the FULL
%   ray affine so a DM lattice point can be placed at a detector pixel
%   DIRECTLY -- flip, rotation, scale and shift are all in the fit, so a
%   folded (OAP) rig whose detector<->DM mapping is a non-90-deg rotation is
%   handled where the 8-parity search of dmg_register cannot (Dave 2026-09-10).
%   frm fields (all in the ray-geometry transverse frames row<->u, col<->v of
%   the mmacos field grid; see doc/opd_conventions.md):
%     .Aaf    2x3, detector-mm(rel first ray) -> DM-mm  (the raw fit)
%     .Lm     2x2 linear part (detector-mm(u2,v2) -> DM-mm(u1,v1))
%     .Linv   2x2 inverse (DM-mm -> detector-mm) -- carries the fold rotation
%     .dxd_mm detector pixel pitch (mm)
%     .u2 .v2 detector in-plane basis (u2=perp(psi), v2=cross(psi,u2))
%     .vpt2   detector vertex (3-vec)
%     .chief  [col row] geometric pixel of the DM-centre (chief) ray, grid
%             centre + its projection/dxd; the reference the diffracted-field
%             anchor (dmg_anchor on an in-pupil poke) refines for the true
%             field-array parity.
%     .N      field-grid size (px);  .cen = (N+1)/2 centre pixel
s1t = macos.trace(iTO);   ito  = macos.get_ray_info(s1t.nRays);
s2t = macos.trace(iDET);  idet = macos.get_ray_info(s2t.nRays);
okr = ito.ok_trace(:) & ito.ok_pass(:) & idet.ok_trace(:) & idet.ok_pass(:);
psi1 = macos.get_elt_psi(iTO);  vpt1 = macos.get_elt_vpt(iTO);
u1 = macos.design.Bench.perp(psi1);  v1 = cross(psi1, u1);
xy_to = [u1.'; v1.'] * (ito.pos - vpt1);
psi2 = macos.get_elt_psi(iDET);  vpt2 = macos.get_elt_vpt(iDET);
u2 = macos.design.Bench.perp(psi2);  v2 = cross(psi2, u2);
xy_d = [u2.'; v2.'] * (idet.pos - idet.pos(:,1));
Aaf = [xy_d(:,okr).' ones(nnz(okr),1)] \ xy_to(:,okr).';
mag = sqrt(abs(det(Aaf(1:2,:).')));
dxd_mm = abs(macos.dx_at(iDET, 'mm'));
if nargout >= 3
    Lm = Aaf(1:2,:).';                          % detector-mm -> DM-mm
    % geometric pixel of the chief (DM-centre) ray: grid centre + its detector
    % projection / pitch.  row<->u2, col<->v2 (opd_conventions.md); the true
    % field-array parity is resolved against a measured in-pupil poke.
    [~, ich] = min(hypot(xy_to(1,:).', xy_to(2,:).') + ~okr*1e9);
    du = u2.' * (idet.pos(:,ich) - vpt2);  dv = v2.' * (idet.pos(:,ich) - vpt2);
    N = size(macos.complex_field(iDET), 1);  cen = (N+1)/2;
    frm = struct('Aaf',Aaf, 'Lm',Lm, 'Linv',inv(Lm), 'dxd_mm',dxd_mm, ...
        'u2',u2(:), 'v2',v2(:), 'vpt2',vpt2(:), 'N',N, 'cen',cen, ...
        'chief',[cen + dv/dxd_mm, cen + du/dxd_mm]);   % [col row]
end
end
