function PL = tg96_place(A, ix, cfg, msk, N_G, DX_G, POKE, measf, place, h0)
%TG96_PLACE  Window placement from the ray-traced affine (Dave 2026-09-11).
%   Maps every DM actuator lattice point to a DETECTOR PIXEL (col,row) of the
%   measured wavefront grid, using the full ray affine (dmg_frame: flip,
%   rotation, scale, shift) composed with the FIXED field-array parity (the
%   deck-dependent 180/reflection between the ray geometry and the diffracted
%   field, opd_conventions.md).  The affine carries the fold's non-90-deg
%   rotation that the 8-parity search of register_two_pokes/dmg_register
%   cannot express; the residual field parity is resolved ONCE against a
%   measured in-pupil reference poke -- not a per-actuator search, and not
%   faking a rotation with a parity.
%
%   Inputs
%     A     arm descriptor (arm_desc): .rx .iTO .iDET ...
%     ix    G.T index struct (.iTO .iDET)
%     cfg   P.dm(1): .nact .pitch
%     msk   detector illuminated mask (field grid, logical N_WF x N_WF)
%     N_G, DX_G  DM surface grid size / pitch (mm)
%     POKE  reference-poke command (mm)
%     measf @(M) -> height map h (field grid) for a DM surface map M(N_G,N_G)
%     place P.place: .resolve
%   Output PL
%     .U .V     nact x nact, predicted detector pixel (col=U, row=V) of every actuator
%     .frm      the dmg_frame affine struct
%     .mag .dxd_mm
%     .lit      nact x nact logical, illuminated actuators
%     .anchor   [bx by tax tay]  (anchor blob px <-> DM mm)
%     .parity   [t su sv]  the resolved field parity (t: transpose u2<->v2)
%     .ref      diagnostics for the reference poke resolution
%     .axg .ayg actuator lattice grids (mm), x=axg (col), y=ayg (row)

macos.load_rx(A.rx);
[mag, dxd_mm, frm] = dmg_frame(ix.iTO, ix.iDET);
xg = ((1:N_G)-(N_G+1)/2)*DX_G;
lat = ((1:cfg.nact)-(cfg.nact+1)/2)*cfg.pitch;
[axg, ayg] = meshgrid(lat);                       % axg = x (col), ayg = y (row)
lit = dmg_lit(msk, dxd_mm, mag, axg, ayg);

% ---- footprint centroid in actuator lattice (the in-pupil anchor site) ---
macos.load_rx(A.rx);  st = macos.trace(ix.iTO);  ri = macos.get_ray_info(st.nRays);
ok = ri.ok_trace(:) & ri.ok_pass(:);
psi1 = macos.get_elt_psi(ix.iTO);  vpt1 = macos.get_elt_vpt(ix.iTO);
u1 = macos.design.Bench.perp(psi1);  v1 = cross(psi1, u1);
dd = ri.pos(:,ok) - vpt1;
ax_c = (u1.'*dd)/cfg.pitch + (cfg.nact+1)/2;      % actuator index along u1  (-> col? resolved by parity)
ax_r = (v1.'*dd)/cfg.pitch + (cfg.nact+1)/2;      % actuator index along v1
cc0 = round(median(ax_c));  rr0 = round(median(ax_r));
hw_c = 0.5*(max(ax_c)-min(ax_c));  hw_r = 0.5*(max(ax_r)-min(ax_r));
cl = @(x) min(max(round(x),1), cfg.nact);
% Anchor + TWO DIRECTIONAL reference pokes, all in-pupil but OFF the
% chief/centre pixel (the four-step map is referenced there, so a centre poke
% reads ~0 -- Dave's reference gotcha).  The refs step in ONE lattice axis each
% (not a diagonal) so they DISAMBIGUATE the transpose parity (a diagonal ref
% cannot tell col<->row from row<->col).
aR  = [cl(rr0 - 0.10*hw_r), cl(cc0 - 0.10*hw_c)];   % anchor (near centre, off it)
bRc = [aR(1),                cl(cc0 + 0.35*hw_c)];  % +column step (x)
bRr = [cl(rr0 + 0.35*hw_r),  aR(2)];               % +row step (y)

% ---- measure anchor + the two directional pokes DIFFERENTIALLY (poke frame
%      minus the flat frame h0 removes the OAP's low-order null background that
%      would bias a weak poke's CoM), then mean-referenced over the mask ------
if nargin < 10 || isempty(h0), h0 = zeros(size(msk)); end
mkref = @(h) (h - h0) - median(h(msk) - h0(msk));
MaA = dm_influence_map(N_G, DX_G, 'nact',cfg.nact,'pitch',cfg.pitch,'act',POKE*(sparse_poke_(cfg.nact,aR)));
hA = mkref(measf(MaA));
[bx, by, tax, tay] = dmg_anchor(hA, MaA, msk, frm.N, xg);
% dmg_anchor: (bx,by)=blob (col,row) px of the anchor; (tax,tay)=truth peak (x,y) mm
[bxc, byc] = blob_com_(mkref(measf(dm_influence_map(N_G,DX_G,'nact',cfg.nact,'pitch',cfg.pitch,'act',POKE*sparse_poke_(cfg.nact,bRc)))), msk, frm.N);
[bxr, byr] = blob_com_(mkref(measf(dm_influence_map(N_G,DX_G,'nact',cfg.nact,'pitch',cfg.pitch,'act',POKE*sparse_poke_(cfg.nact,bRr)))), msk, frm.N);

% ---- the affine linear part: DM-mm -> detector-mm (u2,v2) ---------------
% DM-mm of an actuator relative to the anchor's truth peak.  x pairs with the
% lattice column, y with the row (dmg_anchor's tax=xg(col), tay=xg(row)).
dmx = axg - tax;  dmy = ayg - tay;                % DM-mm (component1=x, component2=y)
% detmm = Linv * [dmx; dmy]  (Linv from the ray affine -- carries the fold
% rotation/reflection); (u2,v2)->pixel via the resolved field parity below.
du = frm.Linv(1,1)*dmx + frm.Linv(1,2)*dmy;       % detector-mm along u2
dv = frm.Linv(2,1)*dmx + frm.Linv(2,2)*dmy;       % detector-mm along v2

% ---- resolve the field-array parity (t,su,sv) against BOTH directional
% pokes: row<->u2, col<->v2 (opd_conventions); t swaps the assignment, su/sv
% the signs.  Anchor pins the offset (dmx=dmy=0 there -> (bx,by)). ----------
tgt = [bRc(1) bRc(2) bxc byc; bRr(1) bRr(2) bxr byr];   % [r c measCol measRow]
cand = [];  errc = [];
for t = 0:1
  for su = [-1 1]
    for sv = [-1 1]
      [Uc, Vc] = tg96_apply_parity(du, dv, dxd_mm, bx, by, [t su sv]);
      e = 0;
      for j = 1:size(tgt,1)
          e = e + hypot(Uc(tgt(j,1),tgt(j,2))-tgt(j,3), Vc(tgt(j,1),tgt(j,2))-tgt(j,4));
      end
      cand(end+1,:) = [t su sv]; %#ok<AGROW>
      errc(end+1,1) = e;         %#ok<AGROW>
    end
  end
end
if place.resolve, [~, ib] = min(errc);  else, ib = 1; end
par = cand(ib,:);
[U, V] = tg96_apply_parity(du, dv, dxd_mm, bx, by, par);
bR = bRc;  bx2 = bxc;  by2 = byc;                   % (kept for the ref diagnostic)

PL = struct('U',U, 'V',V, 'frm',frm, 'mag',mag, 'dxd_mm',dxd_mm, 'lit',lit, ...
    'anchor',[bx by tax tay], 'parity',par, 'axg',axg, 'ayg',ayg, 'xg',{xg}, 'lat',{lat}, ...
    'ref',struct('aR',aR, 'bR',bR, 'meas',[bx2 by2], 'cand',cand, 'err',errc, 'ib',ib));
end

% ---------------------------------------------------------------------------
function A = sparse_poke_(nact, rc)
% unit command map with a single poked actuator at (row,col) = rc
A = zeros(nact);  A(rc(1), rc(2)) = 1;
end

function [cx, cy] = blob_com_(h, msk, N)
% centroid (col,row) px of a single-poke response over the mask
w = abs(h);  w(~msk) = 0;  w(w < 0.5*max(w(:))) = 0;
[cg, rg] = meshgrid(1:N, 1:N);
cx = sum(cg(:).*w(:))/sum(w(:));  cy = sum(rg(:).*w(:))/sum(w(:));
end
