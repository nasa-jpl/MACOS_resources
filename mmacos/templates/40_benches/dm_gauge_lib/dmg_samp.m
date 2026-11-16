function hd = dmg_samp(h, R)
%DMG_SAMP  Sample a detector-frame map into the DM frame under a parity.
%   hd = dmg_samp(H, R): DM point (x,y): candidate maps its offset from
%   the poke-A truth spot through axis-permutation R.P ([ax_x ax_y sx sy])
%   and the ray scale back to detector px about the A-blob centroid.
%   R fields: P, tax, tay, bx, by, dxd_mm, mag, msk, N_WF, gxd, gyd.
%   (Apply the measurement sign R.sgn at the CALLER -- kept explicit so
%   scripts read sgn*dmg_samp(...) as the doctrine states.)
%   Extracted verbatim from zwfs_s3/tg96_s3 @ 10cf593.
off = {R.gxd - R.tax, R.gyd - R.tay};
u = R.P(3)*off{R.P(1)}/(R.dxd_mm*R.mag) + R.bx;
v = R.P(4)*off{R.P(2)}/(R.dxd_mm*R.mag) + R.by;
hn = h;  hn(~R.msk) = NaN;
hd = interp2(1:R.N_WF, (1:R.N_WF).', hn, u, v, 'linear', NaN);
end
