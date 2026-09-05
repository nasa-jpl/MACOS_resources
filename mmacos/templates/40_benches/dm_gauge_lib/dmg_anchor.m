function [bx, by, tax, tay] = dmg_anchor(hA, Ma, msk, N_WF, xg)
%DMG_ANCHOR  Translation registration from ONE CENTER poke (DOF 2).
%   Blob centroid of the measured poke-A map (detector px) paired with
%   the truth poke's peak position (DM mm).  The center is
%   parity-invariant -- right for translation, useless for parity
%   (registration doctrine).  Extracted verbatim from zwfs_s2/tg96_s3.
wA = abs(hA);  wA(~msk) = 0;  wA(wA < 0.5*max(wA(:))) = 0;
[cg, rg] = meshgrid(1:N_WF, 1:N_WF);
bx = sum(cg(:).*wA(:))/sum(wA(:));  by = sum(rg(:).*wA(:))/sum(wA(:));
[~, im] = max(abs(Ma(:)));  [tr, tc] = ind2sub(size(Ma), im);
tax = xg(tc);  tay = xg(tr);
end
