function [U, V] = tg96_apply_parity(du, dv, dxd_mm, bx, by, par)
%TG96_APPLY_PARITY  Detector-mm (u2,v2) -> field pixel (col=U, row=V).
%   The mmacos field grid indexes row<->u2=perp(psi), col<->v2=cross(psi,u2)
%   (opd_conventions.md); the deck-dependent field parity par=[t su sv] gives
%   the transpose t (swap the u2/v2 assignment) and per-axis signs su,sv, and
%   (bx,by) is the anchor pixel that pins the offset (du=dv=0 -> (bx,by)).
%   Shared by tg96_place (window placement) and the D1 non-vacuity check.
t = par(1);  su = par(2);  sv = par(3);
if t == 0
    V = by + su*du/dxd_mm;   U = bx + sv*dv/dxd_mm;      % row from u2, col from v2
else
    V = by + su*dv/dxd_mm;   U = bx + sv*du/dxd_mm;      % transposed assignment
end
end
