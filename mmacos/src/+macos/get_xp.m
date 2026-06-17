function s = get_xp()
%MACOS.GET_XP  Exit-pupil geometry (at element nElt-1).
%   s = macos.get_xp() returns a struct:
%     .vpt  3×1 pupil vertex (BaseUnits, global frame)
%     .psi  3×1 pupil-surface normal (global)
%     .rad  reference-sphere radius (BaseUnits)
%
%   Run macos.fex first to (re)locate the exit pupil.
%   See also: macos.fex, macos.set_xp.
[vpt, psi, rad] = mmacos('xp_get');
s.vpt = vpt(:);
s.psi = psi(:);
s.rad = rad;
end
