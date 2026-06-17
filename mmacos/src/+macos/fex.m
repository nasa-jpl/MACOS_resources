function s = fex(mode)
%MACOS.FEX  Find the Exit Pupil (FEX), return its geometry.
%   s = macos.fex(MODE) runs the exit-pupil finder (placing the XP at
%   element nElt-1) and returns:
%     .vpt  3×1 pupil vertex (BaseUnits, global frame)
%     .psi  3×1 pupil-surface normal (global)
%     .rad  reference-sphere radius (BaseUnits; = Kr at the XP)
%     .xp   7×1 raw xp_fnd result [Kr; psi(3); vpt(3)]
%
%   MODE = 1 (default) centres on the chief ray; MODE = 0 on the
%   centroid.  Requires more than 3 elements and a STOP set (no STOP →
%   the underlying xp_fnd fails and this errors).
%   This is pymacos's fex(); for the chief-ray-to-FP variant see
%   macos.sxp.
%
%   Raw equivalent: mmacos('xp_fnd', mode) + mmacos('xp_get').
%   See also: macos.get_xp, macos.set_xp, macos.sxp.
arguments
    mode (1,1) double = 1
end
if macos.num_elt() <= 3
    error('macos:fex:tooFewElts', ...
        'fex: needs more than 3 elements (have %d).', macos.num_elt());
end
xp = mmacos('xp_fnd', mode);          % 7-vec [Kr, psi(3), vpt(3)]
[vpt, psi, rad] = mmacos('xp_get');
s.vpt = vpt(:);
s.psi = psi(:);
s.rad = rad;
s.xp  = xp(:);
end
