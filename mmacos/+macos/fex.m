function s = fex(mode)
%MACOS.FEX  Find the Exit Pupil (FEX), return focal length + geometry.
%   s = macos.fex(MODE) runs the exit-pupil finder and returns:
%     .f    exit-pupil focal length / reference radius (the XP value)
%     .vpt  3×1 pupil vertex (BaseUnits, global frame)
%     .psi  3×1 pupil-surface normal (global)
%     .rad  reference-sphere radius (BaseUnits)
%
%   MODE defaults to 1.  Requires more than 3 elements defined.
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
f = mmacos('xp_fnd', mode);
[vpt, psi, rad] = mmacos('xp_get');
s.f   = f;
s.vpt = vpt(:);
s.psi = psi(:);
s.rad = rad;
end
