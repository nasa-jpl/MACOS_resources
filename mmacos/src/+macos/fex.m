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
try
    xp = mmacos('xp_fnd', mode);      % 7-vec [Kr, psi(3), vpt(3)]
catch
    % The dispatcher raises a terse 'mmacos: xp_fnd failed' when xp_fnd
    % returns OK=false -- which happens only when the Rx isn't loaded or
    % (the usual cause) no aperture stop is set, leaving the chief ray and
    % hence the pupil undefined.  The engine has already printed the same
    % remedy on stdout; re-throw with an actionable identifier so callers
    % (dw_d*_multi's reset_xp, pupil_quality, ...) abort cleanly instead
    % of building a wavefront on a bogus pupil.
    error('macos:fex:noStop', ...
        ['fex could not find the exit pupil: no aperture stop is set, ' ...
         'so the chief ray (and hence the pupil) is undefined.  Add ' ...
         '"ApStop= 0 0 0" to the Rx header (stop at the primary), or ' ...
         'call macos.stop(iElt) before macos.fex.']);
end
[vpt, psi, rad] = mmacos('xp_get');
s.vpt = vpt(:);
s.psi = psi(:);
s.rad = rad;
s.xp  = xp(:);
end
