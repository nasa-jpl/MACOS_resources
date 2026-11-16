function out = met(unit)
%MACOS.MET  Laser-metrology beam lengths (engine METcalc).
%   out = macos.met() runs the engine metrology compute over the loaded
%   Rx's declared met beams (the nMetPos / tMetElt / metBeamFlg element
%   keywords) and returns
%       out.l   n-by-1 beam lengths, SI metres (default)
%       out.n   system beam count (0 when the Rx declares no metrology)
%   out = macos.met('native') returns lengths in the Rx BaseUnits.
%
%   The engine model (SrfMetCalc) is the straight-line point-to-point
%   distance between the source element's met points and its target
%   element's met points, in the global frame — no line-of-sight /
%   obscuration check.  Met points ride their element under the
%   programmatic perturb path, so perturb -> met() yields finite-
%   difference metrology Jacobians (design-layer dmet_dx).
arguments
    unit (1,:) char {mustBeMember(unit, {'m','native'})} = 'm'
end
n = mmacos('met_calc');
if n == 0
    out = struct('l', zeros(0,1), 'n', 0);
    return
end
l = mmacos('met_get', n);
l = l(:);
if strcmp(unit, 'm')
    c = mmacos('base_unit_to_metres');
    if c == 0.0
        error('macos:met:noCBM', ...
            'CBM unavailable (Rx not loaded or BaseUnits not declared)');
    end
    l = l * c;
end
out = struct('l', l, 'n', n);
end
