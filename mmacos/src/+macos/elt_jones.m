function J = elt_jones(srf)
%MACOS.ELT_JONES  The 2x2 Jones matrix a polarizing element applied.
%   J = macos.elt_jones(SRF) returns the complex 2x2 Jones matrix that the
%   polarizing element at SRF applied during the last trace, expressed in
%   the element's OWN transverse eigenbasis: column/row 1 is the declared
%   axis projected into the ray's transverse plane, 2 is its orthogonal
%   partner (rhat x ahat).
%
%   Diagonal by construction:
%     ideal polarizer   diag(1, 0)
%     waveplate         diag(1, exp(-i*2*pi*R))
%
%   This reads elt_mod's JmatElt, which the trace fills -- an array that was
%   allocated and dead before PLAN_POLARIZATION Phase 3.
%
%   PER ELEMENT, NOT PER RAY, and that is exact rather than an
%   approximation: the coefficients of an ideal polarizer or retarder in its
%   own eigenbasis do not depend on the ray. All of the ray dependence
%   lives in the BASIS, which is why this is not a substitute for
%   macos.jones_pupil -- use that for the pupil-referenced Jones.
%
%   Requires a trace with macos.polarization('on'); returns zeros otherwise
%   (a trace that never ran cannot have applied anything).
%
%   Example -- confirm a quarter-wave plate is unitary:
%     macos.trace(6);
%     J = macos.elt_jones(3);
%     norm(J'*J - eye(2))     % ~1e-16
%
%   See also: macos.polarizer, macos.waveplate, macos.jones_pupil.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end

[re, im] = mmacos('jmat_elt_get', srf);
J = re + 1i*im;
end
