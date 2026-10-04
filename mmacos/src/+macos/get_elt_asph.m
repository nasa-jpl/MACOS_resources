function c = get_elt_asph(srf, n)
%MACOS.GET_ELT_ASPH  Even-radial aspheric coefficients (AsphCoef: h^4, h^6, ...) of an element.
%   c = macos.get_elt_asph(srf) returns the first 4 coefficients (the Rx
%   default count); c = macos.get_elt_asph(srf, n) the first n (n <= 9).
%   What CALIB's OptAsph= DOFs leave on the element (engine 2026-10-04).
if nargin < 2, n = 4; end
c = mmacos('elt_asph_get', double(n), double(srf));
c = c(:).';
end
