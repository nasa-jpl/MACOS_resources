function coefs = get_elt_ff_zrn_coef(iElt, modes)
%MACOS.GET_ELT_FF_ZRN_COEF  Read FFZern coefficients on a FreeForm element.
%   coefs = macos.get_elt_ff_zrn_coef(IELT, MODES) returns COEFS(k) =
%   FFZernCoef(MODES(k)) on element IELT (must be FreeForm SrfType=14),
%   as a column vector.
%
%   See also: macos.set_elt_ff_zrn_coef.
arguments
    iElt  (1,1) double {mustBeInteger, mustBePositive}
    modes (:,1) double {mustBeInteger, mustBePositive}
end
n = numel(modes);
coefs0 = zeros(n, 1);
[~, coefs] = mmacos('elt_srf_ff_zrn_coef', iElt, double(modes), ...
                    coefs0, false, false, n);
coefs = double(coefs(:));
end
