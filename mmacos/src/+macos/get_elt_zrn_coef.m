function coefs = get_elt_zrn_coef(iElt, modes)
%MACOS.GET_ELT_ZRN_COEF  Read Zernike coefficients on a Zern-typed element.
%   coefs = macos.get_elt_zrn_coef(IELT, MODES) returns COEFS(k) =
%   ZernCoef(MODES(k)) on element IELT, as a column vector matching
%   the shape of MODES.  IELT must declare Surface=Zernike or
%   Surface=ZrnGridData in the Rx.
%
%   See also: macos.set_elt_zrn_coef.
arguments
    iElt  (1,1) double {mustBeInteger, mustBePositive}
    modes (:,1) double {mustBeInteger, mustBePositive}
end
n = numel(modes);
coefs0 = zeros(n, 1);
[~, coefs] = mmacos('elt_srf_zrn_coef', iElt, double(modes), coefs0, ...
                    false, false, n);
coefs = double(coefs(:));
end
