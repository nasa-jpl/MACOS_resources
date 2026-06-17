function set_elt_zrn_coef(iElt, modes, coefs, opts)
%MACOS.SET_ELT_ZRN_COEF  Write Zernike coefficients on a Zern-typed element.
%   macos.set_elt_zrn_coef(IELT, MODES, COEFS) writes COEFS(k) to
%   Zernike mode MODES(k) on element IELT for each k.  MODES is a
%   vector of 1-based Zernike mode indices, COEFS is the matching
%   vector of coefficients in BaseUnits.  IELT must declare
%   Surface=Zernike or Surface=ZrnGridData in the Rx.
%
%   macos.set_elt_zrn_coef(IELT, MODES, COEFS, 'reset', true) zeroes
%   all Zernike modes before applying the new coefficients.  Default
%   false (additive into the existing coefficient vector).
%
%   See also: macos.set_elt_mon_zrn_coef, macos.set_elt_ff_zrn_coef,
%             macos.find_zern_elts.
arguments
    iElt        (1,1) double {mustBeInteger, mustBePositive}
    modes       (:,1) double {mustBeInteger, mustBePositive}
    coefs       (:,1) double
    opts.reset  (1,1) logical = false
end
if numel(modes) ~= numel(coefs)
    error('macos:set_elt_zrn_coef:size', ...
        'modes and coefs must have the same length');
end
n = numel(modes);
mmacos('elt_srf_zrn_coef', iElt, double(modes), coefs, true, opts.reset, n);
end
