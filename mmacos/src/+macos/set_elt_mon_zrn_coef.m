function set_elt_mon_zrn_coef(iElt, modes, coefs, opts)
%MACOS.SET_ELT_MON_ZRN_COEF  Write Mon-Zernike coefficients on a FreeForm.
%   macos.set_elt_mon_zrn_coef(IELT, MODES, COEFS) writes COEFS(k) to
%   MonZern mode MODES(k) on element IELT.  IELT must be a FreeForm
%   surface (SrfType=14).  The mode-index basis is the same as the
%   element's MonZernType (e.g. BornWolf / ANSI / Noll).
%
%   macos.set_elt_mon_zrn_coef(IELT, MODES, COEFS, 'reset', true)
%   zeroes all MonZern modes before applying the new coefficients.
%
%   See also: macos.set_elt_ff_zrn_coef, macos.set_elt_zrn_coef,
%             macos.find_freeform_elts.
arguments
    iElt        (1,1) double {mustBeInteger, mustBePositive}
    modes       (:,1) double {mustBeInteger, mustBePositive}
    coefs       (:,1) double
    opts.reset  (1,1) logical = false
end
if numel(modes) ~= numel(coefs)
    error('macos:set_elt_mon_zrn_coef:size', ...
        'modes and coefs must have the same length');
end
n = numel(modes);
mmacos('elt_srf_mon_zrn_coef', iElt, double(modes), coefs, true, ...
       opts.reset, n);
end
