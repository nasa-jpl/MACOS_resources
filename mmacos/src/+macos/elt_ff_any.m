function tf = elt_ff_any()
%MACOS.ELT_FF_ANY  True iff the loaded Rx has any FreeForm (SrfType 14) surfaces.
%   See also: macos.find_freeform_elts, macos.get_elt_ff_zrn_coef.
tf = logical(mmacos('elt_srf_ff_any'));
end
