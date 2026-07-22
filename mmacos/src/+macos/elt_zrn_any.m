function tf = elt_zrn_any()
%MACOS.ELT_ZRN_ANY  True iff the loaded Rx has any Zernike (SrfType 8) surfaces.
%   See also: macos.get_elt_zrn, macos.find_zern_elts.
tf = logical(mmacos('elt_srf_zrn_any'));
end
