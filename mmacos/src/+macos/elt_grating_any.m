function tf = elt_grating_any()
%MACOS.ELT_GRATING_ANY  True iff the loaded Rx has any grating elements.
tf = logical(mmacos('elt_srf_grating_any'));
end
