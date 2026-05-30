function refl = get_elt_grating_type(srf)
%MACOS.GET_ELT_GRATING_TYPE  Reflective (true) vs transmissive (false).
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
refl = logical(mmacos('elt_srf_grating_type', srf, 0, 0));
end
