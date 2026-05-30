function set_elt_grating_type(srf, refl)
%MACOS.SET_ELT_GRATING_TYPE  Set reflective (true) / transmissive (false).
arguments
    srf  (1,1) double {mustBeInteger, mustBePositive}
    refl (1,1) logical
end
mmacos('elt_srf_grating_type', srf, double(refl), 1);
end
