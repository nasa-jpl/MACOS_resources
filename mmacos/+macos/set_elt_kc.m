function set_elt_kc(srf, kc)
%MACOS.SET_ELT_KC  Set the conic constant Kc of element SRF (dimensionless).
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
    kc  (1,1) double
end
mmacos('elt_kc', [srf], kc, 1, 1);
end
