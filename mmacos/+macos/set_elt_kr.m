function set_elt_kr(srf, kr)
%MACOS.SET_ELT_KR  Set the radius of curvature Kr of element SRF (BaseUnits).
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
    kr  (1,1) double
end
mmacos('elt_kr', [srf], kr, 1, 1);
end
