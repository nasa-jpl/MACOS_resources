function set_elt_grating_order(srf, order)
%MACOS.SET_ELT_GRATING_ORDER  Write diffraction order at element SRF.
arguments
    srf   (1,1) double {mustBeInteger, mustBePositive}
    order (1,1) double {mustBeInteger}
end
mmacos('elt_srf_grating_order', srf, order, 1, 1);
end
