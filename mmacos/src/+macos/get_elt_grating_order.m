function order = get_elt_grating_order(srf)
%MACOS.GET_ELT_GRATING_ORDER  Diffraction order at element SRF.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
% Cmd is array-form: iElt(N), Order(N), setter, N.
order_out = mmacos('elt_srf_grating_order', srf, 0, 0, 1);
order = int32(order_out(1));
end
