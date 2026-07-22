function r = get_elt_zrn_norm_radius(srf)
%MACOS.GET_ELT_ZRN_NORM_RADIUS  Zernike normalisation radius (lMon) at SRF.
%   r = macos.get_elt_zrn_norm_radius(SRF) returns the Zernike normalisation
%   radius in BaseUnits, or -1 if SRF is not a Zernike surface.
%   See also: macos.set_elt_zrn_norm_radius, macos.get_elt_zrn.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
r = mmacos('elt_srf_zrn_norm_radius', srf, 0.0, 0);
end
