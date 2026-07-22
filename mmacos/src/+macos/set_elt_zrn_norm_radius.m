function set_elt_zrn_norm_radius(srf, norm_radius)
%MACOS.SET_ELT_ZRN_NORM_RADIUS  Set the Zernike normalisation radius at SRF.
%   macos.set_elt_zrn_norm_radius(SRF, R) sets lMon (BaseUnits, R>0) on a
%   SrfType=Zernike surface.  Errors if SRF is not a Zernike surface.
%   See also: macos.get_elt_zrn_norm_radius, macos.get_elt_zrn.
arguments
    srf         (1,1) double {mustBeInteger, mustBePositive}
    norm_radius (1,1) double {mustBePositive}
end
mmacos('elt_srf_zrn_norm_radius', srf, norm_radius, 1);
end
