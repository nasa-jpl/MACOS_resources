function t = get_elt_zrn_type(srf)
%MACOS.GET_ELT_ZRN_TYPE  Zernike normalisation type id at element SRF.
%   t = macos.get_elt_zrn_type(SRF) returns the ZernType id (1 ANSI,
%   2 BornWolf, 3 Fringe, 4-6 Norm variants, 7 NormHex, 8 NormNoll,
%   9 NormAnnularNoll), or -1 if SRF is not a Zernike surface.
%   See also: macos.set_elt_zrn_type, macos.get_elt_zrn.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
t = double(mmacos('elt_srf_zrn_type', srf, 0, 0, 0));
end
