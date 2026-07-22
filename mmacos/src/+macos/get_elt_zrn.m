function s = get_elt_zrn(srf)
%MACOS.GET_ELT_ZRN  Read the Zernike-surface definition at element SRF.
%   s = macos.get_elt_zrn(SRF) returns a struct for a SrfType=Zernike surface:
%       .norm_radius   scalar   Zernike normalisation radius (lMon, BaseUnits)
%       .type          scalar   ZernType id (1 ANSI, 2 BornWolf, 3 Fringe,
%                               4-6 Norm variants, 7 NormHex, 8 NormNoll,
%                               9 NormAnnularNoll)
%       .coefs         66×1     Zernike coefficients by mode index (BaseUnits)
%       .annular_ratio scalar   inner/outer radius ratio (only ZernType 9)
%
%   Errors if SRF is not a Zernike surface.  For the coefficient-only
%   read/write use macos.get_elt_zrn_coef / set_elt_zrn_coef.  See also:
%   macos.get_elt_zrn_type, macos.get_elt_zrn_norm_radius, macos.elt_zrn_any.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
[lMon, ztype, coef, annu] = mmacos('elt_srf_zrn_get', srf, 1);
s.norm_radius   = lMon;
s.type          = double(ztype);
s.coefs         = coef(:);
s.annular_ratio = annu;
end
