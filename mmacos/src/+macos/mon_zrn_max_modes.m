function n = mon_zrn_max_modes()
%MACOS.MON_ZRN_MAX_MODES  Number of monomial-Zernike coefficient slots per elt.
%   n = macos.mon_zrn_max_modes() returns mMonCoef -- the max Mon-Zernike mode
%   index a FreeForm surface can carry.  Use to size mode arrays without
%   hard-coding.  See also: macos.get_elt_mon_zrn_coef.
n = double(mmacos('elt_srf_mon_zrn_max_modes'));
end
