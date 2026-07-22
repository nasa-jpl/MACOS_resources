function set_elt_zrn_type(srf, zern_type, opts)
%MACOS.SET_ELT_ZRN_TYPE  Set the Zernike normalisation type at element SRF.
%   macos.set_elt_zrn_type(SRF, TYPE) sets the ZernType id (1..9; see
%   macos.get_elt_zrn_type) on a SrfType=Zernike surface.
%   macos.set_elt_zrn_type(..., 'reset', true) zeros the coefficients first.
%   Errors if SRF is not a Zernike surface or TYPE is out of range.
arguments
    srf        (1,1) double {mustBeInteger, mustBePositive}
    zern_type  (1,1) double {mustBeInteger, mustBePositive}
    opts.reset (1,1) logical = false
end
mmacos('elt_srf_zrn_type', srf, zern_type, 1, double(opts.reset));
end
