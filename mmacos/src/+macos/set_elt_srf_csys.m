function set_elt_srf_csys(srf, pMon, xMon, yMon, zMon)
%MACOS.SET_ELT_SRF_CSYS  Set the surface-figure coordinate frame at SRF.
%   macos.set_elt_srf_csys(SRF, PMON, XMON, YMON, ZMON) defines the figure
%   frame (each a 3-vector) for a figured surface (Monomial(4), Zernike(8),
%   GridData(9) or composite).  The x/y/z axes are orthonormalised by the
%   engine.  Errors if SRF is not a figure-frame surface or the axes are
%   degenerate.  See also: macos.get_elt_srf_csys.
arguments
    srf  (1,1) double {mustBeInteger, mustBePositive}
    pMon (3,1) double
    xMon (3,1) double
    yMon (3,1) double
    zMon (3,1) double
end
mmacos('elt_srf_csys_set', srf, pMon, xMon, yMon, zMon, 1);
end
