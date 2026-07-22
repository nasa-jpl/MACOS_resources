function s = get_elt_srf_csys(srfs)
%MACOS.GET_ELT_SRF_CSYS  Surface (monomial/Zernike/grid) coordinate frame.
%   s = macos.get_elt_srf_csys(SRFS) returns the surface-figure coordinate
%   frame (pMon/xMon/yMon/zMon) for each element in SRFS -- the frame in which
%   monomial, Zernike and grid figures are defined:
%       .pMon  3×N   frame origin (vertex position)
%       .xMon  3×N   x-axis (each column a unit vector)
%       .yMon  3×N   y-axis
%       .zMon  3×N   z-axis (surface normal)
%
%   Valid only for surfaces that carry a figure frame (Monomial(4),
%   Zernike(8), GridData(9) and composites); errors otherwise.  This is the
%   figure frame, distinct from macos.get_elt_csys (the element output LCS,
%   TElt) and macos.get_elt_psi (the surface normal alone).
arguments
    srfs (:,1) double {mustBeInteger, mustBePositive}
end
n = numel(srfs);
[pMon, xMon, yMon, zMon] = mmacos('elt_srf_csys_get', srfs, n);
s.pMon = pMon;
s.xMon = xMon;
s.yMon = yMon;
s.zMon = zMon;
end
