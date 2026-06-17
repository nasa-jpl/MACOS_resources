function kr = get_elt_kr(srf)
%MACOS.GET_ELT_KR  Radius of curvature Kr of element SRF (BaseUnits).
%   Flat / FocalPlane elements report a big-magnitude sentinel
%   (~1e22), not a physical radius.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
kr = mmacos('elt_kr', [srf], 0.0, 0, 1);
end
