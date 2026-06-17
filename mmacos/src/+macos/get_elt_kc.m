function kc = get_elt_kc(srf)
%MACOS.GET_ELT_KC  Conic constant Kc of element SRF (dimensionless).
%   Kc = 0 sphere, -1 paraboloid, <-1 hyperboloid, -1<Kc<0 prolate
%   ellipsoid, Kc>0 oblate ellipsoid.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
kc = mmacos('elt_kc', [srf], 0.0, 0, 1);
end
