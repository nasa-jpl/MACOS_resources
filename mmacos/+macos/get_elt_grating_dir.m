function d = get_elt_grating_dir(srf)
%MACOS.GET_ELT_GRATING_DIR  h1HOE ruling-direction vector (column 3-vec).
%   Perpendicular to the ruling direction and to psiElt.  Always
%   returned as a unit vector.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
d = mmacos('elt_srf_grating_rule_dir', srf, zeros(3,1), 0);
d = d(:);
end
