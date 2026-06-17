function z = get_elt_z(srf)
%MACOS.GET_ELT_Z  Propagation distance zElt at element SRF.
%   For Flat / FocalPlane elements this is a big-magnitude sentinel
%   (~1e19–1e22) meaning "propagate to the next defined plane", not a
%   physical distance.  Curved elements report the actual prescribed
%   propagation distance from the previous surface.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
z = mmacos('elt_z', [srf], 0.0, 0, 1);
end
