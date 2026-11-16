function vpt = get_elt_vpt(srf)
%MACOS.GET_ELT_VPT  Vertex point of element SRF (3-vector, BaseUnits).
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
vpt_in = zeros(3, 1);
vpt = mmacos('elt_vpt', [srf], vpt_in, 0, 1);
vpt = vpt(:);
end
