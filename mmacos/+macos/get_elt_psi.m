function psi = get_elt_psi(srf)
%MACOS.GET_ELT_PSI  Vertex surface normal of element SRF (3-vector).
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
psi_in = zeros(3, 1);
psi = mmacos('elt_psi', [srf], psi_in, 0, 1);
psi = psi(:);
end
