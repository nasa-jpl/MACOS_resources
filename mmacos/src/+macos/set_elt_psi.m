function set_elt_psi(srf, psi)
%MACOS.SET_ELT_PSI  Set element SRF's vertex surface normal (3-vector).
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
    psi (3,1) double
end
mmacos('elt_psi', [srf], psi, 1, 1);
end
