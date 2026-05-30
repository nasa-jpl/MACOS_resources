function set_elt_rpt(srf, rpt)
%MACOS.SET_ELT_RPT  Set element SRF's reference point (3-vector).
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
    rpt (3,1) double
end
mmacos('elt_rpt', [srf], rpt, 1, 1);
end
