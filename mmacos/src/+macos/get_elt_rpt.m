function rpt = get_elt_rpt(srf)
%MACOS.GET_ELT_RPT  Reference point of element SRF (3-vector, BaseUnits).
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
rpt_in = zeros(3, 1);
rpt = mmacos('elt_rpt', [srf], rpt_in, 0, 1);
rpt = rpt(:);
end
