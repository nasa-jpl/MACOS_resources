function w = get_elt_grating_rulewidth(srf)
%MACOS.GET_ELT_GRATING_RULEWIDTH  Ruling spacing at element SRF.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
w = mmacos('elt_srf_grating_rule_width', srf, 0.0, 0);
end
