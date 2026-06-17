function set_elt_grating_rulewidth(srf, w)
%MACOS.SET_ELT_GRATING_RULEWIDTH  Write ruling spacing at element SRF.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
    w   (1,1) double {mustBePositive}
end
mmacos('elt_srf_grating_rule_width', srf, w, 1);
end
