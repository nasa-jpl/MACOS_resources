function p = get_elt_grating_params(srf)
%MACOS.GET_ELT_GRATING_PARAMS  Read grating parameters at element SRF.
%   Returns a struct with fields:
%     .reflective   logical: true=reflective, false=transmissive
%     .rule_width   double:  spacing along projected flat plane
%     .diff_order   integer: diffraction order
%     .rule_dir     3×1 double: h1HOE — perpendicular to ruling + psi
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
end
% Cmd 'elt_srf_grating_params' prhs:
%   iElt, Diff_Order, Spacing, h1HOE(3), reflective, setter
% Setter=0 (get); placeholders for the inout args.
[Spacing, Diff_Order, h1HOE, reflective] = mmacos('elt_srf_grating_params', ...
    srf, 0, 0.0, zeros(3,1), 0, 0);
p.reflective = logical(reflective);
p.rule_width = Spacing;
p.diff_order = int32(Diff_Order);
p.rule_dir   = h1HOE(:);
end
