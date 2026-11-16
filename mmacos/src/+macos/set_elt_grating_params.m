function set_elt_grating_params(srf, params)
%MACOS.SET_ELT_GRATING_PARAMS  Write grating parameters at element SRF.
%   PARAMS is a struct with fields:
%     .reflective   logical
%     .rule_width   double > 0
%     .diff_order   integer
%     .rule_dir     3×1 double (will be normalized internally)
arguments
    srf    (1,1) double {mustBeInteger, mustBePositive}
    params (1,1) struct
end
d = params.rule_dir(:);
n = norm(d);
if n == 0
    error('macos:set_elt_grating_params:zeroDir', ...
        'params.rule_dir must be a non-zero 3-vector.');
end
% prhs slot order matches the api signature:
%   iElt, Spacing, Diff_Order, h1HOE, reflective, setter
mmacos('elt_srf_grating_params', srf, ...
    double(params.rule_width), ...
    double(params.diff_order), ...
    d / n, ...
    double(logical(params.reflective)), ...
    1);
end
