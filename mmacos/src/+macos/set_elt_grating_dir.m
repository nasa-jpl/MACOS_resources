function set_elt_grating_dir(srf, d)
%MACOS.SET_ELT_GRATING_DIR  Set h1HOE ruling-direction vector.
%   d is normalized to unit length before storing.  (macos's
%   `elt_srf_grating_rule_dir` does not normalize; we do it here to
%   match pymacos's wrapper semantics — h1HOE is by definition a
%   unit vector.)
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
    d   (3,1) double
end
n = norm(d);
if n == 0
    error('macos:set_elt_grating_dir:zeroVector', ...
        'd must be a non-zero 3-vector.');
end
mmacos('elt_srf_grating_rule_dir', srf, d / n, 1);
end
