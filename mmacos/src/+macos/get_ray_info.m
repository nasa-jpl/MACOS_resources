function r = get_ray_info(n_rays)
%MACOS.GET_RAY_INFO  Read per-ray data from the most recent trace.
%   r = macos.get_ray_info(N) returns a struct with fields:
%       .pos       3×N double  — intersection position with current surface
%       .dir       3×N double  — incoming direction at that surface
%       .opl       N×1 double  — cumulative OPL from source
%       .ok_trace  N×1 logical — true if the ray traced successfully
%       .ok_pass   N×1 logical — true if the ray was not blocked by a mask
%
%   N is the number of rays from the trace.  Get it from the second
%   return of macos.trace().
%
%   See also: macos.trace.
arguments
    n_rays (1,1) double {mustBeInteger, mustBePositive}
end
% Cmd 'ray_info_get' prhs: nRays.  plhs: Pos, Dir, OPL, RayOK, RayPass.
[pos, dir_, opl, ok_trace, ok_pass] = mmacos('ray_info_get', n_rays);
r.pos      = pos;
r.dir      = dir_;
r.opl      = opl(:);
r.ok_trace = logical(ok_trace(:));
r.ok_pass  = logical(ok_pass(:));
end
