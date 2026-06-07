function r = get_ray_status(n_rays)
%MACOS.GET_RAY_STATUS  Per-category ray-status codes after the most recent trace.
%   r = macos.get_ray_status(N) returns a struct with fields:
%       .status      N×1 int32 — per-ray RayStat_* code:
%                       0 = OK, 1 = Obscured, 2 = Miss,
%                       3 = Bracket, 4 = MaxIter, 5 = Undef
%       .fail_elt    N×1 int32 — element index where failure was first
%                       recorded (0 if OK)
%       .n_ok        scalar    — count of OK rays
%       .n_obscured  scalar    — count of Obscured rays
%       .n_miss      scalar    — count of Miss rays
%       .n_bracket   scalar    — count of Bracket rays
%       .n_max_iter  scalar    — count of MaxIter rays
%       .n_undef     scalar    — count of Undef rays
%       .n_lost      scalar    — total rays not in the OK state
%
%   N is the number of rays from the trace (second return of macos.trace).
%
%   Sourced from elt_mod's RayStatus(:) / RayFailElt(:) module arrays,
%   populated by SetRayFail() inside the trace and propagation loops.
%   Complements macos.get_ray_info (binary ok_trace / ok_pass per ray).
%
%   See also: macos.get_ray_info, macos.trace.
arguments
    n_rays (1,1) double {mustBeInteger, mustBePositive}
end
% Cmd 'ray_status_get' prhs: nRays.  plhs: RayStatus_, RayFailElt_.
[status, fail_elt] = mmacos('ray_status_get', n_rays);
r.status     = int32(status(:));
r.fail_elt   = int32(fail_elt(:));
r.n_ok        = nnz(r.status == 0);
r.n_obscured  = nnz(r.status == 1);
r.n_miss      = nnz(r.status == 2);
r.n_bracket   = nnz(r.status == 3);
r.n_max_iter  = nnz(r.status == 4);
r.n_undef     = nnz(r.status == 5);
r.n_lost      = double(n_rays) - r.n_ok;
end
