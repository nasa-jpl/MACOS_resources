function r = ray(iRay)
%MACOS.RAY  Trace ray iRay of the source grid (1 = the chief ray) -- the CLI's RAY.
%   r = macos.ray(iRay)   iRay in 1..nRays of the current grid (macos.trace().nRays)
%   Traces to the last element (or the deck's ray-target element).  Fields as
%   macos.srt (column k = element k, BaseUnits / global frame):
%     r.pos  3 x nElt  the crossing at element k;  r.dir  3 x nElt  the direction leaving it
%     r.n_prev, r.n_cur   1 x nElt  the index before / after
%     r.ok   traced to the end (an OBSCURED ray still is); r.status / r.fail_elt as
%            macos.get_ray_status
%   The engine rebuilds its source grid for the call; the trace is marked modified.
%   api ray_trace (BRIEF_to_bindings B.2, 2026-10-17).
%
%   See also: macos.srt, macos.ray_hist, macos.get_ray_status.
arguments
    iRay (1,1) double {mustBeInteger, mustBePositive}
end
ne = macos.num_elt();
z3 = zeros(3, ne);  z1 = zeros(1, ne);
[P, D, np, nc, ok, st, fe] = mmacos('ray_trace', double(iRay), z3, z3, z1, z1, double(ne));
macos.modify();
r = struct('pos', P, 'dir', D, 'n_prev', reshape(np, 1, []), ...
           'n_cur', reshape(nc, 1, []), 'ok', logical(ok), 'status', double(st), ...
           'fail_elt', double(fe));
end
