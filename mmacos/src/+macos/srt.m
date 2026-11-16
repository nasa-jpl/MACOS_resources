function r = srt(x, y, endElt)
%MACOS.SRT  Trace ONE ray from a normalized source-aperture point (the CLI's SRT).
%   r = macos.srt(x, y)          trace to the last element
%   r = macos.srt(x, y, endElt)  trace through element endElt
%   x, y: the point in the normalized source aperture, |x|, |y| <= 0.5; (0, 0) is
%   the chief ray.  Fields (column k = element k, BaseUnits / global frame):
%     r.pos     3 x nElt  where the ray meets element k
%     r.dir     3 x nElt  its direction leaving element k
%     r.n_prev, r.n_cur   1 x nElt  the index before / after element k
%     r.ok      the ray was traced to endElt (an OBSCURED ray still is)
%     r.status  per-ray status code (as macos.get_ray_status); r.fail_elt the
%               element that set it (0 when OK)
%   Columns past endElt -- or past fail_elt for a ray that was lost -- are 0.
%   The engine re-seeds its source ray set for the one ray; the next
%   macos.trace rebuilds the full grid (the trace is marked modified here).
%   api srt_trace (BRIEF_to_bindings B.2, 2026-10-17).
%
%   See also: macos.ray_hist, macos.trace, macos.get_ray_status.
arguments
    x (1,1) double
    y (1,1) double
    endElt double {mustBeScalarOrEmpty} = []
end
ne = macos.num_elt();
if isempty(endElt), endElt = ne; end
z3 = zeros(3, ne);  z1 = zeros(1, ne);
[P, D, np, nc, ok, st, fe] = mmacos('srt_trace', double(x), double(y), ...
    double(endElt), z3, z3, z1, z1, double(ne));
macos.modify();
r = struct('pos', P, 'dir', D, 'n_prev', reshape(np, 1, []), ...
           'n_cur', reshape(nc, 1, []), 'ok', logical(ok), 'status', double(st), ...
           'fail_elt', double(fe));
end
