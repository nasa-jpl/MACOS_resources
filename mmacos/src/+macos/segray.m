function r = segray(iSeg)
%MACOS.SEGRAY  Trace the ray through Segment element iSeg's centre -- the CLI's SEGRAYTRACE.
%   r = macos.segray(iSeg)   iSeg = the ELEMENT number of a Segment element
%   The ray passes through the segment's reference point (RptElt, the design
%   centre; a perturbation does not move it): parallel to the chief ray for a
%   collimated source, aimed from the chief ray's start for a point source.  It
%   is forced onto segment iSeg and traced to the last element (or the deck's
%   ray-target element).  Fields as macos.ray (column k = element k):
%     r.pos  3 x nElt  the crossing at element k;  r.dir  3 x nElt  the direction leaving it
%     r.n_prev, r.n_cur   1 x nElt  the index before / after
%     r.ok, r.status, r.fail_elt   as macos.ray
%   An element that is not a Segment is refused.  The engine rebuilds its source
%   grid for the call; the trace is marked modified.
%   api segray_trace (BRIEF_to_bindings B.2, 2026-10-17).
%
%   See also: macos.ray, macos.srt.
arguments
    iSeg (1,1) double {mustBeInteger, mustBePositive}
end
ne = macos.num_elt();
z3 = zeros(3, ne);  z1 = zeros(1, ne);
[P, D, np, nc, ok, st, fe] = mmacos('segray_trace', double(iSeg), z3, z3, z1, z1, double(ne));
macos.modify();
r = struct('pos', P, 'dir', D, 'n_prev', reshape(np, 1, []), ...
           'n_cur', reshape(nc, 1, []), 'ok', logical(ok), 'status', double(st), ...
           'fail_elt', double(fe));
end
