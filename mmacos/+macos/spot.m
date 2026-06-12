function s = spot(srf, opts)
%MACOS.SPOT  Ray-surface intersection (spot) at element SRF.
%   s = macos.spot(SRF) returns a struct:
%     .pts       nSpot×2  intersection points in the spot coord frame
%     .centroid  1×2      centroid (x,y)
%     .shift     1×4      shift from the projected reference position
%     .csys      3×3      spot coordinate-frame orientation
%     .nSpot     scalar   number of rays contributing to the spot
%
%   Name-value pairs:
%     'ref'         'beam'(default) | 'tout' | 'telt' — reference
%                   coordinate system the spot is expressed in.
%     'at'          'chief'(default) | 'elt' — reference position
%                   (chief-ray vs element vertex).
%     'reset_trace' logical (default true) — apply MODIFY before tracing.
%
%   Raw equivalents: mmacos('spot_cmd', ...) + mmacos('spot_get', N).
%
%   See also: macos.trace, macos.get_ray_info.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
    opts.ref (1,:) char {mustBeMember(opts.ref, {'beam','tout','telt'})} = 'beam'
    opts.at  (1,:) char {mustBeMember(opts.at,  {'chief','elt'})} = 'chief'
    opts.reset_trace (1,1) logical = true
end
refmap   = struct('beam', 1, 'tout', 2, 'telt', 3);
ref_csys = refmap.(opts.ref);
ref_pos  = double(strcmp(opts.at, 'elt'));
nSpot = mmacos('spot_cmd', srf, ref_csys, ref_pos, double(opts.reset_trace));
[pts, shift, centroid, csys] = mmacos('spot_get', nSpot);
s.pts      = pts;
s.centroid = centroid(:)';
s.shift    = shift(:)';
s.csys     = csys;
s.nSpot    = nSpot;
end
