function F = field_grid(fov, n, opts)
%FIELD_GRID  n x n field-angle GRID -- the "area" field set.
%   F = macos.design.field_grid(fov, n) returns an (n*n) x 2 list of
%   (thx,thy) field points spanning [-fov,+fov] in BOTH axes, edge to edge
%   (corners + edge midpoints + center) -- the field set for AREA scoring
%   (optimize / evaluate over a 2-D field), as opposed to a single +y fan
%   or a CROSS (see macos.design.field_cross).
%
%   fov is a HALF-field; use an ODD n so the center (0,0) is a sample.
%   Returned points are radians by default (units='arcmin' interprets fov
%   in arcmin and still returns radians, to match set_field_points /
%   optimize('fields',...) / realize_apertures('fields',...)).
%
%   Name-value:
%     'origin' (default true)  keep the (0,0) center point.  Pass false for
%              optimize('fields',...), where the on-axis field is IMPLICIT
%              field 1 (optimize also drops it defensively).
%     'units'  'rad' (default) | 'arcmin'.
%
%   CALIB caps a native multi-field optimize at 12 FoV, so the AREA-optimize
%   mode is practical up to a 3x3 grid (8 off-axis + the implicit on-axis =
%   9 FoV); finer grids (e.g. 7x7) are for EVALUATION / the WFE field map.
%
%   Example:
%     opt  = macos.design.field_grid(fov, 3, 'origin', false);  % area optimize
%     fine = macos.design.field_grid(fov, 7);                   % WFE map / scan
%
%   See also macos.design.field_cross, macos.design.Telescope/view_field_map.
    arguments
        fov (1,1) double {mustBePositive}
        n   (1,1) double {mustBeInteger, mustBePositive} = 3
        opts.origin (1,1) logical = true
        opts.units  (1,:) char {mustBeMember(opts.units,{'rad','arcmin'})} = 'rad'
    end
    if strcmpi(opts.units, 'arcmin'), fov = deg2rad(fov/60); end
    a = linspace(-fov, fov, n);
    [X, Y] = meshgrid(a, a);
    F = [X(:), Y(:)];
    if ~opts.origin
        F = F(any(abs(F) > 1e-12, 2), :);   % drop the (0,0) center
    end
end
