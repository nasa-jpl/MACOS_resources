function F = field_cross(fov, n, opts)
%FIELD_CROSS  Field-angle CROSS (x-arm + y-arm) -- the cheap field set.
%   F = macos.design.field_cross(fov, n) returns the (thx,thy) field points
%   on a CROSS through the origin: n samples along +-x and n along +-y,
%   spanning [-fov,+fov] (radians).  The cross captures the two PRINCIPAL
%   field directions at a fraction of a full grid's points -- the right set
%   when the system's field aberration is dominated by one plane (e.g. an
%   eccentric-pupil off-axis section, astigmatic in the decenter plane).
%   For full AREA scoring use macos.design.field_grid.
%
%   fov is a HALF-field; use an ODD n so the center is a sample.  Name-value
%   'origin' (default true) and 'units' ('rad'|'arcmin') as in field_grid.
%
%   field_cross(fov, 3, 'origin', false) is the 4 arm TIPS (+-fov in x and
%   y) -- the minimal cross-mode optimize set.
%
%   See also macos.design.field_grid, macos.design.Telescope/optimize.
    arguments
        fov (1,1) double {mustBePositive}
        n   (1,1) double {mustBeInteger, mustBePositive} = 5
        opts.origin (1,1) logical = true
        opts.units  (1,:) char {mustBeMember(opts.units,{'rad','arcmin'})} = 'rad'
    end
    if strcmpi(opts.units, 'arcmin'), fov = deg2rad(fov/60); end
    a = linspace(-fov, fov, n).';
    F = [ [a, zeros(n,1)] ; [zeros(n,1), a] ];
    F = unique(F, 'rows', 'stable');        % de-dup the shared center
    if ~opts.origin
        F = F(any(abs(F) > 1e-12, 2), :);   % drop the (0,0) center
    end
end
