function F = field_ring(radius, opts)
%FIELD_RING  Ring field set -- the CIRCULAR-field spec.
%   F = macos.design.field_ring(radius) returns (thx,thy) field points on
%   a ring of the given field RADIUS: n equally-spaced azimuths, plus an
%   optional inner ring.  A circular field of diameter 2*radius is the
%   honest spec for a round detector / instrument patch: a square
%   field_grid of half-field h puts its corners at h*sqrt(2) -- 41% more
%   field radius than the circular spec asks for -- and those corners
%   then dominate a field balance.  Balance and score a circular field on
%   a ring + center instead.
%
%   radius is the field RADIUS (half the circular-field diameter).
%   Points are radians by default (units='arcmin' interprets radius in
%   arcmin and still returns radians, matching field_grid / field_cross /
%   optimize('fields',...)).  The (0,0) center is NOT included: for
%   optimize('fields',...) the on-axis field is the implicit field 1; for
%   evaluation loops prepend [0 0] yourself if wanted.
%
%   Name-value:
%     'n'        ring azimuth count (default 8).
%     'inner'    inner-ring radius as a FRACTION of radius (default 0.5;
%                pass 0 to skip the inner samples).
%     'n_inner'  inner-ring azimuth count (default 2: +-x, catching the
%                mid-field behaviour without spending FoV slots).
%     'units'    'rad' (default) | 'arcmin'.
%
%   CALIB caps a native multi-field optimize at 12 FoV: the default
%   8 + 2 + the implicit on-axis = 11 fits.
%
%   Example (a 5-arcmin-diameter circular field):
%     optF = macos.design.field_ring(2.5, 'units','arcmin');
%     t.optimize('fields',optF, ...);
%
%   See also macos.design.field_grid, macos.design.field_cross.
    arguments
        radius (1,1) double {mustBePositive}
        opts.n       (1,1) double {mustBeInteger, mustBePositive} = 8
        opts.inner   (1,1) double {mustBeNonnegative} = 0.5
        opts.n_inner (1,1) double {mustBeInteger, mustBeNonnegative} = 2
        opts.units   (1,:) char {mustBeMember(opts.units,{'rad','arcmin'})} = 'rad'
    end
    if strcmpi(opts.units, 'arcmin'), radius = deg2rad(radius/60); end
    az = (0:opts.n-1).' * (2*pi/opts.n);
    F  = radius * [cos(az), sin(az)];
    if opts.inner > 0 && opts.n_inner > 0
        azi = (0:opts.n_inner-1).' * (2*pi/opts.n_inner);
        F   = [F; opts.inner*radius*[cos(azi), sin(azi)]];
    end
end
