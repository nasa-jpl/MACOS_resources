function [r, info] = through_hole_radius(t, opts)
%THROUGH_HOLE_RADIUS  How big does the perforation have to be, measured.
%
%   [r, info] = THROUGH_HOLE_RADIUS(t) traces the built design T and
%   returns the radius the named element's central hole needs so that
%   every beam leg AFTER the secondary passes through it -- measured from
%   the engine's own ray history, not assumed.
%
%   WHY THIS IS NOT A CONSTANT.  A perforated primary's hole is sized by
%   the beam that crosses its plane, and that beam moves.  It grows with
%   the field (a wider box spreads the returning bundle) and it WALKS OFF
%   THE VERTEX with the field bias -- on the e2e2 TMA the crossing pattern
%   sits 0.45 m off centre at a 2.5 deg bias.  Freezing the hole at a
%   value inherited from some other design and then reporting the primary
%   as an obstruction is a measurement error dressed as a clearance
%   finding: it reads as "the layout does not close" when what actually
%   happened is that nobody re-sized the hole.
%
%   That is exactly how it failed here.  e2e2's stage 2 declared the
%   reference design's scaled 0.308 m hole at every bias and duly reported
%   M1 in the beam from 60' upward -- while the beam needed 0.387 m there
%   and 0.590 m at 90'.  With the hole measured, M1 never obstructs at any
%   bias and only the accepted M2 central obscuration remains.  So: MEASURE
%   IT IN THE DRIVER, at the configuration being judged.
%
%   Name-value:
%     'elt'       element whose plane is crossed (default 1, the primary)
%     'first_leg' first ray-history slot to test (default 3).  Slot 1 is
%                 the source and slot 2 the primary, so 3 is the leg
%                 leaving the secondary -- the first that can come back
%                 through the primary.
%     'margin'    multiply the measured radius by this (default 1.0).
%                 The caller usually wants clearance beyond the marginal
%                 ray.
%     'floor_m'   never return less than this (default 0).  Use it for a
%                 shadow floor -- the secondary already obscures that much,
%                 so a smaller hole buys nothing.
%
%   Returns r (margin and floor applied) and info with .r_raw (the
%   measured maximum, before margin/floor), .centre_off_m (how far the
%   crossing pattern's centroid sits off the element vertex -- a large
%   value says a CONCENTRIC hole is the crude answer and a hole sized and
%   centred on the crossing is the honest one), .n_crossings, and
%   .r_floor_used / .r_margin_used.
%
%   r is NaN when no leg crosses the plane at all (an unfolded on-axis
%   design where nothing returns through the primary).  Callers should
%   treat NaN as "no hole needed", not as a failure.
%
%   See also macos.design.Telescope/set_hole, .../check_clipping.

    arguments
        t
        opts.elt       (1,1) double {mustBeInteger,mustBePositive} = 1
        opts.first_leg (1,1) double {mustBeInteger,mustBePositive} = 3
        opts.margin    (1,1) double {mustBePositive} = 1.0
        opts.floor_m   (1,1) double {mustBeNonnegative} = 0
    end

    % ray_hist must DIRTY the trace or a previously-traced session hands
    % back an empty history (the grid-setter-retrace class); the veneer
    % does that for us.
    macos.ray_hist('on');
    s  = macos.trace();
    hh = macos.ray_hist(s.nRays);
    macos.ray_hist('off');

    e  = t.spec.elt(opts.elt);
    p0 = e.Vpt(:);
    ps = e.psi(:);   ps = ps/norm(ps);

    rad = [];   pts = zeros(3,0);
    for leg = opts.first_leg : size(hh.P,3)-1
        A  = squeeze(hh.P(:,:,leg));
        B  = squeeze(hh.P(:,:,leg+1));
        ok = hh.ok(:,leg) & hh.ok(:,leg+1);
        for i = find(ok(:)).'
            d  = B(:,i) - A(:,i);
            dn = dot(ps, d);
            if abs(dn) < 1e-12, continue; end
            f = dot(ps, p0 - A(:,i)) / dn;
            if f > 0 && f < 1                     % crosses BETWEEN the two
                q = A(:,i) + f*d;
                v = q - p0 - ps*dot(ps, q - p0);  % in-plane offset
                rad(end+1)   = norm(v);           %#ok<AGROW>
                pts(:,end+1) = v;                 %#ok<AGROW>
            end
        end
    end

    info = struct('r_raw',NaN, 'centre_off_m',NaN, 'n_crossings',numel(rad), ...
                  'r_margin_used',opts.margin, 'r_floor_used',opts.floor_m, ...
                  'elt',opts.elt);
    if isempty(rad)
        r = NaN;   return;                        % nothing returns: no hole
    end
    info.r_raw        = max(rad);
    info.centre_off_m = norm(mean(pts,2));
    r = max(opts.margin*info.r_raw, opts.floor_m);
end
