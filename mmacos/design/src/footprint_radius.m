function [r, info] = footprint_radius(t, elt, opts)
%FOOTPRINT_RADIUS  How big is the beam where it lands, measured.
%
%   [r, info] = FOOTPRINT_RADIUS(t, ELT) traces the built design T and
%   returns the largest ray radius at element ELT, measured in that
%   element's own plane about its vertex.  This is the optic's illuminated
%   extent -- what its body has to be at least as big as, and, for a
%   secondary in front of a primary, what it SHADOWS.
%
%   THE SHADOW IS THE HOLE'S FLOOR, and it is free.  A perforated primary
%   loses nothing to a hole that fits inside its secondary's shadow: that
%   light was never going to arrive.  Sizing the hole any smaller buys no
%   aperture back, and sizing it from some other design's geometry can
%   easily do the opposite -- e2e2 inherited a hole 1.39x this design's
%   own secondary shadow and threw away 4.2% of the area where 2.2% was
%   unavoidable.  So the floor is measured HERE, on the design being
%   built, not carried in as a constant.
%
%   Name-value:
%     'margin' multiply the measured radius by this (default 1.0), for a
%              body that has to hold a mount as well as a mirror
%     'quiet'  suppress the printed line (default true)
%
%   Returns r (margin applied) and info with .r_raw, .r_min (the inner
%   edge of the illuminated annulus -- nonzero where an obscuration or a
%   hole already bites), .centre_off_m (the footprint centroid's offset
%   from the vertex, which a field bias makes nonzero), .n_rays.
%
%   See also through_hole_radius, macos.design.Telescope/set_hole.

    arguments
        t
        elt (1,1) double {mustBeInteger,mustBePositive}
        opts.margin (1,1) double {mustBePositive} = 1.0
        opts.quiet  (1,1) logical = true
    end

    n  = numel(t.spec.elt);
    tr = macos.trace(n);                 % full trace, then step back
    macos.trace(elt);
    ri = macos.get_ray_info(tr.nRays);

    e   = t.spec.elt(elt);
    p0  = e.Vpt(:);
    ps  = e.psi(:);   ps = ps/norm(ps);
    ok  = logical(ri.ok_trace(:)) & logical(ri.ok_pass(:));
    Q   = ri.pos(:,ok) - p0;
    Q   = Q - ps*(ps.'*Q);               % in the element's own plane
    rho = sqrt(sum(Q.^2,1)).';

    info = struct('r_raw',NaN,'r_min',NaN,'centre_off_m',NaN, ...
                  'n_rays',nnz(ok),'elt',elt);
    if isempty(rho), r = NaN; return; end
    info.r_raw        = max(rho);
    info.r_min        = min(rho);
    info.centre_off_m = norm(mean(Q,2));
    r = opts.margin * info.r_raw;
    if ~opts.quiet
        fprintf(['    footprint @elt %d: r = %.4f m (raw %.4f, inner %.4f, ' ...
                 'centroid %.4f off vertex, %d rays)\n'], ...
                elt, r, info.r_raw, info.r_min, info.centre_off_m, info.n_rays);
    end
end
