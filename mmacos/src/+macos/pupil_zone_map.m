function z = pupil_zone_map(pupil_elt, image_elt, opts)
%MACOS.PUPIL_ZONE_MAP  Pupil-resolved imaging quality (zone -> image spot).
%   z = macos.pupil_zone_map(PUPIL_ELT, IMAGE_ELT) partitions the ray
%   grid into zones by each ray's TRANSVERSE position at PUPIL_ELT, and
%   for each zone measures the RMS transverse spread of those same rays
%   at IMAGE_ELT.  Conceptually: a cone of rays leaves each patch of the
%   pupil; a perfectly imaging system maps every patch to the SAME image
%   point, so every zone collapses to one tight image spot.  The residual
%   per-zone spot is the pupil-resolved aberration -- the ray-optics
%   standard for pupil/imaging quality (e.g. is a coronagraph pupil a
%   sharp image of the DM).
%
%   Rays are matched across the two surfaces by grid index (same source
%   grid, same order), so no re-emission is needed -- it reads the loaded
%   system's current trace.
%
%   Name-value options:
%     'ngrid'  (default 5)   zones per axis (ngrid x ngrid over the pupil)
%     'shape'  'square'(default) | 'annular'  zone tiling: Cartesian bins,
%              or radial-x-azimuth rings (better for a round pupil)
%     'minrays'(default 4)   ignore zones with fewer live rays
%     'quiet'  (default false) suppress the summary print
%
%   Returns struct z:
%     .nzone         number of populated zones
%     .med_spot      median per-zone RMS image spot (BaseUnits)
%     .max_spot      worst per-zone RMS image spot
%     .rms_spot      RMS over zones
%     .spots         1 x nzone per-zone RMS spots
%     .zctr          2 x nzone pupil-frame zone centres (u,v)
%     .global_spot   RMS spread of ALL rays at IMAGE_ELT (for reference)
%     .pupil_elt, .image_elt, .ngrid, .shape
%
%   See also: macos.spot, macos.pupil_quality, macos.trace.
    arguments
        pupil_elt (1,1) double {mustBeInteger,mustBePositive}
        image_elt (1,1) double {mustBeInteger,mustBePositive}
        opts.ngrid   (1,1) double {mustBeInteger,mustBePositive} = 5
        opts.shape   (1,:) char {mustBeMember(opts.shape,{'square','annular'})} = 'square'
        opts.minrays (1,1) double {mustBeInteger,mustBePositive} = 4
        opts.quiet   (1,1) logical = false
    end

    % pupil-frame transverse coords (perp to the chief at PUPIL_ELT)
    sP = macos.trace(pupil_elt);  iP = macos.get_ray_info(sP.nRays);
    okP = iP.ok_trace(:) & iP.ok_pass(:);
    dch = iP.dir(:,1) / norm(iP.dir(:,1));
    relP = iP.pos - iP.pos(:,1);
    tvP  = relP - dch*(dch.'*relP);
    [e1,e2] = perpbasis(dch);
    u = e1.'*tvP;  v = e2.'*tvP;                 % 1 x N pupil coords

    % image-frame transverse positions of the SAME rays
    sI = macos.trace(image_elt);  iI = macos.get_ray_info(sI.nRays);
    okI = iI.ok_trace(:) & iI.ok_pass(:);
    fch = iI.dir(:,1) / norm(iI.dir(:,1));
    relI = iI.pos - iI.pos(:,1);
    tvI  = relI - fch*(fch.'*relI);              % 3 x N transverse at image

    ok = okP(:).' & okI(:).';
    assert(nnz(ok) >= opts.minrays, ...
        'pupil_zone_map: too few live rays (%d)', nnz(ok));

    R = max(hypot(u(ok), v(ok)));
    spots = [];  zctr = [];
    switch opts.shape
        case 'square'
            eu = linspace(-R, R, opts.ngrid+1);
            ev = linspace(-R, R, opts.ngrid+1);
            for a = 1:opts.ngrid
                for bb = 1:opts.ngrid
                    m = ok & u>=eu(a) & u<eu(a+1) & v>=ev(bb) & v<ev(bb+1);
                    if nnz(m) >= opts.minrays
                        [sp, cu, cv] = zone_spot(tvI, u, v, m);
                        spots(end+1) = sp;         %#ok<AGROW>
                        zctr(:,end+1) = [cu; cv];  %#ok<AGROW>
                    end
                end
            end
        case 'annular'
            rho = hypot(u, v);  th = atan2(v, u);
            er = linspace(0, R, opts.ngrid+1);
            na = max(1, opts.ngrid);
            eth = linspace(-pi, pi, na+1);
            for a = 1:opts.ngrid
                for bb = 1:na
                    m = ok & rho>=er(a) & rho<er(a+1) & th>=eth(bb) & th<eth(bb+1);
                    if nnz(m) >= opts.minrays
                        [sp, cu, cv] = zone_spot(tvI, u, v, m);
                        spots(end+1) = sp;         %#ok<AGROW>
                        zctr(:,end+1) = [cu; cv];  %#ok<AGROW>
                    end
                end
            end
    end

    gd = tvI(:, ok);  gc = mean(gd, 2);
    z.global_spot = sqrt(mean(sum((gd - gc).^2, 1)));
    z.nzone = numel(spots);
    z.spots = spots;  z.zctr = zctr;
    z.med_spot = median(spots);  z.max_spot = max(spots);
    z.rms_spot = sqrt(mean(spots.^2));
    z.pupil_elt = pupil_elt;  z.image_elt = image_elt;
    z.ngrid = opts.ngrid;  z.shape = opts.shape;

    if ~opts.quiet
        fprintf(['pupil_zone_map elt %d -> %d: %d zones (%s %dx%d)  ' ...
                 'median %.4g  worst %.4g  (global %.4g) BaseUnits\n'], ...
            pupil_elt, image_elt, z.nzone, opts.shape, opts.ngrid, ...
            opts.ngrid, z.med_spot, z.max_spot, z.global_spot);
    end
end

function [sp, cu, cv] = zone_spot(tvI, u, v, m)
    d = tvI(:, m);  c = mean(d, 2);
    sp = sqrt(mean(sum((d - c).^2, 1)));
    cu = mean(u(m));  cv = mean(v(m));
end

function [e1,e2] = perpbasis(d)
    d = d / norm(d);
    e1 = cross([0;0;1], d);
    if norm(e1) < 1e-9, e1 = cross([0;1;0], d); end
    e1 = e1 / norm(e1);  e2 = cross(d, e1);
end
