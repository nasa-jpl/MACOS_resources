function rep = packaging_report(t, opts)
%PACKAGING_REPORT  Launch-shroud envelope of a design -- the packaging
%   metric (Dave, 2026-07-04): a space telescope must fit a CYLINDRICAL
%   launch shroud, so what matters is the radial extent of every body
%   and beam about the INCOMING-BEAM axis (keep M2 close to the
%   incoming beam as the PM-SM separation grows), plus the train
%   length.  Compactness trades directly against the coronagraph AOI
%   preference (see aoi_report) -- report both.
%
%   rep = packaging_report(t) measures, about the incoming chief-ray
%   axis (x=0, y=aperture decenter, direction +z):
%     .shroud_radius_m   max radial extent over all beams + bodies
%     .shroud_over_D     2*shroud_radius / D  (shroud diameter in D)
%     .length_m          axial extent (z span) of bodies + beams
%     .elts              per-element [name, r_body, r_beam] table
%
%   Beams come from the two DRAW meridian fans (data-only); bodies from
%   the realized clear apertures (spec.elt.ap, set by realize_apertures)
%   when present, else the element beam footprint.  Run
%   realize_apertures first for honest body sizes.
%
%   Name-value: 'quiet' (default false).
%
%   See also aoi_report, wfe_field_diag.
    arguments
        t
        opts.quiet (1,1) logical = false
    end
    nE = numel(t.spec.elt);
    dy = 0;
    if isfield(t.spec,'aperture_decenter'), dy = t.spec.aperture_decenter; end

    byz = macos.draw_rays('YZ', 0, nE);   % V=Y, U=Z (x ~ 0 for the section)
    bxz = macos.draw_rays('XZ', 0, nE);   % V=X, U=Z (y ~ per-elt center)

    % per-element beam center (for the x-fan's off-plane y)
    cy = zeros(1,nE);
    for k = 1:nE
        my = (byz.elt == k);
        if any(my(:)), cy(k) = mean(byz.V(my)); end
    end

    % beams: radial extent about the axis (x=0, y=dy)
    r_beam = zeros(1,nE);  zlo = inf;  zhi = -inf;
    for k = 1:nE
        my = (byz.elt == k);  mx = (bxz.elt == k);
        r = 0;
        if any(my(:))
            r = max(r, max(abs(byz.V(my) - dy)));            % x ~ 0
            zlo = min(zlo, min(byz.U(my)));  zhi = max(zhi, max(byz.U(my)));
        end
        if any(mx(:))
            r = max(r, max(hypot(bxz.V(mx), cy(k) - dy)));
            zlo = min(zlo, min(bxz.U(mx)));  zhi = max(zhi, max(bxz.U(mx)));
        end
        r_beam(k) = r;
    end

    % bodies: realized clear aperture [hw cx cy] about the element,
    % centered at (cx, cy) in the aperture plane -> radial extent =
    % |(cx, cy) - (0, dy)| + hw.  Fall back to the beam footprint.
    r_body = zeros(1,nE);
    for k = 1:nE
        e = t.spec.elt(k);
        if isfield(e,'ap') && ~isempty(e.ap) && numel(e.ap) >= 3
            r_body(k) = hypot(e.ap(2), e.ap(3) - dy) + e.ap(1);
        elseif isfield(e,'ap_rect') && ~isempty(e.ap_rect)
            xr = max(abs(e.ap_rect(1:2)));
            yr = max(abs(e.ap_rect(3:4) - dy));
            r_body(k) = hypot(xr, yr);
        else
            r_body(k) = r_beam(k);
        end
    end

    D  = t.spec.in.D;
    rs = max([r_beam, r_body]);
    rep = struct('shroud_radius_m',rs, 'shroud_over_D',2*rs/D, ...
                 'length_m',zhi - zlo, ...
                 'elts',struct('name',{t.spec.elt.name}, ...
                               'r_body',num2cell(r_body), ...
                               'r_beam',num2cell(r_beam)));
    if ~opts.quiet
        fprintf('\npackaging_report  (axis at y=%.3f m; D=%.3f m)\n', dy, D);
        fprintf('  %-8s %12s %12s\n','element','r_body(m)','r_beam(m)');
        for k = 1:nE
            fprintf('  %-8s %12.3f %12.3f\n', t.spec.elt(k).name, ...
                    r_body(k), r_beam(k));
        end
        fprintf(['  shroud radius %.3f m (diameter %.2f x D), train length' ...
                 ' %.2f m\n'], rs, rep.shroud_over_D, rep.length_m);
    end
end
