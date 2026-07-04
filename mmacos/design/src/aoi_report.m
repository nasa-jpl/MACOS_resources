function rep = aoi_report(t, opts)
%AOI_REPORT  Per-mirror angle-of-incidence report -- the polarization
%   constraint check for coronagraph-facing designs.
%
%   rep = aoi_report(t) traces the design T (a macos.design.Telescope,
%   already built) and reports, for every powered element, the CHIEF-ray
%   angle of incidence and its VARIATION across the pupil (marginal
%   rays) and, optionally, across the field.
%
%   WHY (Dave, 2026-07-04): coronagraphic systems constrain mirror
%   incidence-angle VARIATION to keep polarization coupling manageable
%   -- the real metric is the AOI SPREAD across the rays of the beam
%   at each mirror (max-min over the pupil, and over the field patch),
%   with < 15 deg preferred.  This acts AGAINST compactness (and a
%   wide field): buying AOI margin generally takes a longer PM-SM
%   separation.  Report the spread as the headline (chief/min/max as
%   context) so the packaging<->polarization trade is explicit.
%
%   MEASUREMENT: engine-true, no surface normals needed -- a mirror
%   turns the beam by 180 deg - 2*AOI (normal incidence REVERSES it),
%   so AOI = 90 deg - acos(din.dout)/2 per ray, taken from the two
%   DRAW meridian fans (macos.draw_rays, data-only).  The per-element spread over the fans is the pupil
%   variation; rerun at field points (opts.fields) for the field
%   variation.
%
%   rep: struct array, one per reflecting element that bends the beam:
%     .name .elt
%     .aoi_chief_deg     chief-ray AOI (deg)
%     .aoi_min_deg/.aoi_max_deg   over pupil (and fields if given)
%     .aoi_spread_deg    max - min: THE polarization metric
%     .ok                aoi_spread_deg <= opts.limit_deg
%   Name-value:
%     'fields'     Kx2 (thx,thy) rad offsets to include (default: none;
%                  nominal field only).  Uses Telescope.trace_at_field.
%     'limit_deg'  the preference threshold (default 15).
%     'quiet'      suppress the printed table (default false).
%
%   See also wfe_field_diag, macos.design.Telescope/trace_at_field.
    arguments
        t
        opts.fields (:,2) double = zeros(0,2)
        opts.limit_deg (1,1) double = 15
        opts.quiet (1,1) logical = false
    end
    nE = numel(t.spec.elt);
    isMir = arrayfun(@(e) strcmp(e.kind,'Reflector'), t.spec.elt);

    % field list: nominal + any requested offsets
    F = [NaN NaN; opts.fields];          % NaN row = nominal (no re-point)
    aoi_min = inf(1,nE);  aoi_max = -inf(1,nE);  aoi_chief = nan(1,nE);
    cleanup = onCleanup(@() t.trace_at_field([]));
    for j = 1:size(F,1)
        if any(isnan(F(j,:)))
            t.trace_at_field([]);        % nominal (also (re)builds + traces)
        else
            t.trace_at_field(F(j,:));
        end
        % per-element beam centers from BOTH fans (as check_clipping):
        % each DRAW fan is a plane curve; its off-plane coordinate is the
        % beam center at that element.  Without this, a y-folded section
        % evaluated in the XZ projection loses the fold angle entirely.
        byz = macos.draw_rays('YZ', 0, nE);   % V=Y, U=Z (x ~ center)
        bxz = macos.draw_rays('XZ', 0, nE);   % V=X, U=Z (y ~ center)
        cx = zeros(1,nE);  cy = zeros(1,nE);
        for k = 1:nE
            mx = (bxz.elt == k);  my = (byz.elt == k);
            if any(mx(:)), cx(k) = mean(bxz.V(mx)); end
            if any(my(:)), cy(k) = mean(byz.V(my)); end
        end
        for pass = 1:2
            if pass == 1, b = byz; else, b = bxz; end
            isy = (pass == 1);
            for r = 1:b.nray
                npr = b.nper(r);
                for i = 2:npr-1
                    k = b.elt(i,r);
                    if k < 1 || k > nE || ~isMir(k), continue; end
                    P = zeros(3,3);           % 3-D points i-1, i, i+1
                    for q = -1:1
                        e2 = b.elt(i+q,r);
                        if e2 >= 1 && e2 <= nE
                            oc_x = cx(e2);  oc_y = cy(e2);
                        else
                            oc_x = cx(k);   oc_y = cy(k);
                        end
                        if isy, P(:,q+2) = [oc_x; b.V(i+q,r); b.U(i+q,r)];
                        else,   P(:,q+2) = [b.V(i+q,r); oc_y; b.U(i+q,r)];
                        end
                    end
                    d1 = P(:,2) - P(:,1);  d2 = P(:,3) - P(:,2);
                    n1 = norm(d1);  n2 = norm(d2);
                    if n1 < 1e-12 || n2 < 1e-12, continue; end
                    c = max(-1, min(1, (d1.'*d2)/(n1*n2)));
                    aoi = max(0, 90 - 0.5*acosd(c));
                    aoi_min(k) = min(aoi_min(k), aoi);
                    aoi_max(k) = max(aoi_max(k), aoi);
                    % chief ray: DRAW's fan includes the chief as the middle
                    % ray; approximate it as the fan-median at the nominal
                    % field on the first pass
                    if j == 1 && pass == 1 && r == ceil(b.nray/2)
                        aoi_chief(k) = aoi;
                    end
                end
            end
        end
    end

    rep = struct('name',{},'elt',{},'aoi_chief_deg',{}, ...
                 'aoi_min_deg',{},'aoi_max_deg',{},'aoi_spread_deg',{},'ok',{});
    for k = 1:nE
        if ~isMir(k) || ~isfinite(aoi_max(k)), continue; end
        spread = aoi_max(k) - aoi_min(k);
        rep(end+1) = struct('name',t.spec.elt(k).name, 'elt',k, ...
            'aoi_chief_deg',aoi_chief(k), 'aoi_min_deg',aoi_min(k), ...
            'aoi_max_deg',aoi_max(k), 'aoi_spread_deg',spread, ...
            'ok', spread <= opts.limit_deg);  %#ok<AGROW>
    end
    if ~opts.quiet
        fprintf('\naoi_report  (limit %g deg, %d field(s))\n', ...
                opts.limit_deg, size(F,1));
        fprintf('  %-8s %10s %10s %10s %12s   %s\n', 'mirror', ...
                'chief(deg)','min','max','SPREAD(deg)','ok');
        for i = 1:numel(rep)
            fprintf('  %-8s %10.2f %10.2f %10.2f %12.2f   %s\n', rep(i).name, ...
                rep(i).aoi_chief_deg, rep(i).aoi_min_deg, ...
                rep(i).aoi_max_deg, rep(i).aoi_spread_deg, ...
                ternary_(rep(i).ok,'yes','** >limit'));
        end
    end
end

function s = ternary_(c, a, b)
    if c, s = a; else, s = b; end
end
