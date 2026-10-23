function f = spectrometer_layout_fig(G, file, opts)
%SPECTROMETER_LAYOUT_FIG  Deck-standard layout of a spectrometer_geom chain.
%   f = spectrometer_layout_fig(G, file) draws two sections of the chain in
%   millimetres with equal aspect and a scale bar: the DISPERSION plane
%   (y-z: slit centre, rays at 380 / 1440 / 2500 nm, chief + marginals) and
%   the SLIT direction (x-z: slit centre and both ends at the band centre).
%   Elements are drawn to their traced footprints (a margin beyond the
%   outermost ray), not as whole spheres: the block as its flat face and
%   spherical face joined, the grating arc, the meniscus faces, the slit and
%   FPA as marks.  Every figure the deck places comes from this producer,
%   unmodified (DECK_STYLE: figures are evidence).
%   Options: 'title' (default G.form), 'lams' (3 wavelengths), 'margin' (0.08).
    arguments
        G struct
        file (1,:) char
        opts.title (1,:) char = ''
        opts.lams (1,:) double = []
        opts.margin (1,1) double = 0.08
    end
    if isempty(opts.lams), opts.lams = [G.P.band_m(1), G.src.lambda_c, G.P.band_m(2)]; end
    if isempty(opts.title), opts.title = G.form; end
    col = [0 0.35 1; 0 0.6 0; 0.85 0 0];
    W = G.P.npix(1)*G.P.pixel_m;  xs3 = [-W/2 0 W/2];
    mm = 1e3;
    % ---- trace a bundle for footprints: 3 slit points x 3 lambdas x (chief + 8 marginals)
    nS = numel(G.surf);  hits = cell(1, nS);
    rays = {};   % {pts(3,:), lam index, slit index, kind}
    ang = linspace(0, 2*pi, 9);  ang(end) = [];
    for i = 1:3
        slit = G.slit + [xs3(i); 0; 0];
        for j = 1:3
            lam = opts.lams(j);  d0 = G.aim(slit, lam);
            ez = d0;  ex = cross([0;1;0], ez);  ex = ex/norm(ex);  ey = cross(ez, ex);
            dirs = [d0, cos(G.src.u)*d0 + sin(G.src.u)*(ex*cos(ang) + ey*sin(ang))];
            for k = 1:size(dirs, 2)
                [pts, ~, ok] = G.trace(slit, dirs(:,k), lam);
                if ~ok, continue; end
                for q = 1:nS, hits{q}(:, end+1) = pts(:, q); end
                rays{end+1} = struct('P', [slit, pts], 'j', j, 'i', i, 'k', k);  %#ok<AGROW>
            end
        end
    end
    f = figure('Visible', 'off', 'Position', [60 60 1250 560], 'Color', 'w');
    secs = {struct('name', 'dispersion plane (y-z)', 'a', 2, 'sel', @(r) r.i == 2), ...
            struct('name', 'slit direction (x-z)',  'a', 1, 'sel', @(r) r.j == 2)};
    for s = 1:2
        ax = subplot(1, 2, s);  hold(ax, 'on');  axis(ax, 'equal');  grid(ax, 'on');
        a = secs{s}.a;
        % ---- elements to their footprints
        for q = 1:nS
            S = G.surf(q);  H = hits{q};
            if isempty(H), continue; end
            switch S.kind
            case 'plane'
                if strcmp(S.act, 'stop')          % the FPA: a mark of its size, in ITS frame
                    if a == 2, ext = G.fpa.H/2;  e = G.fpa.yhat(:); else, ext = G.fpa.W/2;  e = G.fpa.xhat(:); end
                    c = G.fpa.center(:);  p1 = c - ext*e;  p2 = c + ext*e;
                    plot(ax, [p1(3) p2(3)]*mm, [p1(a) p2(a)]*mm, 'm-', 'LineWidth', 3);
                elseif any(strcmp(S.name, {'BlockFaceIn', 'BlockFaceOut'}))   % the block's flat face: drawn once below
                else                               % any other plane (plate, fold, prism exit): its trace through the hits
                    Q = [H(3,:); H(a,:)];  q0 = mean(Q, 2);  [U, ~] = svd(Q - q0, 'econ');  u = U(:,1);
                    tt = u'*(Q - q0);  pad = opts.margin*(max(tt) - min(tt) + 1e-3);
                    p1 = q0 + (min(tt) - pad)*u;  p2 = q0 + (max(tt) + pad)*u;
                    plot(ax, [p1(1) p2(1)]*mm, [p1(2) p2(2)]*mm, 'k-', 'LineWidth', 1.5);
                end
            case {'sphere', 'asph'}
                lo = min(H(a,:));  hi = max(H(a,:));  pad = opts.margin*(hi - lo + 1e-3);
                t = linspace(lo - pad, hi + pad, 121);
                % arc of the sphere in this section through the centre: z = Cz +- sqrt(R^2 - (t-Ca)^2 - off^2)
                other = 3 - a;  off = mean(H(other,:)) - S.C(other);
                rad2 = S.R^2 - (t - S.C(a)).^2 - off^2;  rad2(rad2 < 0) = NaN;
                z = S.C(3) + sign(mean(H(3,:)) - S.C(3))*sqrt(rad2);
                if strcmp(S.act, 'grating'), lw = 2.5;  cl = [0.3 0.3 0.3]; else, lw = 1.5;  cl = [0 0 0]; end
                plot(ax, z*mm, t*mm, '-', 'Color', cl, 'LineWidth', lw);
            end
        end
        if strcmp(G.form, 'dyson')
            % the block: flat face spanning the sphere's footprint, joined to the arc
            ib = find(strcmp({G.surf.name}, 'BlockSphereOut'), 1);  ifc = find(strcmp({G.surf.name}, 'BlockFaceIn'), 1);
            Hs = hits{ib};  lo = min(Hs(a,:));  hi = max(Hs(a,:));  pad = opts.margin*(hi - lo);
            Sb = G.surf(ib);  dz = G.surf(ifc).C(3);
            t = linspace(lo - pad, hi + pad, 121);  other = 3 - a;  off = mean(Hs(other,:)) - Sb.C(other);
            z = Sb.C(3) + sqrt(max(Sb.R^2 - (t - Sb.C(a)).^2 - off^2, 0));
            fill(ax, [dz, z, dz]*mm, [t(1), t, t(end)]*mm, [0.85 0.92 1], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
            plot(ax, [dz dz]*mm, [t(1) t(end)]*mm, 'k-', 'LineWidth', 1.5);
            plot(ax, z*mm, t*mm, 'k-', 'LineWidth', 1.5);
            % meniscus body
            im = find(strcmp({G.surf.name}, 'MenA_out'));
            if ~isempty(im)
                Ha = hits{im};  Hb = hits{im+1};  lo = min([Ha(a,:) Hb(a,:)]);  hi = max([Ha(a,:) Hb(a,:)]);
                t = linspace(lo, hi, 61);
                za = arc_(G.surf(im), t, a, mean(Ha(3-a,:)));  zb = arc_(G.surf(im+1), t, a, mean(Hb(3-a,:)));
                fill(ax, [za, fliplr(zb)]*mm, [t, fliplr(t)]*mm, [0.85 0.92 1], 'EdgeColor', 'k', 'LineWidth', 1, 'FaceAlpha', 0.6);
            end
        end
        % ---- rays
        for r = [rays{:}]
            if ~secs{s}.sel(r), continue; end
            if r.k == 1, lw = 1.2; else, lw = 0.5; end
            plot(ax, r.P(3,:)*mm, r.P(a,:)*mm, '-', 'Color', col(r.j,:), 'LineWidth', lw);
        end
        % slit mark
        if a == 2, plot(ax, G.slit(3)*mm, G.slit(2)*mm, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 5);
        else,      plot(ax, [G.slit(3) G.slit(3)]*mm, [-W/2 W/2]*mm, 'k-', 'LineWidth', 3);
        end
        % axes from the drawn data (padded), then the scale bar INSIDE them
        axis(ax, 'tight');  xl = xlim(ax);  yl = ylim(ax);
        xl = xl + [-0.06 0.04]*diff(xl);  yl = yl + [-0.18 0.08]*diff(yl);
        xlim(ax, xl);  ylim(ax, yl);  L = 100;
        x0 = xl(1) + 0.06*diff(xl);  y0 = yl(1) + 0.06*diff(yl);
        plot(ax, [x0, x0 + L], [y0, y0], 'k-', 'LineWidth', 3);
        text(ax, x0, y0 + 0.04*diff(yl), '100 mm', 'FontSize', 9);
        xlabel(ax, 'z (mm)');
        if a == 2, ylabel(ax, 'y, dispersion direction (mm)'); else, ylabel(ax, 'x, along the slit (mm)'); end
        if a == 2
            title(ax, sprintf('%s: %s -- slit centre, rays at %.0f / %.0f / %.0f nm (blue / green / red)', ...
                  opts.title, secs{s}.name, opts.lams*1e9), 'FontSize', 10);
        else
            title(ax, sprintf('%s: %s -- slit centre and ends at %.0f nm; FPA in magenta', ...
                  opts.title, secs{s}.name, opts.lams(2)*1e9), 'FontSize', 10);
        end
    end
    print(f, file, '-dpng', '-r130');  close(f);
end

function z = arc_(S, t, a, off_other)
    off = off_other - S.C(3 - a);
    rad2 = S.R^2 - (t - S.C(a)).^2 - off^2;  rad2(rad2 < 0) = NaN;
    % the vertex side: the sphere's face nearer the vertex point
    sgn = sign(S.vpt(3) - S.C(3));  if sgn == 0, sgn = -1; end
    z = S.C(3) + sgn*sqrt(rad2);
end
