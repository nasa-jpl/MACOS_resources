function out = e2e6m_shroud_fig(t, png, opts)
%E2E6M_SHROUD_FIG  The launch-shroud fit figure -- the packaging gate, drawn.
%
%   The campaign's packaging constraint is a DEPLOYED, DIAMETER-ONLY gate
%   (Dave 2026-08-24): every optic body and every internal beam leg must
%   sit inside an 8 m diameter cylinder about the launch axis; length is
%   free and stated.  The ENTRY CORRIDOR -- the incoming beam upstream of
%   the primary -- is drawn but reported SEPARATELY, as the sunshade
%   keep-out: it is sky, not hardware, and does not count against the
%   diameter.
%
%   out = E2E6M_SHROUD_FIG(T, PNG) draws two panels for the built
%   Telescope T -- an end-on view down the launch axis with the shroud
%   circle, and a Y-Z elevation with the shroud walls -- saves PNG, and
%   returns the measured numbers (the same ones packaging_report gives,
%   recomputed here from the same fans so the figure and the gate cannot
%   disagree).
%
%   Name-value:
%     'shroud_D_m'  gate diameter (default 8)
%     'entry_m'     how far upstream of the first element to draw the
%                   entry corridor (default 6)
%     'title'       figure title text
%     'visible'     (default false)
%
%   See also PACKAGING_REPORT, macos.design.Telescope/check_clipping.

    arguments
        t
        png (1,:) char
        opts.shroud_D_m (1,1) double = 8.0
        opts.entry_m    (1,1) double = 6.0
        opts.title      (1,:) char   = ''
        opts.visible    (1,1) logical = false
    end
    nE = numel(t.spec.elt);
    dy = 0;
    if isfield(t.spec,'aperture_decenter'), dy = t.spec.aperture_decenter; end

    byz = macos.draw_rays('YZ', 0, nE);    % y-fan: V=Y, U=Z
    bxz = macos.draw_rays('XZ', 0, nE);    % x-fan: V=X, U=Z

    % per-element beam centre and footprint radius, exactly as
    % packaging_report / check_clipping read them
    C = zeros(3,nE);  R = zeros(1,nE);
    zlo = inf;  zhi = -inf;
    for k = 1:nE
        my = (byz.elt == k);  mx = (bxz.elt == k);
        if ~any(my(:)) && ~any(mx(:)), C(:,k) = t.spec.elt(k).Vpt(:); continue; end
        cx = t.spec.elt(k).Vpt(1);  if any(mx(:)), cx = mean(bxz.V(mx)); end
        cy = t.spec.elt(k).Vpt(2);  if any(my(:)), cy = mean(byz.V(my)); end
        zz = [byz.U(my); bxz.U(mx)];
        C(:,k) = [cx; cy; mean(zz(:))];
        ry = 0;  if any(my(:)), ry = max(abs(byz.V(my) - cy)); end
        rx = 0;  if any(mx(:)), rx = max(abs(bxz.V(mx) - cx)); end
        R(k) = max(rx, ry);
        zlo = min(zlo, min(zz(:)));  zhi = max(zhi, max(zz(:)));
    end
    r_rad = hypot(C(1,:), C(2,:) - dy) + R;      % radial extent per element
    out = struct('shroud_D_m', 2*max(r_rad), 'r_elt', r_rad, ...
                 'names', {{t.spec.elt.name}}, ...
                 'length_m', zhi - zlo, 'gate_D_m', opts.shroud_D_m);
    out.pass = out.shroud_D_m <= opts.shroud_D_m;

    f = figure('Visible', tern_(opts.visible,'on','off'), ...
               'Position',[100 100 1250 560]);

    % ---- panel 1: end-on, down the launch axis --------------------------
    ax1 = subplot(1,2,1);  hold(ax1,'on');  axis(ax1,'equal');
    th = linspace(0,2*pi,361);
    Rg = opts.shroud_D_m/2;
    plot(ax1, Rg*cos(th), dy + Rg*sin(th), 'k-', 'LineWidth', 2.0);
    cols = lines(max(nE,7));
    for k = 1:nE
        plot(ax1, C(1,k)+R(k)*cos(th), C(2,k)+R(k)*sin(th), '-', ...
             'Color', cols(k,:), 'LineWidth', 1.6);
    end
    % labels on a leader out to the shroud wall, spread by angle, so the
    % elements crowded into the annulus do not print on top of each other
    lab_r = 1.06*Rg;
    for k = 1:nE
        a  = atan2(C(2,k) - dy, C(1,k));
        if hypot(C(1,k), C(2,k)-dy) < 0.15*Rg, a = pi/2; end   % on-axis elt
        a  = a + (k - (nE+1)/2) * (0.30/max(nE,1)) * pi;
        px = lab_r*cos(a);  py = dy + lab_r*sin(a);
        plot(ax1, [C(1,k) px], [C(2,k) py], '-', ...
             'Color', [cols(k,:) 0.45], 'LineWidth', 0.8);
        ha = 'left';  if px < 0, ha = 'right'; end
        text(ax1, px, py, [' ' t.spec.elt(k).name ' '], ...
             'HorizontalAlignment', ha, 'FontSize', 9, 'Color', cols(k,:)*0.7);
    end
    xlabel(ax1,'x  [m]');  ylabel(ax1,'y  [m]');
    title(ax1, sprintf('end-on: union %.3f m against the %.1f m gate  (%s)', ...
          out.shroud_D_m, opts.shroud_D_m, tern_(out.pass,'FITS','OVER')));
    grid(ax1,'on');  box(ax1,'on');
    lim = 1.45*Rg;  xlim(ax1,[-lim lim]);  ylim(ax1, dy + [-lim lim]);

    % ---- panel 2: Y-Z elevation with the shroud walls -------------------
    % NOT axis-equal: the train is ~2x longer than the shroud is wide, and
    % an equal-aspect elevation shows a tenth of it.
    ax2 = subplot(1,2,2);  hold(ax2,'on');
    ze = [zlo - opts.entry_m, zhi + 0.05*(zhi-zlo)];
    % entry corridor upstream of the first element (drawn, NOT gated)
    r1 = R(1);
    patch(ax2, [ze(1) C(3,1) C(3,1) ze(1)], ...
          [C(2,1)-r1 C(2,1)-r1 C(2,1)+r1 C(2,1)+r1], [0.80 0.80 0.80], ...
          'FaceAlpha',0.30, 'EdgeColor','none');
    plot(ax2, byz.U, byz.V, '-', 'Color',[0.30 0.55 0.85], 'LineWidth',0.5);
    plot(ax2, ze, dy + [Rg Rg], 'k-', 'LineWidth',2.0);
    plot(ax2, ze, dy - [Rg Rg], 'k-', 'LineWidth',2.0);
    for k = 1:nE
        plot(ax2, C(3,k)+[0 0], C(2,k)+[-R(k) R(k)], '-', ...
             'Color', cols(k,:), 'LineWidth', 3.0);
        yl = C(2,k) + R(k) + 0.35 + 0.35*mod(k,2);   % alternate heights
        plot(ax2, C(3,k)+[0 0], [C(2,k)+R(k) yl-0.12], '-', ...
             'Color', [cols(k,:) 0.45], 'LineWidth', 0.8);
        text(ax2, C(3,k), yl, t.spec.elt(k).name, ...
             'HorizontalAlignment','center', 'FontSize',9, 'Color',cols(k,:)*0.7);
    end
    xlabel(ax2,'z  [m]  (launch axis; beam enters +z)');  ylabel(ax2,'y  [m]');
    title(ax2, sprintf(['elevation: train %.2f m long; entry corridor shaded ' ...
                        '(sunshade keep-out, not gated)'], out.length_m));
    grid(ax2,'on');  box(ax2,'on');
    xlim(ax2, ze);  ylim(ax2, dy + [-1.35*Rg 1.35*Rg]);

    if ~isempty(opts.title)
        try, sgtitle(f, opts.title, 'FontWeight','bold'); catch, end
    end
    saveas(f, png);
    if ~opts.visible, close(f); end
    out.png = png;
end

function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
