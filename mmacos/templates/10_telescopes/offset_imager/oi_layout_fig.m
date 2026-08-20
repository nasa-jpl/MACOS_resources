function png = oi_layout_fig(X, G, P, offset_deg, stage_lbl, png)
%OI_LAYOUT_FIG  Y-Z layout figure: mirrors, stop, FP, traced field fans,
%   exit-beam annotation.  Saved to PNG (exportgraphics); the caller's
%   report section lists it.  Constraint annotations: the exit-beam
%   direction arrow and the smallest beam/mirror clearance from OI_GATES
%   are drawn when GT is available in the base workspace of the caller
%   (kept simple: the numbers live in the report; the figure shows the
%   geometry).
%
%   See also OFFSET_IMAGER, OI_GATES, OI_MAP_FIG.

    bx = P.box_deg(1)/2;  by = P.box_deg(2)/2;
    F = [0 offset_deg; 0 offset_deg-by; 0 offset_deg+by];   % Y-Z plane fans

    txt = oi_deck(fill_(X, P));
    sc = oi_score(txt, G, F, 'rays', true);

    fig = figure('Visible','off','Color','w','Position',[100 100 900 620]);
    ax = axes(fig);  hold(ax,'on');
    cols = [0.20 0.45 0.85; 0.85 0.45 0.20; 0.30 0.65 0.30];

    % ---- ray fans (chief + extreme y rays per field) ----------------------
    for q = 1:size(F,1)
        E = sc.rays{q};  if isempty(E), continue; end
        nE = numel(E);
        ok = E{1}.ok;  ok(1) = false;
        % pick chief + the two extreme-y rays at M1
        [~, ilo] = min(E{1}.pos(2,:) + 1e9*~ok');
        [~, ihi] = max(E{1}.pos(2,:) - 1e9*~ok');
        pick = [1 ilo ihi];
        for r = pick
            Z = zeros(1,nE);  Y = zeros(1,nE);
            good = true;
            for ie = 1:nE
                if r > 1 && ~E{ie}.ok(r), good = false; break; end
                Z(ie) = E{ie}.pos(3,r);  Y(ie) = E{ie}.pos(2,r);
            end
            if ~good, continue; end
            % incoming segment
            cd = tancomp_(F(q,1), F(q,2));
            zin = Z(1) - 0.35*cd(3);  yin = Y(1) - 0.35*cd(2);
            plot(ax, [zin Z], [yin Y], '-', 'Color', [cols(q,:) 0.55], ...
                 'LineWidth', 0.6 + 0.6*(r==1));
        end
    end

    % ---- mirrors as sag arcs over their footprints -------------------------
    z_stop = X.z_m1 + X.spacings(1);
    zEl = [X.z_m1, z_stop + X.spacings(2), z_stop + X.spacings(2) + X.spacings(3)];
    mirror_ie = [1 3 4];
    for m = 1:3
        E = sc.rays{1};  e = E{mirror_ie(m)};
        ok = e.ok;  ok(1) = false;
        ylo = min(e.pos(2,ok));  yhi = max(e.pos(2,ok));
        pad = 0.08*(yhi - ylo + 1e-6);
        yy = linspace(ylo - pad, yhi + pad, 60);
        % sag in the local frame (tilt small: draw about the vertex)
        c = 1/X.R(m);  k = X.K(m);
        h2 = (yy - X.yde(m)).^2;
        zz = zEl(m) + sign_(X, m)*0;   %#ok<NASGU>
        sag = c*h2 ./ (1 + sqrt(max(0, 1 - (1+k)*c^2*h2)));
        A = X.asph(m,:);
        sag = sag + A(1)*h2.^2 + A(2)*h2.^3 + A(3)*h2.^4;
        % surface z = vertex z + sag along local +z; local z ~ [0 -sin(a) cos(a)]
        al = X.ade(m);
        zs = zEl(m) + sag*cosd(al) - 0;      % small-tilt draw
        plot(ax, zs, yy, 'k-', 'LineWidth', 2.2);
        text(ax, zEl(m), yhi + 2.5*pad, sprintf('M%d', m), ...
             'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
    end

    % ---- stop + FP -----------------------------------------------------------
    plot(ax, [G.stopC(3) G.stopC(3)], G.stopC(2) + [-1 1]*0.055, 'r-', 'LineWidth', 1.6);
    text(ax, G.stopC(3), G.stopC(2) + 0.065, 'stop', 'Color','r', ...
         'HorizontalAlignment','center','FontSize',9);
    n = G.fpa.psi/norm(G.fpa.psi);
    tvec = [0; n(3); -n(2)];                 % in-plane Y-Z tangent
    fp = G.fpa.Vpt;
    seg = fp + tvec*linspace(-0.2, 0.2, 2);
    plot(ax, seg(3,:), seg(2,:), '-', 'Color',[0.4 0 0.6], 'LineWidth', 2.0);
    text(ax, fp(3), fp(2) - 0.03, 'FP', 'Color',[0.4 0 0.6], ...
         'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');

    % ---- exit-beam annotation --------------------------------------------------
    E = sc.rays{1};  ex_p = E{4}.pos(:,1);  ex_d = E{4}.dir(:,1);
    quiver(ax, ex_p(3), ex_p(2), 0.25*ex_d(3), 0.25*ex_d(2), 0, ...
           'Color',[0.85 0.1 0.1], 'LineWidth',1.4, 'MaxHeadSize',0.8);
    text(ax, ex_p(3)+0.26*ex_d(3), ex_p(2)+0.26*ex_d(2), ...
         sprintf('exit chief %.2f%c', atan2d(ex_d(2),ex_d(3)), char(176)), ...
         'Color',[0.85 0.1 0.1], 'FontSize', 9);

    axis(ax,'equal');  grid(ax,'on');
    xlabel(ax,'z (m)');  ylabel(ax,'y (m)');
    title(ax, sprintf('%s -- %s  (EPD %g mm, F/%.4g, box %gx%g%c at %+g%c)', ...
          P.name, stage_lbl, P.EPD_m*1e3, P.Fno, P.box_deg, char(176), ...
          offset_deg, char(176)), 'Interpreter','none');
    exportgraphics(fig, png, 'Resolution', 140);
    close(fig);
end

function s = sign_(~, ~), s = 1; end

function D = fill_(X, P)
    D = X;
    D.EPD_m = P.EPD_m;  D.WL_m = P.lambda_m;
    D.sampling = P.sampling;  D.name = P.name;
end

function d = tancomp_(xan_deg, yan_deg)
    d = [tand(xan_deg); tand(yan_deg); 1];
    d = d/norm(d);
end
