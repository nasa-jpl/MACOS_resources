function [png, png_fields] = oi_layout_fig(X, G, P, offset_deg, stage_lbl, png)
%OI_LAYOUT_FIG  Stage layout figures: the solid hardware view + the
%   field-envelope elevation.
%
%   [PNG, PNG_FIELDS] = OI_LAYOUT_FIG(X, G, P, OFFSET, LBL, PNG) writes
%   TWO figures for the stage:
%
%     PNG           (<tag>_sN_layout.png)  macos.view_std -- the
%                   standard four-panel solid-body render of the loaded
%                   deck (true sag, footprint-hull apertures, outline
%                   frames for the stop/FP), aimed at the box-centre
%                   field.  This is the interpretable hardware drawing.
%
%     PNG_FIELDS    (<tag>_sN_fields.png)  a Y-Z elevation telling the
%                   FIELD story: per-field beam ENVELOPES as filled
%                   translucent patches (box centre + the two YAN
%                   extremes), chief rays, thick sag-profile mirrors,
%                   stop and FP glyphs, and the exit-chief annotation.
%                   Envelopes, not ray fans -- a 3x27 spaghetti of
%                   crossing lines is not a layout.
%
%   See also OFFSET_IMAGER, MACOS.VIEW_STD, MACOS.VIEW_RX.

    bx = P.box_deg(1)/2;  by = P.box_deg(2)/2; %#ok<NASGU>
    F = [0 offset_deg; 0 offset_deg-by; 0 offset_deg+by];

    txt = oi_deck(fill_(X, P));
    sc = oi_score(txt, G, F, 'rays', true);

    % ================= 1. the solid hardware view (view_std) ===============
    % load the deck aimed at the box-centre chief, then render
    tmp = [tempname '.in'];
    cu  = onCleanup(@() delete_if_(tmp));
    cdir = tancomp_(F(1,1), F(1,2));
    emit_src_(txt, tmp, seed_pos_(G, cdir), cdir);
    macos.load_rx(tmp);
    macos.stop(2, [0 0]);
    macos.view_std('save', png, 'visible', false, ...
        'title', sprintf('%s -- %s', P.name, stage_lbl), ...
        'args', {'nrings', 2, 'nspokes', 8});

    % ================= 2. the field-envelope elevation ======================
    png_fields = regexprep(png, '_layout\.png$', '_fields.png');
    if strcmp(png_fields, png), png_fields = [png(1:end-4) '_fields.png']; end

    fig = figure('Visible','off','Color','w','Position',[80 80 1000 640]);
    ax = axes(fig);  hold(ax,'on');
    cols = [0.20 0.45 0.85; 0.87 0.52 0.15; 0.30 0.62 0.28];
    fldlbl = arrayfun(@(k) sprintf('XAN %+g%c YAN %+g%c', F(k,1), char(176), ...
                      F(k,2), char(176)), 1:3, 'UniformOutput', false);

    % ---- per-field beam envelopes leg by leg ------------------------------
    hleg = gobjects(1,3);
    for q = 1:3
        E = sc.rays{q};  if isempty(E), continue; end
        nE = numel(E);
        ok = true(size(E{1}.ok));
        for ie = 1:nE, ok = ok & E{ie}.ok; end
        ok(1) = false;
        % incoming leg: reconstruct upstream of M1
        cd = tancomp_(F(q,1), F(q,2));
        P1 = E{1}.pos(:,ok);
        P0 = P1 - 0.30*cd;                        % 300 mm of approach
        hleg(q) = env_(ax, P0, P1, cols(q,:));
        for ie = 1:nE-1
            env_(ax, E{ie}.pos(:,ok), E{ie+1}.pos(:,ok), cols(q,:));
        end
        % chief polyline
        Z = zeros(1,nE+1);  Y = zeros(1,nE+1);
        Z(1) = E{1}.pos(3,1) - 0.30*cd(3);  Y(1) = E{1}.pos(2,1) - 0.30*cd(2);
        for ie = 1:nE, Z(ie+1) = E{ie}.pos(3,1); Y(ie+1) = E{ie}.pos(2,1); end
        plot(ax, Z, Y, '-', 'Color', cols(q,:)*0.75, 'LineWidth', 1.4);
    end

    % ---- mirrors as thick sag profiles over their patches ------------------
    z_stop = X.z_m1 + X.spacings(1);
    zEl = [X.z_m1, z_stop + X.spacings(2), z_stop + X.spacings(2) + X.spacings(3)];
    mirror_ie = [1 3 4];
    for m = 1:3
        ylo = inf;  yhi = -inf;
        for q = 1:3
            E = sc.rays{q};  if isempty(E), continue; end
            e = E{mirror_ie(m)};  ok = e.ok;  ok(1) = false;
            ylo = min(ylo, min(e.pos(2,ok)));  yhi = max(yhi, max(e.pos(2,ok)));
        end
        pad = 0.06*(yhi - ylo + 1e-6);
        yy = linspace(ylo - pad, yhi + pad, 80);
        c = 1/X.R(m);  k = X.K(m);
        h2 = (yy - X.yde(m)).^2;
        sag = c*h2 ./ (1 + sqrt(max(0, 1 - (1+k)*c^2*h2)));
        A = X.asph(m,:);
        sag = sag + A(1)*h2.^2 + A(2)*h2.^3 + A(3)*h2.^4;
        al = X.ade(m);
        zs = zEl(m) + sag*cosd(al);
        % substrate: a filled sliver behind the optical face
        tk = 0.012*sign_away_(X, m);
        patch(ax, [zs, fliplr(zs)+tk], [yy, fliplr(yy)], [0.25 0.25 0.28], ...
              'EdgeColor','none', 'FaceAlpha', 1);
        text(ax, zEl(m)+tk*1.6, yhi + 2.2*pad, sprintf('M%d', m), ...
             'HorizontalAlignment','center', 'FontSize', 11, 'FontWeight','bold');
    end

    % ---- stop + FP glyphs ---------------------------------------------------
    rs = 0.055;
    plot(ax, [G.stopC(3) G.stopC(3)], G.stopC(2)+[rs 2.2*rs], 'r-', 'LineWidth', 3);
    plot(ax, [G.stopC(3) G.stopC(3)], G.stopC(2)-[rs 2.2*rs], 'r-', 'LineWidth', 3);
    text(ax, G.stopC(3), G.stopC(2)+2.7*rs, 'stop', 'Color','r', ...
         'HorizontalAlignment','center', 'FontSize', 10, 'FontWeight','bold');
    n = G.fpa.psi/norm(G.fpa.psi);
    tvec = [0; n(3); -n(2)];
    fp = G.fpa.Vpt;
    seg = fp + tvec*linspace(-0.16, 0.16, 2);
    plot(ax, seg(3,:), seg(2,:), '-', 'Color',[0.45 0 0.65], 'LineWidth', 3.5);
    text(ax, fp(3), min(seg(2,:)) - 0.035, 'FP', 'Color',[0.45 0 0.65], ...
         'HorizontalAlignment','center', 'FontSize', 11, 'FontWeight','bold');

    % ---- exit-chief annotation, placed off the beam -------------------------
    E = sc.rays{1};  ex_p = E{4}.pos(:,1);  ex_d = E{4}.dir(:,1);
    a = quiver(ax, ex_p(3)+0.05*ex_d(3), ex_p(2)+0.05*ex_d(2), ...
               0.22*ex_d(3), 0.22*ex_d(2), 0, 'Color',[0.8 0.1 0.1], ...
               'LineWidth', 2, 'MaxHeadSize', 0.9); %#ok<NASGU>
    text(ax, ex_p(3)+0.30*ex_d(3), ex_p(2)+0.30*ex_d(2)-0.03, ...
         sprintf('exit chief %.2f%c', atan2d(ex_d(2), ex_d(3)), char(176)), ...
         'Color',[0.8 0.1 0.1], 'FontSize', 10, 'FontWeight','bold');

    legend(ax, hleg, fldlbl, 'Location','best', 'AutoUpdate','off', 'Box','off');
    axis(ax,'equal');  grid(ax,'on');
    % pad the axes so element labels near the data extremes stay on-canvas
    xl = xlim(ax);  yl = ylim(ax);
    xlim(ax, xl + [-0.05 0.05]*diff(xl));
    ylim(ax, yl + [-0.05 0.05]*diff(yl));
    xlabel(ax, 'z (m)');  ylabel(ax, 'y (m)');
    title(ax, sprintf('%s -- %s -- field envelopes  (EPD %g mm, F/%.4g, box %gx%g%c at %+g%c)', ...
          P.name, stage_lbl, P.EPD_m*1e3, P.Fno, P.box_deg, char(176), ...
          offset_deg, char(176)), 'Interpreter','none');
    exportgraphics(fig, png_fields, 'Resolution', 140);
    close(fig);
end

% =========================================================================
function h = env_(ax, A, B, col)
%ENV_  Filled Y-Z envelope patch between ray sets A (3,N) -> B (3,N).
    [~, ia1] = min(A(2,:));  [~, ia2] = max(A(2,:));
    [~, ib1] = min(B(2,:));  [~, ib2] = max(B(2,:));
    zz = [A(3,ia1) B(3,ib1) B(3,ib2) A(3,ia2)];
    yy = [A(2,ia1) B(2,ib1) B(2,ib2) A(2,ia2)];
    h = patch(ax, zz, yy, col, 'FaceAlpha', 0.22, 'EdgeColor', col, ...
              'EdgeAlpha', 0.6, 'LineWidth', 0.5);
end

function s = sign_away_(X, m)
%SIGN_AWAY_  Substrate direction: behind the optical face (away from the
%   incoming beam), from the mirror order (M1,M3 face -z; M2 faces +z
%   in the W-fold train when spacings alternate sign).
    if m == 2, s = -1; else, s = 1; end
    if X.spacings(1) > 0, s = -s; end
end

function D = fill_(X, P)
    D = X;
    D.EPD_m = P.EPD_m;  D.WL_m = P.lambda_m;
    D.sampling = P.sampling;  D.name = P.name;
end

function p = seed_pos_(G, cdir)
    cdR = [cdir(1); cdir(2); -cdir(3)];
    tq  = (G.z_m1 - G.stopC(3))/cdir(3);
    q   = G.stopC - tq*cdR;
    p   = q - (0.75/cdir(3))*cdir;
end

function emit_src_(txt0, tmp, p0, cdir)
    v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));
    s = regexprep(txt0, '(ChfRayDir=\s*)[^\n]*', ['$1' v3(cdir)]);
    s = regexprep(s,    '(ChfRayPos=\s*)[^\n]*', ['$1' v3(p0)]);
    fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
end

function d = tancomp_(xan_deg, yan_deg)
    d = [tand(xan_deg); tand(yan_deg); 1];
    d = d/norm(d);
end

function delete_if_(p), if exist(p,'file'), delete(p); end, end
