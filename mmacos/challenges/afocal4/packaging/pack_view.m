function fig = pack_view(decks, labels, opts)
%PACK_VIEW  Packaging elevation of one or more decks, against the yardstick.
%
%   FIG = PACK_VIEW({DECK1,...}, {LAB1,...}) draws two elevation panels per
%   deck -- x-z (the plane the folds turn in) over y-z (the plane the field
%   bias and the coldstop tilt live in) -- all on a COMMON scale, with:
%     * the traced rays (a meridian subsample of the real ray grid, from
%       ray_hist -- not the DRAW fan, which caps at 101 rays);
%     * each optic as a bar of its measured footprint, normal to psi;
%     * the STATED envelope: the cylinder r <= r_env about the telescope
%       axis and the slab 0 < z <= z_slab behind the primary;
%     * the M1-M2 spacing drawn as the dimension it is being judged against;
%     * the instrument volume, a box of the stated length and diameter
%       running from the interface plane along the exit chief.
%
%   The point of the figure is the comparison, so the panels share axes: an
%   overhang that needs a different scale to be visible is not the overhang
%   anyone is arguing about.
%
%   Name-value: 'r_env' (0.560) 'z_slab' (0 = the deck's own M1-M2 spacing)
%     'instr_len' (1.000) 'instr_dia' (0.300) 'nray' (25) 'save' 'title'
%
%   See also PACK_LEGS, PACK_CLEAR, MACOS.VIEW_STD.

    arguments
        decks  cell
        labels cell
        opts.r_env     (1,1) double = 0.560
        opts.z_slab    (1,1) double = 0
        opts.instr_len (1,1) double = 1.000
        opts.instr_dia (1,1) double = 0.300
        opts.nray      (1,1) double = 25
        opts.save      (1,:) char = ''
        opts.title     (1,:) char = ''
    end

    n = numel(decks);
    G = cell(1,n);
    for i = 1:n, G{i} = grab_(decks{i}, opts); end

    zl = [min(cellfun(@(g) min(g.zlim(1)), G)), max(cellfun(@(g) max(g.zlim(2)), G))];
    xl = [min(cellfun(@(g) min(g.xlim(1)), G)), max(cellfun(@(g) max(g.xlim(2)), G))];
    pad = 0.05*max(diff(zl), diff(xl));
    zl = zl + [-pad pad];   xl = xl + [-pad pad];

    yl = [min(cellfun(@(g) g.ylim(1), G)), max(cellfun(@(g) g.ylim(2), G))];
    yl = yl + [-pad pad];
    lim = [min([xl yl]) max([xl yl])];

    fig = figure('Visible','off', 'Position',[40 40 660*n 900]);
    tl  = tiledlayout(fig, 2, n, 'Padding','compact', 'TileSpacing','compact');
    for row = 1:2
        for i = 1:n
            ax = nexttile(tl, (row-1)*n + i);  hold(ax,'on');  box(ax,'on');
            draw_(ax, G{i}, opts, row);
            axis(ax,'equal');   xlim(ax, zl);   ylim(ax, lim);
            xlabel(ax,'z (m)   (sky at -z; behind the primary is +z)');
            if row == 1
                ylabel(ax,'x (m)   -- the fold plane');
                title(ax, labels{i}, 'Interpreter','none');
            else
                ylabel(ax,'y (m)   -- the field-bias plane');
            end
            grid(ax,'on');
        end
    end
    if ~isempty(opts.title)
        title(tl, opts.title, 'FontWeight','bold');
    end
    if ~isempty(opts.save)
        exportgraphics(fig, opts.save, 'Resolution', 150);
        fprintf('  wrote %s\n', opts.save);
    end
end

% =====================================================================
function g = grab_(deck, opts)
    macos.load_rx(deck);
    nE = macos.num_elt();
    macos.ray_hist('on');  t = macos.trace();  h = macos.ray_hist(t.nRays);
    macos.ray_hist('off');
    off = size(h.P,3) - nE;

    % Which rays to draw.  Two sets: a MERIDIAN band (the rays whose stop
    % crossing lies near the other transverse axis, so the elevation shows a
    % fan and not a smear) and, riding under it, a thin uniform sample of the
    % whole grid so the bundle ENVELOPE is visible even where the band is
    % narrow.  Drawing the full grid alone is a smear; the band alone can
    % come out as a single line on a grid whose rows do not land on the
    % meridian.
    Q1 = squeeze(h.P(:,:,1+off));
    c1 = mean(Q1(:,h.ok(:,1+off)),2);
    rmax = max(vecnorm(Q1 - c1));
    nAll = size(h.P,2);
    g.idx = cell(1,2);   g.env = 1:max(1,floor(nAll/60)):nAll;
    for cc = 1:2
        other = 3 - cc;
        b = find(abs(Q1(other,:) - c1(other)) < 0.05*rmax);
        if isempty(b), b = g.env; end
        if numel(b) > opts.nray, b = b(round(linspace(1,numel(b),opts.nray))); end
        g.idx{cc} = b;
    end
    g.P = h.P;   g.ok = h.ok;   g.off = off;   g.nE = nE;

    g.vpt = zeros(3,nE);  g.psi = zeros(3,nE);  g.fr = zeros(1,nE);
    g.fc  = zeros(3,nE);
    for k = 1:nE
        g.vpt(:,k) = macos.get_elt_vpt(k);
        g.psi(:,k) = macos.get_elt_psi(k);
        m = h.ok(:,k+off);   Q = squeeze(h.P(:,m,k+off));
        g.fc(:,k) = mean(Q,2);
        g.fr(k)   = max(vecnorm(Q - g.fc(:,k)));
    end
    nm = regexp(fileread(deck), '(?m)^\s*EltName=\s*(\S*)', 'tokens');
    g.name = cellfun(@(c) c{1}, nm, 'UniformOutput', false);
    if numel(g.name) ~= nE
        g.name = arrayfun(@(k) sprintf('e%d',k), 1:nE, 'UniformOutput', false);
    end

    g.dM1M2 = abs(min(g.vpt(3,:)));
    % the instrument: from the interface plane along the exit chief
    Pc = squeeze(h.P(:,1,:));
    a  = Pc(:,nE+off) - Pc(:,nE-1+off);   a = a/norm(a);
    g.instr = struct('p0', g.fc(:,nE), 'dir', a, 'len', opts.instr_len, ...
                     'r', 0.5*opts.instr_dia);
    p1 = g.instr.p0 + a*opts.instr_len;
    g.zlim = [min([g.vpt(3,:), g.instr.p0(3), p1(3)]) - 0.2, ...
              max([g.vpt(3,:), g.instr.p0(3), p1(3)]) + 0.2];
    g.xlim = [min([g.fc(1,:)-g.fr, g.instr.p0(1)-g.instr.r, p1(1)-g.instr.r]), ...
              max([g.fc(1,:)+g.fr, g.instr.p0(1)+g.instr.r, p1(1)+g.instr.r])];
    g.ylim = [min([g.fc(2,:)-g.fr, g.instr.p0(2)-g.instr.r, p1(2)-g.instr.r]), ...
              max([g.fc(2,:)+g.fr, g.instr.p0(2)+g.instr.r, p1(2)+g.instr.r])];
end

function draw_(ax, g, opts, row)
%DRAW_  One elevation.  ROW 1 is the x-z section (the plane the folds turn
%   in); ROW 2 is the y-z section (the plane the field bias and the coldstop
%   tilt live in, and therefore the one that sets how far off the vertex
%   every part has to reach).
    if nargin < 4, row = 1; end
    c = 1;  if row == 2, c = 2; end          % which transverse coordinate

    zs = opts.z_slab;   if zs <= 0, zs = g.dM1M2; end

    % stated envelope: the slab behind the primary, inside the keep-out
    patch(ax, [0 zs zs 0], [-opts.r_env -opts.r_env opts.r_env opts.r_env], ...
          [0.90 0.94 1.00], 'EdgeColor',[0.45 0.58 0.82], 'LineStyle','--', ...
          'FaceAlpha',0.35, 'LineWidth',1.0);
    % the incoming beam, in front of the primary -- outline plus the
    % faintest wash, because the ray fan lives inside it and a solid tint
    % there hides exactly the bundle the figure is about
    patch(ax, [g.zlim(1) 0 0 g.zlim(1)], [-0.5 -0.5 0.5 0.5], ...
          [1.00 0.90 0.80], 'EdgeColor',[0.85 0.62 0.42], 'LineStyle',':', ...
          'FaceAlpha',0.16, 'LineWidth',0.9);

    % rays: the whole-grid envelope first, then the meridian fan over it
    for r = g.env
        j = find(g.ok(r,:));
        plot(ax, squeeze(g.P(3,r,j)), squeeze(g.P(c,r,j)), '-', ...
             'Color',[0.93 0.72 0.55 0.55], 'LineWidth',0.4);
    end
    for r = g.idx{c}
        j = find(g.ok(r,:));
        plot(ax, squeeze(g.P(3,r,j)), squeeze(g.P(c,r,j)), '-', ...
             'Color',[0.78 0.26 0.05 0.90], 'LineWidth',0.9);
    end

    % optics, as bars of the measured footprint normal to psi
    for k = 1:g.nE
        p = g.psi(:,k);
        u = zeros(3,1);   u(c) = -p(3);   u(3) = p(c);
        if norm(u) < 1e-9, u = zeros(3,1);  u(c) = 1; end
        u = u/norm(u);
        a = g.fc(:,k) - u*g.fr(k);   b = g.fc(:,k) + u*g.fr(k);
        plot(ax, [a(3) b(3)], [a(c) b(c)], 'k-', 'LineWidth', 2.4);
        sgn = 1 - 2*mod(k,2);
        dy  = sgn*0.070*(1 + mod(k,3));
        text(ax, g.fc(3,k), g.fc(c,k) + dy, g.name{k}, 'FontSize',8, ...
             'Interpreter','none', 'HorizontalAlignment','center', ...
             'BackgroundColor','w', 'Margin',0.5);
    end

    % instrument volume, along the exit chief
    I = g.instr;   a = I.dir;
    w = zeros(3,1);   w(c) = -a(3);   w(3) = a(c);
    if norm(w) < 1e-9, w = zeros(3,1);  w(c) = 1; end
    w = w/norm(w)*I.r;
    q = [I.p0 + w, I.p0 + a*I.len + w, I.p0 + a*I.len - w, I.p0 - w];
    patch(ax, q(3,:), q(c,:), [0.25 0.58 0.32], 'FaceAlpha',0.20, ...
          'EdgeColor',[0.12 0.42 0.18], 'LineWidth',1.1);
    text(ax, mean(q(3,:)), mean(q(c,:)), 'instrument', 'FontSize',8, ...
         'Color',[0.10 0.35 0.15], 'HorizontalAlignment','center');

    % the yardstick: the M1-M2 spacing, drawn where it is being compared
    yq = opts.r_env*0.90;
    plot(ax, [0 g.dM1M2], [yq yq], '-', 'Color',[0 0.2 0.8], 'LineWidth',1.6);
    plot(ax, [0 0], yq+[-0.035 0.035], '-', 'Color',[0 0.2 0.8], 'LineWidth',1.6);
    plot(ax, g.dM1M2*[1 1], yq+[-0.035 0.035], '-', 'Color',[0 0.2 0.8], 'LineWidth',1.6);
    text(ax, 0.5*g.dM1M2, yq+0.045, sprintf('M1-M2 = %.3f m', g.dM1M2), ...
         'FontSize',8.5, 'Color',[0 0.2 0.8], 'HorizontalAlignment','center');

    zb = max(g.vpt(3,:));
    plot(ax, zb*[1 1], 1.15*opts.r_env*[-1 1], 'r:', 'LineWidth',1.6);
    text(ax, zb, -1.18*opts.r_env, sprintf(' deepest optic %.3f m ', zb), ...
         'FontSize',8.5, 'Color','r', 'HorizontalAlignment','left', ...
         'VerticalAlignment','top');
end
