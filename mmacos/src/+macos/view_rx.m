function fig = view_rx(opts)
%MACOS.VIEW_RX  3-D visualization of the LOADED prescription: beam,
%   optics, and MET paths if present.  Works for ANY Rx -- no design-layer
%   structs required; everything is read back from the engine (the modern
%   equivalent of the old MACOS 3-D model visualizer):
%
%     beam    grey 3-D polylines = the engine DRAW command's real traced
%             ray fans (macos.draw_rays3d, both meridians), true global
%             coordinates -- correct for folded / off-axis systems
%     optics  per-element surface cross-section curves through the fan
%             crossings (drawn to the beam footprint, any surface type --
%             Segment and non-sequential elements included), labeled E<k>
%     MET     if the Rx declares metrology (nMetPos/tMetElt/metBeamFlg):
%             gauge beams launcher->fiducial via macos.met_geom, colored
%             per source element, with launcher/fiducial markers
%
%   fig = macos.view_rx() draws into a new figure and returns it.
%
%   Options:
%     'nrays'   max rays drawn per fan (default 15, subsampled)
%     'elts'    element range [first last] to draw (default all)
%     'hide'    element indices whose surface curve to omit
%     'labels'  label element surfaces (default true)
%     'met'     draw MET paths when present (default true)
%     'ax'      draw into an existing axes instead of a new figure
%     'view'    [az el] initial 3-D view (default [-35 18])
%     'title'   figure title (default: counts summary)
%     'save'    PNG path;  'visible' (default true)
%
%   Positions are global BaseUnits.  Implementation note: the harvest is
%   the DRAW fan capture, NOT per-element macos.trace(k) -- OPD refuses
%   NSRefractor/Segment/NSReflector target elements (and used to
%   infinite-loop on them in batch mode), while the DRAW trace crosses
%   every element type.
%
%   See also: macos.draw_rays3d, macos.met_geom, macos.design.met_view.

arguments
    opts.nrays   (1,1) double  = 15
    opts.elts    (1,:) double  = []
    opts.hide    (1,:) double  = []
    opts.labels  (1,1) logical = true
    opts.met     (1,1) logical = true
    opts.ax                    = []
    opts.view    (1,2) double  = [-35 18]
    opts.title   (1,:) char    = ''
    opts.save    (1,:) char    = ''
    opts.visible (1,1) logical = true
end

if ~macos.has_rx()
    error('macos:view_rx:noRx', 'no prescription is loaded in the engine');
end

nE = macos.num_elt();
k0 = 0;  k1 = nE;
if ~isempty(opts.elts)
    k0 = max(0, opts.elts(1));
    if numel(opts.elts) > 1, k1 = min(nE, opts.elts(2)); end
end

% ---- harvest: both meridian fans, true 3-D crossings -------------------
fans = {macos.draw_rays3d('YZ', k0, k1), macos.draw_rays3d('XZ', k0, k1)};

% ---- figure / axes -----------------------------------------------------
if isempty(opts.ax)
    vis = 'on';  if ~opts.visible, vis = 'off'; end
    fig = figure('Visible', vis, 'Position', [50 50 980 640]);
    ax  = axes('Parent', fig);
else
    ax  = opts.ax;  fig = ancestor(ax, 'figure');
end
hold(ax, 'on');

% ---- beam: subsampled fan polylines ------------------------------------
ndrawn = 0;
for f = 1:2
    b = fans{f};
    live = find(b.nper > 1);
    if isempty(live), continue; end
    pick = live(round(linspace(1, numel(live), min(opts.nrays, numel(live)))));
    for r = unique(pick)
        p = b.P(:, 1:b.nper(r), r);
        plot3(ax, p(1,:), p(2,:), p(3,:), '-', ...
              'Color', [0.55 0.55 0.55 0.4], 'LineWidth', 0.4);
        ndrawn = ndrawn + 1;
    end
end

% ---- optics: per-element cross-section curves through the crossings ----
for k = max(1,k0):k1
    if any(opts.hide == k), continue; end
    ctr = zeros(3,1);  npt = 0;
    for f = 1:2
        b = fans{f};
        % crossing of element k on each ray of this fan, in ray order
        Q = nan(3, b.nray);
        for r = 1:b.nray
            c = find(b.elt(1:b.nper(r), r) == k, 1);
            if ~isempty(c), Q(:, r) = b.P(:, c, r); end
        end
        m = ~isnan(Q(1,:));
        if nnz(m) < 2, continue; end
        Q = Q(:, m);
        plot3(ax, Q(1,:), Q(2,:), Q(3,:), '-', ...
              'Color', [0.2 0.3 0.5], 'LineWidth', 1.4);
        ctr = ctr + sum(Q, 2);  npt = npt + size(Q, 2);
    end
    if opts.labels && npt > 0
        c = ctr / npt;
        text(ax, c(1), c(2), c(3), sprintf('  E%d', k), 'FontSize', 8, ...
             'Color', [0.15 0.2 0.35]);
    end
end

% ---- MET paths, when the Rx declares metrology -------------------------
nbeam = 0;
if opts.met
    g = macos.met_geom();
    nbeam = g.n;
    if nbeam > 0
        cmap = lines(7);
        [ue, ~, ig] = unique(g.src_elt);
        for e = 1:numel(ue)
            m = (ig == e).';
            S = g.src_pts(:, m);  T = g.tgt_pts(:, m);
            n = size(S, 2);
            X = [S(1,:); T(1,:); nan(1,n)];
            Y = [S(2,:); T(2,:); nan(1,n)];
            Z = [S(3,:); T(3,:); nan(1,n)];
            col = cmap(mod(e-1, 7) + 1, :);
            plot3(ax, X(:), Y(:), Z(:), '-', 'Color', [col 0.5], ...
                  'LineWidth', 0.7);
        end
        plot3(ax, g.src_pts(1,:), g.src_pts(2,:), g.src_pts(3,:), 'o', ...
              'MarkerSize', 4, 'MarkerFaceColor', [0.1 0.6 0.2], ...
              'MarkerEdgeColor', 'none', 'LineStyle', 'none');
        fid = unique(g.tgt_pts.', 'rows', 'stable').';
        plot3(ax, fid(1,:), fid(2,:), fid(3,:), 's', 'MarkerSize', 7, ...
              'MarkerFaceColor', [0.85 0.15 0.15], 'MarkerEdgeColor', 'k', ...
              'LineStyle', 'none');
    end
end

axis(ax, 'equal');  grid(ax, 'on');  view(ax, opts.view);
xlabel(ax, 'X');  ylabel(ax, 'Y');  zlabel(ax, 'Z');
if isempty(opts.title)
    opts.title = sprintf('%d elements, %d rays drawn', nE, ndrawn);
    if nbeam > 0
        opts.title = sprintf('%s, %d MET beams', opts.title, nbeam);
    end
end
title(ax, opts.title, 'Interpreter', 'none');
if isempty(opts.ax), fig.Name = opts.title; end

if ~isempty(opts.save), print(fig, opts.save, '-dpng', '-r150'); end
end
