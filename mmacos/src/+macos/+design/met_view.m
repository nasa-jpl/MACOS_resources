function fig = met_view(seg, am, opts)
%MET_VIEW  3-D visualization of a segmented-primary laser-MET setup.
%   fig = macos.design.met_view(SEG, AM) draws the MET configuration built
%   by macos.design.add_met on the segmentation from macos.design.segment_rx:
%
%     LEFT  - 3-D scene: segment hex tiles (face triads), launchers (green),
%             hub fiducials (red), one line per MET gauge beam
%             (launcher->fiducial; segment trusses blue, extra-source truss
%             e.g. the aft-body/M3 ring in orange), the hub disc (fitted
%             from the fiducial ring), and -- when an Rx is loaded in the
%             engine ('rays', default auto) -- a grey real-ray envelope from
%             the actual trace (macos.trace(k) + macos.get_ray_info), the
%             modern equivalent of the old MACOS 3-D model visualizer.
%     RIGHT - face-on view of the primary (tiling plane): hex boundaries,
%             segment labels, launcher positions, and the hub fiducial ring
%             projected along the parent axis.  This is the view that shows
%             the LAYOUT: launcher clearance off the segment edge, pair
%             symmetry about each segment's radial centerline, fiducial
%             clocking.
%
%   SEG: segment_rx output (uses .frames: rpt/xhat/yhat/zhat/lmon, .nseg)
%   AM:  add_met output   (uses .src_pts/.tgt_pts, 3 x n_beams, BaseUnits)
%
%   Options:
%     'rays'        logical; overlay the traced ray envelope (default: true
%                   when an Rx is loaded in the engine, else false).  The
%                   loaded Rx must be the segmented+met system.
%     'nrays'       rays in the envelope (default 24, subsampled)
%     'overlay_pts' 3 x N extra launcher set drawn as open circles in BOTH
%                   panels (e.g. the pre-optimization ring, for comparison)
%     'edge_off'    launcher edge clearance to annotate on the face-on
%                   panel (mm; draws the offset hex dashed; default [] = off)
%     'title'       figure title (default auto from counts)
%     'save'        PNG path;  'visible' (default true)
%
%   Example (after e5_seg.m):
%     macos.design.met_view(seg, am, 'save', 'e5_seg_met_layout.png');
%
%   See also: macos.design.add_met, macos.design.segment_rx,
%             macos.design.dmet_dx, macos.draw_rays.

arguments
    seg (1,1) struct
    am  (1,1) struct
    opts.rays        double  = -1        % -1 = auto (engine has an Rx)
    opts.nrays       (1,1) double = 24
    opts.overlay_pts double  = []
    opts.sensor_pts  double  = []        % 3 x N edge-sensor points drawn
                                         % as small gray DOTS (the SMM Hx
                                         % shared-edge sensors; Dave
                                         % 2026-07-19 -- distinct from the
                                         % open-circle launcher overlay)
    opts.edge_off    double  = []
    opts.title       (1,:) char = ''
    opts.save        (1,:) char = ''
    opts.visible     (1,1) logical = true
end

nseg  = seg.nseg;
nbeam = size(am.src_pts, 2);
nsrc_seg = 6*nseg;                       % add_met: 6 launchers per segment
                                         % first, extra-source beams after
% fiducials = unique target points
fid = unique(am.tgt_pts.', 'rows', 'stable').';
nf  = size(fid, 2);

% hub disc from the fiducial ring: centroid + best-fit plane normal
fc = mean(fid, 2);
[~, ~, Vf] = svd((fid - fc).', 0);
fn = Vf(:, 3);                           % ring-plane normal
fr = max(vecnorm(fid - fc));             % ring radius

% boundary-true segment geometry (hex tiles OR pie wedges; manual 4.x:
% width = flat-to-flat / radial band, ONE global tiling orientation --
% NOT each segment's face-frame xhat)
B0 = macos.design.seg_boundary(seg);
uh = B0.u;  vh = B0.v;  c0 = B0.c0;
Boff = [];
if ~isempty(opts.edge_off)
    Boff = macos.design.seg_boundary(seg, opts.edge_off);
end

% per-segment radial centerline angle in the TILING plane (symmetry axis)
C2 = [uh.'; vh.'] * ([seg.frames.rpt] - c0);
rad_ang = atan2(C2(2,:), C2(1,:));
rad_ang(vecnorm(C2) < 1e-6) = 0;         % center segment

vis = 'on';  if ~opts.visible, vis = 'off'; end
fig = figure('Visible', vis, 'Position', [40 40 1280 620]);
tl  = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

%% ---------------- left: 3-D scene ----------------------------------
ax3 = nexttile(tl);  hold(ax3, 'on');

% optional real-ray + optics underlay from the general viewer (engine
% DRAW fan capture -- NOT per-element trace(k), which infinite-loops on
% Segment elements in batch mode).  MET layer off: drawn below with the
% truss coloring.
want_rays = opts.rays > 0 || (opts.rays < 0 && macos.has_rx());
if want_rays && macos.has_rx()
    macos.view_rx('ax', ax3, 'met', false, 'labels', false, ...
                  'nrays', opts.nrays);
end

% segment tiles (boundary-true: hex or pie)
for s = 1:nseg
    P = B0.poly{s};
    patch(ax3, P(1,:), P(2,:), P(3,:), [0.55 0.55 0.62], ...
          'FaceAlpha', 0.35, 'EdgeColor', [0.25 0.25 0.3], 'LineWidth', 0.8);
end

% hub disc at its REAL physical extent when known (add_met .hub_rad /
% .hub_pv/.hub_ps) so fiducials floating past the mirror rim are
% VISIBLE as a mounting problem; fallback = fit past the fiducial ring.
th = linspace(0, 2*pi, 72);
if isfield(am, 'hub_rad') && isfinite(am.hub_rad)
    hn = am.hub_ps;  hc = am.hub_pv;  rd = am.hub_rad;
else
    hn = fn;  hc = fc;  rd = 1.35*fr;
end
e0 = null(hn.'); ux = e0(:,1); uy = e0(:,2);
D  = hc + rd*(ux*cos(th) + uy*sin(th));
patch(ax3, D(1,:), D(2,:), D(3,:), [0.75 0.7 0.55], ...
      'FaceAlpha', 0.25, 'EdgeColor', [0.5 0.45 0.3]);

% MET beams + launchers, COLORED PER OWNING SEGMENT (Dave 2026-07-16:
% the association must be readable); extra-source truss orange.
segcol = seg_colors_(nseg);
i2 = (min(nsrc_seg, nbeam)+1):nbeam;
hb1 = gobjects(0);
for s = 1:nseg
    js = (s-1)*6 + (1:6);  js = js(js <= nbeam);
    h = beams_(ax3, am.src_pts(:,js), am.tgt_pts(:,js), segcol(s,:));
    plot3(ax3, am.src_pts(1,js), am.src_pts(2,js), am.src_pts(3,js), ...
          'o', 'MarkerSize', 4.5, 'MarkerFaceColor', segcol(s,:), ...
          'MarkerEdgeColor', 'none', 'LineStyle', 'none');
    if isempty(hb1), hb1 = h; end
end
hb2 = beams_(ax3, am.src_pts(:,i2), am.tgt_pts(:,i2), [0.90 0.55 0.10]);
if ~isempty(i2)
    plot3(ax3, am.src_pts(1,i2), am.src_pts(2,i2), am.src_pts(3,i2), ...
          'o', 'MarkerSize', 4.5, 'MarkerFaceColor', [0.90 0.55 0.10], ...
          'MarkerEdgeColor', 'none', 'LineStyle', 'none');
end
hf = plot3(ax3, fid(1,:), fid(2,:), fid(3,:), 's', ...
     'MarkerSize', 7, 'MarkerFaceColor', [0.85 0.15 0.15], ...
     'MarkerEdgeColor', 'k', 'LineStyle', 'none');
if ~isempty(opts.overlay_pts)
    plot3(ax3, opts.overlay_pts(1,:), opts.overlay_pts(2,:), ...
          opts.overlay_pts(3,:), 'o', 'MarkerSize', 5, ...
          'MarkerEdgeColor', [0.4 0.4 0.4], 'LineStyle', 'none');
end
if ~isempty(opts.sensor_pts)
    plot3(ax3, opts.sensor_pts(1,:), opts.sensor_pts(2,:), ...
          opts.sensor_pts(3,:), 'o', 'MarkerSize', 4, ...
          'MarkerFaceColor', [0.45 0.45 0.45], ...
          'MarkerEdgeColor', 'none', 'LineStyle', 'none');
end

axis(ax3, 'equal'); grid(ax3, 'on'); view(ax3, [-35 18]);
xlabel(ax3, 'X'); ylabel(ax3, 'Y'); zlabel(ax3, 'Z');
hh = hf;  lb = {'fiducials'};
if ~isempty(hb1), hh(end+1) = hb1; lb{end+1} = 'segment trusses (color = segment)'; end
if ~isempty(hb2), hh(end+1) = hb2; lb{end+1} = 'extra-source truss'; end
lg = legend(ax3, hh, lb, 'Location', 'northeastoutside'); % clear of the scene
title(ax3, '3-D MET scene');

% small inset below the legend: the M2-M3 MET, face-on in the HUB plane
% (M2 disc at its real radius, rim fiducials, the extra/M3 launcher
% ring + its gauge beams projected along the hub normal) -- makes the
% aft-truss geometry readable at a glance (Dave 2026-07-16).
drawnow;                                  % realize the legend position
lp = lg.Position;
iw = max(lp(3), 0.11);
axi = axes(fig, 'Position', ...
    [lp(1), max(lp(2) - iw - 0.09, 0.05), iw, iw]);
hold(axi, 'on');
pj  = @(P) [ux.'; uy.'] * (P - hc);       % hub-plane coords (disc basis)
thc = linspace(0, 2*pi, 72);
plot(axi, rd*cos(thc), rd*sin(thc), '-', 'Color', [0.5 0.45 0.3]);
if ~isempty(i2)
    Si = pj(am.src_pts(:, i2));
    Ti = pj(am.tgt_pts(:, i2));
    nb2 = size(Si, 2);
    Xb = [Si(1,:); Ti(1,:); nan(1,nb2)];
    Yb = [Si(2,:); Ti(2,:); nan(1,nb2)];
    plot(axi, Xb(:), Yb(:), '-', 'Color', [0.90 0.55 0.10 0.5], ...
         'LineWidth', 0.6);
    plot(axi, Si(1,:), Si(2,:), 'o', 'MarkerSize', 4, ...
         'MarkerFaceColor', [0.90 0.55 0.10], 'MarkerEdgeColor', 'none', ...
         'LineStyle', 'none');
end
Fi = pj(fid);
plot(axi, Fi(1,:), Fi(2,:), 's', 'MarkerSize', 6, ...
     'MarkerFaceColor', [0.85 0.15 0.15], 'MarkerEdgeColor', 'k', ...
     'LineStyle', 'none');
axis(axi, 'equal');  box(axi, 'on');
axi.FontSize = 6;
xlabel(axi, 'hub-plane x, mm', 'FontSize', 7);
ylabel(axi, 'hub-plane y, mm', 'FontSize', 7);
title(axi, 'M2-M3 MET face-on', 'FontSize', 8);

%% ---------------- right: face-on primary layout --------------------
ax2 = nexttile(tl);  hold(ax2, 'on');
prj = @(P) [uh.'; vh.'] * (P - c0);      % tiling-plane coords

for s = 1:nseg
    P = prj(B0.poly{s});
    plot(ax2, P(1,:), P(2,:), '-', ...
         'Color', [0.25 0.25 0.3], 'LineWidth', 0.9);
    if ~isempty(Boff)
        Q = prj(Boff.poly{s});
        plot(ax2, Q(1,:), Q(2,:), '--', ...
             'Color', [0.6 0.6 0.65], 'LineWidth', 0.5);
    end
    cs = C2(:, s);
    text(ax2, cs(1), cs(2), sprintf('S%d', s), ...
         'HorizontalAlignment', 'center', 'FontSize', 10, ...
         'FontWeight', 'bold', 'Color', segcol(s,:));
    % radial centerline (the pair-symmetry axis), sized to the segment
    hw = max(vecnorm(prj(B0.poly{s}) - cs));
    dr = [cos(rad_ang(s)); sin(rad_ang(s))];
    cl = cs + dr .* [-hw, hw];
    plot(ax2, cl(1,:), cl(2,:), ':', 'Color', [0.5 0.5 0.55]);
end

% projected MET beams launcher->fiducial + launchers, colored per
% owning segment (label text matches), fiducials red squares
Fp = prj(fid);
for s = 1:nseg
    js = (s-1)*6 + (1:6);  js = js(js <= nbeam);
    Sp = prj(am.src_pts(:, js));
    Tp = prj(am.tgt_pts(:, js));
    n = size(Sp, 2);
    Xb = [Sp(1,:); Tp(1,:); nan(1,n)];
    Yb = [Sp(2,:); Tp(2,:); nan(1,n)];
    plot(ax2, Xb(:), Yb(:), '-', 'Color', [segcol(s,:) 0.35], ...
         'LineWidth', 0.6);
    plot(ax2, Sp(1,:), Sp(2,:), 'o', 'MarkerSize', 5.5, ...
         'MarkerFaceColor', segcol(s,:), 'MarkerEdgeColor', 'none', ...
         'LineStyle', 'none');
end
plot(ax2, Fp(1,:), Fp(2,:), 's', 'MarkerSize', 8, ...
     'MarkerFaceColor', [0.85 0.15 0.15], 'MarkerEdgeColor', 'k', ...
     'LineStyle', 'none');
if ~isempty(opts.overlay_pts)
    Op = prj(opts.overlay_pts);
    plot(ax2, Op(1,:), Op(2,:), 'o', 'MarkerSize', 6, ...
         'MarkerEdgeColor', [0.4 0.4 0.4], 'LineStyle', 'none');
end
if ~isempty(opts.sensor_pts)
    Sp = prj(opts.sensor_pts);
    plot(ax2, Sp(1,:), Sp(2,:), 'o', 'MarkerSize', 4, ...
         'MarkerFaceColor', [0.45 0.45 0.45], ...
         'MarkerEdgeColor', 'none', 'LineStyle', 'none');
end

axis(ax2, 'equal'); grid(ax2, 'on');
xlabel(ax2, 'tiling-plane x'); ylabel(ax2, 'tiling-plane y');
title(ax2, 'face-on: beams + launchers (color = segment), fiducials red');

if isempty(opts.title)
    opts.title = sprintf('MET setup: %d segments, %d launchers, %d fiducials, %d gauge beams', ...
        nseg, size(am.src_pts, 2), nf, nbeam);
end
sgtitle(tl, opts.title, 'Interpreter', 'none');
fig.Name = opts.title;                   % also queryable (findall cannot
                                         % reach the sgtitle layout Text)

if ~isempty(opts.save), print(fig, opts.save, '-dpng', '-r150'); end
end

% ---------------------------------------------------------------------------
function C = seg_colors_(n)
%SEG_COLORS_  n distinguishable segment colors (stable across panels).
base = lines(7);
C = base(mod(0:n-1, 7) + 1, :);
if n > 7                                  % 2-ring+: darken repeats
    k = ceil((8:n)/7) - 1;
    C(8:n, :) = max(C(8:n, :) - 0.25*k(:), 0);
end
end

function h = beams_(ax, S, T, col)
%BEAMS_  One line object for all gauge beams S(:,k)->T(:,k) (NaN-separated).
if isempty(S), h = gobjects(0); return; end
n = size(S, 2);
X = [S(1,:); T(1,:); nan(1,n)];
Y = [S(2,:); T(2,:); nan(1,n)];
Z = [S(3,:); T(3,:); nan(1,n)];
h = plot3(ax, X(:), Y(:), Z(:), '-', 'Color', [col 0.45], 'LineWidth', 0.6);
end

