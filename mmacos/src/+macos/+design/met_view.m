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

% MET beams: segment trusses (blue) / extra-source truss (orange)
i1 = 1:min(nsrc_seg, nbeam);  i2 = (min(nsrc_seg, nbeam)+1):nbeam;
hb1 = beams_(ax3, am.src_pts(:,i1), am.tgt_pts(:,i1), [0.20 0.45 0.85]);
hb2 = beams_(ax3, am.src_pts(:,i2), am.tgt_pts(:,i2), [0.90 0.55 0.10]);

% launchers + fiducials
hl = plot3(ax3, am.src_pts(1,:), am.src_pts(2,:), am.src_pts(3,:), 'o', ...
     'MarkerSize', 4, 'MarkerFaceColor', [0.1 0.6 0.2], ...
     'MarkerEdgeColor', 'none', 'LineStyle', 'none');
hf = plot3(ax3, fid(1,:), fid(2,:), fid(3,:), 's', ...
     'MarkerSize', 7, 'MarkerFaceColor', [0.85 0.15 0.15], ...
     'MarkerEdgeColor', 'k', 'LineStyle', 'none');
if ~isempty(opts.overlay_pts)
    plot3(ax3, opts.overlay_pts(1,:), opts.overlay_pts(2,:), ...
          opts.overlay_pts(3,:), 'o', 'MarkerSize', 5, ...
          'MarkerEdgeColor', [0.1 0.6 0.2], 'LineStyle', 'none');
end

axis(ax3, 'equal'); grid(ax3, 'on'); view(ax3, [-35 18]);
xlabel(ax3, 'X'); ylabel(ax3, 'Y'); zlabel(ax3, 'Z');
hh = [hl hf];  lb = {'launchers', 'fiducials'};
if ~isempty(hb1), hh(end+1) = hb1; lb{end+1} = 'segment trusses'; end
if ~isempty(hb2), hh(end+1) = hb2; lb{end+1} = 'extra-source truss'; end
legend(ax3, hh, lb, 'Location', 'northeastoutside');   % clear of the scene
title(ax3, '3-D MET scene');

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
         'HorizontalAlignment', 'center', 'FontSize', 9, ...
         'Color', [0.3 0.3 0.35]);
    % radial centerline (the pair-symmetry axis), sized to the segment
    hw = max(vecnorm(prj(B0.poly{s}) - cs));
    dr = [cos(rad_ang(s)); sin(rad_ang(s))];
    cl = cs + dr .* [-hw, hw];
    plot(ax2, cl(1,:), cl(2,:), ':', 'Color', [0.5 0.5 0.55]);
end

Lp = prj(am.src_pts(:, i1));             % segment launchers only
plot(ax2, Lp(1,:), Lp(2,:), 'o', 'MarkerSize', 5, ...
     'MarkerFaceColor', [0.1 0.6 0.2], 'MarkerEdgeColor', 'none', ...
     'LineStyle', 'none');
Fp = prj(fid);
plot(ax2, Fp(1,:), Fp(2,:), 's', 'MarkerSize', 8, ...
     'MarkerFaceColor', [0.85 0.15 0.15], 'MarkerEdgeColor', 'k', ...
     'LineStyle', 'none');
if ~isempty(opts.overlay_pts)
    Op = prj(opts.overlay_pts);
    plot(ax2, Op(1,:), Op(2,:), 'o', 'MarkerSize', 6, ...
         'MarkerEdgeColor', [0.1 0.6 0.2], 'LineStyle', 'none');
end

axis(ax2, 'equal'); grid(ax2, 'on');
xlabel(ax2, 'tiling-plane x'); ylabel(ax2, 'tiling-plane y');
title(ax2, 'face-on: launchers / hub fiducial ring (projected)');

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
function h = beams_(ax, S, T, col)
%BEAMS_  One line object for all gauge beams S(:,k)->T(:,k) (NaN-separated).
if isempty(S), h = gobjects(0); return; end
n = size(S, 2);
X = [S(1,:); T(1,:); nan(1,n)];
Y = [S(2,:); T(2,:); nan(1,n)];
Z = [S(3,:); T(3,:); nan(1,n)];
h = plot3(ax, X(:), Y(:), Z(:), '-', 'Color', [col 0.45], 'LineWidth', 0.6);
end

