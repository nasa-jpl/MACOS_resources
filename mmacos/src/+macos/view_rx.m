function fig = view_rx(opts)
%MACOS.VIEW_RX  3-D visualization of the LOADED prescription: beam,
%   optics, and MET paths if present.  Works for ANY Rx -- no design-layer
%   structs required; everything is read back from the engine (the modern
%   equivalent of the old MACOS 3-D model visualizer):
%
%     beam    a sparse-but-FILLED ray bundle from the engine's per-trace
%             ray-position history (macos.ray_hist): rings-and-spokes
%             pattern across the full pupil by default, true global 3-D
%             polylines -- correct for folded / off-axis systems
%     optics  each optic rendered as a SOLID BODY: the true aperture
%             boundary (Circular/Elliptical/Hexagonal ApVec, Polygonal
%             PolyApVtx, lMon disc, or the ray-footprint hull), lifted
%             onto the real conic sag (KcElt/KrElt), extruded to a THIN
%             sag-following shell (thickness aperture/25), drawn in flat
%             two-tone (light optical face, dark back -- no edge
%             shading) with meridian profile curves.  CONSECUTIVE
%             REFRACTOR pairs are JOINED into one glass solid (front
%             surface + back surface + barrel).  Passive elements
%             (Reference / FocalPlane / Obscuring) draw as outline
%             frames -- they are not hardware.  Return elements (exit-
%             pupil / FP bookkeeping planes) are HIDDEN by default --
%             their declared apertures dwarf the optics ('returns'
%             restores them).  Segment elements draw with alternating
%             face tints and no per-tile profile curves, so the tiling
%             (and the center segment) reads on a segmented primary.
%     MET     if the Rx declares metrology (nMetPos/tMetElt/metBeamFlg):
%             gauge beams launcher->fiducial via macos.met_geom, colored
%             per source element, with launcher/fiducial markers
%
%   fig = macos.view_rx() draws into a new figure and returns it.
%
%   Options:
%     'bundle'  ray pattern: 'rings' (default; chief + nrings x nspokes
%               across the pupil), 'rim' (marginal ring only), 'fans'
%               (legacy dual meridian DRAW fans)
%     'nrings'  rings in the bundle (default 3);  'nspokes' (default 8)
%     'nrays'   ray budget for 'rim'/'fans' subsampling (default 25)
%     'bodies'  'solid' (default) | 'outline' (rims only) | 'patch'
%               (legacy cross-section curves)
%     'thick_frac'  shell thickness as a fraction of the element
%               aperture (default 1/25)
%     'ray_color'  RGB of the traced bundle (default green); overlay
%               several instrument paths into one 'ax' with distinct
%               colors to tell the channels apart
%     'xtra_hist'  cell of pre-harvested ray histories (macos.ray_hist
%               structs from the SAME deck re-aimed at other fields).
%               Each draws as an extra bundle AND joins the footprint
%               union, so body sizes cover every field shown
%     'xtra_color'  Nx3 RGB rows for the extra bundles (default: an
%               internal palette)
%     'show'    layer selection: 'beam' (no MET), 'beam+met' (default),
%               'met' (no traced bundle); optics always draw.  A ring
%               circles the beam at the SOURCE plane so a collimated
%               source's location is unambiguous.
%     'elts'    element range [first last] to draw (default all)
%     'hide'    element indices to omit
%     'returns' draw Return elements too (default false)
%     'labels'  label elements (default true)
%     'met'     draw MET paths when present (default true)
%     'ax'      draw into an existing axes instead of a new figure
%     'view'    [az el] initial 3-D view (default [-35 18])
%     'title'   figure title (default: counts summary)
%     'save'    PNG path;  'visible' (default true)
%
%   Positions are global BaseUnits.  Implementation notes: the beam
%   harvest is the engine ray-position history (RayPosHist via
%   macos.ray_hist -- the full traced grid, so any sparse pattern can be
%   cut from it), NOT per-element macos.trace(k) -- OPD refuses
%   NSRefractor/Segment/NSReflector target elements, while the history
%   records every element type.  The sag SIGN is calibrated per element
%   against the actual ray crossings, so no KrElt orientation convention
%   is baked in.
%
%   See also: macos.ray_hist, macos.get_elt_info, macos.draw_rays3d,
%             macos.met_geom, macos.design.met_view.

arguments
    opts.bundle  (1,:) char {mustBeMember(opts.bundle, ...
                  {'rings','rim','fans'})} = 'rings'
    opts.nrings  (1,1) double  = 3
    opts.nspokes (1,1) double  = 8
    opts.nrays   (1,1) double  = 25
    opts.bodies  (1,:) char {mustBeMember(opts.bodies, ...
                  {'solid','outline','patch'})} = 'solid'
    opts.thick_frac (1,1) double {mustBePositive} = 1/25
    opts.ray_color (1,3) double = [0.0 0.62 0.10]
    opts.xtra_hist (1,:) cell   = {}
    opts.xtra_color (:,3) double = zeros(0,3)
    opts.show    (1,:) char {mustBeMember(opts.show, ...
                  {'beam','beam+met','met'})} = 'beam+met'
    opts.elts    (1,:) double  = []
    opts.hide    (1,:) double  = []
    opts.returns (1,1) logical = false
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

% ---- harvest: full ray-position history (+ legacy fans if asked) -------
macos.ray_hist('on');
t = macos.trace();
h = macos.ray_hist(t.nRays);
macos.ray_hist('off');
% extra pre-harvested histories (other FIELDS of the same deck): they
% join the footprint union, so bodies are sized for every bundle shown
hs = [{h}, opts.xtra_hist];
fans = {};
if strcmp(opts.bundle, 'fans')
    fans = {macos.draw_rays3d('YZ', k0, k1), macos.draw_rays3d('XZ', k0, k1)};
end

% ---- figure / axes -----------------------------------------------------
if isempty(opts.ax)
    vis = 'on';  if ~opts.visible, vis = 'off'; end
    fig = figure('Visible', vis, 'Position', [50 50 980 640]);
    ax  = axes('Parent', fig);
else
    ax  = opts.ax;  fig = ancestor(ax, 'figure');
end
hold(ax, 'on');

% ---- optics ------------------------------------------------------------
% classify each element; then draw solids (with lens joins), outlines,
% or the legacy cross-section curves
SOLID_M = ["Reflector" "Segment" "NSReflector" "Grating" "RfPolarizer"];
GLASS   = ["Refractor" "NSRefractor" "TrGrating" "TrPolarizer" ...
           "CGHNullPlate" "DoeTrGrating" "LensArray" "HOE"];
kk = max(1, k0):k1;
kk = kk(~ismember(kk, opts.hide));
tile = seg_tiling_(kk);          % exact tiles for Segment elements
E = struct('k', {}, 'kind', {}, 'B', {}, 'ctr', {});
for k = kk
    g = elt_geom_(k, hs, tile);
    if isempty(g), continue; end
    if strcmp(g.type, 'Return') && ~opts.returns, continue; end
    if     any(strcmp(g.type, SOLID_M)), g.kind = 'mirror';
    elseif any(strcmp(g.type, GLASS)),   g.kind = 'glass';
    else,                                g.kind = 'passive';
    end
    E(end+1) = struct('k', k, 'kind', g.kind, 'B', g, 'ctr', g.ctr); %#ok<AGROW>
end

switch opts.bodies
    case 'patch'
        for e = 1:numel(E)
            curves_legacy_(ax, E(e).k, h);
        end
    case 'outline'
        for e = 1:numel(E)
            R = E(e).B.rim;
            plot3(ax, R(1,:), R(2,:), R(3,:), '-', ...
                  'Color', [0.2 0.3 0.5], 'LineWidth', 1.2);
        end
    case 'solid'
        e = 1;  si = 0;
        while e <= numel(E)
            if strcmp(E(e).kind, 'glass') && e < numel(E) && ...
               strcmp(E(e+1).kind, 'glass') && E(e+1).k == E(e).k + 1
                lens_(ax, E(e).B, E(e+1).B);          % joined glass solid
                e = e + 2;
            elseif strcmp(E(e).kind, 'passive')
                passive_(ax, E(e).B);
                e = e + 1;
            elseif strcmp(E(e).B.type, 'Segment')
                % alternating face tints, no per-tile profile curves:
                % 7-19 overlapping meridian-curve sets on same-color
                % tiles turned a segmented primary into spoke mush
                si = si + 1;
                if si == 1                            % Seg1 = center cell
                    dt = -0.16;                       % (carries the hole)
                else
                    tints = [0, 0.06, -0.07];
                    dt = tints(mod(si-2, 3) + 1);
                end
                plate_(ax, E(e).B, opts.thick_frac, dt, false);
                e = e + 1;
            else
                plate_(ax, E(e).B, opts.thick_frac);
                e = e + 1;
            end
        end
end
% Rx-declared obscurations (M1 hole / masks): dark disc flush on the
% optical face (+ the shell back face in solid mode), rim outlined --
% the hole reads from both sides (Dave 2026-07-18: show the M1 hole,
% or the segment carrying it, in every layout view).
for e = 1:numel(E)
    B = E(e).B;
    for i = 1:numel(B.obs2)
        C2 = B.obs2{i};
        Q  = B.lift(C2);
        nudge = B.ps * (0.004 * max(B.D, eps));
        plot3(ax, Q(1,:), Q(2,:), Q(3,:), '-', 'Color', [0.1 0.1 0.12], ...
              'LineWidth', 1.2);
        if strcmp(opts.bodies, 'solid')
            Qf = Q + nudge;                          % optical face
            fill3(ax, Qf(1,:), Qf(2,:), Qf(3,:), [0.16 0.16 0.2], ...
                  'EdgeColor', [0.1 0.1 0.12], 'FaceAlpha', 1, ...
                  'FaceLighting', 'none');
            Qb = Q - B.ps*(opts.thick_frac*B.D) - nudge;   % shell back
            fill3(ax, Qb(1,:), Qb(2,:), Qb(3,:), [0.16 0.16 0.2], ...
                  'EdgeColor', [0.1 0.1 0.12], 'FaceAlpha', 1, ...
                  'FaceLighting', 'none');
        end
    end
end
if opts.labels
    % Offset each label OFF the beam, PERPENDICULAR to the local beam and
    % IN the layout plane, so it does not print on the ray lines (the old
    % offset-along-surface-normal failed for near-normal folds, whose
    % normal points along the beam).  Alternate the side element-to-element
    % so adjacent labels do not stack; a faint leader ties label to centre.
    C = zeros(3, numel(E));
    for e = 1:numel(E), C(:,e) = E(e).ctr(:); end
    span = 0;  for e = 1:numel(E), span = max(span, E(e).B.D); end
    off = 1.1 * span;                       % label standoff (base units)
    % layout-plane normal: smallest-variance axis of the element centres
    if numel(E) >= 3
        [U3,~,~] = svd(C - mean(C,2));  npl = U3(:,3);
    else
        npl = [0;0;1];
    end
    for e = 1:numel(E)
        c = C(:,e);
        % local beam tangent from neighbouring centres
        a = C(:, max(e-1,1));  b = C(:, min(e+1,numel(E)));
        t = b - a;  if norm(t) < eps, t = E(e).B.ps(:); end
        t = t / norm(t);
        perp = cross(t, npl);               % in-plane, perp to the beam
        if norm(perp) < 1e-6, perp = E(e).B.ps(:); end
        perp = perp / norm(perp);
        s = 1 - 2*mod(e,2);                 % +1 / -1 alternating
        p = c + s*off*perp;
        plot3(ax, [c(1) p(1)], [c(2) p(2)], [c(3) p(3)], '-', ...
              'Color', [0.7 0.72 0.78], 'LineWidth', 0.5);
        text(ax, p(1), p(2), p(3), sprintf('E%d', E(e).k), ...
             'FontSize', 8, 'Color', [0.15 0.2 0.35], ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'middle');
    end
end

% ---- beam: sparse filled bundle (or legacy fans) ------------------------
ndrawn = 0;
if strcmp(opts.show, 'met')
    % beam layer off (optics + MET only)
elseif strcmp(opts.bundle, 'fans')
    for f = 1:2
        b = fans{f};
        live = find(b.nper > 1);
        if isempty(live), continue; end
        pick = live(round(linspace(1, numel(live), ...
                                   min(opts.nrays, numel(live)))));
        for r = unique(pick)
            p = b.P(:, 1:b.nper(r), r);
            plot3(ax, p(1,:), p(2,:), p(3,:), '-', ...
                  'Color', [opts.ray_color 0.8], 'LineWidth', 0.5);
            ndrawn = ndrawn + 1;
        end
    end
else
    [sel, ring3] = pick_bundle_(h, opts);
    s0 = max(1, k0 + 1);  s1 = k1 + 1;              % history slots
    for r = sel
        % connect the slots the ray actually REACHED: on segmented /
        % non-sequential systems ok is false at the OTHER segments'
        % elements (a ray visits one segment), and a lost ray's slots
        % are false from its failure onward -- both handled by keeping
        % the ok slots in order
        m = squeeze(h.ok(r, s0:s1));
        if nnz(m) < 3, continue; end
        p = squeeze(h.P(:, r, s0:s1));
        p = p(:, m);
        plot3(ax, p(1,:), p(2,:), p(3,:), '-', ...
              'Color', [opts.ray_color 0.8], 'LineWidth', 0.5);
        ndrawn = ndrawn + 1;
    end
    if k0 == 0 && ~isempty(ring3)
        % ring circling the beam AT THE SOURCE PLANE, so a collimated
        % source's location is unambiguous (Dave); a point source
        % collapses the ring to a dot
        plot3(ax, ring3(1,:), ring3(2,:), ring3(3,:), '-', ...
              'Color', 0.7*opts.ray_color, 'LineWidth', 1.4);
    end
    % extra bundles (other fields, pre-harvested via 'xtra_hist'): same
    % sparse pattern, their own colors
    pal = [0.15 0.45 0.80; 0.85 0.50 0.10; 0.55 0.25 0.75; 0.80 0.15 0.35];
    for ih = 1:numel(opts.xtra_hist)
        hx = opts.xtra_hist{ih};
        if ih <= size(opts.xtra_color, 1), cx = opts.xtra_color(ih,:);
        else, cx = pal(mod(ih-1, 4) + 1, :);
        end
        selx = pick_bundle_(hx, opts);
        for r = selx
            m = squeeze(hx.ok(r, s0:s1));
            if nnz(m) < 3, continue; end
            p = squeeze(hx.P(:, r, s0:s1));
            p = p(:, m);
            plot3(ax, p(1,:), p(2,:), p(3,:), '-', ...
                  'Color', [cx 0.8], 'LineWidth', 0.5);
            ndrawn = ndrawn + 1;
        end
    end
end

% ---- MET paths, when the Rx declares metrology -------------------------
nbeam = 0;
if opts.met && ~strcmp(opts.show, 'beam')
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

% ===========================================================================
function g = elt_geom_(k, hs, tile)
%ELT_GEOM_  Per-element drawing geometry from engine truth.
%   Frame (vpt/psi/xa/ya), aperture boundary in the aperture plane (2 x M,
%   relative to vpt), the sag-lifted rim polyline (3 x M+1), the sag
%   function with its sign CALIBRATED against the actual ray crossings,
%   and a label center.  Returns [] when the element has no usable
%   boundary (no aperture, no lMon, no ray hits).  HS is a cell of ray
%   histories (the loaded trace + any 'xtra_hist' fields); the footprint
%   is their UNION, so bodies cover every bundle drawn.
info = macos.get_elt_info(k);
vp = mmacos('elt_vpt', double(k), zeros(3,1), 0, 1);
ps = mmacos('elt_psi', double(k), zeros(3,1), 0, 1);
if norm(ps) == 0, ps = [0;0;1]; else, ps = ps/norm(ps); end
xa = info.x_obs;
if norm(xa) < 1e-9, [~, i0] = min(abs(ps)); xa = zeros(3,1); xa(i0) = 1; end
xa = xa - dot(xa, ps)*ps;  xa = xa / norm(xa);
ya = cross(ps, xa);

% ray crossings at this element, UNION over histories (footprint +
% sag-sign calibration)
Q = zeros(3,0);
for ih = 1:numel(hs)
    hq = hs{ih};
    m = hq.ok(:, k+1);
    Qi = squeeze(hq.P(:, m, k+1));
    if ~isempty(Qi), Q = [Q, Qi]; end %#ok<AGROW>
end
U = [xa.'; ya.'] * (Q - vp);

nb = 48;
th = 2*pi*(0:nb-1)/nb;
switch info.ap_type
    case 1                                          % Circular
        B2 = info.ap_vec(1) * [cos(th); sin(th)];
    case 2                                          % Elliptical
        B2 = [info.ap_vec(1)*cos(th); info.ap_vec(2)*sin(th)];
    case 6                                          % Hexagonal (apothem)
        phic = pi/6 + (0:5)*pi/3;
        B2 = (info.ap_vec(1)/cos(pi/6)) * [cos(phic); sin(phic)];
    case {7, 8}                                     % Polygonal
        B2 = info.ap_vec(1:2) + info.poly;
    otherwise                                       % None: tile / footprint / lMon
        B2 = [];
        if strcmp(info.type, 'Segment') && strcmp(tile.kind, 'hex')
            % exact hex tile from the engine tiling truth (src_seg_get:
            % width = flat-to-flat, ONE global clocking): apothem w/2
            % about the segment's own center -- adjacent tiles are
            % separated by exactly the tiling gap
            rp = mmacos('elt_rpt', double(k), zeros(3,1), 0, 1);
            phic = tile.flat_ang + pi/6 + (0:5)*pi/3;
            C3 = rp + (tile.a/cos(pi/6)) * ...
                 (tile.xa*cos(phic) + tile.ya*sin(phic));
            B2 = [xa.'; ya.'] * (C3 - vp);
        elseif strcmp(info.type, 'Segment') && strcmp(tile.kind, 'pie') ...
               && any(tile.ks == k)
            % exact pie tile: center hexagon / wedge with chord inner
            % edge (physical, non-overlapping -- see seg_tiling_)
            B2 = [xa.'; ya.'] * (pie_tile_(tile, k) - vp);
        elseif size(U, 2) >= 3
            B2 = smooth_hull_(U);
        elseif info.lmon > 0
            B2 = info.lmon * [cos(th); sin(th)];
        end
end
if isempty(B2) || size(B2, 2) < 3, g = []; return; end
% guard absurd declared sizes (unset 1e22-style) with the footprint
if size(U, 2) >= 3 && max(vecnorm(B2)) > 50*max(1, max(vecnorm(U)))
    B2 = smooth_hull_(U);
end

% conic sag, sign calibrated against the crossings
kr = macos.get_elt_kr(k);
kc = macos.get_elt_kc(k);
c = 0;
if isfinite(kr) && abs(kr) > 0 && abs(kr) < 1e15, c = 1/kr; end
sag = @(r2) (c*r2) ./ (1 + sqrt(max(1 - (1+kc)*c^2*r2, 0)));
sgn = 1;
if c ~= 0 && size(Q, 2) >= 3
    w  = ps.' * (Q - vp);                           % actual normal offsets
    sp = sag(sum(U.^2, 1));
    if dot(w, sp) < 0, sgn = -1; end
end
sagf = @(r2) sgn * sag(r2);

lift = @(P2) vp + xa*P2(1,:) + ya*P2(2,:) + ps*sagf(sum(P2.^2, 1));
rim = lift(B2(:, [1:end 1]));

% Rx-declared circular obscurations (a perforated primary's central
% hole, a coronagraph mask...): ObsVec = (r, xc, yc) in the element's
% xObs frame == the (xa, ya) basis used here.  Closed 2-D polylines.
obs2 = {};
ob = macos.get_elt_obs(k);
for i = 1:numel(ob.type)
    if ob.type(i) == 1 && ob.vec(1, i) > 0
        obs2{end+1} = ob.vec(2:3, i) + ...
            ob.vec(1, i) * [cos(th([1:end 1])); sin(th([1:end 1]))]; %#ok<AGROW>
    end
end

g = struct('type', info.type, 'vp', vp, 'ps', ps, 'xa', xa, 'ya', ya, ...
           'B2', B2, 'rim', rim, 'lift', lift, 'sagf', sagf, ...
           'D', max(vecnorm(B2 - mean(B2, 2)))*2, 'ctr', mean(rim, 2), ...
           'obs2', {obs2}, 'kind', '');
end

% ---------------------------------------------------------------------------
function [V, Fq, ring] = surf_mesh_(g, nr)
%SURF_MESH_  Sag-lifted surface disc: boundary-shaped rings scaled toward
%   the centroid + a center vertex.  Returns vertices (3 x nv), quad/tri
%   faces (cell of index rows), and the boundary ring's vertex indices.
B2 = g.B2;  c2 = mean(B2, 2);
M = size(B2, 2);
sc = linspace(1, 0.18, nr);
V = zeros(3, 0);
for j = 1:nr
    V = [V, g.lift(c2 + sc(j)*(B2 - c2))]; %#ok<AGROW>
end
V = [V, g.lift(c2)];                                % center vertex
icen = size(V, 2);
Fq = {};
for j = 1:nr-1
    a = (j-1)*M;  b = j*M;
    for q = 1:M
        q2 = mod(q, M) + 1;
        Fq{end+1} = [a+q, a+q2, b+q2, b+q]; %#ok<AGROW>
    end
end
a = (nr-1)*M;
for q = 1:M
    q2 = mod(q, M) + 1;
    Fq{end+1} = [a+q, a+q2, icen]; %#ok<AGROW>
end
ring = 1:M;
end

function draw_solid_(ax, V, F, col, alpha)
% triangulate mixed quad/tri faces and draw one FLAT-shaded patch
% (LightTools look: no lighting gradient / edge shading -- the body
% reads by silhouette, tone and the profile curves)
T = zeros(0, 3);
for i = 1:numel(F)
    f = F{i};
    if numel(f) == 3, T(end+1, :) = f; %#ok<AGROW>
    else
        T(end+1, :) = f([1 2 3]); %#ok<AGROW>
        T(end+1, :) = f([1 3 4]); %#ok<AGROW>
    end
end
patch(ax, 'Vertices', V.', 'Faces', T, 'FaceColor', col, ...
      'EdgeColor', 'none', 'FaceAlpha', alpha, 'FaceLighting', 'none');
end

function profiles_(ax, g, offs)
%PROFILES_  Meridian profile curves -- the classic cross-section cue
%   that makes the figure (concave dish / convex dome / flat) read at
%   any camera angle.  OFFS = cell of 3x1 offsets (draw the curves on
%   each face of a shell, so one is visible from every side).
if nargin < 3, offs = {zeros(3,1)}; end
c2 = mean(g.B2, 2);
for dirv = {[1;0], [0;1]}
    d = dirv{1};
    pr = d.' * (g.B2 - c2);
    tt = linspace(min(pr), max(pr), 33);
    P0 = g.lift(c2 + d*tt);
    for o = offs
        P = P0 + o{1};
        plot3(ax, P(1,:), P(2,:), P(3,:), '-', ...
              'Color', [0.3 0.34 0.42], 'LineWidth', 1.1);
    end
end
end

function plate_(ax, g, tf, dtint, do_prof)
%PLATE_  Mirror solid: a constant-thickness SHELL that follows the sag
%   (back = the same surface offset along -psi), so the optic's actual
%   figure -- concave dish, convex dome, flat -- reads directly (Dave:
%   the body must be tight to the optical surface, not a cylinder).
%   DTINT shifts the optical-face brightness (segment tiling contrast);
%   DO_PROF=false suppresses the meridian profile curves.
if nargin < 4, dtint = 0; end
if nargin < 5, do_prof = true; end
[V, F, ring] = surf_mesh_(g, 8);
tt = tf * g.D;
nV = size(V, 2);
V = [V, V - g.ps*tt];                                % shell back
Fb = {};
for i = 1:numel(F)
    Fb{end+1} = F{i} + nV; %#ok<AGROW>
end
M = numel(ring);
for q = 1:M                                          % rim wall
    q2 = mod(q, M) + 1;
    Fb{end+1} = [ring(q), ring(q2), nV+ring(q2), nV+ring(q)]; %#ok<AGROW>
end
fcol = min(max([0.72 0.74 0.92] + dtint, 0), 1);
draw_solid_(ax, V, F,  fcol, 1.0);                   % optical face: light
draw_solid_(ax, V, Fb, [0.42 0.44 0.50], 1.0);       % back + wall: dark
plot3(ax, g.rim(1,:), g.rim(2,:), g.rim(3,:), '-', ...
      'Color', [0.25 0.3 0.4], 'LineWidth', 0.8);
Rb = g.rim - g.ps*tt;                                % back rim
plot3(ax, Rb(1,:), Rb(2,:), Rb(3,:), '-', ...
      'Color', [0.25 0.3 0.4], 'LineWidth', 0.5);
if do_prof
    profiles_(ax, g, {zeros(3,1), -g.ps*tt});        % both faces
end
end

function lens_(ax, g1, g2)
%LENS_  Joined glass solid: front surface at g1, back surface at g2,
%   barrel wall between their rims (Dave: join consecutive refractors).
[V1, F1, r1] = surf_mesh_(g1, 5);
[V2, F2, r2] = surf_mesh_(g2, 5);
n1 = size(V1, 2);
V = [V1, V2];
F = F1;
for i = 1:numel(F2), F{end+1} = F2{i} + n1; end %#ok<AGROW>
M = numel(r1);                                       % same nb by construction
for q = 1:M
    q2 = mod(q, M) + 1;
    F{end+1} = [r1(q), r1(q2), n1+r2(q2), n1+r2(q)]; %#ok<AGROW>
end
draw_solid_(ax, V, F, [0.55 0.75 0.95], 0.35);
for g = {g1, g2}
    R = g{1}.rim;
    plot3(ax, R(1,:), R(2,:), R(3,:), '-', ...
          'Color', [0.25 0.45 0.65], 'LineWidth', 0.8);
    profiles_(ax, g{1});
end
end

function passive_(ax, g)
%PASSIVE_  Reference / Return / FocalPlane / Obscuring: outline frame only.
R = g.rim;
st = '--';  col = [0.55 0.55 0.6];
if strcmp(g.type, 'FocalPlane'), st = '-'; col = [0.15 0.15 0.2]; end
if strcmp(g.type, 'Obscuring')
    st = '-'; col = [0.4 0.25 0.25];
    fill3(ax, R(1,:), R(2,:), R(3,:), [0.45 0.3 0.3], ...
          'FaceAlpha', 0.25, 'EdgeColor', 'none');
end
plot3(ax, R(1,:), R(2,:), R(3,:), st, 'Color', col, 'LineWidth', 1.0);
end

function curves_legacy_(ax, k, h)
%CURVES_LEGACY_  The pre-solid look: crossing curves from the history.
m = h.ok(:, k+1);
Q = squeeze(h.P(:, m, k+1));
if size(Q, 2) < 2, return; end
plot3(ax, Q(1,:), Q(2,:), Q(3,:), '.', 'Color', [0.2 0.3 0.5], ...
      'MarkerSize', 3);
end

% ---------------------------------------------------------------------------
function [sel, ring3] = pick_bundle_(h, opts)
%PICK_BUNDLE_  Sparse-but-filled ray selection from the source plane.
P0 = squeeze(h.P(:, :, 1));
ok0 = h.ok(:, 1).';
c = mean(P0(:, ok0), 2);
A = P0(:, ok0) - c;
[Ub, ~, ~] = svd(A, 'econ');
uv = Ub(:, 1:2).' * (P0 - c);                        % source-plane coords
r  = vecnorm(uv);
thr = atan2(uv(2,:), uv(1,:));
rmax = max(r(ok0));
live = find(ok0);
sel = zeros(1, 0);
    function grab(rt, at)
        % nearest live ray to (radius rt, azimuth at)
        d2 = (r(live) - rt).^2 + (rt * wrap_(thr(live) - at)).^2;
        [~, i] = min(d2);
        sel(end+1) = live(i);
    end
switch opts.bundle
    case 'rim'
        na = max(6, opts.nrays);
        for a = 2*pi*(0:na-1)/na, grab(0.98*rmax, a); end
    otherwise                                        % 'rings'
        [~, i0] = min(r(live));                      % chief
        sel = live(i0);
        for j = 1:opts.nrings
            rt = j/opts.nrings * 0.98 * rmax;
            st = pi/opts.nspokes * mod(j, 2);        % stagger rings
            for a = st + 2*pi*(0:opts.nspokes-1)/opts.nspokes
                grab(rt, a);
            end
        end
end
sel = unique(sel, 'stable');
thr2 = linspace(0, 2*pi, 91);
ring3 = c + rmax*(Ub(:,1)*cos(thr2) + Ub(:,2)*sin(thr2));
end

function w = wrap_(a)
w = mod(a + pi, 2*pi) - pi;
end

function B2 = smooth_hull_(U)
%SMOOTH_HULL_  Ray-footprint boundary: convex hull, grown 8%, resampled
%   by arc length and lightly smoothed so the body reads as the optic's
%   smooth outline instead of a kinked sampling polygon.
Kh = convhull(U(1,:), U(2,:));
H = U(:, Kh(1:end-1));
c2 = mean(H, 2);
H = c2 + 1.08*(H - c2);
Hc = H(:, [1:end 1]);
d = [0, cumsum(vecnorm(diff(Hc, 1, 2)))];
tq = linspace(0, d(end), 49);  tq(end) = [];
B2 = [interp1(d, Hc(1,:), tq); interp1(d, Hc(2,:), tq)];
w = 5;                                               % closed moving average
kern = ones(1, w)/w;
for i = 1:2
    x = [B2(i, end-w+1:end), B2(i,:), B2(i, 1:w)];
    x = conv(x, kern, 'same');
    B2(i,:) = x(w+1:end-w);
end
end

function tile = seg_tiling_(kk)
%SEG_TILING_  Engine segmentation-tiling truth for exact Segment tiles.
%   src_seg_get gives GridType/nSeg/width/gap.  Hex: one global tiling
%   clocking from the consensus of nearest-neighbor segment-center
%   directions (mod 60 deg).  Pie: center segment = HEXAGON at the
%   physical (width-gap)/2, ring wedges = outer arc + straight CHORD
%   inner edge facing the center hexagon's flat -- the same physical
%   model as macos.design.seg_boundary (tiles must not overlap and the
%   gaps must read; Dave).
tile = struct('kind', 'none');
[gid, nsg, w, gp] = mmacos('src_seg_get');
if nsg < 2 || w <= 0 || ~any(gid == [3 4]), return; end
ks = zeros(1, 0);  Cs = zeros(3, 0);
for k = kk
    ii = macos.get_elt_info(k);
    if strcmp(ii.type, 'Segment')
        ks(end+1) = k; %#ok<AGROW>
        Cs(:, end+1) = mmacos('elt_rpt', double(k), zeros(3,1), 0, 1); %#ok<AGROW>
    end
end
if numel(ks) < 2, return; end
ps = mmacos('elt_psi', double(ks(1)), zeros(3,1), 0, 1);
if norm(ps) == 0, return; end
ps = ps/norm(ps);
[~, i0] = min(abs(ps));  xa = zeros(3,1);  xa(i0) = 1;
xa = xa - dot(xa, ps)*ps;  xa = xa/norm(xa);  ya = cross(ps, xa);
c0 = mean(Cs, 2);
C2 = [xa, ya].' * (Cs - c0);
if gid == 3                                          % Hex
    n = size(C2, 2);
    D = squeeze(vecnorm(reshape(C2, 2, 1, n) - reshape(C2, 2, n, 1)));
    D(1:n+1:end) = inf;
    [iq, jq] = find(D < 1.05*min(D(:)));
    angs = zeros(1, numel(iq));
    for q = 1:numel(iq)
        dv = C2(:, jq(q)) - C2(:, iq(q));
        angs(q) = mod(atan2(dv(2), dv(1)), pi/3);
    end
    tile = struct('kind', 'hex', 'a', w/2, ...
                  'flat_ang', angle(mean(exp(1i*6*angs)))/6, ...
                  'ps', ps, 'xa', xa, 'ya', ya);
else                                                 % Pie
    % ring classification with the width-scaled tolerance (shared
    % helper -- micron radius scatter on figured parents must not
    % split a ring)
    R = macos.design.pie_rings(C2, w);
    az = atan2(C2(2, R.iring == 1), C2(1, R.iring == 1));
    tile = struct('kind', 'pie', 'w', w, 'g', gp, 'ks', ks, ...
                  'C2', C2, 'c0', c0, 'rc', R.rc, 'isctr', R.isctr, ...
                  'rings', R.rings, 'iring', R.iring, 'nmem', R.nmem, ...
                  'flat_ang', angle(mean(exp(1i*6*az)))/6, ...
                  'ps', ps, 'xa', xa, 'ya', ya);
end
end

function P3 = pie_tile_(tile, k)
%PIE_TILE_  Global boundary polyline of pie Segment element k (open).
i = find(tile.ks == k, 1);
w = tile.w;  g = tile.g;
lift = @(P2) tile.c0 + tile.xa*P2(1,:) + tile.ya*P2(2,:);
if tile.isctr(i)
    % center cell: hexagon at the physical (w-g)/2, flats facing ring 1
    phic = tile.flat_ang + pi/6 + (0:5)*pi/3;
    P3 = lift(((w-g)/2/cos(pi/6)) * [cos(phic); sin(phic)]);
    return
end
rc = tile.rc(i);
a0 = atan2(tile.C2(2,i), tile.C2(1,i));
has_outer  = tile.iring(i) < numel(tile.rings);
inner_ring = tile.iring(i) == 1;
% side edges parallel to the sector boundary rays (uniform-width gap
% slots -- shared geometry with seg_boundary/seg_apertures)
W = macos.design.pie_wedge_geom(a0, 2*pi/tile.nmem(tile.iring(i)), ...
        rc, w, g, 0, has_outer, inner_ring && any(tile.isctr));
tho = linspace(W.th1, W.th2, 25);
if inner_ring && any(tile.isctr)
    % straight chord facing the center hexagon's flat
    P2 = [W.A, W.ro*[cos(tho); sin(tho)], W.B];
else
    thi = linspace(W.ti2, W.ti1, 25);
    P2 = [W.ro*[cos(tho); sin(tho)], W.ri*[cos(thi); sin(thi)]];
end
P3 = lift(P2);
end
