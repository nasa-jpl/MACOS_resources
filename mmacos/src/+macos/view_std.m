function fig = view_std(opts)
%MACOS.VIEW_STD  Standard 3-view layout figure for the LOADED prescription.
%   fig = macos.view_std() draws four beam-aligned panels via
%   macos.view_rx -- FRONT (POV behind the SOURCE, angled, looking at
%   the first optic's reflective face), BACK (POV behind the first
%   optic), ISO (angled), SIDE (elevation) -- with the optical-layout
%   convention: the LIGHT SOURCE IS AT THE LEFT and light travels to
%   the RIGHT in the iso and side panels.
%
%   The camera frames are built from the BEAM AXIS (source-plane
%   centroid toward the first surface, from the engine ray history), so
%   the views are meaningful for any prescription orientation.
%
%   Options:
%     'front','back','iso','side'  [az el] fine-tune angles in
%             DEGREES, in the beam frame: az rotates the camera about
%             the beam's vertical (0 = broadside; +90 = camera behind
%             the SOURCE looking along the light at the first optic's
%             face; -90 = from behind the first optic at its back),
%             el tilts it up.  Defaults [75 12], [-75 12], [-35 22],
%             [0 0].  Set an entry to [] to omit that panel.
%     'args'  cell of Name,Value pairs forwarded to every
%             macos.view_rx call ('ray_color', 'show', 'nrings', ...)
%     'title'   overall figure title (default: Rx summary)
%     'save'    PNG path;  'visible' (default true)
%
%   See also: macos.view_rx.
arguments
    opts.front   double = [75 12]
    opts.back    double = [-75 12]
    opts.iso     double = [-35 22]
    opts.side    double = [0 0]
    opts.args    cell   = {}
    opts.title   (1,:) char = ''
    opts.save    (1,:) char = ''
    opts.visible (1,1) logical = true
end
if ~macos.has_rx()
    error('macos:view_std:noRx', 'no prescription is loaded in the engine');
end

% ---- beam frame from the traced history --------------------------------
macos.ray_hist('on');
t = macos.trace();
h = macos.ray_hist(t.nRays);
macos.ray_hist('off');
c0 = mean(squeeze(h.P(:, h.ok(:,1), 1)), 2);         % source plane
j = find(any(h.ok(:, 2:end), 1), 1) + 1;             % first reached surface
c1 = mean(squeeze(h.P(:, h.ok(:,j), j)), 2);
b  = c1 - c0;  b = b / norm(b);                      % light direction
[~, i0] = min(abs(b));  xb = zeros(3,1);  xb(i0) = 1;
xb = xb - dot(xb, b)*b;  xb = xb / norm(xb);         % broadside axis
yb = cross(b, xb);                                   % screen-up axis

panels = {};
if ~isempty(opts.front), panels{end+1} = {opts.front, 'front view (behind source)'}; end
if ~isempty(opts.back),  panels{end+1} = {opts.back,  'back view'};  end
if ~isempty(opts.iso),   panels{end+1} = {opts.iso,   'iso view'};   end
if ~isempty(opts.side),  panels{end+1} = {opts.side,  'side view'};  end
np = numel(panels);

vis = 'on';  if ~opts.visible, vis = 'off'; end
% Grid layout (near-square) instead of a single 1xN row: a 4-panel row on
% a short figure makes each panel tiny.  ncol = ceil(sqrt(np)) gives 2x2
% for 4 panels, 2x1 for 2, etc.; the figure is sized so each TILE is a
% large ~720x620 px regardless of paper count.
ncol = ceil(sqrt(np));  nrow = ceil(np/ncol);
tilew = 720;  tileh = 620;
fig = figure('Visible', vis, ...
    'Position', [40 40 min(ncol*tilew, 2400) min(nrow*tileh, 1800)]);
tl = tiledlayout(fig, nrow, ncol, 'Padding', 'tight', 'TileSpacing', 'tight');
for q = 1:np
    ax = nexttile(tl);
    macos.view_rx('ax', ax, 'title', panels{q}{2}, opts.args{:});
    a = deg2rad(panels{q}{1});
    % camera forward in the beam frame: az about yb from +xb toward +b,
    % el toward yb.  Screen-right = forward x up = the beam direction
    % at az=0 -- light travels LEFT -> RIGHT.
    f = cos(a(2))*(cos(a(1))*xb + sin(a(1))*b) + sin(a(2))*yb;
    axis(ax, 'equal');
    xl = xlim(ax);  yl = ylim(ax);  zl = zlim(ax);
    tgt = [mean(xl); mean(yl); mean(zl)];
    d = 3 * max([diff(xl), diff(yl), diff(zl)]);
    set(ax, 'CameraTarget', tgt.', 'CameraPosition', (tgt - d*f).', ...
            'CameraUpVector', yb.', 'Projection', 'orthographic');
    camva(ax, 'auto');         % frame the scene from the manual camera
    % ---- tighten the frame to the PROJECTED drawn content (2026-08-29).
    % camva('auto') frames the whole axis bounding BOX; an oblique camera
    % projects that box far larger than the content, which is where the
    % wide panel margins in every 4-view figure came from.  Project each
    % drawn point (lines, patches, surfaces, text anchors) onto the
    % camera screen axes and set the view angle to the content extent
    % plus breathing room.  Orthographic: visible half-height =
    % d*tan(camva/2).
    % camva('auto') frames the full axis bounding BOX; an oblique camera
    % projects that box far larger than the drawn content, which is where
    % the wide panel margins in every 4-view figure came from.  Zoom IN by
    % the measured ratio of the BOX's projected extent to the CONTENT's
    % projected extent, per screen axis, taking the smaller ratio.  Fully
    % relative (camzoom), so it is independent of MATLAB's camva
    % semantics and of the tile pixel geometry.
    Pd = viewstd_points_(ax);
    if ~isempty(Pd)
        fw = f / norm(f);
        rt = cross(fw, yb);  rt = rt / norm(rt);
        [cx, cy, cz] = ndgrid(xl, yl, zl);
        B = [cx(:), cy(:), cz(:)].';               % the 8 box corners
        relC = Pd - tgt;   relB = B - tgt;
        zu = max(abs(yb.' * relB)) / max([abs(yb.' * relC), 1e-12]);
        zr = max(abs(rt.' * relB)) / max([abs(rt.' * relC), 1e-12]);
        z = min(zu, zr) / 1.10;                    % 10% breathing room
        if z > 1, camzoom(ax, z); end              % only ever tighter
    end
    axis(ax, 'off');           % LightTools-clean panels; the title stays
end
if isempty(opts.title)
    opts.title = sprintf('%d elements -- standard views', macos.num_elt());
end
title(tl, opts.title, 'Interpreter', 'none');
fig.Name = opts.title;
if ~isempty(opts.save), print(fig, opts.save, '-dpng', '-r150'); end
end

function P = viewstd_points_(ax)
%VIEWSTD_POINTS_  All finite drawn coordinates in AX, as a 3xN array.
P = zeros(3, 0);
for h = findall(ax)'
    switch h.Type
        case 'line'
            z = h.ZData(:).';
            if isempty(z), z = zeros(1, numel(h.XData)); end
            P = [P, [h.XData(:).'; h.YData(:).'; z]];         %#ok<AGROW>
        case 'patch'
            V = h.Vertices;
            if size(V, 2) == 2, V(:, 3) = 0; end
            P = [P, V.'];                                     %#ok<AGROW>
        case 'surface'
            P = [P, [h.XData(:).'; h.YData(:).'; h.ZData(:).']]; %#ok<AGROW>
        case 'text'
            p = h.Position(:);
            if numel(p) == 2, p(3) = 0; end
            P = [P, p];                                       %#ok<AGROW>
    end
end
P = P(:, all(isfinite(P), 1));
end
