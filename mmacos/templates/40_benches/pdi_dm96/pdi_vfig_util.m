function out = pdi_vfig_util(varargin)
%PDI_VFIG_UTIL  The deck-figure recipe, in one place (Dave 2026-09-12, via
%   zwfs_dm96/zwfs_vlayout.m): layouts are drawn BY THE ENGINE from the
%   emitted deck (macos.view_rx), the fold plane seen from above, passive
%   bookkeeping planes hidden, elements NAMED (not E-numbers) with leader
%   lines off the beam, the crowded node as a cropped panel at full slide
%   width, type readable at slide size (15-17 pt in an 1800 px figure).
%
%   Calls:
%     pdi_vfig_util('flat',  ax, title, fontsize [, az])  from above, equal,
%                                                   titled; az rotates the
%                                                   fold plane in the page
%     pdi_vfig_util('frame', ax, pts, padfrac)      limits about these points
%     pdi_vfig_util('label', ax, lab [, fontsize])  leader-lined names
%
%   lab is an N x 3 cell: {vertex (3x1), offset [fx fy], name}.  THE OFFSET
%   IS A FRACTION OF THE AXIS SPAN, not bench mm -- a panel that crops to
%   600 mm and one that shows 1900 mm then place their labels the same
%   distance off the beam on the printed page, which is the only distance
%   that matters.  Call 'label' AFTER 'frame' (it reads the limits).
%
%   padfrac is [left right bottom top] as fractions of the point cloud's
%   own extent.  Titles are wrapped to WRAPCOL characters a line: a title
%   wider than its axes is a title the slide cuts off at both ends.
WRAPCOL = 105;
switch varargin{1}
case 'flat'
    [ax, ttl, fs] = varargin{2:4};
    az = 0;  if numel(varargin) > 4, az = varargin{5}; end   % rotate the fold plane in the page:
    axes(ax);  axis(ax, 'equal');  view(ax, az, 90);         %#ok<LAXES> a portrait node at full width
    title(ax, wrap_(ttl, WRAPCOL), 'Color', [11 11 11]/255, 'FontWeight', 'normal', 'FontSize', fs);
    out = ax;
case 'frame'
    [ax, pts, pad] = varargin{2:4};
    x0 = min(pts(1,:));  x1 = max(pts(1,:));  y0 = min(pts(2,:));  y1 = max(pts(2,:));
    w = max(x1 - x0, eps);  h = max(y1 - y0, eps);
    xlim(ax, [x0 - pad(1)*w, x1 + pad(2)*w]);
    ylim(ax, [y0 - pad(3)*h, y1 + pad(4)*h]);
    xlabel(ax, 'bench x, mm', 'Color', [11 11 11]/255, 'FontSize', 14);
    ylabel(ax, 'bench y, mm', 'Color', [11 11 11]/255, 'FontSize', 14);
    set(ax, 'FontSize', 13);  grid(ax, 'on');
    set(ax, 'GridColor', [225 224 217]/255, 'Color', 'w');
    out = ax;
case 'label'
    [ax, lab] = varargin{2:3};
    fs = 16;  if numel(varargin) > 3, fs = varargin{4}; end
    xl = xlim(ax);  yl = ylim(ax);  sx = diff(xl);  sy = diff(yl);
    for k = 1:size(lab, 1)
        p = lab{k,1};  d = [lab{k,2}(1)*sx, lab{k,2}(2)*sy];
        plot3(ax, [p(1) p(1)+d(1)], [p(2) p(2)+d(2)], [0.2 0.2], '-', ...
            'Color', [137 135 129]/255, 'LineWidth', 1.0);
        va = 'bottom';  if d(2) < 0, va = 'top'; end
        text(ax, p(1)+d(1), p(2)+d(2), 0.3, lab{k,3}, 'Color', [11 11 11]/255, ...
            'FontSize', fs, 'HorizontalAlignment', 'center', 'VerticalAlignment', va, ...
            'BackgroundColor', 'w', 'Margin', 1);
    end
    out = ax;
otherwise
    error('pdi_vfig_util: unknown action %s', varargin{1});
end
end

function t = wrap_(s, n)
% word-wrap to n characters a line (a cell of lines: MATLAB stacks them)
w = strsplit(strtrim(s), ' ');  t = {};  cur = '';
for i = 1:numel(w)
    if isempty(cur), cand = w{i}; else, cand = [cur ' ' w{i}]; end
    if numel(cand) > n && ~isempty(cur), t{end+1} = cur;  cur = w{i}; %#ok<AGROW>
    else, cur = cand; end
end
if ~isempty(cur), t{end+1} = cur; end
end
