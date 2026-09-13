function out = pdi_vfig_util(varargin)
%PDI_VFIG_UTIL  The deck-figure recipe, in one place (Dave 2026-09-12, via
%   zwfs_dm96/zwfs_vlayout.m): layouts are drawn BY THE ENGINE from the
%   emitted deck (macos.view_rx), the fold plane seen from above, passive
%   bookkeeping planes hidden, elements NAMED (not E-numbers) with leader
%   lines off the beam, the crowded node as a cropped panel at full slide
%   width, type readable at slide size (15-17 pt in an 1800 px figure).
%
%   Calls:
%     ax = pdi_vfig_util('axes', fig, tl, span)     a tile of span rows
%     pdi_vfig_util('flat', ax, title)              from above, equal, titled
%     pdi_vfig_util('label', ax, lab)               leader-lined names
%     pdi_vfig_util('frame', ax, pts, pad)          limits about these points
%   lab is an N x 3 cell: {vertex (3x1), offset [dx dy], name}.
switch varargin{1}
case 'axes'
    [~, tl, span] = varargin{2:4};  out = nexttile(tl, [span 1]);
case 'flat'
    [ax, ttl, fs] = varargin{2:4};
    axes(ax);  axis(ax, 'equal');  view(ax, 0, 90);                 %#ok<LAXES>
    title(ax, ttl, 'Color', [11 11 11]/255, 'FontWeight', 'normal', 'FontSize', fs);
    out = ax;
case 'label'
    [ax, lab] = varargin{2:3};
    fs = 17;  if numel(varargin) > 3, fs = varargin{4}; end
    for k = 1:size(lab, 1)
        p = lab{k,1};  d = lab{k,2};
        plot3(ax, [p(1) p(1)+d(1)], [p(2) p(2)+d(2)], [0.2 0.2], '-', ...
            'Color', [137 135 129]/255, 'LineWidth', 1.0);
        text(ax, p(1)+d(1), p(2)+d(2), 0.3, lab{k,3}, 'Color', [11 11 11]/255, ...
            'FontSize', fs, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'BackgroundColor', 'w', 'Margin', 1);
    end
    out = ax;
case 'frame'
    [ax, pts, pad] = varargin{2:4};
    xlim(ax, [min(pts(1,:))-pad(1), max(pts(1,:))+pad(2)]);
    ylim(ax, [min(pts(2,:))-pad(3), max(pts(2,:))+pad(4)]);
    xlabel(ax, 'bench x, mm', 'Color', [11 11 11]/255, 'FontSize', 14);
    ylabel(ax, 'bench y, mm', 'Color', [11 11 11]/255, 'FontSize', 14);
    set(ax, 'FontSize', 13);  grid(ax, 'on');
    set(ax, 'GridColor', [225 224 217]/255, 'Color', 'w');
    out = ax;
otherwise
    error('pdi_vfig_util: unknown action %s', varargin{1});
end
end
