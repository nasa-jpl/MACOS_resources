function png = seq_centroid_maps(C, Frel, png)
%SEQ_CENTROID_MAPS  The two field maps, deck-quality.
%
%   png = SEQ_CENTROID_MAPS()            re-render from rodgers1_seq_centroid.mat
%   png = SEQ_CENTROID_MAPS(C, Frel, png)
%
%   Top row  : CENTROID DISPLACEMENT (um on the detector) -- the coma tracker.
%   Bottom   : departure of the centroid grid from the ideal f.theta grid
%              (um), scale held at f, detector placement/clocking removed.
%
%   Per-panel colour scales, deliberately: stage 2's displacement is ~6x
%   stage 4's, and a shared scale would flatten the very shrink the map
%   exists to show.  Each panel therefore carries its own bar, and the
%   headline range is printed in its subtitle so the panels stay comparable
%   by reading rather than by colour.

    if nargin < 1 || isempty(C)
        here = fileparts(mfilename('fullpath'));
        M = load(fullfile(here,'rodgers1_seq_centroid.mat'));
        C = M.C;
        P = rodgers_common('seq');  Frel = P.seq.Frel;
        png = fullfile(here,'rodgers1_seq_centroid_maps.png');
    end

    idx = find(arrayfun(@(x) ~isempty(x.name), C));
    idx = idx(1:min(4,numel(idx)));
    x = Frel(:,1)*180/pi*60;   y = Frel(:,2)*180/pi*60;
    pad = 1.2;

    fig = figure('Visible','off','Position',[60 60 1400 760],'Color','w');
    tl  = tiledlayout(fig, 2, numel(idx), 'TileSpacing','compact','Padding','compact');

    for j = 1:numel(idx)
        i = idx(j);

        ax = nexttile(tl, j);
        v  = C(i).dcen_um;
        panel_(ax, x, y, v, pad);
        title(ax, C(i).name, 'FontSize',10, 'FontWeight','bold');
        subtitle(ax, sprintf('centroid displacement  %.2f - %.2f \\mum', min(v), max(v)), ...
                 'FontSize',8.5);
        if j == 1, ylabel(ax, '\DeltaYAN (arcmin)'); else, ylabel(ax,''); end

        ax = nexttile(tl, numel(idx)+j);
        if ~isempty(C(i).dist)
            w = C(i).dist.raw_um;
            panel_(ax, x, y, w, pad);
            subtitle(ax, sprintf('vs ideal f\\cdot\\theta   %.0f - %.0f \\mum', ...
                                 min(w), max(w)), 'FontSize',8.5);
        else
            axis(ax,'off');
        end
        xlabel(ax, 'XAN (arcmin)');
        if j == 1, ylabel(ax, '\DeltaYAN (arcmin)'); else, ylabel(ax,''); end
    end

    title(tl, ['Rodgers TMA at the .seq truth   —   centroid displacement (top) ' ...
               'and distortion (bottom)'], 'FontWeight','bold','FontSize',13);
    subtitle(tl, ['EPD 5000 mm, \lambda = 1 \mum, his 15-point half box.  ' ...
                  'Displacement shrinks as the solves correct the field; ' ...
                  'distortion is a property of the layout and does not.'], ...
             'FontSize',9.5);
    exportgraphics(fig, png, 'Resolution', 170);
    close(fig);
end

function panel_(ax, x, y, v, pad)
    scatter(ax, x, y, 300, v, 'filled', 'MarkerEdgeColor',[0.25 0.25 0.25]);
    axis(ax,'equal');  box(ax,'on');  grid(ax,'on');
    xlim(ax, [min(x)-pad, max(x)+pad]);
    ylim(ax, [min(y)-pad, max(y)+pad]);
    set(ax,'FontSize',8.5,'Layer','top','GridAlpha',0.15);
    colormap(ax, parula);
    cb = colorbar(ax);  cb.FontSize = 8;
end
