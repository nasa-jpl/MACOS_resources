function [png, mp] = oi_map_fig(X, G, P, offset_deg, stage_lbl, png, deck_txt)
%OI_MAP_FIG  Dense strict-WFE-vs-field map over the box + its figure.
%
%   [PNG, MP] = OI_MAP_FIG(X, G, P, OFFSET_DEG, LBL, PNG) scores the
%   P.map_n x P.map_n dense map (per-field exit-pupil anchors -- the
%   reportable metric) and writes the figure.  MP: .W (map, nm),
%   .XG/.YG (deg), .max_nm/.avg_nm/.std_nm/.min_nm, .max_at (XAN,YAN).
%   The map statistic is the one Mike's ladder quotes: the MAP MAXIMUM.
%
%   See also OI_SCORE, OFFSET_IMAGER.

    F = oi_fieldset(P, offset_deg, P.map_n);
    if nargin < 7 || isempty(deck_txt)
        txt = oi_deck(fill_(X, P));
    else
        txt = deck_txt;     % gate hook: score a committed deck verbatim
    end
    sc = oi_score(txt, G, F);
    W = reshape(sc.wfe_cen_nm, P.map_n, P.map_n);
    XG = reshape(F(:,1), P.map_n, P.map_n);
    YG = reshape(F(:,2), P.map_n, P.map_n);

    % F6/F7 (t5 experiment): fields that lose every ray are REPORTED,
    % never silently folded in.  mp.valid is false the moment ANY field
    % failed -- a "dense-map maximum" computed from a partial map is not
    % the packet metric and must never print as one.
    fin = isfinite(W);
    nfld = numel(W);  nbad = nfld - nnz(fin);
    if ~any(fin(:))
        mp = struct('W',W,'XG',XG,'YG',YG, ...
                    'max_nm',NaN, 'avg_nm',NaN, 'std_nm',NaN, ...
                    'min_nm',NaN, 'max_at',[NaN NaN], ...
                    'wfe_cen_nm',sc.wfe_cen_nm, 'fields',F, ...
                    'n_fields',nfld, 'n_failed',nbad, 'valid',false);
        fig = figure('Visible','off','Color','w','Position',[100 100 720 600]);
        ax = axes(fig);  axis(ax,'off');
        text(ax, 0.5, 0.55, sprintf(['%s -- %s\nDENSE MAP FAILED: all ' ...
             '%d field points lost every ray at YAN %+g deg.\nNo map ' ...
             'statistic exists for this stage.'], P.name, stage_lbl, ...
             nfld, offset_deg), 'HorizontalAlignment','center', ...
             'FontSize', 13, 'Interpreter','none');
        exportgraphics(fig, png, 'Resolution', 140);
        close(fig);
        return
    end
    [mx, im] = max(W(:) - 1e12*~fin(:));
    mp = struct('W',W,'XG',XG,'YG',YG, ...
                'max_nm',mx, 'avg_nm',mean(W(fin)), 'std_nm',std(W(fin)), ...
                'min_nm',min(W(fin)), 'max_at',[XG(im) YG(im)], ...
                'wfe_cen_nm',sc.wfe_cen_nm, 'fields',F, ...
                'n_fields',nfld, 'n_failed',nbad, 'valid',nbad == 0);

    fig = figure('Visible','off','Color','w','Position',[100 100 720 600]);
    ax = axes(fig);
    imagesc(ax, XG(1,:), YG(:,1), W, 'AlphaData', double(fin));  % NaN = blank
    set(ax,'YDir','normal');
    colormap(ax, parula);
    cb = colorbar(ax);  cb.Label.String = 'strict RMS WFE (nm, centroid ref)';
    hold(ax,'on');
    plot(ax, XG(im), YG(im), 'r^', 'MarkerSize',9, 'LineWidth',1.5);
    if nbad == 0
        xlabel(ax, sprintf('XAN (deg)   [map max %.1f nm, avg %.1f, std %.1f]', ...
               mp.max_nm, mp.avg_nm, mp.std_nm));
    else
        xlabel(ax, sprintf(['XAN (deg)   [INVALID: %d/%d fields lost every ' ...
               'ray; finite-only max %.1f nm]'], nbad, nfld, mp.max_nm));
    end
    ylabel(ax,'YAN (deg)');
    title(ax, sprintf('%s -- %s -- RMS WFE vs field (%dx%d)', ...
          P.name, stage_lbl, P.map_n, P.map_n), 'Interpreter','none');
    exportgraphics(fig, png, 'Resolution', 140);
    close(fig);

    % numeric self-check (headless-PNG trap): the stats must describe W
    assert(abs(mp.max_nm - max(W(fin))) < 1e-9, 'oi_map_fig:stats');
end

function D = fill_(X, P)
    D = X;
    D.EPD_m = P.EPD_m;  D.WL_m = P.lambda_m;
    D.sampling = P.sampling;  D.name = P.name;
end
