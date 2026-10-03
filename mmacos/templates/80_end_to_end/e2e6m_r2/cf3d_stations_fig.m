function cf3d_stations_fig(orient)
%CF3D_STATIONS_FIG  Render the CF3d internal-stations figure(s) from the
%   cached fields in cf3d_stations.mat (written by CF3D_STATIONS).
%     'tall'  7 stations x 2 states (the ctb_coro_compare column layout)
%             -> cf3d_stations.png
%     'wide'  2 states x 7 stations -- the slide strip; colorbars on the
%             bottom row only, short station titles
%             -> cf3d_stations_wide.png
%     'both'  (default)
%
%   See also CF3D_STATIONS.
    arguments
        orient (1,:) char {mustBeMember(orient,{'tall','wide','both'})} = 'both'
    end
    here = fileparts(mfilename('fullpath'));
    D = load(fullfile(here, 'cf3d_stations.mat'));
    S = D.S;  pb = D.pb;  N = D.N;  c = D.c;
    w = round(N/8);
    fr = max(1,round(c-w)) : min(N,round(c+w));
    % {long title, short title, field, scale, crop}
    st = { ...
      'pupil (Apodizer plane, before masks)',  'pupil',      'pup',   'amp', []; ...
      'after circular stop + prolate apodizer','apodized',   'apod',  'amp', []; ...
      'FPM plane, before occulter',            'FPM pre',    'fpm0',  'log', fr; ...
      'FPM plane, after occulter',             'FPM post',   'fpm1',  'log', fr; ...
      'Lyot plane, before stop',               'Lyot pre',   'lyot0', 'log', []; ...
      'Lyot plane, after 0.90 stop',           'Lyot post',  'lyot1', 'log', []; ...
      'science plane -- CONTRAST',             'science (contrast)', 'fpa', 'con', fr};
    ttl = sprintf(['e2e6m CF3d internal stations -- d=1.10 m apl, %s | ' ...
        'DZ 3-15 lambda/D: flat %.2e -> dug %.2e'], D.tag, D.czf, D.czd);

    if any(strcmp(orient, {'tall','both'}))
        fig = figure('Position',[40 40 760 2200], 'Color','w', 'Visible','off');
        tl = tiledlayout(fig, size(st,1), 2, 'TileSpacing','compact', ...
                         'Padding','compact');
        title(tl, ttl, 'Interpreter','none', 'FontSize',10, 'FontWeight','bold');
        for r = 1:size(st,1)
            for s = 1:2
                ax = nexttile(tl);
                panel_(ax, S(s), st(r,:), pb, true);
                if s == 1, title(ax, st{r,1}, 'FontSize',8, ...
                                 'FontWeight','normal'); end
                if r == 1, subtitle(ax, S(s).name, 'FontSize',9, ...
                                    'FontWeight','bold'); end
            end
        end
        exportgraphics(fig, fullfile(here,'cf3d_stations.png'), 'Resolution',130);
        fprintf('wrote cf3d_stations.png\n');
    end

    if any(strcmp(orient, {'wide','both'}))
        fig = figure('Position',[40 40 2300 620], 'Color','w', 'Visible','off');
        tl = tiledlayout(fig, 2, size(st,1), 'TileSpacing','tight', ...
                         'Padding','compact');
        title(tl, ttl, 'Interpreter','none', 'FontSize',11, 'FontWeight','bold');
        for s = 1:2
            for r = 1:size(st,1)
                ax = nexttile(tl);
                panel_(ax, S(s), st(r,:), pb, s == 2);   % colorbars: bottom row
                if s == 1, title(ax, st{r,2}, 'FontSize',10, ...
                                 'FontWeight','normal'); end
                if r == 1
                    axis(ax, 'on');  box(ax,'off');
                    set(ax,'XTick',[],'YTick',[],'XColor','none','YColor','none');
                    ylabel(ax, S(s).name, 'FontSize',10, 'FontWeight','bold', ...
                           'Color','k', 'Visible','on');
                end
            end
        end
        exportgraphics(fig, fullfile(here,'cf3d_stations_wide.png'), 'Resolution',130);
        fprintf('wrote cf3d_stations_wide.png\n');
    end
end

function panel_(ax, Ss, row, pb, withcb)
    E = Ss.(row{3});
    if ~isempty(row{5}), E = E(row{5}, row{5}); end
    switch row{4}
        case 'amp'
            imagesc(ax, abs(E));  colormap(ax, gray);
        case 'log'
            imagesc(ax, log10(abs(E).^2 / max(abs(E(:)).^2) + 1e-12));
            colormap(ax, parula);  clim(ax, [-10 0]);
            if withcb
                cb = colorbar(ax);  cb.Label.String = 'log_{10} norm I';
            end
        case 'con'
            imagesc(ax, log10(abs(E).^2 / pb + 1e-14));
            colormap(ax, parula);  clim(ax, [-11 -3]);
            if withcb
                cb = colorbar(ax);  cb.Label.String = 'log_{10} contrast';
            end
    end
    axis(ax, 'image', 'off');
end
