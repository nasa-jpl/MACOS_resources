% fig_walk_trend.m -- map max + clearance floor vs box width: the walk's
% five steps as two trends, with the endgame re-solve as the closing
% point.  Every number is read from the COMMITTED records
% (t5_walk/t5_walk_run.mat + t5_endgame/wfe.mat) -- nothing hand-typed.
% Writes walk_trend.png beside this script (consumed by
% deck_rodgers3_walk.py slide F16).  Run from anywhere:
%   matlab -batch "run('.../fig_walk_trend.m')"
here = fileparts(mfilename('fullpath'));
tpl  = fullfile(here,'..','..','templates','10_telescopes','offset_imager');
W = load(fullfile(tpl,'t5_walk','t5_walk_run.mat'));
E = load(fullfile(tpl,'t5_endgame','wfe.mat'));

w  = [W.rec.width];                 % box full width per step (deg)
mp = [W.rec.map_max_nm];            % dense-map max (nm)
cl = [W.rec.clear_min_mm];          % signed clearance floor (mm)
eg = E.rows(E.win);                 % the endgame winner (hinge 32 mm)

f = figure('Visible','off','Position',[100 100 880 500],'Color','w');
ax = axes(f); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

yyaxis(ax,'left');
plot(ax, w, mp, '-o', 'LineWidth',1.8, 'MarkerSize',7, ...
     'MarkerFaceColor','auto');
plot(ax, 15, eg.map_max_nm, 'p', 'MarkerSize',17, 'LineWidth',1.2, ...
     'MarkerFaceColor','auto');
ylabel(ax, 'dense-map max WFE (nm)');
ylim(ax, [0 80]);

yyaxis(ax,'right');
plot(ax, w, cl, '-s', 'LineWidth',1.8, 'MarkerSize',7, ...
     'MarkerFaceColor','auto');
plot(ax, 15, eg.disk_floor_mm, 'p', 'MarkerSize',17, 'LineWidth',1.2, ...
     'MarkerFaceColor','auto');
yline(ax, 25, '--', '25 mm clearance gate', 'LineWidth',1.1, ...
      'LabelHorizontalAlignment','left', 'FontSize',10);
ylabel(ax, 'signed clearance floor (mm)');
ylim(ax, [0 108]);

xlabel(ax, 'field box full width (deg)');
xlim(ax, [4.4 15.9]); xticks(ax, w);
legend(ax, {'walk: map max', 'endgame re-solve', ...
            'walk: clearance floor', 'endgame re-solve'}, ...
       'Location','north', 'NumColumns',2, 'FontSize',9);
text(ax, 14.9, 62, sprintf('%.1f nm @ %.1f mm', ...
     eg.map_max_nm, eg.disk_floor_mm), ...
     'HorizontalAlignment','right', 'FontSize',10);
set(ax, 'FontSize', 11);

png = fullfile(here, 'walk_trend.png');
exportgraphics(f, png, 'Resolution', 150);
fprintf('wrote %s  (walk: %s nm / %s mm; endgame %.2f nm @ %.2f mm)\n', ...
    png, mat2str(round(mp,1)), mat2str(round(cl,1)), ...
    eg.map_max_nm, eg.disk_floor_mm);
close(f);
