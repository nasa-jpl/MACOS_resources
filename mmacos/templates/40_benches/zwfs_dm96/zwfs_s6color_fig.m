function zwfs_s6color_fig()
%ZWFS_S6COLOR_FIG  Deck figure for the S6 color stage: modal transfer vs
%   spatial frequency per color, both instruments, with the five-color
%   combination's transfer.  Reads zwfs_s6color.mat + ../tg_psi_dm96/
%   tg96_s6color.mat; writes zwfs_s6color.png.  Ordinal (wavelength-
%   ordered) one-hue ramp, light -> dark = 480 -> 780 nm; combination in
%   ink.  Run:  cd <this dir>;  matlab -batch "zwfs_s6color_fig; exit(0)"
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
cd(exdir);
Z = load('zwfs_s6color.mat');  Z = Z.out;
I = load(fullfile('..', 'tg_psi_dm96', 'tg96_s6color.mat'));  I = I.out;
lams = Z.lams_nm;  K = numel(lams);
[~, order] = sort(lams);                        % ramp follows wavelength
ramp = [134 182 239; 85 152 231; 42 120 214; 28 92 171; 16 66 129]/255;
ink = [11 11 11]/255;  ink2 = [82 81 78]/255;  muted = [137 135 129]/255;
grid_c = [225 224 217]/255;  axis_c = [195 194 183]/255;  surf_c = [252 252 251]/255;
f = figure('Color', surf_c, 'Position', [100 100 1500 620], 'Visible', 'off');
tl = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
tl.Title.String = 'Modal transfer vs wavelength, 96x96 DM: the ZWFS null migrates with color, the IFO does not move';
tl.Title.FontSize = 13;  tl.Title.Color = ink;

% ---- ZWFS: (p,0) rows ---------------------------------------------------
ax = nexttile;  hold(ax, 'on');
is1d = Z.PQ(:,2) == 0;  f1 = Z.PQ(is1d,1)/2;  g1 = Z.gk(is1d,:);
Gc = sum(g1.^2, 2) ./ (sum(g1.^2, 2) + Z.beta^2);
for j = 1:K
    k = order(j);
    plot(ax, f1, g1(:,k), '-o', 'Color', ramp(j,:), 'LineWidth', 2, ...
        'MarkerSize', 6, 'MarkerFaceColor', ramp(j,:), 'MarkerEdgeColor', surf_c, ...
        'DisplayName', sprintf('%g nm', lams(k)));
end
plot(ax, f1, Gc, '-s', 'Color', ink, 'LineWidth', 2, 'MarkerSize', 6, ...
    'MarkerFaceColor', ink, 'MarkerEdgeColor', surf_c, 'DisplayName', '5-color combination');
yline(ax, 0, 'Color', axis_c, 'LineWidth', 1, 'HandleVisibility', 'off');
[gmin, imin] = min(g1(:,1));
text(ax, f1(imin)+1.2, gmin, sprintf('632.8 nm null, %.2f at %g cyc/ap', gmin, f1(imin)), ...
    'Color', ink2, 'FontSize', 10, 'VerticalAlignment', 'middle');
text(ax, 9, 1.22, sprintf('5-color combination, min %.3f across the band', min(Gc)), ...
    'Color', ink2, 'FontSize', 10, 'HorizontalAlignment', 'left');
title(ax, 'ZWFS: (p,0) lattice modes through the calibrated estimator', 'Color', ink, 'FontWeight', 'normal');
xlabel(ax, 'spatial frequency, cycles per aperture');  ylabel(ax, 'transfer gain');
ylim(ax, [-0.7 2.5]);  xlim(ax, [0 42]);
legend(ax, 'Location', 'northwest', 'Box', 'off', 'TextColor', ink2, 'FontSize', 9);
style_(ax, grid_c, axis_c, muted, surf_c);

% ---- IFO: diagonal rows below the zero command --------------------------
ax = nexttile;  hold(ax, 'on');
use = I.fk < max(I.fk);  fk = I.fk(use);  gI = I.gk(use,:);
GcI = sum(gI.^2, 2) ./ (sum(gI.^2, 2) + I.beta^2);
for j = 1:K
    k = order(j);
    plot(ax, fk, gI(:,k), '-o', 'Color', ramp(j,:), 'LineWidth', 2, ...
        'MarkerSize', 6, 'MarkerFaceColor', ramp(j,:), 'MarkerEdgeColor', surf_c, ...
        'DisplayName', sprintf('%g nm', lams(k)));
end
plot(ax, fk, GcI, '-s', 'Color', ink, 'LineWidth', 2, 'MarkerSize', 6, ...
    'MarkerFaceColor', ink, 'MarkerEdgeColor', surf_c, 'DisplayName', '5-color combination');
spread = max(gI, [], 2) - min(gI, [], 2);
text(ax, fk(end), gI(end,1)-0.07, sprintf('five colors coincide: max spread %.4f', max(spread)), ...
    'Color', ink2, 'FontSize', 10, 'HorizontalAlignment', 'right');
title(ax, 'IFO: (p,p) lattice modes through the true-kernel estimator', 'Color', ink, 'FontWeight', 'normal');
xlabel(ax, 'spatial frequency, cycles per aperture');  ylabel(ax, 'transfer gain');
ylim(ax, [0 1.1]);  xlim(ax, [0 60]);
legend(ax, 'Location', 'southwest', 'Box', 'off', 'TextColor', ink2, 'FontSize', 9);
style_(ax, grid_c, axis_c, muted, surf_c);

exportgraphics(f, 'zwfs_s6color.png', 'Resolution', 130);
close(f);
fprintf('wrote zwfs_s6color.png\n');
end

function style_(ax, grid_c, axis_c, muted, surf_c)
ax.Color = surf_c;  ax.XColor = muted;  ax.YColor = muted;
ax.GridColor = grid_c;  ax.GridAlpha = 1;  ax.YGrid = 'on';  ax.XGrid = 'off';
ax.Box = 'off';  ax.LineWidth = 1;  ax.TickDir = 'out';
ax.XAxis.Color = axis_c;  ax.YAxis.Color = axis_c;
ax.XLabel.Color = [82 81 78]/255;  ax.YLabel.Color = [82 81 78]/255;
ax.XAxis.TickLabelColor = [82 81 78]/255;  ax.YAxis.TickLabelColor = [82 81 78]/255;
ax.FontSize = 10;
end
