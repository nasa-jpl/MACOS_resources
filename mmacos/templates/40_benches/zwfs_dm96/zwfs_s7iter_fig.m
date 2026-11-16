function zwfs_s7iter_fig()
%ZWFS_S7ITER_FIG  Deck figure for S7: (left) the 96x96 modal transfer on
%   the CORRECTED (DM-conjugate) model for the three reading classes,
%   against the legacy-model linear record (zwfs_s3.mat); (right) the
%   working-state ladder (single-actuator 10 nm differential gain vs base
%   rms) per reading.  Reads zwfs_s7iter.mat + zwfs_s3.mat; writes
%   zwfs_s7iter.png.  Categorical palette = the dataviz reference order
%   (blue/orange/aqua/yellow, validated 2026-09-09; the two low-contrast
%   hues carry direct labels).  Run: cd <this dir>; matlab -batch "zwfs_s7iter_fig; exit(0)"
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
cd(exdir);
Z = load('zwfs_s7iter.mat');  Z = Z.out;  R = Z.n96;
S3 = load('zwfs_s3.mat');  S3 = S3.out.n96;
PQ3 = [1 0;2 0;4 0;8 0;16 0;24 0;32 0;48 0;64 0;80 0;8 8;24 24];   % S3's probe set
c_L = [42 120 214]/255;  c_I = [235 104 52]/255;  c_S = [27 175 122]/255;  c_Ip = [237 161 0]/255;
ink = [11 11 11]/255;  ink2 = [82 81 78]/255;  muted = [137 135 129]/255;
grid_c = [225 224 217]/255;  axis_c = [195 194 183]/255;  surf_c = [252 252 251]/255;
f = figure('Color', surf_c, 'Position', [100 100 1500 620], 'Visible', 'off');
tl = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
tl.Title.String = 'ZWFS on the corrected (DM-conjugate) model, 96x96: the Talbot null is gone; the one-frame exact reading with the refined base prior holds to 60 nm rms';
tl.Title.FontSize = 13;  tl.Title.Color = ink;

% ---- left: modal transfer -------------------------------------------
ax = nexttile;  hold(ax, 'on');
is1 = R.PQ(:,2) == 0;  f1 = R.PQ(is1,1)/2;
is3 = PQ3(:,2) == 0;   f3 = PQ3(is3,1)/2;
plot(ax, f3, S3.gk(is3), '--', 'Color', muted, 'LineWidth', 2, 'DisplayName', 'linear, LEGACY model (S3 record)');
ser = {R.gk(is1,1), c_L, 'linear (L)'; R.gk(is1,2), c_I, 'exact, iterated b (I)'; R.gk(is1,3), c_S, 'phase-stepped (S)'};
for k = 1:3
    plot(ax, f1, ser{k,1}, '-o', 'Color', ser{k,2}, 'LineWidth', 2, 'MarkerSize', 6, ...
        'MarkerFaceColor', ser{k,2}, 'MarkerEdgeColor', surf_c, 'DisplayName', ser{k,3});
end
yline(ax, 0, 'Color', axis_c, 'LineWidth', 1, 'HandleVisibility', 'off');
yline(ax, 1, ':', 'Color', axis_c, 'LineWidth', 1, 'HandleVisibility', 'off');
[gmin, imin] = min(S3.gk(is3));
text(ax, f3(imin)+1.0, gmin, sprintf('legacy null %.2f at %g cyc/ap', gmin, f3(imin)), ...
    'Color', ink2, 'FontSize', 10, 'VerticalAlignment', 'middle');
yl = spread_(cellfun(@(v) v(end), ser(:,1)), 0.05);
for k = 1:3
    text(ax, f1(end)+0.8, yl(k), ser{k,3}, 'Color', ink, 'FontSize', 10, 'VerticalAlignment', 'middle');
end
xlabel(ax, 'spatial frequency, cycles per aperture ((p,0) probes)', 'Color', ink2);
ylabel(ax, 'modal transfer gain through the actuator-space estimator', 'Color', ink2);
title(ax, 'Modal transfer, raw (before the Wiener)', 'Color', ink, 'FontWeight', 'normal');
xlim(ax, [0 52]);  grid(ax, 'on');  style_(ax, grid_c, axis_c, ink2, surf_c);
legend(ax, 'Location', 'southwest', 'TextColor', ink, 'Color', surf_c, 'EdgeColor', axis_c);

% ---- right: the working-state ladder ---------------------------------
ax = nexttile;  hold(ax, 'on');
lad = R.ladder;  amps = [lad.amp]*1e6;
G = reshape([lad.g], 5, []).';                 % rungs x readings (L F I I+ S)
serb = {G(:,1), c_L, 'linear (L)'; G(:,2), muted, 'exact, frozen b (F)'; G(:,3), c_I, 'exact, iterated b (I)'; ...
        G(:,4), c_Ip, 'I + refined base prior (I+)'; G(:,5), c_S, 'phase-stepped (S)'};
inm = amps <= 60;                              % the measurement regime; past it the recoveries alias
for k = 1:5
    semilogx(ax, amps, serb{k,1}, '-o', 'Color', serb{k,2}, 'LineWidth', 2, 'MarkerSize', 6, ...
        'MarkerFaceColor', serb{k,2}, 'MarkerEdgeColor', surf_c, 'DisplayName', serb{k,3});
end
yl = spread_(cellfun(@(v) v(1), serb(:,1)), 0.05);
for k = 1:5
    text(ax, amps(1)/1.05, yl(k), serb{k,3}, 'Color', ink, 'FontSize', 10, 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'right');
end
yline(ax, 0, 'Color', axis_c, 'LineWidth', 1, 'HandleVisibility', 'off');
yline(ax, 1, ':', 'Color', axis_c, 'LineWidth', 1, 'HandleVisibility', 'off');
xline(ax, 85, '-', 'Color', axis_c, 'LineWidth', 1, 'HandleVisibility', 'off');
text(ax, 90, 0.95, {'past the fold every one-frame reading', 'aliases: quote gain, not SNR (S4)'}, ...
    'Color', ink2, 'FontSize', 9, 'VerticalAlignment', 'top');
set(ax, 'XScale', 'log', 'XTick', amps, 'XTickLabel', arrayfun(@(a) sprintf('%g', a), amps, 'UniformOutput', false));
xlim(ax, [amps(1)/3.2, amps(end)*1.3]);
xlabel(ax, 'working-state rms, nm (random base; same rng(7) field, scaled)', 'Color', ink2);
ylabel(ax, 'gain at the poked actuator (10 nm differential, corrected)', 'Color', ink2);
title(ax, 'Break scale: single-actuator differential on a growing base', 'Color', ink, 'FontWeight', 'normal');
grid(ax, 'on');  style_(ax, grid_c, axis_c, ink2, surf_c);
legend(ax, 'Location', 'southwest', 'TextColor', ink, 'Color', surf_c, 'EdgeColor', axis_c);
exportgraphics(f, 'zwfs_s7iter.png', 'Resolution', 110, 'BackgroundColor', surf_c);
fprintf('wrote zwfs_s7iter.png\n');
end

function y = spread_(y, gap)
% push label anchors apart (in data units) so direct labels never overlap
[ys, i] = sort(y(:));
for k = 2:numel(ys)
    if ys(k) - ys(k-1) < gap, ys(k) = ys(k-1) + gap; end
end
y(i) = ys;
end

function style_(ax, grid_c, axis_c, ink2, surf_c)
ax.GridColor = grid_c;  ax.GridAlpha = 1;  ax.XColor = axis_c;  ax.YColor = axis_c;
ax.Color = surf_c;  ax.Box = 'off';  ax.FontSize = 10;
ax.XLabel.Color = ink2;  ax.YLabel.Color = ink2;
ax.XAxis.TickLabelColor = ink2;  ax.YAxis.TickLabelColor = ink2;
end
