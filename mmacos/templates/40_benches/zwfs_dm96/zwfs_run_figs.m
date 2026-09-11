function zwfs_run_figs(out)
%ZWFS_RUN_FIGS  Figures for a zwfs_run result (called by stage 'figs').
%   zwfs_run_figs(out)            from the struct zwfs_run returns
%   zwfs_run_figs('runs/x/x.mat') from a saved run
%   Writes into the run's output dir:
%     <tag>_battery_n<NACT>.png   modal transfer per reading class (left) and the
%                                 working-state ladder per reading (right), per DM config
%     <tag>_color.png             per-color modal transfer per class + the K-color
%                                 combination's transfer
%     <tag>_noise.png             photon-noise sigma vs photons per state, per reading
%     <tag>_loop.png              closed-loop hold: residual vs cycle per photon level
%                                 (left, the walk drift) and the steady-state hold error
%                                 vs photons per cycle per reading and drift, with the
%                                 hold spec line (right)
%   Palette: the dataviz categorical order used across the campaign figures.
if ischar(out) || isstring(out), q = load(out);  out = q.out; end
P = out.P;
if ~exist(P.outdir, 'dir'), mkdir(P.outdir); end
pal = struct('L',[42 120 214]/255, 'F',[137 135 129]/255, 'I',[235 104 52]/255, ...
             'Ip',[237 161 0]/255, 'S',[27 175 122]/255);
ink = [11 11 11]/255;  ink2 = [82 81 78]/255;  muted = [137 135 129]/255;
grid_c = [225 224 217]/255;  axis_c = [195 194 183]/255;  surf_c = [252 252 251]/255;
cls_c = {pal.L, pal.I, pal.S};  cls_n = {'linear map (L)', 'exact map (I class)', 'stepped map (S)'};
name = @(rd) strrep(rd, '+', 'p');
colof = @(rd) pal.(name(rd));
lbl = struct('L','linear (L)', 'F','exact, frozen b (F)', 'I','exact, iterated b (I)', ...
             'Ip','I + refined base prior (I+)', 'S','phase-stepped (S)');

% ---- battery ------------------------------------------------------------
if isfield(out, 'battery')
    fns = fieldnames(out.battery);
    for i = 1:numel(fns)
        R = out.battery.(fns{i});  cfg = R.cfg;
        f = figure('Color', surf_c, 'Position', [100 100 1500 620], 'Visible', 'off');
        tl = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
        tl.Title.String = sprintf('%s: DM %dx%d, NGRID %d, spot %.2f lam/D, %s', P.tag, cfg.nact, cfg.nact, P.NGRID, P.mask.DIA_LAMD, P.bench.mask_prop);
        tl.Title.FontSize = 13;  tl.Title.Color = ink;
        ax = nexttile;  hold(ax, 'on');
        is1 = R.PQ(:,2) == 0;  f1 = R.PQ(is1,1)/2;
        for k = find(~isnan(R.gk(1,:)))
            plot(ax, f1, R.gk(is1,k), '-o', 'Color', cls_c{k}, 'LineWidth', 2, 'MarkerSize', 6, ...
                'MarkerFaceColor', cls_c{k}, 'MarkerEdgeColor', surf_c, 'DisplayName', cls_n{k});
        end
        yline(ax, 0, 'Color', axis_c, 'HandleVisibility', 'off');  yline(ax, 1, ':', 'Color', axis_c, 'HandleVisibility', 'off');
        xlabel(ax, 'spatial frequency, cycles per aperture ((p,0) probes)', 'Color', ink2);
        ylabel(ax, 'modal transfer gain through the actuator-space estimator', 'Color', ink2);
        title(ax, 'Modal transfer, raw (before the Wiener)', 'Color', ink, 'FontWeight', 'normal');
        grid(ax, 'on');  style_(ax, grid_c, axis_c, ink2, surf_c);
        legend(ax, 'Location', 'southwest', 'TextColor', ink, 'Color', surf_c, 'EdgeColor', axis_c);
        ax = nexttile;  hold(ax, 'on');
        lad = R.ladder;  amps = [lad.amp]*1e6;  G = reshape([lad.g], numel(R.readings), []).';
        for k = 1:numel(R.readings)
            rd = R.readings{k};
            semilogx(ax, amps, G(:,k), '-o', 'Color', colof(rd), 'LineWidth', 2, 'MarkerSize', 6, ...
                'MarkerFaceColor', colof(rd), 'MarkerEdgeColor', surf_c, 'DisplayName', lbl.(name(rd)));
        end
        yline(ax, 0, 'Color', axis_c, 'HandleVisibility', 'off');  yline(ax, 1, ':', 'Color', axis_c, 'HandleVisibility', 'off');
        set(ax, 'XScale', 'log', 'XTick', amps, 'XTickLabel', arrayfun(@(a) sprintf('%g', a), amps, 'UniformOutput', false));
        xlim(ax, [amps(1)/1.5, amps(end)*1.3]);
        xlabel(ax, 'working-state rms, nm (random base, same field scaled)', 'Color', ink2);
        ylabel(ax, sprintf('gain at the poked actuator (%g nm differential, corrected)', P.battery.dev_single*1e6), 'Color', ink2);
        title(ax, 'Break scale: single-actuator differential on a growing base', 'Color', ink, 'FontWeight', 'normal');
        grid(ax, 'on');  style_(ax, grid_c, axis_c, ink2, surf_c);
        legend(ax, 'Location', 'southwest', 'TextColor', ink, 'Color', surf_c, 'EdgeColor', axis_c);
        fn = fullfile(P.outdir, sprintf('%s_battery_%s.png', P.tag, fns{i}));
        exportgraphics(f, fn, 'Resolution', 110, 'BackgroundColor', surf_c);  close(f);
        fprintf('wrote %s\n', fn);
    end
end

% ---- color ---------------------------------------------------------------
if isfield(out, 'color')
    C = out.color;  K = numel(C.lams_nm);  is1 = C.PQ(:,2) == 0;  f1 = C.PQ(is1,1)/2;
    classes = find(~isnan(C.gk(1,:,1)));
    ramp = [ [8 48 107]; [33 102 172]; [67 147 195]; [146 197 222]; [209 229 240] ]/255;   % ordinal blues
    f = figure('Color', surf_c, 'Position', [100 100 520*numel(classes)+200 560], 'Visible', 'off');
    tl = tiledlayout(1, numel(classes), 'Padding', 'compact', 'TileSpacing', 'compact');
    tl.Title.String = sprintf('%s: one physical mask at %s nm -- per-color transfer and the %d-color combination', P.tag, num2str(C.lams_nm), K);
    tl.Title.FontSize = 13;  tl.Title.Color = ink;
    [~, order] = sort(C.lams_nm);
    for c = classes
        ax = nexttile;  hold(ax, 'on');
        for j = 1:K
            k = order(j);  cc = ramp(max(1, round((j-1)/(K-1)*(size(ramp,1)-1))+1), :);
            plot(ax, f1, squeeze(C.gk(is1,c,k)), '-o', 'Color', cc, 'LineWidth', 1.8, 'MarkerSize', 5, ...
                'MarkerFaceColor', cc, 'MarkerEdgeColor', surf_c, 'DisplayName', sprintf('%g nm', C.lams_nm(k)));
        end
        plot(ax, f1, C.Gc(:,c), '-', 'Color', pal.I, 'LineWidth', 2.6, 'DisplayName', sprintf('%d-color combination (Geff)', K));
        yline(ax, 0, 'Color', axis_c, 'HandleVisibility', 'off');  yline(ax, 1, ':', 'Color', axis_c, 'HandleVisibility', 'off');
        xlabel(ax, 'cycles per aperture', 'Color', ink2);  ylabel(ax, 'modal transfer gain (raw)', 'Color', ink2);
        title(ax, cls_n{c}, 'Color', ink, 'FontWeight', 'normal');
        grid(ax, 'on');  style_(ax, grid_c, axis_c, ink2, surf_c);
        legend(ax, 'Location', 'southwest', 'TextColor', ink, 'Color', surf_c, 'EdgeColor', axis_c);
    end
    fn = fullfile(P.outdir, sprintf('%s_color.png', P.tag));
    exportgraphics(f, fn, 'Resolution', 110, 'BackgroundColor', surf_c);  close(f);
    fprintf('wrote %s\n', fn);
end

% ---- noise -----------------------------------------------------------------
if isfield(out, 'noise')
    N = out.noise;
    f = figure('Color', surf_c, 'Position', [100 100 760 560], 'Visible', 'off');
    ax = axes(f);  hold(ax, 'on');
    for c = 1:numel(N.cols)
        rd = regexprep(N.cols{c}, '\(.*\)', '');  ls = '-';
        if contains(N.cols{c}, 'noiseless'), ls = '--'; elseif contains(N.cols{c}, 'full'), ls = ':'; end
        loglog(ax, N.nstates, N.sig_pm(:,c), [ls 'o'], 'Color', colof(rd), 'LineWidth', 2, 'MarkerSize', 6, ...
            'MarkerFaceColor', colof(rd), 'MarkerEdgeColor', surf_c, 'DisplayName', N.cols{c});
    end
    yline(ax, 1, ':', 'Color', muted, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    text(ax, N.nstates(1)*1.5, 1.25, '1 pm target', 'Color', ink2, 'FontSize', 10);
    set(ax, 'XScale', 'log', 'YScale', 'log');
    xlabel(ax, 'photons per DM state (a reading''s frames share it)', 'Color', ink2);
    ylabel(ax, 'noise sigma of the poked-actuator estimate, pm', 'Color', ink2);
    title(ax, sprintf('%s: photon-noise pricing of the single-actuator differential', P.tag), 'Color', ink, 'FontWeight', 'normal');
    grid(ax, 'on');  style_(ax, grid_c, axis_c, ink2, surf_c);
    legend(ax, 'Location', 'southwest', 'TextColor', ink, 'Color', surf_c, 'EdgeColor', axis_c);
    fn = fullfile(P.outdir, sprintf('%s_noise.png', P.tag));
    exportgraphics(f, fn, 'Resolution', 110, 'BackgroundColor', surf_c);  close(f);
    fprintf('wrote %s\n', fn);
end

% ---- loop ------------------------------------------------------------------
if isfield(out, 'loop')
    LO = out.loop;  res = LO.res;  NPH = LO.nph;  RD = LO.readings;
    kinds = LO.drifts;  kshow = kinds{end};                 % the last drift model on the left panel
    if any(strcmp(kinds, 'walk')), kshow = 'walk'; end
    ramp = [ [209 229 240]; [146 197 222]; [67 147 195]; [33 102 172]; [8 48 107] ]/255;   % ordinal blues, light = few photons
    f = figure('Color', surf_c, 'Position', [100 100 1500 620], 'Visible', 'off');
    tl = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    tl.Title.String = sprintf('%s: closed-loop hold at gain %.2f on the %s, %d cycles, matrix calibration', P.tag, LO.g, LO.surface, LO.K);
    tl.Title.FontSize = 13;  tl.Title.Color = ink;
    ax = nexttile;  hold(ax, 'on');
    j0 = find(strcmp(RD, 'S'), 1);  if isempty(j0), j0 = numel(RD); end   % the stepped reading if run
    for q = 1:numel(NPH)
        i = find(strcmp({res.rd}, RD{j0}) & strcmp({res.drift}, kshow) & [res.nph] == NPH(q), 1);
        if isempty(i), continue; end
        cc = ramp(max(1, round((q-1)/max(numel(NPH)-1,1)*(size(ramp,1)-1))+1), :);
        semilogy(ax, 1:LO.K, res(i).L.rms*1e9, '-', 'Color', cc, 'LineWidth', 1.8, 'DisplayName', sprintf('%.0e photons per cycle', NPH(q)));
    end
    i = find(strcmp({res.rd}, RD{j0}) & strcmp({res.drift}, 'step'), 1);
    if ~isempty(i), semilogy(ax, 1:LO.K, res(i).L.rms*1e9, '--', 'Color', muted, 'LineWidth', 1.5, 'DisplayName', sprintf('noiseless %g nm step', res(i).amp*1e6)); end
    set(ax, 'YScale', 'log');
    yl = ylim(ax);  ylim(ax, [min(yl(1), LO.hold_spec*1e9/3), max(yl(2), LO.hold_spec*1e9*3)]);
    yline(ax, LO.hold_spec*1e9, ':', 'Color', ink2, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    text(ax, 2, LO.hold_spec*1e9*1.3, sprintf('%g pm hold spec', LO.hold_spec*1e9), 'Color', ink2, 'FontSize', 10);
    xlabel(ax, 'cycle', 'Color', ink2);  ylabel(ax, 'residual surface error over lit, pm rms', 'Color', ink2);
    title(ax, sprintf('Reading %s, %s drift: residual per cycle', RD{j0}, kshow), 'Color', ink, 'FontWeight', 'normal');
    grid(ax, 'on');  style_(ax, grid_c, axis_c, ink2, surf_c);
    legend(ax, 'Location', 'northeast', 'TextColor', ink, 'Color', surf_c, 'EdgeColor', axis_c);
    ax = nexttile;  hold(ax, 'on');
    lst = struct('none', ':', 'walk', '-', 'thermal', '--');
    for j = 1:numel(RD)
        for kd = 1:numel(kinds)
            ss = nan(1, numel(NPH));
            for q = 1:numel(NPH)
                i = find(strcmp({res.rd}, RD{j}) & strcmp({res.drift}, kinds{kd}) & [res.nph] == NPH(q), 1);
                if ~isempty(i), ss(q) = res(i).L.ss*1e9; end
            end
            loglog(ax, NPH, ss, [lst.(kinds{kd}) 'o'], 'Color', colof(RD{j}), 'LineWidth', 2, 'MarkerSize', 6, ...
                'MarkerFaceColor', colof(RD{j}), 'MarkerEdgeColor', surf_c, 'DisplayName', sprintf('%s, %s', lbl.(name(RD{j})), kinds{kd}));
        end
    end
    set(ax, 'XScale', 'log', 'YScale', 'log');
    yl = ylim(ax);  ylim(ax, [min(yl(1), LO.hold_spec*1e9/3), max(yl(2), LO.hold_spec*1e9*3)]);
    yline(ax, LO.hold_spec*1e9, ':', 'Color', ink2, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    text(ax, NPH(1)*1.3, LO.hold_spec*1e9*1.25, sprintf('%g pm hold spec', LO.hold_spec*1e9), 'Color', ink2, 'FontSize', 10);
    xlabel(ax, 'photons per cycle (per DM state; a reading''s frames share it)', 'Color', ink2);
    ylabel(ax, 'steady-state hold error over lit, pm rms', 'Color', ink2);
    title(ax, 'Hold error vs photons per cycle (drift: dotted none, solid walk, dashed thermal)', 'Color', ink, 'FontWeight', 'normal');
    grid(ax, 'on');  style_(ax, grid_c, axis_c, ink2, surf_c);
    legend(ax, 'Location', 'southwest', 'TextColor', ink, 'Color', surf_c, 'EdgeColor', axis_c, 'FontSize', 8);
    fn = fullfile(P.outdir, sprintf('%s_loop.png', P.tag));
    exportgraphics(f, fn, 'Resolution', 110, 'BackgroundColor', surf_c);  close(f);
    fprintf('wrote %s\n', fn);
end
end

function style_(ax, grid_c, axis_c, ink2, surf_c)
ax.GridColor = grid_c;  ax.GridAlpha = 1;  ax.XColor = axis_c;  ax.YColor = axis_c;
ax.Color = surf_c;  ax.Box = 'off';  ax.FontSize = 10;
ax.XLabel.Color = ink2;  ax.YLabel.Color = ink2;
ax.XAxis.TickLabelColor = ink2;  ax.YAxis.TickLabelColor = ink2;
end
