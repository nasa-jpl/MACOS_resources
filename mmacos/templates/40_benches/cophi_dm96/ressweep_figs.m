function T = ressweep_figs(varargin)
%RESSWEEP_FIGS  The resolution sweep's composite figure (the deck slide "How much
%   resolution?"): hold at the record cycle (pm) and seconds per pm against camera
%   pixels per actuator, one line per gauge, the collapse band shaded.  Reads
%   runs/ressweep_{ifo,het,het_iq,zwfs}/*.mat (whatever exists); writes
%   runs/ressweep_figs/ressweep_composite.png + ressweep_composite.txt (the same
%   numbers as a table).  Two panels, two y-scales of their own -- never one figure
%   with two y-axes.
%
%   The FAILURE BAND is read from the records, not assumed: the samplings at which
%   ANY gauge's single-actuator row fails -- its noise-free floor (one-step) reaches
%   o.floor_fail pm (the records' anchors sit at 1-5 pm), its converged gain is off
%   unity by more than o.gain_tol, or a +-0.1 pitch shift reads the wrong actuator.
%   Shaded from the plot's left edge to midway between the largest failing sampling
%   and the next one that holds (the boundary lies between them).  The brief's
%   'collapse' (the fit reading the wrong actuator) appears at NO sampling.
exdir = fileparts(mfilename('fullpath'));
o = struct('gain_tol', 0.05, 'floor_fail', 10, 'out', fullfile(exdir, 'runs', 'ressweep_figs'));
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
if ~exist(o.out, 'dir'), mkdir(o.out); end
% fixed categorical order (validated: dataviz palette slots 1-7, light surface)
col = {'#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7'};
mk  = {'o', 's', '^', 'v', 'd', '>', '<'};
G = {  % label, gauge file, reading/camera selector
    'IFO snapshot (pol. camera)',      'ifo',    'polcam';
    'vector ZWFS (pol. camera)',       'zwfs',   'V';
    'stepped ZWFS (mono)',             'zwfs',   'S';
    'pinhole PDI (mono)',              'zwfs',   'P';
    'HET, Feng''s camera',             'het',    'baseline';
    'HET, Photonfocus',                'het',    'photonfocus';
    'HET, lock-in C4',                 'het_iq', 'lockin'};
PT = cell(0, 4);
T = struct('label', {}, 's', {}, 'hold_pm', {}, 'spm', {}, 'cycle_s', {}, 'g1', {}, 'wrong', {}, 'fl1', {}, 'col', {}, 'mk', {});
for i = 1:size(G, 1)
    f = fullfile(exdir, 'runs', ['ressweep_' G{i, 2}], ['ressweep_' G{i, 2} '.mat']);
    if ~isfile(f), continue; end
    L = load(f);  res = L.res;  [~, ix] = sort([res.s_target]);  res = res(ix);
    % the ZWFS's native 1.65 is a POINT-sampled grid (NGRID 193, the camera records'),
    % not a camera's pixels: kept out of the lines, drawn as its own hollow marker
    pt = abs([res.s_target] - 1.65) < 1e-6;  respt = res(pt);  res = res(~pt);
    s = nan(1, numel(res));  hp = s;  sp = s;  cy = s;  g1 = s;  wr = s;  f1 = s;
    for j = 1:numel(res)
        R = res(j).R;  s(j) = R.s;
        if strcmp(G{i, 2}, 'zwfs')
            rd = G{i, 3};  k = find(strcmp(R.readings, rd));
            nz = R.noise(strcmp({R.noise.rd}, rd) & strcmp({R.noise.binning}, 'digital'));
            lp = R.loop(strcmp({R.loop.rd}, rd) & strcmp({R.loop.drift}, 'none'));
            rn = find(strcmp(R.rows_conv_names, 'rand30/single10'));
            g1(j) = R.rows_conv(rn, k, 1);  wr(j) = any(R.reg(2:end, k, 5));
            rw = res(j).rows;  q = find(strcmp({rw.row}, 'rand30/single10') & strcmp({rw.rd}, rd), 1);
            f1(j) = rw(q).raw(3);
        else
            nz = R.noise(strcmp({R.noise.cam}, G{i, 3}) & strcmp({R.noise.binning}, 'digital'));
            lp = R.loop(strcmp({R.loop.cam}, G{i, 3}) & strcmp({R.loop.drift}, 'none'));
            g1(j) = R.rows_conv{1, 2};  wr(j) = any(R.reg(2:end, 8));  f1(j) = R.rows{1, 4};
        end
        sp(j) = nz.s_per_pm;  hp(j) = lp.ss_pf_pm;  cy(j) = lp.cycle_s;
    end
    if ~isempty(respt)
        Rp = respt(1).R;  kp = strcmp({Rp.noise.rd}, G{i, 3}) & strcmp({Rp.noise.binning}, 'digital');
        lpp = Rp.loop(strcmp({Rp.loop.rd}, G{i, 3}) & strcmp({Rp.loop.drift}, 'none'));
        PT(end+1, :) = {i, Rp.s, lpp.ss_pf_pm, Rp.noise(kp).s_per_pm}; %#ok<AGROW>
    end
    T(end+1) = struct('label', G{i, 1}, 's', s, 'hold_pm', hp, 'spm', sp, 'cycle_s', cy, 'g1', g1, 'wrong', wr, 'fl1', f1, ...
        'col', col{i}, 'mk', mk{i}); %#ok<AGROW>
end
assert(~isempty(T), 'ressweep_figs: no ressweep records under runs/');
% ---- the collapse band, from the records ---------------------------------
bad = [];  allS = [];
for i = 1:numel(T)
    bad = [bad, T(i).s(abs(T(i).g1 - 1) > o.gain_tol | T(i).wrong > 0 | T(i).fl1 >= o.floor_fail)]; %#ok<AGROW>
    allS = [allS, T(i).s]; %#ok<AGROW>
end
sbad = max([bad, -Inf]);  sc = sbad;
nxt = min(allS(allS > sbad + 0.05));
if isfinite(sbad) && ~isempty(nxt), sc = (sbad + nxt)/2; end
% ---- figure: two panels, shared x ------------------------------------------
fg = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [0 0 11 4.6], ...
    'PaperPositionMode', 'manual', 'PaperUnits', 'inches', 'PaperPosition', [0 0 11 4.6]);
tl = tiledlayout(fg, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ttl = {'Hold at each sensor''s loop cycle (noise only), pm', 'Averaging time to 1 pm per actuator, s'};
fld = {'hold_pm', 'spm'};
xs = [0.9 4.3];
for p = 1:2
    ax = nexttile(tl);  hold(ax, 'on');
    yl = [min(cellfun(@(v) min(v), {T.(fld{p})}))/2, max(cellfun(@(v) max(v), {T.(fld{p})}))*2];
    if isfinite(sc)
        patch(ax, [xs(1) sc sc xs(1)], yl([1 1 2 2]), [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.6, ...
            'HandleVisibility', 'off');
        text(ax, xs(1) + 0.03, yl(1)*1.6, {'single-', 'actuator', 'row fails'}, 'Color', [0.35 0.35 0.35], 'FontSize', 9, ...
            'VerticalAlignment', 'bottom');
    end
    for i = 1:numel(T)
        plot(ax, T(i).s, T(i).(fld{p}), '-', 'Color', T(i).col, 'LineWidth', 2, 'Marker', T(i).mk, ...
            'MarkerFaceColor', T(i).col, 'MarkerEdgeColor', 'w', 'MarkerSize', 7, ...
            'DisplayName', sprintf('%s, %.3g s cycle', T(i).label, T(i).cycle_s(1)));
    end
    for q = 1:size(PT, 1)
        v = PT{q, 2 + p};
        plot(ax, PT{q, 2}, v, T(PT{q, 1}).mk, 'Color', T(PT{q, 1}).col, 'MarkerFaceColor', 'w', 'LineWidth', 1.5, ...
            'MarkerSize', 8, 'HandleVisibility', 'off');
    end
    if ~isempty(PT) && p == 2
        text(ax, 2.15, 18, {'hollow at 1.65: the ZWFS on a POINT-sampled', 'grid (the camera records''), not a camera''s pixels'}, ...
            'Color', [0.35 0.35 0.35], 'FontSize', 8);
    end
    set(ax, 'YScale', 'log', 'XLim', xs, 'YLim', yl, 'XTick', [1 1.5 2 3.3 4], 'Box', 'off', ...
        'XColor', [0.3 0.3 0.3], 'YColor', [0.3 0.3 0.3], 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 1);
    grid(ax, 'on');
    if p == 1, set(ax, 'YTick', [1 2 3 5 10 20 30 50], 'YTickLabel', {'1','2','3','5','10','20','30','50'}); end
    xlabel(ax, 'Camera pixels per actuator pitch');  title(ax, ttl{p}, 'FontWeight', 'normal');
    if p == 2, legend(ax, 'Location', 'eastoutside', 'Box', 'off'); end
end
exportgraphics(fg, fullfile(o.out, 'ressweep_composite.png'), 'Resolution', 160);  close(fg);
% ---- the same numbers as a table ---------------------------------------------
fid = fopen(fullfile(o.out, 'ressweep_composite.txt'), 'w');  cl = onCleanup(@() fclose(fid));
fprintf(fid, ['ressweep composite -- %s; failure band: samplings <= %.3f px/act fail the single-actuator row (noise-free floor >= %g pm, ' ...
    '|g_conv - 1| > %.2f, or a +-0.1 pitch shift reads the wrong actuator); shaded to %.2f\n'], char(datetime('now')), sbad, o.floor_fail, o.gain_tol, sc);
for i = 1:numel(T)
    for j = 1:numel(T(i).s)
        fprintf(fid, '%-30s %6.3f px/act  hold %7.3g pm (cycle %.3g s)  %9.3g s per pm  single: floor %5.1f pm, g_conv %.4f  wrong %d\n', T(i).label, T(i).s(j), ...
            T(i).hold_pm(j), T(i).cycle_s(j), T(i).spm(j), T(i).fl1(j), T(i).g1(j), T(i).wrong(j));
    end
end
end
