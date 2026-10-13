function out = cophi_hold_vs_cycle(varargin)
%COPHI_HOLD_VS_CYCLE  Budgeted hold (pm rms) vs loop cycle time, sensor by sensor.
%   out = cophi_hold_vs_cycle            writes runs/figs_hold_vs_cycle/hold_vs_cycle.png
%   out = cophi_hold_vs_cycle('cophi')   the three COPHI cameras only
%
%   The budget (Dave 2026-10-13, deck_cophi "the advantage the fast lock-in cycle
%   provides: much more averaging at a fixed cycle time"): a reading's noise
%   integrates as 1/sqrt(readings), so at cycle time T a sensor with s_pm seconds
%   per picometer (SNR 1 at 1 pm, the record's column) reads sigma_n = sqrt(s_pm/T)
%   pm per cycle, and the loop at gain g holds sigma_n sqrt(g/(2-g)) (dmg_loop's
%   steady state; g 0.5 -> 0.577).  A drift RATE costs the lag (rate T)/(g G) per
%   cycle, so the two cross at an optimum cycle time: the record's thermal ramp
%   (5 pm per 16 s cycle = 0.31 pm/s) held 10.3 pm at 16 s = 2.06 x (pm per cycle).
%   Panel 1: noise only, with the loop RECORDS as points (stage loop, mirror rig,
%   plain splitter; lock-in runs/lockin_*).  Panel 2: noise + the thermal ramp.
%   Seconds per pm are the records' mirror-rig values (stages noise / format /
%   lockin; the other gauges from tg_psi_dm96_oap/runs/camI_oap, zwfs_dm96/runs/
%   camS_oap -- the polarization-camera timings).  Pure MATLAB, no engine.
only_cophi = nargin >= 1 && strcmp(varargin{1}, 'cophi');
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
outdir = fullfile(exdir, 'runs', 'figs_hold_vs_cycle');  if ~exist(outdir, 'dir'), mkdir(outdir); end

g = 0.5;  kg = sqrt(g/(2-g));                 % loop steady-state factor, gain 0.5
ramp = 5/16;                                  % pm/s: the record's thermal ramp (5 pm per 16 s cycle)
lag_per = 10.3/5;                             % hold per (pm of ramp per cycle): the record's 10.3 at 5
% sensor, seconds per pm (mirror rig), record source, line style, colour
S = { ...
  'COPHI, Feng''s camera (1k, 60 ke, 30 fps)',       385,  'noise_oap_plain',          '-',  [0.85 0.33 0.10]; ...
  'COPHI, Photonfocus (200 ke, 150 fps)',            36,   'noise_oap_plain',          '-',  [0.00 0.45 0.74]; ...
  'COPHI, lock-in camera C4 (well-capped, 1 kHz)',   3.35, 'lockin_noise_oap',         '-',  [0.47 0.20 0.67]; ...
  'COPHI, lock-in, well-free (2.5e9 e-/s, analytic)', 0.34, 'format_lens_plain (analytic)', '--', [0.47 0.20 0.67]; ...
  'IFO, Twyman-Green (pol. camera)',                 160,  'tg_psi_dm96_oap camI_oap', ':',  [0.30 0.30 0.30]; ...
  'vector ZWFS (pol. camera)',                       519,  'zwfs_dm96 camS_oap',       ':',  [0.00 0.60 0.30]; ...
  'stepped ZWFS (mono camera)',                      475,  'zwfs_dm96 camS_oap',       ':',  [0.00 0.60 0.30]*0.6; ...
  'pinhole / P-SRI (mono camera)',                   342,  'zwfs_dm96 camS_oap',       ':',  [0.60 0.45 0.00]};
if only_cophi, S = S(1:4, :); end
% the loop RECORDS (mirror rig, noise only): cycle time s, hold pm rms, sensor row
R = {16, 2.67, 1; 1.6, 2.52, 2; 0.125, 3.09, 3; 0.044, 5.22, 3};
T = logspace(-2.5, 2.5, 200);
f = figure('Color', 'w', 'Position', [20 20 1500 620], 'Visible', 'off');
tl = tiledlayout(f, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
for panel = 1:2
    ax = nexttile(tl);  hold(ax, 'on');
    for k = 1:size(S, 1)
        sig = sqrt(S{k,2} ./ T) * kg;                       % noise-only hold, pm
        if panel == 2, sig = hypot(sig, lag_per * ramp * T); end
        plot(ax, T, sig, S{k,4}, 'Color', S{k,5}, 'LineWidth', 1.8, 'DisplayName', S{k,1});
    end
    if panel == 1
        for r = 1:size(R, 1)
            plot(ax, R{r,1}, R{r,2}, 'o', 'MarkerSize', 9, 'MarkerFaceColor', S{R{r,3},5}, ...
                'Color', 'k', 'LineWidth', 1.0, 'HandleVisibility', 'off');
        end
        plot(ax, nan, nan, 'ko', 'MarkerSize', 9, 'MarkerFaceColor', [0.7 0.7 0.7], 'DisplayName', 'loop records (stage loop, mirror rig)');
    end
    yline(ax, 3, ':', '3 pm hold', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.2, 'HandleVisibility', 'off', 'LabelHorizontalAlignment', 'left');
    set(ax, 'XScale', 'log', 'YScale', 'log', 'FontSize', 13);
    xlim(ax, [T(1) T(end)]);  ylim(ax, [0.3 300]);  grid(ax, 'on');
    xlabel(ax, 'loop cycle time, s', 'FontSize', 14);
    ylabel(ax, 'hold, pm rms', 'FontSize', 14);
    if panel == 1
        title(ax, 'Noise only: hold = sqrt(s_{pm}/T) x 0.577 (gain 0.5); points = the loop records', 'FontSize', 14, 'FontWeight', 'normal');
        legend(ax, 'Location', 'northeast', 'FontSize', 11);
    else
        title(ax, sprintf('With the record''s thermal ramp (%.2f pm/s): lag = 2.06 x ramp x T; the optimum cycle', ramp), 'FontSize', 14, 'FontWeight', 'normal');
    end
end
fn = fullfile(outdir, iff_(only_cophi, 'hold_vs_cycle_cophi.png', 'hold_vs_cycle.png'));
print(f, fn, '-dpng', '-r150');  close(f);
% the optimum per sensor (panel 2) and the hold at three cycle times, for the deck's table
out = struct('sensor', {S(:,1)}, 's_pm', cell2mat(S(:,2)), 'record', {S(:,3)});
Tq = [0.125 1.6 16];
fid = fopen(fullfile(outdir, 'hold_vs_cycle.txt'), 'w');
fprintf(fid, 'hold pm rms (noise only, gain 0.5) at cycle %g / %g / %g s; optimum cycle and hold with the %.2f pm/s ramp\n', Tq, ramp);
for k = 1:size(S, 1)
    hq = sqrt(S{k,2} ./ Tq) * kg;
    tot = hypot(sqrt(S{k,2} ./ T) * kg, lag_per * ramp * T);  [hmin, im] = min(tot);
    out.hold_at(k, :) = hq;  out.T_opt(k) = T(im);  out.hold_opt(k) = hmin;
    fprintf(fid, '%-52s s_pm %7.2f : %6.2f / %6.2f / %6.2f pm ; optimum %6.3f s -> %5.2f pm   [%s]\n', S{k,1}, S{k,2}, hq, T(im), hmin, S{k,3});
end
fclose(fid);
fprintf('wrote %s (+ hold_vs_cycle.txt)\n', fn);
end

function v = iff_(c, a, b)
if c, v = a; else, v = b; end
end
