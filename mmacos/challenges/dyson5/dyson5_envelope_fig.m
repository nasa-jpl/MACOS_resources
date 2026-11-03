function dyson5_envelope_fig(T, axes_, P, file)
%DYSON5_ENVELOPE_FIG  The closure-envelope figure: per axis, each spec metric
%   divided by ITS limit (1 = the limit; CRF / P.xrf_px, SRF / the slit-floor
%   limit max(P.srf_px(end), P.slit_px) + 0.1 px, max(smile, keystone) /
%   P.smile_px), points that do not close marked.  Called by dyson5_envelope
%   and standalone from the record: S = load('dyson5_s4env.mat');
%   dyson5_envelope_fig(S.S.table, S.S.axes, S.P, 'dyson5_s4env.png').
    srf_lim = max(P.srf_px(end), P.slit_px) + 0.1;
    f = figure('Visible', 'off', 'Position', [40 40 1400 330], 'Color', 'w');
    for a = 1:numel(axes_)
        ax = subplot(1, numel(axes_), a);  hold(ax, 'on');  grid(ax, 'on');
        sel = find(strcmp(T.axis, axes_(a).label));
        xv = 1:numel(sel);  lab = T.value(sel);
        plot(ax, xv, T.CRF_px(sel)/P.xrf_px, 'o-', xv, T.SRF_px(sel)/srf_lim, 's-', xv, max(T.smile_px(sel), T.keystone_px(sel))/P.smile_px, 'd-');
        yline(ax, 1, 'k--');  set(ax, 'XTick', xv, 'XTickLabel', lab, 'FontSize', 8);
        cl = T.closes(sel);  plot(ax, xv(~cl), ones(1, nnz(~cl))*1.02, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
        title(ax, axes_(a).label, 'FontSize', 9);  if a == 1, ylabel(ax, 'metric / its limit (1 = limit)'); end
        if a == numel(axes_), legend(ax, {'CRF / 1.5 px', sprintf('SRF / %.1f px (slit floor)', srf_lim), 'smile|keystone / 0.1 px', 'does not close'}, 'Location', 'best', 'FontSize', 7); end
        ylim(ax, [0, max(1.3, max(ylim(ax)))]);
    end
    sgtitle(f, 'dyson5 closure envelope: R4 re-solved from the record, one axis at a time (engine scores / limits; x = on a bound or over a limit)', 'FontSize', 10);
    print(f, file, '-dpng', '-r130');  close(f);
end
