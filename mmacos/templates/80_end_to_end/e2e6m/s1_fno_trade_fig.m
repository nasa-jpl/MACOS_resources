function out = s1_fno_trade_fig(matfile, png, opts)
%S1_FNO_TRADE_FIG  The f/# vs wavefront trade, drawn from a closure run.
%
%   `s1_close_fno` walks the M3 base radius trying to pull the CORRECTED
%   f/# into band.  Whether it succeeds or refuses, the run's own iterates
%   ARE the trade curve -- what a faster layout costs in wavefront -- and
%   that curve is the reportable result either way.  This draws it from
%   the saved `s1_close_fno.mat`, so the exhibit is the measurement, not a
%   redrawing of it.
%
%   Two panels sharing the R3 axis: the corrected f/# against the band,
%   and the dense-map -tilt max against the diffraction limit.  A point
%   is filled when it meets EVERY gate (f/#, wavefront, shroud, clear) and
%   hollow otherwise, so a refusal reads as plainly as a success.
%
%   out = S1_FNO_TRADE_FIG(MATFILE, PNG) -> the plotted table.
%
%   Name-value: 'band','dl_waves','shroud_D_m' (defaults from
%   e2e6m_params), 'visible' (false).
%
%   See also S1_CLOSE_FNO, E2E6M_PARAMS.

    arguments
        matfile (1,:) char
        png     (1,:) char
        opts.band       (1,2) double = [0 0]
        opts.dl_waves   (1,1) double = 0
        opts.shroud_D_m (1,1) double = 0
        opts.visible    (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    setup_(here);
    P = e2e6m_params();
    if all(opts.band == 0),      opts.band = P.fno_band;         end
    if opts.dl_waves == 0,       opts.dl_waves = P.dl_waves;     end
    if opts.shroud_D_m == 0,     opts.shroud_D_m = P.shroud_D_m; end

    S = load(matfile);  R = S.OUT.R;
    [~, o] = sort([R.R3]);  R = R(o);
    x  = [R.R3];  y1 = [R.fno];  y2 = [R.wfe_tilt];
    ok = [R.ok];
    out = struct('R3',x, 'fno',y1, 'wfe_tilt',y2, 'ok',ok, ...
                 'shroud',[R.shroud], 'clear',[R.clear]);

    f = figure('Visible', tern_(opts.visible,'on','off'), ...
               'Position',[100 100 1100 460]);

    ax1 = subplot(1,2,1);  hold(ax1,'on');
    patch(ax1, [min(x) max(x) max(x) min(x)], ...
          [opts.band(1) opts.band(1) opts.band(2) opts.band(2)], ...
          [0.85 0.92 0.85], 'EdgeColor','none');
    plot(ax1, x, y1, '-', 'Color',[0.25 0.25 0.25], 'LineWidth',1.2);
    scat_(ax1, x, y1, ok);
    xlabel(ax1,'M3 base radius R3  [m]');  ylabel(ax1,'corrected f/#');
    title(ax1, sprintf('f/# at the FP (band %g-%g shaded)', opts.band));
    grid(ax1,'on');  box(ax1,'on');

    ax2 = subplot(1,2,2);  hold(ax2,'on');
    plot(ax2, [min(x) max(x)], opts.dl_waves*[1 1], '--', ...
         'Color',[0.20 0.45 0.20], 'LineWidth',1.4);
    plot(ax2, x, y2, '-', 'Color',[0.25 0.25 0.25], 'LineWidth',1.2);
    scat_(ax2, x, y2, ok);
    set(ax2,'YScale','log');
    xlabel(ax2,'M3 base radius R3  [m]');
    ylabel(ax2,'dense-map RMS WFE, -tilt max  [waves @ 500 nm]');
    title(ax2, sprintf('wavefront (diffraction limit %.3f dashed)', opts.dl_waves));
    grid(ax2,'on');  box(ax2,'on');

    try
        sgtitle(f, ['what a faster layout costs: the freeform stage spends ' ...
                    'optical power, so f/# and wavefront trade'], ...
                'FontWeight','bold', 'Interpreter','none');
    catch
    end
    saveas(f, png);
    if ~opts.visible, close(f); end
    out.png = png;
end

% =========================================================================
function scat_(ax, x, y, ok)
%SCAT_  Filled = meets every gate; hollow = does not.  Labelled by iterate
%   so a reader can find the run directory that produced each point.
    for k = 1:numel(x)
        if ok(k)
            plot(ax, x(k), y(k), 'o', 'MarkerSize',9, ...
                 'MarkerFaceColor',[0.15 0.35 0.70], 'MarkerEdgeColor','k');
        else
            plot(ax, x(k), y(k), 'o', 'MarkerSize',9, ...
                 'MarkerFaceColor','w', 'MarkerEdgeColor',[0.55 0.20 0.20], ...
                 'LineWidth',1.4);
        end
        text(ax, x(k), y(k), sprintf('  %d', k), 'FontSize',9, ...
             'VerticalAlignment','bottom');
    end
end

function setup_(here)
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
end

function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
