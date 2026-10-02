function f = telescope_maps_fig(R, file, opts)
%TELESCOPE_MAPS_FIG  The telescope's score at the slit, deck standard.
%   f = telescope_maps_fig(R, file) draws a 1 x 5 row from a telescope_score
%   (engine) or telescope_score_chain result over the field along the slit:
%   the rms spot (along / across the slit, px) with the slit's admittance,
%   the ensquared fractions, the chief's angle to the slit normal
%   (telecentricity) and to the spectrometer's own aim (pupil match) with
%   the chief's miss of the grating vertex, the best-focus offset along the
%   slit normal (field flatness) with the rms at best focus, and the mapping
%   (local IFOV against the design IFOV, the slit landing against the slit
%   ends).  Pixel units where the scorer uses them; the convention of each
%   quantity in its axis label.  Options: 'title', 'pixel_um' (18).
    arguments
        R struct
        file (1,:) char
        opts.title (1,:) char = ''
        opts.pixel_um (1,1) double = 18
    end
    th = R.fields*180/pi;
    f = figure('Visible', 'off', 'Position', [40 40 1600 320], 'Color', 'w');
    subplot(1, 5, 1);  plot(th, R.SU, 'o-', th, R.SV, 's-', th, R.S, 'k.-', 'LineWidth', 1.2);  grid on
    yline(0.5, '--', 'half a pixel');  xlabel('field along the slit (deg)');  ylabel('rms spot (px)');
    legend({'along the slit (u)', 'across the slit (v)', 'radius'}, 'Location', 'best');  title(sprintf('spot at the slit, max %.2f px', max(R.S)));
    subplot(1, 5, 2);  plot(th, R.EE1, 'o-', th, R.SLIT, 's-', 'LineWidth', 1.2);  grid on;  ylim([0 1.05])
    yline(0.75, '--', 'paper > 0.75');  xlabel('field (deg)');  ylabel('fraction');
    legend({'in 1 px about the centroid', sprintf('through the %g px slit', R.slit_px)}, 'Location', 'best');  title(sprintf('ensquared, min %.2f / slit %.2f', min(R.EE1), min(R.SLIT)));
    subplot(1, 5, 3);  yyaxis left;  plot(th, R.tel_rad*180/pi, 'o-', th, R.err_rad*180/pi, 's-', 'LineWidth', 1.2);  ylabel('chief angle (deg)');  grid on
    yyaxis right;  plot(th, R.walk_m*1e3, 'd-', 'LineWidth', 1.2);  ylabel('chief miss of the grating vertex (mm)');
    xlabel('field (deg)');  legend({'to the slit normal (telecentricity)', 'to the spectrometer''s aim (pupil match)', 'miss on the grating'}, 'Location', 'best');
    title(sprintf('pupil match: max miss %.1f mm', max(R.walk_m)*1e3));
    subplot(1, 5, 4);  yyaxis left;  plot(th, R.zbf_m*1e6, 'o-', 'LineWidth', 1.2);  ylabel('best focus along the slit normal (um)');  grid on
    yyaxis right;  plot(th, R.sbf_px, 's-', 'LineWidth', 1.2);  ylabel('rms spot at best focus (px)');
    xlabel('field (deg)');  title(sprintf('field flatness: p-v %.0f um', (max(R.zbf_m) - min(R.zbf_m))*1e6));
    subplot(1, 5, 5);  thm = 0.5*(th(1:end-1) + th(2:end));
    yyaxis left;  plot(thm, R.ifov_ratio, 'o-', 'LineWidth', 1.2);  ylabel('local IFOV / design IFOV');  grid on
    yyaxis right;  plot(th, R.map_lin/R.pixel_m, 's-', 'LineWidth', 1.2);  ylabel('departure from the linear map (px)');
    xlabel('field (deg)');  title(sprintf('mapping: ends %+.1f / %+.1f px off the slit ends', R.end_err_m/R.pixel_m));
    if ~isempty(opts.title), sgtitle(opts.title, 'FontWeight', 'bold'); end
    print(f, file, '-dpng', '-r110');  close(f);
end
