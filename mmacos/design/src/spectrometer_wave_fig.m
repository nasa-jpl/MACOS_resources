function f = spectrometer_wave_fig(S, file, opts)
%SPECTROMETER_WAVE_FIG  The propagation twin's evidence, deck standard.
%   f = spectrometer_wave_fig(S, file) draws, from the s2w stage's struct S
%   (fields offner_m0 = the order-0 validation, offner, dyson = the twins):
%   (1) the order-0 Offner PSF at the band centre, log stretch, with the
%   18 um pixel drawn (the terminal reproduces the Airy spot);
%   (2) per form, the wave - ray centroid offset |dv| (dispersion) map in
%   pixels over (slit x, lambda) -- the agreement the twin measures;
%   (3) per form, SRF and CRF from the propagated PSF against the ray chain
%   at the slit centre vs wavelength, with the 2-px slit floor.
    arguments
        S struct
        file (1,:) char
        opts.pixel_um (1,1) double = 18
    end
    px = opts.pixel_um*1e-6;
    f = figure('Visible', 'off', 'Position', [40 40 1500 420], 'Color', 'w');
    % (1) order-0 PSF
    ax = subplot(1, 4, 1);  P0 = S.offner_m0.psf{1,1};  I = double(P0.I);  dx = P0.dx;  N = size(I,1);  c0 = N/2 + 1;
    g = ((1:N) - c0)*dx*1e6;  w = 40;  k = abs(g) <= w;
    imagesc(ax, g(k), g(k), log10(max(I(k,k)'/max(I(:)), 1e-6)));  axis(ax, 'xy', 'equal', 'tight');  colormap(ax, 'gray');
    hold(ax, 'on');  rectangle(ax, 'Position', [-px/2 -px/2 px px]*1e6, 'EdgeColor', 'm', 'LineWidth', 1.5);
    xlabel(ax, 'u, along the slit (um)');  ylabel(ax, 'v, dispersion (um)');
    title(ax, sprintf('Offner at order 0: PSF (log10), 1 px box; EE %.3f', S.offner_m0.ee), 'FontSize', 10);
    cb = colorbar(ax);  cb.Label.String = 'log10 I / I_{peak}';
    % (2) agreement maps
    forms = {'offner', 'dyson'};
    for q = 1:2
        R = S.(forms{q});  ax = subplot(1, 4, 1 + q);
        D = sqrt(R.d_du.^2 + R.d_dv.^2);
        imagesc(ax, R.lams*1e9, R.xs*1e3, D);  axis(ax, 'xy');  cb = colorbar(ax);
        cb.Label.String = sprintf('|wave - ray centroid| (px), max %.4f', max(D(:)));
        xlabel(ax, '\lambda (nm)');  ylabel(ax, 'slit x (mm)');  title(ax, sprintf('%s: PSF centroid vs ray centroid', forms{q}), 'FontSize', 10);
    end
    % (3) SRF/CRF wave vs lambda at the slit centre
    ax = subplot(1, 4, 4);  hold(ax, 'on');  grid(ax, 'on');
    for q = 1:2
        R = S.(forms{q});  im = ceil(numel(R.xs)/2);
        plot(ax, R.lams*1e9, R.SRF(im, :), '-o', 'DisplayName', [forms{q} ' SRF (wave)']);
        plot(ax, R.lams*1e9, R.CRF(im, :), '--s', 'DisplayName', [forms{q} ' CRF (wave)']);
    end
    yline(ax, 2, ':', '2-px slit floor', 'HandleVisibility', 'off');  xlabel(ax, '\lambda (nm)');  ylabel(ax, 'FWHM (px)');
    legend(ax, 'Location', 'northwest', 'FontSize', 8);  title(ax, 'response functions from the propagated PSF, slit centre', 'FontSize', 10);
    print(f, file, '-dpng', '-r130');  close(f);
end
