function f = spectrometer_maps_fig(R, file, opts)
%SPECTROMETER_MAPS_FIG  Performance maps of one scored deck, deck standard.
%   f = spectrometer_maps_fig(R, file) draws the 1 x 6 panel row of a
%   spectrometer_score result: the field-angle map u_c, the keystone map
%   u_c - u_c(lambda_c), the smile map v_c - v_c(slit centre), SRF FWHM,
%   CRF FWHM and the geometric ensquared energy -- all in pixel units with
%   the convention of each quantity in its axis/colorbar label (sign and
%   axis stated on the figure, not in a caption).  Options: 'title',
%   'pixel_um' (18).
    arguments
        R struct
        file (1,:) char
        opts.title (1,:) char = ''
        opts.pixel_um (1,1) double = 18
    end
    im = ceil(numel(R.xs)/2);  jm = ceil(numel(R.lams)/2);
    K = R.U - R.U(:, jm);  Sm = R.V - R.V(im, :);
    panels = {R.U, 'u_c (px): spatial centroid, +x along the slit', 'field-angle map'; ...
              K,   sprintf('u_c - u_c(%.0f nm) (px): keystone, max %.4f', R.lams(jm)*1e9, max(abs(K(:)))), 'keystone map'; ...
              Sm,  sprintf('v_c - v_c(slit centre) (px): smile, max %.4f', max(abs(Sm(:)))), 'smile map'; ...
              R.SRF, sprintf('SRF FWHM (px), max %.3f', max(R.SRF(:))), 'SRF'; ...
              R.CRF, sprintf('CRF FWHM (px), max %.3f', max(R.CRF(:))), 'CRF'; ...
              R.EE, sprintf('ensquared in 1 px (geometric), min %.3f', min(R.EE(:))), 'ensquared energy'};
    f = figure('Visible', 'off', 'Position', [40 40 1500 300], 'Color', 'w');
    for k = 1:6
        ax = subplot(1, 6, k);
        imagesc(ax, R.lams*1e9, R.xs*1e3, panels{k,1});  axis(ax, 'xy');
        cb = colorbar(ax);  cb.Label.String = panels{k,2};  cb.Label.FontSize = 8;
        xlabel(ax, '\lambda (nm)');  if k == 1, ylabel(ax, 'slit x (mm)'); end
        title(ax, panels{k,3}, 'FontSize', 10);
    end
    if ~isempty(opts.title)
        sgtitle(f, sprintf('%s -- %.0f um pixels; per (slit x, lambda): engine ray centroids and spots', opts.title, opts.pixel_um), 'FontSize', 11, 'Interpreter', 'none');
    end
    print(f, file, '-dpng', '-r130');  close(f);
end
