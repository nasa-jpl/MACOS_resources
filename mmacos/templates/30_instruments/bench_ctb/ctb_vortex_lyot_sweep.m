function out = ctb_vortex_lyot_sweep(opts)
%CTB_VORTEX_LYOT_SWEEP  Vortex Lyot-fraction trade against APLC / band-limited.
%   out = CTB_VORTEX_LYOT_SWEEP() sweeps the Lyot stop fraction for the
%   8x-binned scalar vortex (charges 4 and 6, ctb_vortex_matched with a
%   dense fraction grid) and plots dark-zone mean contrast against
%   THROUGHPUT, with the fixed-design reference points -- apodized-pupil
%   Lyot, band-limited, hard occulter -- read from the committed
%   head-to-head (ctb_mask_compare.mat) so every marker shares the same
%   grid, annulus (3-15 lambda/D), and normalization.
%
%   The point of the figure: with the sampled core cured
%   (ctb_mask_vortex), the Lyot fraction is a real depth-for-throughput
%   dial -- charge 4 spans APLC-class depth at 25% throughput to
%   1e-8-class at 81% -- where the fixed designs are single points.
%
%   Name-value:
%     'lyot_fracs'  fraction grid (default [0.50 0.60 0.70 0.80 0.85
%                   0.90 0.95 0.99])
%     'charges'     vortex charges (default [4 6])
%     'outdir'      figure dir (this dir)
%     'visible'     show figure (false)
%
%   out: the ctb_vortex_matched sweep struct plus .refs (the comparison
%   points) and .figure.
%
%   Run:  >> out = ctb_vortex_lyot_sweep;        (~5 min at N=1024)
%   See also: ctb_vortex_matched, ctb_mask_compare, ctb_mask_vortex.
    arguments
        opts.lyot_fracs (1,:) double = [0.50 0.60 0.70 0.80 0.85 0.90 0.95 0.99]
        opts.charges    (1,:) double = [4 6]
        opts.outdir     (1,:) char = ''
        opts.visible    (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', '..', 'src'));
    if isempty(opts.outdir), opts.outdir = here; end

    % ---- the sweep (engine runs) ---------------------------------------
    out = ctb_vortex_matched('charges', opts.charges, ...
                             'lyot_fracs', opts.lyot_fracs);

    % ---- fixed-design reference points (committed head-to-head) --------
    cmp = load(fullfile(here, 'ctb_mask_compare.mat'));
    pick = @(fam) struct( ...
        'C', cmp.tbl.dz_mean(strcmp(cmp.tbl.family, fam)), ...
        'T', cmp.tbl.throughput(strcmp(cmp.tbl.family, fam)));
    refs = struct('aplc', pick('aplc'), 'blc', pick('blc'), ...
                  'hard', pick('hard'));
    out.refs = refs;

    % ---- figure --------------------------------------------------------
    vis = 'off'; if opts.visible, vis = 'on'; end
    fig = figure('Visible',vis, 'Color','w', 'Position',[80 80 860 560]);
    ax = axes(fig);  hold(ax, 'on');  grid(ax, 'on');
    mk = {'o-', 's-'};
    for k = 1:numel(opts.charges)
        T = 100 * out.throughput;
        C = out.contrast_grid(k, :);
        semilogy(ax, T, C, mk{k}, 'LineWidth', 1.5, 'MarkerSize', 6, ...
            'DisplayName', sprintf('vortex, charge %d (Lyot swept)', ...
                                   opts.charges(k)));
        for j = 1:numel(T)
            text(ax, T(j), C(j)*1.35, sprintf('%.2f', opts.lyot_fracs(j)), ...
                'FontSize', 8, 'HorizontalAlignment', 'center', ...
                'Color', [0.35 0.35 0.35]);
        end
    end
    rr = {refs.aplc, 'p', 'apodized-pupil Lyot'; ...
          refs.blc,  'd', 'band-limited (4th)'; ...
          refs.hard, '^', 'hard occulter'};
    for k = 1:size(rr, 1)
        semilogy(ax, 100*rr{k,1}.T, rr{k,1}.C, rr{k,2}, ...
            'MarkerSize', 11, 'LineWidth', 1.5, ...
            'MarkerFaceColor', [0.85 0.85 0.85], 'Color', [0.1 0.1 0.1], ...
            'DisplayName', rr{k,3});
    end
    set(ax, 'YScale', 'log');
    xlabel(ax, 'throughput (%)');
    ylabel(ax, sprintf('dark-zone mean contrast (3--15 \\lambda/D)'));
    title(ax, sprintf(['Vortex Lyot-fraction sweep vs fixed designs -- ' ...
        '8x-binned masks, N=1024, 500 nm']), 'FontWeight', 'bold');
    legend(ax, 'Location', 'northwest');
    xlim(ax, [15 100]);

    fp = fullfile(opts.outdir, 'ctb_vortex_lyot_sweep.png');
    exportgraphics(fig, fp, 'Resolution', 150);
    close(fig);
    out.figure = fp;
    fprintf('[sweep] wrote %s\n', fp);
    save(fullfile(opts.outdir, 'ctb_vortex_lyot_sweep.mat'), '-struct', 'out');
end
