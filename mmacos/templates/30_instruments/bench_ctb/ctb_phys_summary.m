function out = ctb_phys_summary(opts)
%CTB_PHYS_SUMMARY  Summary figure for the physics-layer EFC campaign.
%   Reads the saved ctb_efc_phys_*.mat run states (gitignored; regen
%   lines in CTB_PROP_STATUS SESSION 13) and builds the two-panel
%   deck figure: convergence of each physics configuration, and the
%   rebalanced-Lyot floor-vs-throughput trade.  Asset-gated: errors
%   with the regen instruction when the run states are absent.
%
%   Note: ctb_efc_phys_bb.mat predates the band-normalization fix and
%   its contrast series is scaled by nlam=3 here (uniform factor; the
%   loop's decisions were unaffected).
%
%   Run:  >> ctb_phys_summary;
%   See also: ctb_efc_physics, ctb_efc.
    arguments
        opts.outdir  (1,:) char = ''
        opts.visible (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    if isempty(opts.outdir), opts.outdir = here; end
    need = {'ctb_efc_phys_pol.mat','ctb_efc_phys_bb.mat', ...
            'ctb_efc_phys_bbpol.mat','ctb_efc_phys_bbpol_r1.mat', ...
            'ctb_efc_phys_bbpol_L70.mat','ctb_efc_phys_bbpol_L80.mat'};
    for k = 1:numel(need)
        assert(isfile(fullfile(here, need{k})), ...
            'ctb_phys_summary: %s absent -- regen per CTB_PROP_STATUS SESSION 13', need{k});
    end
    P   = load(fullfile(here, 'ctb_efc_phys_pol.mat'));
    B   = load(fullfile(here, 'ctb_efc_phys_bb.mat'));
    BP  = load(fullfile(here, 'ctb_efc_phys_bbpol.mat'));
    BPr = load(fullfile(here, 'ctb_efc_phys_bbpol_r1.mat'));
    L70 = load(fullfile(here, 'ctb_efc_phys_bbpol_L70.mat'));
    L80 = load(fullfile(here, 'ctb_efc_phys_bbpol_L80.mat'));
    Bc  = 3 * B.contrast;                 % pre-fix normalization (see help)

    vis = 'off'; if opts.visible, vis = 'on'; end
    fig = figure('Visible',vis, 'Color','w', 'Position',[80 80 1240 480]);
    tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact', 'Padding','compact');
    title(tl, ['CTB physics-layer EFC -- vortex charge 4, 10% band, ' ...
        'coated-train polarization, rebalanced Lyot'], 'FontWeight','bold');

    % ---- panel 1: convergence ------------------------------------------
    ax = nexttile(tl);  hold(ax,'on');  grid(ax,'on');
    semilogy(ax, 0:numel(P.contrast)-1, P.contrast, 'o-', ...
        'DisplayName','polarization only (mono)');
    semilogy(ax, 0:numel(Bc)-1, Bc, 's-', 'DisplayName','10% band only');
    it0 = 0:numel(BP.contrast)-1;
    it1 = numel(BP.contrast)-1 + (0:numel(BPr.contrast)-1);
    semilogy(ax, it0, BP.contrast, 'd-', 'DisplayName','band + polarization');
    semilogy(ax, it1, BPr.contrast, 'd--', ...
        'DisplayName','band + pol, re-measured matrix');
    yline(ax, 6.78e-15, ':', 'monochromatic floor 6.8\times10^{-15}', ...
        'LabelHorizontalAlignment','left', 'FontSize', 8, ...
        'HandleVisibility','off');
    set(ax, 'YScale','log');
    xlabel(ax, 'EFC iteration'); ylabel(ax, 'dark-zone mean contrast');
    legend(ax, 'Location','northeast', 'FontSize', 8);
    title(ax, 'Lyot 0.60 (36% throughput)');

    % ---- panel 2: rebalanced Lyot --------------------------------------
    ax = nexttile(tl);  hold(ax,'on');  grid(ax,'on');
    T = 100 * [0.36 0.49 0.64];
    C = [BP.c_after L70.c_after L80.c_after];
    fr = [0.60 0.70 0.80];
    semilogy(ax, T, C, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7, ...
        'DisplayName', 'band + pol floor (fixed matrix)');
    for k = 1:3
        text(ax, T(k), C(k)*1.5, sprintf('Lyot %.2f', fr(k)), ...
            'FontSize', 8, 'HorizontalAlignment','center', 'Color',[.35 .35 .35]);
    end
    semilogy(ax, 100*0.36, BPr.c_after, 'p', 'MarkerSize', 12, ...
        'MarkerFaceColor', [0.85 0.85 0.85], 'Color', [0.1 0.1 0.1], ...
        'DisplayName', 'with re-measured matrix');
    yline(ax, 3.82e-9, '--', 'hard occulter, mono, re-measured: 3.8\times10^{-9}', ...
        'LabelHorizontalAlignment','left', 'FontSize', 8, ...
        'HandleVisibility','off');
    set(ax, 'YScale','log');
    xlabel(ax, 'throughput (%)');
    ylabel(ax, 'dark-zone mean contrast (10% band, unpolarized)');
    legend(ax, 'Location','southeast', 'FontSize', 8);
    title(ax, 'Rebalancing the Lyot under full physics');

    fp = fullfile(opts.outdir, 'ctb_phys_summary.png');
    exportgraphics(fig, fp, 'Resolution', 150);
    close(fig);
    fprintf('[physfig] wrote %s\n', fp);
    out = struct('figure', fp, 'floors', struct( ...
        'pol_mono', P.c_after, 'band', 3*B.c_after, 'bandpol', BP.c_after, ...
        'bandpol_relin', BPr.c_after, 'L70', L70.c_after, 'L80', L80.c_after, ...
        'pol_floor', BPr.pol_floor));
end
