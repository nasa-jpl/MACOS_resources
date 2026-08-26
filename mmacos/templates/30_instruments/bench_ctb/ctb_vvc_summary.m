function out = ctb_vvc_summary(opts)
%CTB_VVC_SUMMARY  Scalar vs vector vortex: the bandwidth verdict figure.
%   Reads the ctb_vvc_*.mat run states (gitignored; regen lines in
%   CTB_PROP_STATUS SESSION 14) and the scalar bandwidth sweep, and
%   overlays the closed-loop floors on one bandwidth axis:
%
%     - scalar vortex (achromatic angle-map mask): the gentle chromatic
%       floor of slide 12;
%     - zero-order VECTOR vortex, unpolarized: leakage-pinned at
%       cos^2(delta/2) -- the leak amplitude flips sign across band
%       center and a common surface cannot null it (0% point = the
%       ideal-plate two-spiral compromise, 2.0e-9);
%     - zero-order VECTOR vortex in the circular polarizer/analyzer
%       sandwich: the analyzer rejects the leakage OPTICALLY (statics
%       return to the mono value) and the loop digs 37x-4200x beyond
%       the unpolarized chain;
%     - zero-order VECTOR vortex in the crossed LINEAR sandwich (x in,
%       y analyzed): the leakage is co-polarized with the input and
%       equally rejected; star-side floors are circular-class.  The
%       difference sits on the PLANET side -- transmission
%       sin^2(m*theta_p)/2 has 2m azimuthal nulls (8 blind spots at
%       charge 4) where the circular sandwich is flat 1/2;
%     - markers: ideal plate + analyzer (reproduces the scalar loop),
%       the full stacks (10% + coating screens, both sandwiches), and
%       the per-lambda-control recovery point at 5%.
%
%   Asset-gated: errors with the regen instruction when run states are
%   absent (the per-lambda point and the linear ladder are optional --
%   skipped if missing).
%
%   Run:  >> ctb_vvc_summary;
%   See also: ctb_vvc, ctb_vortex_bandwidth, ctb_mask_vvc.
    arguments
        opts.outdir  (1,:) char = ''
        opts.visible (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    if isempty(opts.outdir), opts.outdir = here; end
    need = {'ctb_vvc_ideal.mat','ctb_vvc_ideal_analyzed.mat', ...
            'ctb_vvc_c05.mat','ctb_vvc_c10.mat','ctb_vvc_c20.mat', ...
            'ctb_vvc_circ00.mat','ctb_vvc_circ05.mat','ctb_vvc_circ10.mat', ...
            'ctb_vvc_circ20.mat','ctb_vvc_circ10s.mat', ...
            'ctb_vortex_bandwidth.mat'};
    for k = 1:numel(need)
        assert(isfile(fullfile(here, need{k})), ...
            'ctb_vvc_summary: %s absent -- regen per CTB_PROP_STATUS SESSION 14', need{k});
    end
    L = @(f) load(fullfile(here, f));
    I0   = L('ctb_vvc_ideal.mat');          % unpolarized ideal (compromise)
    IA   = L('ctb_vvc_ideal_analyzed.mat'); % ideal + analyzer (validation)
    Cu   = cellfun(L, {'ctb_vvc_c05.mat','ctb_vvc_c10.mat','ctb_vvc_c20.mat'});
    Cc   = cellfun(L, {'ctb_vvc_circ00.mat','ctb_vvc_circ05.mat', ...
                       'ctb_vvc_circ10.mat','ctb_vvc_circ20.mat'});
    S10  = L('ctb_vvc_circ10s.mat');        % full stack
    SW   = L('ctb_vortex_bandwidth.mat');   % scalar sweep
    linf = {'ctb_vvc_lin00.mat','ctb_vvc_lin05.mat', ...
            'ctb_vvc_lin10.mat','ctb_vvc_lin20.mat'};
    has_lin = all(cellfun(@(f) isfile(fullfile(here, f)), linf));
    if has_lin, Cl = cellfun(L, linf); end

    vis = 'off'; if opts.visible, vis = 'on'; end
    fig = figure('Visible',vis, 'Color','w', 'Position',[80 80 860 580]);
    ax = axes(fig);  hold(ax,'on');  grid(ax,'on');

    semilogy(ax, 100*SW.bands, SW.floors, 'o-', 'LineWidth', 1.6, ...
        'DisplayName', 'scalar vortex (achromatic mask)');
    bwu = [0, 100*[Cu.band]];
    flu = [I0.c_after, [Cu.c_after]];
    semilogy(ax, bwu, flu, 's-', 'LineWidth', 1.6, ...
        'DisplayName', 'vector vortex, zero-order, unpolarized');
    semilogy(ax, 100*[Cc.band], [Cc.c_after], '^-', 'LineWidth', 1.6, ...
        'DisplayName', 'vector vortex in circular polarizer/analyzer sandwich');
    if has_lin
        semilogy(ax, 100*[Cl.band], [Cl.c_after], 'v-', 'LineWidth', 1.6, ...
            'DisplayName', 'crossed linear sandwich (star side; planet has 2m nulls)');
    end
    fls = fullfile(here, 'ctb_vvc_lin10s.mat');
    if isfile(fls)
        LS = load(fls);
        semilogy(ax, 100*LS.band, LS.c_after, 'd', 'MarkerSize', 10, ...
            'MarkerFaceColor', [0.85 0.92 1], 'Color', [0.1 0.1 0.1], ...
            'DisplayName', 'full stack: crossed linear + coating screens (10%)');
    end
    semilogy(ax, 0, IA.c_after, 'p', 'MarkerSize', 13, ...
        'MarkerFaceColor', [0.85 0.85 0.85], 'Color', [0.1 0.1 0.1], ...
        'DisplayName', 'ideal plate + analyzer (= scalar loop)');
    semilogy(ax, 100*S10.band, S10.c_after, 'd', 'MarkerSize', 10, ...
        'MarkerFaceColor', [1 1 1], 'Color', [0.1 0.1 0.1], ...
        'DisplayName', 'full stack: sandwich + coating screens (10%)');
    fpl = fullfile(here, 'ctb_vvc_circ05_perlam.mat');
    if isfile(fpl)
        PL = load(fpl);
        semilogy(ax, 100*PL.band, PL.c_after, 'h', 'MarkerSize', 12, ...
            'MarkerFaceColor', [1 1 0.6], 'Color', [0.1 0.1 0.1], ...
            'DisplayName', 'sandwich + per-\lambda control (5%)');
    end
    for b = 1:numel(Cu)
        text(ax, 100*Cu(b).band, Cu(b).c_after*1.6, ...
            sprintf('leak %.0e', Cu(b).leak_frac), 'FontSize', 8, ...
            'HorizontalAlignment','center', 'Color',[.35 .35 .35]);
    end
    set(ax, 'YScale', 'log');
    xlabel(ax, 'bandwidth (%)');
    ylabel(ax, 'dark-zone mean contrast (3-15 \lambda_0/D), closed loop');
    title(ax, {'Scalar vs vector vortex under bandwidth', ...
        'charge 4, Lyot 0.60, zero-order plate: \delta(\lambda) = \pi\lambda_0/\lambda'}, ...
        'FontWeight','bold');
    legend(ax, 'Location', 'southeast', 'FontSize', 8);
    fp = fullfile(opts.outdir, 'ctb_vvc_summary.png');
    exportgraphics(fig, fp, 'Resolution', 150);
    close(fig);
    fprintf('[vvcfig] wrote %s\n', fp);
    out = struct('figure', fp);
end
