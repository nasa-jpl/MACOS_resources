function OUT = cf1_families(over)
%CF1_FAMILIES  Coronagraph-family campaign S1: the head-to-head, PRE-CONTROL.
%
%   The bench_ctb slide-5 replay (ctb_mask_compare) on the e2e6m
%   segmented train: every literature mask family through ONE runner
%   (cf_chain), the SAME deck, grid, dark-zone annulus and Strehl
%   normalisation, at band center.  All numbers are OPEN LOOP -- the DMs
%   are flat; S2 adds the closed-loop column.
%
%   Rows (parameters stated per row; HLC is a recorded DEFERRAL, see
%   below):
%     classical Lyot     hard occulter 2.8 lambda/D + Lyot 0.50, no
%                        apodizer -- the pure-occulter baseline.
%     apodized Lyot      clear-disc prolate + occulter + Lyot 0.90 --
%                        the R1 baseline re-expressed through cf_chain
%                        (the prolate is designed for a CLEAR disc and
%                        merely laid over the gapped pupil).
%     APLC               the prolate solved ON the traced gapped pupil
%                        (ctb_apod_prolate 'support'; N'Diaye, Zimmerman
%                        & Soummer 2016) + occulter + Lyot 0.90 -- the
%                        aperture-matched co-design.
%     band-limited       Kuchner & Traub 2002 order-4, eps 0.40,
%                        separable, Lyot (1-eps) = 0.60.
%     vortex c4 / c6     scalar vortex (8x complex-binned), Lyot 0.60
%                        (the CTB study operating point); the segment
%                        gaps break the ideal-vortex null -- the point
%                        of the row is to MEASURE the leak.  NOTE: this
%                        train is UNOBSCURED -- no secondary -- so the
%                        leak is gap-driven only.
%
%   HYBRID LYOT -- DEFERRED, the bench_ctb SESSION-6 ruling verbatim:
%   the HLC's metal+dielectric complex FPM is a CO-OPTIMIZATION product
%   of the FALCO/EFC design loop, not a closed-form profile; shipping a
%   parameterized placeholder un-validated against a published design
%   would put a fabricated row in the table.  Recorded here and in the
%   deck; it enters when FALCO is wired in.
%
%   OUT = CF1_FAMILIES()      defaults
%   OUT = CF1_FAMILIES(OVER)  with e2e6m_r2_params overrides
%
%   See also CF_CHAIN, CF0_GATES, ctb_mask_compare, ctb_apod_prolate.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    rx = fullfile(P.outdir, 'r1_seg_prop.in');

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m CF1 -- mask families, head-to-head (PRE-CONTROL)');
    L = say_(L, 'deck %s', rx);
    L = say_(L, 'model %d, %g nm (band center), annulus %g-%g lambda/D, Strehl-normalised', ...
             P.co.model, P.lambda_m*1e9, P.co.inner_lamD, P.co.outer_lamD);
    L = say_(L, 'ALL NUMBERS PRE-CONTROL: DMs flat, open loop (S2 adds the closed-loop column)');

    F = struct( ...
      'key',  {'hard',            'apl',                'aplc',               'blc',              'v4',            'v6'}, ...
      'name', {'classical Lyot',  'apodized Lyot (R1)', 'APLC (ap.-matched)', 'band-limited 4th', 'vortex chg 4',  'vortex chg 6'}, ...
      'cfg',  { ...
        {'apod_kind','none',        'fpm_kind','hard',   'r_fpm_lamD',P.co.r_occ_lamD, 'r_lyot_frac',0.50}, ...
        {'apod_kind','prolate',     'fpm_kind','hard',   'r_fpm_lamD',P.co.r_occ_lamD, 'r_lyot_frac',P.co.r_lyot_frac}, ...
        {'apod_kind','prolate_seg', 'fpm_kind','hard',   'r_fpm_lamD',P.co.r_occ_lamD, 'r_lyot_frac',P.co.r_lyot_frac}, ...
        {'apod_kind','none',        'fpm_kind','blc',    'blc_eps',0.40, 'blc_order',4, 'r_lyot_frac',0.60}, ...
        {'apod_kind','none',        'fpm_kind','vortex', 'charge',4,     'r_lyot_frac',0.60}, ...
        {'apod_kind','none',        'fpm_kind','vortex', 'charge',6,     'r_lyot_frac',0.60}}, ...
      'note', { ...
        'hard occulter + Lyot 0.50, no apodizer', ...
        'clear-disc prolate over the gapped pupil + Lyot 0.90 (= R1)', ...
        'prolate solved ON the traced gapped pupil + Lyot 0.90', ...
        'K&T 2002 order-4, \epsilon=0.40, Lyot 0.60', ...
        'scalar vortex, 8x-binned, Lyot 0.60', ...
        'scalar vortex, 8x-binned, Lyot 0.60'}, ...
      'res',  {[],[],[],[],[],[]});

    for k = 1:numel(F)
        L = say_(L, '\n---- %s ----', F(k).name);
        ch = cf_chain('rx', rx, 'model_size', P.co.model, ...
                      'prolate_iter', P.co.prolate_iter, F(k).cfg{:});
        E  = ch.run();
        I  = abs(E).^2;
        dz = macos.dark_zone_metrics(I, ch.peak_bare, ch.lamD_px, ...
                                     P.co.inner_lamD, P.co.outer_lamD);
        supp = ch.peak_bare / max(max(I(:)), eps);
        [rr, cc] = macos.radial_contrast(I, ch.peak_bare, ch.lamD_px, ...
                                         P.co.outer_lamD + 3);
        w = round(2*(P.co.outer_lamD+3)*ch.lamD_px);
        F(k).res = struct('tag', ch.tag, 'config', {ch.config}, ...
            'dz', dz, 'supp', supp, 'thru', ch.thru, ...
            'thru_apod', ch.thru_apod, 'lamD_px', ch.lamD_px, ...
            'peak_bare', ch.peak_bare, 'rr', rr, 'cc', cc, ...
            'thumb', crop_(log10(max(I/ch.peak_bare, 1e-14)), w), ...
            'prolate_info', ch.prolate_info);
        L = say_(L, '    tag %s | DZ mean %.3e median %.3e | suppr %.3e | thru %.3f', ...
                 ch.tag, dz.mean, dz.median, supp, ch.thru);
        if strcmp(F(k).key, 'apl')          % consistency thread back to R1
            R1 = load(fullfile(P.outdir,'r1_coro_run.mat'));
            r1m = R1.OUT.V(strcmp({R1.OUT.V.tag},'seg')).res.dz_aplc.mean;
            L = say_(L, '    vs R1 committed %.3e (rel %.3g) [consistency thread]', ...
                     r1m, abs(dz.mean-r1m)/r1m);
        end
    end

    % ---- the table ------------------------------------------------------
    L = say_(L, '\n==== the S1 table (PRE-CONTROL, DMs flat; annulus %g-%g lambda/D) ====', ...
             P.co.inner_lamD, P.co.outer_lamD);
    L = say_(L, '  %-20s | %-10s | %-10s | %-9s | %-6s | %s', ...
             'family', 'DZ mean', 'DZ median', 'suppress', 'thru', 'note');
    L = say_(L, '  %s', repmat('-', 1, 108));
    for k = 1:numel(F)
        L = say_(L, '  %-20s | %.3e | %.3e | %.3e | %5.1f%% | %s', ...
                 F(k).name, F(k).res.dz.mean, F(k).res.dz.median, ...
                 F(k).res.supp, 100*F(k).res.thru, F(k).note);
    end
    L = say_(L, '  %-20s | %s', 'hybrid Lyot', ...
             'DEFERRED -- FALCO co-design product, no validated closed form (SESSION-6 ruling)');
    L = say_(L, '  %s', repmat('-', 1, 108));
    L = say_(L, '  (throughput = off-axis proxy: apodizer Phi^2-fill x Lyot area');
    L = say_(L, '   x (1-eps)^2 for the BLC -- the ctb_mask_compare convention)');
    L = say_(L, '  This train is UNOBSCURED: vortex leak is segment-gap-driven only.');

    % ---- figures --------------------------------------------------------
    png1 = fullfile(P.outdir, 'cf1_families.png');
    fig_summary_(F, P, png1);
    png2 = fullfile(P.outdir, 'cf1_radial.png');
    fig_radial_(F, P, png2);
    L = say_(L, '\n  figures: %s | %s', png1, png2);

    L = say_(L, '\nCF1 DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf1_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);

    % prune the thumbnails' source arrays are already cropped; keep them
    OUT = struct('P',P, 'F',F, 'text',txt, 'figures',{{png1,png2}}, ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf1_run.mat'),'OUT');
end

% =========================================================================
function fig_summary_(F, P, png)
    f = figure('Visible','off','Color','w','Position',[40 40 1760 940]);
    set(f,'DefaultAxesFontSize',16,'DefaultTextFontSize',16);
    tl = tiledlayout(f, 2, 5, 'TileSpacing','compact', 'Padding','compact');
    title(tl, sprintf(['e2e6m segmented train -- coronagraph mask families, ' ...
        'PRE-CONTROL (DMs flat)\nannulus %g-%g \\lambda/D, %g nm, N=%d'], ...
        P.co.inner_lamD, P.co.outer_lamD, P.lambda_m*1e9, P.co.model), ...
        'FontWeight','bold', 'Interpreter','tex', 'FontSize',20);

    ax = nexttile(tl, [2 2]); hold(ax,'on'); set(ax,'YScale','log');
    cols = lines(numel(F));
    yv = arrayfun(@(k) F(k).res.dz.mean, 1:numel(F));
    ylim(ax, [min(yv)/2.5, max(yv)*2.5]);
    yl = ylim(ax);
    % stagger labels of near-coincident points (small log-y dodge, clamped
    % inside the axis so no label detaches from its marker or the box)
    yoff = ones(1, numel(F));
    [~, ord] = sort(yv);
    for q = 2:numel(ord)
        if yv(ord(q)) / yv(ord(q-1)) < 1.6
            yoff(ord(q))   = 1.18;
            yoff(ord(q-1)) = 1/1.18;
        end
    end
    for k = 1:numel(F)
        plot(ax, 100*F(k).res.thru, F(k).res.dz.mean, 'o', 'MarkerSize',14, ...
             'MarkerFaceColor',cols(k,:), 'MarkerEdgeColor','k', 'LineWidth',1.1);
        ylab = min(max(yv(k) * yoff(k), yl(1)*1.25), yl(2)/1.25);
        text(ax, 100*F(k).res.thru+2.0, ylab, F(k).name, ...
             'FontSize',15, 'FontWeight','bold');
    end
    grid(ax,'on'); box(ax,'on'); xlim(ax,[0 100]);
    xlabel(ax,'off-axis throughput proxy (%)');
    ylabel(ax,'mean dark-zone contrast (PRE-CONTROL)');
    title(ax,'lower-right is better');

    for k = 1:numel(F)
        ax = nexttile(tl);
        imagesc(ax, F(k).res.thumb); axis(ax,'image','off');
        colormap(ax, parula); clim(ax, [-10 0]);
        cb = colorbar(ax); cb.Label.String = 'log_{10} contrast';
        title(ax, F(k).name, 'FontSize',15);
    end
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function fig_radial_(F, P, png)
    f = figure('Visible','off','Color','w','Position',[80 80 980 640]);
    ax = axes(f); hold(ax,'on'); set(ax,'YScale','log');
    cols = lines(numel(F));
    h = gobjects(1, numel(F));
    for k = 1:numel(F)
        h(k) = plot(ax, F(k).res.rr, max(F(k).res.cc, 1e-14), '-', ...
                    'Color', cols(k,:), 'LineWidth', 1.7);
    end
    yl = ylim(ax);
    p = patch(ax, [P.co.inner_lamD P.co.outer_lamD P.co.outer_lamD P.co.inner_lamD], ...
              [yl(1) yl(1) yl(2) yl(2)], [0.90 0.90 0.95], ...
              'FaceAlpha',0.45, 'EdgeColor','none', 'HandleVisibility','off');
    uistack(p, 'bottom');
    grid(ax,'on'); box(ax,'on');
    xlabel(ax,'separation  [\lambda/D]');  ylabel(ax,'contrast');
    legend(ax, h, arrayfun(@(k) sprintf('%s  (%.1e)', F(k).name, ...
           F(k).res.dz.mean), 1:numel(F), 'UniformOutput', false), ...
           'Location','northeastoutside');
    title(ax, sprintf(['mask families on the segmented train -- radial contrast, ' ...
        'PRE-CONTROL\n(dark zone %g-%g \\lambda/D shaded)'], ...
        P.co.inner_lamD, P.co.outer_lamD), 'Interpreter','tex');
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function o = crop_(img, w)
    n = size(img,1); if w >= n, o = img; return; end
    c = floor(n/2)+1; lo = max(c-floor(w/2),1); hi = min(lo+w-1,n); o = img(lo:hi,lo:hi);
end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
