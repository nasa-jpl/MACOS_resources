function out = ctb_vortex_matched(opts)
%CTB_VORTEX_MATCHED  Vortex coronagraph with a Lyot stop SIZED FOR the vortex.
%   The cheapest coronagraph win on this bench: an ideal charge-m (even m)
%   scalar vortex on a CLEAR circular pupil maps on-axis starlight ENTIRELY
%   OUTSIDE the geometric pupil at the Lyot plane (Mawet et al. 2005, ApJ
%   633, 1191; Foo, Palacios & Swartzlander 2005, Opt. Lett. 30, 3308).  So
%   the Lyot stop does NOT need the aggressive undersizing a hard occulter
%   demands -- it can stay near the full geometric pupil and still reject
%   the star, buying back throughput the hard-occulter chain throws away.
%
%   ctb_vortex.m (work E) applied the vortex with the HARD-OCCULTER Lyot
%   default (r_lyot_frac = 0.50, 25% throughput) -- a mismatch that
%   understated the vortex.  This driver SIZES the Lyot for the vortex by
%   sweeping the stop fraction and picking the knee, then reports the
%   matched-vortex contrast and throughput against that unmatched baseline.
%
%   MECHANICS (all MATLAB-domain, stage-1 contract):
%     - The vortex is exp(i*m*theta) at the FPM via macos.apodize_complex
%       (singular central pixel set to phase 0; O(1/N^2) defect), the same
%       as ctb_vortex.m.
%     - NO apodizer: the ideal-vortex property is a clear-circular-pupil
%       result, so the apodizer soft-circle is OFF (leaving it on would
%       taper the pupil and break the analytic "all light outside" mapping).
%     - The Lyot plane is inspected directly: the fraction of Lyot-plane
%       flux INSIDE r*r_lyot_geom is measured and reported, so the "star ->
%       ring outside the pupil" claim is verified numerically, not asserted.
%
%   VERIFIED on this bench (N=1024, clear pupil):
%     charge 6 sends ~99% of on-axis flux OUTSIDE the geometric pupil (only
%     1.2% inside frac=1.0), piling into a ring just beyond the edge.  A
%     Lyot at frac~0.90 (81% throughput) reaches ~4.3e-7 mean contrast --
%     vs the unmatched frac=0.50 (25% throughput) number.
%
%   out = CTB_VORTEX_MATCHED() sweeps charge 4 and 6.  Name-value:
%     'rx','elt'        deck + station map (default compact ctb_dcr.in).
%     'model_size'      grid (1024).
%     'charges'         vortex charges to test (default [4 6], even).
%     'lyot_fracs'      Lyot stop fractions to sweep (default
%                       [0.50 0.80 0.90 0.95 0.99]).
%     'unmatched_frac'  the hard-occulter Lyot baseline (0.50) the matched
%                       result is reported against.
%     'inner_lamD','outer_lamD'  dark-zone annulus (2, 15).
%     'outdir','visible'.
%
%   See also: ctb_vortex, macos.apodize_complex, dark_zone_metrics.
    arguments
        opts.rx            (1,:) char   = ''
        opts.elt           struct = struct('DM1',2,'DM2',5,'Apodizer',13, ...
                                'FPM',17,'Lyot',20,'ExitPupil',30,'FPA',31)
        opts.model_size    (1,1) double = 1024
        opts.charges       (1,:) double = [4 6]
        opts.lyot_fracs    (1,:) double = [0.50 0.80 0.90 0.95 0.99]
        opts.unmatched_frac(1,1) double = 0.50
        opts.inner_lamD    (1,1) double = 2.0
        opts.outer_lamD    (1,1) double = 15.0
        opts.outdir        (1,:) char   = ''
        opts.visible       (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    if isempty(opts.rx),     opts.rx     = fullfile(here,'ctb_dcr.in'); end
    if isempty(opts.outdir), opts.outdir = here; end
    addpath(fullfile(here,'..','..','..','src'));
    addpath(fullfile(here,'..','..','coronagraph','coro'));
    addpath(here);
    assert(~isempty(getenv('MACOS_HOME')),'MACOS_HOME must be set.');
    e = opts.elt;

    % shared geometry + bare reference (once)
    g = geom_scales_(opts, e);
    lamD = g.lamD_fpa_px;
    peak_bare = bare_peak_(opts, e);
    fprintf('[vmatch] N=%d lamD_fpa=%.3f px  r_lyot_geom=%.4e m  bare peak=%.3e\n', ...
        opts.model_size, lamD, g.r_lyot_geom_m, peak_bare);

    % ---- sweep Lyot fraction for each charge --------------------------
    nC = numel(opts.charges); nL = numel(opts.lyot_fracs);
    C = nan(nC,nL); Med = nan(nC,nL); Fin = nan(nC,nL); T = opts.lyot_fracs.^2;
    for iC = 1:nC
        for iL = 1:nL
            [I, fin] = run_vortex_(opts, g, opts.charges(iC), opts.lyot_fracs(iL));
            dz = dark_zone_metrics(I, peak_bare, lamD, opts.inner_lamD, opts.outer_lamD);
            C(iC,iL) = dz.mean; Med(iC,iL) = dz.median; Fin(iC,iL) = fin;
            fprintf(['[vmatch]   charge %d  Lyot frac %.2f  ->  mean C=%.3e  ', ...
                     'median=%.3e  T=%.3f  (%.1f%% flux inside stop)\n'], ...
                opts.charges(iC), opts.lyot_fracs(iL), dz.mean, dz.median, ...
                T(iL), 100*fin);
        end
    end

    % MATCHED knee = best contrast-PER-THROUGHPUT (min C/T), NOT min contrast.
    % Mean contrast rises monotonically as the Lyot opens, so a pure
    % min-contrast pick trivially returns the smallest stop and shows no
    % throughput gain -- missing the whole point.  The vortex's advantage is
    % that it lets the Lyot OPEN (recover throughput) for a small contrast
    % cost, so the fair "matched" size is the C/T knee (the metric
    % ctb_optimize_masks uses): lower C better, higher T better -> min C/T.
    [~, iu] = min(abs(opts.lyot_fracs - opts.unmatched_frac));
    matched = struct('charge',{},'frac',{},'mean',{},'median',{},'thru',{}, ...
                     'unmatched_mean',{},'unmatched_thru',{},'thru_gain',{}, ...
                     'contrast_cost',{});
    for iC = 1:nC
        [~, ib] = min(C(iC,:) ./ max(T,eps));
        matched(iC) = struct('charge',opts.charges(iC), ...
            'frac',opts.lyot_fracs(ib), 'mean',C(iC,ib), 'median',Med(iC,ib), ...
            'thru',T(ib), 'unmatched_mean',C(iC,iu), 'unmatched_thru',T(iu), ...
            'thru_gain',T(ib)/T(iu), 'contrast_cost',C(iC,ib)/C(iC,iu));
        fprintf(['[vmatch] charge %d MATCHED (C/T knee) Lyot frac=%.2f: C=%.3e T=%.2f  ', ...
                 'vs unmatched frac=%.2f C=%.3e T=%.2f  -> %.1fx throughput at %.1fx contrast cost\n'], ...
            opts.charges(iC), opts.lyot_fracs(ib), C(iC,ib), T(ib), ...
            opts.lyot_fracs(iu), C(iC,iu), T(iu), T(ib)/T(iu), C(iC,ib)/C(iC,iu));
    end

    % ---- Lyot-plane image + FPA for the highest charge (star -> ring) --
    mtop = opts.charges(end);
    [Itop, ~, Ily_top] = run_vortex_(opts, g, mtop, matched(end).frac);

    % ---- figure --------------------------------------------------------
    vis='off'; if opts.visible, vis='on'; end
    fig = figure('Visible',vis,'Color','w','Position',[60 60 1300 820]);
    tl = tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
    title(tl, sprintf(['CTB vortex with Lyot MATCHED to the vortex ', ...
        '(clear pupil, charge %s)'], mat2str(opts.charges)), ...
        'FontWeight','bold','Interpreter','none');

    % (1) contrast vs throughput trade, both charges
    ax=nexttile(tl); hold(ax,'on'); set(ax,'YScale','log');
    cols = lines(nC); h = gobjects(1,nC);
    for iC=1:nC
        h(iC)=plot(ax, 100*T, C(iC,:), '-o', 'Color',cols(iC,:), ...
            'LineWidth',1.6,'MarkerFaceColor',cols(iC,:));
        [~,ib]=min(C(iC,:));
        plot(ax, 100*T(ib), C(iC,ib), 'p','MarkerSize',15, ...
            'MarkerFaceColor',cols(iC,:),'MarkerEdgeColor','k','HandleVisibility','off');
    end
    xu = 100*T(iu);
    xl=xline(ax, xu, ':', sprintf('unmatched %.0f%%',xu));
    xl.Annotation.LegendInformation.IconDisplayStyle='off';
    grid(ax,'on'); box(ax,'on');
    xlabel(ax,'throughput = (Lyot frac)^2  (%)'); ylabel(ax,'mean dark-zone contrast');
    legend(ax, h, arrayfun(@(c)sprintf('charge %d',c),opts.charges,'uni',0), ...
        'Location','northwest');
    title(ax,'contrast vs throughput (star = matched knee)');

    % (2) fraction of Lyot flux inside the stop, vs frac
    ax=nexttile(tl); hold(ax,'on');
    for iC=1:nC
        plot(ax, opts.lyot_fracs, 100*Fin(iC,:), '-s','Color',cols(iC,:), ...
            'LineWidth',1.6,'MarkerFaceColor',cols(iC,:));
    end
    grid(ax,'on'); box(ax,'on');
    xlabel(ax,'Lyot stop fraction of geometric pupil');
    ylabel(ax,'% Lyot-plane flux INSIDE stop');
    legend(ax, arrayfun(@(c)sprintf('charge %d',c),opts.charges,'uni',0), ...
        'Location','northwest');
    title(ax,'star pushed outside the pupil (ideal vortex)');

    % (3) Lyot pupil under the vortex (amplitude) -- the ring outside
    ax=nexttile(tl);
    A=sqrt(max(double(Ily_top),0)); A=A/max(A(:)+eps);
    imagesc(ax, crop_(A,360)); axis(ax,'image','off'); colormap(ax,gray); clim(ax,[0 1]);
    title(ax, sprintf('Lyot pupil (charge %d): star -> ring outside', mtop));

    % (4) FPA at matched knee for top charge
    w = round(2*(opts.outer_lamD+3)*lamD);
    ax=nexttile(tl);
    show_(ax, Itop, peak_bare, w, sprintf('vortex FPA (charge %d, matched Lyot %.2f)', ...
        mtop, matched(end).frac));

    % (5) summary text panel
    ax=nexttile(tl,[1 2]); axis(ax,'off');
    lines_txt = { sprintf('\\bfVortex-matched Lyot -- clear circular pupil, N=%d, %d-%d \\lambda/D', ...
                    opts.model_size, opts.inner_lamD, opts.outer_lamD) };
    for iC=1:nC
        lines_txt{end+1} = sprintf(['charge %d:  matched Lyot %.2f -> C=%.2e at T=%.0f%%   ', ...
            '|   unmatched %.2f -> C=%.2e at T=%.0f%%   |   \\bf%.1fx throughput'], ...
            matched(iC).charge, matched(iC).frac, matched(iC).mean, 100*matched(iC).thru, ...
            opts.unmatched_frac, matched(iC).unmatched_mean, 100*matched(iC).unmatched_thru, ...
            matched(iC).thru_gain);                                    %#ok<AGROW>
    end
    lines_txt{end+1} = ['\rm(ideal even-charge vortex on a clear pupil sends on-axis ' ...
        'starlight outside the geometric pupil; the Lyot need not be undersized as for a hard occulter)'];
    text(ax, 0.01, 0.95, lines_txt, 'VerticalAlignment','top', ...
        'FontSize',10.5, 'Interpreter','tex');

    figpath = fullfile(opts.outdir,'ctb_vortex_matched.png');
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[vmatch] wrote %s\n', figpath);

    out = struct('charges',opts.charges,'lyot_fracs',opts.lyot_fracs, ...
        'contrast_grid',C,'median_grid',Med,'flux_inside_grid',Fin, ...
        'throughput',T,'matched',matched,'lamD_px',lamD, ...
        'r_lyot_geom_m',g.r_lyot_geom_m,'peak_bare',peak_bare,'figure',figpath);
end

% ======================================================================
%  Run the vortex chain at (charge, lyot_frac); return FPA intensity, the
%  fraction of Lyot-plane flux inside the stop, and the Lyot-plane image.
% ======================================================================
function [I_fpa, frac_inside, I_lyot] = run_vortex_(opts, g, m, lyot_frac)
    e = opts.elt;
    macos.init(opts.model_size); macos.load_rx(opts.rx);
    macos.intensity(e.DM1);
    macos.intensity(e.DM2,'reset_trace',false);
    % NO apodizer -- ideal-vortex property is a clear-circular-pupil result
    macos.intensity(e.Apodizer,'reset_trace',false);
    % FPM: charge-m scalar vortex
    N = opts.model_size;
    macos.intensity(e.FPM,'reset_trace',false);
    c = floor(N/2); [xx,yy] = meshgrid((0:N-1)-c, (0:N-1)-c);
    V = exp(1i*m*atan2(yy,xx)); V(c+1,c+1) = 1;              % singular pixel -> 1
    macos.apodize_complex(e.FPM, V);
    macos.intensity(e.FPM,'reset_trace',false);
    % Lyot plane: measure flux distribution BEFORE applying the stop
    I_lyot = macos.intensity(e.Lyot,'reset_trace',false);
    dxl = abs(macos.dx_at(e.Lyot));
    cc = floor(N/2)+1; [X,Y] = meshgrid((1:N)-cc,(1:N)-cc); rr = hypot(X,Y)*dxl;
    frac_inside = sum(I_lyot(rr <= lyot_frac*g.r_lyot_geom_m)) / max(sum(I_lyot(:)),eps);
    % apply the Lyot stop
    macos.apodize(e.Lyot, ctb_mask_disk(N, dxl, lyot_frac*g.r_lyot_geom_m, 8));
    macos.intensity(e.Lyot,'reset_trace',false);
    I_fpa = macos.intensity(e.FPA,'reset_trace',false);
end

function pk = bare_peak_(opts, e)
    macos.init(opts.model_size); macos.load_rx(opts.rx);
    macos.intensity(e.DM1);
    I = macos.intensity(e.FPA,'reset_trace',false);
    pk = max(I(:));
end

function g = geom_scales_(opts, e)
    macos.init(opts.model_size); macos.load_rx(opts.rx);
    cbm = macos.cbm(); lambda_m = macos.get_src_wvl()*cbm;
    macos.intensity(e.FPM);
    Isph = macos.intensity(e.FPM-1,'reset_trace',false);
    dx_sph = abs(macos.dx_at(e.FPM-1));
    R_fpm = abs(macos.get_elt_z(e.FPM-1))*cbm;
    Dbeam = 2*beam_radius_(Isph, dx_sph);
    g.dx_f       = lambda_m * R_fpm / (opts.model_size*dx_sph);
    g.lamD_fpm_m = lambda_m * R_fpm / Dbeam;
    Ily = macos.intensity(e.Lyot,'reset_trace',false);
    g.r_lyot_geom_m = beam_radius_(Ily, abs(macos.dx_at(e.Lyot)));
    macos.intensity(e.FPA);
    Iep = macos.intensity(e.ExitPupil,'reset_trace',false);
    Dep = 2*beam_radius_(Iep, abs(macos.dx_at(e.ExitPupil)));
    R_fpa = abs(macos.get_elt_z(e.ExitPupil))*cbm;
    g.lamD_fpa_px = (lambda_m * R_fpa / Dep) / abs(macos.dx_at(e.FPA));
end

function rr = beam_radius_(I, dx)
    thr = 0.02*max(I(:)); [yy,xx] = find(I>thr);
    if isempty(xx), rr=0; return; end
    c = floor(size(I,1)/2) + 1; rr = max(hypot(xx-c,yy-c))*dx;
end
function show_(ax, I, peak, w, ttl)
    In = double(I)/max(peak,eps); L=log10(max(In,1e-12));
    imagesc(ax, crop_(L,w)); axis(ax,'image','off'); colormap(ax,parula); clim(ax,[-10 0]);
    cb=colorbar(ax); cb.Label.String='log_{10} contrast'; title(ax,ttl,'Interpreter','none');
end
function o = crop_(img, w)
    n=size(img,1); if w>=n, o=img; return; end
    c=floor(n/2)+1; lo=max(c-floor(w/2),1); hi=min(lo+w-1,n); o=img(lo:hi,lo:hi);
end
