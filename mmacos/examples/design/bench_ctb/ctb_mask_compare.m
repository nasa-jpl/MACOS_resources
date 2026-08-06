function out = ctb_mask_compare(opts)
%CTB_MASK_COMPARE  Head-to-head comparison of the CTB coronagraph mask kinds.
%   Runs every literature focal-plane-mask family on the SAME CTB deck, the
%   SAME grid, the SAME dark-zone annulus, and the SAME Strehl normalisation
%   (bare on-axis peak), so their dark-zone contrast and throughput are
%   apples-to-apples.  One ROW per mask kind (the work-order deliverable):
%
%     hard occulter   -- 2.70 lambda/D opaque disk + soft-circle apodizer
%                        + Lyot 0.50 (the ctb_coro_compare baseline).
%     band-limited     -- Kuchner & Traub 2002 order-4 mask, Lyot trimmed
%                        (1-eps); clear pupil.
%     APLC             -- Soummer 2005 prolate apodizer + 2.8 lambda/D hard
%                        occulter + near-full Lyot.
%     vortex (matched) -- charge-6 scalar vortex, Lyot matched to the vortex.
%     Roddier pi-mask  -- Roddier & Roddier 1997, 0.53 lambda/D pi spot.
%     dual-zone        -- N'Diaye 2012 achromatic dual-zone phase mask.
%
%   THROUGHPUT is the off-axis (planet) amplitude throughput proxy each
%   family reports: hard/BLC/vortex = Lyot-area (times (1-eps)^2 or apodizer
%   fill where applicable); APLC = apodizer Phi^2-fill * Lyot area.  These
%   are first-order proxies (the true value integrates the off-axis PSF core
%   through the masks); adequate for the comparison ranking.
%
%   out = CTB_MASK_COMPARE() writes a table (stdout + .mat) and a summary
%   figure (contrast-vs-throughput scatter + FPA thumbnails).  Name-value:
%     'rx','elt','model_size' (1024), 'inner_lamD','outer_lamD' (3,15),
%     'outdir','visible'.  Per-family params use each family's defaults.
%
%   See also: ctb_bandlimited, ctb_aplc, ctb_vortex_matched, ctb_phase_masks,
%             ctb_coro_compare, dark_zone_metrics.
    arguments
        opts.rx            (1,:) char   = ''
        opts.elt           struct = struct('DM1',2,'DM2',5,'Apodizer',13, ...
                                'FPM',17,'Lyot',20,'ExitPupil',30,'FPA',31)
        opts.model_size    (1,1) double = 1024
        opts.inner_lamD    (1,1) double = 3.0
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

    g = geom_scales_(opts, e);
    lamD = g.lamD_fpa_px;
    peak_bare = bare_peak_(opts, e);
    N = opts.model_size;
    fprintf('[cmp] N=%d lamD_fpa=%.3f px  annulus %.0f-%.0f lam/D  bare peak=%.3e\n', ...
        N, lamD, opts.inner_lamD, opts.outer_lamD, peak_bare);

    % ---- run each family ----------------------------------------------
    rows = struct('name',{},'family',{},'I',{},'dz',{},'supp',{},'thru',{},'note',{});

    % 1) hard occulter (the ctb_coro_compare baseline)
    [I,thru] = arm_hard_(opts, g);
    rows(end+1) = mkrow_('hard occulter','hard',I,peak_bare,lamD,opts,thru, ...
        '2.70 \lambda/D opaque + soft apod + Lyot 0.50');

    % 2) band-limited Lyot (K&T order 4, eps=0.40)
    [I,thru] = arm_blc_(opts, g, 0.40);
    rows(end+1) = mkrow_('band-limited','blc',I,peak_bare,lamD,opts,thru, ...
        'K&T 2002 order-4, \epsilon=0.40, Lyot (1-\epsilon)');

    % 3) APLC (prolate + 2.8 lam/D occulter + Lyot 0.90)
    [I,thru] = arm_aplc_(opts, g, 2.8, 0.90);
    rows(end+1) = mkrow_('APLC','aplc',I,peak_bare,lamD,opts,thru, ...
        'Soummer 2005 prolate + 2.8 \lambda/D occulter');

    % 4) vortex matched (charge 6, Lyot 0.90)
    [I,thru] = arm_vortex_(opts, g, 6, 0.90);
    rows(end+1) = mkrow_('vortex (matched)','vortex',I,peak_bare,lamD,opts,thru, ...
        'charge-6, Lyot 0.90 (matched)');

    % 5) Roddier pi-mask (0.53 lam/D)
    [I,thru] = arm_phase_(opts, g, 'roddier', 0.90);
    rows(end+1) = mkrow_('Roddier \pi-mask','roddier',I,peak_bare,lamD,opts,thru, ...
        'R&R 1997, 0.53 \lambda/D \pi spot');

    % 6) dual-zone phase mask
    [I,thru] = arm_phase_(opts, g, 'dualzone', 0.90);
    rows(end+1) = mkrow_('dual-zone','dualzone',I,peak_bare,lamD,opts,thru, ...
        'N''Diaye 2012 achromatic DZPM');

    % ---- print the table ----------------------------------------------
    fprintf('\n');
    fprintf('  %-18s | %-10s | %-10s | %-10s | %-8s\n', ...
        'mask kind','DZ mean','DZ median','suppress','thruput');
    fprintf('  %s\n', repmat('-',1,72));
    for k = 1:numel(rows)
        fprintf('  %-18s | %.3e | %.3e | %.2e | %6.1f%%\n', ...
            rows(k).name, rows(k).dz.mean, rows(k).dz.median, rows(k).supp, 100*rows(k).thru);
    end
    fprintf('  %s\n', repmat('-',1,72));
    fprintf('  (annulus %.0f-%.0f lam/D, Strehl-normalised to bare peak; ', ...
        opts.inner_lamD, opts.outer_lamD);
    fprintf('throughput = off-axis proxy)\n\n');

    % ---- summary figure -----------------------------------------------
    vis='off'; if opts.visible, vis='on'; end
    fig = figure('Visible',vis,'Color','w','Position',[40 40 1760 940]);
    set(fig,'DefaultAxesFontSize',18,'DefaultTextFontSize',18);   % readable when shrunk onto a slide
    tl = tiledlayout(fig,2,4,'TileSpacing','compact','Padding','compact');
    title(tl, sprintf(['CTB coronagraph mask families -- contrast vs throughput ', ...
        '(annulus %.0f-%.0f \\lambda/D, N=%d)'], opts.inner_lamD, opts.outer_lamD, N), ...
        'FontWeight','bold','Interpreter','tex','FontSize',22);

    % big scatter panel (contrast vs throughput), lower-left = better
    ax=nexttile(tl,[2 2]); hold(ax,'on'); set(ax,'YScale','log','FontSize',18);
    cols = lines(numel(rows));
    for k=1:numel(rows)
        plot(ax, 100*rows(k).thru, rows(k).dz.mean, 'o', 'MarkerSize',16, ...
            'MarkerFaceColor',cols(k,:),'MarkerEdgeColor','k','LineWidth',1.2);
        text(ax, 100*rows(k).thru+1.5, rows(k).dz.mean, rows(k).name, ...
            'FontSize',17,'FontWeight','bold','Interpreter','tex');
    end
    grid(ax,'on'); box(ax,'on'); xlim(ax,[0 108]);
    xlabel(ax,'off-axis throughput (%)','FontSize',19);
    ylabel(ax,'mean dark-zone contrast','FontSize',19);
    title(ax,'deeper + more throughput = lower-right is better','Interpreter','none','FontSize',19);

    % FPA thumbnails
    w = round(2*(opts.outer_lamD+3)*lamD);
    for k=1:numel(rows)
        if k>4, continue; end                            % 4 thumbnail slots
        ax=nexttile(tl);
        show_(ax, rows(k).I, peak_bare, w, rows(k).name);
    end

    figpath = fullfile(opts.outdir,'ctb_mask_compare.png');
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[cmp] wrote %s\n', figpath);

    % ---- save the table (.mat) ----------------------------------------
    tbl = struct('name',{ {rows.name} }, 'family',{ {rows.family} }, ...
        'dz_mean',[cellfun(@(d)d.mean,{rows.dz})], ...
        'dz_median',[cellfun(@(d)d.median,{rows.dz})], ...
        'suppression',[rows.supp], 'throughput',[rows.thru], ...
        'note',{ {rows.note} });
    matpath = fullfile(opts.outdir,'ctb_mask_compare.mat');
    save(matpath,'tbl','lamD','peak_bare','opts');
    fprintf('[cmp] wrote %s\n', matpath);

    out = struct('rows',rows,'table',tbl,'lamD_px',lamD,'peak_bare',peak_bare, ...
        'figure',figpath,'mat',matpath);
end

% ======================================================================
%  Per-family arms (each returns FPA intensity + off-axis throughput proxy).
% ======================================================================
function [I,thru] = arm_hard_(opts, g)
    e = opts.elt; N = opts.model_size;
    macos.init(N); macos.load_rx(opts.rx);
    macos.intensity(e.DM1); macos.intensity(e.DM2,'reset_trace',false);
    Ia = macos.intensity(e.Apodizer,'reset_trace',false);
    macos.apodize(e.Apodizer, ctb_mask_softcircle(size(Ia,1),abs(macos.dx_at(e.Apodizer)),15e-3,2e-3,8));
    macos.intensity(e.Apodizer,'reset_trace',false);
    If = macos.intensity(e.FPM,'reset_trace',false);
    macos.apodize(e.FPM, 1 - ctb_mask_disk(size(If,1), g.dx_f, 2.70*g.lamD_fpm_m, 8));
    macos.intensity(e.FPM,'reset_trace',false);
    Il = macos.intensity(e.Lyot,'reset_trace',false);
    macos.apodize(e.Lyot, ctb_mask_disk(size(Il,1),abs(macos.dx_at(e.Lyot)),0.50*g.r_lyot_geom_m,8));
    macos.intensity(e.Lyot,'reset_trace',false);
    I = macos.intensity(e.FPA,'reset_trace',false); thru = 0.50^2;
end

function [I,thru] = arm_blc_(opts, g, eps)
    e = opts.elt; N = opts.model_size;
    macos.init(N); macos.load_rx(opts.rx);
    macos.intensity(e.DM1); macos.intensity(e.DM2,'reset_trace',false);
    macos.intensity(e.Apodizer,'reset_trace',false);
    macos.intensity(e.FPM,'reset_trace',false);
    macos.apodize(e.FPM, ctb_mask_bandlimited(N, g.dx_f, g.lamD_fpm_m, eps, 4, 'separable'));
    macos.intensity(e.FPM,'reset_trace',false);
    Il = macos.intensity(e.Lyot,'reset_trace',false);
    macos.apodize(e.Lyot, ctb_mask_disk(size(Il,1),abs(macos.dx_at(e.Lyot)),(1-eps)*g.r_lyot_geom_m,8));
    macos.intensity(e.Lyot,'reset_trace',false);
    I = macos.intensity(e.FPA,'reset_trace',false); thru = (1-eps)^2;
end

function [I,thru] = arm_aplc_(opts, g, r_occ_lamD, r_lyot_frac)
    e = opts.elt; N = opts.model_size;
    macos.init(N); macos.load_rx(opts.rx);
    macos.intensity(e.DM1);
    Iap = macos.intensity(e.Apodizer,'reset_trace',false);
    r_apod_px = beam_radius_(Iap,1);
    [Phi,~] = ctb_apod_prolate(N, r_apod_px, r_occ_lamD);
    thru_apod = phi2_fill_(Phi, r_apod_px);
    macos.init(N); macos.load_rx(opts.rx);
    macos.intensity(e.DM1); macos.intensity(e.DM2,'reset_trace',false);
    macos.intensity(e.Apodizer,'reset_trace',false);
    macos.apodize(e.Apodizer, Phi);
    macos.intensity(e.Apodizer,'reset_trace',false);
    If = macos.intensity(e.FPM,'reset_trace',false);
    macos.apodize(e.FPM, 1 - ctb_mask_disk(size(If,1), g.dx_f, r_occ_lamD*g.lamD_fpm_m, 8));
    macos.intensity(e.FPM,'reset_trace',false);
    Il = macos.intensity(e.Lyot,'reset_trace',false);
    macos.apodize(e.Lyot, ctb_mask_disk(size(Il,1),abs(macos.dx_at(e.Lyot)),r_lyot_frac*g.r_lyot_geom_m,8));
    macos.intensity(e.Lyot,'reset_trace',false);
    I = macos.intensity(e.FPA,'reset_trace',false); thru = thru_apod*r_lyot_frac^2;
end

function [I,thru] = arm_vortex_(opts, g, m, r_lyot_frac)
    e = opts.elt; N = opts.model_size;
    macos.init(N); macos.load_rx(opts.rx);
    macos.intensity(e.DM1); macos.intensity(e.DM2,'reset_trace',false);
    macos.intensity(e.Apodizer,'reset_trace',false);
    macos.intensity(e.FPM,'reset_trace',false);
    c=floor(N/2); [xx,yy]=meshgrid((0:N-1)-c,(0:N-1)-c); V=exp(1i*m*atan2(yy,xx)); V(c+1,c+1)=1;
    macos.apodize_complex(e.FPM, V);
    macos.intensity(e.FPM,'reset_trace',false);
    Il = macos.intensity(e.Lyot,'reset_trace',false);
    macos.apodize(e.Lyot, ctb_mask_disk(size(Il,1),abs(macos.dx_at(e.Lyot)),r_lyot_frac*g.r_lyot_geom_m,8));
    macos.intensity(e.Lyot,'reset_trace',false);
    I = macos.intensity(e.FPA,'reset_trace',false); thru = r_lyot_frac^2;
end

function [I,thru] = arm_phase_(opts, g, kind, r_lyot_frac)
    e = opts.elt; N = opts.model_size;
    macos.init(N); macos.load_rx(opts.rx);
    macos.intensity(e.DM1); macos.intensity(e.DM2,'reset_trace',false);
    macos.intensity(e.Apodizer,'reset_trace',false);
    macos.intensity(e.FPM,'reset_trace',false);
    switch kind
        case 'roddier',  V = ctb_mask_phase(N,g.dx_f,g.lamD_fpm_m,'roddier',struct('rho0_lamD',0.53));
        case 'dualzone', V = ctb_mask_phase(N,g.dx_f,g.lamD_fpm_m,'dualzone',struct());
    end
    macos.apodize_complex(e.FPM, V);
    macos.intensity(e.FPM,'reset_trace',false);
    Il = macos.intensity(e.Lyot,'reset_trace',false);
    macos.apodize(e.Lyot, ctb_mask_disk(size(Il,1),abs(macos.dx_at(e.Lyot)),r_lyot_frac*g.r_lyot_geom_m,8));
    macos.intensity(e.Lyot,'reset_trace',false);
    I = macos.intensity(e.FPA,'reset_trace',false); thru = r_lyot_frac^2;
end

% ======================================================================
function row = mkrow_(name, family, I, peak_bare, lamD, opts, thru, note)
    dz = dark_zone_metrics(I, peak_bare, lamD, opts.inner_lamD, opts.outer_lamD);
    row = struct('name',name,'family',family,'I',I,'dz',dz, ...
        'supp',peak_bare/max(max(I(:)),eps),'thru',thru,'note',note);
end

function t = phi2_fill_(Phi, r_pup_px)
    N=size(Phi,1); c=floor(N/2)+1; [X,Y]=meshgrid((1:N)-c,(1:N)-c);
    P = hypot(X,Y) <= r_pup_px; t = sum(Phi(P).^2)/max(sum(P(:)),1);
end

% ---- shared geometry / helpers ---------------------------------------
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
function pk = bare_peak_(opts, e)
    macos.init(opts.model_size); macos.load_rx(opts.rx);
    macos.intensity(e.DM1);
    I = macos.intensity(e.FPA,'reset_trace',false); pk = max(I(:));
end
function rr = beam_radius_(I, dx)
    thr = 0.02*max(I(:)); [yy,xx] = find(I>thr);
    if isempty(xx), rr=0; return; end
    c = floor(size(I,1)/2) + 1; rr = max(hypot(xx-c,yy-c))*dx;
end
function show_(ax, I, peak, w, ttl)
    In = double(I)/max(peak,eps); L=log10(max(In,1e-12));
    imagesc(ax, crop_(L,w)); axis(ax,'image','off'); colormap(ax,parula); clim(ax,[-10 0]);
    cbh=colorbar(ax); cbh.Label.String='log_{10} contrast'; cbh.FontSize=14; cbh.Label.FontSize=15;
    title(ax,ttl,'Interpreter','tex','FontSize',18);
end
function o = crop_(img, w)
    n=size(img,1); if w>=n, o=img; return; end
    c=floor(n/2)+1; lo=max(c-floor(w/2),1); hi=min(lo+w-1,n); o=img(lo:hi,lo:hi);
end
