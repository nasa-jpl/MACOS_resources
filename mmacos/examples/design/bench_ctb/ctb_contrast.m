function out = ctb_contrast(opts)
%CTB_CONTRAST  Radial dark-zone contrast at the CTB FPA (work B).
%   Propagates ONE model twice -- bare (no masks) and coronagraphic
%   (apodizer+FPM+Lyot) -- then scores the coronagraphic FPA against the
%   BARE on-axis peak (Strehl-normalised contrast, the coronagraph-
%   literature convention).  Reuses the validated coro/ helpers
%   (lambda_over_D_pixels, radial_contrast, dark_zone_metrics).
%
%   The CTB is a 2-DM relay -> a full 360-deg ANNULAR dark zone is the
%   fair scoring region.  Default annulus 3-15 lambda/D (inner edge set by
%   the FPM occulter radius, outer by the DM control bandwidth ~N/2 cycles).
%
%   out = CTB_CONTRAST() scores the compact deck; writes a contrast curve
%   + a dark-zone-metrics print.  Name-value:
%     'rx','elt'         model deck + station map (default compact).
%     'model_size'       grid (512).
%     'r_fpm_lamD'       FPM occulter radius, lambda/D (3).
%     'inner_lamD','outer_lamD'  dark-zone annulus (3, 15).
%     'r_apod_m','r_apod_taper_m','r_lyot_frac'  mask params.
%     'outdir','visible'.
%
%   See also: ctb_coro_compare, radial_contrast, dark_zone_metrics.
    arguments
        opts.rx            (1,:) char   = ''
        opts.elt           struct = struct('DM1',2,'DM2',5,'Apodizer',13, ...
                                'FPM',17,'Lyot',20,'ExitPupil',30,'FPA',31)
        opts.model_size    (1,1) double = 512
        opts.r_fpm_lamD    (1,1) double = 3.0
        opts.inner_lamD    (1,1) double = 3.0
        opts.outer_lamD    (1,1) double = 15.0
        opts.r_apod_m      (1,1) double = 15e-3
        opts.r_apod_taper_m(1,1) double = 2e-3
        opts.r_lyot_frac   (1,1) double = 0.85
        opts.outdir        (1,:) char   = ''
        opts.visible       (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    if isempty(opts.rx),     opts.rx     = fullfile(here,'ctb_dcr.in'); end
    if isempty(opts.outdir), opts.outdir = here; end
    addpath(fullfile(here,'..','..','..','src'));
    coro = fullfile(here,'..','..','coronagraph','coro');
    addpath(coro);
    assert(~isempty(getenv('MACOS_HOME')),'MACOS_HOME must be set.');

    % ---- bare + coro FPA via the shared driver (same normalisation) ----
    m = struct('name','model','rx',opts.rx,'elt',opts.elt);
    argc = {'models',m, 'model_size',opts.model_size, ...
            'r_fpm_lamD',opts.r_fpm_lamD, 'r_apod_m',opts.r_apod_m, ...
            'r_apod_taper_m',opts.r_apod_taper_m, 'r_lyot_frac',opts.r_lyot_frac, ...
            'outdir',opts.outdir};
    ob = ctb_coro_compare('coro',false, argc{:});
    oc = ctb_coro_compare('coro',true,  argc{:});
    I_bare = ob.I{1}{end};
    I_coro = oc.I{1}{end};

    % ---- lambda/D at the FPA -- DETERMINISTIC (geometry), not Airy-null --
    % SYSPROP's lamD_px is 0 for this deck (finite conjugate source, zSrc=25:
    % the marginal-ray EFL analysis needs a source at infinity), and the
    % Airy-null finder locks onto the wrong feature on this undersampled core
    % (returned a spurious 16.8).  Compute lambda/D from the exit-pupil FF
    % geometry: lamD = lambda * R_fpa / D_ep, R_fpa = zElt(ExitPupil), D_ep
    % the geometric beam diameter at the exit pupil (both engine reads).
    peak_bare = max(I_bare(:));
    lamD_px = fpa_lamD_px_(opts.rx, opts.elt, opts.model_size);
    fprintf('[contrast] lambda/D = %.3f px (geometric)  bare peak = %.4e\n', ...
        lamD_px, peak_bare);

    % ---- radial contrast curves (bare + coro), Strehl-normalised -------
    [r_b, c_b] = radial_contrast(I_bare, peak_bare, lamD_px, opts.outer_lamD+3);
    [r_c, c_c] = radial_contrast(I_coro, peak_bare, lamD_px, opts.outer_lamD+3);

    % ---- dark-zone metrics over the annulus ----------------------------
    dz = dark_zone_metrics(I_coro, peak_bare, lamD_px, ...
                           opts.inner_lamD, opts.outer_lamD);      % annulus
    fprintf(['[contrast] dark zone %.1f-%.1f lam/D (annulus): ', ...
             'mean=%.3e median=%.3e peak(worst)=%.3e floor(best)=%.3e n=%d\n'], ...
        opts.inner_lamD, opts.outer_lamD, dz.mean, dz.median, dz.peak, dz.floor, dz.n_pix);

    % ---- figure --------------------------------------------------------
    vis = 'off'; if opts.visible, vis='on'; end
    fig = figure('Visible',vis,'Color','w','Position',[80 80 1150 460]);
    tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
    title(tl,'CTB dark-zone contrast (Strehl-normalised to bare peak)', ...
        'FontWeight','bold','Interpreter','none');

    ax1 = nexttile; hold(ax1,'on');
    set(ax1,'YScale','log');
    hb = semilogy(ax1, r_b, max(c_b,1e-12), '-', 'Color',[.5 .5 .5], 'LineWidth',1.3);
    hc = semilogy(ax1, r_c, max(c_c,1e-12), '-', 'Color',[0 0.35 0.8], 'LineWidth',1.7);
    xr = [opts.inner_lamD opts.outer_lamD];
    yl = ylim(ax1);
    p = patch(ax1,[xr(1) xr(2) xr(2) xr(1)],[yl(1) yl(1) yl(2) yl(2)], ...
        [0.75 0.85 1.0],'FaceAlpha',0.30,'EdgeColor','none','HandleVisibility','off');
    uistack(p,'bottom');
    yline(ax1, dz.mean, ':', sprintf('DZ mean %.1e',dz.mean), ...
        'Color',[0 0.35 0.8], 'LabelHorizontalAlignment','left', ...
        'HandleVisibility','off');
    set(ax1,'YScale','log'); grid(ax1,'on'); box(ax1,'on');
    xlabel(ax1,'separation (\lambda/D)'); ylabel(ax1,'contrast');
    legend(ax1,[hb hc],{'bare (no coro)','coronagraphic'},'Location','northeast');
    title(ax1, sprintf('radial contrast (annulus %.0f-%.0f \\lambda/D shaded)', ...
        opts.inner_lamD, opts.outer_lamD));

    ax2 = nexttile;
    In = I_coro / max(peak_bare,eps);
    L = log10(max(In,1e-12));
    w = round(2*(opts.outer_lamD+3)*lamD_px);
    imagesc(ax2, crop_(L,w)); axis(ax2,'image','off');
    colormap(ax2,parula); clim(ax2,[-10 0]);
    cb=colorbar(ax2); cb.Label.String='log_{10} contrast';
    title(ax2,'coronagraphic FPA (contrast units)');

    figpath = fullfile(opts.outdir,'ctb_contrast.png');
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[contrast] wrote %s\n', figpath);

    out = struct('lamD_px',lamD_px,'peak_bare',peak_bare, ...
        'r_bare',r_b,'c_bare',c_b,'r_coro',r_c,'c_coro',c_c, ...
        'dark_zone',dz,'inner_lamD',opts.inner_lamD,'outer_lamD',opts.outer_lamD, ...
        'figure',figpath);
end

function o = crop_(img, w)
    n = size(img,1); if w>=n, o=img; return; end
    c = floor(n/2)+1; lo=max(c-floor(w/2),1); hi=min(lo+w-1,n); o=img(lo:hi,lo:hi);
end

function lamD_px = fpa_lamD_px_(rx, elt, N)
%FPA_LAMD_PX_  Deterministic lambda/D (px) at the FPA from EP FF geometry.
%   lamD = lambda * R_fpa / D_ep  (metres), then / dx_FPA (px).
%   R_fpa = zElt(ExitPupil) (the terminal FarField sphere radius); D_ep is
%   the geometric beam diameter measured on the exit-pupil plane.  Robust
%   for finite-conjugate decks where SYSPROP returns 0.
    macos.init(N); macos.load_rx(rx);
    cbm = macos.cbm(); lambda_m = macos.get_src_wvl()*cbm;
    macos.intensity(elt.FPA);
    Iep  = macos.intensity(elt.ExitPupil, 'reset_trace', false);
    dxep = abs(macos.dx_at(elt.ExitPupil));
    Dep  = 2 * beam_radius_(Iep, dxep);
    R_m  = abs(macos.get_elt_z(elt.ExitPupil)) * cbm;
    dxfpa = abs(macos.dx_at(elt.FPA));
    lamD_px = (lambda_m * R_m / Dep) / dxfpa;
end

function rr = beam_radius_(I, dx)
    thr = 0.02*max(I(:)); [yy,xx] = find(I>thr);
    if isempty(xx), rr=0; return; end
    c = (size(I,1)-1)/2 + 1; rr = max(hypot(xx-c,yy-c))*dx;
end
