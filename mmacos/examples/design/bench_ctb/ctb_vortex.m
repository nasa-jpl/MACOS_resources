function out = ctb_vortex(opts)
%CTB_VORTEX  Scalar vortex coronagraph on the CTB (work E, MATLAB domain).
%   Replaces the hard-edge occulter with an idealized CHARGE-m scalar
%   vortex phase mask exp(i*m*theta) at the FPM, applied to the complex
%   field IN MATLAB via macos.apodize_complex -- the same stage-1 "modify
%   the cfield in MATLAB" contract as the amplitude masks.
%
%   NO ENGINE CHANGE IS NEEDED: the complex-mask API the recipe's stage-2
%   called for (cfield_apodize_c) ALREADY EXISTS in the engine as
%   cfield_apodize_complex(OK, MASK_RE, MASK_IM, N, iElt) (macos_api_mod.F90),
%   surfaced as macos.apodize_complex.  A scalar vortex is a pure phase
%   mask, so it rides that API directly.
%
%   The vortex sends on-axis starlight entirely OUTSIDE the geometric pupil
%   at the Lyot plane (the defining property of a charge-even vortex), so
%   the Lyot stop then rejects it -- a much deeper, inner-working-angle-
%   smaller suppression than a hard occulter, with no light blocked on-axis.
%   An idealized exp(i*m*theta) vortex is achromatic BY CONSTRUCTION (the
%   phase has no lambda dependence), so it composes cleanly with the
%   bandpass driver.
%
%   SINGULAR PIXEL: exp(i*m*theta) is undefined at theta origin (the chief
%   pierce).  The central pixel's phase is set to 0 (transmission 1) -- a
%   single-pixel defect whose energy is O(1/N^2) of the core and does not
%   affect the dark zone; documented, not hidden.
%
%   out = CTB_VORTEX() runs a charge-6 vortex vs the hard occulter and the
%   bare PSF, at 500 nm.  Name-value:
%     'rx','elt'        deck + station map (default compact ctb_dcr.in).
%     'model_size'      grid (512).
%     'charge'          vortex topological charge m (default 6, even).
%     'r_apod_m','r_apod_taper_m','r_lyot_frac'  mask params.
%     'r_fpm_lamD'      hard-occulter radius for the comparison arm (3).
%     'inner_lamD','outer_lamD'  dark-zone annulus (2, 15; vortex reaches
%                       smaller inner working angle than the hard occulter).
%     'outdir','visible'.
%
%   See also: macos.apodize_complex, ctb_coro_compare, ctb_contrast.
    arguments
        opts.rx            (1,:) char   = ''
        opts.elt           struct = struct('DM1',2,'DM2',5,'Apodizer',13, ...
                                'FPM',17,'Lyot',20,'ExitPupil',30,'FPA',31)
        opts.model_size    (1,1) double = 1024
        opts.charge        (1,1) double = 6
        opts.r_apod_m      (1,1) double = 15e-3
        opts.r_apod_taper_m(1,1) double = 2e-3
        opts.r_lyot_frac   (1,1) double = 0.50
        opts.r_fpm_lamD    (1,1) double = 2.70
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
    assert(~isempty(getenv('MACOS_HOME')),'MACOS_HOME must be set.');
    e = opts.elt;

    g = geom_scales_(opts, e);
    lamD0 = g.lamD_fpa_px;
    fprintf('[vortex] charge=%d  lamD_fpa=%.3f px\n', opts.charge, lamD0);

    % three arms on one grid: bare, hard occulter, charge-m vortex
    [I_bare, peak_bare] = arm_(opts, g, 'bare');
    I_hard   = arm_(opts, g, 'hard');
    [I_vort, I_lyot_v]  = arm_(opts, g, 'vortex');

    % contrast metrics (Strehl-norm to bare peak)
    dz_hard = dark_zone_metrics(I_hard, peak_bare, lamD0, opts.inner_lamD, opts.outer_lamD);
    dz_vort = dark_zone_metrics(I_vort, peak_bare, lamD0, opts.inner_lamD, opts.outer_lamD);
    fprintf('[vortex] dark zone %.0f-%.0f lam/D mean contrast: hard=%.3e  vortex=%.3e (%.1fx deeper)\n', ...
        opts.inner_lamD, opts.outer_lamD, dz_hard.mean, dz_vort.mean, dz_hard.mean/dz_vort.mean);
    [rh,ch] = radial_contrast(I_hard, peak_bare, lamD0, opts.outer_lamD+3);
    [rv,cv] = radial_contrast(I_vort, peak_bare, lamD0, opts.outer_lamD+3);

    % ---- figure --------------------------------------------------------
    vis='off'; if opts.visible, vis='on'; end
    fig = figure('Visible',vis,'Color','w','Position',[60 60 1300 760]);
    tl = tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
    title(tl, sprintf(['CTB scalar vortex (charge %d) vs hard occulter ', ...
        '-- MATLAB apodize\\_complex'], opts.charge), ...
        'FontWeight','bold','Interpreter','tex');
    w = round(2*(opts.outer_lamD+3)*lamD0);
    show_(nexttile(tl), I_bare, peak_bare, w, 'bare PSF (no coro)');
    show_(nexttile(tl), I_hard, peak_bare, w, sprintf('hard occulter (%.0f \\lambda/D)',opts.r_fpm_lamD));
    show_(nexttile(tl), I_vort, peak_bare, w, sprintf('vortex FPA (charge %d)',opts.charge));
    % Lyot pupil under the vortex: starlight pushed to a ring outside the pupil
    axl = nexttile(tl);
    A = sqrt(max(double(I_lyot_v),0)); A = A/max(A(:)+eps);
    imagesc(axl, crop_(A, 320)); axis(axl,'image','off'); colormap(axl,gray); clim(axl,[0 1]);
    title(axl,'Lyot pupil (vortex): star -> ring outside');
    % contrast curves
    axc = nexttile(tl,[1 2]); hold(axc,'on'); set(axc,'YScale','log');
    hh=plot(axc,rh,max(ch,1e-12),'-','Color',[0.5 0.5 0.5],'LineWidth',1.5);
    hv=plot(axc,rv,max(cv,1e-12),'-','Color',[0.6 0.1 0.6],'LineWidth',1.8);
    xr=[opts.inner_lamD opts.outer_lamD]; yl=ylim(axc);
    p=patch(axc,[xr(1) xr(2) xr(2) xr(1)],[yl(1) yl(1) yl(2) yl(2)], ...
        [0.93 0.85 0.95],'FaceAlpha',0.30,'EdgeColor','none','HandleVisibility','off');
    uistack(p,'bottom');
    grid(axc,'on'); box(axc,'on'); xlabel(axc,'separation (\lambda/D)'); ylabel(axc,'contrast');
    legend(axc,[hh hv],{sprintf('hard (mean %.1e)',dz_hard.mean), ...
        sprintf('vortex (mean %.1e)',dz_vort.mean)},'Location','northeast');
    title(axc,'radial dark-zone contrast: hard occulter vs vortex');

    figpath = fullfile(opts.outdir,'ctb_vortex.png');
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[vortex] wrote %s\n', figpath);

    out = struct('charge',opts.charge,'lamD_px',lamD0,'peak_bare',peak_bare, ...
        'I_bare',I_bare,'I_hard',I_hard,'I_vort',I_vort,'I_lyot_vortex',I_lyot_v, ...
        'dz_hard',dz_hard,'dz_vort',dz_vort,'r_hard',rh,'c_hard',ch, ...
        'r_vort',rv,'c_vort',cv,'figure',figpath);
end

% ======================================================================
function varargout = arm_(opts, g, kind)
    e = opts.elt;
    macos.init(opts.model_size); macos.load_rx(opts.rx);
    macos.intensity(e.DM1);
    macos.intensity(e.DM2,'reset_trace',false);
    % apodizer (all arms except bare get it; keep bare truly bare)
    if ~strcmp(kind,'bare')
        Ia = macos.intensity(e.Apodizer,'reset_trace',false);
        macos.apodize(e.Apodizer, mask_softcircle_(size(Ia,1), abs(macos.dx_at(e.Apodizer)), ...
            opts.r_apod_m, opts.r_apod_taper_m));
        macos.intensity(e.Apodizer,'reset_trace',false);
    end
    % FPM
    If = macos.intensity(e.FPM,'reset_trace',false);
    N = size(If,1);
    switch kind
        case 'bare'
            % no FPM
        case 'hard'
            r_fpm_m = opts.r_fpm_lamD * g.lamD_fpm_m;
            macos.apodize(e.FPM, 1 - mask_harddisk_(N, g.dx_f, r_fpm_m));
        case 'vortex'
            V = vortex_mask_(N, opts.charge);            % complex exp(i m theta)
            macos.apodize_complex(e.FPM, V);
    end
    if ~strcmp(kind,'bare'), macos.intensity(e.FPM,'reset_trace',false); end
    % Lyot
    if ~strcmp(kind,'bare')
        Il = macos.intensity(e.Lyot,'reset_trace',false);
        macos.apodize(e.Lyot, mask_harddisk_(size(Il,1), abs(macos.dx_at(e.Lyot)), ...
            opts.r_lyot_frac * g.r_lyot_geom_m));
        I_lyot = macos.intensity(e.Lyot,'reset_trace',false);
    else
        I_lyot = macos.intensity(e.Lyot,'reset_trace',false);
    end
    I_fpa = macos.intensity(e.FPA,'reset_trace',false);
    switch kind
        case 'bare',   varargout = {I_fpa, max(I_fpa(:))};
        case 'hard',   varargout = {I_fpa};
        case 'vortex', varargout = {I_fpa, I_lyot};
    end
end

function V = vortex_mask_(N, m)
%VORTEX_MASK_  Charge-m scalar vortex exp(i*m*theta), centred on the BEAM
%   pixel floor(N/2) (0-based) where the NF2 focus lands (centering fix).
%   Central singular pixel set to phase 0 (transmission 1); O(1/N^2) defect.
    c = floor(N/2); [xx,yy] = meshgrid((0:N-1)-c, (0:N-1)-c);
    th = atan2(yy, xx);
    V = exp(1i * m * th);
    V(c+1, c+1) = 1;                                    % singular pixel (1-based)
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
function M = mask_harddisk_(N, dx, r_m), M = ctb_mask_disk(N,dx,r_m,8); end
function M = mask_softcircle_(N, dx, r0_m, sigma_m)
    M = ctb_mask_softcircle(N,dx,r0_m,sigma_m,8);
end
function M = disk_ss_(N, dx, r_m, K)
    c=(N-1)/2; off=((0:K-1)-(K-1)/2)/K; M=zeros(N);
    [ox,oy]=meshgrid(off,off); ox=ox(:).'; oy=oy(:).';
    for i=1:N
        yc=(i-1-c); xs=((0:N-1)-c).'; acc=zeros(N,1);
        for s=1:numel(ox)
            xx=(xs+ox(s))*dx; yy=(yc+oy(s))*dx; acc=acc+double(xx.^2+yy.^2<=r_m^2);
        end
        M(i,:)=acc.'/numel(ox);
    end
end
function show_(ax, I, peak, w, ttl)
    In = double(I)/max(peak,eps); L=log10(max(In,1e-12));
    imagesc(ax, crop_(L,w)); axis(ax,'image','off'); colormap(ax,parula); clim(ax,[-10 0]);
    cb=colorbar(ax); cb.Label.String='log_{10} contrast'; title(ax,ttl,'Interpreter','tex');
end
function o = crop_(img, w)
    n=size(img,1); if w>=n, o=img; return; end
    c=floor(n/2)+1; lo=max(c-floor(w/2),1); hi=min(lo+w-1,n); o=img(lo:hi,lo:hi);
end
