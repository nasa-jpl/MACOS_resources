function out = ctb_phase_masks(opts)
%CTB_PHASE_MASKS  Phase-mask coronagraphs on the CTB (Roddier & Roddier +
%   dual-zone), reported beside the scalar vortex.
%   The first LITERATURE COMPLEX focal masks on this bench (the vortex is a
%   phase ramp; these are phase STEPS).  Both applied to the complex field
%   at the FPM via macos.apodize_complex -- the stage-1 "modify the cfield
%   in MATLAB" contract.
%
%   ARMS (all clear-pupil; masks from ctb_mask_phase):
%     'roddier'  -- Roddier & Roddier 1997 (PASP 109, 815): a PI-phase focal
%        disk of radius 0.53 lambda/D on the clear pupil (extinction by flux
%        balance -- R&R use NO apodizer; the prolate-apodized RRPM is later
%        Aime/Soummer work).  The pi-shifted core interferes destructively
%        with the outer field -> starlight pushed outside the Lyot pupil.
%     'dualzone' -- achromatic dual-zone phase mask (N'Diaye et al. 2012,
%        A&A 538, A55 = arXiv:1111.3194; Soummer, Dohlen & Aime 2003): inner
%        phase disk (d1=0.874 lambda0/D) + outer phase ring (d2=1.445), both
%        PURE PHASE (non-pi: phi1=1.94, phi2=4.26 rad at lambda0), the phases
%        given as OPDs that slide with wavelength (the achromatization
%        mechanism).  The full DZPM's amplitude apodization lives in the
%        entrance pupil; this driver applies the MASK arm (the achromatic
%        design point of interest here).
%     'vortex'   -- charge-6 scalar vortex (ctb_vortex reference), for a
%        side-by-side contrast comparison.
%
%   Each arm gets a Lyot stop (r_lyot_frac of the geometric pupil).  Scored
%   over a dark-zone annulus and shown as FPA + Lyot-pupil + contrast curves.
%
%   out = CTB_PHASE_MASKS().  Name-value:
%     'rx','elt'        deck + station map (default compact ctb_dcr.in).
%     'model_size'      grid (1024).
%     'rho0_lamD'       R&R spot radius, lambda/D (default 0.53).
%     'dz_d1_lamD','dz_d2_lamD'  dual-zone diameters (0.874, 1.445).
%     'dz_phi1','dz_phi2'  dual-zone phases at lambda0 (1.94, 4.26 rad).
%     'charge'          vortex charge for the comparison arm (6).
%     'r_lyot_frac'     Lyot stop fraction of geometric pupil (0.90).
%     'inner_lamD','outer_lamD'  dark-zone annulus (2, 15).
%     'outdir','visible'.
%
%   See also: ctb_mask_phase, ctb_vortex, ctb_vortex_matched, macos.apodize_complex.
    arguments
        opts.rx            (1,:) char   = ''
        opts.elt           struct = struct('DM1',2,'DM2',5,'Apodizer',13, ...
                                'FPM',17,'Lyot',20,'ExitPupil',30,'FPA',31)
        opts.model_size    (1,1) double = 1024
        opts.rho0_lamD     (1,1) double = 0.53
        opts.dz_d1_lamD    (1,1) double = 0.874
        opts.dz_d2_lamD    (1,1) double = 1.445
        opts.dz_phi1       (1,1) double = 1.94
        opts.dz_phi2       (1,1) double = 4.26
        opts.charge        (1,1) double = 6
        opts.r_lyot_frac   (1,1) double = 0.90
        opts.inner_lamD    (1,1) double = 2.0
        opts.outer_lamD    (1,1) double = 15.0
        opts.outdir        (1,:) char   = ''
        opts.visible       (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    if isempty(opts.rx),     opts.rx     = fullfile(here,'ctb_dcr.in'); end
    if isempty(opts.outdir), opts.outdir = here; end
    addpath(fullfile(here,'..','..','..','src'));
    addpath(here);
    assert(~isempty(getenv('MACOS_HOME')),'MACOS_HOME must be set.');
    e = opts.elt;

    g = geom_scales_(opts, e);
    lamD = g.lamD_fpa_px;
    peak_bare = bare_peak_(opts, e);
    fprintf('[phase] N=%d lamD_fpa=%.3f px  r_lyot_geom=%.4e m  bare peak=%.3e\n', ...
        opts.model_size, lamD, g.r_lyot_geom_m, peak_bare);

    arms = {'roddier','dualzone','vortex'};
    labels = {sprintf('Roddier \\pi-mask (%.2f \\lambda/D)',opts.rho0_lamD), ...
              sprintf('dual-zone (d_1=%.2f d_2=%.2f)',opts.dz_d1_lamD,opts.dz_d2_lamD), ...
              sprintf('vortex (charge %d)',opts.charge)};
    R = struct('arm',{},'I_fpa',{},'I_lyot',{},'dz',{},'supp',{},'r',{},'c',{});
    for k = 1:numel(arms)
        [Ifpa, Ilyot] = run_arm_(opts, g, arms{k});
        dz = macos.dark_zone_metrics(Ifpa, peak_bare, lamD, opts.inner_lamD, opts.outer_lamD);
        [rr,cc] = macos.radial_contrast(Ifpa, peak_bare, lamD, opts.outer_lamD+3);
        R(k) = struct('arm',arms{k},'I_fpa',Ifpa,'I_lyot',Ilyot,'dz',dz, ...
            'supp',peak_bare/max(max(Ifpa(:)),eps),'r',rr,'c',cc);
        fprintf('[phase] %-9s DZ mean=%.3e median=%.3e floor=%.3e  suppression=%.2e\n', ...
            arms{k}, dz.mean, dz.median, dz.floor, R(k).supp);
    end

    % ================= figure ===========================================
    vis='off'; if opts.visible, vis='on'; end
    fig = figure('Visible',vis,'Color','w','Position',[50 50 1500 900]);
    tl = tiledlayout(fig,3,3,'TileSpacing','compact','Padding','compact');
    title(tl, 'CTB phase-mask coronagraphs (Roddier & Roddier / dual-zone) vs vortex', ...
        'FontWeight','bold','Interpreter','tex');
    w = round(2*(opts.outer_lamD+3)*lamD);

    for k = 1:3
        % row: FPA | Lyot pupil | mask-phase inset
        ax=nexttile(tl); show_(ax, R(k).I_fpa, peak_bare, w, ...
            sprintf('%s -- FPA (suppr %.1e)',labels{k},R(k).supp));
        ax=nexttile(tl);
        A=sqrt(max(double(R(k).I_lyot),0)); A=A/max(A(:)+eps);
        imagesc(ax, crop_(A,360)); axis(ax,'image','off'); colormap(ax,gray); clim(ax,[0 1]);
        title(ax, sprintf('%s -- Lyot pupil', arms{k}),'Interpreter','none');
        ax=nexttile(tl);
        Vp = mask_phase_inset_(opts, g, arms{k});
        imagesc(ax, crop_(Vp, round(6*lamD))); axis(ax,'image','off');
        colormap(ax,hsv); clim(ax,[-pi pi]); cb=colorbar(ax); cb.Label.String='phase (rad)';
        title(ax, sprintf('%s -- mask phase', arms{k}),'Interpreter','none');
    end

    figpath = fullfile(opts.outdir,'ctb_phase_masks.png');
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[phase] wrote %s\n', figpath);

    % contrast-curve companion figure
    fig2 = figure('Visible',vis,'Color','w','Position',[80 80 900 520]);
    ax=axes(fig2); hold(ax,'on'); set(ax,'YScale','log');
    cols=[0.85 0.33 0.10; 0.2 0.6 0.2; 0.6 0.1 0.6]; h=gobjects(1,3);
    for k=1:3, h(k)=plot(ax,R(k).r,max(R(k).c,1e-12),'-','Color',cols(k,:),'LineWidth',1.7); end
    xr=[opts.inner_lamD opts.outer_lamD]; yl=ylim(ax);
    p=patch(ax,[xr(1) xr(2) xr(2) xr(1)],[yl(1) yl(1) yl(2) yl(2)], ...
        [0.9 0.9 0.92],'FaceAlpha',0.4,'EdgeColor','none','HandleVisibility','off');
    uistack(p,'bottom'); grid(ax,'on'); box(ax,'on');
    xlabel(ax,'separation (\lambda/D)'); ylabel(ax,'contrast');
    legend(ax,h,arrayfun(@(k)sprintf('%s (mean %.1e)',arms{k},R(k).dz.mean),1:3,'uni',0), ...
        'Location','northeast','Interpreter','none');
    title(ax,'phase-mask radial dark-zone contrast','Interpreter','none');
    figpath2 = fullfile(opts.outdir,'ctb_phase_masks_contrast.png');
    exportgraphics(fig2, figpath2, 'Resolution',150);
    if ~opts.visible, close(fig2); end
    fprintf('[phase] wrote %s\n', figpath2);

    out = struct('arms',{arms},'results',R,'lamD_px',lamD,'peak_bare',peak_bare, ...
        'figure',figpath,'figure_contrast',figpath2);
end

% ======================================================================
function [I_fpa, I_lyot] = run_arm_(opts, g, arm)
    e = opts.elt; N = opts.model_size;
    macos.init(N); macos.load_rx(opts.rx);
    macos.intensity(e.DM1);
    macos.intensity(e.DM2,'reset_trace',false);
    macos.intensity(e.Apodizer,'reset_trace',false);     % clear pupil (no apodizer)
    macos.intensity(e.FPM,'reset_trace',false);
    V = mask_for_(opts, g, arm, N);
    macos.apodize_complex(e.FPM, V);
    macos.intensity(e.FPM,'reset_trace',false);
    I_lyot = macos.intensity(e.Lyot,'reset_trace',false);
    dxl = abs(macos.dx_at(e.Lyot));
    macos.apodize(e.Lyot, ctb_mask_disk(N, dxl, opts.r_lyot_frac*g.r_lyot_geom_m, 8));
    macos.intensity(e.Lyot,'reset_trace',false);
    I_fpa = macos.intensity(e.FPA,'reset_trace',false);
end

function V = mask_for_(opts, g, arm, N)
    switch arm
        case 'roddier'
            V = ctb_mask_phase(N, g.dx_f, g.lamD_fpm_m, 'roddier', ...
                struct('rho0_lamD',opts.rho0_lamD));
        case 'dualzone'
            V = ctb_mask_phase(N, g.dx_f, g.lamD_fpm_m, 'dualzone', ...
                struct('d1_lamD',opts.dz_d1_lamD,'d2_lamD',opts.dz_d2_lamD, ...
                       'phi1',opts.dz_phi1,'phi2',opts.dz_phi2));
        case 'vortex'
            c=floor(N/2); [xx,yy]=meshgrid((0:N-1)-c,(0:N-1)-c);
            V=exp(1i*opts.charge*atan2(yy,xx)); V(c+1,c+1)=1;
    end
end

function Vp = mask_phase_inset_(opts, g, arm)
    V = mask_for_(opts, g, arm, opts.model_size);
    Vp = angle(V);
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
    cbh=colorbar(ax); cbh.Label.String='log_{10} contrast'; title(ax,ttl,'Interpreter','tex');
end
function o = crop_(img, w)
    n=size(img,1); if w>=n, o=img; return; end
    c=floor(n/2)+1; lo=max(c-floor(w/2),1); hi=min(lo+w-1,n); o=img(lo:hi,lo:hi);
end
