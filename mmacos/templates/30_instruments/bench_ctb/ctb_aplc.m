function out = ctb_aplc(opts)
%CTB_APLC  Apodized Pupil Lyot Coronagraph on the CTB, compared to the BLC.
%   The APLC (Soummer 2005, ApJ 618, L161; Soummer, Aime & Falloon 2003,
%   A&A 397, 1161) pairs a PROLATE-SPHEROIDAL pupil apodizer with a small
%   HARD occulter and a Lyot stop.  The apodizer is the dominant prolate
%   eigenfunction of the APLC operator (ctb_apod_prolate); it reshapes the
%   pupil so the occulter + Lyot suppress on-axis starlight far better than
%   a hard occulter alone, at the cost of the apodizer's throughput.
%
%   CHAIN (all MATLAB-domain, stage-1 contract):
%     clear pupil -> PROLATE apodizer at the Apodizer pupil (macos.apodize)
%     -> HARD occulter (radius r_occ_lamD) at the FPM -> Lyot stop -> FPA.
%   Occulter radius default 2.8 lambda/D (Soummer et al. 2011 GPI design:
%   occulter DIAMETER 5.6 lambda/D; cross-checked vs arXiv:1103.6085).
%
%   THROUGHPUT.  The APLC's off-axis (planet) throughput is set by the
%   apodizer transmission (integral of Phi^2 over the pupil / pupil area)
%   times the Lyot area fraction.  To make the "APLC vs BLC" comparison
%   FAIR, this driver measures the APLC throughput, then runs the BLC
%   (ctb_bandlimited chain) at the epsilon whose throughput (1-eps)^2
%   MATCHES it, and compares dark-zone contrast at equal throughput -- the
%   apples-to-apples question the work order asks.
%
%   out = CTB_APLC() runs the prolate APLC at 2.8 lambda/D and the
%   throughput-matched BLC.  Name-value:
%     'rx','elt'         deck + station map (default compact ctb_dcr.in).
%     'model_size'       grid (1024).
%     'r_occ_lamD'       hard occulter radius, lambda/D (default 2.8).
%     'r_lyot_frac'      Lyot stop fraction of geometric pupil (default 0.90;
%                        APLC uses a near-full Lyot -- the apodizer, not the
%                        Lyot, does the suppression).
%     'blc_order','blc_form'  BLC comparison arm (default 4, 'separable').
%     'inner_lamD','outer_lamD'  dark-zone annulus (3, 15).
%     'outdir','visible'.
%
%   See also: ctb_apod_prolate, ctb_bandlimited, ctb_mask_disk, dark_zone_metrics.
    arguments
        opts.rx            (1,:) char   = ''
        opts.elt           struct = struct('DM1',2,'DM2',5,'Apodizer',13, ...
                                'FPM',17,'Lyot',20,'ExitPupil',30,'FPA',31)
        opts.model_size    (1,1) double = 1024
        opts.r_occ_lamD    (1,1) double = 2.8
        opts.r_lyot_frac   (1,1) double = 0.90
        opts.blc_order     (1,1) double {mustBeMember(opts.blc_order,[4 8])} = 4
        opts.blc_form      (1,:) char {mustBeMember(opts.blc_form,{'separable','radial','linear'})} = 'separable'
        opts.inner_lamD    (1,1) double = 3.0
        opts.outer_lamD    (1,1) double = 15.0
        opts.prolate_iter  (1,1) double = 200   % power-iteration cap for
                                                % ctb_apod_prolate.  200 is
                                                % this example's committed
                                                % setting; a large pupil on
                                                % a fine grid can need
                                                % thousands (measured: the
                                                % e2e6m 6 m pupil converges
                                                % at 2387, and at 200 it
                                                % reports Lambda0 = 1.0017,
                                                % i.e. ABOVE the eigenvalue's
                                                % physical bound of 1).
        opts.apodizer      (:,:) double = []   % supply an apodizer instead
                                              % of solving for the prolate.
                                              % Same N as 'model_size'.
                                              % Used by the e2e6m LP slice
                                              % (apodizer_lp) so an
                                              % externally-designed mask is
                                              % scored through EXACTLY this
                                              % chain rather than a copy of
                                              % it.
        opts.skip_blc      (1,1) logical = false  % skip the throughput-
                                              % matched BLC arm (it is a
                                              % second full propagation and
                                              % is not part of every
                                              % question this driver is
                                              % asked).
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
    N = opts.model_size;

    % apodizer beam radius (px) at the Apodizer plane
    r_apod_px = apod_beam_px_(opts, e);
    fprintf('[aplc] N=%d lamD_fpa=%.3f px  r_apod=%.1f px  r_occ=%.2f lam/D  bare peak=%.3e\n', ...
        N, lamD, r_apod_px, opts.r_occ_lamD, peak_bare);

    % ---- the apodizer: supplied, or the prolate -------------------------
    if ~isempty(opts.apodizer)
        Phi = opts.apodizer;
        assert(isequal(size(Phi),[N N]), ...
            'ctb_aplc: supplied apodizer is %dx%d, model_size is %d', ...
            size(Phi,1), size(Phi,2), N);
        pinfo = struct('lambda0',NaN,'converged',NaN,'n_iter_used',0, ...
                       'supplied',true);
        thru_apod = apod_throughput_(Phi, r_apod_px);
        fprintf('[aplc] apodizer SUPPLIED (%dx%d)  apodizer throughput=%.3f\n', ...
            N, N, thru_apod);
    else
        [Phi, pinfo] = ctb_apod_prolate(N, r_apod_px, opts.r_occ_lamD, ...
                                        'n_iter', opts.prolate_iter);
        pinfo.supplied = false;
        thru_apod = apod_throughput_(Phi, r_apod_px);    % Phi^2-weighted fill
        fprintf('[aplc] prolate: Lambda0=%.4f (conv=%d, %d it)  apodizer throughput=%.3f\n', ...
            pinfo.lambda0, pinfo.converged, pinfo.n_iter_used, thru_apod);
    end

    % ---- run APLC ------------------------------------------------------
    [I_aplc, I_lyot_a] = run_aplc_(opts, g, Phi);
    dz_aplc = macos.dark_zone_metrics(I_aplc, peak_bare, lamD, opts.inner_lamD, opts.outer_lamD);
    supp_aplc = peak_bare / max(max(I_aplc(:)),eps);
    thru_aplc = thru_apod * opts.r_lyot_frac^2;          % apodizer * Lyot area
    fprintf('[aplc] APLC: DZ mean=%.3e median=%.3e floor=%.3e  suppression=%.2e  throughput=%.3f\n', ...
        dz_aplc.mean, dz_aplc.median, dz_aplc.floor, supp_aplc, thru_aplc);

    % ---- throughput-matched BLC ----------------------------------------
    % BLC 2-D throughput = (1-eps)^2 -> match: eps = 1 - sqrt(thru_aplc).
    eps_match = max(0.02, min(0.9, 1 - sqrt(thru_aplc)));
    if opts.skip_blc
        I_blc = zeros(N); dz_blc = macos.dark_zone_metrics(I_blc, peak_bare, ...
            lamD, opts.inner_lamD, opts.outer_lamD);
        supp_blc = NaN; thru_blc = NaN;
        [r_aplc,c_aplc] = macos.radial_contrast(I_aplc, peak_bare, lamD, opts.outer_lamD+3);
        out = struct('r_occ_lamD',opts.r_occ_lamD,'r_lyot_frac',opts.r_lyot_frac, ...
            'prolate_info',pinfo,'apodizer_throughput',thru_apod, ...
            'dz_aplc',dz_aplc,'supp_aplc',supp_aplc,'thru_aplc',thru_aplc, ...
            'eps_match',eps_match,'dz_blc',dz_blc,'supp_blc',supp_blc,'thru_blc',thru_blc, ...
            'lamD_px',lamD,'peak_bare',peak_bare,'figure','', ...
            'r_aplc',r_aplc,'c_aplc',c_aplc, ...
            'I_aplc',I_aplc,'I_blc',I_blc,'Phi',Phi,'r_apod_px',r_apod_px);
        fprintf('[aplc] BLC arm skipped (skip_blc)\n');
        return
    end
    [I_blc, ~] = run_blc_(opts, g, eps_match);
    dz_blc = macos.dark_zone_metrics(I_blc, peak_bare, lamD, opts.inner_lamD, opts.outer_lamD);
    supp_blc = peak_bare / max(max(I_blc(:)),eps);
    thru_blc = (1-eps_match)^2;
    fprintf(['[aplc] BLC (throughput-matched, eps=%.3f): DZ mean=%.3e median=%.3e  ', ...
             'suppression=%.2e  throughput=%.3f\n'], ...
        eps_match, dz_blc.mean, dz_blc.median, supp_blc, thru_blc);
    fprintf('[aplc] AT EQUAL THROUGHPUT (~%.2f): APLC mean=%.3e  vs  BLC mean=%.3e (%.2fx)\n', ...
        thru_aplc, dz_aplc.mean, dz_blc.mean, dz_blc.mean/dz_aplc.mean);

    % ---- radial contrast curves ----------------------------------------
    [r_aplc,c_aplc] = macos.radial_contrast(I_aplc, peak_bare, lamD, opts.outer_lamD+3);
    [r_blc, c_blc ] = macos.radial_contrast(I_blc,  peak_bare, lamD, opts.outer_lamD+3);

    % ================= figure ===========================================
    vis='off'; if opts.visible, vis='on'; end
    fig = figure('Visible',vis,'Color','w','Position',[50 50 1500 900]);
    tl = tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
    title(tl, sprintf(['CTB APLC (prolate apodizer + %.1f \\lambda/D occulter) ', ...
        'vs throughput-matched BLC'], opts.r_occ_lamD),'FontWeight','bold','Interpreter','tex');

    % (1) apodizer amplitude image
    ax=nexttile(tl); imagesc(ax, crop_(Phi,round(2.4*r_apod_px))); axis(ax,'image','off');
    colormap(ax,gray); clim(ax,[0 1]); cb=colorbar(ax); cb.Label.String='amplitude \Phi';
    title(ax,'prolate apodizer \Phi (pupil)','Interpreter','tex');

    % (2) apodizer radial profile
    ax=nexttile(tl); c=floor(N/2)+1; prof=Phi(c,c:c+round(r_apod_px)+2);
    rp=(0:numel(prof)-1)/r_apod_px;
    plot(ax, rp, prof, '-','Color',[0 0.35 0.8],'LineWidth',1.8);
    grid(ax,'on'); box(ax,'on'); xlabel(ax,'radius / R_{pupil}'); ylabel(ax,'\Phi'); ylim(ax,[0 1.05]);
    title(ax,sprintf('apodizer profile (\\Lambda_0=%.3f)',pinfo.lambda0),'Interpreter','tex');

    % (3) APLC Lyot pupil (post-occulter, pre-stop)
    ax=nexttile(tl); A=sqrt(max(double(I_lyot_a),0)); A=A/max(A(:)+eps);
    imagesc(ax, crop_(A,360)); axis(ax,'image','off'); colormap(ax,gray); clim(ax,[0 1]);
    title(ax,'APLC Lyot pupil (post-occulter)');

    % (4) APLC FPA
    w = round(2*(opts.outer_lamD+3)*lamD);
    ax=nexttile(tl); show_(ax, I_aplc, peak_bare, w, sprintf('APLC FPA (suppr %.1e)',supp_aplc));

    % (5) BLC FPA (throughput matched)
    ax=nexttile(tl); show_(ax, I_blc, peak_bare, w, ...
        sprintf('BLC FPA (\\epsilon=%.2f, suppr %.1e)',eps_match,supp_blc));

    % (6) contrast curves + summary
    ax=nexttile(tl); hold(ax,'on'); set(ax,'YScale','log');
    ha=plot(ax,r_aplc,max(c_aplc,1e-12),'-','Color',[0.6 0.1 0.6],'LineWidth',1.8);
    hb=plot(ax,r_blc, max(c_blc, 1e-12),'-','Color',[0 0.35 0.8],'LineWidth',1.6);
    xr=[opts.inner_lamD opts.outer_lamD]; yl=ylim(ax);
    p=patch(ax,[xr(1) xr(2) xr(2) xr(1)],[yl(1) yl(1) yl(2) yl(2)], ...
        [0.9 0.9 0.95],'FaceAlpha',0.4,'EdgeColor','none','HandleVisibility','off');
    uistack(p,'bottom');
    grid(ax,'on'); box(ax,'on'); xlabel(ax,'separation (\lambda/D)'); ylabel(ax,'contrast');
    legend(ax,[ha hb],{sprintf('APLC (mean %.1e, T=%.2f)',dz_aplc.mean,thru_aplc), ...
        sprintf('BLC \\epsilon=%.2f (mean %.1e, T=%.2f)',eps_match,dz_blc.mean,thru_blc)}, ...
        'Location','northeast','Interpreter','tex');
    title(ax,'contrast at EQUAL throughput: APLC vs BLC','Interpreter','tex');

    figpath = fullfile(opts.outdir,'ctb_aplc.png');
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[aplc] wrote %s\n', figpath);

    out = struct('r_occ_lamD',opts.r_occ_lamD,'r_lyot_frac',opts.r_lyot_frac, ...
        'prolate_info',pinfo,'apodizer_throughput',thru_apod, ...
        'dz_aplc',dz_aplc,'supp_aplc',supp_aplc,'thru_aplc',thru_aplc, ...
        'eps_match',eps_match,'dz_blc',dz_blc,'supp_blc',supp_blc,'thru_blc',thru_blc, ...
        'lamD_px',lamD,'peak_bare',peak_bare,'figure',figpath, ...
        'r_aplc',r_aplc,'c_aplc',c_aplc,'Phi',Phi,'r_apod_px',r_apod_px, ...
        'I_aplc',I_aplc,'I_blc',I_blc);   % the FPA images, so a caller can
                                          % re-score or re-plot without
                                          % re-running the chain
end

% ======================================================================
function [I_fpa, I_lyot] = run_aplc_(opts, g, Phi)
    e = opts.elt; N = opts.model_size;
    macos.init(N); macos.load_rx(opts.rx);
    macos.intensity(e.DM1);
    macos.intensity(e.DM2,'reset_trace',false);
    % prolate apodizer at the Apodizer pupil
    macos.intensity(e.Apodizer,'reset_trace',false);
    macos.apodize(e.Apodizer, Phi);
    macos.intensity(e.Apodizer,'reset_trace',false);
    % hard occulter at the FPM
    macos.intensity(e.FPM,'reset_trace',false);
    r_occ_m = opts.r_occ_lamD * g.lamD_fpm_m;
    macos.apodize(e.FPM, 1 - ctb_mask_disk(N, g.dx_f, r_occ_m, 8));
    macos.intensity(e.FPM,'reset_trace',false);
    % Lyot stop
    I_lyot = macos.intensity(e.Lyot,'reset_trace',false);
    dxl = abs(macos.dx_at(e.Lyot));
    macos.apodize(e.Lyot, ctb_mask_disk(N, dxl, opts.r_lyot_frac*g.r_lyot_geom_m, 8));
    macos.intensity(e.Lyot,'reset_trace',false);
    I_fpa = macos.intensity(e.FPA,'reset_trace',false);
end

function [I_fpa, I_lyot] = run_blc_(opts, g, eps_mask)
    e = opts.elt; N = opts.model_size;
    macos.init(N); macos.load_rx(opts.rx);
    macos.intensity(e.DM1);
    macos.intensity(e.DM2,'reset_trace',false);
    macos.intensity(e.Apodizer,'reset_trace',false);     % no apodizer (BLC clear pupil)
    macos.intensity(e.FPM,'reset_trace',false);
    M = ctb_mask_bandlimited(N, g.dx_f, g.lamD_fpm_m, eps_mask, opts.blc_order, opts.blc_form);
    macos.apodize(e.FPM, M);
    macos.intensity(e.FPM,'reset_trace',false);
    I_lyot = macos.intensity(e.Lyot,'reset_trace',false);
    dxl = abs(macos.dx_at(e.Lyot));
    macos.apodize(e.Lyot, ctb_mask_disk(N, dxl, (1-eps_mask)*g.r_lyot_geom_m, 8));
    macos.intensity(e.Lyot,'reset_trace',false);
    I_fpa = macos.intensity(e.FPA,'reset_trace',false);
end

function r_px = apod_beam_px_(opts, e)
    macos.init(opts.model_size); macos.load_rx(opts.rx);
    macos.intensity(e.DM1);
    I = macos.intensity(e.Apodizer,'reset_trace',false);
    r_px = beam_radius_(I, 1);                            % dx=1 -> pixels
end

function t = apod_throughput_(Phi, r_pup_px)
%APOD_THROUGHPUT_  Phi^2-weighted fill fraction over the geometric pupil
%   (the off-axis point-source amplitude throughput of the apodizer alone).
    N=size(Phi,1); c=floor(N/2)+1; [X,Y]=meshgrid((1:N)-c,(1:N)-c);
    P = hypot(X,Y) <= r_pup_px;
    t = sum(Phi(P).^2) / max(sum(P(:)),1);
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
    cb=colorbar(ax); cb.Label.String='log_{10} contrast'; title(ax,ttl,'Interpreter','tex');
end
function o = crop_(img, w)
    n=size(img,1); if w>=n, o=img; return; end
    c=floor(n/2)+1; lo=max(c-floor(w/2),1); hi=min(lo+w-1,n); o=img(lo:hi,lo:hi);
end
