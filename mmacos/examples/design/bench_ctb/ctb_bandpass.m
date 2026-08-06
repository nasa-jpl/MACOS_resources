function out = ctb_bandpass(opts)
%CTB_BANDPASS  Finite-bandpass CTB coronagraph (work D).
%   Runs the CTB coronagraph at nwf wavelengths across a stated band and
%   sums the FPA intensities INCOHERENTLY (a broadband PSF), then compares
%   monochromatic vs broadband dark-zone contrast.
%
%   HARD RULE (recipe): each focal-plane mask is re-evaluated PER WAVELENGTH
%   on that lambda's grid -- the focal sampling dx_f = lambda*R/(N*dx_sphere)
%   scales with lambda, so a mask cached at one lambda is wrong at the
%   others.  This driver obeys the rule (the mask is rebuilt inside the
%   per-lambda loop), and the FPA grid pitch dx_at(FPA) also scales with
%   lambda (verified: 2.28e-5 / 2.40e-5 / 2.52e-5 m at 475/500/525 nm).
%
%   FPM SIZING MODE controls the chromatic behaviour and is the whole point
%   of a bandpass study:
%     'fpm_size' = 'meters'  (DEFAULT): the occulter has a FIXED PHYSICAL
%       radius (metres), so it subtends a DIFFERENT lambda/D at each
%       wavelength (larger lambda -> smaller lambda/D radius) -> genuine
%       chromatic contrast smearing, the realistic hard-mask case.
%     'fpm_size' = 'lamD': the occulter is sized in FPM-local lambda/D and
%       auto-scales with the grid, so it subtends the SAME lambda/D at every
%       wavelength -> the dark zone is wavelength-INVARIANT in lambda/D units
%       (mono==broadband to numerical precision).  This is the idealized
%       achromatic-mask reference; it isolates the "does the machinery add
%       chromatic error on its own" question (answer: no).
%
%   macos.compose (COMPOSE+ADD) sums only the engine's own per-lambda traces
%   -- it does NOT apply the MATLAB-domain CTB masks -- so the coronagraphic
%   broadband sum is done in MATLAB (same incoherent COMPOSE ADD physics,
%   mask-aware).
%
%   out = CTB_BANDPASS() sums 5 wavelengths across a 10% band about 500 nm
%   with a fixed-physical-size occulter (the chromatic case).
%   Name-value:
%     'rx','elt'         deck + station map (default compact ctb_dcr.in).
%     'model_size'       grid (512).
%     'nwf'              number of wavelengths (default 5).
%     'band_frac'        fractional bandwidth (default 0.10 = 10%).
%     'fpm_size'         'meters' (chromatic, default) | 'lamD' (achromatic).
%     'r_fpm_lamD','r_apod_m','r_apod_taper_m','r_lyot_frac'  mask params.
%     'inner_lamD','outer_lamD'  dark-zone annulus (3, 15).
%     'outdir','visible'.
%
%   See also: ctb_contrast, ctb_coro_compare, macos.compose, macos.set_src_wvl.
    arguments
        opts.rx            (1,:) char   = ''
        opts.elt           struct = struct('DM1',2,'DM2',5,'Apodizer',13, ...
                                'FPM',17,'Lyot',20,'ExitPupil',30,'FPA',31)
        opts.model_size    (1,1) double = 1024
        opts.nwf           (1,1) double = 5
        opts.band_frac     (1,1) double = 0.10
        opts.fpm_size      (1,:) char {mustBeMember(opts.fpm_size,{'meters','lamD'})} = 'meters'
        opts.r_fpm_lamD    (1,1) double = 2.70
        opts.r_apod_m      (1,1) double = 15e-3
        opts.r_apod_taper_m(1,1) double = 2e-3
        opts.r_lyot_frac   (1,1) double = 0.50
        opts.inner_lamD    (1,1) double = 3.0
        opts.outer_lamD    (1,1) double = 15.0
        opts.outdir        (1,:) char   = ''
        opts.visible       (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    if isempty(opts.rx),     opts.rx     = fullfile(here,'ctb_dcr.in'); end
    if isempty(opts.outdir), opts.outdir = here; end
    addpath(fullfile(here,'..','..','..','src'));
    addpath(fullfile(here,'..','..','coronagraph','coro'));    % contrast helpers
    assert(~isempty(getenv('MACOS_HOME')),'MACOS_HOME must be set.');
    e = opts.elt;

    % nominal wavelength + band
    macos.init(opts.model_size); macos.load_rx(opts.rx);
    wvl0 = macos.get_src_wvl();                                 % WaveUnits
    if opts.nwf == 1
        wvls = wvl0;
    else
        wvls = wvl0 * (1 + opts.band_frac*(linspace(-0.5,0.5,opts.nwf)));
    end
    fprintf('[band] nwf=%d band=%.0f%%  wvls(WaveUnits)= %s\n', ...
        opts.nwf, 100*opts.band_frac, mat2str(wvls,4));

    % lambda/D (px) at the FPA at the NOMINAL wavelength (display scale)
    lamD0 = fpa_lamD_px_(opts, wvl0);

    % fixed physical occulter radius (metres) at the nominal lambda -- held
    % constant across the band in 'meters' mode (the chromatic case).
    g0 = [];
    macos.init(opts.model_size); macos.load_rx(opts.rx); macos.set_src_wvl(wvl0);
    g0 = geom_scales_(opts, e);
    r_fpm_fixed_m = opts.r_fpm_lamD * g0.lamD_fpm_m;
    fprintf('[band] fpm_size=%s  r_fpm(nominal)=%.4e m = %.2f lam/D @ %.0f nm\n', ...
        opts.fpm_size, r_fpm_fixed_m, opts.r_fpm_lamD, wvl0*1e6);

    % ---- per-wavelength coronagraphic FPA (masks rebuilt per lambda) ---
    % CRITICAL: MACOS's FarField FPA RE-GRIDS per wavelength (dx_at(FPA)
    % scales with lambda), so each monochromatic PSF is the SAME pixel size
    % on the NxN array -- summing them there cancels the chromatic effect.
    % The physically correct broadband PSF resamples every lambda's PSF onto
    % ONE COMMON PHYSICAL detector grid (the nominal-lambda pitch): a longer
    % lambda is physically larger -> lands on MORE detector pixels -> radial
    % smearing.  This is exactly what macos.compose does internally; we do it
    % in MATLAB because the masks are MATLAB-applied.
    N = opts.model_size;
    dx0 = ref_dxfpa_(opts, wvl0);                              % common grid pitch
    I_coro_sum = zeros(N); I_bare_sum = zeros(N);
    for w = wvls
        [Ic, Ib, dxw] = run_coro_at_wvl_(opts, w, r_fpm_fixed_m);
        I_coro_sum = I_coro_sum + resample_to_(Ic, dxw, dx0);
        I_bare_sum = I_bare_sum + resample_to_(Ib, dxw, dx0);
    end
    I_coro_bb = I_coro_sum / opts.nwf;
    I_bare_bb = I_bare_sum / opts.nwf;

    % ---- monochromatic reference (nominal lambda only; its grid == dx0) -
    [I_coro_mono, I_bare_mono] = run_coro_at_wvl_(opts, wvl0, r_fpm_fixed_m);

    % ---- contrast: mono vs broadband -----------------------------------
    peak_bare_mono = max(I_bare_mono(:));
    peak_bare_bb   = max(I_bare_bb(:));
    dz_mono = dark_zone_metrics(I_coro_mono, peak_bare_mono, lamD0, ...
                                opts.inner_lamD, opts.outer_lamD);
    dz_bb   = dark_zone_metrics(I_coro_bb,   peak_bare_bb,   lamD0, ...
                                opts.inner_lamD, opts.outer_lamD);
    [r_m, c_m] = radial_contrast(I_coro_mono, peak_bare_mono, lamD0, opts.outer_lamD+3);
    [r_b, c_b] = radial_contrast(I_coro_bb,   peak_bare_bb,   lamD0, opts.outer_lamD+3);
    fprintf('[band] dark zone %.0f-%.0f lam/D mean contrast:  mono=%.3e  broadband=%.3e (%.1fx)\n', ...
        opts.inner_lamD, opts.outer_lamD, dz_mono.mean, dz_bb.mean, dz_bb.mean/dz_mono.mean);

    % ---- figure --------------------------------------------------------
    vis='off'; if opts.visible, vis='on'; end
    fig = figure('Visible',vis,'Color','w','Position',[60 60 1300 760]);
    tl = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
    title(tl, sprintf(['CTB finite bandpass -- %d wavelengths over %.0f%% band ', ...
        '(masks rebuilt per \\lambda)'], opts.nwf, 100*opts.band_frac), ...
        'FontWeight','bold','Interpreter','tex');

    w = round(2*(opts.outer_lamD+3)*lamD0);
    ax=nexttile; show_(ax,I_coro_mono,peak_bare_mono,w,'monochromatic FPA');
    ax=nexttile; show_(ax,I_coro_bb,  peak_bare_bb,  w,'broadband FPA (incoherent)');
    ax=nexttile([1 2]); hold(ax,'on'); set(ax,'YScale','log');
    hm=plot(ax,r_m,max(c_m,1e-12),'-','Color',[0 0.35 0.8],'LineWidth',1.6);
    hb=plot(ax,r_b,max(c_b,1e-12),'-','Color',[0.85 0.33 0.10],'LineWidth',1.6);
    xr=[opts.inner_lamD opts.outer_lamD]; yl=ylim(ax);
    p=patch(ax,[xr(1) xr(2) xr(2) xr(1)],[yl(1) yl(1) yl(2) yl(2)], ...
        [0.75 0.85 1.0],'FaceAlpha',0.30,'EdgeColor','none','HandleVisibility','off');
    uistack(p,'bottom');
    grid(ax,'on'); box(ax,'on'); xlabel(ax,'separation (\lambda/D)'); ylabel(ax,'contrast');
    legend(ax,[hm hb],{sprintf('mono (mean %.1e)',dz_mono.mean), ...
        sprintf('broadband (mean %.1e)',dz_bb.mean)},'Location','northeast');
    title(ax,'radial dark-zone contrast: mono vs broadband');

    figpath = fullfile(opts.outdir,'ctb_bandpass.png');
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[band] wrote %s\n', figpath);

    out = struct('wvls',wvls,'lamD0',lamD0,'I_coro_mono',I_coro_mono, ...
        'I_coro_bb',I_coro_bb,'I_bare_bb',I_bare_bb,'dz_mono',dz_mono, ...
        'dz_bb',dz_bb,'r_mono',r_m,'c_mono',c_m,'r_bb',r_b,'c_bb',c_b, ...
        'figure',figpath);
end

% ======================================================================
function [I_coro, I_bare, dxfpa] = run_coro_at_wvl_(opts, wvl, r_fpm_fixed_m)
    e = opts.elt;
    macos.init(opts.model_size); macos.load_rx(opts.rx);
    macos.set_src_wvl(wvl);
    % geometry scales at THIS wavelength (dx_f and lamD scale with lambda)
    g = geom_scales_(opts, e);
    % bare
    macos.intensity(e.DM1);
    I_bare = macos.intensity(e.FPA,'reset_trace',false);
    dxfpa  = abs(macos.dx_at(e.FPA));
    % coronagraphic
    I_coro = run_coro_chain_(opts, g, r_fpm_fixed_m);
end

function dx0 = ref_dxfpa_(opts, wvl0)
    e = opts.elt;
    macos.init(opts.model_size); macos.load_rx(opts.rx); macos.set_src_wvl(wvl0);
    macos.intensity(e.FPA);
    dx0 = abs(macos.dx_at(e.FPA));
end

function J = resample_to_(I, dx_src, dx_dst)
    % resample intensity I (pitch dx_src, centred on the array centre) onto a
    % grid of the SAME size with pitch dx_dst, conserving total flux.  A
    % longer-lambda PSF (larger dx_src) maps to a physically wider footprint
    % on the common dx_dst grid -> the chromatic radial smear.
    if abs(dx_src/dx_dst - 1) < 1e-9, J = I; return; end
    N = size(I,1); c = (N-1)/2;
    [xs,ys] = meshgrid(((0:N-1)-c), ((0:N-1)-c));          % dst pixel indices
    % physical dst coords -> src pixel indices
    xi = xs * (dx_dst/dx_src) + c;
    yi = ys * (dx_dst/dx_src) + c;
    J = interp2(0:N-1, (0:N-1).', I, xi, yi, 'linear', 0);
    % conserve flux: intensity is per-pixel; a src pixel of area dx_src^2
    % maps to dx_dst^2 -> scale by (dx_src/dx_dst)^2
    J = J * (dx_src/dx_dst)^2;
end

function I = run_coro_chain_(opts, g, r_fpm_fixed_m)
    e = opts.elt;
    macos.intensity(e.DM1);
    macos.intensity(e.DM2,'reset_trace',false);
    Ia = macos.intensity(e.Apodizer,'reset_trace',false);
    macos.apodize(e.Apodizer, mask_softcircle_(size(Ia,1), abs(macos.dx_at(e.Apodizer)), ...
        opts.r_apod_m, opts.r_apod_taper_m));
    macos.intensity(e.Apodizer,'reset_trace',false);
    If = macos.intensity(e.FPM,'reset_trace',false);
    % FPM radius: FIXED metres (chromatic) or per-lambda lambda/D (achromatic)
    switch opts.fpm_size
        case 'meters', r_fpm_m = r_fpm_fixed_m;               % subtends varying lam/D
        case 'lamD',   r_fpm_m = opts.r_fpm_lamD * g.lamD_fpm_m; % constant lam/D
    end
    macos.apodize(e.FPM, 1 - mask_harddisk_(size(If,1), g.dx_f, r_fpm_m));
    macos.intensity(e.FPM,'reset_trace',false);
    Il = macos.intensity(e.Lyot,'reset_trace',false);
    macos.apodize(e.Lyot, mask_harddisk_(size(Il,1), abs(macos.dx_at(e.Lyot)), ...
        opts.r_lyot_frac * g.r_lyot_geom_m));
    macos.intensity(e.Lyot,'reset_trace',false);
    I = macos.intensity(e.FPA,'reset_trace',false);
end

function g = geom_scales_(opts, e)
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
end

function lamD = fpa_lamD_px_(opts, wvl)
    e = opts.elt;
    macos.init(opts.model_size); macos.load_rx(opts.rx); macos.set_src_wvl(wvl);
    cbm = macos.cbm(); lambda_m = macos.get_src_wvl()*cbm;
    macos.intensity(e.FPA);
    Iep = macos.intensity(e.ExitPupil,'reset_trace',false);
    Dep = 2*beam_radius_(Iep, abs(macos.dx_at(e.ExitPupil)));
    R_m = abs(macos.get_elt_z(e.ExitPupil))*cbm;
    lamD = (lambda_m * R_m / Dep) / abs(macos.dx_at(e.FPA));
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
function show_(ax, I, peak, w, ttl)
    In = double(I)/max(peak,eps); L=log10(max(In,1e-12));
    imagesc(ax, crop_(L,w)); axis(ax,'image','off'); colormap(ax,parula); clim(ax,[-10 0]);
    cb=colorbar(ax); cb.Label.String='log_{10} contrast'; title(ax,ttl,'Interpreter','tex');
end
function o = crop_(img, w)
    n=size(img,1); if w>=n, o=img; return; end
    c=floor(n/2)+1; lo=max(c-floor(w/2),1); hi=min(lo+w-1,n); o=img(lo:hi,lo:hi);
end
