function out = ctb_planet(opts)
%CTB_PLANET  Off-axis companion in the CTB dark zone (work C).
%   Runs the CTB coronagraph twice on one grid -- an on-axis STAR and a
%   dim OFF-AXIS PLANET -- and forms the incoherent star+planet scene at
%   the FPA, showing the planet standing in the coronagraphic dark zone.
%
%   PLANET INJECTION (MATLAB domain, stage-1 consistent).  The CoroExample
%   WINDOW/FFP recipe does NOT displace a companion on THIS deck: Dave's
%   rule puts every reference-surface vertex ON the chief ray, so after the
%   ffp+ors/fex re-alignment the FPM and FPA grids BOTH re-centre on the
%   (now-tilted) chief -- the planet lands back at pixel centre and the
%   centred occulter clips it (verified empirically).  Instead we inject
%   the companion as a PUPIL PHASE RAMP on the complex field at DM1
%   (macos.apodize_complex): a ramp of k cycles across the pupil diameter
%   focuses the planet at exactly k lambda/D off-centre, at the FIXED
%   centred FPM (so the planet image clears the occulter while the on-axis
%   star is blocked) and at the FPA.  This is the same "modify the complex
%   field in MATLAB" contract as the masks -- no engine source tilt, no
%   reference re-alignment.
%
%   The star and planet are propagated through the SAME centred masks (the
%   occulter/Lyot are fixed on the array), summed incoherently (they are
%   mutually incoherent sources), with the planet scaled to a chosen
%   star-planet flux ratio.
%
%   out = CTB_PLANET() places a companion at 6 lambda/D, flux ratio 1e-3
%   relative to the star's UNOCCULTED peak.  Name-value:
%     'rx','elt'          deck + station map (default compact ctb_dcr.in).
%     'model_size'        grid (512).
%     'sep_lamD'          companion separation, lambda/D (default 6).
%     'pa_deg'            position angle, deg CCW from +x (default 0).
%     'flux_ratio'        planet/star flux (default 1e-3).
%     'r_fpm_lamD','r_apod_m','r_apod_taper_m','r_lyot_frac'  mask params.
%     'inner_lamD','outer_lamD'  dark-zone annulus for the contrast ref.
%     'outdir','visible'.
%
%   See also: ctb_coro_compare, ctb_contrast, macos.apodize_complex.
    arguments
        opts.rx            (1,:) char   = ''
        opts.elt           struct = struct('DM1',2,'DM2',5,'Apodizer',13, ...
                                'FPM',17,'Lyot',20,'ExitPupil',30,'FPA',31)
        opts.model_size    (1,1) double = 512
        opts.sep_lamD      (1,1) double = 6.0
        opts.pa_deg        (1,1) double = 0.0
        opts.flux_ratio    (1,1) double = 1e-3
        opts.r_fpm_lamD    (1,1) double = 3.0
        opts.r_apod_m      (1,1) double = 15e-3
        opts.r_apod_taper_m(1,1) double = 2e-3
        opts.r_lyot_frac   (1,1) double = 0.85
        opts.view_lamD     (1,1) double = 18.0
        opts.outdir        (1,:) char   = ''
        opts.visible       (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    if isempty(opts.rx),     opts.rx     = fullfile(here,'ctb_dcr.in'); end
    if isempty(opts.outdir), opts.outdir = here; end
    addpath(fullfile(here,'..','..','..','src'));
    assert(~isempty(getenv('MACOS_HOME')),'MACOS_HOME must be set.');
    e = opts.elt;

    % geometry scales (deterministic) -- FPM sizing + FPA lambda/D + pupil dia
    g = geom_scales_(opts.rx, e, opts.model_size);
    fprintf('[planet] lamD_fpa=%.3f px  pupil=%.1f px  sep=%.1f lam/D  flux=%.1e\n', ...
        g.lamD_fpa_px, g.pupil_px, opts.sep_lamD, opts.flux_ratio);

    % ---- STAR: on-axis coronagraphic FPA (unocculted peak for norm) ----
    [I_star, peak_unocc] = run_star_(opts, g);

    % ---- PLANET: pupil phase ramp -> off-axis, same centred masks ------
    kx = opts.sep_lamD * cosd(opts.pa_deg);
    ky = opts.sep_lamD * sind(opts.pa_deg);
    I_planet_raw = run_planet_(opts, g, kx, ky);

    % ---- incoherent scene ----------------------------------------------
    % Both star and planet fields were propagated through the SAME
    % coronagraph at unit source flux, so their FPA intensities are already
    % correctly suppressed relative to the same unocculted reference peak.
    % The planet is astrophysically fainter by flux_ratio -> scale and add
    % incoherently (star and planet are mutually incoherent sources).
    I_planet = I_planet_raw * opts.flux_ratio;
    I_scene  = I_star + I_planet;
    I_diff   = max(I_scene - I_star, 0);

    % planet peak location
    [~,ip] = max(I_planet(:)); [pr,pc] = ind2sub(size(I_planet),ip);
    cen = floor(opts.model_size/2)+1;
    off = hypot(pr-cen,pc-cen)/g.lamD_fpa_px;
    fprintf('[planet] planet peak @ [%d %d] = %.2f lam/D from centre (target %.1f)\n', ...
        pr, pc, off, opts.sep_lamD);
    fprintf('[planet] star residual peak=%.3e  planet peak=%.3e (norm to unocc star)\n', ...
        max(I_star(:))/peak_unocc, max(I_planet(:))/peak_unocc);

    % ---- figure: star | planet | scene | difference --------------------
    vis = 'off'; if opts.visible, vis='on'; end
    fig = figure('Visible',vis,'Color','w','Position',[60 60 1300 360]);
    tl = tiledlayout(fig,1,4,'TileSpacing','compact','Padding','compact');
    title(tl, sprintf(['CTB planet injection -- companion at %.1f \\lambda/D, ', ...
        'flux ratio %.0e (log_{10} contrast)'], opts.sep_lamD, opts.flux_ratio), ...
        'FontWeight','bold','Interpreter','tex');
    w = round(2*opts.view_lamD * g.lamD_fpa_px);
    show_(tl, I_star,   peak_unocc, w, g.lamD_fpa_px, 'suppressed star');
    show_(tl, I_planet, peak_unocc, w, g.lamD_fpa_px, sprintf('planet alone (%.1f \\lambda/D)',opts.sep_lamD));
    show_(tl, I_scene,  peak_unocc, w, g.lamD_fpa_px, 'star + planet (incoherent)');
    show_(tl, I_diff,   peak_unocc, w, g.lamD_fpa_px, 'difference: scene - star');

    figpath = fullfile(opts.outdir,'ctb_planet.png');
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[planet] wrote %s\n', figpath);

    out = struct('I_star',I_star,'I_planet',I_planet,'I_scene',I_scene, ...
        'I_diff',I_diff,'peak_unocc',peak_unocc,'lamD_fpa_px',g.lamD_fpa_px, ...
        'sep_lamD',opts.sep_lamD,'pa_deg',opts.pa_deg,'flux_ratio',opts.flux_ratio, ...
        'planet_peak_lamD',off,'figure',figpath);
end

% ======================================================================
function [I_star, peak_unocc] = run_star_(opts, g)
    e = opts.elt;
    % unocculted reference peak (masks OFF) for astrophysical normalisation
    macos.init(opts.model_size); macos.load_rx(opts.rx);
    macos.intensity(e.DM1);
    I_unocc = macos.intensity(e.FPA,'reset_trace',false);
    peak_unocc = max(I_unocc(:));
    % coronagraphic star
    I_star = run_coro_chain_(opts, g, []);
end

function I = run_planet_(opts, g, kx, ky)
    % pupil phase ramp injected at DM1: k cycles across the pupil diameter
    N = opts.model_size; c = (N-1)/2;
    [xx,yy] = meshgrid((0:N-1)-c,(0:N-1)-c);
    ramp = exp(1i*2*pi*(kx*xx + ky*yy)/g.pupil_px);
    I = run_coro_chain_(opts, g, ramp);
end

function I = run_coro_chain_(opts, g, pupil_ramp)
    % one coronagraphic forward pass with centred MATLAB masks; if
    % pupil_ramp is non-empty it is multiplied onto the complex field at
    % DM1 (the planet tilt).
    e = opts.elt;
    macos.init(opts.model_size); macos.load_rx(opts.rx);
    macos.intensity(e.DM1);
    if ~isempty(pupil_ramp)
        macos.apodize_complex(e.DM1, pupil_ramp);
    end
    macos.intensity(e.DM1,'reset_trace',false);
    macos.intensity(e.DM2,'reset_trace',false);
    % apodizer
    Ia = macos.intensity(e.Apodizer,'reset_trace',false);
    M = mask_softcircle_(size(Ia,1), abs(macos.dx_at(e.Apodizer)), ...
                         opts.r_apod_m, opts.r_apod_taper_m);
    macos.apodize(e.Apodizer, M);
    macos.intensity(e.Apodizer,'reset_trace',false);
    % FPM occulter (centred, deterministic sizing)
    If = macos.intensity(e.FPM,'reset_trace',false);
    r_fpm_m = opts.r_fpm_lamD * g.lamD_fpm_m;
    Mf = 1 - mask_harddisk_(size(If,1), g.dx_f, r_fpm_m);
    macos.apodize(e.FPM, Mf);
    macos.intensity(e.FPM,'reset_trace',false);
    % Lyot
    Il = macos.intensity(e.Lyot,'reset_trace',false);
    Ml = mask_harddisk_(size(Il,1), abs(macos.dx_at(e.Lyot)), ...
                        opts.r_lyot_frac * g.r_lyot_geom_m);
    macos.apodize(e.Lyot, Ml);
    macos.intensity(e.Lyot,'reset_trace',false);
    % FPA
    I = macos.intensity(e.FPA,'reset_trace',false);
end

% ======================================================================
function g = geom_scales_(rx, e, N)
    macos.init(N); macos.load_rx(rx);
    cbm = macos.cbm(); lambda_m = macos.get_src_wvl()*cbm;
    macos.intensity(e.FPM);
    Isph = macos.intensity(e.FPM-1,'reset_trace',false);
    dx_sph = abs(macos.dx_at(e.FPM-1));
    R_fpm = abs(macos.get_elt_z(e.FPM-1))*cbm;
    Dbeam = 2*beam_radius_(Isph, dx_sph);
    g.dx_f       = lambda_m * R_fpm / (N*dx_sph);
    g.lamD_fpm_m = lambda_m * R_fpm / Dbeam;
    Ily = macos.intensity(e.Lyot,'reset_trace',false);
    g.r_lyot_geom_m = beam_radius_(Ily, abs(macos.dx_at(e.Lyot)));
    % FPA lambda/D + pupil diameter (px) at DM1
    macos.intensity(e.FPA);
    Iep = macos.intensity(e.ExitPupil,'reset_trace',false);
    Dep = 2*beam_radius_(Iep, abs(macos.dx_at(e.ExitPupil)));
    R_fpa = abs(macos.get_elt_z(e.ExitPupil))*cbm;
    dxfpa = abs(macos.dx_at(e.FPA));
    g.lamD_fpa_px = (lambda_m * R_fpa / Dep) / dxfpa;
    Idm = macos.intensity(e.DM1,'reset_trace',false);
    dxdm = abs(macos.dx_at(e.DM1));
    g.pupil_px = 2*beam_radius_(Idm, dxdm) / dxdm;
end

function rr = beam_radius_(I, dx)
    thr = 0.02*max(I(:)); [yy,xx] = find(I>thr);
    if isempty(xx), rr=0; return; end
    c = (size(I,1)-1)/2 + 1; rr = max(hypot(xx-c,yy-c))*dx;
end

function M = mask_harddisk_(N, dx, r_m), M = disk_ss_(N,dx,r_m,8); end

function M = mask_softcircle_(N, dx, r0_m, sigma_m)
    r1 = r0_m + 4*sigma_m; base = disk_ss_(N,dx,r1,8);
    c=(N-1)/2; [xx,yy]=meshgrid(0:N-1,0:N-1); rr=hypot(xx-c,yy-c)*dx;
    tap=ones(N); out=rr>r0_m; tap(out)=exp(-((rr(out)-r0_m)/sigma_m).^2);
    M = base.*tap;
end

function M = disk_ss_(N, dx, r_m, K)
    c=(N-1)/2; off=((0:K-1)-(K-1)/2)/K; M=zeros(N);
    [ox,oy]=meshgrid(off,off); ox=ox(:).'; oy=oy(:).';
    for i=1:N
        yc=(i-1-c); xs=((0:N-1)-c).'; acc=zeros(N,1);
        for s=1:numel(ox)
            xx=(xs+ox(s))*dx; yy=(yc+oy(s))*dx;
            acc=acc+double(xx.^2+yy.^2<=r_m^2);
        end
        M(i,:)=acc.'/numel(ox);
    end
end

function show_(tl, I, peak_unocc, w, lamD, ttl)
    nexttile(tl);
    In = double(I)/max(peak_unocc,eps);
    L = log10(max(In,1e-12));
    imagesc(crop_(L,w)); axis image off; colormap(gca,parula); clim([-10 0]);
    cb=colorbar; cb.Label.String='log_{10} contrast';
    title(ttl,'Interpreter','tex');
end

function o = crop_(img, w)
    n=size(img,1); if w>=n, o=img; return; end
    c=floor(n/2)+1; lo=max(c-floor(w/2),1); hi=min(lo+w-1,n); o=img(lo:hi,lo:hi);
end
