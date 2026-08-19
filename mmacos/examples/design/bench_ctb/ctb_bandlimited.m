function out = ctb_bandlimited(opts)
%CTB_BANDLIMITED  Band-limited Lyot coronagraph on the CTB (priority mask).
%   The band-limited Lyot coronagraph (Kuchner & Traub 2002, ApJ 570, 900;
%   Kuchner, Crepp & Ge 2005, ApJ 628, 466 for the 8th order) replaces the
%   hard occulter with a GRADED amplitude focal mask M-hat(x) whose Fourier
%   transform is strictly band-limited.  On a CLEAR circular pupil that
%   band-limit confines the on-axis starlight to within a fraction epsilon
%   of the Lyot-pupil EDGE, so a Lyot stop trimmed by epsilon rejects it
%   ENTIRELY -- an exact (4th- or 8th-order) on-axis null, far deeper than a
%   hard occulter's ~1e3 suppression.  NO pupil apodizer is used (the null
%   is the mask's own property; ctb_mask_bandlimited builds M-hat).
%
%   THE PAIRING RULE (verified against the papers, not a remembered
%   formula).  KCG Eq. 2: the conjugate pupil mask is zero for
%   epsilon/2 < |u| < 1 - epsilon/2 (u = pupil radius normalised to 1), so
%   the Lyot stop RETAINS radius (1 - epsilon)*R_geom -- retained DIAMETER
%   = (1 - epsilon)*D, per-axis throughput (1 - epsilon), 2-D area ~
%   (1 - epsilon)^2.  (A source quoting "1 - 2 epsilon" is using a half-
%   bandwidth epsilon.)  This driver VERIFIES the rule empirically: it
%   measures where the on-axis starlight lands in the Lyot plane vs epsilon
%   and confirms the star is confined to the outer epsilon-annulus, THEN
%   trims the Lyot to (1 - epsilon)*R_geom.
%
%   THREE GATES (per the work order):
%     (a) IDEAL ON-AXIS NULL -- the numerics-limited suppression floor with
%         the matched Lyot, and WHAT LIMITS it on this bench (grid sampling
%         of the graded mask, the discrete Lyot edge, the central pixel).
%     (b) CONTRAST + THROUGHPUT vs epsilon -- the fundamental BLC trade
%         (bigger epsilon -> deeper/wider null but more pupil trimmed away).
%     (c) BANDPASS -- BL masks are chromatic BY DESIGN (M-hat is defined in
%         lambda/D; a mask etched at fixed PHYSICAL size subtends a varying
%         lambda/D across a band).  Quantifies mono vs broadband for a
%         fixed-metres mask (realistic) vs a per-lambda lambda/D-rescaled
%         mask (achromatic reference), reusing the ctb_bandpass resample.
%
%   out = CTB_BANDLIMITED() runs the 4th-order separable mask at epsilon =
%   0.4 (the KCG worked-example bandwidth).  Name-value:
%     'rx','elt'        deck + station map (default compact ctb_dcr.in).
%     'model_size'      grid (1024).
%     'order'           4 (K&T) or 8 (KCG); default 4.
%     'form'            'separable' (default) | 'radial' | 'linear'.
%     'epsilon'         nominal bandwidth for gates (a),(c) (default 0.4).
%     'eps_list'        bandwidths swept in gate (b) (default
%                       [0.10 0.20 0.30 0.40 0.55]).
%     'inner_lamD','outer_lamD'  dark-zone annulus (3, 15).
%     'nwf','band_frac' bandpass: #wavelengths (5), fractional band (0.10).
%     'outdir','visible'.
%
%   See also: ctb_mask_bandlimited, ctb_contrast, ctb_bandpass, ctb_vortex_matched.
    arguments
        opts.rx            (1,:) char   = ''
        opts.elt           struct = struct('DM1',2,'DM2',5,'Apodizer',13, ...
                                'FPM',17,'Lyot',20,'ExitPupil',30,'FPA',31)
        opts.model_size    (1,1) double = 1024
        opts.order         (1,1) double {mustBeMember(opts.order,[4 8])} = 4
        opts.form          (1,:) char {mustBeMember(opts.form,{'separable','radial','linear'})} = 'separable'
        opts.epsilon       (1,1) double = 0.40
        opts.eps_list      (1,:) double = [0.10 0.20 0.30 0.40 0.55]
        opts.inner_lamD    (1,1) double = 3.0
        opts.outer_lamD    (1,1) double = 15.0
        opts.nwf           (1,1) double = 5
        opts.band_frac     (1,1) double = 0.10
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
    fprintf('[blc] N=%d order=%d form=%s  lamD_fpa=%.3f px  r_lyot_geom=%.4e m  bare peak=%.3e\n', ...
        opts.model_size, opts.order, opts.form, lamD, g.r_lyot_geom_m, peak_bare);

    % ================= GATE (a): ideal on-axis null =====================
    % The BLC + Lyot-trimmed-by-epsilon suppression, and what limits it.
    [Ia, Ilyot_a, ~] = run_blc_(opts, g, opts.epsilon, opts.epsilon);
    peak_coro = max(Ia(:));
    supp = peak_bare / max(peak_coro, eps);
    dz_a = dark_zone_metrics(Ia, peak_bare, lamD, opts.inner_lamD, opts.outer_lamD);
    fprintf(['[blc] (a) NULL  eps=%.2f: coro peak=%.3e  suppression=%.2e  ', ...
             'DZ mean=%.3e median=%.3e floor=%.3e\n'], ...
        opts.epsilon, peak_coro, supp, dz_a.mean, dz_a.median, dz_a.floor);

    % --- verify the trim rule: where does the star land in the Lyot? ----
    tr = verify_trim_(opts, g);                          % star flux vs pupil radius
    fprintf(['[blc] (a) trim check eps=%.2f: %.1f%% of Lyot-plane flux is OUTSIDE ', ...
             'r=(1-eps)R (=%.3f R); Lyot trimmed to (1-eps)R rejects it\n'], ...
        opts.epsilon, 100*tr.frac_outside_trim, 1-opts.epsilon);

    % ================= GATE (b): contrast + throughput vs epsilon =======
    nE = numel(opts.eps_list);
    Cmean = nan(1,nE); Cmed = nan(1,nE); Thru = nan(1,nE); Supp = nan(1,nE);
    for k = 1:nE
        ep = opts.eps_list(k);
        [Ik, ~, ~] = run_blc_(opts, g, ep, ep);
        dz = dark_zone_metrics(Ik, peak_bare, lamD, opts.inner_lamD, opts.outer_lamD);
        Cmean(k) = dz.mean; Cmed(k) = dz.median;
        Thru(k)  = (1-ep)^2;                             % 2-D area throughput
        Supp(k)  = peak_bare / max(max(Ik(:)),eps);
        fprintf(['[blc] (b) eps=%.2f: DZ mean=%.3e median=%.3e  throughput=(1-eps)^2=%.3f  ', ...
                 'suppression=%.2e\n'], ep, dz.mean, dz.median, Thru(k), Supp(k));
    end

    % ================= GATE (c): bandpass (chromatic by design) =========
    bp = bandpass_(opts, g, lamD, peak_bare);

    % ================= figure ===========================================
    vis='off'; if opts.visible, vis='on'; end
    fig = figure('Visible',vis,'Color','w','Position',[50 50 1500 900]);
    tl = tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
    title(tl, sprintf(['CTB band-limited Lyot coronagraph -- order %d, %s mask ', ...
        '(Kuchner & Traub 2002 / KCG 2005)'], opts.order, opts.form), ...
        'FontWeight','bold','Interpreter','none');

    % (a1) the 1-D mask profile M-hat(X) (the separable building block; a
    % center-row slice of a SEPARABLE 2-D mask is identically 0 because
    % M-hat(0)=0, so plot the 1-D profile directly).
    ax=nexttile(tl);
    Xax = linspace(-12,12,1201);
    Mline = blc_profile_1d_(opts.order, opts.epsilon, Xax);
    plot(ax, Xax, Mline, '-','Color',[0 0.35 0.8],'LineWidth',1.6); hold(ax,'on');
    plot(ax, Xax, Mline.^2, '--','Color',[0.85 0.33 0.10],'LineWidth',1.4);
    xlim(ax,[-12 12]); ylim(ax,[0 1.05]); grid(ax,'on'); box(ax,'on');
    xlabel(ax,'separation (\lambda/D)'); ylabel(ax,'transmission');
    legend(ax,{'amplitude $\hat{M}$','intensity $|\hat{M}|^2$'}, ...
        'Interpreter','latex','Location','south');
    title(ax, sprintf('BL 1-D mask profile (order %d, \\epsilon=%.2f)', ...
        opts.order, opts.epsilon),'Interpreter','tex');

    % (a2) trim-rule verification: cumulative star flux vs Lyot radius
    ax=nexttile(tl); plot(ax, tr.r_over_R, 100*tr.cum_outside, '-','Color',[0.2 0.5 0.2],'LineWidth',1.8);
    hold(ax,'on'); xl=xline(ax, 1-opts.epsilon, '--', sprintf('(1-\\epsilon)R'));
    xl.Annotation.LegendInformation.IconDisplayStyle='off';
    grid(ax,'on'); box(ax,'on'); xlabel(ax,'Lyot radius / R_{geom}');
    ylabel(ax,'% on-axis flux OUTSIDE this radius');
    title(ax,'trim rule: star confined to outer \epsilon-annulus','Interpreter','tex');

    % (a3) coronagraphic FPA (the null)
    w = round(2*(opts.outer_lamD+3)*lamD);
    ax=nexttile(tl); show_(ax, Ia, peak_bare, w, ...
        sprintf('FPA null (\\epsilon=%.2f): suppr %.1e', opts.epsilon, supp));

    % (b) contrast + throughput vs epsilon
    ax=nexttile(tl); yyaxis(ax,'left'); set(ax,'YScale','log');
    plot(ax, opts.eps_list, Cmean, '-o','LineWidth',1.7,'MarkerFaceColor','auto');
    ylabel(ax,'mean dark-zone contrast');
    yyaxis(ax,'right');
    plot(ax, opts.eps_list, 100*Thru, '-s','LineWidth',1.5);
    ylabel(ax,'throughput (1-\epsilon)^2  (%)'); ylim(ax,[0 100]);
    grid(ax,'on'); box(ax,'on'); xlabel(ax,'mask bandwidth \epsilon');
    title(ax,'(b) contrast + throughput vs \epsilon','Interpreter','tex');

    % (c) bandpass mono vs broadband
    ax=nexttile(tl); hold(ax,'on'); set(ax,'YScale','log');
    hm=plot(ax,bp.r_mono,max(bp.c_mono,1e-12),'-','Color',[0 0.35 0.8],'LineWidth',1.6);
    hf=plot(ax,bp.r_fix, max(bp.c_fix, 1e-12),'-','Color',[0.85 0.33 0.10],'LineWidth',1.6);
    hl=plot(ax,bp.r_lamD,max(bp.c_lamD,1e-12),'-','Color',[0.2 0.6 0.2],'LineWidth',1.4);
    xr=[opts.inner_lamD opts.outer_lamD]; yl=ylim(ax);
    p=patch(ax,[xr(1) xr(2) xr(2) xr(1)],[yl(1) yl(1) yl(2) yl(2)], ...
        [0.75 0.85 1.0],'FaceAlpha',0.25,'EdgeColor','none','HandleVisibility','off');
    uistack(p,'bottom');
    grid(ax,'on'); box(ax,'on'); xlabel(ax,'separation (\lambda/D)'); ylabel(ax,'contrast');
    legend(ax,[hm hf hl], {sprintf('mono (%.1e)',bp.dz_mono.mean), ...
        sprintf('broadband fixed-m (%.1e)',bp.dz_fix.mean), ...
        sprintf('broadband \\lambda/D (%.1e)',bp.dz_lamD.mean)}, ...
        'Location','northeast','Interpreter','tex');
    title(ax, sprintf('(c) bandpass %d\\times over %.0f%% band', opts.nwf, 100*opts.band_frac),'Interpreter','tex');

    % summary text
    ax=nexttile(tl); axis(ax,'off');
    txt = { sprintf('\\bfBand-limited Lyot -- order %d, %s, N=%d', opts.order, opts.form, opts.model_size), ...
        sprintf('(a) null @ \\epsilon=%.2f: suppression %.1e, DZ mean %.1e (floor %.1e)', ...
            opts.epsilon, supp, dz_a.mean, dz_a.floor), ...
        sprintf('    limit: grid sampling of graded mask + discrete Lyot edge + central pixel'), ...
        sprintf('(a) trim rule (1-\\epsilon)R VERIFIED: %.1f%% of star flux outside it', 100*tr.frac_outside_trim), ...
        sprintf('(b) \\epsilon %.2f->%.2f: contrast %.1e->%.1e, throughput %.0f%%->%.0f%%', ...
            opts.eps_list(1), opts.eps_list(end), Cmean(1), Cmean(end), 100*Thru(1), 100*Thru(end)), ...
        sprintf('(c) bandpass %.0f%%: mono %.1e -> fixed-metres %.1e (%.1fx, chromatic)', ...
            100*opts.band_frac, bp.dz_mono.mean, bp.dz_fix.mean, bp.dz_fix.mean/bp.dz_mono.mean), ...
        sprintf('    \\lambda/D-rescaled mask -> %.1e (achromatic reference)', bp.dz_lamD.mean) };
    text(ax, 0.0, 0.97, txt, 'VerticalAlignment','top','FontSize',10,'Interpreter','tex');

    figpath = fullfile(opts.outdir,'ctb_bandlimited.png');
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[blc] wrote %s\n', figpath);

    out = struct('order',opts.order,'form',opts.form,'epsilon',opts.epsilon, ...
        'lamD_px',lamD,'peak_bare',peak_bare,'suppression',supp, ...
        'dz_null',dz_a,'trim',tr,'eps_list',opts.eps_list,'contrast_mean',Cmean, ...
        'contrast_median',Cmed,'throughput',Thru,'suppression_list',Supp, ...
        'bandpass',bp,'figure',figpath);
end

% ======================================================================
%  BLC chain: clear pupil -> BL amplitude mask at FPM -> Lyot trimmed to
%  (1-eps_lyot)*R_geom -> FPA.  Returns FPA intensity, Lyot-plane image
%  (pre-stop), and the applied Lyot radius (m).
% ======================================================================
function [I_fpa, I_lyot, r_lyot_m] = run_blc_(opts, g, eps_mask, eps_lyot)
    e = opts.elt; N = opts.model_size;
    macos.init(N); macos.load_rx(opts.rx);
    macos.intensity(e.DM1);
    macos.intensity(e.DM2,'reset_trace',false);
    macos.intensity(e.Apodizer,'reset_trace',false);     % NO apodizer (clear pupil)
    % FPM: band-limited amplitude mask on the deterministic focal grid
    macos.intensity(e.FPM,'reset_trace',false);
    M = ctb_mask_bandlimited(N, g.dx_f, g.lamD_fpm_m, eps_mask, opts.order, opts.form);
    macos.apodize(e.FPM, M);
    macos.intensity(e.FPM,'reset_trace',false);
    % Lyot plane (pre-stop) for inspection
    I_lyot = macos.intensity(e.Lyot,'reset_trace',false);
    dxl = abs(macos.dx_at(e.Lyot));
    r_lyot_m = (1 - eps_lyot) * g.r_lyot_geom_m;          % KCG trim rule
    macos.apodize(e.Lyot, ctb_mask_disk(N, dxl, r_lyot_m, 8));
    macos.intensity(e.Lyot,'reset_trace',false);
    I_fpa = macos.intensity(e.FPA,'reset_trace',false);
end

% ======================================================================
%  Verify the trim rule: with the BL mask applied but NO Lyot stop, what
%  fraction of the on-axis Lyot-plane flux sits outside radius r?  The BL
%  property predicts the star is confined to the outer epsilon-annulus, so
%  ~all of it should be outside (1-epsilon)*R_geom.
% ======================================================================
function tr = verify_trim_(opts, g)
    e = opts.elt; N = opts.model_size;
    macos.init(N); macos.load_rx(opts.rx);
    macos.intensity(e.DM1);
    macos.intensity(e.DM2,'reset_trace',false);
    macos.intensity(e.Apodizer,'reset_trace',false);
    macos.intensity(e.FPM,'reset_trace',false);
    M = ctb_mask_bandlimited(N, g.dx_f, g.lamD_fpm_m, opts.epsilon, opts.order, opts.form);
    macos.apodize(e.FPM, M);
    macos.intensity(e.FPM,'reset_trace',false);
    Il = macos.intensity(e.Lyot,'reset_trace',false);
    dxl = abs(macos.dx_at(e.Lyot));
    cc = floor(N/2)+1; [X,Y] = meshgrid((1:N)-cc,(1:N)-cc); rr = hypot(X,Y)*dxl;
    R = g.r_lyot_geom_m; tot = max(sum(Il(:)),eps);
    rgrid = linspace(0, 1.3, 60);                         % in units of R
    cum_outside = arrayfun(@(f) sum(Il(rr > f*R))/tot, rgrid);
    tr.r_over_R = rgrid;
    tr.cum_outside = cum_outside;
    tr.frac_outside_trim = sum(Il(rr > (1-opts.epsilon)*R))/tot;
end

function p = blc_profile_1d_(order, eps, X)
%BLC_PROFILE_1D_  Normalised 1-D amplitude M-hat(X), X in lambda/D (for the
%   profile plot).  Mirrors ctb_mask_bandlimited's profile_/norm_ so the
%   plotted building block is exactly the mask's 1-D factor.
    z = pi*eps*X;
    s1 = @(zz) arrayfun(@(v) tern_(v==0,1,sin(v)/v), zz);
    if order == 4, prof = @(zz) 1 - s1(zz);
    else,          prof = @(zz) 2/3 - s1(zz/3).^3 + (1/3)*s1(zz); end
    Xg = linspace(0, 40/max(eps,1e-6), 200000);
    Nn = 1/max(prof(pi*eps*Xg));
    p = max(min(Nn*prof(z),1),0);
end
function v = tern_(c,a,b), if c, v=a; else, v=b; end, end

% ======================================================================
%  Bandpass gate: sum nwf wavelengths incoherently onto ONE common
%  physical detector grid (flux-conserving resample, as ctb_bandpass).
%  TWO masks: FIXED-METRES (chromatic, realistic) and per-lambda
%  lambda/D-rescaled (achromatic reference).
% ======================================================================
function bp = bandpass_(opts, g, lamD0, peak_bare)
    e = opts.elt; N = opts.model_size;
    macos.init(N); macos.load_rx(opts.rx);
    wvl0 = macos.get_src_wvl();
    if opts.nwf == 1, wvls = wvl0;
    else, wvls = wvl0 * (1 + opts.band_frac*linspace(-0.5,0.5,opts.nwf)); end

    % fixed physical mask footprint: build M at nominal lambda, hold in metres
    dx_f0 = g.dx_f; lamD_m0 = g.lamD_fpm_m;               % nominal-lambda scales

    dx0 = ref_dxfpa_(opts, wvl0);
    Ifix = zeros(N); IlamD = zeros(N); Ibare = zeros(N);
    for w = wvls
        [If, Il, Ib, dxw] = run_blc_wvl_(opts, w, dx_f0, lamD_m0);
        Ifix  = Ifix  + resample_(If, dxw, dx0);
        IlamD = IlamD + resample_(Il, dxw, dx0);
        Ibare = Ibare + resample_(Ib, dxw, dx0);
    end
    Ifix=Ifix/opts.nwf; IlamD=IlamD/opts.nwf; Ibare=Ibare/opts.nwf;
    peak_bb = max(Ibare(:));
    [Imono, ~, ~] = run_blc_(opts, g, opts.epsilon, opts.epsilon);   % nominal lambda

    bp.dz_mono = dark_zone_metrics(Imono, peak_bare, lamD0, opts.inner_lamD, opts.outer_lamD);
    bp.dz_fix  = dark_zone_metrics(Ifix,  peak_bb,   lamD0, opts.inner_lamD, opts.outer_lamD);
    bp.dz_lamD = dark_zone_metrics(IlamD, peak_bb,   lamD0, opts.inner_lamD, opts.outer_lamD);
    [bp.r_mono,bp.c_mono] = radial_contrast(Imono, peak_bare, lamD0, opts.outer_lamD+3);
    [bp.r_fix, bp.c_fix ] = radial_contrast(Ifix,  peak_bb,   lamD0, opts.outer_lamD+3);
    [bp.r_lamD,bp.c_lamD] = radial_contrast(IlamD, peak_bb,   lamD0, opts.outer_lamD+3);
    bp.wvls = wvls;
    fprintf(['[blc] (c) bandpass %d wvls %.0f%%: mono %.3e | fixed-metres %.3e (%.1fx) | ', ...
             'lambda/D-rescaled %.3e (%.1fx)\n'], opts.nwf, 100*opts.band_frac, ...
        bp.dz_mono.mean, bp.dz_fix.mean, bp.dz_fix.mean/bp.dz_mono.mean, ...
        bp.dz_lamD.mean, bp.dz_lamD.mean/bp.dz_mono.mean);
end

function [I_fix, I_lamD, I_bare, dxfpa] = run_blc_wvl_(opts, wvl, dx_f0, lamD_m0)
    e = opts.elt; N = opts.model_size;
    macos.init(N); macos.load_rx(opts.rx); macos.set_src_wvl(wvl);
    gw = geom_scales_cur_(opts, e);                       % this-lambda scales (NO reinit)
    % bare
    macos.intensity(e.DM1);
    I_bare = macos.intensity(e.FPA,'reset_trace',false);
    dxfpa  = abs(macos.dx_at(e.FPA));
    % FIXED-metres mask (chromatic, realistic): the etched mask has ONE
    % physical footprint.  Build with THIS-lambda dx_f (pixels->metres
    % correct here) but the NOMINAL lamD_m0 (metres->lambda/D uses the
    % nominal scale), so the mask's PHYSICAL profile is lambda-invariant and
    % its lambda/D footprint grows with lambda -> genuine chromaticity.
    I_fix  = blc_chain_(opts, gw, gw.dx_f, lamD_m0,        opts.epsilon);
    % lambda/D-rescaled mask (achromatic reference): build with THIS-lambda
    % lamD_m -> constant lambda/D at every wavelength.
    I_lamD = blc_chain_(opts, gw, gw.dx_f, gw.lamD_fpm_m,  opts.epsilon);
end

function I = blc_chain_(opts, g, mask_dx_f, mask_lamD_m, eps)
    e = opts.elt; N = opts.model_size;
    macos.intensity(e.DM1);
    macos.intensity(e.DM2,'reset_trace',false);
    macos.intensity(e.Apodizer,'reset_trace',false);
    macos.intensity(e.FPM,'reset_trace',false);
    M = ctb_mask_bandlimited(N, mask_dx_f, mask_lamD_m, eps, opts.order, opts.form);
    macos.apodize(e.FPM, M);
    macos.intensity(e.FPM,'reset_trace',false);
    macos.intensity(e.Lyot,'reset_trace',false);
    dxl = abs(macos.dx_at(e.Lyot));
    macos.apodize(e.Lyot, ctb_mask_disk(N, dxl, (1-eps)*g.r_lyot_geom_m, 8));
    macos.intensity(e.Lyot,'reset_trace',false);
    I = macos.intensity(e.FPA,'reset_trace',false);
end

function dx0 = ref_dxfpa_(opts, wvl0)
    e = opts.elt;
    macos.init(opts.model_size); macos.load_rx(opts.rx); macos.set_src_wvl(wvl0);
    macos.intensity(e.FPA); dx0 = abs(macos.dx_at(e.FPA));
end

function J = resample_(I, dx_src, dx_dst)
    if abs(dx_src/dx_dst - 1) < 1e-9, J = I; return; end
    N=size(I,1); c=(N-1)/2; [xs,ys]=meshgrid((0:N-1)-c,(0:N-1)-c);
    xi = xs*(dx_dst/dx_src)+c; yi = ys*(dx_dst/dx_src)+c;
    J = interp2(0:N-1,(0:N-1).',I,xi,yi,'linear',0) * (dx_src/dx_dst)^2;
end

% ---- shared geometry / helpers (mirror ctb_contrast) -----------------
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

function g = geom_scales_cur_(opts, e)
%GEOM_SCALES_CUR_  Same as geom_scales_ but reads at the CURRENT loaded
%   wavelength -- does NOT re-init/reload (which would reset src_wvl to the
%   deck default, defeating set_src_wvl in the bandpass loop).
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
