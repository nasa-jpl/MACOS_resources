function out = ctb_optimize_masks(opts)
%CTB_OPTIMIZE_MASKS  Sweep FPM occulter radius x Lyot stop fraction to
%   MINIMISE mean dark-zone contrast in a target band (default 3-7 lambda/D).
%
%   Classical-Lyot design box from the literature (Wikipedia "Coronagraph";
%   Sivaramakrishnan et al. 2001, ApJ 552, 397; the HCIT/CoroExample class):
%   occulting spot radius ~1-3 lambda/D, Lyot stop ~75-90% of the geometric
%   pupil.  The BEST combination for a given dark-zone band on a SPECIFIC
%   bench is not analytic (it depends on the residual diffraction structure),
%   so this is an empirical 2-D sweep over that box, scored on the mean
%   per-pixel contrast (dark_zone_metrics.mean) over [band_inner, band_outer]
%   lambda/D, Strehl-normalised to the bare peak.
%
%   Smaller occulter -> smaller inner working angle but more starlight leaks;
%   more Lyot undersizing -> better diffracted-light rejection but lower
%   throughput and larger effective lambda/D.  The sweep finds the knee.
%
%   out = CTB_OPTIMIZE_MASKS() sweeps a default 4x4 grid at N=512 (fast),
%   prints a table, writes a heatmap, and returns the best (r_fpm_lamD,
%   r_lyot_frac).  Confirm the winner at N=1024 via ctb_contrast.
%
%   Name-value:
%     'rx','elt','model_size' (512 for the sweep; bump to confirm).
%     'r_fpm_list'   occulter radii, lambda/D (default [1.5 2 2.5 3]).
%     'r_lyot_list'  Lyot fractions           (default [0.75 0.8 0.85 0.9]).
%     'band_inner','band_outer'  scoring band (default 3, 7).
%     'r_apod_m','r_apod_taper_m'  apodizer params.
%     'outdir','visible'.
%
%   See also: ctb_contrast, ctb_coro_compare, dark_zone_metrics.
    arguments
        opts.rx           (1,:) char   = ''
        opts.elt          struct = struct('DM1',2,'DM2',5,'Apodizer',13, ...
                                'FPM',17,'Lyot',20,'ExitPupil',30,'FPA',31)
        opts.model_size   (1,1) double = 512
        opts.r_fpm_list   (1,:) double = [1.5 2.0 2.5 3.0]
        opts.r_lyot_list  (1,:) double = [0.75 0.80 0.85 0.90]
        opts.band_inner   (1,1) double = 3.0
        opts.band_outer   (1,1) double = 7.0
        opts.r_apod_m     (1,1) double = 15e-3
        opts.r_apod_taper_m (1,1) double = 2e-3
        opts.outdir       (1,:) char   = ''
        opts.visible      (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    if isempty(opts.rx),     opts.rx     = fullfile(here,'ctb_dcr.in'); end
    if isempty(opts.outdir), opts.outdir = here; end
    addpath(fullfile(here,'..','..','..','src'));
    assert(~isempty(getenv('MACOS_HOME')),'MACOS_HOME must be set.');
    e = opts.elt;

    % fixed geometry + bare reference (compute ONCE)
    g = geom_scales_(opts, e);
    lamD = g.lamD_fpa_px;
    peak_bare = bare_peak_(opts, e);
    fprintf('[opt] N=%d lamD_fpa=%.3f px  band %.1f-%.1f lam/D  bare peak=%.3e\n', ...
        opts.model_size, lamD, opts.band_inner, opts.band_outer, peak_bare);

    nF = numel(opts.r_fpm_list); nL = numel(opts.r_lyot_list);
    C = nan(nF, nL);                                     % mean contrast grid
    T = nan(nF, nL);                                     % throughput (Lyot flux frac)
    for iF = 1:nF
        for iL = 1:nL
            [I, thr] = run_coro_(opts, g, opts.r_fpm_list(iF), opts.r_lyot_list(iL));
            dz = macos.dark_zone_metrics(I, peak_bare, lamD, opts.band_inner, opts.band_outer);
            C(iF,iL) = dz.mean;  T(iF,iL) = thr;
            fprintf('[opt]   r_fpm=%.2f lam/D  r_lyot=%.2f  ->  mean C=%.3e  T=%.3f  C/T=%.3e\n', ...
                opts.r_fpm_list(iF), opts.r_lyot_list(iL), dz.mean, thr, dz.mean/max(thr,eps));
        end
    end

    [best, idx] = min(C(:));
    [bF, bL] = ind2sub(size(C), idx);
    fprintf('[opt] BEST contrast: r_fpm=%.2f lam/D  r_lyot=%.2f  ->  C=%.3e  T=%.3f\n', ...
        opts.r_fpm_list(bF), opts.r_lyot_list(bL), best, T(bF,bL));
    % contrast-per-throughput knee (penalises throwing away light for contrast)
    CT = C ./ max(T,eps);
    [~, idk] = min(CT(:)); [kF,kL] = ind2sub(size(CT), idk);
    fprintf('[opt] BEST C/T knee: r_fpm=%.2f lam/D  r_lyot=%.2f  ->  C=%.3e  T=%.3f  C/T=%.3e\n', ...
        opts.r_fpm_list(kF), opts.r_lyot_list(kL), C(kF,kL), T(kF,kL), CT(kF,kL));

    % heatmap
    vis='off'; if opts.visible, vis='on'; end
    fig = figure('Visible',vis,'Color','w','Position',[80 80 700 560]);
    imagesc(opts.r_lyot_list, opts.r_fpm_list, log10(C));
    set(gca,'YDir','normal'); colormap(parula); cb=colorbar;
    cb.Label.String = sprintf('log_{10} mean contrast (%.0f-%.0f \\lambda/D)', ...
        opts.band_inner, opts.band_outer);
    xlabel('Lyot stop fraction of pupil'); ylabel('FPM occulter radius (\lambda/D)');
    title(sprintf('CTB mask/Lyot sweep -- best r_{fpm}=%.2f r_{lyot}=%.2f (C=%.2e)', ...
        opts.r_fpm_list(bF), opts.r_lyot_list(bL), best), 'Interpreter','tex');
    hold on; plot(opts.r_lyot_list(bL), opts.r_fpm_list(bF), 'rp', ...
        'MarkerSize',16,'MarkerFaceColor','r');            % best contrast
    plot(opts.r_lyot_list(kL), opts.r_fpm_list(kF), 'wd', ...
        'MarkerSize',13,'MarkerFaceColor','w','LineWidth',1.2);  % best C/T knee
    % annotate each cell: contrast (top) + throughput (bottom)
    for iF=1:nF, for iL=1:nL
        text(opts.r_lyot_list(iL), opts.r_fpm_list(iF), ...
            sprintf('%.1e\\newlineT=%.2f',C(iF,iL),T(iF,iL)), ...
            'HorizontalAlignment','center','FontSize',7.5, ...
            'Color', ternary_(C(iF,iL)>median(C(:)),'k','w'));
    end, end
    figpath = fullfile(opts.outdir,'ctb_optimize_masks.png');
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[opt] wrote %s\n', figpath);

    out = struct('r_fpm_list',opts.r_fpm_list,'r_lyot_list',opts.r_lyot_list, ...
        'contrast_grid',C,'best_r_fpm_lamD',opts.r_fpm_list(bF), ...
        'best_r_lyot_frac',opts.r_lyot_list(bL),'best_contrast',best, ...
        'band',[opts.band_inner opts.band_outer],'lamD_px',lamD,'figure',figpath);
end

% ======================================================================
function [I, thr] = run_coro_(opts, g, r_fpm_lamD, r_lyot_frac)
    e = opts.elt;
    macos.init(opts.model_size); macos.load_rx(opts.rx);
    macos.intensity(e.DM1);
    macos.intensity(e.DM2,'reset_trace',false);
    Ia = macos.intensity(e.Apodizer,'reset_trace',false);
    macos.apodize(e.Apodizer, ctb_mask_softcircle(size(Ia,1), abs(macos.dx_at(e.Apodizer)), ...
        opts.r_apod_m, opts.r_apod_taper_m, 8));
    macos.intensity(e.Apodizer,'reset_trace',false);
    If = macos.intensity(e.FPM,'reset_trace',false);
    macos.apodize(e.FPM, 1 - ctb_mask_disk(size(If,1), g.dx_f, r_fpm_lamD*g.lamD_fpm_m, 8));
    macos.intensity(e.FPM,'reset_trace',false);
    Il = macos.intensity(e.Lyot,'reset_trace',false);
    macos.apodize(e.Lyot, ctb_mask_disk(size(Il,1), abs(macos.dx_at(e.Lyot)), ...
        r_lyot_frac*g.r_lyot_geom_m, 8));
    macos.intensity(e.Lyot,'reset_trace',false);
    % first-order off-axis (planet) throughput = Lyot stop AREA fraction of
    % the pupil (the star-flux-through-Lyot is mostly rejected starlight, a
    % misleading proxy).  The FPM cost to an off-axis source outside a few
    % lambda/D is negligible, so r_lyot_frac^2 is the dominant term.
    thr = r_lyot_frac^2;
    I = macos.intensity(e.FPA,'reset_trace',false);
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

function v = ternary_(c,a,b), if c, v=a; else, v=b; end, end
