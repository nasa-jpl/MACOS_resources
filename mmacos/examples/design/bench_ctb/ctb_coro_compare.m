function out = ctb_coro_compare(opts)
%CTB_CORO_COMPARE  Side-by-side compact-vs-full coronagraph propagation.
%   out = CTB_CORO_COMPARE() propagates the CTB through TWO diffraction
%   models and shows the intensity at the key coronagraph planes side by
%   side, so a user can compare a "compact" model (near-field props only
%   through the mask conjugates) against a "full" surface-to-surface model
%   (every leg propagated) -- with or without the coronagraph masks.
%
%   The coronagraph elements (apodizer, focal-plane mask, Lyot stop) are
%   applied IN MATLAB: the complex field is propagated to the mask plane,
%   multiplied by the (real, 0..1) mask array, and propagation continues.
%   This is the stage-1 convention (see Coro_propagation_summary.md): the
%   deck's mask sites stay passive; masking is a cfield multiply so the
%   diffraction wavefront is actually modified (an obscuration declared on
%   a Reference clips rays only -- the Phase-5 lesson).
%
%   Displayed planes (columns = models, rows = planes):
%     DM1, DM2 (pupils), Apodizer (pupil), FPM (focus), Lyot (pupil),
%     final ExitPupil (pupil), FPA (focus / the PSF).
%
%   ------------------------------------------------------------------
%   PARAMETERS (edit these or pass as name-value pairs)
%   ------------------------------------------------------------------
%     'models'   struct array of models to compare, each with fields
%                .name, .rx (path), .elt (station-index struct).  Default:
%                the shipped compact (ctb_dcr.in) + full (ctb_s2s_dcr.in).
%     'model_size' macos grid size (default 512).
%     'coro'     logical, insert the coronagraph masks (default true).
%     'apodizer' logical, apply the apodizer at the Apodizer pupil (true).
%     'fpm'      logical, apply the focal-plane mask at the FPM (true).
%     'lyot'     logical, apply the Lyot stop at the Lyot pupil (true).
%     'r_apod_m' apodizer soft-edge radius, metres (default 15e-3).
%     'r_apod_taper_m' apodizer Gaussian edge sigma, metres (2e-3).
%     'r_fpm_lamD' FPM occulting-spot radius in lambda/D (default 3).
%     'r_lyot_frac' Lyot radius as a fraction of the pupil beam (0.85).
%     'outdir'   where to write the figure (default: this example dir).
%     'visible'  show the figure (default false; headless-safe).
%
%   To compare YOUR OWN models, pass 'models' with your .in files and the
%   element indices of the seven planes:
%     m(1) = struct('name','mine', 'rx','/path/my.in', ...
%                   'elt', struct('DM1',2,'DM2',5,'Apodizer',13,'FPM',17, ...
%                                 'Lyot',20,'ExitPupil',30,'FPA',31));
%     ctb_coro_compare('models', m);
%
%   To add a NEW mask kind, add a builder to build_masks_ below and a call
%   in run_model_ at the plane you want it applied.
%
%   Run:  >> out = ctb_coro_compare;            (requires MACOS_HOME)
%   See also: macos.apodize, ctb_prop_layout, Coro_propagation_summary.md.
    arguments
        opts.models          struct = default_models_()
        opts.model_size      (1,1) double {mustBeInteger,mustBePositive} = 512
        opts.coro            (1,1) logical = true
        opts.apodizer        (1,1) logical = true
        opts.fpm             (1,1) logical = true
        opts.lyot            (1,1) logical = true
        opts.r_apod_m        (1,1) double = 15e-3
        opts.r_apod_taper_m  (1,1) double = 2e-3
        opts.r_fpm_lamD      (1,1) double = 3.0
        opts.r_lyot_frac     (1,1) double = 0.85
        opts.outdir          (1,:) char = default_outdir_()
        opts.visible         (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', '..', 'src'));           % mmacos/src
    assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
    if ~exist(opts.outdir,'dir'); mkdir(opts.outdir); end

    % masks OFF entirely if coro=false (a bare-optics comparison)
    if ~opts.coro
        opts.apodizer = false; opts.fpm = false; opts.lyot = false;
    end

    planes = {'DM1','DM2','Apodizer','FPM','Lyot','ExitPupil','FPA'};
    kind   = {'pupil','pupil','pupil','focal','pupil','pupil','focal'};

    % --- SHARED lambda/D --------------------------------------------------
    % All models share the same optics, so lambda/D at the FPA is ONE
    % number.  Compute it once (analytic first-order value preferred) and
    % feed it to every model, so the FPM (sized in lambda/D) is IDENTICAL
    % across models -- otherwise the comparison is not apples-to-apples.
    lamD = shared_lamD_(opts.models, opts.model_size);
    fprintf('[compare] shared lambda/D = %.3f px (all models)\n', lamD);

    % --- run every model ------------------------------------------------
    R = struct('name',{},'I',{},'dx',{},'lamD',{});
    for k = 1:numel(opts.models)
        m = opts.models(k);
        r = run_model_(m, planes, opts, lamD);
        r.name = m.name;
        R(end+1) = r;                                            %#ok<AGROW>
        fprintf('[compare] %-6s  FPA peak=%.4e sum=%.4e dx=%.3e m  lam/D=%.2f px\n', ...
            r.name, max(r.I{end}(:)), sum(r.I{end}(:)), r.dx(end), r.lamD);
    end

    % --- figure: rows = planes, cols = models --------------------------
    nM = numel(R);  nP = numel(planes);
    vis = 'off';  if opts.visible, vis = 'on'; end
    fig = figure('Visible',vis, 'Color','w', 'Position',[60 60 460*nM 220*nP]);
    tl = tiledlayout(fig, nP, nM, 'TileSpacing','compact','Padding','compact');
    ttl = sprintf('CTB coronagraph propagation -- INT at key planes (%s)', ...
        ternary_(opts.coro, sprintf('coro: apod=%d fpm=%d lyot=%d', ...
            opts.apodizer, opts.fpm, opts.lyot), 'NO coronagraph (bare optics)'));
    title(tl, ttl, 'FontWeight','bold', 'Interpreter','none');

    for ip = 1:nP
        for im = 1:nM
            nexttile(tl);
            I = R(im).I{ip};
            show_plane_(I, kind{ip}, R(im).lamD);
            if im == 1, ylabel(planes{ip}, 'FontWeight','bold'); end
            if ip == 1, title(R(im).name, 'Interpreter','none'); end
        end
    end
    tag = ternary_(opts.coro, 'coro', 'bare');
    figpath = fullfile(opts.outdir, sprintf('ctb_coro_compare_%s.png', tag));
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[compare] wrote %s\n', figpath);

    out = struct('models',{{R.name}}, 'planes',{planes}, 'kind',{kind}, ...
                 'I',{{R.I}}, 'dx',{{R.dx}}, 'lamD',[R.lamD], ...
                 'figure',figpath, 'opts',opts);
end

% ======================================================================
%  Propagate ONE model, applying the MATLAB masks in light order.
% ======================================================================
function r = run_model_(m, planes, opts, lamD)
    e = m.elt;
    macos.init(opts.model_size);
    nE = macos.load_rx(m.rx);
    assert(nE == e.FPA, '%s: nElt=%d but FPA index=%d', m.name, nE, e.FPA);

    % ---- single forward pass, masks multiplied in place ---------------
    I = cell(1, numel(planes));
    macos.intensity(e.DM1);                              % first: full trace
    I{1} = macos.intensity(e.DM1, 'reset_trace', false);
    I{2} = macos.intensity(e.DM2, 'reset_trace', false);

    % Apodizer pupil
    I{3} = macos.intensity(e.Apodizer, 'reset_trace', false);
    if opts.apodizer
        M = mask_softcircle_(size(I{3},1), abs(macos.dx_at(e.Apodizer)), ...
                             opts.r_apod_m, opts.r_apod_taper_m);
        macos.apodize(e.Apodizer, M);
        I{3} = macos.intensity(e.Apodizer, 'reset_trace', false);
    end

    % FPM focus
    I{4} = macos.intensity(e.FPM, 'reset_trace', false);
    if opts.fpm
        dxf = abs(macos.dx_at(e.FPM));
        r_fpm_m = opts.r_fpm_lamD * lamD * dxf;          % lam/D -> pixels -> m
        M = 1 - mask_harddisk_(size(I{4},1), dxf, r_fpm_m);  % opaque occulter
        macos.apodize(e.FPM, M);
        I{4} = macos.intensity(e.FPM, 'reset_trace', false);
    end

    % Lyot pupil
    I{5} = macos.intensity(e.Lyot, 'reset_trace', false);
    if opts.lyot
        dxl = abs(macos.dx_at(e.Lyot));
        r_beam = beam_radius_(I{5}, dxl);                % measured pupil radius
        M = mask_harddisk_(size(I{5},1), dxl, opts.r_lyot_frac * r_beam);
        macos.apodize(e.Lyot, M);
        I{5} = macos.intensity(e.Lyot, 'reset_trace', false);
    end

    % final exit pupil + FPA
    I{6} = macos.intensity(e.ExitPupil, 'reset_trace', false);
    I{7} = macos.intensity(e.FPA,       'reset_trace', false);

    dx = zeros(1, numel(planes));
    fld = planes;
    for i = 1:numel(planes), dx(i) = abs(macos.dx_at(e.(fld{i}))); end
    r = struct('name','', 'I',{I}, 'dx',dx, 'lamD',lamD);
end

% ======================================================================
%  Mask builders (real 0..1 arrays, centred on the array centre, which is
%  the chief-ray pierce for these axis-aligned decks).  Supersampled edges.
% ======================================================================
function M = mask_harddisk_(N, dx, r_m, K)
%MASK_HARDDISK_  1 inside radius r_m, 0 outside; K-supersampled edge.
    if nargin < 4, K = 8; end
    M = disk_ss_(N, dx, r_m, K);
end

function M = mask_softcircle_(N, dx, r0_m, sigma_m, K)
%MASK_SOFTCIRCLE_  1 inside r0, Gaussian roll-off outside; truncated at
%   r0+4*sigma.  Amplitude apodizer.
    if nargin < 5, K = 8; end
    r1 = r0_m + 4*sigma_m;
    base = disk_ss_(N, dx, r1, K);                       % hard truncation
    c = (N-1)/2;  [xx,yy] = meshgrid(0:N-1, 0:N-1);
    rr = hypot(xx-c, yy-c) * dx;
    tap = ones(N);  out = rr > r0_m;
    tap(out) = exp(-((rr(out)-r0_m)/sigma_m).^2);
    M = base .* tap;
end

function M = disk_ss_(N, dx, r_m, K)
%DISK_SS_  K x K supersampled binary disk of radius r_m (metres).
    c = (N-1)/2;
    off = ((0:K-1) - (K-1)/2) / K;                       % sub-pixel offsets
    M = zeros(N);
    [ox, oy] = meshgrid(off, off);
    ox = ox(:).'; oy = oy(:).';                          % 1 x K^2
    for i = 1:N
        yc = (i-1-c);
        xs = ((0:N-1)-c).';                              % N x 1
        % accumulate sub-sample hits
        acc = zeros(N,1);
        for s = 1:numel(ox)
            xx = (xs + ox(s)) * dx;
            yy = (yc + oy(s)) * dx;
            acc = acc + double(xx.^2 + yy.^2 <= r_m^2);
        end
        M(i,:) = acc.' / numel(ox);
    end
end

% ======================================================================
%  Helpers
% ======================================================================
function lamD = shared_lamD_(models, model_size)
%SHARED_LAMD_  ONE lambda/D (px) at the FPA for all models (they share
%   optics).  Try each model's analytic first-order value; if none is
%   valid, fall back to the Airy-null finder on a bare FPA PSF; then a
%   constant.  Computed with NO masks (bare optics).
    here = fileparts(mfilename('fullpath'));
    coro = fullfile(here, '..', '..', 'coronagraph', 'coro');
    if exist(fullfile(coro,'lambda_over_D_pixels.m'),'file'), addpath(coro); end
    lamD = [];
    for k = 1:numel(models)
        e = models(k).elt;
        macos.init(model_size);
        try, macos.load_rx(models(k).rx); catch, continue; end
        macos.intensity(e.FPA);
        % analytic first-order lambda/D (preferred)
        try
            p = macos.first_order_properties(e.FPA);
            if isfield(p,'lamD_px') && isfinite(p.lamD_px) && p.lamD_px > 0
                lamD = p.lamD_px;  return;
            end
        catch
        end
        % Airy-null finder as a per-model fallback
        if isempty(lamD) && exist('lambda_over_D_pixels','file')
            try, lamD = lambda_over_D_pixels(macos.intensity(e.FPA)); catch, end
        end
    end
    if isempty(lamD) || ~isfinite(lamD) || lamD <= 0, lamD = 4.0; end
end

function rr = beam_radius_(I, dx)
%BEAM_RADIUS_  Physical radius (m) enclosing the illuminated pupil, from
%   the intensity footprint (99th-pct of the support).
    thr = 0.02 * max(I(:));
    [yy,xx] = find(I > thr);
    if isempty(xx), rr = 0; return; end
    c = (size(I,1)-1)/2 + 1;
    rr = max(hypot(xx-c, yy-c)) * dx;
end

function show_plane_(I, kind, lamD)
    I = double(I);
    if strcmp(kind,'pupil')
        A = sqrt(max(I,0));                              % amplitude
        A = A / max(A(:)+eps);
        imagesc(crop_(A, 300)); axis image off; colormap(gca,gray); clim([0 1]);
    else
        In = I / max(I(:)+eps);
        L = log10(max(In, 1e-10));
        w = max(40, round(2*15*max(lamD,2)));            % +/-15 lam/D
        imagesc(crop_(L, w)); axis image off; colormap(gca,parula); clim([-10 0]);
        cb = colorbar; cb.Label.String = 'log_{10} norm I';
    end
end

function o = crop_(img, w)
    n = size(img,1);
    if w >= n, o = img; return; end
    c = floor(n/2)+1;  lo = max(c-floor(w/2),1);  hi = min(lo+w-1,n);
    o = img(lo:hi, lo:hi);
end

function v = ternary_(c,a,b), if c, v=a; else, v=b; end, end

function d = default_outdir_(), d = fileparts(mfilename('fullpath')); end

function m = default_models_()
%DEFAULT_MODELS_  The shipped compact + full CTB decks with station maps.
    here = fileparts(mfilename('fullpath'));
    m(1) = struct('name','compact', ...
        'rx', fullfile(here,'ctb_dcr.in'), ...
        'elt', struct('DM1',2,'DM2',5,'Apodizer',13,'FPM',17, ...
                      'Lyot',20,'ExitPupil',30,'FPA',31));
    m(2) = struct('name','full', ...
        'rx', fullfile(here,'ctb_s2s_dcr.in'), ...
        'elt', struct('DM1',2,'DM2',5,'Apodizer',16,'FPM',22, ...
                      'Lyot',27,'ExitPupil',43,'FPA',44));
end
