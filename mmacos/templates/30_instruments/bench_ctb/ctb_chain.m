function ch = ctb_chain(opts)
%CTB_CHAIN  Reusable masked-chain runner for the CTB (EFC/Jacobian core).
%   ch = CTB_CHAIN() loads a CTB deck once, sizes the coronagraph masks
%   once (the deterministic geometry of ctb_coro_compare: FPM pitch from
%   the Fraunhofer relation on the NF1 sphere, Lyot radius from the bare
%   geometric pupil), and returns a runner that evaluates the chain
%   repeatedly -- the loop primitive the DM Jacobian and EFC drivers
%   need.  Each run is a fresh forward pass: full trace, masks
%   multiplied in place at their planes (macos.apodize, the stage-1
%   convention), complex field read at the FPA.  DM state is whatever
%   the element grids currently hold (apply commands with ctb_dm
%   BEFORE calling run).
%
%   Name-value (defaults match ctb_contrast):
%     'rx'            deck (default ctb_dm.in, emitted if absent)
%     'elt'           station map (default compact-deck indices)
%     'model_size'    grid (512)
%     'coro'          insert masks in run() (true)
%     'apodizer','fpm','lyot'  individual mask switches (true)
%     'r_fpm_lamD'    FPM occulter radius, lambda/D (2.70)
%     'r_apod_m','r_apod_taper_m','r_lyot_frac'  mask params
%
%   ch fields:
%     .run()        -> E at the FPA (N x N complex) through the masked
%                      chain (or bare when 'coro' false)
%     .run_bare()   -> E at the FPA, no masks (reference/normalization)
%     .elt, .N, .lamD_px, .center_px, .masks (A/F/L arrays + scales),
%     .peak_bare (bare on-axis peak intensity, the contrast normalizer),
%     .dz_mask(inner_lamD, outer_lamD) -> logical FPA annulus
%
%   Load-state contract: ctb_chain OWNS the session (macos.init +
%   load_rx at construction).  Do not load another Rx between runs.
%
%   Run:  >> ch = ctb_chain;  E = ch.run();
%   See also: ctb_dm, ctb_dm_jacobian, ctb_efc, ctb_coro_compare.
    arguments
        opts.rx              (1,:) char = ''
        opts.elt             struct = struct('DM1',2,'DM2',5,'Apodizer',13, ...
                                  'FPM',17,'Lyot',20,'ExitPupil',30,'FPA',31)
        opts.model_size      (1,1) double {mustBeInteger,mustBePositive} = 512
        opts.coro            (1,1) logical = true
        opts.apodizer        (1,1) logical = true
        opts.fpm             (1,1) logical = true
        opts.lyot            (1,1) logical = true
        opts.fpm_kind        (1,:) char {mustBeMember(opts.fpm_kind, ...
                                  {'hard','vortex'})} = 'hard'
        opts.charge          (1,1) double = 4
        opts.r_fpm_lamD      (1,1) double = 2.70
        opts.r_apod_m        (1,1) double = 15e-3
        opts.r_apod_taper_m  (1,1) double = 2e-3
        opts.r_lyot_frac     (1,1) double = 0.50
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', '..', 'src'));
    assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
    if isempty(opts.rx)
        opts.rx = fullfile(here, 'ctb_dm.in');
        if ~isfile(opts.rx), ctb_dm_rx(); end
    end
    e = opts.elt;  N = opts.model_size;

    % ---- load + deterministic scales (bare, maskless pre-pass) --------
    macos.init(N);
    nE = macos.load_rx(opts.rx);
    assert(nE == e.FPA, 'ctb_chain: nElt=%d but FPA index=%d', nE, e.FPA);
    cbm      = macos.cbm();
    lambda_m = macos.get_src_wvl() * cbm;

    % FPM leg scales (Fraunhofer on the NF1 sphere; finding-2 pattern)
    macos.intensity(e.FPM);
    Isph   = macos.intensity(e.FPM-1, 'reset_trace', false);
    dx_sph = abs(macos.dx_at(e.FPM-1));
    R_m    = abs(macos.get_elt_z(e.FPM-1)) * cbm;
    Dbeam  = 2 * footprint_radius_(Isph, dx_sph);
    dx_f       = lambda_m * R_m / (N * dx_sph);
    lamD_fpm_m = lambda_m * R_m / Dbeam;

    % bare geometric Lyot radius (finding-4 pattern)
    Ily = macos.intensity(e.Lyot, 'reset_trace', false);
    r_lyot_geom_m = footprint_radius_(Ily, abs(macos.dx_at(e.Lyot)));

    % FPA lambda/D in px (exit-pupil FarField geometry; shared_lamD_ pattern)
    macos.intensity(e.FPA);
    Iep  = macos.intensity(e.ExitPupil, 'reset_trace', false);
    Dep  = 2 * footprint_radius_(Iep, abs(macos.dx_at(e.ExitPupil)));
    Rfpa = abs(macos.get_elt_z(e.ExitPupil)) * cbm;
    lamD_px = (lambda_m * Rfpa / Dep) / abs(macos.dx_at(e.FPA));

    % ---- masks, built once --------------------------------------------
    masks = struct();
    if opts.apodizer
        masks.A = ctb_mask_softcircle(N, abs(macos.dx_at(e.Apodizer)), ...
                                      opts.r_apod_m, opts.r_apod_taper_m, 8);
    end
    if opts.fpm
        switch opts.fpm_kind
            case 'hard'
                masks.F = 1 - ctb_mask_disk(N, dx_f, ...
                                            opts.r_fpm_lamD * lamD_fpm_m, 8);
            case 'vortex'
                masks.F = ctb_mask_vortex(N, opts.charge);   % complex, 8x-binned
        end
    end
    if opts.lyot
        masks.L = ctb_mask_disk(N, abs(macos.dx_at(e.Lyot)), ...
                                opts.r_lyot_frac * r_lyot_geom_m, 8);
    end
    masks.dx_f = dx_f;  masks.lamD_fpm_m = lamD_fpm_m;
    masks.r_lyot_geom_m = r_lyot_geom_m;

    % ---- runner --------------------------------------------------------
    ch = struct();
    ch.elt = e;  ch.N = N;  ch.rx = opts.rx;
    ch.config = {'fpm_kind', opts.fpm_kind, 'charge', opts.charge, ...
        'apodizer', opts.apodizer, 'fpm', opts.fpm, 'lyot', opts.lyot, ...
        'r_fpm_lamD', opts.r_fpm_lamD, 'r_apod_m', opts.r_apod_m, ...
        'r_apod_taper_m', opts.r_apod_taper_m, 'r_lyot_frac', opts.r_lyot_frac};
    ch.lambda_m = lambda_m;  ch.lamD_px = lamD_px;
    ch.center_px = floor(N/2) + 1;
    ch.masks = masks;  ch.coro = opts.coro;
    ch.run      = @() run_(opts.coro);
    ch.run_bare = @() run_(false);
    ch.run_screened      = @(S) run_screened_(opts.coro, S);
    ch.run_bare_screened = @(S) run_screened_(false, S);
    ch.dz_mask  = @dz_mask_;

    % bare on-axis peak (current DM state = as-loaded) for normalization
    Eb = ch.run_bare();
    ch.peak_bare = max(abs(Eb(:)).^2);

    function E = run_(withMasks)
        if ~withMasks
            E = macos.complex_field(e.FPA);        % full fresh pass
            return;
        end
        macos.intensity(e.Apodizer);               % fresh trace to apodizer
        if opts.apodizer
            macos.apodize(e.Apodizer, masks.A);
        end
        macos.intensity(e.FPM, 'reset_trace', false);
        if opts.fpm
            if isreal(masks.F)
                macos.apodize(e.FPM, masks.F);
            else
                macos.apodize_complex(e.FPM, masks.F);
            end
        end
        macos.intensity(e.Lyot, 'reset_trace', false);
        if opts.lyot
            macos.apodize(e.Lyot, masks.L);
        end
        E = macos.complex_field(e.FPA, 'reset_trace', false);
    end

    function E = run_screened_(withMasks, S)
        % run_ with an extra complex pupil screen multiplied at the
        % Apodizer plane (a Jones-component screen; [] = none).  The
        % bare variant keeps the screen but drops the coronagraph masks
        % -- the per-component normalization runs.
        macos.intensity(e.Apodizer);
        if withMasks && opts.apodizer
            macos.apodize(e.Apodizer, masks.A);
        end
        if ~isempty(S)
            macos.apodize_complex(e.Apodizer, S);
        end
        macos.intensity(e.FPM, 'reset_trace', false);
        if withMasks && opts.fpm
            if isreal(masks.F)
                macos.apodize(e.FPM, masks.F);
            else
                macos.apodize_complex(e.FPM, masks.F);
            end
        end
        macos.intensity(e.Lyot, 'reset_trace', false);
        if withMasks && opts.lyot
            macos.apodize(e.Lyot, masks.L);
        end
        E = macos.complex_field(e.FPA, 'reset_trace', false);
    end

    function M = dz_mask_(inner_lamD, outer_lamD)
        c = ch.center_px;
        [ii, jj] = ndgrid(1:N, 1:N);
        rl = hypot(ii - c, jj - c) / lamD_px;
        M = rl >= inner_lamD & rl <= outer_lamD;
    end
end

function rr = footprint_radius_(I, dx)
%FOOTPRINT_RADIUS_  Radius (m) of the illuminated support (2% threshold).
    thr = 0.02 * max(I(:));
    [yy, xx] = find(I > thr);
    if isempty(xx), rr = 0; return; end
    c = (size(I,1)-1)/2 + 1;
    rr = max(hypot(xx - c, yy - c)) * dx;
end
