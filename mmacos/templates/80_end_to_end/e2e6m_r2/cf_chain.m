function ch = cf_chain(opts)
%CF_CHAIN  Config-driven masked-chain runner for the e2e6m round-2 train.
%   ch = CF_CHAIN('rx',DECK,...) loads an e2e6m diffraction deck once,
%   sizes the coronagraph masks once from ENGINE-measured scales (the
%   ctb_chain/ctb_aplc recipe: FPM pitch from the Fraunhofer relation on
%   the feeding sphere, Lyot radius from the bare geometric pupil, FPA
%   lambda/D from the exit-pupil FarField geometry), and returns a runner
%   that evaluates the masked chain repeatedly.  This is r1_coro's scoring
%   walk lifted into the ctb_chain pattern so every mask FAMILY runs
%   through ONE runner with its configuration recorded on the struct --
%   the substrate of the coronagraph-family campaign
%   (macos/BRIEF_e2e6m_coro_families.md S0).
%
%   Mask primitives are IMPORTED from bench_ctb (ctb_mask_disk /
%   ctb_mask_vortex / ctb_mask_bandlimited / ctb_apod_prolate) -- 8x
%   supersampled/binned throughout, the SESSION-11 rule.
%
%   Name-value (defaults reproduce the R1 APLC configuration):
%     'rx'           deck (default r1_seg_prop.in beside this file)
%     'elt'          station map (default: parsed from EltName= lines --
%                    DM1/DM2/Apodizer/FPM/Lyot/ExitPupil/Science)
%     'model_size'   grid (1024, the R1 scoring grid)
%     'coro'         insert masks in run() (true)
%     'apod_kind'    'prolate' (clear-disc Soummer prolate, the R1
%                    apodizer) | 'prolate_seg' (prolate over the TRACED
%                    gapped pupil support -- the aperture-matched APLC of
%                    N'Diaye/Zimmerman/Soummer 2016) | 'supplied'
%                    ('apod_mask') | 'none'
%     'apod_mask'    NxN amplitude apodizer when apod_kind='supplied'
%     'prolate_iter' power-iteration cap (5000; the e2e6m pupil converges
%                    near 2400 -- the ctb default 200 is NOT enough here)
%     'fpm_kind'     'hard' | 'vortex' | 'blc' | 'none'
%     'r_fpm_lamD'   hard-occulter radius, lambda/D (2.8) -- also the
%                    prolate design radius (APLC co-design: one number)
%     'charge'       vortex charge (4)
%     'blc_eps'      band-limited epsilon (0.40)
%     'blc_order'    4 | 8 (4); 'blc_form' 'separable'|'radial'|'linear'
%     'lyot'         apply the Lyot stop (true)
%     'r_lyot_frac'  Lyot radius / geometric pupil radius (0.90)
%
%   ch fields:
%     .run()               -> E at the Science plane through the masked
%                             chain (N x N complex; 'coro' false = bare)
%     .run_bare()          -> E with no masks (normalization reference)
%     .run_screened(S)     -> run() with an extra complex pupil screen
%                             multiplied at the Apodizer plane (Jones-
%                             component screens; [] = none)
%     .run_bare_screened(S)-> screen kept, coronagraph masks dropped
%     .dz_mask(in,out)     -> logical FPA annulus in lambda/D
%     .config              name-value stamp (ctb_jac_check vocabulary;
%                          every Jacobian/run cache must carry it)
%     .tag                 filesystem tag derived from the config
%     .thru                off-axis throughput proxy (apodizer Phi^2-fill
%                          x Lyot area; the ctb_mask_compare convention)
%     .elt,.N,.rx,.lambda_m,.lamD_px,.center_px,.peak_bare,.masks,
%     .r_apod_px,.support (traced pupil, prolate_seg), .prolate_info
%
%   Load-state contract: cf_chain OWNS the session (macos.init + load_rx
%   at construction).  Do not load another Rx between runs.  DM state is
%   whatever the element grids hold (apply commands with ctb_dm BEFORE
%   run) -- grid state survives the fresh trace each run makes.
%
%   See also: ctb_chain, ctb_aplc, ctb_jac_check, r1_coro, cf0_gates.
    arguments
        opts.rx            (1,:) char = ''
        opts.elt           struct = struct([])
        opts.model_size    (1,1) double {mustBeInteger,mustBePositive} = 1024
        opts.coro          (1,1) logical = true
        opts.apod_kind     (1,:) char {mustBeMember(opts.apod_kind, ...
                               {'none','prolate','prolate_seg','supplied'})} = 'prolate'
        opts.apod_mask     (:,:) double = []
        opts.prolate_iter  (1,1) double = 5000
        opts.fpm_kind      (1,:) char {mustBeMember(opts.fpm_kind, ...
                               {'hard','vortex','blc','none'})} = 'hard'
        opts.r_fpm_lamD    (1,1) double = 2.8
        opts.charge        (1,1) double = 4
        opts.blc_eps       (1,1) double = 0.40
        opts.blc_order     (1,1) double {mustBeMember(opts.blc_order,[4 8])} = 4
        opts.blc_form      (1,:) char {mustBeMember(opts.blc_form, ...
                               {'separable','radial','linear'})} = 'separable'
        opts.lyot          (1,1) logical = true
        opts.r_lyot_frac   (1,1) double = 0.90
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', '..', 'src'));
    addpath(fullfile(here, '..', '..', '30_instruments', 'bench_ctb'));
    assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
    if isempty(opts.rx)
        opts.rx = fullfile(here, 'r1_seg_prop.in');
    end
    assert(isfile(opts.rx), 'cf_chain: %s not found -- run r1_coro first', opts.rx);
    N = opts.model_size;

    % ---- load + station map --------------------------------------------
    macos.init(N);
    nE = macos.load_rx(opts.rx);
    [e_auto, dm_elt] = elt_map_(opts.rx);
    if isempty(fieldnames(opts.elt))
        e = e_auto;
    else
        e = opts.elt;
    end
    assert(nE == e.FPA, 'cf_chain: nElt=%d but Science index=%d', nE, e.FPA);
    cbm      = macos.cbm();
    lambda_m = macos.get_src_wvl() * cbm;

    % ---- deterministic scales (bare pre-passes; the ctb_aplc recipe) ---
    % FPM leg: Fraunhofer on the feeding sphere
    macos.intensity(e.FPM);
    Isph   = macos.intensity(e.FPM-1, 'reset_trace', false);
    dx_sph = abs(macos.dx_at(e.FPM-1));
    R_fpm  = abs(macos.get_elt_z(e.FPM-1)) * cbm;
    Dbeam  = 2 * beam_radius_(Isph, dx_sph);
    dx_f       = lambda_m * R_fpm / (N * dx_sph);
    lamD_fpm_m = lambda_m * R_fpm / Dbeam;

    % bare geometric Lyot radius
    Ily = macos.intensity(e.Lyot, 'reset_trace', false);
    r_lyot_geom_m = beam_radius_(Ily, abs(macos.dx_at(e.Lyot)));

    % FPA lambda/D in px (exit-pupil FarField geometry)
    macos.intensity(e.FPA);
    Iep  = macos.intensity(e.ExitPupil, 'reset_trace', false);
    Dep  = 2 * beam_radius_(Iep, abs(macos.dx_at(e.ExitPupil)));
    Rfpa = abs(macos.get_elt_z(e.ExitPupil)) * cbm;
    lamD_px = (lambda_m * Rfpa / Dep) / abs(macos.dx_at(e.FPA));

    % apodizer pupil (traced): radius + support
    macos.intensity(e.DM1);
    Iap = macos.intensity(e.Apodizer, 'reset_trace', false);
    r_apod_px = beam_radius_(Iap, 1);
    support = double(Iap > 0.02 * max(Iap(:)));   % the TRACED gapped pupil

    % ---- masks, built once (all 8x supersampled/binned) ----------------
    masks = struct();
    pinfo = struct('lambda0', NaN, 'converged', NaN, 'n_iter_used', 0);
    switch opts.apod_kind
        case 'prolate'
            [masks.A, pinfo] = ctb_apod_prolate(N, r_apod_px, opts.r_fpm_lamD, ...
                                                'n_iter', opts.prolate_iter);
        case 'prolate_seg'
            [masks.A, pinfo] = ctb_apod_prolate(N, r_apod_px, opts.r_fpm_lamD, ...
                                                'n_iter', opts.prolate_iter, ...
                                                'support', support);
        case 'supplied'
            assert(isequal(size(opts.apod_mask), [N N]), ...
                'cf_chain: apod_mask must be %dx%d', N, N);
            masks.A = opts.apod_mask;
        case 'none'
            masks.A = [];
    end
    switch opts.fpm_kind
        case 'hard'
            masks.F = 1 - ctb_mask_disk(N, dx_f, opts.r_fpm_lamD * lamD_fpm_m, 8);
        case 'vortex'
            masks.F = ctb_mask_vortex(N, opts.charge);       % complex, 8x-binned
        case 'blc'
            masks.F = ctb_mask_bandlimited(N, dx_f, lamD_fpm_m, opts.blc_eps, ...
                                           opts.blc_order, opts.blc_form);
        case 'none'
            masks.F = [];
    end
    if opts.lyot
        masks.L = ctb_mask_disk(N, abs(macos.dx_at(e.Lyot)), ...
                                opts.r_lyot_frac * r_lyot_geom_m, 8);
    else
        masks.L = [];
    end
    masks.dx_f = dx_f;  masks.lamD_fpm_m = lamD_fpm_m;
    masks.r_lyot_geom_m = r_lyot_geom_m;

    % ---- off-axis throughput proxy (ctb_mask_compare convention) -------
    if isempty(masks.A)
        thru_apod = 1;
    else
        thru_apod = phi2_fill_(masks.A, r_apod_px);
    end
    thru = thru_apod;
    if opts.lyot, thru = thru * opts.r_lyot_frac^2; end
    if strcmp(opts.fpm_kind, 'blc')
        % the BLC transmits (1-eps) in amplitude off-axis on top of its Lyot
        thru = thru * (1 - opts.blc_eps)^2;
    end

    % ---- config stamp + tag --------------------------------------------
    config = {'apod_kind', opts.apod_kind, 'prolate_iter', opts.prolate_iter, ...
              'fpm_kind', opts.fpm_kind, 'charge', opts.charge, ...
              'r_fpm_lamD', opts.r_fpm_lamD, ...
              'blc_eps', opts.blc_eps, 'blc_order', opts.blc_order, ...
              'blc_form', opts.blc_form, ...
              'lyot', opts.lyot, 'r_lyot_frac', opts.r_lyot_frac, ...
              'support_npx', nnz(support)};
    if strcmp(opts.apod_kind, 'supplied')
        config = [config, {'apod_fro', norm(masks.A(:))}];
    end
    ak = struct('none','n', 'prolate','p', 'prolate_seg','s', 'supplied','u');
    switch opts.fpm_kind
        case 'hard',   fk = sprintf('h%03d', round(100*opts.r_fpm_lamD));
        case 'vortex', fk = sprintf('v%d', opts.charge);
        case 'blc',    fk = sprintf('b%02d', round(100*opts.blc_eps));
        case 'none',   fk = 'x';
    end
    if opts.lyot, lk = sprintf('L%03d', round(100*opts.r_lyot_frac));
    else,         lk = 'L---'; end
    tag = sprintf('%s_%s_%s', ak.(opts.apod_kind), fk, lk);

    % ---- runner --------------------------------------------------------
    ch = struct();
    ch.elt = e;  ch.dm_elt = dm_elt;  ch.N = N;  ch.rx = opts.rx;
    ch.config = config;  ch.tag = tag;  ch.thru = thru;
    ch.thru_apod = thru_apod;
    ch.lambda_m = lambda_m;  ch.lamD_px = lamD_px;
    ch.center_px = floor(N/2) + 1;
    ch.masks = masks;  ch.coro = opts.coro;
    ch.r_apod_px = r_apod_px;  ch.support = support;
    ch.prolate_info = pinfo;
    ch.run               = @() run_(opts.coro, []);
    ch.run_bare          = @() run_bare_();
    ch.run_screened      = @(S) run_(opts.coro, S);
    ch.run_bare_screened = @(S) run_(false, S);
    ch.dz_mask = @dz_mask_;

    % bare on-axis peak (current DM state) -- the contrast normalizer
    Eb = ch.run_bare();
    ch.peak_bare = max(abs(Eb(:)).^2);

    function E = run_bare_()
        % ctb_aplc's bare_peak_ walk exactly: one DM1 stop, then the FPA
        % -- NO other intermediate stops (each read-and-continue perturbs
        % the field at the 1e-10 level; measured, cf0).
        macos.intensity(e.DM1);                       % fresh trace
        E = macos.complex_field(e.FPA, 'reset_trace', false);
    end

    function E = run_(withMasks, S)
        % r1_coro's scoring walk (ctb_aplc run_aplc_), masks per config.
        macos.intensity(e.DM1);                       % fresh trace
        macos.intensity(e.DM2, 'reset_trace', false);
        macos.intensity(e.Apodizer, 'reset_trace', false);
        if withMasks && ~isempty(masks.A)
            macos.apodize(e.Apodizer, masks.A);
            macos.intensity(e.Apodizer, 'reset_trace', false);
        end
        if ~isempty(S)
            macos.apodize_complex(e.Apodizer, S);
        end
        macos.intensity(e.FPM, 'reset_trace', false);
        if withMasks && ~isempty(masks.F)
            if isreal(masks.F)
                macos.apodize(e.FPM, masks.F);
            else
                macos.apodize_complex(e.FPM, masks.F);
            end
            macos.intensity(e.FPM, 'reset_trace', false);
        end
        macos.intensity(e.Lyot, 'reset_trace', false);
        if withMasks && ~isempty(masks.L)
            macos.apodize(e.Lyot, masks.L);
            macos.intensity(e.Lyot, 'reset_trace', false);
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

% =========================================================================
function [e, dm_elt] = elt_map_(rx)
%ELT_MAP_  Chain stations from EltName= lines.  The DM1/DM2 STOPS are the
%   near-field SEED PLANES (Prop*_start/end, the planes prop_layout put ON
%   the DM1->DM2 leg) -- r1_coro's walk, which the S0 bit-consistency gate
%   pins; the elements NAMED DM1/DM2 (the grid surfaces the control layer
%   pokes) are returned separately as dm_elt.  Stopping at the named DMs
%   instead partitions the near-field legs differently and shifts the
%   dark-zone numbers at the 1e-8 level (measured, cf0).
    nm = regexp(fileread(rx), '^\s*EltName=\s*(\S+)', 'tokens', 'lineanchors');
    nm = cellfun(@(c) c{1}, nm, 'UniformOutput', false);
    at = @(s) find(strcmp(nm, s), 1);
    ps = find(~cellfun('isempty', regexp(nm, '^Prop\d+_start$', 'once')), 1);
    pe = find(~cellfun('isempty', regexp(nm, '^Prop\d+_end$',   'once')), 1);
    if isempty(ps), ps = at('DM1'); end          % decks without seed planes
    if isempty(pe), pe = at('DM2'); end
    e = struct('DM1',ps, 'DM2',pe, ...
               'Apodizer',at('Apodizer'), 'FPM',at('FPM'), ...
               'Lyot',at('Lyot'), 'ExitPupil',at('ExitPupil'), ...
               'FPA',at('Science'));
    f = fieldnames(e);
    for k = 1:numel(f)
        assert(~isempty(e.(f{k})), 'cf_chain: %s not in %s', f{k}, rx);
    end
    dm_elt = [at('DM1'), at('DM2')];
end

function rr = beam_radius_(I, dx)
%BEAM_RADIUS_  Max radial extent of the illuminated support (2% threshold),
%   about the FFT DC pixel -- the ctb_aplc/ctb_chain definition, verbatim.
    thr = 0.02 * max(I(:));
    [yy, xx] = find(I > thr);
    if isempty(xx), rr = 0; return; end
    c = floor(size(I,1)/2) + 1;
    rr = max(hypot(xx - c, yy - c)) * dx;
end

function t = phi2_fill_(Phi, r_pup_px)
%PHI2_FILL_  Phi^2-weighted fill over the geometric pupil disc (off-axis
%   amplitude throughput of the apodizer alone; gap area counts as loss).
    N = size(Phi,1);  c = floor(N/2) + 1;
    [X, Y] = meshgrid((1:N) - c, (1:N) - c);
    P = hypot(X, Y) <= r_pup_px;
    t = sum(Phi(P).^2) / max(sum(P(:)), 1);
end
