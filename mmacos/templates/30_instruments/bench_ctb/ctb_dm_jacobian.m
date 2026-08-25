function out = ctb_dm_jacobian(opts)
%CTB_DM_JACOBIAN  Engine-measured EFC Jacobian: dE(dark zone)/d(actuator).
%   out = CTB_DM_JACOBIAN() pokes every active actuator of the CTB DMs
%   through the FULL masked diffraction chain (ctb_chain: apodizer +
%   FPM + Lyot applied in place) and records the finite-difference
%   change of the complex FPA field on the dark-zone pixels.  This is
%   the engine-faithful G matrix EFC needs -- measured against the
%   engine, not a Fourier model of it (the e2e6m S3b lesson: an
%   optimizer mines any model gap; the design operator must BE the
%   engine).
%
%   G(:,c) = (E_poke(dz) - E0(dz)) / h,  c = (DM, actuator) column.
%
%   Name-value:
%     'model_size'    grid (512)
%     'nact'          actuators across each DM lattice (32)
%     'beam_d_mm'     controlled beam diameter on the DMs (21.3, the
%                     measured CTB footprint at DM1/DM2)
%     'coupling'      influence-function nearest-neighbor coupling (0.12)
%     'h_mm'          poke, mm of surface (2e-6 = 2 nm; OPD nonlinearity
%                     is quadratic in the poke phase -- ~0.1% here)
%     'dms'           which DMs to include (default [1 2])
%     'inner_lamD','outer_lamD'  dark-zone annulus (3, 15)
%     'chain'         name-value cell forwarded to ctb_chain (mask config)
%     'save'          write ctb_dm_jacobian_N<model>.mat + .fp.json (true)
%     'outdir'        (this dir)
%     'verbose'       progress prints every 50 pokes (true)
%
%   out: G (npix x ncol complex single), col_dm/col_act (column maps),
%   dz_idx (linear FPA indices), E0_dz, peak_bare, lamD_px, center_px,
%   dm (model geometry per DM), h_mm, config, timing.
%
%   The .mat is the EFC input; it is fingerprinted (jac_fingerprint) and
%   the full array is gitignored above ~10 MB by the bench_ctb policy --
%   regen with this driver.
%
%   Run:  >> out = ctb_dm_jacobian;                 (~25 min at 512)
%   See also: ctb_dm, ctb_chain, ctb_efc, jac_fingerprint.
    arguments
        opts.model_size (1,1) double {mustBeInteger,mustBePositive} = 512
        opts.nact       (1,1) double {mustBeInteger,mustBePositive} = 32
        opts.beam_d_mm  (1,1) double {mustBePositive} = 21.3
        opts.coupling   (1,1) double = 0.12
        opts.h_mm       (1,1) double {mustBePositive} = 2e-6
        opts.dms        (1,:) double = [1 2]
        opts.inner_lamD (1,1) double = 3.0
        opts.outer_lamD (1,1) double = 15.0
        opts.chain      (1,:) cell = {}
        opts.tag        (1,:) char = ''
        opts.a0         (1,:) cell = {}
        opts.save       (1,1) logical = true
        opts.outdir     (1,:) char = ''
        opts.verbose    (1,1) logical = true
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', '..', 'src'));
    if isempty(opts.outdir), opts.outdir = here; end

    % ---- deck + chain + DM models --------------------------------------
    r  = ctb_dm_rx();
    ch = ctb_chain('rx', r.rx_out, 'model_size', opts.model_size, ...
                   opts.chain{:});
    ndm = numel(opts.dms);
    dm = cell(1, ndm);
    if isempty(opts.a0)
        opts.a0 = arrayfun(@(~) zeros(opts.nact^2, 1), 1:ndm, ...
                           'UniformOutput', false);
    end
    for k = 1:ndm
        j = opts.dms(k);
        dm{k} = ctb_dm('ielt', r.ielt(j), 'ng', r.ng, 'gdx_mm', r.gdx_mm(j), ...
                       'nact', opts.nact, 'beam_d_mm', opts.beam_d_mm, ...
                       'coupling', opts.coupling);
        dm{k}.apply(opts.a0{k});          % linearization point (flat default)
    end

    % ---- dark zone + nominal field -------------------------------------
    M  = ch.dz_mask(opts.inner_lamD, opts.outer_lamD);
    dz_idx = find(M);
    E0 = ch.run();
    e0 = E0(dz_idx);
    c0 = mean(abs(e0).^2) / ch.peak_bare;
    fprintf('[jac] dark zone %g-%g lam/D: %d px, initial mean contrast %.3e\n', ...
        opts.inner_lamD, opts.outer_lamD, numel(dz_idx), c0);

    % ---- poke loop ------------------------------------------------------
    ncol = sum(cellfun(@(d) d.nact_active, dm));
    G = complex(zeros(numel(dz_idx), ncol, 'single'));
    col_dm  = zeros(1, ncol);  col_act = zeros(1, ncol);
    h = opts.h_mm;
    t0 = tic;  c = 0;
    for k = 1:ndm
        acts = find(dm{k}.active).';
        for j = acts
            a = opts.a0{k};  a(j) = a(j) + h;
            dm{k}.apply(a);
            E = ch.run();
            c = c + 1;
            G(:, c) = single((E(dz_idx) - e0) / h);
            col_dm(c) = opts.dms(k);  col_act(c) = j;
            if opts.verbose && mod(c, 50) == 0
                el = toc(t0);
                fprintf('[jac] %d/%d pokes  %.1f s  (%.2f s/poke, ETA %.1f min)\n', ...
                    c, ncol, el, el/c, el/c*(ncol-c)/60);
            end
        end
        dm{k}.apply(opts.a0{k});
    end
    dt = toc(t0);
    fprintf('[jac] done: %d pokes in %.1f min (%.2f s/poke)\n', ncol, dt/60, dt/ncol);

    % ---- package + save -------------------------------------------------
    for k = ndm:-1:1
        d = dm{k};
        dmgeo(k) = struct('ielt',d.ielt, 'nact',d.nact, ...
            'pitch_mm',d.pitch_mm, 'coupling',d.coupling, ...
            'beam_d_mm',d.beam_d_mm, 'active',d.active, ...
            'acx',d.acx, 'acy',d.acy, 'ng',d.ng, 'gdx_mm',d.gdx_mm);
    end
    out = struct('G',G, 'col_dm',col_dm, 'col_act',col_act, ...
        'dz_idx',dz_idx, 'E0_dz',e0, 'peak_bare',ch.peak_bare, ...
        'lamD_px',ch.lamD_px, 'center_px',ch.center_px, 'N',ch.N, ...
        'dm',dmgeo, 'h_mm',h, 'inner_lamD',opts.inner_lamD, ...
        'outer_lamD',opts.outer_lamD, 'contrast0',c0, ...
        'chain_opts',{ch.config}, 'a0',{opts.a0}, 'tag',opts.tag, ...
        'rx',ch.rx, 'timing_s',dt, 'date',datestr(now)); %#ok<TNOW1,DATST>
    if opts.save
        tg = '';  if ~isempty(opts.tag), tg = ['_' opts.tag]; end
        mat = fullfile(opts.outdir, sprintf('ctb_dm_jacobian_N%d%s.mat', ch.N, tg));
        save(mat, '-struct', 'out', '-v7.3');
        try
            meta = struct('rx', out.rx, 'h_mm', h, 'model_size', ch.N, ...
                'nact', opts.nact, 'beam_d_mm', opts.beam_d_mm, ...
                'coupling', opts.coupling, 'inner_lamD', opts.inner_lamD, ...
                'outer_lamD', opts.outer_lamD, 'date', out.date);
            jac_fingerprint('write', regexprep(mat, '\.mat$', '.fp.json'), ...
                out, meta);
        catch me
            warning('ctb_dm_jacobian:fp', 'fingerprint failed: %s', me.message);
        end
        fprintf('[jac] saved %s (%.1f MB)\n', mat, dir(mat).bytes/1e6);
    end
end
