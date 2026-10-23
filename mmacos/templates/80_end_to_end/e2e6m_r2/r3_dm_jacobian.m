function OUT = r3_dm_jacobian(over)
%R3_DM_JACOBIAN  Engine-measured EFC Jacobian for the e2e6m train.
%
%   G(:,c) = (E_dz(poke_c) - E_dz(0)) / h -- the change of the complex
%   Science-plane field on the dark-zone pixels per metre of surface on
%   each active actuator of DM1/DM2, measured through the FULL masked
%   chain (prolate + FPM + Lyot applied in place).  Engine-faithful by
%   construction: the S3b lesson says an optimizer mines any model gap,
%   so the design operator IS the engine.
%
%   Machinery is the committed CTB core, pointed at this deck:
%   `ctb_chain` (which takes 'rx' + 'elt') runs the masked walk with the
%   PROLATE injected via run_screened (its own apodizer mask disabled);
%   `ctb_dm` provides the influence-function DM models on the GridData
%   surfaces r1_dm built.  All '_mm' fields carry METRES (deck units).
%
%   The Jacobian runs at P.dj.model (512): every scale is re-measured
%   at that N, so it is self-consistent, and R4's EFC scores at the
%   same N.  Artifact: r3_dmjac.mat (gitignored) + committed
%   fingerprint r3_dmjac.fp.json.
%
%   OUT = R3_DM_JACOBIAN()      defaults (~1-2 h)
%   OUT = R3_DM_JACOBIAN(OVER)  with e2e6m_r2_params overrides
%
%   See also CTB_CHAIN, CTB_DM, CTB_DM_JACOBIAN, R1_DM, R4 (the EFC).

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    addpath(fullfile(here,'..','..','..','design','src'));

    rx = fullfile(P.outdir, 'r1_seg_dm.in');
    assert(isfile(rx), 'r3_dm_jacobian: %s not found -- run r1_dm first', rx);
    A = load(fullfile(P.outdir,'r1_dm_run.mat'));
    aug = A.OUT.aug;

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m R3 -- the EFC Jacobian');
    L = say_(L, 'deck %s, model %d, poke %g nm surface', rx, P.dj.model, P.dj.h*1e9);

    % ---- the masked chain on this deck ----------------------------------
    e = elt_map_(rx);
    N = P.dj.model;
    ch = ctb_chain('rx', rx, 'elt', e, 'model_size', N, ...
                   'apodizer', false, ...     % the prolate rides in as a screen
                   'fpm', true, 'r_fpm_lamD', P.co.r_occ_lamD, ...
                   'lyot', true, 'r_lyot_frac', P.co.r_lyot_frac);
    Iap = macos.intensity(e.Apodizer);
    r_apod_px = beam_radius_(Iap, 1);
    Phi = ctb_apod_prolate(N, r_apod_px, P.co.r_occ_lamD, ...
                           'n_iter', P.co.prolate_iter);
    L = say_(L, 'chain: lambda/D %.3f px, apodizer r %.1f px, peak_bare %.4e', ...
             ch.lamD_px, r_apod_px, ch.peak_bare);

    dz = ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD);
    dz_idx = find(dz);
    L = say_(L, 'dark zone %g-%g lambda/D: %d pixels', ...
             P.co.inner_lamD, P.co.outer_lamD, numel(dz_idx));

    % ---- the DM models ---------------------------------------------------
    beam_d = 2 * 0.023771;                  % measured pupil at the DMs (R1)
    ndm = numel(aug.ielt);
    dm = cell(1, ndm);
    for k = 1:ndm
        dm{k} = ctb_dm('ielt', aug.ielt(k), 'ng', aug.ng, ...
                       'gdx_mm', aug.gdx_mm(k), 'nact', P.dj.nact, ...
                       'beam_d_mm', beam_d, 'pitch_mm', beam_d/P.dj.nact, ...
                       'coupling', P.dj.coupling);
    end
    nacts = cellfun(@(d) d.nact_active, dm);
    ncol  = sum(nacts);
    L = say_(L, 'DMs: %d lattices %dx%d, active %s -> %d columns', ...
             ndm, P.dj.nact, P.dj.nact, mat2str(nacts), ncol);

    % ---- the sweep -------------------------------------------------------
    E0 = ch.run_screened(Phi);
    E0_dz = single(E0(dz_idx));
    G = zeros(numel(dz_idx), ncol, 'single');  G = complex(G, G);
    col_dm = zeros(1, ncol);  col_act = zeros(1, ncol);
    c = 0;  h = P.dj.h;  tswp = tic;
    for k = 1:ndm
        act = find(dm{k}.active(:)).';
        for a = act
            c = c + 1;
            v = zeros(dm{k}.nact^2, 1);  v(a) = h;
            dm{k}.apply(v);
            E = ch.run_screened(Phi);
            G(:,c) = (single(E(dz_idx)) - E0_dz) / h;
            col_dm(c) = k;  col_act(c) = a;
            if mod(c, 50) == 0
                L = say_(L, '    %4d/%d columns, %.2f s per poke', ...
                         c, ncol, toc(tswp)/c);
            end
        end
        dm{k}.clear();
    end
    L = say_(L, 'sweep done: %d columns in %.1f min (%.2f s per poke)', ...
             ncol, toc(tswp)/60, toc(tswp)/ncol);

    % ---- gates -----------------------------------------------------------
    cn = sqrt(sum(abs(G).^2, 1));
    L = say_(L, 'column norms: median %.4g, min %.4g, zero-cols %d  [%s]', ...
             median(cn), min(cn), nnz(cn == 0), ...
             gate_(nnz(cn == 0) == 0 && all(isfinite(cn))));

    OUT = struct('G', G, 'col_dm', col_dm, 'col_act', col_act, ...
                 'dz_idx', dz_idx, 'E0_dz', E0_dz, 'h', h, ...
                 'lamD_px', ch.lamD_px, 'center_px', ch.center_px, ...
                 'peak_bare', ch.peak_bare, 'r_apod_px', r_apod_px, ...
                 'N', N, 'rx', rx, 'aug', aug, 'nacts', nacts, ...
                 'beam_d', beam_d, 'P', P, 'when', datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'r3_dmjac.mat'), 'OUT', '-v7.3');
    jac_fingerprint('write', fullfile(P.outdir,'r3_dmjac.fp.json'), ...
        struct('G_re', real(G), 'G_im', imag(G)), struct( ...
        'rx', string(rx), 'model', N, 'nact', P.dj.nact, ...
        'h_m', h, 'ncol', ncol, 'npix', numel(dz_idx), ...
        'inner_lamD', P.co.inner_lamD, 'outer_lamD', P.co.outer_lamD, ...
        'when', string(datestr(now,31)))); %#ok<TNOW1,DATST>

    L = say_(L, '\nR3 dm_jacobian DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'r3_dmjac_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
end

% =========================================================================
function e = elt_map_(rx)
    nm = regexp(fileread(rx), '^\s*EltName=\s*(\S+)', 'tokens','lineanchors');
    nm = cellfun(@(c) c{1}, nm, 'UniformOutput', false);
    at = @(s) find(strcmp(nm, s), 1);
    e = struct('DM1',at('DM1'), 'DM2',at('DM2'), ...
               'Apodizer',at('Apodizer'), 'FPM',at('FPM'), ...
               'Lyot',at('Lyot'), 'ExitPupil',at('ExitPupil'), ...
               'FPA',at('Science'));
    f = fieldnames(e);
    for k = 1:numel(f)
        assert(~isempty(e.(f{k})), 'r3_dm_jacobian: %s not in %s', f{k}, rx);
    end
end

function r = beam_radius_(I, dx)
    m = I > 0.02*max(I(:));
    [rr,cc] = find(m);
    if isempty(rr), r = 0;  return; end
    r = 0.5 * max(max(rr)-min(rr), max(cc)-min(cc)) * dx;
end

function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
function s = gate_(ok), if ok, s = 'PASS'; else, s = 'FAIL'; end, end
