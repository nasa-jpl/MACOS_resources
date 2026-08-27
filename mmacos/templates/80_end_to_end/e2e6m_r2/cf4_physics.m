function OUT = cf4_physics(over)
%CF4_PHYSICS  Coronagraph-family campaign S4: bandwidth + polarization.
%
%   The ctb_efc_physics / ctb_vortex_bandwidth replay on the e2e6m
%   winner operating point, N = P.dj.model (512), resumable per leg:
%
%   LEG 'pol'      mono + the coated train's Jones-pupil screens.
%                  Screens: protected Al (MgF2 90.6 nm over Al 220 nm)
%                  on all 31 reflectors, macos.jones_pupil at the exit
%                  pupil, normalized by the COMPLEX mean of Jxx (trap 5:
%                  magnitude-only normalization leaves the stack's
%                  global reflection phase in the screens and the
%                  correction lands with a 2-theta error, ADDING
%                  energy).  Per-component propagation via
%                  run_screened; ONE shared Jacobian (the S2 cache --
%                  screens are near-identity, verified and reported);
%                  control drives the CO-POLARIZED mean; the component
%                  spread about it is the pol floor.
%   LEG 'band'     the bandwidth ladder 0/5/10/20% under Dave's 2.5%
%                  color-spacing rule (1/3/5/9 colors), ONE 9-color
%                  superset Jacobian (0.90:0.025:1.10), per-band block
%                  subsets (the ctb_vortex_bandwidth pattern).  Focal
%                  masks are FIXED-METRES objects rebuilt per lambda
%                  (cf_chain.set_lambda); the dark zone is the FIXED
%                  PHYSICAL annulus (per-lambda pixel radii [3/lf,
%                  15/lf]); contrast normalizes by the band-MEAN peak
%                  (trap 4).
%   LEG 'bandpol'  band (10%, 5 colors) + screens together at the
%                  THROUGHPUT-REBALANCED Lyot (from the cf3a frontier),
%                  with its own 5-color Jacobian at that Lyot.
%
%   over.cf4 fields: family ('' = the S2 winner by relin floor),
%   legs ({'pol','band','bandpol'}), rebal_lyot (0.70), bands
%   ([0 .05 .10 .20]), niter (12), force (false).
%
%   See also CF2_EFC, CF3_LYOT, CF_CHAIN, ctb_efc_physics,
%   ctb_vortex_bandwidth.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    ov = over;  cf4 = struct();
    if isfield(ov, 'cf4'), cf4 = ov.cf4;  ov = rmfield(ov, 'cf4'); end
    P = e2e6m_r2_params(ov);
    if ~isfield(cf4,'family'),     cf4.family = '';                  end
    if ~isfield(cf4,'legs'),       cf4.legs = {'pol','band','bandpol'}; end
    if ~isfield(cf4,'rebal_lyot'), cf4.rebal_lyot = 0.70;            end
    if ~isfield(cf4,'bands'),      cf4.bands = [0 0.05 0.10 0.20];   end
    if ~isfield(cf4,'niter'),      cf4.niter = 12;                   end
    if ~isfield(cf4,'alphas'),     cf4.alphas = logspace(-6,-2,5);   end
    if ~isfield(cf4,'force'),      cf4.force = false;                end
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    lib = cf_efc_lib();  %#ok<NASGU>  (mono machinery; the band loop is local)

    % ---- the operating point -------------------------------------------
    C2 = load(fullfile(P.outdir,'cf2_run.mat'));
    if isempty(cf4.family)
        rel = structfun(@(r) r.c_relin, C2.OUT.R);
        [~, iw] = min(rel);
        cf4.family = C2.OUT.keys{iw};
    end
    key = cf4.family;
    C1 = load(fullfile(P.outdir,'cf1_run.mat'));
    fam = C1.OUT.F(strcmp({C1.OUT.F.key}, key));
    rx = fullfile(P.outdir, 'r1_seg_dm.in');
    A = load(fullfile(P.outdir,'r1_dm_run.mat'));  aug = A.OUT.aug;
    beam_d = 2 * 0.023771;

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m CF4 -- physics layers on %s', fam.name);
    L = say_(L, 'deck %s, model %d, physical annulus %g-%g lambda0/D', ...
             rx, P.dj.model, P.co.inner_lamD, P.co.outer_lamD);

    % ---- screens (cached; coated train, then a clean reload) ------------
    sp = fullfile(P.outdir, 'cf4_pol_screens.mat');
    if ~isfile(sp)
        L = say_(L, '\n[screens] coating the train + Jones pupil...');
        macos.init(P.dj.model);
        nE = macos.load_rx(rx);
        refl = find_reflectors_(rx);
        for e = refl
            macos.coating(e, 'index',[1.38 0.77], 'extinc',[0 6.08], ...
                          'thickness',[9.06e-8 2.2e-7]);   % metres (deck units)
        end
        jp = macos.jones_pupil(nE - 1);           % the exit pupil
        J = jp.J;  J(isnan(J)) = 0;
        s0 = mean(nonzeros(J(:,:,1,1)));          % COMPLEX mean of Jxx (trap 5)
        J = J / s0;
        SC = struct('J', J, 'norm', s0, 'leak', jp.leak, 'nrefl', numel(refl), ...
                    'coating', 'MgF2 90.6nm / Al 220nm on all reflectors');
        save(sp, '-struct', 'SC');
        L = say_(L, '[screens] %d reflectors coated; |mean Jxx| %.4f, leak %.3g -> %s', ...
                 numel(refl), abs(s0), jp.leak, sp);
    else
        SC = load(sp);
        L = say_(L, '\n[screens] cached: %s', sp);
    end
    screens = {SC.J(:,:,1,1), SC.J(:,:,2,1), SC.J(:,:,1,2), SC.J(:,:,2,2)};
    wcomp = [0.5 0.5 0.5 0.5];  ctrl = [1 4];
    dev = cellfun(@(S) max(abs(nonzeros(S) - 1)), screens([1 4]));
    xpk = cellfun(@(S) max(abs(S(:))), screens([2 3]));
    L = say_(L, '[screens] co-pol identity deviation %.3g/%.3g; cross-pol peak %.3g/%.3g', ...
             dev(1), dev(2), xpk(1), xpk(2));
    L = say_(L, '          (shared-Jacobian argument holds iff co-pol dev << 1; measured above)');

    R = struct();

    % =====================================================================
    if any(strcmp(cf4.legs, 'pol'))
        state = fullfile(P.outdir, sprintf('cf4_pol_%s.mat', key));
        if isfile(state) && ~cf4.force
            R.pol = load(state);
            L = say_(L, '\n[pol] SKIP (state exists): floor %.3e, pol_floor %.3e', ...
                     R.pol.c_after, R.pol.pol_floor);
        else
            L = say_(L, '\n[pol] mono + screens, shared S2 Jacobian');
            [ch, dm] = chain_(P, rx, fam, aug, beam_d, []);
            dzpix = find(ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD));
            j2f = fullfile(P.outdir, sprintf('cf2_G_%s.mat', ch.tag));
            assert(isfile(j2f), 'cf4: %s missing -- run cf2_efc first', j2f);
            J2 = load(j2f);
            ctb_jac_check(J2, ch.config, j2f);
            lb = cf_efc_lib();  lb.stamp_parity(J2, ch.config, j2f);
            r = efc_multi_(ch, dm, J2, {1.0}, {dzpix}, screens, wcomp, ctrl, ...
                           cf4.niter, cf4.alphas, L);
            r.leg = 'pol';  save(state, '-struct', 'r');
            R.pol = r;
            L = say_(L, '[pol] static %.3e -> floor %.3e | pol_floor %.3e', ...
                     r.c_before, r.c_after, r.pol_floor);
        end
    end

    % =====================================================================
    if any(strcmp(cf4.legs, 'band'))
        lf9 = 0.90:0.025:1.10;                     % the 9-color superset
        supercache = fullfile(P.outdir, sprintf('cf4_G_super_%s.mat', key));
        [ch, dm] = chain_(P, rx, fam, aug, beam_d, []);
        % GATE (brief amendment): the set_lambda mask memoization must be
        % lambda-correct -- a stale memo on the superset-Jacobian path is
        % silent and plausible.  Build path vs memo path must agree to
        % the bit at both ends of the ladder.
        for lfg = [min(lf9) max(lf9)]
            ch.set_lambda(lfg);   E1 = ch.run();     % first build
            ch.set_lambda(1.0);
            ch.set_lambda(lfg);   E2 = ch.run();     % memo hit
            dmax = max(abs(E1(:) - E2(:)));
            assert(dmax == 0, ...
                'cf4: set_lambda memo NOT lambda-correct at lf=%.3f (max dE %.3g)', ...
                lfg, dmax);
            L = say_(L, '[memo gate] lf %.3f: build vs memo bit-identical  [PASS]', lfg);
        end
        ch.set_lambda(1.0);
        JJ = jac_multi_(ch, dm, lf9, P, supercache, L);
        bres = struct('band',{}, 'ncol',{}, 'c_static',{}, 'c_after',{}, ...
                      'pol_floor',{});
        for b = cf4.bands
            nc = max(3, 2*round(b/0.05) + 1) * (b > 0) + (b == 0);
            lf = 1 + ((1:nc) - (nc+1)/2) * 0.025;
            state = fullfile(P.outdir, sprintf('cf4_band%02d_%s.mat', ...
                                               round(100*b), key));
            if isfile(state) && ~cf4.force
                r = load(state);
                L = say_(L, '\n[band %g%%] SKIP (state exists): floor %.3e', ...
                         100*b, r.c_after);
            else
                L = say_(L, '\n[band %g%%] %d colors %s', 100*b, nc, mat2str(lf));
                Jb = subset_(JJ, lf);
                [dzc, ~] = dz_sets_(ch, P, lf);
                r = efc_multi_(ch, dm, Jb, num2cell(lf), dzc, {[]}, 1, 1, ...
                               cf4.niter, cf4.alphas, L);
                r.leg = sprintf('band%02d', round(100*b));  r.lfracs = lf;
                save(state, '-struct', 'r');
                L = say_(L, '[band %g%%] static %.3e -> floor %.3e', ...
                         100*b, r.c_before, r.c_after);
            end
            bres(end+1) = struct('band',b, 'ncol',nc, 'c_static',r.c_before, ...
                'c_after',r.c_after, 'pol_floor',NaN); %#ok<AGROW>
        end
        R.band = bres;
    end

    % =====================================================================
    if any(strcmp(cf4.legs, 'bandpol'))
        state = fullfile(P.outdir, sprintf('cf4_bandpol_%s_L%03d.mat', ...
                                           key, round(100*cf4.rebal_lyot)));
        if isfile(state) && ~cf4.force
            R.bandpol = load(state);
            L = say_(L, '\n[bandpol] SKIP (state exists): floor %.3e, pol_floor %.3e', ...
                     R.bandpol.c_after, R.bandpol.pol_floor);
        else
            lf5 = 1 + (-2:2) * 0.025;              % the 10% band, 5 colors
            L = say_(L, '\n[bandpol] 10%% band + screens at the rebalanced Lyot %.2f', ...
                     cf4.rebal_lyot);
            [ch, dm] = chain_(P, rx, fam, aug, beam_d, cf4.rebal_lyot);
            Jr = jac_multi_(ch, dm, lf5, P, ...
                fullfile(P.outdir, sprintf('cf4_G_rebal_%s_L%03d.mat', ...
                    key, round(100*cf4.rebal_lyot))), L);
            [dzc, ~] = dz_sets_(ch, P, lf5);
            r = efc_multi_(ch, dm, Jr, num2cell(lf5), dzc, screens, wcomp, ...
                           ctrl, cf4.niter, cf4.alphas, L);
            r.leg = 'bandpol';  r.lfracs = lf5;  r.rebal_lyot = cf4.rebal_lyot;
            save(state, '-struct', 'r');
            R.bandpol = r;
            L = say_(L, '[bandpol] static %.3e -> floor %.3e | pol_floor %.3e | thru %.3f', ...
                     r.c_before, r.c_after, r.pol_floor, ch.thru);
        end
    end

    % ---- the table + figure ---------------------------------------------
    L = say_(L, '\n==== CF4 physics floors (%s, N=%d, physical annulus %g-%g lambda0/D) ====', ...
             fam.name, P.dj.model, P.co.inner_lamD, P.co.outer_lamD);
    L = say_(L, '  %-26s | %-10s | %-10s | %-10s', 'leg', 'static', 'floor', 'pol floor');
    L = say_(L, '  %s', repmat('-', 1, 66));
    if isfield(R,'pol')
        L = say_(L, '  %-26s | %.3e | %.3e | %.3e', 'polarization only (mono)', ...
                 R.pol.c_before, R.pol.c_after, R.pol.pol_floor);
    end
    if isfield(R,'band')
        for q = 1:numel(R.band)
            L = say_(L, '  %-26s | %.3e | %.3e | %-10s', ...
                     sprintf('band %g%% (%d colors)', 100*R.band(q).band, ...
                             R.band(q).ncol), ...
                     R.band(q).c_static, R.band(q).c_after, '--');
        end
    end
    if isfield(R,'bandpol')
        L = say_(L, '  %-26s | %.3e | %.3e | %.3e', ...
                 sprintf('band 10%% + pol @ Lyot %.2f', cf4.rebal_lyot), ...
                 R.bandpol.c_before, R.bandpol.c_after, R.bandpol.pol_floor);
    end
    L = say_(L, '  %s', repmat('-', 1, 66));

    png = fullfile(P.outdir, 'cf4_physics.png');
    fig_(R, P, fam, png);
    L = say_(L, '\n  figure: %s', png);
    L = say_(L, '\nCF4 DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf4_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('P',P, 'R',R, 'family',key, 'text',txt, 'figure',png, ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf4_run.mat'),'OUT');
end

% =========================================================================
function [ch, dm] = chain_(P, rx, fam, aug, beam_d, lyot_override)
    cfg = fam.cfg;
    if ~isempty(lyot_override)
        for i = 1:2:numel(cfg)-1
            if strcmp(cfg{i}, 'r_lyot_frac'), cfg{i+1} = lyot_override; end
        end
    end
    ch = cf_chain('rx', rx, 'model_size', P.dj.model, ...
                  'prolate_iter', P.co.prolate_iter, ...
                  'circ_stop_frac', P.cf.circ_stop_frac, cfg{:});
    dm = cell(1, numel(aug.ielt));
    for k = 1:numel(dm)
        dm{k} = ctb_dm('ielt', aug.ielt(k), 'ng', aug.ng, ...
                       'gdx_mm', aug.gdx_mm(k), 'nact', P.dj.nact, ...
                       'beam_d_mm', beam_d, 'pitch_mm', beam_d/P.dj.nact, ...
                       'coupling', P.dj.coupling);
        dm{k}.clear();
    end
end

function refl = find_reflectors_(rx)
%FIND_REFLECTORS_  Element indices with Element= Reflector or Segment.
    kinds = regexp(fileread(rx), '^\s*Element=\s*(\S+)', 'tokens', 'lineanchors');
    kinds = cellfun(@(c) c{1}, kinds, 'UniformOutput', false);
    refl = find(ismember(kinds, {'Reflector','Segment'})).';
end

function [dzc, dzM] = dz_sets_(ch, P, lfracs)
%DZ_SETS_  Per-lambda dark-zone pixel sets: the FIXED PHYSICAL annulus
%   [inner,outer] lambda0/D is the pixel annulus [inner/lf, outer/lf] on
%   each wavelength's grid (FPA pitch ~ lambda; trap 4).
    c = ch.center_px;  N = ch.N;
    [ii, jj] = ndgrid(1:N, 1:N);
    rl = hypot(ii - c, jj - c) / ch.lamD_px;
    dzc = cell(1, numel(lfracs));  dzM = cell(1, numel(lfracs));
    for l = 1:numel(lfracs)
        lf = lfracs(l);
        dzM{l} = rl >= P.co.inner_lamD/lf & rl <= P.co.outer_lamD/lf;
        dzc{l} = find(dzM{l});
    end
end

function JJ = jac_multi_(ch, dm, lfracs, P, cache, L) %#ok<INUSD>
%JAC_MULTI_  Per-lambda block Jacobian (no screens), cached + stamped.
%   Rows: per-lambda dark-zone pixel sets, stacked (rowoff blocks) --
%   the ctb_efc_physics pattern.
    if isfile(cache)
        JJ = load(cache);
        ctb_jac_check(JJ, ch.config, cache);
        assert(isequal(JJ.lfracs(:), lfracs(:)), ...
            'cf4: %s carries lfracs %s, requested %s', cache, ...
            mat2str(JJ.lfracs), mat2str(lfracs));
        fprintf('    [jacM] loaded %s (%d lambda x %d cols)\n', ...
                cache, numel(lfracs), size(JJ.G,2));
        return
    end
    nlam = numel(lfracs);
    dzc = dz_sets_(ch, P, lfracs);
    rowoff = [0 cumsum(cellfun(@numel, dzc))];
    nacts = cellfun(@(d) d.nact_active, dm);
    ncol = sum(nacts);
    G = complex(zeros(rowoff(end), ncol, 'single'));
    col_dm = zeros(1, ncol);  col_act = zeros(1, ncol);
    a0 = cellfun(@(d) zeros(d.nact^2,1), dm, 'UniformOutput', false);
    for k = 1:numel(dm), dm{k}.apply(a0{k}); end
    e0l = cell(1, nlam);
    for l = 1:nlam
        ch.set_lambda(lfracs(l));
        E = ch.run();
        e0l{l} = double(E(dzc{l}));
    end
    h = P.dj.h;  c = 0;  tswp = tic;
    for k = 1:numel(dm)
        act = find(dm{k}.active(:)).';
        for a = act
            c = c + 1;
            v = a0{k};  v(a) = v(a) + h;
            dm{k}.apply(v);
            for l = 1:nlam
                ch.set_lambda(lfracs(l));
                E = ch.run();
                G(rowoff(l)+1:rowoff(l+1), c) = ...
                    single((double(E(dzc{l})) - e0l{l}) / h);
            end
            col_dm(c) = k;  col_act(c) = a;
            if mod(c, 100) == 0
                el = toc(tswp);
                fprintf('    [jacM] %4d/%d pokes, %.1f min (ETA %.1f min)\n', ...
                        c, ncol, el/60, el/c*(ncol-c)/60);
            end
        end
        dm{k}.apply(a0{k});
    end
    ch.set_lambda(1.0);
    JJ = struct('G', G, 'col_dm', col_dm, 'col_act', col_act, ...
        'rowoff', rowoff, 'lfracs', lfracs, 'h', h, 'a0', {a0}, ...
        'chain_opts', {ch.config}, 'N', ch.N, 'rx', ch.rx, ...
        'lamD_px', ch.lamD_px, 'when', datestr(now,31)); %#ok<TNOW1,DATST>
    save(cache, '-struct', 'JJ', '-v7.3');
    jac_fingerprint('write', [cache(1:end-4) '.fp.json'], ...
        struct('G_re', real(G), 'G_im', imag(G)), ...
        struct('rx', string(ch.rx), 'model', ch.N, 'tag', string(ch.tag), ...
               'lfracs', lfracs, 'ncol', ncol, 'h_m', h, ...
               'when', string(datestr(now,31)))); %#ok<TNOW1,DATST>
    fprintf('    [jacM] measured %d lambda x %d cols in %.1f min -> %s\n', ...
            numel(lfracs), ncol, toc(tswp)/60, cache);
end

function Jb = subset_(JJ, lf)
%SUBSET_  Per-band block subset of the superset Jacobian.
    [tf, il] = ismember(round(lf*1e6), round(JJ.lfracs*1e6));
    assert(all(tf), 'cf4: band colors %s not all in the superset %s', ...
           mat2str(lf), mat2str(JJ.lfracs));
    rows = [];
    rowoff = 0;
    for q = il(:).'
        rows = [rows, JJ.rowoff(q)+1 : JJ.rowoff(q+1)]; %#ok<AGROW>
        rowoff(end+1) = numel(rows); %#ok<AGROW>
    end
    Jb = struct('G', JJ.G(rows,:), 'col_dm', JJ.col_dm, ...
        'col_act', JJ.col_act, 'rowoff', rowoff, 'lfracs', lf, ...
        'h', JJ.h, 'a0', {JJ.a0}, 'chain_opts', {JJ.chain_opts}, ...
        'N', JJ.N, 'rx', JJ.rx, 'lamD_px', JJ.lamD_px);
end

function r = efc_multi_(ch, dm, JJ, lfcell, dzc, screens, wcomp, ctrl, ...
                        niter, alphas, L) %#ok<INUSD>
%EFC_MULTI_  The ctb_efc_physics loop on cf_chain: per-(lambda,component)
%   measured fields, co-polarized per-lambda mean drive against the
%   stacked per-lambda Jacobian blocks, measured-contrast alpha line
%   search, monotone accept, accepted state restored.  Contrast = the
%   lambda-mean dark-zone mean over the BAND-MEAN peak (trap 4);
%   pol_floor = dark-zone mean of the component variance.
    nlam = numel(lfcell);  ncomp = numel(screens);
    lf = cell2mat(lfcell);
    % band-mean peak, screens included in the weighting
    pkband = 0;
    for l = 1:nlam
        ch.set_lambda(lf(l));
        pk_l = 0;
        for q = 1:ncomp
            Eb = ch.run_bare_screened(screens{q});
            pk_l = pk_l + wcomp(q) * abs(Eb).^2;
        end
        pkband = pkband + max(pk_l(:));
    end
    pkband = pkband / nlam;

    rowoff = JJ.rowoff;
    Gr = [real(double(JJ.G)); imag(double(JJ.G))];
    [U, S, V] = svd(Gr, 'econ');
    sv = diag(S);

    a = cellfun(@(d) zeros(d.nact^2,1), dm, 'UniformOutput', false);
    for k = 1:numel(dm), dm{k}.apply(a{k}); end
    [ef, C0] = measure_();
    contrast = zeros(1, niter+1);  contrast(1) = C0;
    fprintf('    [efcM] iter 0: %.3e\n', C0);
    for it = 1:niter
        em = zeros(rowoff(end), 1);
        for l = 1:nlam
            acc = 0;
            for q = ctrl
                acc = acc + wcomp(q) * ef{l}{q};
            end
            em(rowoff(l)+1:rowoff(l+1)) = acc / sum(wcomp(ctrl));
        end
        Ue = U' * [real(em); imag(em)];
        best = struct('c', inf);
        for al = alphas
            da = -V * ((sv ./ (sv.^2 + al*sv(1)^2)) .* Ue);
            at = a;
            for k = 1:numel(dm)
                sel = JJ.col_dm == k;
                at{k}(JJ.col_act(sel)) = at{k}(JJ.col_act(sel)) + da(sel);
                dm{k}.apply(at{k});
            end
            [eft, Ct] = measure_();
            if Ct < best.c
                best = struct('c',Ct, 'a',{at}, 'ef',{eft}, 'alpha',al);
            end
        end
        if best.c >= contrast(it)
            fprintf('    [efcM] iter %d: no alpha improves (best %.3e) -- stop\n', it, best.c);
            contrast = contrast(1:it);
            for k = 1:numel(dm), dm{k}.apply(a{k}); end   % restore ACCEPTED
            break
        end
        a = best.a;  ef = best.ef;
        contrast(it+1) = best.c;
        fprintf('    [efcM] iter %d: %.3e (alpha %.1e)\n', it, best.c, best.alpha);
    end
    for k = 1:numel(dm), dm{k}.apply(a{k}); end
    [ef, c_final] = measure_();
    contrast(end) = c_final;
    pf = pol_floor_(ef);
    strokes = cellfun(@(x) 1e9*rms_(x(x~=0)), a);
    for k = 1:numel(dm), dm{k}.clear(); end
    ch.set_lambda(1.0);
    r = struct('c_before', contrast(1), 'c_after', c_final, ...
        'contrast', contrast, 'pol_floor', pf, 'a', {a}, ...
        'strokes_nm', strokes, 'chain_opts', {ch.config});

    function [efields, C] = measure_()
        efields = cell(1, nlam);
        C = 0;
        for l2 = 1:nlam
            ch.set_lambda(lf(l2));
            efields{l2} = cell(1, ncomp);
            for q2 = 1:ncomp
                E = ch.run_screened(screens{q2});
                efields{l2}{q2} = double(E(dzc{l2}));
                C = C + wcomp(q2) * mean(abs(efields{l2}{q2}).^2);
            end
        end
        C = C / nlam / pkband;
    end

    function pfl = pol_floor_(efields)
        % the ctb_efc_physics definition verbatim: co-pol spread about
        % the co-pol mean + the full cross-polarized energy -- the part
        % no common surface can correct
        if ncomp == 1, pfl = NaN; return; end
        v = 0;
        for l2 = 1:nlam
            acc = 0;
            for q2 = ctrl, acc = acc + wcomp(q2) * efields{l2}{q2}; end
            acc = acc / sum(wcomp(ctrl));
            sprd = 0;
            for q2 = 1:ncomp
                if ismember(q2, ctrl)
                    sprd = sprd + wcomp(q2) * mean(abs(efields{l2}{q2} - acc).^2);
                else
                    sprd = sprd + wcomp(q2) * mean(abs(efields{l2}{q2}).^2);
                end
            end
            v = v + sprd;
        end
        pfl = v / nlam / pkband;
    end
end

function fig_(R, P, fam, png)
    f = figure('Visible','off','Color','w','Position',[60 60 900 620]);
    ax = axes(f); hold(ax,'on'); set(ax,'YScale','log');
    if isfield(R, 'band')
        b = [R.band.band];
        st = [R.band.c_static];  fl = [R.band.c_after];
        plot(ax, 100*b, st, 'ks--', 'MarkerFaceColor',[0.7 0.7 0.7], ...
             'LineWidth',1.2, 'MarkerSize',9);
        plot(ax, 100*b, fl, 'bo-', 'MarkerFaceColor',[0.4 0.6 0.9], ...
             'LineWidth',1.7, 'MarkerSize',9);
    end
    lg = {'static (band-mean)', 'EFC floor (per-band colors)'};
    if isfield(R, 'pol')
        yline(ax, R.pol.c_after, '-', sprintf('pol-only floor %.1e', R.pol.c_after), ...
              'Color',[0.2 0.6 0.3], 'LineWidth', 1.4);
        yline(ax, R.pol.pol_floor, ':', sprintf('pol floor (uncontrollable) %.1e', ...
              R.pol.pol_floor), 'Color',[0.2 0.6 0.3]);
    end
    if isfield(R, 'bandpol')
        plot(ax, 10, R.bandpol.c_after, 'p', 'MarkerSize', 16, ...
             'MarkerFaceColor',[0.9 0.5 0.2], 'MarkerEdgeColor','k');
        lg{end+1} = sprintf('band+pol @ Lyot %.2f', R.bandpol.rebal_lyot);
    end
    grid(ax,'on'); box(ax,'on');
    xlabel(ax,'bandwidth (%)');
    ylabel(ax,'dark-zone mean contrast (band-mean-peak normalized)');
    legend(ax, lg, 'Location','best');
    title(ax, sprintf(['e2e6m %s -- physics floors (N=%d, 2.5%% color rule, ' ...
        'physical annulus %g-%g \\lambda_0/D)'], fam.name, P.dj.model, ...
        P.co.inner_lamD, P.co.outer_lamD), 'Interpreter','tex');
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function r = rms_(v), v = v(:); if isempty(v), r = 0; else, r = sqrt(mean(v.^2)); end, end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
