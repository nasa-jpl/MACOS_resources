function out = ctb_efc_physics(opts)
%CTB_EFC_PHYSICS  EFC with the next physics layers: finite band + polarization.
%   out = CTB_EFC_PHYSICS('band',true,'pol',true) closes the EFC loop on
%   the vortex chain under a finite bandpass and/or coating polarization,
%   separately or together, and finds the floor.
%
%   PHYSICS MODELS
%   Band: per-wavelength propagation (macos.set_src_wvl; the SESSION-5
%   rules -- per-lambda grids, FPA pitch ~ lambda, pupil pitch invariant,
%   the vortex mask is angle-only so nothing rebuilds).  The dark zone is
%   the FIXED PHYSICAL annulus (3-15 lambda0/D at band center), which on
%   each wavelength's grid is the pixel annulus [3/lf, 15/lf].
%   Polarization: the coated train's Jones pupil (protected Al: MgF2
%   quarter-wave over 220 nm Al on all 10 reflectors), computed by ray
%   trace (macos.jones_pupil at the exit pupil) and applied as
%   PER-COMPONENT PUPIL SCREENS via apodize_complex -- the roadmap
%   decomposition: each Jones component propagates as its own scalar
%   chain; unpolarized input = the 1/2-weighted sum of the four.  (The
%   engine's native vector mode would miss the downstream coated OAPs --
%   the documented Tranche-2 gap -- so the screen decomposition is the
%   honest path.)  Screens vary from identity by <4e-3 on this bench, so
%   ONE Jacobian per wavelength (no screen) serves every component; only
%   the MEASURED fields are per-component.  With a shared G the stacked
%   least squares is exactly EFC on the COMPONENT-MEAN field, and the
%   component spread about that mean is the irreducible polarization
%   floor -- reported separately as out.pol_floor.
%
%   Name-value:
%     'band'       false (mono) | true (default grid [0.95 1 1.05])
%     'lfracs'     wavelength fractions when band=true
%     'pol'        false | true (coat + Jones screens, unpolarized input)
%     'chain'      ctb_chain config (default: vortex charge 4, Lyot 0.60,
%                  no apodizer)
%     'jac'        Jacobian .mat path ('' = measure; tagged save)
%     'niter'      EFC iterations (15)
%     'alphas'     regularization grid (logspace(-6,-2,5))
%     'a0'         warm-start commands (relinearization)
%     'tag'        run tag for saved products
%     'nact','beam_d_mm','coupling','h_mm'  DM/Jacobian params (as
%                  ctb_dm_jacobian defaults)
%     'inner_lamD','outer_lamD'  physical annulus at band center (3, 15)
%     'save','outdir','verbose'
%
%   out: contrast (per iteration, band+pol-weighted, Strehl-normalized),
%   c_before/c_after, pol_floor (dark-zone mean of the component
%   variance -- the uncontrollable part), a (commands), per-config meta.
%
%   Run:  >> out = ctb_efc_physics('band',true,'pol',true);
%   See also: ctb_chain, ctb_dm, ctb_efc, ctb_dm_jacobian, macos.jones_pupil.
    arguments
        opts.band       (1,1) logical = false
        opts.lfracs     (1,:) double = [0.95 1.00 1.05]
        opts.pol        (1,1) logical = false
        opts.chain      (1,:) cell = {'fpm_kind','vortex','charge',4, ...
                                      'apodizer',false,'r_lyot_frac',0.60}
        opts.jac        (1,:) char = ''
        opts.niter      (1,1) double {mustBeInteger,mustBePositive} = 15
        opts.alphas     (1,:) double = logspace(-6, -2, 5)
        opts.a0         (1,:) cell = {}
        opts.tag        (1,:) char = ''
        opts.nact       (1,1) double = 32
        opts.beam_d_mm  (1,1) double = 21.3
        opts.coupling   (1,1) double = 0.12
        opts.h_mm       (1,1) double = 2e-6
        opts.inner_lamD (1,1) double = 3.0
        opts.outer_lamD (1,1) double = 15.0
        opts.save       (1,1) logical = true
        opts.outdir     (1,:) char = ''
        opts.verbose    (1,1) logical = true
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', '..', 'src'));
    if isempty(opts.outdir), opts.outdir = here; end
    if ~opts.band, opts.lfracs = 1.0; end
    nlam = numel(opts.lfracs);

    % ---- deck + chain + DMs --------------------------------------------
    r  = ctb_dm_rx();
    ch = ctb_chain('rx', r.rx_out, 'model_size', 512, opts.chain{:});
    N = ch.N;
    lam0 = macos.get_src_wvl();
    restore_wvl = onCleanup(@() macos.set_src_wvl(lam0));
    ndm = 2;
    dm = cell(1, ndm);
    for k = 1:ndm
        dm{k} = ctb_dm('ielt', r.ielt(k), 'ng', r.ng, 'gdx_mm', r.gdx_mm(k), ...
                       'nact', opts.nact, 'beam_d_mm', opts.beam_d_mm, ...
                       'coupling', opts.coupling);
    end
    if isempty(opts.a0)
        opts.a0 = arrayfun(@(~) zeros(opts.nact^2,1), 1:ndm, 'UniformOutput', false);
    end
    for k = 1:ndm, dm{k}.apply(opts.a0{k}); end

    % ---- polarization screens ------------------------------------------
    % Components are the four Jones entries; unpolarized input weights 1/2
    % per input state.  Screens are computed once (coated train, ray
    % trace) and cached beside the drivers.  Scalar chain runs keep the
    % engine polarization OFF -- coatings act only through the screens.
    if opts.pol
        sp = fullfile(here, 'ctb_pol_screens.mat');
        if isfile(sp)
            SC = load(sp);
        else
            fprintf('[phys] computing coated-train Jones screens...\n');
            refl = [1 2 5 6 11 14 19 21 26 28];
            for e = refl
                macos.coating(e, 'index',[1.38 0.77], 'extinc',[0 6.08], ...
                              'thickness',[9.06e-5 2.2e-4]);
            end
            jp = macos.jones_pupil(ch.elt.ExitPupil);
            J = jp.J;  J(isnan(J)) = 0;
            % normalize by the COMPLEX mean of Jxx: removes the flux scale
            % AND the coating stack's global reflection phase (a common
            % piston -- exact for contrast).  Magnitude-only normalization
            % leaves that phase in the screens, and fields measured through
            % them rotate ~143 deg against the screen-free Jacobian: the
            % correction then lands with a 2-theta phase error and ADDS
            % energy (measured: achieved-vs-predicted corr -0.80).
            s0 = mean(nonzeros(J(:,:,1,1)));
            J = J / s0;
            SC = struct('J', J, 'norm', s0, 'leak', jp.leak, ...
                        'coating', 'MgF2 90.6nm / Al 220nm on 10 reflectors');
            save(sp, '-struct', 'SC');
            fprintf('[phys] screens cached: %s\n', sp);
        end
        screens = {SC.J(:,:,1,1), SC.J(:,:,2,1), SC.J(:,:,1,2), SC.J(:,:,2,2)};
        wcomp = [0.5 0.5 0.5 0.5];         % unpolarized: 1/2 per input state
        % CONTROL drives only the co-polarized components: the cross terms
        % are ~4 decades weaker and not correctable by a phase common to
        % both polarizations -- averaging their near-zero fields into the
        % drive would only dilute it 2x.  They stay in the SCORE and in
        % the pol-floor metric.
        ctrl = [1 4];
    else
        screens = {[]};
        wcomp = 1;
        ctrl = 1;
    end
    ncomp = numel(screens);

    % ---- per-lambda dark zones + normalization -------------------------
    c = ch.center_px;
    [ii, jj] = ndgrid(1:N, 1:N);
    rl = hypot(ii - c, jj - c) / ch.lamD_px;
    dzM = cell(1, nlam);  dz = cell(1, nlam);
    pkband = 0;
    for l = 1:nlam
        lf = opts.lfracs(l);
        dzM{l} = rl >= opts.inner_lamD/lf & rl <= opts.outer_lamD/lf;
        dz{l} = find(dzM{l});
        macos.set_src_wvl(lam0 * lf);
        pk_l = 0;
        for q = 1:ncomp
            Eb = ch.run_bare_screened(screens{q});
            pk_l = pk_l + wcomp(q) * abs(Eb).^2;
        end
        pkband = pkband + max(pk_l(:));
    end
    pkband = pkband / nlam;      % band-MEAN peak: contrast is then the
                                 % lambda-mean dark-zone mean over the
                                 % lambda-mean peak (a summed peak would
                                 % understate the contrast by nlam)

    % ---- measured-field + contrast closures ----------------------------
    function [efields, C] = measure_()
        % efields{l}{q}: dark-zone complex field at (lambda l, comp q);
        % C: band+pol-weighted dark-zone mean contrast
        efields = cell(1, nlam);
        for l2 = 1:nlam
            macos.set_src_wvl(lam0 * opts.lfracs(l2));
            efields{l2} = cell(1, ncomp);
            for q2 = 1:ncomp
                E = ch.run_screened(screens{q2});
                efields{l2}{q2} = double(E(dz{l2}));
            end
        end
        % weighted mean over lambda of dark-zone means, / band-summed peak
        C = 0;
        for l2 = 1:nlam
            for q2 = 1:ncomp
                C = C + wcomp(q2) * mean(abs(efields{l2}{q2}).^2);
            end
        end
        C = C / nlam / pkband;
    end

    % ---- Jacobian: one per lambda, no screens --------------------------
    tg = '';  if ~isempty(opts.tag), tg = ['_' opts.tag]; end
    jp_ = opts.jac;
    if isempty(jp_)
        jp_ = fullfile(opts.outdir, sprintf('ctb_dm_jacobian_N%d_phys%s.mat', N, tg));
    end
    if isfile(jp_)
        JJ = load(jp_);
        fprintf('[phys] Jacobian loaded: %s\n', jp_);
    else
        ncol = sum(cellfun(@(d) d.nact_active, dm));
        % block rows: per-lambda dark-zone pixel sets, stacked
        rowoff = [0 cumsum(cellfun(@numel, dz))];
        G = complex(zeros(rowoff(end), ncol, 'single'));
        col_dm = zeros(1, ncol);  col_act = zeros(1, ncol);
        % baseline fields per lambda (no screen)
        e0l = cell(1, nlam);
        for l = 1:nlam
            macos.set_src_wvl(lam0 * opts.lfracs(l));
            E = ch.run_screened([]);
            e0l{l} = double(E(dz{l}));
        end
        h = opts.h_mm;  t0 = tic;  cc = 0;
        for k = 1:ndm
            acts = find(dm{k}.active).';
            for j = acts
                a = opts.a0{k};  a(j) = a(j) + h;
                dm{k}.apply(a);
                cc = cc + 1;
                for l = 1:nlam
                    macos.set_src_wvl(lam0 * opts.lfracs(l));
                    E = ch.run_screened([]);
                    G(rowoff(l)+1:rowoff(l+1), cc) = ...
                        single((E(dz{l}) - e0l{l}) / h);
                end
                col_dm(cc) = k;  col_act(cc) = j;
                if opts.verbose && mod(cc, 100) == 0
                    el = toc(t0);
                    fprintf('[phys jac] %d/%d pokes  %.1f min (ETA %.1f min)\n', ...
                        cc, ncol, el/60, el/cc*(ncol-cc)/60);
                end
            end
            dm{k}.apply(opts.a0{k});
        end
        JJ = struct('G',G, 'col_dm',col_dm, 'col_act',col_act, ...
            'rowoff',rowoff, 'lfracs',opts.lfracs, 'h_mm',h, ...
            'a0',{opts.a0}, 'chain_opts',{ch.config}, 'N',N, ...
            'timing_s',toc(t0));
        save(jp_, '-struct', 'JJ', '-v7.3');
        fprintf('[phys] Jacobian saved: %s (%.1f min)\n', jp_, JJ.timing_s/60);
    end
    Gr = [real(double(JJ.G)); imag(double(JJ.G))];
    [U, S, V] = svd(Gr, 'econ');
    sv = diag(S);
    rowoff = JJ.rowoff;

    % ---- EFC loop -------------------------------------------------------
    a = opts.a0;
    [ef, C0] = measure_();
    contrast = zeros(1, opts.niter + 1);
    contrast(1) = C0;
    fprintf('[phys] iter 0: contrast %.3e  (band=%d pol=%d)\n', C0, opts.band, opts.pol);
    for it = 1:opts.niter
        % co-polarized mean field per lambda (shared-G equivalence), stacked
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
        for al = opts.alphas
            da = -V * ((sv ./ (sv.^2 + al * sv(1)^2)) .* Ue);
            at = a;
            for k = 1:ndm
                sel = JJ.col_dm == k;
                at{k}(JJ.col_act(sel)) = at{k}(JJ.col_act(sel)) + da(sel);
                dm{k}.apply(at{k});
            end
            [eft, Ct] = measure_();
            if Ct < best.c
                best = struct('c', Ct, 'a', {at}, 'ef', {eft}, 'alpha', al);
            end
        end
        if best.c >= contrast(it)
            fprintf('[phys] iter %d: no alpha improves (best %.3e) -- stop\n', it, best.c);
            contrast = contrast(1:it);
            for k = 1:ndm, dm{k}.apply(a{k}); end
            break;
        end
        a = best.a;  ef = best.ef;
        contrast(it+1) = best.c;
        str = cellfun(@(x) 1e6 * rms(x(x~=0)), a);
        fprintf('[phys] iter %d: contrast %.3e (alpha %.0e, stroke rms [%s] nm)\n', ...
            it, best.c, best.alpha, num2str(str, '%.2f '));
    end
    for k = 1:ndm, dm{k}.apply(a{k}); end
    [ef, c_final] = measure_();

    % ---- the polarization floor: component spread about the mean -------
    % uncontrollable-by-a-common-surface part: co-pol spread about the
    % co-pol mean, plus the full cross-polarized energy
    pol_floor = NaN;
    if opts.pol
        v = 0;
        for l = 1:nlam
            acc = 0;
            for q = ctrl, acc = acc + wcomp(q) * ef{l}{q}; end
            acc = acc / sum(wcomp(ctrl));
            sprd = 0;
            for q = 1:ncomp
                if ismember(q, ctrl)
                    sprd = sprd + wcomp(q) * mean(abs(ef{l}{q} - acc).^2);
                else
                    sprd = sprd + wcomp(q) * mean(abs(ef{l}{q}).^2);
                end
            end
            v = v + sprd;
        end
        pol_floor = v / nlam / pkband;
    end

    out = struct('contrast',contrast, 'c_before',contrast(1), ...
        'c_after',c_final, 'pol_floor',pol_floor, 'a',{a}, ...
        'band',opts.band, 'pol',opts.pol, 'lfracs',opts.lfracs, ...
        'chain',{opts.chain}, 'niter_run',numel(contrast)-1, ...
        'stroke_rms_nm',cellfun(@(x) 1e6*rms(x(x~=0)), a), 'jac',jp_, ...
        'inner_lamD',opts.inner_lamD, 'outer_lamD',opts.outer_lamD);
    fprintf('[phys] final: %.3e -> %.3e (%.1fx)%s\n', out.c_before, out.c_after, ...
        out.c_before/max(out.c_after, realmin), ...
        ternary_(opts.pol, sprintf('  pol floor %.3e', pol_floor), ''));
    if opts.save
        save(fullfile(opts.outdir, ['ctb_efc_phys' tg '.mat']), '-struct', 'out', '-v7.3');
    end
end

function o = ternary_(c, x, y)
    if c, o = x; else, o = y; end
end
