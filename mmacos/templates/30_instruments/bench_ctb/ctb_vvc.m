function out = ctb_vvc(opts)
%CTB_VVC  Vector-vortex coronagraph on the CTB: ideal, chromatic, +coatings.
%   out = CTB_VVC() runs the charge-4 VECTOR vortex (Lyot 0.60, no
%   apodizer) through the component-chain machinery and closes the EFC
%   loop.  Three tiers ('tier'):
%     'ideal'      delta = pi at every wavelength -- the validation
%                  tier: an ideal even-charge VVC acts on each
%                  polarization like the scalar vortex, so its static
%                  contrast and floor must reproduce the scalar chain.
%     'chromatic'  zero-order plate, delta(lambda) = pi*lambda0/lambda:
%                  the leakage term cos(delta/2)*I -- starlight with no
%                  spiral -- passes the coronagraph and maps the classic
%                  VVC bandwidth limitation.
%   'pol' true additionally composes the coated-train Jones screens at
%   the pupil (input-side), the full stack.
%
%   MECHANICS.  The VVC is a 2x2 Jones focal-plane mask
%     J = e^{i d/2}[cos(d/2) I - i sin(d/2) M],  M = [Vc Vs; Vs -Vc]
%   (ctb_mask_vvc, 8x-binned entries).  Everything downstream of the
%   FPM is linear and scalar, so each OUTPUT component is the coherent
%   sum of TWO chain runs, one per FPM entry mask -- no engine work.
%   Unpolarized input: 2 inputs x 2 outputs x 2 terms = 8 runs per
%   wavelength per evaluation.
%
%   CONTROL.  Two Jacobians serve every tier: pokes through the Vc mask
%   (G_cos) and the Vs mask (G_sin), measured at band center.  The
%   dominant poke responses are dE_xx ~ +b G_cos, dE_yx ~ b G_sin,
%   dE_xy ~ b G_sin, dE_yy ~ -b G_cos (b = -i e^{i d/2} sin(d/2)); the
%   alpha-leakage response and the per-wavelength G variation are
%   DROPPED from the model (<~10% at 20%-band edges) -- the line search
%   scores measured contrast, so model error slows convergence rather
%   than corrupting it.  With a shared column space the stacked least
%   squares collapses to two blocks:
%     e_cos = conj(b)(e_xx - e_yy)/(2|b|^2),  weight 2|b|^2
%     e_sin = conj(b)(e_yx + e_xy)/(2|b|^2),  weight 2|b|^2
%   summed over wavelengths, solved against [G_cos; G_sin] real-stacked.
%
%   Name-value: 'tier' ('chromatic'), 'pol' (false), 'band' (0 = mono;
%   or fractional bandwidth, colors per the 2.5%-spacing rule),
%   'charge' (4), 'niter' (12), 'tag', 'save', 'outdir', 'verbose'.
%
%   out: contrast series, c_before/c_after, leak_frac (band-mean
%   cos^2(d/2)), a (commands), config meta.
%
%   Run:  >> out = ctb_vvc('tier','ideal');        % validation, ~30 min
%         >> out = ctb_vvc('band',0.10);           % chromatic 10%, 5 colors
%   See also: ctb_mask_vvc, ctb_efc_physics, ctb_vortex_bandwidth.
    arguments
        opts.tier    (1,:) char {mustBeMember(opts.tier, {'ideal','chromatic'})} = 'chromatic'
        opts.analyzer (1,:) char {mustBeMember(opts.analyzer, {'none','circular'})} = 'none'
        opts.input   (1,:) char {mustBeMember(opts.input, {'unpolarized','circular'})} = 'unpolarized'
        opts.jac_perlam (1,1) logical = false   % per-wavelength Jacobian
                                                % blocks + stacked solve
                                                % (circular input only) --
                                                % the chromatic-control
                                                % recovery; costs nlam x
                                                % the poke bill
        opts.pol     (1,1) logical = false
        opts.band    (1,1) double {mustBeNonnegative} = 0
        opts.charge  (1,1) double = 4
        opts.niter   (1,1) double = 12
        opts.alphas  (1,:) double = logspace(-6, -2, 5)
        opts.tag     (1,:) char = ''
        opts.h_mm    (1,1) double = 2e-6
        opts.nact    (1,1) double = 32
        opts.save    (1,1) logical = true
        opts.outdir  (1,:) char = ''
        opts.verbose (1,1) logical = true
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', '..', 'src'));
    if isempty(opts.outdir), opts.outdir = here; end

    % control wavelengths: 2.5%-spacing rule
    if opts.band == 0
        lfracs = 1.0;
    else
        ncol = max(3, 2*round(opts.band/0.05) + 1);
        lfracs = 1 + opts.band * linspace(-0.5, 0.5, ncol);
    end
    nlam = numel(lfracs);

    % ---- chain (vortex config with the FPM left OFF: the VVC entries
    % are applied per run via run_screened's mask hook is not enough --
    % we apply the entry mask ourselves at the FPM, so build the chain
    % with fpm=false and multiply explicitly) -------------------------
    r  = ctb_dm_rx();
    ch = ctb_chain('rx', r.rx_out, 'model_size', 512, ...
        'fpm_kind','vortex', 'charge',opts.charge, 'apodizer',false, ...
        'fpm',false, 'r_lyot_frac',0.60);
    N = ch.N;  e = ch.elt;
    lam0 = macos.get_src_wvl();
    restore_wvl = onCleanup(@() macos.set_src_wvl(lam0));
    [Vc, Vs] = ctb_mask_vvc(N, opts.charge);

    dm = cell(1,2);
    for k = 1:2
        dm{k} = ctb_dm('ielt', r.ielt(k), 'ng', r.ng, 'gdx_mm', r.gdx_mm(k), ...
                       'nact', opts.nact);
        dm{k}.clear();
    end

    % input-side screens (coating Jones) when pol is on
    if opts.pol
        sp = fullfile(here, 'ctb_pol_screens.mat');
        assert(isfile(sp), 'ctb_vvc: cache screens first (ctb_efc_physics pol run)');
        SC = load(sp);
        Sin = {SC.J(:,:,1,1), SC.J(:,:,2,1); SC.J(:,:,1,2), SC.J(:,:,2,2)};
        % Sin{c,q}: input state q -> component c at the pupil
    else
        Sin = {[], 0; 0, []};   % identity input: component c==q only
    end

    % retardance per wavelength
    function [al, be] = ab_(lf)
        if strcmp(opts.tier, 'ideal'), d = pi; else, d = pi / lf; end
        al = exp(1i*d/2) * cos(d/2);
        be = -1i * exp(1i*d/2) * sin(d/2);
    end

    % one chain run: pupil screen Spup (or []), FPM entry mask Mf
    function E = run1_(Spup, Mf)
        macos.intensity(e.Apodizer);
        if ~isempty(Spup)
            macos.apodize_complex(e.Apodizer, Spup);
        end
        macos.intensity(e.FPM, 'reset_trace', false);
        macos.apodize_complex(e.FPM, Mf);
        macos.intensity(e.Lyot, 'reset_trace', false);
        macos.apodize(e.Lyot, ch.masks.L);
        E = macos.complex_field(e.FPA, 'reset_trace', false);
    end

    % full evaluation at one wavelength: the 4 output fields (dz pixels)
    % E_pq = sum_c A[J_pc . C[Sin_cq . P]]
    % entry masks: Jxx = al + be*Vc, Jyx = be*Vs, Jxy = be*Vs, Jyy = al - be*Vc
    function ef = fields_(dzidx, lf)
        [al, be] = ab_(lf);
        Jm = {al + be*Vc, be*Vs; be*Vs, al - be*Vc};   % Jm{p,c}
        if strcmp(opts.input, 'circular')
            % ONE input state, R = (x - i y)/sqrt(2): the component
            % pupil fields carry weights (1, -i)/sqrt(2) (through the
            % coating screens when pol is on).  4 runs per wavelength.
            win = [1, -1i] / sqrt(2);
            ef = cell(2,1);                             % ef{p}
            for p = 1:2
                acc = 0;
                for c = 1:2
                    if opts.pol
                        Sc = (Sin{c,1} * win(1) + Sin{c,2} * win(2));
                        E = run1_(Sc, Jm{p,c});
                        acc = acc + E(dzidx);
                    else
                        E = run1_([], Jm{p,c});
                        acc = acc + win(c) * E(dzidx);
                    end
                end
                ef{p} = double(acc);
            end
            return
        end
        ef = cell(2,2);                                 % ef{p,q}
        for q = 1:2
            for p = 1:2
                acc = 0;
                for c = 1:2
                    S = Sin{c,q};
                    if isequal(S, 0), continue; end      % identity: c~=q absent
                    E = run1_(S, Jm{p,c});
                    acc = acc + E(dzidx);
                end
                ef{p,q} = double(acc);
            end
        end
    end

    % ---- dark zones + normalization ------------------------------------
    c0px = ch.center_px;
    [ii, jj] = ndgrid(1:N, 1:N);
    rl = hypot(ii - c0px, jj - c0px) / ch.lamD_px;
    dz = cell(1, nlam);  pkband = 0;
    for l = 1:nlam
        lf = lfracs(l);
        dz{l} = find(rl >= 3/lf & rl <= 15/lf);
        macos.set_src_wvl(lam0 * lf);
        % unpolarized bare peak: no FPM, no Lyot -- component chains
        pk_l = 0;
        if strcmp(opts.input, 'circular')
            win = [1, -1i] / sqrt(2);
            Eacc = {0, 0};
            for c = 1:2
                macos.intensity(e.Apodizer);
                if opts.pol
                    Sc = (Sin{c,1} * win(1) + Sin{c,2} * win(2));
                    macos.apodize_complex(e.Apodizer, Sc);
                    Eb = macos.complex_field(e.FPA, 'reset_trace', false);
                    Eacc{c} = double(Eb);
                else
                    Eb = macos.complex_field(e.FPA, 'reset_trace', false);
                    Eacc{c} = win(c) * double(Eb);
                end
            end
            pk_l = abs(Eacc{1}).^2 + abs(Eacc{2}).^2;
        else
            for q = 1:2
                for c = 1:2
                    S = Sin{c,q};
                    if isequal(S, 0), continue; end
                    macos.intensity(e.Apodizer);
                    if ~isempty(S), macos.apodize_complex(e.Apodizer, S); end
                    Eb = macos.complex_field(e.FPA, 'reset_trace', false);
                    pk_l = pk_l + 0.5 * abs(Eb).^2;
                end
            end
        end
        pkband = pkband + max(pk_l(:));
    end
    pkband = pkband / nlam;

    function [efs, C] = measure_()
        efs = cell(1, nlam);  C = 0;
        for l = 1:nlam
            macos.set_src_wvl(lam0 * lfracs(l));
            efs{l} = fields_(dz{l}, lfracs(l));
            if strcmp(opts.input, 'circular')
                if strcmp(opts.analyzer, 'circular')
                    eL = (efs{l}{1} - 1i*efs{l}{2}) / sqrt(2);
                    C = C + mean(abs(eL).^2);
                else
                    C = C + mean(abs(efs{l}{1}).^2) + mean(abs(efs{l}{2}).^2);
                end
            elseif strcmp(opts.analyzer, 'circular')
                % L-analyzed output selects the SINGLE (+m) spiral for
                % both input states; unpolarized analyzed intensity,
                % normalized by the analyzed bare peak (pkband/2), so
                % the contrast is directly comparable within-channel
                epx = (efs{l}{1,1} + 1i*efs{l}{2,1}) / sqrt(2);
                epy = (efs{l}{1,2} + 1i*efs{l}{2,2}) / sqrt(2);
                C = C + 0.5 * (mean(abs(epx).^2) + mean(abs(epy).^2));
            else
                for q = 1:2
                    for p = 1:2
                        C = C + 0.5 * mean(abs(efs{l}{p,q}).^2);
                    end
                end
            end
        end
        if strcmp(opts.input, 'circular')
            C = C / nlam / pkband;      % vs the star's direct (bare) peak
        elseif strcmp(opts.analyzer, 'circular')
            C = C / nlam / (pkband/2);
        else
            C = C / nlam / pkband;
        end
    end

    % ---- the two Jacobians (band center, Vc and Vs masks, no screens) --
    tg = '';  if ~isempty(opts.tag), tg = ['_' opts.tag]; end
    jp = fullfile(opts.outdir, sprintf('ctb_dm_jacobian_N%d_vvc_c%d.mat', N, opts.charge));
    dzc = find(rl >= 3 & rl <= 15);
    if isfile(jp)
        JJ = load(jp);
    else
        macos.set_src_wvl(lam0);
        masks2 = {Vc, Vs};
        ncols = sum(cellfun(@(d) d.nact_active, dm));
        G = complex(zeros(2*numel(dzc), ncols, 'single'));
        col_dm = zeros(1, ncols);  col_act = zeros(1, ncols);
        e0 = cell(1,2);
        for mI = 1:2
            E = run1_([], masks2{mI});
            e0{mI} = double(E(dzc));
        end
        t0 = tic;  cc = 0;
        for k = 1:2
            acts = find(dm{k}.active).';
            for j = acts
                a = zeros(opts.nact^2, 1);  a(j) = opts.h_mm;
                dm{k}.apply(a);
                cc = cc + 1;
                for mI = 1:2
                    E = run1_([], masks2{mI});
                    rows = (mI-1)*numel(dzc) + (1:numel(dzc));
                    G(rows, cc) = single((double(E(dzc)) - e0{mI}) / opts.h_mm);
                end
                col_dm(cc) = k;  col_act(cc) = j;
                if opts.verbose && mod(cc, 100) == 0
                    el = toc(t0);
                    fprintf('[vvc jac] %d/%d pokes  %.1f min (ETA %.1f min)\n', ...
                        cc, ncols, el/60, el/cc*(ncols-cc)/60);
                end
            end
            dm{k}.clear();
        end
        JJ = struct('G',G, 'col_dm',col_dm, 'col_act',col_act, ...
            'npix',numel(dzc), 'charge',opts.charge, 'h_mm',opts.h_mm, ...
            'timing_s',toc(t0));
        save(jp, '-struct', 'JJ', '-v7.3');
        fprintf('[vvc] Jacobian saved: %s (%.1f min)\n', jp, JJ.timing_s/60);
    end
    np = JJ.npix;
    if opts.jac_perlam
        assert(strcmp(opts.input, 'circular'), ...
            'ctb_vvc: jac_perlam is implemented for circular input');
        jpl = fullfile(opts.outdir, sprintf( ...
            'ctb_dm_jacobian_N%d_vvc_c%d_perlam_b%02d.mat', N, ...
            opts.charge, round(100*opts.band)));
        if isfile(jpl)
            JL = load(jpl);
        else
            masks2 = {Vc, Vs};
            ncols = sum(cellfun(@(d) d.nact_active, dm));
            GL = complex(zeros(numel(dzc)*nlam, ncols, 'single'));
            e0L = cell(nlam, 2);
            for l = 1:nlam
                macos.set_src_wvl(lam0 * lfracs(l));
                for mI = 1:2
                    E = run1_([], masks2{mI});
                    e0L{l, mI} = double(E(dzc));
                end
            end
            t0 = tic;  cc = 0;
            col_dm = zeros(1, ncols);  col_act = zeros(1, ncols);
            for k = 1:2
                acts = find(dm{k}.active).';
                for j = acts
                    a = zeros(opts.nact^2, 1);  a(j) = opts.h_mm;
                    dm{k}.apply(a);
                    cc = cc + 1;
                    for l = 1:nlam
                        macos.set_src_wvl(lam0 * lfracs(l));
                        gc = 0;  gs = 0;
                        for mI = 1:2
                            E = run1_([], masks2{mI});
                            g = (double(E(dzc)) - e0L{l, mI}) / opts.h_mm;
                            if mI == 1, gc = g; else, gs = g; end
                        end
                        rows = (l-1)*numel(dzc) + (1:numel(dzc));
                        GL(rows, cc) = single(gc - 1i*gs);   % G_L(lambda)
                    end
                    col_dm(cc) = k;  col_act(cc) = j;
                    if opts.verbose && mod(cc, 100) == 0
                        el = toc(t0);
                        fprintf('[vvc jacL] %d/%d pokes  %.1f min (ETA %.1f min)\n', ...
                            cc, ncols, el/60, el/cc*(ncols-cc)/60);
                    end
                end
                dm{k}.clear();
            end
            JL = struct('G',GL, 'col_dm',col_dm, 'col_act',col_act, ...
                'npix',numel(dzc), 'lfracs',lfracs, 'timing_s',toc(t0));
            save(jpl, '-struct', 'JL', '-v7.3');
            fprintf('[vvc] per-lambda Jacobian saved: %s (%.1f min)\n', ...
                jpl, JL.timing_s/60);
        end
        % fold the per-lambda vortex weight beta into the blocks
        Gb = double(JL.G);
        for l = 1:nlam
            [~, be] = ab_(lfracs(l));
            rows = (l-1)*np + (1:np);
            Gb(rows, :) = be * Gb(rows, :);
        end
        Gr = [real(Gb); imag(Gb)];
        JJ.col_dm = JL.col_dm;  JJ.col_act = JL.col_act;
    elseif strcmp(opts.input, 'circular')
        % R input: dE_L = be (G_cos - i G_sin) da -- one complex block
        Gc_ = double(JJ.G(1:np, :));  Gs_ = double(JJ.G(np+1:2*np, :));
        Gr = [real(Gc_ - 1i*Gs_); imag(Gc_ - 1i*Gs_)];
    elseif strcmp(opts.analyzer, 'circular')
        % analyzed channel: single complex block G+ = (G_cos + i G_sin)/sqrt(2)
        Gc_ = double(JJ.G(1:np, :));  Gs_ = double(JJ.G(np+1:2*np, :));
        Gr = [real((Gc_ + 1i*Gs_)/sqrt(2)); imag((Gc_ + 1i*Gs_)/sqrt(2))];
    else
        Gr = [real(double(JJ.G)); imag(double(JJ.G))];
    end
    [U, S, V] = svd(Gr, 'econ');  sv = diag(S);

    % NOTE the Jacobian dark zone is the band-center annulus; per-lambda
    % dz sets differ by <5% in radius -- the reduced fields are formed on
    % the CENTER annulus for the solve, measured contrast on per-lambda
    % annuli (the honest score).
    dzsolve = dzc;

    % ---- EFC loop -------------------------------------------------------
    a = cellfun(@(d) zeros(opts.nact^2,1), dm, 'UniformOutput', false);
    [efs, C0] = measure_();
    contrast = zeros(1, opts.niter+1);  contrast(1) = C0;
    lk = 0;
    for l = 1:nlam, [al,~] = ab_(lfracs(l)); lk = lk + abs(al)^2; end
    lk = lk / nlam;
    fprintf('[vvc] tier=%s band=%g%% pol=%d: iter 0 contrast %.3e (leak frac %.2e)\n', ...
        opts.tier, 100*opts.band, opts.pol, C0, lk);

    function em = reduced_()
        if opts.jac_perlam
            % stacked per-lambda targets: e_L(lambda), no collapse (the
            % lambda-differential field is what per-lambda control nulls)
            em = zeros(np * nlam, 1);
            for l = 1:nlam
                macos.set_src_wvl(lam0 * lfracs(l));
                ef = fields_(dzsolve, lfracs(l));
                em((l-1)*np + (1:np)) = (ef{1} - 1i*ef{2}) / sqrt(2);
            end
            return
        end
        if strcmp(opts.input, 'circular')
            % control the L-projection (leakage-free channel for R input)
            em = 0;  W2 = 0;
            for l = 1:nlam
                macos.set_src_wvl(lam0 * lfracs(l));
                ef = fields_(dzsolve, lfracs(l));
                [~, be] = ab_(lfracs(l));
                em = em + conj(be) * (ef{1} - 1i*ef{2}) / sqrt(2);
                W2 = W2 + abs(be)^2;
            end
            em = em / W2;
            return
        end
        if strcmp(opts.analyzer, 'circular')
            % control the analyzed (+) channel of the x input, measured
            % on the band-center annulus, weight-summed over wavelengths
            em = 0;  W2 = 0;
            for l = 1:nlam
                macos.set_src_wvl(lam0 * lfracs(l));
                ef = fields_(dzsolve, lfracs(l));
                [~, be] = ab_(lfracs(l));
                em = em + conj(be) * (ef{1,1} + 1i*ef{2,1}) / sqrt(2);
                W2 = W2 + abs(be)^2;
            end
            em = em / W2;
            return
        end
        % collapse the 4 component fields onto the [G_cos; G_sin] blocks,
        % weight-summed over wavelengths (fields re-read on the solve
        % annulus via fields_ already using per-lambda dz -- here we
        % re-measure on the CENTER annulus for consistency)
        ec = 0;  es = 0;  W = 0;
        for l = 1:nlam
            macos.set_src_wvl(lam0 * lfracs(l));
            ef = fields_(dzsolve, lfracs(l));
            [~, be] = ab_(lfracs(l));
            w = 2 * abs(be)^2;
            ec = ec + conj(be) * (ef{1,1} - ef{2,2});
            es = es + conj(be) * (ef{2,1} + ef{1,2});
            W = W + w;
        end
        em = [ec; es] / W;
    end

    for it = 1:opts.niter
        % reduced_ measures from the ENGINE: restore the accepted state
        % first (the line-search trials leave the last trial's commands
        % applied -- solving about that state corrupts every iteration
        % after the first; the one-step-then-stall signature)
        for k = 1:2, dm{k}.apply(a{k}); end
        em = reduced_();
        Ue = U' * [real(em); imag(em)];
        best = struct('c', inf);
        for al2 = opts.alphas
            da = -V * ((sv ./ (sv.^2 + al2 * sv(1)^2)) .* Ue);
            at = a;
            for k = 1:2
                sel = JJ.col_dm == k;
                at{k}(JJ.col_act(sel)) = at{k}(JJ.col_act(sel)) + da(sel);
                dm{k}.apply(at{k});
            end
            [eft, Ct] = measure_();
            if Ct < best.c, best = struct('c',Ct,'a',{at},'ef',{eft},'alpha',al2); end
        end
        if best.c >= contrast(it)
            fprintf('[vvc] iter %d: no alpha improves (best %.3e) -- stop\n', it, best.c);
            contrast = contrast(1:it);
            for k = 1:2, dm{k}.apply(a{k}); end
            break;
        end
        a = best.a;  efs = best.ef;
        contrast(it+1) = best.c;
        str = cellfun(@(x) 1e6*rms(x(x~=0)), a);
        fprintf('[vvc] iter %d: contrast %.3e (alpha %.0e, stroke rms [%s] nm)\n', ...
            it, best.c, best.alpha, num2str(str, '%.2f '));
    end
    for k = 1:2, dm{k}.apply(a{k}); end
    [~, c_final] = measure_();
    fprintf('[vvc] final: %.3e -> %.3e (%.1fx)\n', C0, c_final, C0/max(c_final,realmin));

    out = struct('tier',opts.tier, 'pol',opts.pol, 'band',opts.band, ...
        'lfracs',lfracs, 'charge',opts.charge, 'contrast',contrast, ...
        'c_before',C0, 'c_after',c_final, 'leak_frac',lk, 'a',{a}, ...
        'stroke_rms_nm',cellfun(@(x) 1e6*rms(x(x~=0)), a));
    if opts.save
        save(fullfile(opts.outdir, ['ctb_vvc' tg '.mat']), '-struct', 'out', '-v7.3');
    end
end
