function out = ctb_efc(opts)
%CTB_EFC  Electric-field conjugation dark hole on the CTB, engine in the loop.
%   out = CTB_EFC() digs a dark hole with the two CTB DMs: the classic
%   EFC iteration (Tikhonov-regularized least squares on the complex
%   dark-zone field) driven by the ENGINE-measured Jacobian
%   (ctb_dm_jacobian) and closed on the ENGINE-measured field -- every
%   iterate re-propagates the full masked diffraction chain, so model
%   error cannot be mined (the e2e6m S3b lesson).  Sensing is assumed
%   perfect: the complex FPA field is read directly from the engine;
%   pairwise probing is the lab-facing extension, not built here.
%
%   Iteration:  da = -argmin ||G da + e||^2 + alpha ||da||^2
%   solved via the SVD of G; the regularization alpha is LINE-SEARCHED
%   each iteration against the MEASURED contrast (engine evaluations
%   are ~0.25 s, so trying a few alphas and keeping the best measured
%   result is affordable -- a luxury lab EFC does with probing).
%
%   Name-value:
%     'jac'         Jacobian .mat path or ctb_dm_jacobian output struct
%                   (default ctb_dm_jacobian_N512.mat beside this file;
%                   REGENERATED via ctb_dm_jacobian() if absent)
%     'niter'       EFC iterations (10)
%     'alphas'      relative regularization grid, fractions of s_max^2
%                   (default logspace(-6,-2,5); searched every iteration)
%     'stroke_warn_nm'  warn if any actuator exceeds this (50)
%     'outdir'      figure/result dir (this dir)
%     'save'        write ctb_efc.mat + ctb_efc.png (true)
%     'visible'     show the figure (false)
%
%   out: contrast (1 x niter+1, dark-zone mean, Strehl-normalized),
%   a (per-DM command vectors, mm), E_before/E_after (FPA fields),
%   alpha_used, stroke_rms_nm, jac meta passthrough.
%
%   Run:  >> out = ctb_efc;
%   See also: ctb_dm_jacobian, ctb_chain, ctb_dm, ctb_contrast.
    arguments
        opts.jac             = ''
        opts.niter          (1,1) double {mustBeInteger,mustBePositive} = 10
        opts.alphas         (1,:) double = logspace(-6, -2, 5)
        opts.stroke_warn_nm (1,1) double = 50
        opts.outdir         (1,:) char = ''
        opts.save           (1,1) logical = true
        opts.visible        (1,1) logical = false
        opts.replot         (1,:) char = ''     % saved ctb_efc.mat: figure only
        opts.compare        struct = struct([]) % .contrast + .label overlay
        opts.tag            (1,:) char = ''     % suffix for saved .mat/.png
        opts.a0             (1,:) cell = {}     % warm-start commands (relin)
        opts.quant_nm       (1,1) double {mustBeNonnegative} = 0  % Lane 2e:
                            % DM LSB noise, nm.  0 = off.  PLANT ONLY -- the
                            % controller solves in continuous strokes; the
                            % engine is commanded with fresh U(+/-LSB/2)
                            % dither per actuator per evaluation (DM
                            % electronics noise, Ruane 2020).  The solved
                            % state stays continuous; only ch.run() sees the
                            % dithered command.
        opts.quant_seed     (1,1) double = 1      % rng seed for the dither
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', '..', 'src'));
    if isempty(opts.outdir), opts.outdir = here; end

    % ---- Jacobian -------------------------------------------------------
    if isstruct(opts.jac)
        J = opts.jac;
    else
        jp = opts.jac;
        if isempty(jp), jp = fullfile(here, 'ctb_dm_jacobian_N512.mat'); end
        if ~isfile(jp)
            fprintf('[efc] %s absent -- measuring the Jacobian first\n', jp);
            J = ctb_dm_jacobian();
        else
            J = load(jp);
        end
    end
    % ---- replot-only mode: rebuild the figure from a saved run ----------
    % (DM geometry from the Jacobian; no engine session, no loop)
    if ~isempty(opts.replot)
        out = load(opts.replot);
        dmr = cell(1, numel(J.dm));
        for k = 1:numel(J.dm)
            d = J.dm(k);
            dmr{k} = ctb_dm('ielt', d.ielt, 'ng', d.ng, 'gdx_mm', d.gdx_mm, ...
                'nact', d.nact, 'pitch_mm', d.pitch_mm, ...
                'beam_d_mm', d.beam_d_mm, 'coupling', d.coupling);
        end
        fig_(out, dmr, opts);
        return
    end

    G = double(J.G);
    % REAL-stacked system: DM commands are real, so the complex least
    % squares must be solved as [Re G; Im G] da = -[Re e; Im e].  (A
    % complex SVD solve returns complex da whose imaginary part the mex
    % layer silently drops -- the achieved field then diverges from the
    % prediction and the line search collapses to tiny steps.)
    Gr = [real(G); imag(G)];
    [U, S, V] = svd(Gr, 'econ');
    s = diag(S);
    fprintf('[efc] G: %d px x %d acts (real-stacked), s_max=%.3e cond=%.2e\n', ...
        size(G,1), size(G,2), s(1), s(1)/s(end));

    % ---- chain + DM models (must match the Jacobian's config) ----------
    cargs = {};
    if isfield(J, 'chain_opts'), cargs = J.chain_opts; end
    ch = ctb_chain('rx', J.rx, 'model_size', J.N, cargs{:});
    ndm = numel(J.dm);
    dm = cell(1, ndm);
    for k = 1:ndm
        d = J.dm(k);
        dm{k} = ctb_dm('ielt', d.ielt, 'ng', d.ng, 'gdx_mm', d.gdx_mm, ...
                       'nact', d.nact, 'pitch_mm', d.pitch_mm, ...
                       'beam_d_mm', d.beam_d_mm, 'coupling', d.coupling);
        dm{k}.clear();
    end
    % column -> (dm slot, actuator) map: J.col_dm holds the DM index as
    % passed to the Jacobian's 'dms' option, in J.dm order
    udm = unique(J.col_dm, 'stable');
    dmof = arrayfun(@(c) find(udm == c, 1), J.col_dm);

    dz = J.dz_idx;
    pb = J.peak_bare;

    % ---- EFC loop -------------------------------------------------------
    if ~isempty(opts.a0)
        a = opts.a0;                       % warm start (relinearized round)
    elseif isfield(J, 'a0') && ~isempty(J.a0)
        a = J.a0;                          % the Jacobian's linearization point
    else
        a = cellfun(@(d) zeros(d.nact^2, 1), dm, 'UniformOutput', false);
    end
    if opts.quant_nm > 0
        rng(opts.quant_seed);
        fprintf('[efc] PLANT LSB dither %.2f nm (seed %d) -- controller stays continuous\n', ...
                opts.quant_nm, opts.quant_seed);
    end
    papply_(dm, a, opts.quant_nm);
    E = ch.run();
    E_before = E;
    contrast = zeros(1, opts.niter + 1);
    contrast(1) = mean(abs(E(dz)).^2) / pb;
    alpha_used = zeros(1, opts.niter);
    fprintf('[efc] iter 0: contrast %.3e\n', contrast(1));

    for it = 1:opts.niter
        e = double(E(dz));
        Ue = U' * [real(e); imag(e)];
        best = struct('c', inf, 'da', [], 'alpha', NaN, 'E', []);
        for al = opts.alphas
            alpha = al * s(1)^2;
            da = -V * ((s ./ (s.^2 + alpha)) .* Ue);
            % apply trial
            at = a;
            for k = 1:ndm
                sel = dmof == k;
                at{k}(J.col_act(sel)) = at{k}(J.col_act(sel)) + da(sel);
            end
            papply_(dm, at, opts.quant_nm);
            Et = ch.run();
            c = mean(abs(Et(dz)).^2) / pb;
            if c < best.c
                best = struct('c', c, 'da', da, 'alpha', al, 'E', Et, 'a', {at});
            end
        end
        if best.c >= contrast(it)
            fprintf('[efc] iter %d: no alpha improves (best %.3e) -- stop\n', ...
                it, best.c);
            contrast = contrast(1:it);
            alpha_used = alpha_used(1:it-1);
            % restore best-so-far state
            papply_(dm, a, opts.quant_nm);
            E = ch.run();
            break;
        end
        a = best.a;
        E = best.E;
        contrast(it+1) = best.c;
        alpha_used(it) = best.alpha;
        str_nm = cellfun(@(x) 1e6 * rms(x(x~=0)), a);
        fprintf('[efc] iter %d: contrast %.3e (alpha %.1e, stroke rms [%s] nm)\n', ...
            it, best.c, best.alpha, num2str(str_nm, '%.2f '));
        if any(cellfun(@(x) 1e6*max(abs(x)), a) > opts.stroke_warn_nm)
            warning('ctb_efc:stroke', 'actuator stroke exceeds %g nm', ...
                opts.stroke_warn_nm);
        end
    end
    % ensure engine state = final commands (line search may have left a trial)
    papply_(dm, a, opts.quant_nm);
    E_after = ch.run();
    c_final = mean(abs(E_after(dz)).^2) / pb;
    fprintf('[efc] final: %.3e -> %.3e (%.1fx) in %d iterations\n', ...
        contrast(1), c_final, contrast(1)/c_final, numel(contrast)-1);

    % ---- outputs --------------------------------------------------------
    out = struct('contrast',contrast, 'alpha_used',alpha_used, ...
        'a',{a}, 'E_before',E_before, 'E_after',E_after, ...
        'c_before',contrast(1), 'c_after',c_final, ...
        'lamD_px',J.lamD_px, 'center_px',J.center_px, 'N',J.N, ...
        'inner_lamD',J.inner_lamD, 'outer_lamD',J.outer_lamD, ...
        'peak_bare',pb, 'dz_idx',dz, 'jac_h_mm',J.h_mm, ...
        'quant_nm',opts.quant_nm, 'quant_seed',opts.quant_seed);
    stroke_rms_nm = cellfun(@(x) 1e6 * rms(x(x~=0)), a);
    out.stroke_rms_nm = stroke_rms_nm;
    if opts.save
        tg = '';  if ~isempty(opts.tag), tg = ['_' opts.tag]; end
        save(fullfile(opts.outdir, ['ctb_efc' tg '.mat']), ...
             '-struct', 'out', '-v7.3');
        fig_(out, dm, opts);
    end
end

% ======================================================================
function papply_(dm, cmd, quant_nm)
%PAPPLY_  Apply a command to the PLANT, with optional LSB dither (Lane 2e).
%   quant_nm > 0 adds fresh U(+/-quant_nm/2) nm noise per actuator (DM
%   electronics LSB noise, Ruane 2020; commands are mm so nm*1e-6).  The
%   controller's solved state is unchanged -- only what the engine is
%   commanded here -- so this is a PLANT-only perturbation.
    for k = 1:numel(dm)
        c = cmd{k};
        if quant_nm > 0
            c = c + (rand(size(c)) - 0.5) * quant_nm * 1e-6;
        end
        dm{k}.apply(c);
    end
end

% ======================================================================
function fig_(out, dm, opts)
    vis = 'off'; if opts.visible, vis = 'on'; end
    fig = figure('Visible',vis, 'Color','w', 'Position',[60 60 1860 420]);
    tl = tiledlayout(fig, 1, 5, 'TileSpacing','compact', 'Padding','compact');
    title(tl, sprintf(['CTB EFC dark hole -- %d DM actuators, engine-measured ' ...
        'Jacobian, engine-closed loop'], sum(cellfun(@(d) d.nact_active, dm))), ...
        'FontWeight','bold');

    c = out.center_px;  w = ceil((out.outer_lamD + 3) * out.lamD_px);
    ix = c-w : c+w;
    lamD_ax = ((ix) - c) / out.lamD_px;
    % color floor adapts to the achieved depth (a fixed -10 saturates a
    % 1e-13-class dark hole into a blank panel)
    lo = max(-16, min(-10, floor(log10(max(out.c_after, 1e-16))) - 1));
    for p = 1:2
        nexttile(tl);
        if p == 1, E = out.E_before; t = sprintf('before  (%.2e)', out.c_before);
        else,      E = out.E_after;  t = sprintf('after  (%.2e)', out.c_after);
        end
        L = log10(max(abs(E(ix,ix)).^2 / out.peak_bare, 10^lo));
        imagesc(lamD_ax, lamD_ax, L.'); axis image xy;
        colormap(gca, parula); clim([lo -4]);
        hold on
        th = linspace(0, 2*pi, 200);
        for rr = [out.inner_lamD out.outer_lamD]
            plot(rr*cos(th), rr*sin(th), 'w--', 'LineWidth', 0.8);
        end
        title(t, 'Interpreter','none'); xlabel('\lambda/D'); ylabel('\lambda/D');
        cb = colorbar; cb.Label.String = 'log_{10} contrast';
    end

    nexttile(tl);
    semilogy(0:numel(out.contrast)-1, out.contrast, 'o-', 'LineWidth', 1.4, ...
        'DisplayName', 'DM1 + DM2');
    grid on; hold on;
    if ~isempty(opts.compare)
        semilogy(0:numel(opts.compare.contrast)-1, opts.compare.contrast, ...
            's--', 'LineWidth', 1.2, 'DisplayName', opts.compare.label);
        legend('Location', 'northeast');
    end
    xlabel('EFC iteration'); ylabel('dark-zone mean contrast');
    title(sprintf('%.2e \\rightarrow %.2e', out.c_before, out.c_after));

    for k = 1:numel(out.a)
        nexttile(tl);
        d = dm{k};
        S = reshape(1e6 * out.a{k}, d.nact, d.nact);
        imagesc(S.'); axis image xy; colormap(gca, parula);
        cb = colorbar; cb.Label.String = 'nm';
        title(sprintf('DM%d commands (rms %.2f nm)', k, out.stroke_rms_nm(k)));
    end

    tg = '';  if ~isempty(opts.tag), tg = ['_' opts.tag]; end
    fp = fullfile(opts.outdir, ['ctb_efc' tg '.png']);
    exportgraphics(fig, fp, 'Resolution', 150);
    close(fig);
    fprintf('[efc] figure %s\n', fp);
end
