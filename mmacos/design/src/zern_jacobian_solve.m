function out = zern_jacobian_solve(t, elts, opts)
%ZERN_JACOBIAN_SOLVE  Linear least-squares Zernike figure solve (SVD).
%   out = zern_jacobian_solve(t, ELTS, 'modes',M, 'fields',F, ...) solves
%   the multi-field figure-correction problem for the Zernike departures
%   on mirrors ELTS of the built telescope T by DIRECT LINEAR ALGEBRA
%   instead of CALIB's finite-difference Levenberg-Marquardt:
%
%     1. poke each (mirror, mode) once, trace the field set, and build
%        the OPD Jacobian J = dW/dc (dW/dZern is nearly linear -- these
%        are figure departures, not geometry);
%     2. project per-field PISTON + TIP/TILT out of both the residual
%        and the Jacobian columns (tilt is DISTORTION/pointing, not
%        blur -- and a mirror-tilt Zernike is a pure gauge for a
%        chief-referenced merit: projecting it out makes the gauge
%        directions vanish instead of wandering);
%     3. solve min ||J*dc + r|| by TRUNCATED SVD -- the singular-value
%        spectrum is printed, so degeneracy (canceling-coefficient null
%        spaces, over-tight/over-wide lMon) is VISIBLE, and the
%        minimum-norm solution excludes the pathological canceling
%        combinations by construction;
%     4. apply, re-emit, re-evaluate; iterate (default 2 passes -- the
%        residual nonlinearity is small).
%
%   One Jacobian costs (1 + nElts*nModes) traces per field -- roughly
%   what ONE CALIB iteration pays in finite differences; the whole solve
%   is typically ~100x cheaper than a 200-iteration CALIB run.
%
%   T must be built (t.build()).  Coefficients are seeded from / written
%   back to t.spec.elt(k).freeform (same contract as optimize_freeform),
%   so the two engines compose: jacobian first, CALIB to polish.
%
%   Name-value:
%     'modes'    Zernike mode list (default [3 4 5 9:13 19:25], e5mono)
%     'type'     Zernike type string (default 'BornWolf')
%     'lmon'     normalization radius: scalar or one per ELTS entry.
%                FIELD-ZONE radius (footprint + field walk), fixed for
%                the life of the coefficients (see optimize_freeform).
%     'fields'   Kx2 [thx thy] rad field OFFSETS; [0 0] rows allowed
%                (default [0 0]).  The center is NOT implicit.
%     'weights'  1xK per-field weights (default ones)
%     'iters'    outer linearize-solve-apply passes (default 2)
%     'svd_rel'  relative singular-value cutoff (default 1e-4)
%     'dstep'    poke size in metres of surface coefficient (default
%                LAM/4 equivalent: 1.25e-7)
%     'quiet'    suppress prints (default false)
%
%   Returns struct:
%     .wfe       (iters+1)xK tilt-removed RMS WFE per field (m):
%                row 1 = before, row i+1 = after pass i
%     .sv        singular values of the last pass (scaled)
%     .rank      retained rank of the last pass
%     .coef      final coefficient state, cell per ELTS entry
%     .modes .elts .fields
%
%   See also: macos.design.Telescope/optimize_freeform, wfe_field_diag.
    arguments
        t
        elts (1,:) double {mustBeInteger, mustBePositive}
        opts.modes (1,:) double = [3 4 5 9 10 11 12 13 19 20 21 22 23 24 25]
        opts.type (1,:) char = 'BornWolf'
        opts.lmon (1,:) double = NaN
        opts.fields (:,2) double = [0 0]
        opts.weights (1,:) double = []
        opts.iters (1,1) double {mustBeInteger, mustBePositive} = 2
        opts.svd_rel (1,1) double = 1e-4
        opts.lambda_rel (1,1) double = 3e-3   % Tikhonov damping (rel sv1)
        opts.alphas (1,:) double = [1 0.5 0.25 0.1]  % line-search steps
        opts.dstep (1,1) double = 1.25e-7
        opts.quiet (1,1) logical = false
    end
    modes = opts.modes(:).';  nm = numel(modes);
    lmon = opts.lmon;
    if isscalar(lmon), lmon = repmat(lmon, 1, numel(elts)); end
    assert(numel(lmon) == numel(elts), ...
        'zern_jacobian_solve: lmon must be scalar or one per ELTS entry');
    F = opts.fields;  K = size(F,1);
    w = opts.weights;  if isempty(w), w = ones(1,K); end
    assert(numel(w) == K, 'weights must match fields');
    ne = numel(elts);  nc = ne*nm;

    % declare/keep the Zernike surfaces (seeds from existing freeform)
    for j = 1:ne
        k = elts(j);
        seed = zeros(1, nm);
        ff = t.spec.elt(k).freeform;
        if isstruct(ff) && ~isempty(ff) && isfield(ff,'modes')
            for i = 1:nm
                kk = find(ff.modes == modes(i), 1);
                if ~isempty(kk), seed(i) = ff.coef(kk); end
            end
            if isnan(lmon(j)) && isfield(ff,'lmon') && ~isnan(ff.lmon)
                lmon(j) = ff.lmon;      % lMon continuity (see optimize_freeform)
            end
        end
        t.set_freeform(k, modes, seed, 'type', opts.type, 'lmon', lmon(j));
    end
    t.build('','init',false);

    wfe = zeros(opts.iters+1, K);
    sv = [];  rankk = 0;
    for pass = 1:opts.iters
        % ---- assemble residual + Jacobian over the field set ----------
        rows_r = cell(K,1);  rows_J = cell(K,1);
        for f = 1:K
            if any(abs(F(f,:)) > 1e-15), t.trace_at_field(F(f,:));
            else, t.trace_at_field([]); end
            [W0, P, ii] = opd_() ;                 % base OPD + projector
            if pass == 1, wfe(1,f) = std(P*W0); end
            Jf = zeros(numel(W0), nc);
            c = 0;
            for j = 1:ne
                k = elts(j);
                base = coef_(t, k, modes);
                for m = 1:nm
                    c = c + 1;
                    poke_(k, modes(m), base(m) + opts.dstep);
                    Wp = opd_at_(ii);
                    Jf(:,c) = (Wp - W0) / opts.dstep;
                    poke_(k, modes(m), base(m));
                end
            end
            rows_r{f} = sqrt(w(f)) * (P*W0);
            rows_J{f} = sqrt(w(f)) * (P*Jf);
        end
        t.trace_at_field([]);
        r = vertcat(rows_r{:});  J = vertcat(rows_J{:});

        % ---- truncated-SVD minimum-norm solve --------------------------
        [U,S,V] = svd(J, 'econ');
        sv = diag(S);
        if sv(1) <= 0
            error(['zern_jacobian_solve: zero Jacobian -- the pokes did ' ...
                   'not reach the trace (MODIFY missing?)']);
        end
        keep = sv >= opts.svd_rel * sv(1);
        rankk = nnz(keep);
        % Tikhonov-damped minimum-norm step: hard-truncate the deep null
        % space (svd_rel), damp the small-sv directions (they otherwise
        % take huge strokes into the nonlinear regime -- the raw Newton
        % step OVERSHOOTS at tens-of-waves amplitude).
        lam = opts.lambda_rel * sv(1);
        g = sv(keep) ./ (sv(keep).^2 + lam^2);
        dc = -V(:,keep) * (g .* (U(:,keep).'*r));
        if ~opts.quiet
            fprintf(['[jacobian] pass %d: %d rows x %d dofs, rank %d/%d ' ...
                     '(sv %.2e..%.2e, lambda %.1e)\n'], pass, numel(r), nc, ...
                    rankk, nc, sv(1), sv(end), lam);
        end

        % ---- damped step with a short line search -----------------------
        base_all = cell(1,ne);
        for j = 1:ne, base_all{j} = coef_(t, elts(j), modes); end
        alphas = opts.alphas;
        best = struct('alpha',0, 'worst',max(wfe(pass,:)), 'wfe',wfe(pass,:));
        for a = alphas
            c = 0;
            for j = 1:ne
                k = elts(j);
                newc = base_all{j} + a*dc(c+(1:nm)).';  c = c + nm;
                t.set_freeform(k, modes, newc, 'type', opts.type, 'lmon', lmon(j));
            end
            t.build('','init',false);
            wa = zeros(1,K);
            for f = 1:K
                if any(abs(F(f,:)) > 1e-15), t.trace_at_field(F(f,:));
                else, t.trace_at_field([]); end
                [W0, P] = opd_();
                wa(f) = std(P*W0);
            end
            t.trace_at_field([]);
            if max(wa) < best.worst
                best = struct('alpha',a, 'worst',max(wa), 'wfe',wa);
            end
        end
        % leave the best state applied (alpha=0 -> restore the base)
        c = 0;
        for j = 1:ne
            k = elts(j);
            newc = base_all{j} + best.alpha*dc(c+(1:nm)).';  c = c + nm;
            t.set_freeform(k, modes, newc, 'type', opts.type, 'lmon', lmon(j));
        end
        t.build('','init',false);
        wfe(pass+1,:) = best.wfe;
        if ~opts.quiet
            fprintf(['[jacobian] pass %d WFE (waves @ %g nm): worst %.4f -> ' ...
                     '%.4f (alpha %.2f)\n'], pass, t.spec.wavelength*1e9, ...
                    max(wfe(pass,:))/t.spec.wavelength, ...
                    best.worst/t.spec.wavelength, best.alpha);
        end
        if best.alpha == 0, break; end     % no improving step -- stop
    end

    out = struct('wfe',wfe, 'sv',sv, 'rank',rankk, 'modes',modes, ...
                 'elts',elts, 'fields',F, 'lmon',lmon);
    out.coef = arrayfun(@(k) t.spec.elt(k).freeform.coef, elts, 'uni', 0);

    % ---- helpers --------------------------------------------------------
    function [W, P, ii] = opd_()
        nE = numel(t.spec.elt);
        s = macos.trace(nE);
        Wm = macos.opd();
        v = Wm(:);
        ii = find(isfinite(v) & v ~= 0 & abs(v) < 1e30);
        W = v(ii);
        % piston + tip/tilt projector over the lit samples (pupil coords
        % from the OPD grid indices -- adequate for projection)
        [ry, rx] = ind2sub(size(Wm), ii);
        A = [ones(numel(ii),1), rx - mean(rx), ry - mean(ry)];
        P = eye(numel(ii)) - A*((A.'*A)\A.');
        %#ok<NASGU>
        if s.nRays == 0, error('zern_jacobian_solve: no rays'); end
    end
    function W = opd_at_(ii)
        nE = numel(t.spec.elt);
        macos.trace(nE);
        Wm = macos.opd();  v = Wm(:);
        W = v(ii);
        W(~isfinite(W) | abs(W) > 1e30) = 0;
    end
    function c = coef_(t_, k, modes_)
        ff = t_.spec.elt(k).freeform;
        c = zeros(1, numel(modes_));
        for i2 = 1:numel(modes_)
            kk2 = find(ff.modes == modes_(i2), 1);
            if ~isempty(kk2), c(i2) = ff.coef(kk2); end
        end
    end
    function poke_(k, mode, val)
        macos.set_elt_zrn_coef(k, mode, val);   % absolute write, in-place
        macos.modify();     % invalidate trace caches -- without MODIFY the
                            % next trace reuses the cached MonCoef and the
                            % poke silently reads back as zero (the same
                            % trap the dw_dz ZernikeCoefChannel documents)
    end
end
