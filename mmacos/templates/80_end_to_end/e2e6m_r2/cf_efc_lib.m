function h = cf_efc_lib()
%CF_EFC_LIB  Shared EFC machinery for the coronagraph-family campaign.
%   h = CF_EFC_LIB() returns handles used by cf2_efc / cf3_spacing /
%   later stages, so the Jacobian sweep and the loop exist ONCE:
%
%     h.jacobian(ch, dm, a0, dz_idx, P, cache)
%         Engine-measured G = dE(dark zone)/d(actuator) about the
%         commands a0, through ch.run() (the FULL masked chain).
%         Cached at `cache` with the chain_opts stamp (verified on load
%         by ctb_jac_check -- the file NAME is a hint, the stamp is the
%         authority), the linearization point (asserted on load), and a
%         committed .fp.json fingerprint.
%     h.efc(ch, dm, J, a_in, dz_idx, niter, alphas)
%         The ctb_efc idiom: REAL-STACKED solve ([Re G; Im G] da =
%         -[Re e; Im e] -- the mex silently drops Im(da), trap 1),
%         per-iteration Tikhonov line search against MEASURED contrast,
%         monotone accept, accepted state RE-APPLIED before returning
%         (trap 2: line-search trials leave the engine at the last
%         trial).  Returns [a, contrast, alpha_used].
%     h.seta(dm, a)          apply per-DM command cells
%     h.sym_frac(E, c, dz)   180-deg-symmetric energy fraction of the
%                            dark-zone field (amplitude-type speckle is
%                            even about the star; phase-type odd)
%
%   See also CF2_EFC, CF3_SPACING, ctb_efc, ctb_dm_jacobian,
%   ctb_jac_check, jac_fingerprint.
    h.jacobian     = @jacobian_;
    h.efc          = @efc_;
    h.seta         = @seta_;
    h.sym_frac     = @sym_frac_;
    h.stamp_parity = @stamp_parity_;
    h.linfloor     = @linfloor_;
end

% =========================================================================
function stamp_parity_(J, config, src)
%STAMP_PARITY_  The campaign's STRICT complement to ctb_jac_check: every
%   key of the REQUESTED config must exist in the cached stamp.  Our own
%   caches always carry the full config, so a missing key means the cache
%   predates that config field (a stale GENERATION -- e.g. the S0b
%   circ_stop_frac amendment: a no-stop cache's stamp simply lacks the
%   key, and ctb_jac_check's compare-what-both-have contract PASSES it).
%   ctb_jac_check itself stays untouched (CTB's legacy caches rely on the
%   partial-compare contract).
    a = nv2struct_(J.chain_opts);
    b = nv2struct_(config);
    kb = fieldnames(b);
    miss = kb(~isfield(a, kb));
    if ~isempty(miss)
        error('cf_efc_lib:stale_generation', ...
            ['%s predates the config field(s) %s -- a STALE GENERATION ' ...
             '(measured before that knob existed).  Delete it or use a ' ...
             'distinct tag.'], src, strjoin(miss, ', '));
    end
end

function s = nv2struct_(c)
    if isstruct(c), s = c; return; end
    s = struct();
    for i = 1:2:numel(c)-1, s.(c{i}) = c{i+1}; end
end

% =========================================================================
function la = linfloor_(J, stroke_bound_nm)
%LINFLOOR_  Linear-achievable floor of a measured Jacobian: the residual
%   of the top-N-mode real-stacked least squares on the STORED static
%   field, as a function of rank, with the stroke bound applied -- the
%   bench_ctb attribution pattern ("linear-achievable floor 4.5e-9 at
%   11 nm rms").  Closed form per rank from the SVD:
%     ||res_r||^2 = ||e0||^2 - sum_{i<=r} (u_i' e0)^2,
%     ||a_r||^2   = sum_{i<=r} ((u_i' e0)/s_i)^2,
%   so the whole rank curve costs one SVD.  The floor is the contrast at
%   the largest rank whose stroke rms is within the bound.
    if nargin < 2, stroke_bound_nm = 50; end
    G = double(J.G);
    Gr = [real(G); imag(G)];
    e0 = double(J.E0_dz(:));
    e0r = [real(e0); imag(e0)];
    [U, S, ~] = svd(Gr, 'econ');
    s = diag(S);
    Ue = U' * e0r;
    npix = numel(e0);
    pk = J.peak_bare;
    res2 = max(sum(e0r.^2) - cumsum(Ue.^2), 0);
    con = res2 / npix / pk;
    a2 = cumsum((Ue ./ s).^2);
    stroke_nm = 1e9 * sqrt(a2 / size(Gr, 2));
    ok = stroke_nm <= stroke_bound_nm;
    if ~any(ok)
        rbest = 1;
    else
        rbest = find(ok, 1, 'last');
    end
    la = struct('c_static', sum(e0r.^2)/npix/pk, ...
                'floor', con(rbest), 'rank', rbest, ...
                'stroke_nm', stroke_nm(rbest), ...
                'bound_nm', stroke_bound_nm, ...
                'curve_con', con(:).', 'curve_stroke_nm', stroke_nm(:).');
end

% =========================================================================
function [J, meta] = jacobian_(ch, dm, a0, dz_idx, P, cache)
    if isfile(cache)
        J = load(cache);
        ctb_jac_check(J, ch.config, cache);
        stamp_parity_(J, ch.config, cache);
        a0v = cell2mat(cellfun(@(x) x(:), a0,  'UniformOutput', false));
        acv = cell2mat(cellfun(@(x) x(:), J.a0, 'UniformOutput', false));
        assert(max(abs(a0v - acv)) < 1e-15, ...
            'cf_efc_lib: %s was measured about a DIFFERENT DM state -- delete or retag', cache);
        fprintf('    [jac] loaded %s (%d cols)\n', cache, size(J.G,2));
        meta = struct('file', cache, 'ncol', size(J.G,2), 'cached', true);
        return
    end
    seta_(dm, a0);
    E0 = ch.run();
    E0_dz = single(E0(dz_idx));
    nacts = cellfun(@(d) d.nact_active, dm);
    ncol = sum(nacts);
    G = complex(zeros(numel(dz_idx), ncol, 'single'));
    col_dm = zeros(1, ncol);  col_act = zeros(1, ncol);
    h = P.dj.h;  c = 0;  tswp = tic;
    for k = 1:numel(dm)
        act = find(dm{k}.active(:)).';
        for a = act
            c = c + 1;
            v = a0{k};  v(a) = v(a) + h;
            dm{k}.apply(v);
            E = ch.run();
            G(:,c) = (single(E(dz_idx)) - E0_dz) / h;
            col_dm(c) = k;  col_act(c) = a;
            if mod(c, 200) == 0
                fprintf('    [jac] %4d/%d cols, %.2f s/poke\n', c, ncol, toc(tswp)/c);
            end
        end
        dm{k}.apply(a0{k});
    end
    cn = sqrt(sum(abs(G).^2, 1));
    assert(nnz(cn == 0) == 0 && all(isfinite(cn)), ...
        'cf_efc_lib: null/inf Jacobian columns');
    J = struct('G', G, 'col_dm', col_dm, 'col_act', col_act, ...
        'dz_idx', dz_idx, 'E0_dz', E0_dz, 'h', h, 'a0', {a0}, ...
        'chain_opts', {ch.config}, 'N', ch.N, 'rx', ch.rx, ...
        'lamD_px', ch.lamD_px, 'peak_bare', ch.peak_bare, ...
        'when', datestr(now,31)); %#ok<TNOW1,DATST>
    save(cache, '-struct', 'J', '-v7.3');
    jac_fingerprint('write', [cache(1:end-4) '.fp.json'], ...
        struct('G_re', real(G), 'G_im', imag(G)), ...
        struct('rx', string(ch.rx), 'model', ch.N, 'tag', string(ch.tag), ...
               'chain', strjoin(string(cellfun(@(x) fmtv_(x), ch.config, ...
                   'UniformOutput', false)), ' '), ...
               'ncol', ncol, 'npix', numel(dz_idx), 'h_m', h, ...
               'when', string(datestr(now,31)))); %#ok<TNOW1,DATST>
    fprintf('    [jac] measured %d cols in %.1f min -> %s\n', ...
            ncol, toc(tswp)/60, cache);
    meta = struct('file', cache, 'ncol', ncol, 'cached', false);
end

function [a, contrast, alpha_used] = efc_(ch, dm, J, a_in, dz_idx, niter, alphas)
    G = double(J.G);
    Gr = [real(G); imag(G)];
    [U, S, V] = svd(Gr, 'econ');
    s = diag(S);
    a = a_in;
    seta_(dm, a);
    E = ch.run();
    pb = ch.peak_bare;
    contrast = zeros(1, niter + 1);
    contrast(1) = mean(abs(E(dz_idx)).^2) / pb;
    alpha_used = zeros(1, niter);
    fprintf('    [efc] iter 0: %.3e\n', contrast(1));
    for it = 1:niter
        e = double(E(dz_idx));
        Ue = U' * [real(e); imag(e)];
        best = struct('c', inf, 'a', [], 'E', [], 'alpha', NaN);
        for al = alphas
            alpha = al * s(1)^2;
            da = -V * ((s ./ (s.^2 + alpha)) .* Ue);
            at = a;
            for k = 1:numel(dm)
                sel = J.col_dm == k;
                at{k}(J.col_act(sel)) = at{k}(J.col_act(sel)) + da(sel);
            end
            seta_(dm, at);
            Et = ch.run();
            c = mean(abs(Et(dz_idx)).^2) / pb;
            if c < best.c
                best = struct('c', c, 'a', {at}, 'E', Et, 'alpha', al);
            end
        end
        if best.c >= contrast(it)
            fprintf('    [efc] iter %d: no alpha improves (best %.3e) -- stop\n', it, best.c);
            contrast = contrast(1:it);
            alpha_used = alpha_used(1:it-1);
            seta_(dm, a);                  % restore the ACCEPTED state
            ch.run();
            return
        end
        a = best.a;  E = best.E;
        contrast(it+1) = best.c;
        alpha_used(it) = best.alpha;
        fprintf('    [efc] iter %d: %.3e (alpha %.1e)\n', it, best.c, best.alpha);
    end
    seta_(dm, a);                          % engine = accepted final state
end

function seta_(dm, a)
    for k = 1:numel(dm), dm{k}.apply(a{k}); end
end

function s = sym_frac_(E, c, dz_idx)
    N = size(E,1);
    idx = mod(2*c - (1:N) - 1, N) + 1;     % i -> 2c-i (reflect about c; the
    Er = E(idx, idx);                      % wrapped edge rows are far
    Es = (E + Er) / 2;                     % outside the dark zone)
    s = sum(abs(Es(dz_idx)).^2) / max(sum(abs(E(dz_idx)).^2), realmin);
end

function t = fmtv_(v)
    if ischar(v) || isstring(v), t = char(v);
    elseif islogical(v), w = {'false','true'}; t = w{v+1};
    else, t = mat2str(v, 6);
    end
end
