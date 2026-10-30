function est = ctb_probe_estimate(ch, jac, a0, opts)
%CTB_PROBE_ESTIMATE  Pairwise-probe estimate of the CTB dark-zone field.
%   EST = CTB_PROBE_ESTIMATE(CH, JAC, A0) estimates the complex dark-zone
%   field from intensity images only, the way a lab does it (Give'on, Kern
%   & Shaklan 2011, "Pair-wise, deformable mirror, image plane-based
%   diffraction sensing and control", Proc. SPIE 8151):
%
%     1. I_0      the image at the DM commands A0;
%     2. I_k+-    for each probe k, the image with A0 +/- a_k on the probe
%                 DM (each a fresh CH.run() through the full masked chain);
%     3. dE_k  =  G a_k, the Jacobian's model of the probe field;
%     4. per pixel, I_k+ - I_k- = 4 Re(conj(dE_k) E) for the npairs probes,
%        solved for [Re E; Im E] by least squares (ctb_probe_solve).
%
%   The engine's true field is read ONLY to score the estimate (EST.E_true,
%   EST.floor); it is never fed back.  The probe-intensity diagnostic,
%   (I_k+ + I_k-)/2 - I_0 against |dE_k|^2, is the model-mismatch meter:
%   its dark-zone ratio is 1.0 when the Jacobian predicts the probe.
%
%   CH    a ctb_chain (the plant), on the Jacobian's chain
%   JAC   a ctb_dm_jacobian struct, or a path to its .mat.  A path is
%         checked as ctb_efc checks its cache: ctb_jac_check against CH's
%         config, and the <name>.fp.json truth beside it when there is one.
%         A struct is checked against CH's config (ctb_jac_check).
%   A0    current DM commands, one [nact^2] vector (mm) per Jacobian DM;
%         {} = the Jacobian's linearization point (JAC.a0, else zeros)
%
%   Name-value:
%     'npairs'    probe pairs (3; >= 2)
%     'probe_nm'  probe amplitude, the peak actuator stroke in nm.
%                 [] (default) sets it so the probes' mean model intensity
%                 on the dark zone equals the mean of I_0 (the speckle
%                 level); EST.probe_nm reports the value used.
%     'probes'    explicit probe shapes, [nact^2 x npairs] on the probe DM
%                 (scaled to peak 1, then by 'probe_nm').  [] (default) =
%                 sinc-sine probes after Give'on: a narrow row in y (so
%                 each covers the whole annulus height), along x a sinc
%                 whose focal-plane lobe spans (outer+2) lambda/D centred
%                 at +/-(outer+1)/2 lambda/D (each half of the annulus is
%                 seen by one lobe), phases theta_k = (k-1) pi / npairs.
%                 Even-numbered probes are the same shape turned 90 deg:
%                 where the two x lobes overlap (|u| < 1/2 lambda/D) a
%                 pixel otherwise gets one phase from every x probe, and
%                 its 2x2 solve is singular (condition 1e7 measured on the
%                 shipped CTB; 9 with the turned probes).
%     'probe_dm'  which Jacobian DM carries the probes (1)
%     'photons'   0 = noiseless; else photons per image at the bare peak,
%                 Poisson noise on every image (PLANT ONLY, like ctb_efc's
%                 'quant_nm': the estimator sees noisy images, the score
%                 uses the true field)
%     'seed'      rng seed for the photon noise (1)
%     'dm'        reuse ctb_dm objects (cell, in JAC.dm order) ({})
%     'check_jac' check a struct JAC against CH's config (true); ctb_efc
%                 passes false for the Jacobian it has already checked
%
%   EST fields: E_hat, E_true (dark-zone vectors), floor (mean
%   |E_hat - E_true|^2 / peak_bare: the estimation floor), rel_err
%   (||E_hat - E_true|| / ||E_true||), err_px (per-pixel |dE|^2/peak, for
%   the histogram), amp_err / phase_err (per pixel), c_true / c_est (mean
%   contrast of each), ratio (1 x npairs diagnostic), cond2 (per-pixel
%   2x2 conditioning), I0 / Ip / Im (images, dark-zone vectors), dE,
%   probes, probe_nm, photons, nruns.
%
%   Run:  >> est = ctb_probe_estimate(ch, J, {});
%   See also: ctb_probe_solve, ctb_efc, ctb_dm_jacobian, ctb_chain.
    arguments
        ch  struct
        jac
        a0  cell = {}
        opts.npairs   (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(opts.npairs, 2)} = 3
        opts.probe_nm double {mustBeScalarOrEmpty, mustBePositive} = []
        opts.probes   double = []
        opts.probe_dm (1,1) double {mustBeInteger, mustBePositive} = 1
        opts.photons  (1,1) double {mustBeNonnegative} = 0
        opts.seed     (1,1) double = 1
        opts.dm       cell = {}
        opts.check_jac (1,1) logical = true
    end
    if opts.check_jac || ~isstruct(jac)
        J = load_jac_(jac, ch);
    else
        J = jac;               % the caller has checked it (ctb_efc)
    end

    % ---- DM models and the column map (as ctb_efc) ---------------------
    ndm = numel(J.dm);
    dm = opts.dm;
    if isempty(dm)
        dm = cell(1, ndm);
        for k = 1:ndm
            d = J.dm(k);
            dm{k} = ctb_dm('ielt', d.ielt, 'ng', d.ng, 'gdx_mm', d.gdx_mm, ...
                           'nact', d.nact, 'pitch_mm', d.pitch_mm, ...
                           'beam_d_mm', d.beam_d_mm, 'coupling', d.coupling);
        end
    end
    assert(opts.probe_dm <= ndm, 'ctb_probe_estimate: probe_dm %d of %d DMs', ...
        opts.probe_dm, ndm);
    udm  = unique(J.col_dm, 'stable');
    dmof = arrayfun(@(c) find(udm == c, 1), J.col_dm);
    if isempty(a0)
        if isfield(J, 'a0') && ~isempty(J.a0), a0 = J.a0;
        else, a0 = arrayfun(@(d) zeros(d.nact^2, 1), J.dm, 'UniformOutput', false);
        end
    end
    assert(numel(a0) == ndm, 'ctb_probe_estimate: A0 needs one command per DM (%d)', ndm);

    dz = J.dz_idx;  pb = J.peak_bare;
    cbm = 1e-3;  if isfield(J, 'cbm'), cbm = J.cbm; end     % the deck's base unit, m
    u_nm = 1e9 * cbm;                                          % deck units -> nm
    sel = dmof == opts.probe_dm;
    Gp  = double(J.G(:, sel));               % probe DM's columns
    cact = J.col_act(sel);                    % their actuator indices
    pd  = J.dm(opts.probe_dm);

    % ---- probe shapes (peak 1) -----------------------------------------
    if isempty(opts.probes)
        S = default_probes_(pd, J.outer_lamD, opts.npairs);
    else
        S = opts.probes;
        assert(size(S, 1) == pd.nact^2, ...
            'ctb_probe_estimate: probes must be [nact^2 x npairs] = [%d x n]', pd.nact^2);
        opts.npairs = size(S, 2);
        S = S ./ max(abs(S), [], 1);
    end
    K = opts.npairs;
    dE1 = Gp * S(cact, :);                    % model field per unit probe (mm)

    % ---- images ----------------------------------------------------------
    rs = []; if opts.photons > 0, rs = RandStream('mt19937ar', 'Seed', opts.seed); end
    nruns = 0;
    apply_(dm, a0);
    E0 = ch.run();  nruns = nruns + 1;
    E_true = double(E0(dz));
    I0 = noisy_(abs(E_true).^2, pb, opts.photons, rs);

    if isempty(opts.probe_nm)
        A = sqrt(mean(I0) / mean(abs(dE1).^2, 'all'));   % mm, peak stroke
    else
        A = opts.probe_nm / u_nm;
    end
    dE = A * dE1;
    Ip = zeros(numel(dz), K);  Im = Ip;
    for k = 1:K
        for sgn = [1 -1]
            at = a0;
            at{opts.probe_dm} = at{opts.probe_dm} + sgn * A * S(:, k);
            apply_(dm, at);
            Ek = ch.run();  nruns = nruns + 1;
            Ik = noisy_(abs(double(Ek(dz))).^2, pb, opts.photons, rs);
            if sgn > 0, Ip(:, k) = Ik; else, Im(:, k) = Ik; end
        end
    end
    apply_(dm, a0);                           % leave the plant as found

    % ---- estimate, diagnostic, score -------------------------------------
    [E_hat, cond2] = ctb_probe_solve(Ip - Im, dE);
    ratio = mean((Ip + Im)/2 - I0, 1) ./ mean(abs(dE).^2, 1);
    d = E_hat - E_true;
    err_px = abs(d).^2 / pb;
    est = struct('E_hat', E_hat, 'E_true', E_true, ...
        'floor', mean(err_px), 'rel_err', norm(d) / norm(E_true), ...
        'err_px', err_px, ...
        'amp_err', abs(E_hat) ./ abs(E_true) - 1, ...
        'phase_err', angle(E_hat .* conj(E_true)), ...
        'c_true', mean(abs(E_true).^2) / pb, 'c_est', mean(abs(E_hat).^2) / pb, ...
        'ratio', ratio, 'cond2', cond2, 'I0', I0, 'Ip', Ip, 'Im', Im, ...
        'dE', dE, 'probes', A * S, 'probe_nm', A * u_nm, ...
        'photons', opts.photons, 'nruns', nruns, 'dz_idx', dz, 'peak_bare', pb);
end

% ======================================================================
function J = load_jac_(jac, ch)
    if isstruct(jac)
        J = jac;
        ctb_jac_check(J, ch.config, '(struct)');
        return
    end
    jp = char(jac);
    assert(isfile(jp), 'ctb_probe_estimate: Jacobian %s not found', jp);
    J = load(jp);
    ctb_jac_check(J, ch.config, jp);
    fpp = regexprep(jp, '\.mat$', '.fp.json');
    if isfile(fpp)
        [ok, rep] = jac_fingerprint('check', J, fpp);
        if ~ok
            error('ctb_efc:fingerprint', ...
                ['%s does not match its fingerprint truth %s (%s): it was ' ...
                 'measured on another deck, mask or engine.'], jp, fpp, rep);
        end
    end
end

% ======================================================================
function S = default_probes_(d, outer_lamD, K)
    % actuator centres in aperture units (D = the controlled beam)
    x = d.acx(:) / d.beam_d_mm;  y = d.acy(:) / d.beam_d_mm;
    xi0 = (outer_lamD + 1) / 2;               % lobe centre, lambda/D
    Wx  = outer_lamD + 2;                     % lobe width in x, lambda/D
    Wy  = 2 * (outer_lamD + 1);               % full annulus height
    S = zeros(numel(x), K);
    for k = 1:K
        th = (k - 1) * pi / K;
        if mod(k, 2) == 1      % lobes at +/- xi0 along x
            S(:, k) = sinc_(Wx * x) .* sinc_(Wy * y) .* cos(2*pi*xi0*x + th);
        else                   % the same probe turned 90 degrees
            S(:, k) = sinc_(Wx * y) .* sinc_(Wy * x) .* cos(2*pi*xi0*y + th);
        end
    end
    S(~d.active(:), :) = 0;
    S = S ./ max(abs(S), [], 1);
end

function s = sinc_(t)
    s = ones(size(t));
    nz = t ~= 0;
    s(nz) = sin(pi * t(nz)) ./ (pi * t(nz));
end

function apply_(dm, cmd)
    for k = 1:numel(dm), dm{k}.apply(cmd{k}); end
end

function I = noisy_(I, pb, photons, rs)
%NOISY_  Poisson photon noise at PHOTONS per image at the bare peak.
    if photons <= 0, return; end
    lam = max(I, 0) / pb * photons;
    n = poisson_(lam, rs);
    I = n / photons * pb;
end

function n = poisson_(lam, rs)
    % Knuth for small means, normal approximation (with continuity) above 50
    n = zeros(size(lam));
    big = lam > 50;
    n(big) = max(0, round(lam(big) + sqrt(lam(big)) .* randn(rs, nnz(big), 1)));
    idx = find(~big);
    for m = idx(:)'
        L = exp(-lam(m));  k = 0;  p = rand(rs);
        while p > L, k = k + 1;  p = p * rand(rs); end
        n(m) = k;
    end
end
