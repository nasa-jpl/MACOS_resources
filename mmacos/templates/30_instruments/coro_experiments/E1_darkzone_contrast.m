function out = E1_darkzone_contrast()
%E1_DARKZONE_CONTRAST  Sprint 1 / E1 division-of-labor experiment.
%   MATLAB-side outer merit: read intensity() at the science focal
%   plane of the Phase-5 Rx_Coro coronagraph, compute annular
%   dark-zone contrast entirely in MATLAB (ported contrast.py lambda/D
%   machinery), reproduce the ~3e-10 dark-zone baseline at 7-10 lambda/D,
%   and measure per-evaluation wall time at n_lambda = 3.
%
%   This is the MATLAB-side proof that the OUTER MERIT can live in
%   MATLAB with zero Fortran (PLAN_DESIGN_LAYER.md §8 Sprint 1, E1).
%   The measured per-eval cost is the input that scopes whether the
%   Sprint-3 DarkZone CALIB target must move into Fortran (E2/E3).
%
%   Returns a struct OUT with the no-mask / with-mask curves, the
%   derived lambda/D, the dark-zone contrast, and per-eval timings.
%
%   Run:  out = E1_darkzone_contrast();   (then exit(0) under -batch)

    here = fileparts(mfilename('fullpath'));
    addpath(here);                       % radial_contrast, etc.
    cleanup_log = coro_log_start('E1_darkzone_contrast'); %#ok<NASGU>

    MODEL_SIZE   = 1024;                 % Rx_Coro is a 1024-class grid
    DETECTOR_ELT = 21;                   % science focal plane
    RX_NOMASK    = 'Rx_Coro_noLyot.in';  % Phase 5.1 reference
    RX_CORO      = 'Rx_Coro_FPM.in';     % Phase 5.2: FPM=400um + Lyot=14mm
    DZ_INNER     = 7;                    % dark-zone inner edge (lambda/D)
    DZ_OUTER     = 10;                   % dark-zone outer edge (lambda/D)

    rxdir = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
                     'pymacos', 'tests', 'Rx');
    rx = @(n) fullfile(rxdir, n);

    macos.init(MODEL_SIZE);

    % --- No-mask reference: Strehl normaliser + empirical lambda/D ----
    macos.load_rx(rx(RX_NOMASK));
    I_no    = macos.intensity(DETECTOR_ELT);
    peak_no = max(I_no(:));
    lamD    = macos.lambda_over_D_pixels(I_no);
    fprintf('[E1] lambda/D = %.2f px (first null at 1.22 l/D = %.2f px)\n', ...
            lamD, 1.22 * lamD);
    fprintf('[E1] no-mask peak = %.4e\n', peak_no);

    % --- Coronagraph PSF: scoring target -----------------------------
    macos.load_rx(rx(RX_CORO));
    I_co    = macos.intensity(DETECTOR_ELT);
    peak_co = max(I_co(:));
    fprintf('[E1] with-mask peak = %.4e (suppression factor %.2e)\n', ...
            peak_co, peak_no / peak_co);

    % --- Radial contrast curves (Strehl-normalised to no-mask peak) --
    [r_no, c_no] = macos.radial_contrast(I_no, peak_no, lamD, 20.0);
    [r_co, c_co] = macos.radial_contrast(I_co, peak_no, lamD, 20.0);

    % --- Dark-zone digest --------------------------------------------
    fprintf('\n[E1] Radial contrast at key separations:\n');
    fprintf('%14s  %10s  %10s  %9s\n', 'r (lambda/D)', 'no-mask', ...
            'with-mask', 'gain');
    for rt = [0 1 2 3 5 7 10 15]
        [~, i_no] = min(abs(r_no - rt));
        [~, i_co] = min(abs(r_co - rt));
        gain = c_no(i_no) / max(c_co(i_co), 1e-300);
        fprintf('%14.2f  %10.3e  %10.3e  %9.2e\n', ...
                r_co(i_co), c_no(i_no), c_co(i_co), gain);
    end

    % Mean dark-zone contrast over [DZ_INNER, DZ_OUTER] lambda/D.
    dz = (r_co >= DZ_INNER) & (r_co <= DZ_OUTER) & isfinite(c_co);
    dz_contrast = mean(c_co(dz));
    fprintf('\n[E1] mean dark-zone contrast (%g-%g l/D) = %.3e\n', ...
            DZ_INNER, DZ_OUTER, dz_contrast);

    % --- Per-eval wall time at n_lambda = 3 --------------------------
    % An outer-merit evaluation = set wavelength, re-trace + propagate
    % to the detector (intensity re-traces), score the dark zone.  Time
    % that, with the coronagraph loaded, across a 3-wavelength band.
    macos.load_rx(rx(RX_CORO));
    wvl0    = macos.get_src_wvl();              % WaveUnits (typ. um)
    factors = [0.95, 1.00, 1.05];               % ~10% band, 3 samples
    n_lam   = numel(factors);
    t_eval  = zeros(1, n_lam);
    dz_lam  = zeros(1, n_lam);
    for k = 1:n_lam
        macos.set_src_wvl(wvl0 * factors(k));
        tic;
        Ik          = macos.intensity(DETECTOR_ELT);   % re-trace + propagate
        [rk, ck]    = macos.radial_contrast(Ik, peak_no, lamD, 20.0);
        dzk         = (rk >= DZ_INNER) & (rk <= DZ_OUTER) & isfinite(ck);
        dz_lam(k)   = mean(ck(dzk));
        t_eval(k)   = toc;
    end
    macos.set_src_wvl(wvl0);                    % restore
    fprintf('\n[E1] per-eval wall time at n_lambda=%d (coronagraph, model %d):\n', ...
            n_lam, MODEL_SIZE);
    for k = 1:n_lam
        fprintf('     lambda x %.3f : %.3f s   (dz contrast %.3e)\n', ...
                factors(k), t_eval(k), dz_lam(k));
    end
    fprintf('[E1] mean per-eval %.3f s, n_lambda=3 merit %.3f s total\n', ...
            mean(t_eval), sum(t_eval));

    out = struct('lamD', lamD, 'peak_no', peak_no, 'peak_co', peak_co, ...
        'suppression', peak_no / peak_co, ...
        'r_no', r_no, 'c_no', c_no, 'r_co', r_co, 'c_co', c_co, ...
        'dz_contrast', dz_contrast, 'dz_inner', DZ_INNER, ...
        'dz_outer', DZ_OUTER, 't_eval', t_eval, 'dz_lam', dz_lam, ...
        'model_size', MODEL_SIZE);
end
