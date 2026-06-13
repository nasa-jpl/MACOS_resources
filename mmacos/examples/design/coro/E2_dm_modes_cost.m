function out = E2_dm_modes_cost(mode_counts, metric)
%E2_DM_MODES_COST  Sprint 1 / E2 division-of-labor experiment.
%   DM control entirely in the outer loop: DM Fourier modes as fmincon
%   design vars, no inner loop, reusing the E1 MATLAB-side dark-zone
%   merit.  Measures wall-time cost as a function of mode count to find
%   where a MATLAB-driven fmincon DM solve becomes prohibitive — the
%   bar a Sprint-3 Fortran DarkZone target must beat
%   (PLAN_DESIGN_LAYER.md §8 Sprint 1, E2).
%
%   The cost driver is fmincon's finite-difference gradient: each
%   iteration needs (nModes + 1) objective evals (forward differences),
%   each one a full CPROPAGATE trace to the detector (~E1's ~3 s at
%   model 1024).  We confirm that scaling empirically and extrapolate.
%
%   TWO Jacobian cost regimes are reported (see §"Jacobian cost"):
%     - GLOBAL modal basis (Zernike/Fourier, used here): (nMode+1)
%       traces/iteration -> goes prohibitive at large mode count.
%     - LOCAL actuator / influence-function basis: actuator influence
%       has compact support (< ~3 act spacings), so poking every ~6th
%       actuator gives spatially-separable responses -> the FULL
%       Jacobian costs ~(stencil+1) ≈ 37 traces INDEPENDENT of actuator
%       count.  This is the influence-function-DM payoff.
%
%   The DM (Elt 4 of Rx_Coro_DM, a 1024x1024 grid-data Reflector) is
%   driven via the grid-data surface setter — the SAME carrier that
%   holds influence-function / measured DM maps going forward.
%
%   The objective METRIC is selectable ('mean' | 'peak' | 'floor' |
%   'median' | 'energy'); the full metric set is reported for the
%   flat-DM baseline and each run regardless of which one is minimised.
%
%   out = E2_DM_MODES_COST([3 8 15], 'mean')
    arguments
        mode_counts (1,:) double = [3 8 15]
        metric      (1,:) char {mustBeMember(metric, ...
            {'mean','peak','floor','median','energy'})} = 'mean'
    end

    here = fileparts(mfilename('fullpath'));
    addpath(here);
    cleanup_log = coro_log_start('E2_dm_modes_cost'); %#ok<NASGU>

    MODEL_SIZE = 1024;
    DM_ELT     = 4;            % grid-data Reflector DM
    DET_ELT    = 21;           % science focal plane
    DZ_INNER   = 7;  DZ_OUTER = 10;
    LAM_MM     = 8.5e-4;       % 850 nm in BaseUnits (mm)
    FD_STEP    = 1e-3;         % fmincon FD step (= 1e-3 wave DM stroke)
    MAXITER    = 2;            % cost probe: a couple iters times the loop
    ACT_STENCIL = 36;          % ~6x6 separable-poke coloring (E2 note)
    rxdir = fullfile(getenv('HOME'),'dev','MACOS_resources','pymacos','tests','Rx');
    rx = @(n) fullfile(rxdir, n);

    macos.init(MODEL_SIZE);

    % --- Strehl reference + lambda/D from the no-mask PSF (E1) --------
    macos.load_rx(rx('Rx_Coro_noLyot.in'));
    I_no    = macos.intensity(DET_ELT);
    peak_no = max(I_no(:));
    lamD    = lambda_over_D_pixels(I_no);
    fprintf('[E2] lambda/D = %.2f px, no-mask peak = %.4e\n', lamD, peak_no);

    % --- Load the DM coronagraph, read DM grid geometry --------------
    macos.load_rx(rx('Rx_Coro_DM.in'));
    N  = mmacos('elt_srf_grid_size', double(DM_ELT), 1);
    [dx, ~] = mmacos('elt_srf_grid_data', double(DM_ELT), 0.0, ...
                     zeros(N), 0.0, double(N), double(N));
    fprintf('[E2] DM Elt %d grid = %dx%d, dx = %.4e\n', DM_ELT, N, N, dx);

    metrics_of = @(I) dark_zone_metrics(I, peak_no, lamD, DZ_INNER, DZ_OUTER);

    % Flat-DM baseline image + metrics + per-eval baseline.
    t1 = tic;
    I_flat = eval_dm(zeros(1,1), {zeros(N)}, DM_ELT, DET_ELT, dx, N);
    t_eval1 = toc(t1);
    m_flat = metrics_of(I_flat);
    fprintf(['[E2] single merit eval (flat DM) = %.3f s | flat-DM dark zone: ', ...
             'mean %.3e peak %.3e floor %.3e energy %.3e (%d px)\n'], ...
            t_eval1, m_flat.mean, m_flat.peak, m_flat.floor, ...
            m_flat.energy, m_flat.n_pix);

    % --- Cost sweep over mode counts ---------------------------------
    rows = struct('nModes',{},'funcCount',{},'iters',{},'wall',{}, ...
                  'per_eval',{},'per_iter',{},'metrics',{});
    I_runs = cell(1, numel(mode_counts));
    x_runs = cell(1, numel(mode_counts));
    opts = optimoptions('fmincon', 'Algorithm','sqp', ...
        'Display','off', 'SpecifyObjectiveGradient',false, ...
        'FiniteDifferenceType','forward', ...
        'FiniteDifferenceStepSize', FD_STEP, ...
        'MaxIterations', MAXITER, 'MaxFunctionEvaluations', 10000);

    for j = 1:numel(mode_counts)
        nm    = mode_counts(j);
        modes = fourier_modes(N, nm, LAM_MM);
        merit = @(coef) pick(metrics_of( ...
                    eval_dm(coef, modes, DM_ELT, DET_ELT, dx, N)), metric);
        x0 = zeros(1, nm);
        lb = -0.5*ones(1, nm);  ub = 0.5*ones(1, nm);
        tw = tic;
        [xopt, ~, ~, info] = fmincon(merit, x0, [],[],[],[], lb, ub, [], opts);
        wall = toc(tw);

        % Capture the final-state image + full metric set at xopt.
        I_fin = eval_dm(xopt, modes, DM_ELT, DET_ELT, dx, N);
        m_fin = metrics_of(I_fin);
        I_runs{j} = I_fin;  x_runs{j} = xopt;

        r.nModes=nm; r.funcCount=info.funcCount; r.iters=info.iterations;
        r.wall=wall; r.per_eval=wall/max(info.funcCount,1);
        r.per_iter=wall/max(info.iterations,1); r.metrics=m_fin;
        rows(end+1)=r; %#ok<AGROW>
        fprintf(['[E2] nModes=%2d: %3d evals, %d iters, %.1f s wall, ', ...
                 '%.2f s/eval, %.1f s/iter | final dz: mean %.3e peak %.3e ', ...
                 'floor %.3e energy %.3e\n'], ...
                nm, info.funcCount, info.iterations, wall, r.per_eval, ...
                r.per_iter, m_fin.mean, m_fin.peak, m_fin.floor, m_fin.energy);
    end

    % --- Jacobian cost: global modal vs local multiplexed ------------
    per_eval = median([rows.per_eval]);
    fprintf('\n[E2] median per-eval = %.2f s (objective minimised: %s)\n', ...
            per_eval, metric);
    fprintf('[E2] GLOBAL modal basis FD cost/iter = (nModes+1) x %.2f s\n', per_eval);
    K = 30; budget_s = 3600;
    for nlam = [1 3]
        nmax = budget_s/(K*per_eval*nlam) - 1;
        fprintf(['[E2]   prohibitive @ K=%d iters, nlambda=%d, 1-hr budget: ', ...
                 'nModes ~ %.0f\n'], K, nlam, nmax);
    end
    fprintf(['[E2] LOCAL actuator/influence-fn basis: separable %dx-poke ', ...
             'coloring => FULL Jacobian ~ (%d+1) x %.2f s = %.0f s, ', ...
             'INDEPENDENT of actuator count\n'], ACT_STENCIL, ACT_STENCIL, ...
            per_eval, (ACT_STENCIL+1)*per_eval);
    fprintf(['[E2]   -> a 2000-actuator DM Jacobian: global FD ', ...
             '%.0f s vs multiplexed %.0f s (factor %.0f)\n'], ...
            2001*per_eval, (ACT_STENCIL+1)*per_eval, ...
            2001/(ACT_STENCIL+1));

    out = struct('lamD',lamD,'peak_no',peak_no,'grid',N,'dx',dx, ...
        't_eval1',t_eval1,'rows',rows,'per_eval',per_eval, ...
        'metric',metric,'m_flat',m_flat,'I_flat',I_flat,'I_runs',{I_runs}, ...
        'x_runs',{x_runs},'mode_counts',mode_counts, ...
        'act_stencil',ACT_STENCIL,'model_size',MODEL_SIZE, ...
        'rx_dm','Rx_Coro_DM.in','dz',[DZ_INNER DZ_OUTER]);

    % Persist the full workspace (incl. 1024^2 images) for resume.
    save_coro_workspace('E2_dm_modes_cost', out, 2);
end

% ---------------------------------------------------------------------
function I = eval_dm(coef, modes, dm_elt, det_elt, dx, N)
% Set DM grid = sum_i coef_i*modes{i}, propagate, return detector image.
    G = zeros(N);
    for i = 1:numel(coef)
        G = G + coef(i) * modes{i};
    end
    mmacos('elt_srf_grid_data', double(dm_elt), dx, G, 1.0, double(N), double(N));
    I = macos.intensity(det_elt);
end

function s = pick(m, name)
    s = m.(name);
end

% ---------------------------------------------------------------------
function modes = fourier_modes(N, nmodes, lam_mm)
% Frequency-ordered 2D cosine ripple modes over the inscribed DM
% aperture (unit disk).  1 coef-unit = 1 wave (lam_mm) RMS of surface
% inside the aperture.  EFC/dark-hole "ripple" basis; the exact basis
% is immaterial to the COST measurement.
    [xx, yy] = meshgrid(linspace(-1,1,N), linspace(-1,1,N));
    ap = (xx.^2 + yy.^2) <= 1.0;
    kmax = ceil(sqrt(nmodes)) + 2;
    klist = [];
    for s = 0:(2*kmax)
        for kx = 0:s
            ky = s - kx;
            if kx==0 && ky==0, continue; end
            klist(end+1,:) = [kx ky]; %#ok<AGROW>
        end
    end
    [~, ord] = sort(sum(klist.^2, 2));
    klist = klist(ord, :);
    modes = cell(1, nmodes);
    for i = 1:nmodes
        kx = klist(i,1); ky = klist(i,2);
        m = cos(pi*(kx*xx + ky*yy));  m(~ap) = 0;
        rms = sqrt(mean(m(ap).^2));
        modes{i} = (lam_mm/rms) * m;
    end
end
