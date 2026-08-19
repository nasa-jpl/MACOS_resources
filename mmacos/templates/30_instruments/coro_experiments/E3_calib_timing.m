function out = E3_calib_timing()
%E3_CALIB_TIMING  Sprint 1 / E3 division-of-labor experiment.
%   Time the existing (ray-trace) CALIB inner loop on a DM element of
%   Rx_Coro_DM at nGridpts=511, and measure the per-eval cost ratio of
%   a ray-trace objective (WFE) vs a diffraction objective (intensity
%   through the full CPROPAGATE chain).  Together these answer the
%   Sprint-1 question: would a Sprint-3 diffraction-scoring DarkZone
%   inner loop be seconds or hours?  (PLAN_DESIGN_LAYER.md §8, E3.)
%
%   Key distinction surfaced here: every shipping CALIB target
%   (WFE / WFE_ZMODE / BEAM / SPOT / OPL) is computed from the RAY
%   TRACE at the target element — none runs the diffraction FFT chain.
%   A DarkZone target (Sprint 3) WOULD, so its per-objective-eval cost
%   is the diffraction eval (~E1/E2's ~3.4 s), not the ray-trace eval.
%
%   out = E3_CALIB_TIMING()
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    cleanup_log = coro_log_start('E3_calib_timing'); %#ok<NASGU>

    MODEL_SIZE = 1024;
    DM_ELT     = 4;            % grid-data Reflector DM
    DET_ELT    = 21;
    rxdir = fullfile(getenv('HOME'),'dev','MACOS_resources','pymacos','tests','Rx');
    rx = @(n) fullfile(rxdir, n);

    macos.init(MODEL_SIZE);
    macos.load_rx(rx('Rx_Coro_DM.in'));

    % --- Per-eval cost primitives: ray-trace vs diffraction ----------
    % Force a REAL re-trace each time: a bare trace() with no preceding
    % modify() short-circuits on cached state (~0 s), which would be a
    % no-op measurement.  modify()+trace() re-traces; INT(reset=false)
    % then times the diffraction propagate alone on that traced state.
    macos.modify(); macos.trace();              % warm up
    t = tic; for i=1:3, macos.modify(); macos.trace(); end; t_trace = toc(t)/3;
    t = tic; for i=1:3, macos.intensity(DET_ELT,'reset_trace',false); end
    t_prop = toc(t)/3;
    t = tic; for i=1:3, macos.intensity(DET_ELT); end; t_obj_diff = toc(t)/3;
    fprintf(['[E3] per-eval: ray-trace(modify+trace) = %.3f s, ', ...
             'diffraction-propagate(INT only) = %.3f s, ', ...
             'full diffraction objective(trace+INT) = %.3f s\n'], ...
            t_trace, t_prop, t_obj_diff);
    fprintf('[E3] diffraction/ray-trace cost ratio = %.1fx\n', t_obj_diff/max(t_trace,1e-6));

    % --- Existing (ray-trace) CALIB inner loop on the DM -------------
    % Inject a recoverable tip on the DM, then let CALIB drive the DM
    % TIP/TILT to minimise RMS WFE (WFE target = ray-trace objective).
    macos.perturb(DM_ELT, 'rotation', [3e-6; 0; 0], 'frame', 'local');
    macos.modify();
    s0 = macos.trace();
    fprintf('[E3] injected DM tip; pre-CALIB rmsWFE = %.4e\n', s0.rmsWFE);

    macos.calib_clear_var_elts();
    macos.calib_set_var_elt(DM_ELT, 'TIP', 'TILT');
    macos.calib_set_target('WFE');
    macos.calib_set_iter(50);
    macos.calib_set_tol(1e-12);

    tc = tic;
    res = macos.calib();
    t_calib = toc(tc);
    fprintf(['[E3] CALIB (WFE, 2 DOF, ray-trace objective): %.2f s wall, ', ...
             'converged=%d, WFE %.4e -> %.4e\n'], ...
            t_calib, res.converged, res.old_wfe(1), res.new_wfe(1));

    % --- Project a diffraction-scoring (DarkZone) inner loop ---------
    % The EXISTING CALIB above is a RAY-TRACE objective (WFE): fast
    % (~%.0f s for 2 DOF).  A Sprint-3 DarkZone target replaces each
    % objective eval with a FULL diffraction eval (t_obj_diff), and an
    % FD gradient over nDOF costs (nDOF+1) of them per iteration.  The
    % E2 multiplexed local-actuator Jacobian instead measures the full
    % Jacobian in ~37 traces regardless of nDOF.
    fprintf(['\n[E3] PROJECTED DarkZone inner loop (diffraction objective, ', ...
             '%.2f s/eval, K=%d iters):\n'], t_obj_diff, 30);
    K = 30;
    for nDOF = [12 100 2000]
        naive_iter = (nDOF+1) * t_obj_diff;
        mux_jac    = 37 * t_obj_diff;          % E2 separable-poke Jacobian
        fprintf(['[E3]   nDOF=%4d: naive FD %.0f s/iter -> %.1f h for K=%d ', ...
                 '| multiplexed Jacobian %.0f s (N-indep)\n'], ...
                nDOF, naive_iter, naive_iter*K/3600, K, mux_jac);
    end
    fprintf(['[E3] verdict: ray-trace CALIB inner loop = SECONDS; naive-FD ', ...
             'diffraction DarkZone = HOURS at real DM scale; multiplexed/EFC ', ...
             '= ~2 min, N-independent.\n']);

    out = struct('t_trace',t_trace,'t_prop',t_prop,'t_obj_diff',t_obj_diff, ...
        'ratio',t_obj_diff/max(t_trace,1e-6), ...
        't_calib',t_calib,'calib',res,'pre_wfe',s0.rmsWFE, ...
        'model_size',MODEL_SIZE,'rx_dm','Rx_Coro_DM.in');
    save_coro_workspace('E3_calib_timing', out, 2);
end
