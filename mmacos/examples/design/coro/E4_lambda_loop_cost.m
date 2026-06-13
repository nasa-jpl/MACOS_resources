function out = E4_lambda_loop_cost(n_lambda)
%E4_LAMBDA_LOOP_COST  Sprint 1 / E4 division-of-labor experiment.
%   Measure the load / set-wavelength / ray-trace / diffraction-
%   propagate cost split per wavelength on the coronagraph, to quantify
%   what a future MACOS-side `spectral_run` would buy over a MATLAB
%   loop of `set_src_wvl` + `intensity` calls (PLAN_DESIGN_LAYER §8, E4).
%
%   The decisive split is RAY-TRACE vs DIFFRACTION-PROPAGATE:
%     - The ray geometry of a REFLECTIVE system is WAVELENGTH-
%       INDEPENDENT.  A spectral_run could trace ONCE and loop only the
%       diffraction propagation over lambda, saving (nlambda-1)*t_trace.
%     - A MATLAB loop with intensity(reset_trace=true) RE-TRACES every
%       wavelength -- redundant work for reflective systems.
%     - (Refractive systems disperse: glass re-resolves per lambda, so
%       the trace is NOT shareable -- see §6.4 glass re-resolution.)
%
%   out = E4_LAMBDA_LOOP_COST(5)
    arguments
        n_lambda (1,1) double = 5
    end
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    cleanup_log = coro_log_start('E4_lambda_loop_cost'); %#ok<NASGU>

    MODEL_SIZE = 1024;
    DET_ELT    = 21;
    rxdir = fullfile(getenv('HOME'),'dev','MACOS_resources','pymacos','tests','Rx');
    rx = @(n) fullfile(rxdir, n);

    macos.init(MODEL_SIZE);

    % --- One-time load cost (amortized away by any per-lambda loop) --
    tl = tic; macos.load_rx(rx('Rx_Coro_FPM.in')); t_load = toc(tl);
    fprintf('[E4] load_rx (one-time) = %.3f s\n', t_load);

    wvl0    = macos.get_src_wvl();
    factors = linspace(0.95, 1.05, n_lambda);

    % --- Per-wavelength split: set / trace / propagate ---------------
    % modify() between set_src_wvl and trace() forces a REAL re-trace
    % (a bare trace() on cached state short-circuits to ~0 s).  The
    % diffraction propagate is then INT with reset_trace=false on the
    % freshly-traced state, so the split is clean.
    macos.set_src_wvl(wvl0); macos.modify(); macos.trace();  % warm up
    t_set = zeros(1,n_lambda); t_trace = zeros(1,n_lambda);
    t_prop = zeros(1,n_lambda); t_loop = zeros(1,n_lambda);
    for k = 1:n_lambda
        tk = tic;
        ts = tic; macos.set_src_wvl(wvl0*factors(k)); macos.modify(); t_set(k) = toc(ts);
        tt = tic; macos.trace();                                      t_trace(k) = toc(tt);
        tp = tic; macos.intensity(DET_ELT, 'reset_trace', false);     t_prop(k) = toc(tp);
        t_loop(k) = toc(tk);
    end
    macos.set_src_wvl(wvl0);

    mset=mean(t_set); mtr=mean(t_trace); mpr=mean(t_prop); mlp=mean(t_loop);
    fprintf('[E4] per-lambda means: set %.4f s, trace %.3f s, propagate %.3f s, loop-iter %.3f s\n', ...
            mset, mtr, mpr, mlp);
    overhead = mlp - (mset+mtr+mpr);
    fprintf('[E4] MATLAB/mex per-lambda overhead = %.4f s (loop-iter - sum of parts)\n', overhead);

    % --- What spectral_run would buy ---------------------------------
    matlab_loop = n_lambda*(mset+mtr+mpr) + n_lambda*max(overhead,0);
    % spectral_run, reflective: trace ONCE, propagate per lambda.
    spectral_refl = mtr + n_lambda*(mset+mpr);
    fprintf('\n[E4] nlambda=%d total:\n', n_lambda);
    fprintf('[E4]   MATLAB loop (re-trace each lambda)      = %.2f s\n', matlab_loop);
    fprintf('[E4]   spectral_run (reflective: trace once)   = %.2f s  (saves %.2f s, %.0f%%)\n', ...
            spectral_refl, matlab_loop-spectral_refl, ...
            100*(matlab_loop-spectral_refl)/matlab_loop);
    fprintf('[E4]   verdict: spectral_run amortization is worth it iff trace >> propagate\n');
    fprintf('[E4]            and the system is reflective (lambda-independent geometry).\n');
    fprintf('[E4]            trace/propagate ratio here = %.2f\n', mtr/mpr);

    out = struct('t_load',t_load,'t_set',t_set,'t_trace',t_trace, ...
        't_prop',t_prop,'t_loop',t_loop,'mean_set',mset,'mean_trace',mtr, ...
        'mean_prop',mpr,'overhead',overhead,'matlab_loop',matlab_loop, ...
        'spectral_refl',spectral_refl,'n_lambda',n_lambda, ...
        'model_size',MODEL_SIZE,'rx','Rx_Coro_FPM.in');
    save_coro_workspace('E4_lambda_loop_cost', out, 2);
end
