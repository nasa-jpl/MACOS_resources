function OUT = s5_timeseries(over)
%S5_TIMESERIES  e2e6m stage 5: a drifting observatory, and the contrast.
%
%   Plays a random + correlated DRIFT history of the 19 segments through
%   the ENGINE, uncorrected and corrected, and reports the two series
%   that matter for this use case side by side:
%
%     rms WFE vs time       at the coronagraph exit pupil, and
%     dark-zone CONTRAST vs time at the science plane
%
%   The contrast series is the payoff and the expensive half: every point
%   is a full diffraction propagation plus the APLC mask chain, so the
%   timeline is SUBSAMPLED (`P.ts.every`) and the report says by how much.
%
%   ------------------------------------------------------------------
%   MET IS NOT IN SCOPE HERE -- stated, per the brief
%   ------------------------------------------------------------------
%   The committed `run_compare` / `run_simulator` runners drive an RBCS
%   METROLOGY loop and both REQUIRE the MET products (`run_met`: the
%   Stewart truss, dedx/dldx, the estimator blocks).  Standing that up on
%   this train means reconciling run_met's body list with a Jacobian
%   harvested on the FULL train, which is integration work this stage
%   does not do.  So:
%
%     * there is no metrology loop and no sensed-measurement bars.  The
%       control here is IMAGE-BASED: it sees the wavefront, not a truss.
%       That makes the corrected leg an OPTIMISTIC bound -- a real loop
%       estimates the state from noisy metrology and does worse.
%     * `run_compare`'s substance is kept: [1] below pokes DOFs and
%       checks the ENGINE against the linear model, which is what
%       validates the S4 Jacobian.  What is dropped is the l/e
%       measurement bars, which are metrology.
%
%   ------------------------------------------------------------------
%   THE TWO-PASS RULE (correctness, not convenience)
%   ------------------------------------------------------------------
%   The engine perturb path is INCREMENTAL and single-axis rotation
%   increments do not commute, so toggling +-u every frame leaves a
%   systematic ~|u_rot|^2 non-closure per cycle that accumulates into a
%   phantom rotation.  The whole UNCORRECTED history is played first, the
%   Rx is reloaded for a clean state, and the CORRECTED history is played
%   in a second pass -- within a pass the large state is applied once and
%   the per-frame increments are nm-scale.
%
%   METRIC TAGS.  WFE: rms OPD at the CORONAGRAPH exit pupil, piston
%   removed, in waves at 500 nm.  Contrast: dark-zone mean over
%   3-15 lambda/D, Strehl-normalised to the BARE on-axis peak of the
%   NOMINAL train (one fixed reference for the whole series, so the
%   curve is comparable frame to frame).
%
%   OUT = S5_TIMESERIES()      run at the default parameter set
%   OUT = S5_TIMESERIES(OVER)  ... with e2e6m_params overrides
%
%   See also E2E6M_PARAMS, S4_SENSITIVITIES, S3_CORO.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    setup_(here);
    P = e2e6m_params(over);
    if isempty(P.outdir), P.outdir = here; end
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    rx = fullfile(P.outdir, P.sn.rx);
    assert(isfile(rx), 's5_timeseries: %s not found -- run S3 first', rx);
    S4 = fullfile(P.outdir, 's4_sens.mat');
    assert(isfile(S4), 's5_timeseries: %s not found -- run S4 first', S4);

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m S5 -- the drift time series');
    L = say_(L, 'deck %s', rx);
    L = say_(L, 'MET NOT IN SCOPE: image-based control, no metrology loop --');
    L = say_(L, '   the corrected leg is an OPTIMISTIC bound (see the header).');
    L = say_(L, 'WFE: rms OPD at the coronagraph exit pupil, piston removed, waves @ %g nm', ...
             P.lambda_m*1e9);
    L = say_(L, 'contrast: dark-zone mean %g-%g lambda/D, normalised to the NOMINAL bare peak', ...
             P.co.inner_lamD, P.co.outer_lamD);

    % ---- the control basis, from S4 --------------------------------------
    Z = load(S4, 'ox');
    [dwdu, ucols, uelts] = control_basis_(Z.ox, P);
    L = say_(L, '\n[0] control basis: %d rows x %d columns (%d segments x 6 DOF)', ...
             size(dwdu,1), size(dwdu,2), numel(uelts));

    % ---- [1] engine vs linear model (run_compare's substance) ------------
    L = say_(L, '\n[1] engine vs linear model, %d sample DOFs at %g nm / %g nrad:', ...
             P.ts.n_check, P.ts.d_trans*1e9, P.ts.d_rot*1e9);
    chk = linear_check_(rx, P, dwdu, ucols, uelts);
    for k = 1:numel(chk.col)
        L = say_(L, '    elt %2d dof %d: |engine| %.4g  |model| %.4g  rel.err %.3g', ...
                 chk.elt(k), chk.dof(k), chk.n_eng(k), chk.n_mod(k), chk.rel(k));
    end
    L = say_(L, '    worst relative error %.3g  [%s]', chk.worst, ...
             gate_(chk.worst < P.ts.tol_linear));

    % ---- [2] the drift history -------------------------------------------
    rng(P.ts.seed);
    [X, tvec] = drift_history_(P, numel(uelts));
    L = say_(L, '\n[2] history: %d frames, dt %g s, %d states', ...
             size(X,2), P.ts.dt, size(X,1));
    L = say_(L, '    random walk %g nm / %g nrad per step, correlated drift %g nm / %g nrad per 100 s', ...
             P.ts.walk_trans*1e9, P.ts.walk_rot*1e9, ...
             P.ts.drift_trans*1e9, P.ts.drift_rot*1e9);

    % ---- [3] two passes through the engine -------------------------------
    L = say_(L, '\n[3] coronagraph setup (built ONCE: prolate + masks + scales)');
    C = coro_setup_(rx, P);
    L = say_(L, '    lambda/D at the FPA %.3f px | apodizer radius %.1f px', ...
             C.lamD_fpa_px, C.r_apod_px);
    L = say_(L, '    nominal bare on-axis peak %.4e (the series normalisation)', ...
             C.peak_bare);

    L = say_(L, '\n[3] pass 1 of 2: UNCORRECTED');
    A = play_(rx, P, X, zeros(size(X)), uelts, C);
    L = say_(L, '    WFE %.4f -> %.4f waves; contrast %.3e -> %.3e (%d scored frames)', ...
             A.wfe(1), A.wfe(end), A.con(find(isfinite(A.con),1)), ...
             A.con(find(isfinite(A.con),1,'last')), nnz(isfinite(A.con)));

    L = say_(L, '\n    solving the image-based correction on frame %d', P.ts.wfc_frame);
    U = wfc_(rx, P, X, uelts, ucols, dwdu);
    L = say_(L, '    control |u| rms %.3g nm / %.3g nrad', ...
             1e9*rms_(U(4:6:end)), 1e9*rms_(U(1:6:end)));

    L = say_(L, '\n[3] pass 2 of 2: CORRECTED (control held, history drifts on)');
    B = play_(rx, P, X, repmat(U,1,size(X,2)), uelts, C);
    L = say_(L, '    WFE %.4f -> %.4f waves; contrast %.3e -> %.3e', ...
             B.wfe(1), B.wfe(end), B.con(find(isfinite(B.con),1)), ...
             B.con(find(isfinite(B.con),1,'last')));

    % ---- [4] the payoff figure -------------------------------------------
    png = fullfile(P.outdir,'s5_series.png');
    series_fig_(tvec, A, B, P, png);
    L = say_(L, '\n[4] payoff figure: %s', png);
    L = say_(L, '    contrast scored every %d frames (%d of %d) -- each point is a', ...
             P.ts.every, nnz(isfinite(A.con)), numel(tvec));
    L = say_(L, '    full diffraction propagation plus the APLC mask chain.');

    L = say_(L, '\nS5 DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'s5_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);

    OUT = struct('P',P, 'chk',chk, 'X',X, 't',tvec, 'unc',A, 'cor',B, ...
                 'U',U, 'uelts',uelts, 'figure',png, 'text',txt, ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'s5_run.mat'),'OUT','-v7.3');
end

% =========================================================================
function setup_(here)
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
end

function [dwdu, cols, elts] = control_basis_(ox, P)
%CONTROL_BASIS_  The SEGMENT columns of dwdx -- the control authority.
%   The group columns are appended AFTER the per-element block and are
%   NOT control DOFs (they are the same motion, counted once as a body);
%   including them would double-count and make the ridge singular.
    A = ox.dwdxall;
    isgrp = false(1, size(A,2));
    if isfield(ox,'kind'), isgrp = strcmp(ox.kind, 'Group'); end
    e = ox.elt(:).';
    elts = unique(e(~isgrp & ismember(e, P.ts.control_elts)), 'stable');
    cols = find(~isgrp & ismember(e, elts));
    dwdu = A(:, cols);
end

function chk = linear_check_(rx, P, dwdu, cols, elts)
%LINEAR_CHECK_  Poke a sample of control DOFs and compare the ENGINE's
%   OPD change with the Jacobian column.  This is what says the S4
%   Jacobian describes THIS deck; without it the whole time series is
%   an assertion.
    macos.init(P.sn.model);
    n = macos.load_rx(rx);
    macos.opd_ref('chief');
    macos.trace(n);
    W0 = macos.opd();
    m = finite_(W0);
    pick = round(linspace(1, numel(cols), P.ts.n_check));
    chk = struct('col',cols(pick), 'elt',zeros(1,numel(pick)), ...
                 'dof',zeros(1,numel(pick)), 'n_eng',zeros(1,numel(pick)), ...
                 'n_mod',zeros(1,numel(pick)), 'rel',zeros(1,numel(pick)));
    for k = 1:numel(pick)
        c = cols(pick(k));
        j = mod(pick(k)-1, 6);                     % 0..5 = Rx..Tz
        ie = elts(1 + floor((pick(k)-1)/6));
        d = zeros(6,1);
        if j < 3, d(j+1) = P.ts.d_rot; else, d(j+1) = P.ts.d_trans; end
        macos.perturb(ie, 'rotation', d(1:3), 'translation', d(4:6), ...
                      'frame','local');
        macos.modify();  macos.trace(n);
        W1 = macos.opd();
        macos.perturb(ie, 'rotation', -d(1:3), 'translation', -d(4:6), ...
                      'frame','local');
        macos.modify();
        dW = W1(m) - W0(m);
        amp = P.ts.d_rot;  if j >= 3, amp = P.ts.d_trans; end
        mod = dwdu(1:nnz(m), pick(k)) * amp;       % centre-field block
        chk.elt(k) = ie;  chk.dof(k) = j;
        chk.n_eng(k) = rms_(dW);
        chk.n_mod(k) = rms_(mod);
        chk.rel(k) = rms_(dW(:) - mod(:)) / max(rms_(dW), realmin);
    end
    chk.worst = max(chk.rel);
end

function [X, t] = drift_history_(P, nb)
%DRIFT_HISTORY_  Random walk + a correlated drift, per segment, 6 DOF.
%   The correlated part is a COMMON direction shared by every segment
%   (a thermal soak moves the whole assembly one way); the random part
%   is independent per DOF.  Both are what a real observatory does
%   between wavefront-control updates.
    nT = P.ts.frames;  t = (0:nT-1)*P.ts.dt;
    X = zeros(6*nb, nT);
    w = zeros(6*nb, 1);
    dir = randn(6*nb, 1);
    dir(4:6:end) = dir(4:6:end);            % keep the per-DOF scaling below
    dir = dir / max(norm(dir), realmin);
    for k = 2:nT
        s = zeros(6*nb,1);
        for j = 0:5
            a = P.ts.walk_rot;  if j >= 3, a = P.ts.walk_trans; end
            s(j+1:6:end) = a*randn(nb,1);
        end
        w = w + s;
        f = (t(k)/100) * ones(6*nb,1);
        g = zeros(6*nb,1);
        for j = 0:5
            a = P.ts.drift_rot;  if j >= 3, a = P.ts.drift_trans; end
            g(j+1:6:end) = a;
        end
        X(:,k) = w + dir .* g .* f;
    end
end

function R = play_(rx, P, X, U, elts, C)
%PLAY_  One pass: apply state+control frame by frame, INCREMENTALLY, and
%   record WFE every frame and contrast every P.ts.every frames.
    nT = size(X,2);
    R = struct('wfe',nan(1,nT), 'con',nan(1,nT));
    macos.init(P.co.model);
    n = macos.load_rx(rx);
    macos.opd_ref('chief');
    prev = zeros(size(X,1),1);
    for k = 1:nT
        d = (X(:,k) + U(:,k)) - prev;
        apply_(elts, d);
        prev = X(:,k) + U(:,k);
        macos.modify();  macos.trace(n);
        W = macos.opd();  m = finite_(W);
        v = W(m) - mean(W(m));
        R.wfe(k) = std(v) / P.lambda_m;
        if mod(k-1, P.ts.every) == 0 || k == nT
            R.con(k) = contrast_now_(C, P);
        end
    end
    apply_(elts, -prev);  macos.modify();
end

function apply_(elts, d)
    for b = 1:numel(elts)
        q = d(6*(b-1)+1 : 6*b);
        if ~any(q), continue; end
        macos.perturb(elts(b), 'rotation', q(1:3), 'translation', q(4:6), ...
                      'frame','local');
    end
end

function U = wfc_(rx, P, X, elts, cols, dwdu) %#ok<INUSD>
%WFC_  Image-based wavefront control at one frame: an ITERATED
%   Tikhonov-ridge least squares on the engine wavefront.  Iterated
%   because the state at that frame is not in the linear regime -- one
%   pinv step off a micron-scale error leaves a large residual.
    macos.init(P.sn.model);
    n = macos.load_rx(rx);
    macos.opd_ref('chief');
    macos.trace(n);
    W0 = macos.opd();  m = finite_(W0);
    kf = min(P.ts.wfc_frame, size(X,2));
    apply_(elts, X(:,kf));
    macos.modify();  macos.trace(n);
    nrow = nnz(m);
    A = dwdu(1:nrow, :);
    lam = P.ts.ridge * max(vecnorm(A))^2;
    R = (A.'*A + lam*eye(size(A,2))) \ A.';
    U = zeros(size(A,2),1);
    for it = 1:P.ts.wfc_iters
        W = macos.opd();
        v = W(m) - mean(W(m));
        du = -R * v(:);
        apply_(elts, du);  U = U + du;
        macos.modify();  macos.trace(n);
    end
    apply_(elts, -(X(:,kf) + U));  macos.modify();
end

function C = coro_setup_(rx, P)
%CORO_SETUP_  Everything about the coronagraph that does NOT depend on
%   the state: the station map, the focal/pupil scales, the prolate
%   apodizer, the occulter and the Lyot stop, and the ONE bare on-axis
%   peak the whole series is normalised to.
%
%   Hoisted deliberately.  The prolate is 2392 power iterations of
%   1024^2 FFTs -- ~1.5 min -- and it depends on the PUPIL, not on where
%   the segments happen to be this frame.  Rebuilding it per frame would
%   dominate the run and change nothing.
%
%   The masks likewise: a fixed mask is the physical object.  Only the
%   FIELD changes frame to frame.
    C.e = elt_map_(rx);
    N = P.co.model;
    macos.init(N);
    macos.load_rx(rx);

    cbm = macos.cbm();  lam = macos.get_src_wvl()*cbm;
    macos.intensity(C.e.FPM);
    Isph  = macos.intensity(C.e.FPM-1, 'reset_trace', false);
    dxsph = abs(macos.dx_at(C.e.FPM-1));
    R_fpm = abs(macos.get_elt_z(C.e.FPM-1))*cbm;
    Dbeam = 2*beam_radius_(Isph, dxsph);
    C.dx_f       = lam * R_fpm / (N*dxsph);
    C.lamD_fpm_m = lam * R_fpm / Dbeam;

    Ily = macos.intensity(C.e.Lyot, 'reset_trace', false);
    C.dx_lyot = abs(macos.dx_at(C.e.Lyot));
    C.r_lyot_geom_m = beam_radius_(Ily, C.dx_lyot);

    macos.intensity(C.e.FPA);
    Iep  = macos.intensity(C.e.ExitPupil, 'reset_trace', false);
    Dep  = 2*beam_radius_(Iep, abs(macos.dx_at(C.e.ExitPupil)));
    R_fpa = abs(macos.get_elt_z(C.e.ExitPupil))*cbm;
    C.lamD_fpa_px = (lam * R_fpa / Dep) / abs(macos.dx_at(C.e.FPA));

    Iap = macos.intensity(C.e.Apodizer);
    C.r_apod_px = beam_radius_(Iap, 1);          % already in pixels

    C.Phi  = ctb_apod_prolate(N, C.r_apod_px, P.co.r_occ_lamD, ...
                              'n_iter', P.co.prolate_iter);
    C.Mocc = 1 - ctb_mask_disk(N, C.dx_f, P.co.r_occ_lamD*C.lamD_fpm_m, 8);
    C.Mlyo = ctb_mask_disk(N, C.dx_lyot, P.co.r_lyot_frac*C.r_lyot_geom_m, 8);

    % the ONE normalisation for the whole series: the BARE on-axis peak of
    % the NOMINAL train.  Re-normalising per frame would divide out
    % exactly the degradation the series exists to show.
    I = macos.intensity(C.e.FPA);
    C.peak_bare = max(I(:));
end

function c = contrast_now_(C, P)
%CONTRAST_NOW_  Dark-zone mean through the pre-built APLC chain on the
%   CURRENT engine state.  No mask is rebuilt here; only the field moves.
    try
        macos.intensity(C.e.seed1);
        macos.intensity(C.e.Apodizer,'reset_trace',false);
        macos.apodize(C.e.Apodizer, C.Phi);
        macos.intensity(C.e.FPM,'reset_trace',false);
        macos.apodize(C.e.FPM, C.Mocc);
        macos.intensity(C.e.Lyot,'reset_trace',false);
        macos.apodize(C.e.Lyot, C.Mlyo);
        I = macos.intensity(C.e.FPA,'reset_trace',false);
        dz = macos.dark_zone_metrics(I, C.peak_bare, C.lamD_fpa_px, ...
                                     P.co.inner_lamD, P.co.outer_lamD);
        c = dz.mean;
    catch
        c = NaN;
    end
end

function e = elt_map_(rx)
%ELT_MAP_  Station indices by NAME.  Never an index table: the segment
%   count and the quartet layout both move them.
    nm = regexp(fileread(rx), '^\s*EltName=\s*(\S+)', 'tokens','lineanchors');
    nm = cellfun(@(c) c{1}, nm, 'UniformOutput', false);
    at = @(s) find(strcmp(nm, s), 1);
    e = struct('Apodizer',at('Apodizer'), 'FPM',at('FPM'), 'Lyot',at('Lyot'), ...
               'ExitPupil',at('ExitPupil'), 'FPA',at('Science'));
    p = find(~cellfun('isempty', regexp(nm, '^Prop\d+_start$', 'once')), 1);
    e.seed1 = p;
    f = fieldnames(e);
    for k = 1:numel(f)
        assert(~isempty(e.(f{k})), 's5: %s not found in %s', f{k}, rx);
    end
end

function r = beam_radius_(I, dx)
%BEAM_RADIUS_  Half the illuminated extent, from the intensity support.
    m = I > 0.02*max(I(:));
    [rr,cc] = find(m);
    if isempty(rr), r = 0;  return; end
    r = 0.5 * max(max(rr)-min(rr), max(cc)-min(cc)) * dx;
end

function png = series_fig_(t, A, B, P, png)
    f = figure('Visible','off','Color','w','Position',[80 80 1100 760]);
    ax1 = subplot(2,1,1); hold(ax1,'on'); set(ax1,'YScale','log');
    plot(ax1, t, A.wfe, '-', 'Color',[0.75 0.15 0.15], 'LineWidth',1.6);
    plot(ax1, t, B.wfe, '-', 'Color',[0.15 0.35 0.75], 'LineWidth',1.6);
    grid(ax1,'on'); box(ax1,'on');
    ylabel(ax1, sprintf('rms WFE  [waves @ %g nm]', P.lambda_m*1e9));
    legend(ax1, {'uncorrected','corrected'}, 'Location','northwest');
    title(ax1, 'wavefront at the coronagraph exit pupil');
    ax2 = subplot(2,1,2); hold(ax2,'on'); set(ax2,'YScale','log');
    ia = isfinite(A.con);  ib = isfinite(B.con);
    plot(ax2, t(ia), A.con(ia), 'o-', 'Color',[0.75 0.15 0.15], ...
         'LineWidth',1.6, 'MarkerSize',4);
    plot(ax2, t(ib), B.con(ib), 'o-', 'Color',[0.15 0.35 0.75], ...
         'LineWidth',1.6, 'MarkerSize',4);
    grid(ax2,'on'); box(ax2,'on');
    xlabel(ax2,'time  [s]');
    ylabel(ax2, sprintf('dark-zone mean contrast  (%g-%g \\lambda/D)', ...
                        P.co.inner_lamD, P.co.outer_lamD));
    legend(ax2, {'uncorrected','corrected'}, 'Location','northwest');
    title(ax2, sprintf(['contrast at the science plane -- every %d frames, ' ...
                        'each a full propagation + APLC'], P.ts.every));
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function m = finite_(W)
    m = isfinite(W) & W ~= 0 & abs(W) < 1e30;
end
function r = rms_(v), v = v(:); if isempty(v), r = 0; else, r = sqrt(mean(v.^2)); end, end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
function s = gate_(ok), if ok, s = 'PASS'; else, s = 'FAIL'; end, end
