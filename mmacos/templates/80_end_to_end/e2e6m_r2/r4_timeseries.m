function OUT = r4_timeseries(over)
%R4_TIMESERIES  e2e6m round 2: the closed-loop drift series.
%
%   Round 1's S5 upgraded three ways, all on ONE deck (the DM-augmented
%   full-train diffraction deck, so rigid-body drift, MET correction
%   and the coronagraph all describe the same object):
%
%   1. SIX-DOF control.  R0 closed the check on all six rigid-body
%      DOFs (evaluation at the harvest's wf_elt), so control is no
%      longer piston-only.  Every engine WFE/check in this runner
%      traces to ox.wf_elt -- the R0 rule.
%   2. MET-DRIVEN correction -- a PER-FRAME RBCS loop, not a held
%      one-shot.  Each frame the CORRECTED system is measured
%      (m = [dldx; dedx]*(x+u) + noise, the validated linear model --
%      the engine holds met/sensor points rigid, the s6/s7 result),
%      a BLUE estimator (weighted LS + ridge on the SEGMENT state --
%      the Tesch doctrine) estimates the residual and an integrator
%      updates the control: u <- u - g*x_hat, g = 0.5.  Two rejected
%      mechanizations, both MEASURED: a held one-shot injects noise
%      (the frame-2 drift sits below the 1 nm edge noise; corrected
%      leg ended WORSE, 0.056 vs 0.013 waves), and the SEGMENT SLICE
%      of run_met's full-body MMSE gain is NON-CONTRACTIVE (spectral
%      radius 1.154; the engine loop diverged to 19 nm on a 3 nm
%      drift, matching the pure-linear prediction to 3 digits --
%      r4_loop_diag.m).  BLUE+ridge: radius 0.9998, floor 0.37 nm.
%      Stated bound: measurements are simulated from the linear model,
%      not a second engine trace of the truss Rx.
%   3. THE DMs CLOSE THE LOOP.  At every contrast-scored frame of the
%      corrected pass, an EFC step (Tikhonov, the ctb_efc idiom, the
%      engine-measured r3_dmjac G) re-solves the DM commands against
%      the CURRENT dark-zone field, and the contrast series is scored
%      open-loop (DMs flat, uncorrected drift) vs closed-loop
%      (MET-corrected rigid bodies + EFC-held dark zone).
%
%   METRIC TAGS.  WFE: rms OPD at the coronagraph exit pupil
%   (ox.wf_elt), piston removed, waves at 500 nm.  Contrast: dark-zone
%   mean 3-15 lambda/D at the Science plane, Strehl-normalised to the
%   nominal bare peak, at model P.dj.model (=512, the Jacobian grid;
%   ONE grid for G, EFC and scoring -- consistency beats resolution in
%   a series).  Contrast every P.ts.every frames, stated.
%
%   TWO-PASS RULE kept from round 1 (incremental perturbs do not
%   commute; whole uncorrected pass, reload, corrected pass).
%
%   OUT = R4_TIMESERIES()      defaults
%   OUT = R4_TIMESERIES(OVER)  with e2e6m_r2_params overrides
%
%   See also ../e2e6m/s5_timeseries, R3_SENSITIVITIES, R3_MET,
%   R3_DM_JACOBIAN, ctb_efc.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));

    rx = fullfile(P.outdir, 'r1_seg_dm.in');
    S3 = fullfile(P.outdir, 'r3_sens.mat');
    SJ = fullfile(P.outdir, 'r3_dmjac.mat');
    SM = fullfile(P.outdir, 'r3_met.mat');
    assert(isfile(rx), 'r4: %s missing -- run r1_dm', rx);
    assert(isfile(S3), 'r4: %s missing -- run r3_sensitivities', S3);
    assert(isfile(SJ), 'r4: %s missing -- run r3_dm_jacobian', SJ);
    assert(isfile(SM), 'r4: %s missing -- run r3_met', SM);

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m R4 -- the closed-loop drift series');
    L = say_(L, 'deck %s, model %d', rx, P.dj.model);
    L = say_(L, 'WFE: rms OPD at the coronagraph exit pupil (wf_elt), waves @ %g nm', ...
             P.lambda_m*1e9);
    L = say_(L, 'contrast: dark-zone mean %g-%g lambda/D, nominal bare-peak norm,', ...
             P.co.inner_lamD, P.co.outer_lamD);
    L = say_(L, '          every %d frames; EFC re-solved at each scored frame (closed leg)', ...
             P.ts.every);

    % ---- inputs ---------------------------------------------------------
    Z  = load(S3, 'ox');   ox = Z.ox;
    J  = load(SJ);         DJ = J.OUT;
    met = load(SM, 'dldx', 'dedx', 'dxdl', 'dxde');
    wf = ox.wf_elt;
    B    = basis_(ox, P, 0:5);                 % control: ALL SIX (R0)
    uelts = B.elts(:).';
    L = say_(L, '\n[0] control basis: %d rows x %d cols (%d segments x 6 DOF)', ...
             size(B.A,1), size(B.A,2), numel(uelts));
    L = say_(L, '    wf_elt %d (the harvest surface; the R0 rule)', wf);

    % MET estimator: BLUE (weighted LS + ridge) on the SEGMENT state.
    % run_met's H rows are [gauges; edges] over [segs, hub] x 6; the
    % drift moves the segments, so the estimator solves for them alone
    % -- slicing the full-body MMSE gain instead is non-contractive
    % (measured; see the header and r4_loop_diag.m).
    nb = numel(uelts);
    nx = 6*nb;
    H  = [met.dldx(:,1:nx); met.dedx(:,1:nx)];
    sig = [1e-12*ones(size(met.dldx,1),1); ...
           1e-9*ones(size(met.dedx,1),1)];    % gauge 1 pm, edge 1 nm class
    W  = 1./sig.^2;
    Ab = H.' * (W .* H);
    Kb = (Ab + 1e-6*max(diag(Ab))*eye(nx)) \ (H.' .* W.');
    L = say_(L, '    MET: %d gauges + %d edges -> BLUE gain %dx%d; noise 1 pm / 1 nm', ...
             size(met.dldx,1), size(met.dedx,1), size(Kb,1), size(Kb,2));

    % ---- the chain + masks + EFC operator -------------------------------
    N = P.dj.model;
    e = DJ.aug;  %#ok<NASGU>
    em = elt_map_(rx);
    ch = ctb_chain('rx', rx, 'elt', em, 'model_size', N, ...
                   'apodizer', false, ...
                   'fpm', true, 'r_fpm_lamD', P.co.r_occ_lamD, ...
                   'lyot', true, 'r_lyot_frac', P.co.r_lyot_frac);
    Iap = macos.intensity(em.Apodizer);
    Phi = ctb_apod_prolate(N, beam_radius_(Iap,1), P.co.r_occ_lamD, ...
                           'n_iter', P.co.prolate_iter);
    % EFC normal equations, REAL-constrained (the actuator vector is
    % real): da = -(Re(G'G) + lam I) \ Re(G'E) -- the ctb_efc idiom.
    G  = double(DJ.G);
    GtGr = real(G'*G);
    lam_efc = 1e-2 * max(diag(GtGr));
    Fefc = GtGr + lam_efc*eye(size(G,2));
    dm = cell(1, numel(DJ.aug.ielt));
    for k = 1:numel(dm)
        dm{k} = ctb_dm('ielt', DJ.aug.ielt(k), 'ng', DJ.aug.ng, ...
                       'gdx_mm', DJ.aug.gdx_mm(k), 'nact', P.dj.nact, ...
                       'beam_d_mm', DJ.beam_d, 'pitch_mm', DJ.beam_d/P.dj.nact, ...
                       'coupling', P.dj.coupling);
    end
    L = say_(L, '\n[1] chain at N=%d: lambda/D %.3f px, bare peak %.4e; EFC ridge %.3g', ...
             N, ch.lamD_px, ch.peak_bare, lam_efc);

    % ---- the history ----------------------------------------------------
    rng(P.ts.seed);
    [X, tvec] = drift_(P, nb);
    L = say_(L, '\n[2] history: %d frames, dt %g s; walk %g nm / %g nrad per step', ...
             size(X,2), P.ts.dt, P.ts.walk_trans*1e9, P.ts.walk_rot*1e9);

    % ---- pass 1: UNCORRECTED (DMs flat) ---------------------------------
    L = say_(L, '\n[3] pass 1 of 2: UNCORRECTED, DMs flat');
    UN = play_(rx, P, X, uelts, wf, ch, Phi, [], [], [], []);
    L = say_(L, '    WFE %.4f -> %.4f waves; contrast %.3e -> %.3e', ...
             UN.wfe(1), UN.wfe(end), first_(UN.con), last_(UN.con));

    % ---- pass 2: CORRECTED (per-frame RBCS loop) + EFC ------------------
    MET = struct('H', H, 'K', Kb, 'nx', nx, 'sig', sig, 'gain', 0.5);
    L = say_(L, '\n[3] pass 2 of 2: RBCS loop (gain %.2f, per frame) + EFC at scored frames', ...
             MET.gain);
    CR = play_(rx, P, X, uelts, wf, ch, Phi, ...
               struct('G',G,'F',Fefc), dm, DJ, MET);
    L = say_(L, '    WFE %.4f -> %.4f waves; contrast %.3e -> %.3e', ...
             CR.wfe(1), CR.wfe(end), first_(CR.con), last_(CR.con));
    L = say_(L, '    closed-loop WFE floor (median, last 10 frames): %.4f waves', ...
             median(CR.wfe(end-9:end)));
    L = say_(L, '    residual state at the last frame: |x+u| rms %.3g (drift |x| rms %.3g)', ...
             rms_(X(:,end) + CR.Ulast), rms_(X(:,end)));
    if ~isempty(CR.efc_dig)
        L = say_(L, '    EFC first dig (frame 1): %s', ...
                 strjoin(compose('%.3e', CR.efc_dig), ' -> '));
    end

    % ---- the payoff figure ----------------------------------------------
    png = fullfile(P.outdir,'r4_series.png');
    fig_(tvec, UN, CR, P, png);
    L = say_(L, '\n[4] payoff figure: %s', png);

    L = say_(L, '\nR4 DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'r4_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('P',P, 'X',X, 't',tvec, 'unc',UN, 'cor',CR, ...
                 'uelts',uelts, 'wf_elt',wf, 'figure',png, 'text',txt, ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'r4_run.mat'),'OUT','-v7.3');
end

% =========================================================================
function B = basis_(ox, P, dofs)
    ic = find(strcmp(ox.field_names, 'C'), 1);
    B.Wnom = ox.per_field_w_nom_2d{ic};
    B.mnom = fin_(B.Wnom);
    A0     = ox.per_field_dwdx{ic};
    keep = strcmp(ox.kind(:), 'RigidBody') ...
         & ismember(ox.iElt(:),    P.ts.control_elts(:)) ...
         & ismember(ox.dof_idx(:), dofs(:));
    B.cols = find(keep);
    B.elts = unique(ox.iElt(B.cols), 'stable');
    B.dof  = ox.dof_idx(B.cols);
    B.ielt = ox.iElt(B.cols);
    B.A    = A0(:, B.cols);
end

function [X, t] = drift_(P, nb)
%DRIFT_  Round 1's history generator, verbatim: random walk + a common
%   correlated drift, 6 DOF per segment.
    nT = P.ts.frames;  t = (0:nT-1)*P.ts.dt;
    X = zeros(6*nb, nT);
    w = zeros(6*nb, 1);
    dir = randn(6*nb, 1);
    dir = dir / max(norm(dir), realmin);
    for k = 2:nT
        s = zeros(6*nb,1);
        for j = 0:5
            a = P.ts.walk_rot;  if j >= 3, a = P.ts.walk_trans; end
            s(j+1:6:end) = a*randn(nb,1);
        end
        w = w + s;
        g = zeros(6*nb,1);
        for j = 0:5
            a = P.ts.drift_rot;  if j >= 3, a = P.ts.drift_trans; end
            g(j+1:6:end) = a;
        end
        X(:,k) = w + dir .* g .* (t(k)/100);
    end
end

function R = play_(rx, P, X, elts, wf, ch, Phi, EFC, dm, DJ, MET)
%PLAY_  One pass.  WFE every frame at WF (the harvest surface); contrast
%   every P.ts.every frames through the masked chain.  With MET given,
%   the RBCS loop runs per frame: measure the corrected system through
%   the linear model + noise, estimate with the MMSE gain, integrate
%   u <- u - g*x_hat.  With DM/EFC given, an EFC step re-solves the DM
%   commands at each scored frame (5 iterations on the first, 2 after)
%   and the setting is HELD between scored frames.
    nT = size(X,2);
    R = struct('wfe',nan(1,nT), 'con',nan(1,nT), 'dm_rms',nan(1,nT), ...
               'u_rms',nan(1,nT), 'x_rms',nan(1,nT), 'efc_dig',[], ...
               'Ulast',zeros(size(X,1),1));
    macos.init(P.dj.model);
    n = macos.load_rx(rx); %#ok<NASGU>
    a_dm = [];
    if ~isempty(dm), a_dm = zeros(size(EFC.G,2),1); end
    U = zeros(size(X,1),1);
    prev = zeros(size(X,1),1);
    first_score = true;
    for k = 1:nT
        if ~isempty(MET)
            % measure the CORRECTED system as it stands entering frame k
            mv = MET.H*(X(:,k) + U) + MET.sig.*randn(numel(MET.sig),1);
            U  = U - MET.gain*(MET.K*mv);
        end
        d = (X(:,k) + U) - prev;
        apply_(elts, d);
        prev = X(:,k) + U;
        R.u_rms(k) = rms_(U);
        R.x_rms(k) = rms_(prev);
        macos.modify();  macos.trace(wf);
        W = macos.opd();  m = fin_(W);
        v = W(m) - mean(W(m));
        R.wfe(k) = std(v) / P.lambda_m;
        if mod(k-1, P.ts.every) == 0 || k == nT
            if ~isempty(dm)
                % DAMPED, LEAKY EFC.  The segmented-pupil speckle is
                % amplitude-dominated and the 0.15 m DM spacing gives
                % weak Talbot authority (z/z_T ~ 0.4% at 15 lambda/D),
                % so undamped re-solves push strokes along near-null
                % directions and DIVERGE (measured: contrast 1.8e-7 ->
                % 2.9e-6 over 9 re-solves).  Damping gamma bounds the
                % step; the leak mu drains null-space accumulation.
                gam = 0.7;  mu = 0.02;
                ni = 1;  if first_score, ni = 8; end
                dig = nan(1, ni+1);
                for it = 1:ni
                    E  = ch.run_screened(Phi);
                    if first_score
                        I = abs(E).^2 / ch.peak_bare;
                        dzm = ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD);
                        dig(it) = mean(I(dzm));
                    end
                    da = -(EFC.F \ real(EFC.G' * double(E(DJ.dz_idx))));
                    a_dm = (1 - mu)*a_dm + gam*da;
                    seta_(dm, a_dm);
                end
                R.dm_rms(k) = rms_(a_dm);
            end
            E = ch.run_screened(Phi);
            I = abs(E).^2 / ch.peak_bare;
            dzm = ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD);
            R.con(k) = mean(I(dzm));
            if ~isempty(dm) && first_score
                dig(end) = R.con(k);
                R.efc_dig = dig;
            end
            first_score = false;
        end
    end
    R.Ulast = U;
    apply_(elts, -prev);  macos.modify();
    if ~isempty(dm), seta_(dm, zeros(size(a_dm))); end
end

function seta_(dm, a)
    c = 0;
    for k = 1:numel(dm)
        na = dm{k}.nact_active;
        v  = zeros(dm{k}.nact^2, 1);
        v(dm{k}.active) = a(c+1:c+na);
        dm{k}.apply(v);
        c = c + na;
    end
end

function apply_(elts, d)
    for b = 1:numel(elts)
        q = d(6*(b-1)+1 : 6*b);
        if ~any(q), continue; end
        macos.perturb(elts(b), 'rotation', q(1:3), 'translation', q(4:6), ...
                      'frame','local');
    end
end

function e = elt_map_(rx)
    nm = regexp(fileread(rx), '^\s*EltName=\s*(\S+)', 'tokens','lineanchors');
    nm = cellfun(@(c) c{1}, nm, 'UniformOutput', false);
    at = @(s) find(strcmp(nm, s), 1);
    e = struct('DM1',at('DM1'), 'DM2',at('DM2'), ...
               'Apodizer',at('Apodizer'), 'FPM',at('FPM'), ...
               'Lyot',at('Lyot'), 'ExitPupil',at('ExitPupil'), ...
               'FPA',at('Science'));
end

function png = fig_(t, UN, CR, P, png)
    f = figure('Visible','off','Color','w','Position',[80 80 1100 1020]);
    ax0 = subplot(3,1,1); hold(ax0,'on'); set(ax0,'YScale','log');
    plot(ax0, t, UN.x_rms*1e9, '-', 'Color',[0.75 0.15 0.15], 'LineWidth',1.6);
    plot(ax0, t, CR.x_rms*1e9, '-', 'Color',[0.15 0.35 0.75], 'LineWidth',1.6);
    grid(ax0,'on'); box(ax0,'on');
    ylabel(ax0,'rigid-body state rms  [nm, nrad]');
    legend(ax0, {'drift (uncontrolled)','residual under the RBCS loop'}, ...
           'Location','northwest');
    title(ax0, 'segment rigid-body state (19 segments x 6 DOF)');
    ax1 = subplot(3,1,2); hold(ax1,'on'); set(ax1,'YScale','log');
    plot(ax1, t, UN.wfe, '-', 'Color',[0.75 0.15 0.15], 'LineWidth',1.6);
    plot(ax1, t, CR.wfe, '-', 'Color',[0.15 0.35 0.75], 'LineWidth',1.6);
    grid(ax1,'on'); box(ax1,'on');
    ylabel(ax1, sprintf('rms WFE  [waves @ %g nm]', P.lambda_m*1e9));
    legend(ax1, {'uncorrected', ...
        'closed loop -- INCLUDES the deliberate EFC pupil shaping'}, ...
           'Location','northwest');
    title(ax1, ['wavefront at the coronagraph exit pupil -- the closed-loop ' ...
        'line carries the DM strokes EFC spends buying contrast']);
    ax2 = subplot(3,1,3); hold(ax2,'on'); set(ax2,'YScale','log');
    ia = isfinite(UN.con);  ib = isfinite(CR.con);
    plot(ax2, t(ia), UN.con(ia), 'o-', 'Color',[0.75 0.15 0.15], ...
         'LineWidth',1.6, 'MarkerSize',4);
    plot(ax2, t(ib), CR.con(ib), 'o-', 'Color',[0.15 0.35 0.75], ...
         'LineWidth',1.6, 'MarkerSize',4);
    grid(ax2,'on'); box(ax2,'on');
    xlabel(ax2,'time  [s]');
    ylabel(ax2, sprintf('dark-zone mean  (%g-%g \\lambda/D)', ...
                        P.co.inner_lamD, P.co.outer_lamD));
    legend(ax2, {'open loop (DMs flat)','closed loop (RBCS + damped EFC)'}, ...
           'Location','northwest');
    title(ax2, sprintf(['contrast at the Science plane -- every %d frames; ' ...
                        'EFC re-solved per scored frame'], P.ts.every));
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function r = beam_radius_(I, dx)
    m = I > 0.02*max(I(:));
    [rr,cc] = find(m);
    if isempty(rr), r = 0;  return; end
    r = 0.5 * max(max(rr)-min(rr), max(cc)-min(cc)) * dx;
end
function v = first_(c), v = c(find(isfinite(c),1)); end
function v = last_(c),  v = c(find(isfinite(c),1,'last')); end
function m = fin_(W)
    m = isfinite(W) & W ~= 0 & abs(W) < 1e30;
end
function r = rms_(v), v = v(:); if isempty(v), r = 0; else, r = sqrt(mean(v.^2)); end, end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
