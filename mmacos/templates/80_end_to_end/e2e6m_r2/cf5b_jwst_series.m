function OUT = cf5b_jwst_series(over)
%CF5B_JWST_SERIES  The drift series re-posed at the CF3d operating point
%   (Dave 2026-09-01): START from the new solution, drift ~10 nm/hr,
%   time scale JWST-like (24 h at 30-min frames), on the d=1.10 m apl
%   train.
%
%   Differences from R4_TIMESERIES, all deliberate:
%   - Deck/chain: the CF3d configuration exactly -- r1_seg_d110_dm.in
%     through cf_chain with the apl masks + circularized stop, so the
%     contrast normalizer is CF3d's own and frame 1 must reproduce the
%     1.133e-9 dug floor (a built-in gate).
%   - BOTH passes start at the DUG DM state (cf3d_run.mat).  The open
%     leg HOLDS the DMs frozen and lets the drift decay the dark hole;
%     the closed leg runs the R4 RBCS loop (BLUE + ridge, gain 0.5 per
%     frame, all six DOFs) plus a GUARDED EFC hold at scored frames.
%   - EFC hold, not dig: G is the CF3d round-13 cache (measured AT the
%     dug state), one damped step per scored frame, NO leak (the leak
%     would drain the dug solution), and the step is accept-reject
%     guarded (score before/after, revert if worse -- the ladder's
%     monotone-acceptance lesson, since there is no line search here).
%   - History: dt 1800 s x 49 frames = 24 h; ramp 10 nm/hr (10 nrad/hr
%     rotations) along a random direction + a 0.5 nm/step walk.  The
%     ~10 nm/hr number is the SEGMENT-STATE rate (the JWST-experience
%     disturbance class); the induced wavefront rate is measured and
%     reported, not assumed.
%
%   Writes cf5b_report.txt, cf5b_jwst.png, cf5b_run.mat.
%
%   See also R4_TIMESERIES, CF3D_DEEPDIG, CF_CHAIN, R3_MET.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    c5 = struct();
    if isfield(over,'cf5b'), c5 = over.cf5b;  over = rmfield(over,'cf5b'); end
    def = struct('frames',49, 'dt',1800, 'every',2, ...
                 'rate_trans',10e-9/3600, 'rate_rot',10e-9/3600, ...  % m,rad per s
                 'walk_trans',0.5e-9, 'walk_rot',0.5e-9, ...          % per step
                 'gain',0.5, 'gam',0.7);
    for f = fieldnames(def).', if ~isfield(c5,f{1}), c5.(f{1}) = def.(f{1}); end, end
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m CF5b -- JWST-class drift on the dug state');
    L = say_(L, 'start = CF3d solution; drift %.1f nm/hr (+%.1f nrad/hr) ramp + %.1f nm/step walk', ...
             c5.rate_trans*3600e9, c5.rate_rot*3600e9, c5.walk_trans*1e9);
    L = say_(L, 'history %d frames x %g s = %.1f h; contrast/EFC every %d frames', ...
             c5.frames, c5.dt, c5.frames*c5.dt/3600, c5.every);

    % ---- chain + DMs: the CF3d construction verbatim --------------------
    beam_d = 2 * 0.023771;
    C1 = load(fullfile(P.outdir,'cf1_run.mat'));
    FC = struct();
    for k = 1:numel(C1.OUT.F), FC.(C1.OUT.F(k).key) = C1.OUT.F(k); end
    rx   = fullfile(P.outdir, 'r1_seg_d110_dm.in');
    assert(isfile(rx), 'cf5b: run cf3b first (d110 decks)');
    Adm = ctb_dm_rx('rx_in', fullfile(P.outdir,'r1_seg_d110_prop.in'), ...
                    'rx_out', rx, 'dms', P.dm.names, 'ng', P.dm.ng);
    ch = cf_chain('rx', rx, 'model_size', P.dj.model, ...
                  'prolate_iter', P.co.prolate_iter, ...
                  'circ_stop_frac', P.cf.circ_stop_frac, FC.apl.cfg{:});
    dm = cell(1, numel(Adm.ielt));
    for k = 1:numel(dm)
        dm{k} = ctb_dm('ielt', Adm.ielt(k), 'ng', Adm.ng, ...
            'gdx_mm', Adm.gdx_mm(k), 'nact', P.dj.nact, ...
            'beam_d_mm', beam_d, 'pitch_mm', beam_d/P.dj.nact, ...
            'coupling', P.dj.coupling);
        dm{k}.clear();
    end
    dz_idx = find(ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD));
    wf = ch.elt.ExitPupil;
    L = say_(L, 'chain %s | dz %d px | WFE at ExitPupil (elt %d)', ch.tag, numel(dz_idx), wf);

    % ---- the dug state + its Jacobian -----------------------------------
    CK = load(fullfile(P.outdir, 'cf3d_run.mat'));
    a_dug = CK.a;
    Gc = fullfile(P.outdir, sprintf('cf3d_G_seg_d110_%s_r13.mat', ch.tag));
    assert(isfile(Gc), 'cf5b: %s missing (the round-13 dug-state G)', Gc);
    J = load(Gc);
    av = cell2mat(cellfun(@(x) x(:), a_dug(:).', 'UniformOutput', false));
    jv = cell2mat(cellfun(@(x) x(:), J.a0(:).',  'UniformOutput', false));
    assert(max(abs(av(:)-jv(:))) < 1e-15, 'cf5b: G was measured about a different state');
    assert(isequal(J.dz_idx(:), dz_idx(:)), 'cf5b: dz_idx mismatch vs the G cache');
    G = double(J.G);
    GtGr = real(G'*G);
    F = GtGr + 1e-2*max(diag(GtGr))*eye(size(G,2));
    L = say_(L, 'G: %s (%d cols, measured about the dug state; EFC ridge 1e-2 rel)', ...
             Gc, size(G,2));

    % ---- MET (the R4 BLUE estimator, verbatim) --------------------------
    SM = load(fullfile(P.outdir,'r3_met.mat'), 'dldx', 'dedx');
    uelts = P.ts.control_elts(:).';
    nb = numel(uelts);  nx = 6*nb;
    H  = [SM.dldx(:,1:nx); SM.dedx(:,1:nx)];
    sig = [1e-12*ones(size(SM.dldx,1),1); 1e-9*ones(size(SM.dedx,1),1)];
    W  = 1./sig.^2;
    Ab = H.' * (W .* H);
    Kb = (Ab + 1e-6*max(diag(Ab))*eye(nx)) \ (H.' .* W.');
    L = say_(L, 'MET: %d gauges + %d edges -> BLUE gain; noise 1 pm / 1 nm; loop gain %.2f', ...
             size(SM.dldx,1), size(SM.dedx,1), c5.gain);

    % ---- the history ----------------------------------------------------
    rng(P.ts.seed);
    nT = c5.frames;  tvec = (0:nT-1)*c5.dt;
    X = zeros(nx, nT);
    w = zeros(nx, 1);
    dir = randn(nx,1);  dir = dir/max(norm(dir),realmin);
    for k = 2:nT
        s = zeros(nx,1);
        for j = 0:5
            a = c5.walk_rot;  if j >= 3, a = c5.walk_trans; end
            s(j+1:6:end) = a*randn(nb,1);
        end
        w = w + s;
        g = zeros(nx,1);
        for j = 0:5
            a = c5.rate_rot;  if j >= 3, a = c5.rate_trans; end
            g(j+1:6:end) = a;
        end
        X(:,k) = w + dir .* g * tvec(k);
    end
    L = say_(L, 'history: |x| rms at 24 h = %.1f nm-class (ramp-dominated)', rms_(X(:,end))*1e9);

    % ---- pass 1: OPEN (DMs frozen at the dug state) ---------------------
    L = say_(L, '\npass 1 of 2: OPEN -- DMs frozen dug, drift uncorrected');
    UN = play_(rx, P, c5, X, uelts, wf, ch, dz_idx, a_dug, dm, [], [], []);
    L = say_(L, '   contrast %.3e -> %.3e | WFE %.4f -> %.4f waves', ...
             first_(UN.con), last_(UN.con), UN.wfe(1), UN.wfe(end));

    % ---- pass 2: CLOSED (RBCS + guarded EFC hold) -----------------------
    MET = struct('H',H, 'K',Kb, 'sig',sig, 'gain',c5.gain);
    EFC = struct('G',G, 'F',F, 'col_dm',J.col_dm, 'col_act',J.col_act, ...
                 'gam',c5.gam);
    L = say_(L, 'pass 2 of 2: CLOSED -- RBCS per frame + guarded EFC hold at scored frames');
    CR = play_(rx, P, c5, X, uelts, wf, ch, dz_idx, a_dug, dm, MET, EFC, []);
    L = say_(L, '   contrast %.3e -> %.3e | WFE %.4f -> %.4f waves', ...
             first_(CR.con), last_(CR.con), CR.wfe(1), CR.wfe(end));
    L = say_(L, '   state residual at 24 h: |x+u| rms %.3g nm vs drift %.3g nm', ...
             rms_(X(:,end)+CR.Ulast)*1e9, rms_(X(:,end))*1e9);
    L = say_(L, '   EFC steps: %d taken, %d reverted by the guard', ...
             CR.efc_taken, CR.efc_reverted);
    L = say_(L, '   closed-loop contrast (median, scored frames after hour 2): %.3e', ...
             median(CR.con(isfinite(CR.con) & tvec > 7200)));

    % ---- gate: frame 1 must reproduce the CF3d floor --------------------
    L = say_(L, '\nGATE frame-1 closed contrast %.3e vs CF3d floor 1.133e-09 (ratio %.3f)', ...
             first_(CR.con), first_(CR.con)/1.133e-9);

    % ---- figure ---------------------------------------------------------
    png = fullfile(P.outdir,'cf5b_jwst.png');
    fig_(tvec/3600, UN, CR, P, c5, png);
    L = say_(L, 'figure: %s', png);

    L = say_(L, '\nCF5b DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf5b_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('P',P, 'c5',c5, 'X',X, 't',tvec, 'unc',UN, 'cor',CR, ...
                 'figure',png, 'text',txt, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf5b_run.mat'),'OUT','-v7.3');
end

% =========================================================================
function R = play_(rx, P, c5, X, elts, wf, ch, dz_idx, a_dug, dm, MET, EFC, ~)
    nT = size(X,2);
    R = struct('wfe',nan(1,nT), 'con',nan(1,nT), 'u_rms',nan(1,nT), ...
               'x_rms',nan(1,nT), 'dm_rms',nan(1,nT), ...
               'Ulast',zeros(size(X,1),1), 'efc_taken',0, 'efc_reverted',0);
    macos.init(P.dj.model);
    macos.load_rx(rx);
    a = a_dug;                                   % both passes start dug
    seta_(dm, a);
    U = zeros(size(X,1),1);
    prev = zeros(size(X,1),1);
    for k = 1:nT
        if ~isempty(MET)
            mv = MET.H*(X(:,k) + U) + MET.sig.*randn(numel(MET.sig),1);
            U  = U - MET.gain*(MET.K*mv);
        end
        d = (X(:,k) + U) - prev;
        apply_(elts, d);
        prev = X(:,k) + U;
        R.u_rms(k) = rms_(U);  R.x_rms(k) = rms_(prev);
        macos.modify();  macos.trace(wf);
        Wo = macos.opd();  m = fin_(Wo);
        v = Wo(m) - mean(Wo(m));
        R.wfe(k) = std(v) / P.lambda_m;
        if mod(k-1, c5.every) == 0 || k == nT
            E = ch.run();
            c_now = dzmean_(E, ch, dz_idx);
            if ~isempty(EFC)
                % one damped step about the dug-state G, accept-reject
                da = -(EFC.F \ real(EFC.G' * double(E(dz_idx))));
                a_try = a;
                for c = 1:numel(da)
                    kdm = EFC.col_dm(c);  ka = EFC.col_act(c);
                    a_try{kdm}(ka) = a_try{kdm}(ka) + EFC.gam*da(c);
                end
                seta_(dm, a_try);
                E2 = ch.run();
                c_try = dzmean_(E2, ch, dz_idx);
                if c_try < c_now
                    a = a_try;  c_now = c_try;
                    R.efc_taken = R.efc_taken + 1;
                else
                    seta_(dm, a);                % revert
                    R.efc_reverted = R.efc_reverted + 1;
                end
            end
            R.con(k) = c_now;
            R.dm_rms(k) = 1e9*rms_(cell2mat(cellfun(@(x) x(x~=0), ...
                a(:).', 'UniformOutput', false).'));
        end
    end
    R.Ulast = U;
    apply_(elts, -prev);  macos.modify();
    seta_(dm, cellfun(@(x) zeros(size(x)), a, 'UniformOutput', false));
end

function c = dzmean_(E, ch, dz_idx)
    I = abs(E).^2 / ch.peak_bare;
    c = mean(I(dz_idx));
end

function seta_(dm, a)
    for k = 1:numel(dm), dm{k}.apply(a{k}); end
end

function apply_(elts, d)
    for b = 1:numel(elts)
        q = d(6*(b-1)+1 : 6*b);
        if ~any(q), continue; end
        macos.perturb(elts(b), 'rotation', q(1:3), 'translation', q(4:6), ...
                      'frame','local');
    end
end

function fig_(th, UN, CR, P, c5, png)
    f = figure('Visible','off','Color','w','Position',[80 80 1100 1020]);
    ax0 = subplot(3,1,1); hold(ax0,'on'); set(ax0,'YScale','log');
    plot(ax0, th, UN.x_rms*1e9, '-', 'Color',[0.75 0.15 0.15], 'LineWidth',1.6);
    plot(ax0, th, CR.x_rms*1e9, '-', 'Color',[0.15 0.35 0.75], 'LineWidth',1.6);
    grid(ax0,'on'); box(ax0,'on');
    ylabel(ax0,'rigid-body state rms  [nm, nrad]');
    legend(ax0, {'drift (uncontrolled)','residual under the RBCS loop'}, ...
           'Location','northwest');
    title(ax0, sprintf('segment rigid-body state -- %.0f nm/hr JWST-class drift over %.0f h', ...
          c5.rate_trans*3600e9, th(end)));
    ax1 = subplot(3,1,2); hold(ax1,'on'); set(ax1,'YScale','log');
    plot(ax1, th, UN.wfe, '-', 'Color',[0.75 0.15 0.15], 'LineWidth',1.6);
    plot(ax1, th, CR.wfe, '-', 'Color',[0.15 0.35 0.75], 'LineWidth',1.6);
    grid(ax1,'on'); box(ax1,'on');
    ylabel(ax1, sprintf('rms WFE  [waves @ %g nm]', P.lambda_m*1e9));
    legend(ax1, {'uncorrected','closed loop -- carries the deliberate EFC shaping'}, ...
           'Location','northwest');
    title(ax1, 'wavefront at the coronagraph exit pupil');
    ax2 = subplot(3,1,3); hold(ax2,'on'); set(ax2,'YScale','log');
    ia = isfinite(UN.con);  ib = isfinite(CR.con);
    plot(ax2, th(ia), UN.con(ia), 'o-', 'Color',[0.75 0.15 0.15], ...
         'LineWidth',1.6, 'MarkerSize',4);
    plot(ax2, th(ib), CR.con(ib), 'o-', 'Color',[0.15 0.35 0.75], ...
         'LineWidth',1.6, 'MarkerSize',4);
    yline(ax2, 1.133e-9, ':', 'CF3d floor', 'Color',[0.4 0.4 0.4]);
    grid(ax2,'on'); box(ax2,'on');
    xlabel(ax2,'time  [hours]');
    ylabel(ax2, sprintf('dark-zone mean  (%g-%g \\lambda/D)', ...
                        P.co.inner_lamD, P.co.outer_lamD));
    legend(ax2, {'open loop (DMs frozen dug)','closed loop (RBCS + guarded EFC hold)'}, ...
           'Location','northwest');
    title(ax2, 'contrast at the Science plane -- both passes START at the dug dark hole');
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function v = first_(c), v = c(find(isfinite(c),1)); end
function v = last_(c),  v = c(find(isfinite(c),1,'last')); end
function m = fin_(W),   m = isfinite(W) & W ~= 0 & abs(W) < 1e30; end
function r = rms_(v), v = v(:); if isempty(v), r = 0; else, r = sqrt(mean(v.^2)); end, end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
