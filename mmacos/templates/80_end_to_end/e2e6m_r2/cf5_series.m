function OUT = cf5_series(over)
%CF5_SERIES  Coronagraph-family campaign S5: the drift series, toned down 10x.
%
%   Dave: "The time-series sequence also needs toning down by 10x."
%   Two legs, one process (N = P.dj.model):
%
%   LEG A (continuity): r4_timeseries VERBATIM -- same estimator (BLUE +
%   ridge on the segment state, gain 0.5), same damped leaky EFC
%   (gamma 0.7, mu 0.02), same two-pass rule -- with every drift
%   amplitude divided by 10 (walk 0.03 nm/step, correlated drift
%   0.6 nm per 100 s -> end-state ~0.3 nm rms).  This REGENERATES
%   r4_report.txt / r4_series.png / r4_run.mat as the deck's series.
%
%   LEG B (the winner): the same mechanization on the S2/S3 winner's
%   STOPPED chain (cf_chain + the winner's S2 Jacobian, stamp-verified).
%   The EFC operator is r4's real-constrained Tikhonov normal equations
%   on the winner's G.
%
%   NOISE-REGIME HONESTY (the brief): at 10x smaller drift the state
%   sits BELOW the 1 nm edge-sensor noise for most of the soak; the
%   report carries the measured estimator behavior (residual vs drift
%   vs the noise floor) rather than a silent retune -- nothing is
%   retuned.
%
%   over.cf5 fields: family ('' = the S2 winner by relin floor),
%   scale (0.1), legs ({'baseline','winner'}).
%
%   See also R4_TIMESERIES, CF2_EFC, CF_CHAIN.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    ov = over;  cf5 = struct();
    if isfield(ov, 'cf5'), cf5 = ov.cf5;  ov = rmfield(ov, 'cf5'); end
    P = e2e6m_r2_params(ov);
    if ~isfield(cf5,'family'), cf5.family = '';                    end
    if ~isfield(cf5,'scale'),  cf5.scale = 0.1;                    end
    if ~isfield(cf5,'legs'),   cf5.legs = {'baseline','winner'};   end
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));

    ts10 = P.ts;
    ts10.walk_trans  = ts10.walk_trans  * cf5.scale;
    ts10.walk_rot    = ts10.walk_rot    * cf5.scale;
    ts10.drift_trans = ts10.drift_trans * cf5.scale;
    ts10.drift_rot   = ts10.drift_rot   * cf5.scale;

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m CF5 -- the drift series at %.0fx smaller drift', ...
             1/cf5.scale);
    L = say_(L, 'walk %g nm / %g nrad per step; drift %g nm / %g nrad per 100 s', ...
             ts10.walk_trans*1e9, ts10.walk_rot*1e9, ...
             ts10.drift_trans*1e9, ts10.drift_rot*1e9);
    L = say_(L, 'edge noise UNCHANGED (1 pm gauge / 1 nm edge): the drift now sits BELOW');
    L = say_(L, 'the edge noise for most of the soak -- measured behavior below, NOT retuned.');

    R = struct();

    % ---- LEG A: r4_timeseries verbatim at scale --------------------------
    if any(strcmp(cf5.legs, 'baseline'))
        L = say_(L, '\n[A] r4_timeseries (R1-baseline chain, no stop), drift x%.2f:', cf5.scale);
        RA = r4_timeseries(struct('ts', ts10));
        R.baseline = summarize_(RA);
        L = report_leg_(L, R.baseline, 'baseline (r4 chain)');
    end

    % ---- LEG B: the winner's stopped chain -------------------------------
    if any(strcmp(cf5.legs, 'winner'))
        C2 = load(fullfile(P.outdir,'cf2_run.mat'));
        if isempty(cf5.family)
            rel = structfun(@(r) r.c_relin, C2.OUT.R);
            [~, iw] = min(rel);
            cf5.family = C2.OUT.keys{iw};
        end
        key = cf5.family;
        C1 = load(fullfile(P.outdir,'cf1_run.mat'));
        fam = C1.OUT.F(strcmp({C1.OUT.F.key}, key));
        L = say_(L, '\n[B] winner leg: %s (stopped chain), drift x%.2f:', fam.name, cf5.scale);

        rx = fullfile(P.outdir, 'r1_seg_dm.in');
        S3 = load(fullfile(P.outdir,'r3_sens.mat'), 'ox');  ox = S3.ox;
        met = load(fullfile(P.outdir,'r3_met.mat'), 'dldx', 'dedx');
        A  = load(fullfile(P.outdir,'r1_dm_run.mat'));  aug = A.OUT.aug;
        beam_d = 2 * 0.023771;

        ch = cf_chain('rx', rx, 'model_size', P.dj.model, ...
                      'prolate_iter', P.co.prolate_iter, ...
                      'circ_stop_frac', P.cf.circ_stop_frac, fam.cfg{:});
        jf = fullfile(P.outdir, sprintf('cf2_G_%s.mat', ch.tag));
        assert(isfile(jf), 'cf5: %s missing -- run cf2_efc first', jf);
        J = load(jf);
        ctb_jac_check(J, ch.config, jf);
        lb = cf_efc_lib();  lb.stamp_parity(J, ch.config, jf);
        G = double(J.G);
        GtGr = real(G'*G);
        lam_efc = 1e-2 * max(diag(GtGr));
        Fefc = GtGr + lam_efc*eye(size(G,2));
        dm = cell(1, numel(aug.ielt));
        for k = 1:numel(dm)
            dm{k} = ctb_dm('ielt', aug.ielt(k), 'ng', aug.ng, ...
                           'gdx_mm', aug.gdx_mm(k), 'nact', P.dj.nact, ...
                           'beam_d_mm', beam_d, 'pitch_mm', beam_d/P.dj.nact, ...
                           'coupling', P.dj.coupling);
            dm{k}.clear();
        end

        % control basis + BLUE estimator: r4's [0]-block verbatim
        B = basis_(ox, P, 0:5);
        uelts = B.elts(:).';
        wf = ox.wf_elt;
        nb = numel(uelts);  nx = 6*nb;
        H  = [met.dldx(:,1:nx); met.dedx(:,1:nx)];
        sig = [1e-12*ones(size(met.dldx,1),1); 1e-9*ones(size(met.dedx,1),1)];
        W  = 1./sig.^2;
        Ab = H.' * (W .* H);
        Kb = (Ab + 1e-6*max(diag(Ab))*eye(nx)) \ (H.' .* W.');
        MET = struct('H', H, 'K', Kb, 'nx', nx, 'sig', sig, 'gain', 0.5);

        Pts = P;  Pts.ts = ts10;
        rng(Pts.ts.seed);
        [X, tvec] = drift_(Pts, nb);
        L = say_(L, '    chain %s | lambda/D %.3f px | peak_bare %.4e | G %d cols (ridge %.3g)', ...
                 ch.tag, ch.lamD_px, ch.peak_bare, size(G,2), lam_efc);

        UN = play_(rx, Pts, X, uelts, wf, ch, [], [], [], []);
        L = say_(L, '    open loop:   WFE %.4f -> %.4f waves; contrast %.3e -> %.3e', ...
                 UN.wfe(1), UN.wfe(end), first_(UN.con), last_(UN.con));
        CR = play_(rx, Pts, X, uelts, wf, ch, ...
                   struct('G',G,'F',Fefc,'dz',J.dz_idx), dm, [], MET);
        L = say_(L, '    closed loop: WFE %.4f -> %.4f waves; contrast %.3e -> %.3e', ...
                 CR.wfe(1), CR.wfe(end), first_(CR.con), last_(CR.con));
        L = say_(L, '    residual |x+u| rms %.3g vs drift |x| rms %.3g (edge noise 1e-9)', ...
                 rms_(X(:,end) + CR.Ulast), rms_(X(:,end)));
        if ~isempty(CR.efc_dig)
            L = say_(L, '    EFC first dig: %s', strjoin(compose('%.3e', CR.efc_dig), ' -> '));
        end

        png = fullfile(P.outdir, 'cf5_series.png');
        fig_(tvec, UN, CR, Pts, fam, png);
        L = say_(L, '    figure: %s', png);
        R.winner = struct('family', key, 'tag', ch.tag, 'X', X, 't', tvec, ...
                          'unc', UN, 'cor', CR, 'figure', png);
    end

    L = say_(L, '\nCF5 DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf5_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('P',P, 'ts10',ts10, 'R',R, 'text',txt, ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf5_run.mat'),'OUT','-v7.3');
end

% =========================================================================
function s = summarize_(RA)
    s = struct('wfe_unc',[RA.unc.wfe(1) RA.unc.wfe(end)], ...
               'wfe_cor',[RA.cor.wfe(1) RA.cor.wfe(end)], ...
               'con_unc',[first_(RA.unc.con) last_(RA.unc.con)], ...
               'con_cor',[first_(RA.cor.con) last_(RA.cor.con)], ...
               'x_end', rms_(RA.X(:,end)), ...
               'res_end', rms_(RA.X(:,end) + RA.cor.Ulast));
end

function L = report_leg_(L, s, name)
    L = say_(L, '    %s:', name);
    L = say_(L, '      open:   WFE %.4f -> %.4f waves; contrast %.3e -> %.3e', ...
             s.wfe_unc(1), s.wfe_unc(2), s.con_unc(1), s.con_unc(2));
    L = say_(L, '      closed: WFE %.4f -> %.4f waves; contrast %.3e -> %.3e', ...
             s.wfe_cor(1), s.wfe_cor(2), s.con_cor(1), s.con_cor(2));
    L = say_(L, '      drift end-state %.3g rms; closed-loop residual %.3g rms', ...
             s.x_end, s.res_end);
    if s.res_end > s.x_end
        L = say_(L, '      NOISE REGIME: the estimator residual EXCEEDS the drift -- the');
        L = say_(L, '      0.3 nm state sits below the 1 nm edge noise; the RBCS loop is');
        L = say_(L, '      noise-fed here (measured, not retuned; the EFC leg still holds');
        L = say_(L, '      contrast against what drift there is).');
    end
end

% ---- r4_timeseries internals, carried for the winner leg ---------------
function B = basis_(ox, P, dofs)
    keep = strcmp(ox.kind(:), 'RigidBody') ...
         & ismember(ox.iElt(:),    P.ts.control_elts(:)) ...
         & ismember(ox.dof_idx(:), dofs(:));
    B.cols = find(keep);
    B.elts = unique(ox.iElt(B.cols), 'stable');
end

function [X, t] = drift_(P, nb)
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

function R = play_(rx, P, X, elts, wf, ch, EFC, dm, DJ, MET) %#ok<INUSD>
%PLAY_  r4_timeseries' pass runner on cf_chain: WFE every frame at the
%   harvest surface; contrast every P.ts.every frames through ch.run()
%   (masks + stop internal); damped leaky EFC at scored frames.
    nT = size(X,2);
    R = struct('wfe',nan(1,nT), 'con',nan(1,nT), 'dm_rms',nan(1,nT), ...
               'u_rms',nan(1,nT), 'x_rms',nan(1,nT), 'efc_dig',[], ...
               'Ulast',zeros(size(X,1),1));
    macos.init(P.dj.model);
    macos.load_rx(rx);
    a_dm = [];
    if ~isempty(dm), a_dm = zeros(size(EFC.G,2),1); end
    U = zeros(size(X,1),1);
    prev = zeros(size(X,1),1);
    first_score = true;
    for k = 1:nT
        if ~isempty(MET)
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
                gam = 0.7;  mu = 0.02;              % r4's damped leaky EFC
                ni = 1;  if first_score, ni = 8; end
                dig = nan(1, ni+1);
                for it = 1:ni
                    E  = ch.run();
                    if first_score
                        I = abs(E).^2 / ch.peak_bare;
                        dzm = ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD);
                        dig(it) = mean(I(dzm));
                    end
                    da = -(EFC.F \ real(EFC.G' * double(E(EFC.dz))));
                    a_dm = (1 - mu)*a_dm + gam*da;
                    seta_(dm, a_dm);
                end
                R.dm_rms(k) = rms_(a_dm);
            end
            E = ch.run();
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

function fig_(t, UN, CR, P, fam, png)
    f = figure('Visible','off','Color','w','Position',[80 80 1100 1020]);
    ax0 = subplot(3,1,1); hold(ax0,'on'); set(ax0,'YScale','log');
    plot(ax0, t, UN.x_rms*1e9, '-', 'Color',[0.75 0.15 0.15], 'LineWidth',1.6);
    plot(ax0, t, CR.x_rms*1e9, '-', 'Color',[0.15 0.35 0.75], 'LineWidth',1.6);
    yline(ax0, 1, ':', '1 nm edge noise', 'Color',[0.3 0.3 0.3]);
    grid(ax0,'on'); box(ax0,'on');
    ylabel(ax0,'rigid-body state rms  [nm, nrad]');
    legend(ax0, {'drift (uncontrolled)','residual under the RBCS loop'}, ...
           'Location','southeast');
    title(ax0, sprintf('segment rigid-body state -- drift toned down 10x (%s leg)', fam.name));
    ax1 = subplot(3,1,2); hold(ax1,'on'); set(ax1,'YScale','log');
    plot(ax1, t, UN.wfe, '-', 'Color',[0.75 0.15 0.15], 'LineWidth',1.6);
    plot(ax1, t, CR.wfe, '-', 'Color',[0.15 0.35 0.75], 'LineWidth',1.6);
    grid(ax1,'on'); box(ax1,'on');
    ylabel(ax1, sprintf('rms WFE  [waves @ %g nm]', P.lambda_m*1e9));
    legend(ax1, {'uncorrected', ...
        'closed loop -- INCLUDES the deliberate EFC pupil shaping'}, ...
           'Location','northwest');
    title(ax1, 'wavefront at the coronagraph exit pupil');
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
    title(ax2, 'contrast at the Science plane');
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
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
