function OUT = cf3e_noapod(over)
%CF3E_NOAPOD  The UNAPODIZED d=1.10 restart ladder (Dave 2026-09-01) --
%   the no-apodizer A/B of the apodizer-trade plan.
%
%   The CF3d configuration EXACTLY, minus the prolate: apod_kind 'none',
%   everything else held (hard occulter, Lyot 0.90, and the CIRCULAR
%   PUPIL STOP c098 STAYS at the apodizer plane -- cf_chain's S0b rule
%   applies it to every pass including the bare normalizer, so it is
%   the pupil, not a coronagraph mask; Dave's explicit requirement).
%   Holding L=0.90 isolates the APODIZER alone; the classical family's
%   own L=0.50 operating point is a separate rung if wanted.
%
%   Ladder: the widened schedule from the start (alphas 1e-10..1e-2,
%   niter 20), target 5e-11, two-stall convergence, wall-clock bounded.
%   Throughput proxy logged up front -- the trade's other axis.
%
%   See also CF3D_DEEPDIG, CF_CHAIN, cf_efc_lib.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    ov = over;  c3 = struct();
    if isfield(ov,'cf3e'), c3 = ov.cf3e;  ov = rmfield(ov,'cf3e'); end
    P = e2e6m_r2_params(ov);
    if ~isfield(c3,'max_rounds'), c3.max_rounds = 40; end
    if ~isfield(c3,'wall_h'),     c3.wall_h = 6.0; end
    if ~isfield(c3,'niter'),      c3.niter = 20; end
    if ~isfield(c3,'alphas'),     c3.alphas = logspace(-10,-2,9); end
    if ~isfield(c3,'target'),     c3.target = 5e-11; end
    if ~isfield(c3,'bump_alphas'), c3.bump_alphas = [1e-4 1e-6]; end
    if ~isfield(c3,'bump_nrec'),   c3.bump_nrec = 8; end
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    lib = cf_efc_lib();

    rep = fullfile(P.outdir, 'cf3e_report.txt');
    logf_(rep, '==== e2e6m CF3e -- UNAPODIZED d=1.10 restart ladder %s', ...
          datestr(now,31)); %#ok<DATST>
    t0 = tic;

    % ---- chain + DMs: cf3d verbatim, apodizer OFF -----------------------
    beam_d = 2 * 0.023771;
    C1 = load(fullfile(P.outdir,'cf1_run.mat'));
    FC = struct();
    for k = 1:numel(C1.OUT.F), FC.(C1.OUT.F(k).key) = C1.OUT.F(k); end
    cfg = FC.apl.cfg;
    ia = find(strcmp(cfg, 'apod_kind'), 1);
    assert(~isempty(ia), 'cf3e: apod_kind not in the apl config');
    cfg{ia+1} = 'none';
    prop_in = fullfile(P.outdir, 'r1_seg_d110_prop.in');
    dmrx    = fullfile(P.outdir, 'r1_seg_d110_dm.in');
    assert(isfile(prop_in) && isfile(dmrx), 'cf3e: run cf3b first (d110 decks)');
    Adm = ctb_dm_rx('rx_in', prop_in, 'rx_out', dmrx, ...
                    'dms', P.dm.names, 'ng', P.dm.ng);
    ch = cf_chain('rx', dmrx, 'model_size', P.dj.model, ...
                  'prolate_iter', P.co.prolate_iter, ...
                  'circ_stop_frac', P.cf.circ_stop_frac, cfg{:});
    assert(~isempty(ch.masks.S), 'cf3e: circular stop MISSING at the apodizer plane');
    assert(isempty(ch.masks.A), 'cf3e: apodizer mask present -- config override failed');
    tag = sprintf('seg_d110_%s', ch.tag);
    dm = cell(1, numel(Adm.ielt));
    for k = 1:numel(dm)
        dm{k} = ctb_dm('ielt', Adm.ielt(k), 'ng', Adm.ng, ...
            'gdx_mm', Adm.gdx_mm(k), 'nact', P.dj.nact, ...
            'beam_d_mm', beam_d, 'pitch_mm', beam_d/P.dj.nact, ...
            'coupling', P.dj.coupling);
        dm{k}.clear();
    end
    dz_idx = find(ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD));
    logf_(rep, 'chain %s | dz %d px | THRU %.3f (apl: 0.091) | niter %d | alphas %.0e..%.0e | budget %.1f h', ...
          tag, numel(dz_idx), ch.thru, c3.niter, c3.alphas(1), c3.alphas(end), c3.wall_h);

    % ---- resume or start ------------------------------------------------
    ckpt = fullfile(P.outdir, 'cf3e_run.mat');
    hist = struct('round',{}, 'c_start',{}, 'c_end',{}, 'alpha',{}, ...
                  'stroke_nm',{}, 'la_floor',{}, 'la_unbound',{}, ...
                  'stroke_unbound',{}, 'minutes',{});
    a = cellfun(@(dd) zeros(dd.nact^2,1), dm, 'UniformOutput', false);
    r0 = 1;
    if isfile(ckpt)
        C = load(ckpt);
        a = C.a;  hist = C.hist;  r0 = numel(hist) + 1;
        if ~isfield(hist, 'la_unbound')      % pre-release checkpoints
            [hist.la_unbound] = deal(NaN);  [hist.stroke_unbound] = deal(NaN);
        end
        logf_(rep, 'RESUMED at round %d (last floor %.3e)', r0, hist(end).c_end);
    end

    % ---- the ladder -----------------------------------------------------
    stall = 0;
    for r = r0:c3.max_rounds
        tr = tic;
        Gc = fullfile(P.outdir, sprintf('cf3e_G_%s_r%d.mat', tag, r));
        [G, ~] = lib.jacobian(ch, dm, a, dz_idx, P, Gc);
        la = lib.linfloor(G, P.cf.stroke_bound_nm);
        laU = lib.linfloor(G, 1e9);          % stroke limit RELEASED (Dave 2026-09-01)
        [a, cvec, alph] = lib.efc(ch, dm, G, a, dz_idx, c3.niter, c3.alphas);
        if isempty(alph)
            logf_(rep, '[round %d] NO ACCEPTED STEP (line search exhausted at %.3e)', ...
                  r, cvec(end));
            alph = NaN;
        end
        % stroke RELEASED (Dave 2026-09-01): non-monotone large-step walk
        ach0 = 1e9 * rms_(cell2mat(cellfun(@(x) x(x~=0), a(:).', ...
                                           'UniformOutput', false).'));
        ptargets = unique(round(min([2 4 8] * max(ach0, 50), 8000)));
        [a, pvec, pinfo] = lib.push(ch, dm, G, a, dz_idx, ptargets);
        if pinfo.c1 < pinfo.c0
            logf_(rep, '[round %d PUSH] %.3e -> %.3e | target %.0f nm | stroke %.1f nm | %d runs', ...
                  r, pinfo.c0, pinfo.c1, pinfo.target, pinfo.stroke, numel(pvec)-1);
            if pinfo.c1 < cvec(end), cvec(end+1) = pinfo.c1; end %#ok<AGROW>
        else
            logf_(rep, '[round %d PUSH] no gain at targets %s nm', r, mat2str(ptargets));
            % beta bump (Dave): accept a worse aggressive step, recover
            [a, binfo] = lib.bump(ch, dm, G, a, dz_idx, c3.bump_alphas, c3.bump_nrec, c3.alphas);
            if binfo.c1 < binfo.c0
                logf_(rep, '[round %d BUMP] %.3e -> kick %.3e -> %.3e | alpha_b %.0e', ...
                      r, binfo.c0, binfo.c_kick, binfo.c1, binfo.alpha_bump);
                if binfo.c1 < cvec(end), cvec(end+1) = binfo.c1; end %#ok<AGROW>
            else
                logf_(rep, '[round %d BUMP] no gain', r);
            end
        end
        ach = 1e9 * rms_(cell2mat(cellfun(@(x) x(x~=0), a(:).', ...
                                          'UniformOutput', false).'));
        hist(end+1) = struct('round',r, 'c_start',cvec(1), 'c_end',cvec(end), ...
            'alpha',alph(end), 'stroke_nm',ach, 'la_floor',la.floor, ...
            'la_unbound',laU.floor, 'stroke_unbound',laU.stroke_nm, ...
            'minutes',toc(tr)/60); %#ok<AGROW>
        save(ckpt, 'a', 'hist', 'tag');
        logf_(rep, '[round %d] %.3e -> %.3e | alpha %.1e | stroke %.1f nm | la(G) %.3e | laU %.3e @ %.0f nm | %.1f min', ...
              r, cvec(1), cvec(end), alph(end), ach, la.floor, ...
              laU.floor, laU.stroke_nm, toc(tr)/60);
        if c3.target > 0 && hist(end).c_end < c3.target
            logf_(rep, 'TARGET REACHED: %.3e < %.1e -- stopping', hist(end).c_end, c3.target);
            break
        end
        if r >= r0+1 && hist(end-1).c_end / hist(end).c_end < 1.03
            stall = stall + 1;
            if stall >= 2
                logf_(rep, 'CONVERGED: <3%% over two consecutive rounds -- floor %.3e', hist(end).c_end);
                break
            end
        else
            stall = 0;
        end
        if toc(t0) > c3.wall_h*3600
            logf_(rep, 'WALL CLOCK: %.1f h -- stopping after round %d (floor %.3e)', ...
                  toc(t0)/3600, r, hist(end).c_end);
            break
        end
    end

    OUT = struct('P',P, 'tag',tag, 'hist',hist, 'a',{a}, 'thru',ch.thru);
    save(ckpt, 'a', 'hist', 'tag', 'OUT');
    logf_(rep, 'CF3e DONE in %.1f h: %.3e -> %.3e over %d rounds (lin-ach %.3e; thru %.3f vs apl 0.091)', ...
          toc(t0)/3600, hist(1).c_start, hist(end).c_end, numel(hist), ...
          hist(end).la_floor, ch.thru);
end

function logf_(rep, varargin)
    s = sprintf(varargin{:});
    fid = fopen(rep, 'a');  fprintf(fid, '%s\n', s);  fclose(fid);
    fprintf('%s\n', s);
end

function v = rms_(x)
    v = sqrt(mean(double(x(:)).^2));
end
