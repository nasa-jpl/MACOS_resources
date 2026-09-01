function OUT = cf3d_deepdig(over)
%CF3D_DEEPDIG  The restart ladder at d = 1.10 m (Dave 2026-08-31 pm).
%
%   FALCO-style rinse-and-repeat on the apodized-Lyot leg of the
%   d=1.10 m train: EFC to a floor, RELINEARIZE about the dug state,
%   repeat, with a widened Tikhonov schedule (logspace(-8,-2,7) vs
%   cf3b's -6..-2) and 20 iterations per round.
%
%   WHY: cf3b measured lin-ach 6.2e-11 at d=1.10 while its single-relin
%   dig stalled at 9.0e-7 -- four orders above the linear floor.  At
%   this spacing the bottleneck is the CONTROLLER, not the substrate
%   (the reverse of d=0.15, where relin == lin-ach).  Colleagues'
%   FALCO runs on the same pupil class dig much deeper; this ladder is
%   the minimal in-record version of that machinery.
%
%   Checkpointed per round to cf3d_run.mat (resumable: re-run resumes);
%   report appended per round to cf3d_report.txt (tail -f friendly).
%   Wall-clock bounded (default 6.0 h) -- hand back whatever rounds
%   completed; every round is a finished floor.
%
%   OUT = CF3D_DEEPDIG() defaults; over.cf3d fields: max_rounds (8),
%   wall_h (6.0), niter (20), alphas (logspace(-8,-2,7)).
%
%   See also CF3B_SPACING, CF2_EFC, cf_efc_lib.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    ov = over;  c3 = struct();
    if isfield(ov,'cf3d'), c3 = ov.cf3d;  ov = rmfield(ov,'cf3d'); end
    P = e2e6m_r2_params(ov);
    if ~isfield(c3,'max_rounds'), c3.max_rounds = 8; end
    if ~isfield(c3,'wall_h'),     c3.wall_h = 6.0; end
    if ~isfield(c3,'niter'),      c3.niter = 20; end
    if ~isfield(c3,'alphas'),     c3.alphas = logspace(-8,-2,7); end
    if ~isfield(c3,'target'),     c3.target = 0; end   % stop below this floor (0 = off)
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    lib = cf_efc_lib();

    rep = fullfile(P.outdir, 'cf3d_report.txt');
    logf_(rep, '==== e2e6m CF3d -- deep dig, d=1.10 m apl (restart ladder) %s', ...
          datestr(now,31)); %#ok<DATST>
    t0 = tic;

    % ---- chain + DMs, exactly the cf3b d=1.10 setup ---------------------
    beam_d = 2 * 0.023771;
    C1 = load(fullfile(P.outdir,'cf1_run.mat'));
    FC = struct();
    for k = 1:numel(C1.OUT.F), FC.(C1.OUT.F(k).key) = C1.OUT.F(k); end
    prop_in = fullfile(P.outdir, 'r1_seg_d110_prop.in');
    dmrx    = fullfile(P.outdir, 'r1_seg_d110_dm.in');
    assert(isfile(prop_in) && isfile(dmrx), 'cf3d: run cf3b first (d110 decks)');
    Adm = ctb_dm_rx('rx_in', prop_in, 'rx_out', dmrx, ...
                    'dms', P.dm.names, 'ng', P.dm.ng);   % deterministic re-emit
    ch = cf_chain('rx', dmrx, 'model_size', P.dj.model, ...
                  'prolate_iter', P.co.prolate_iter, ...
                  'circ_stop_frac', P.cf.circ_stop_frac, FC.apl.cfg{:});
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
    logf_(rep, 'chain %s | dz %d px | niter %d | alphas %.0e..%.0e | budget %.1f h', ...
          tag, numel(dz_idx), c3.niter, c3.alphas(1), c3.alphas(end), c3.wall_h);

    % ---- resume or start ------------------------------------------------
    ckpt = fullfile(P.outdir, 'cf3d_run.mat');
    hist = struct('round',{}, 'c_start',{}, 'c_end',{}, 'alpha',{}, ...
                  'stroke_nm',{}, 'la_floor',{}, 'minutes',{});
    a = cellfun(@(dd) zeros(dd.nact^2,1), dm, 'UniformOutput', false);
    r0 = 1;
    if isfile(ckpt)
        C = load(ckpt);
        a = C.a;  hist = C.hist;  r0 = numel(hist) + 1;
        logf_(rep, 'RESUMED at round %d (last floor %.3e)', r0, hist(end).c_end);
    end

    % ---- the ladder -----------------------------------------------------
    stall = 0;
    for r = r0:c3.max_rounds
        tr = tic;
        if r == 1
            Gc = fullfile(P.outdir, sprintf('cf2_G_%s.mat', tag));   % cf3b cache
        else
            Gc = fullfile(P.outdir, sprintf('cf3d_G_%s_r%d.mat', tag, r));
        end
        [G, ~] = lib.jacobian(ch, dm, a, dz_idx, P, Gc);
        la = lib.linfloor(G, P.cf.stroke_bound_nm);
        [a, cvec, alph] = lib.efc(ch, dm, G, a, dz_idx, c3.niter, c3.alphas);
        if isempty(alph)
            logf_(rep, '[round %d] NO ACCEPTED STEP (line search exhausted at %.3e)', ...
                  r, cvec(end));
            alph = NaN;
        end
        ach = 1e9 * rms_(cell2mat(cellfun(@(x) x(x~=0), a(:).', ...
                                          'UniformOutput', false).'));
        hist(end+1) = struct('round',r, 'c_start',cvec(1), 'c_end',cvec(end), ...
            'alpha',alph(end), 'stroke_nm',ach, 'la_floor',la.floor, ...
            'minutes',toc(tr)/60); %#ok<AGROW>
        save(ckpt, 'a', 'hist', 'tag');
        logf_(rep, '[round %d] %.3e -> %.3e | alpha %.1e | stroke %.1f nm | la(G) %.3e | %.1f min', ...
              r, cvec(1), cvec(end), alph(end), ach, la.floor, toc(tr)/60);
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

    OUT = struct('P',P, 'tag',tag, 'hist',hist, 'a',{a});
    save(ckpt, 'a', 'hist', 'tag', 'OUT');
    logf_(rep, 'CF3d DONE in %.1f h: %.3e -> %.3e over %d rounds (lin-ach %.3e)', ...
          toc(t0)/3600, hist(1).c_start, hist(end).c_end, numel(hist), ...
          hist(end).la_floor);
end

function logf_(rep, varargin)
    s = sprintf(varargin{:});
    fid = fopen(rep, 'a');  fprintf(fid, '%s\n', s);  fclose(fid);
    fprintf('%s\n', s);
end

function v = rms_(x)
    v = sqrt(mean(double(x(:)).^2));
end
