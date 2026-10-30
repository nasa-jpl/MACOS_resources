function S = e2e_cycles(cfg)
%E2E_CYCLES  Recompute-and-restart EFC cycles: the Jacobian re-measured about the loop's own state, the loop restarted.
%   S = E2E_CYCLES(CFG) drives the CTB's EFC (ctb_efc) in cycles on the deck
%   and chain CFG names: cycle k measures (or reuses) the Jacobian about the
%   DM state a_k, runs the loop from a_k until it stalls or reaches
%   CFG.niter iterations, and the next cycle starts from where it ended.
%   This is the standard practice for driving contrast down (Dave
%   2026-10-30: ten or more recompute-and-restart cycles to approach the
%   goal is usual); stage 9 did one such round, this driver does as many as
%   the stop rule allows.  Resumable: the chain's state is saved after
%   every cycle in CFG.exdir/s11_<tag>_state.mat and a call with the same
%   CFG continues from the last finished cycle (each cycle's Jacobian is
%   saved beside its fingerprint truth and reused through e2e_jacobian when
%   it is this chain's).
%
%   CFG fields:
%     exdir, tag        the stage directory and the chain's name
%                       ('sens', 'perf'): files s11_<tag>_state.mat,
%                       Jacobians ctb_dm_jacobian_N<model>_e2e_<tag>_c<k>,
%                       loops ctb_efc_e2e_<tag>_c<k>
%     r, e, ch, chain_pick, MODEL   as s9 / s10 (the DM deck, the station
%                       map, the chain, its options, the model size)
%     sensing           'true' (ctb_efc's perfect sensing) or a struct of
%                       ctb_probe_estimate options (the probed loop)
%     a0                the start state ({} = flat)
%     niter, alphas     per-cycle loop settings (ctb_efc)
%     max_cycles        the cap: cycles 0 .. max_cycles
%     stop_gain         stop when a cycle's start/end contrast ratio is
%                       below this (1.05: a 5 % cycle is convergence)
%     goal              stop when the true contrast reaches this (0 = never)
%     reuse             cell of tags: cycle k may reuse the Jacobian with
%                       tag reuse{k+1} when that Jacobian's linearization
%                       point IS a_k (stage 9's 'e2e_coupled' for cycle 0,
%                       'e2e_coupled_r1' for the perfect chain's cycle 1);
%                       a tag whose point differs is not touched (the
%                       cycle measures its own) -- so a committed truth is
%                       never deleted by a chain it does not belong to
%     jac_opts          further ctb_dm_jacobian options (cell, default {};
%                       the tests' small Jacobians)
%     say               the printer
%
%   S fields: cycles (struct array, one per finished cycle: k, jac_tag,
%   jac_time_s, jac_reused, c_start, c_end, gain, contrast, contrast_est,
%   iters, stroke_rms_nm, est_rel0 / est_floor0 (the probe estimate's
%   error at the cycle start with the fresh Jacobian; NaN under perfect
%   sensing), a_end), a_final, c_final, stop_reason ('converged', 'goal',
%   'cap', '' while running), total_iters, state_file.
%
%   See also S11_CYCLES, E2E_JACOBIAN, CTB_EFC, CTB_PROBE_ESTIMATE.
    say = cfg.say;
    sf = fullfile(cfg.exdir, sprintf('s11_%s_state.mat', cfg.tag));
    if isfile(sf)
        S = load(sf);
        say('[cycles %s] resuming from %s: %d cycles done, contrast %.3e, stop ''%s''', cfg.tag, sf, numel(S.cycles), S.c_final, S.stop_reason);
        if ~isempty(S.stop_reason) && ~strcmp(S.stop_reason, 'cap'), return; end
        if numel(S.cycles) - 1 >= cfg.max_cycles, S.stop_reason = 'cap';  save(sf, '-struct', 'S');  return; end
        S.stop_reason = '';
        a = S.a_final;  k0 = numel(S.cycles);
    else
        S = struct('cycles', struct('k', {}, 'jac_tag', {}, 'jac_time_s', {}, 'jac_reused', {}, 'c_start', {}, 'c_end', {}, 'gain', {}, ...
                   'contrast', {}, 'contrast_est', {}, 'iters', {}, 'stroke_rms_nm', {}, 'est_rel0', {}, 'est_floor0', {}, 'a_end', {}), ...
                   'a_final', {cfg.a0}, 'c_final', NaN, 'stop_reason', '', 'total_iters', 0, 'state_file', sf, 'tag', cfg.tag);
        a = cfg.a0;  k0 = 0;
    end
    probe = isstruct(cfg.sensing);
    for k = k0:cfg.max_cycles
        % ---- the Jacobian about a_k ------------------------------------------------
        own = sprintf('e2e_%s_c%d', cfg.tag, k);  tagk = own;
        if isfield(cfg, 'reuse') && numel(cfg.reuse) >= k+1 && ~isempty(cfg.reuse{k+1})
            cand = cfg.reuse{k+1};
            jp = fullfile(cfg.exdir, sprintf('ctb_dm_jacobian_N%d_%s.mat', cfg.MODEL, cand));
            if isfile(jp)
                Jc = load(jp, 'a0');                                  % a flat point is {} or a cell of zero vectors
                flat_a = isempty(a) || all(cellfun(@(x) ~any(x), a));
                flat_J = isempty(Jc.a0) || all(cellfun(@(x) ~any(x), Jc.a0));
                if (flat_a && flat_J) || (~flat_a && ~isempty(Jc.a0) && isequal(Jc.a0, a)), tagk = cand;
                else, say('[cycles %s] cycle %d: %s is about another state; measuring %s', cfg.tag, k, cand, own); end
            end
        end
        t1 = tic;
        jo = {};  if isfield(cfg, 'jac_opts'), jo = cfg.jac_opts; end
        Jk = e2e_jacobian(cfg.exdir, tagk, cfg.r, cfg.e, cfg.ch, cfg.chain_pick, a, cfg.MODEL, say, jo);
        jt = toc(t1);  reused = jt < 60;
        % ---- the estimate at the cycle start, with the fresh Jacobian (the probed chain) -----
        rel0 = NaN;  fl0 = NaN;
        if probe
            pa = namedargs2cell(cfg.sensing);
            es = ctb_probe_estimate(cfg.ch, Jk, a, pa{:}, 'check_jac', false);
            rel0 = es.rel_err;  fl0 = es.floor;
            say('[cycles %s] cycle %d: the estimate at the start, Jacobian about this state: relative error %.3e, floor %.2e on %.3e', cfg.tag, k, rel0, fl0, es.c_true);
        end
        % ---- the loop from a_k --------------------------------------------------------
        out = ctb_efc('jac', Jk, 'a0', a, 'tag', own, 'outdir', cfg.exdir, 'niter', cfg.niter, 'alphas', cfg.alphas, 'visible', false, 'sensing', cfg.sensing);
        c = struct('k', k, 'jac_tag', tagk, 'jac_time_s', jt, 'jac_reused', reused, 'c_start', out.c_before, 'c_end', out.c_after, ...
                   'gain', out.c_before/out.c_after, 'contrast', out.contrast, 'contrast_est', out.contrast_est, 'iters', numel(out.contrast)-1, ...
                   'stroke_rms_nm', out.stroke_rms_nm, 'est_rel0', rel0, 'est_floor0', fl0, 'a_end', {out.a});
        S.cycles(end+1) = c;
        a = out.a;  S.a_final = a;  S.c_final = out.c_after;  S.total_iters = S.total_iters + c.iters;
        say('[cycles %s] cycle %d: Jacobian %s (%s, %.1f min), loop %.3e -> %.3e in %d iterations (x%.2f), strokes %s nm rms; %d iterations so far', ...
            cfg.tag, k, tagk, tern_(reused, 'reused', 'measured'), jt/60, c.c_start, c.c_end, c.iters, c.gain, mat2str(c.stroke_rms_nm, 3), S.total_iters);
        if cfg.goal > 0 && c.c_end <= cfg.goal, S.stop_reason = 'goal';
        elseif c.gain < cfg.stop_gain, S.stop_reason = 'converged';
        elseif k == cfg.max_cycles, S.stop_reason = 'cap'; end
        save(sf, '-struct', 'S');
        if ~isempty(S.stop_reason)
            say('[cycles %s] stop: %s after cycle %d at %.3e', cfg.tag, S.stop_reason, k, S.c_final);
            return
        end
    end
end

function s = tern_(c, a, b)
    if c, s = a; else, s = b; end
end
