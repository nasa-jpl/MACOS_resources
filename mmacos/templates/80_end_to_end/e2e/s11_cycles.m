%S11_CYCLES  e2e stage 11: recompute-and-restart EFC cycles on the coupled model, probed and perfect.
%
%   Stage 10 found that the pairwise-probe estimate needs a Jacobian about
%   the loop's CURRENT state (33 % off at a neighbouring state, 1.2 % at the
%   Jacobian's own point) and that every probed fixed-G loop stalls at
%   three iterations.  Dave's ruling (2026-10-30): recomputing the Jacobian
%   is standard practice in driving contrast lower; ten or more
%   recompute-and-restart cycles to approach the goal is usual, and the
%   time it takes is too.  This stage does that (e2e_cycles): two chains on
%   the stage-9 model behind the annular Lyot stop, from flat mirrors --
%     'sens'  the probed loop (ctb_efc 'sensing' = ctb_probe_estimate at
%             stage 10's pick, P.sens pick_rel; the controller sees the
%             estimate only)
%     'perf'  the same cycles with perfect sensing, the reference (its first
%             two Jacobians are stage 9's: the flat one and the one about
%             its fixed-G dug state, reused when the loop reproduces that
%             state to the bit; else measured)
%   each cycle a Jacobian about the chain's own state (about an hour at
%   model 512 on the coupled train, central differences, 3520 runs), the
%   loop from it until it stalls or P.sens.cycles.niter, the next cycle
%   from where it ended, until a cycle gains less than
%   P.sens.cycles.stop_gain, the goal P.sens.cycles.goal is reached, or
%   the cap P.sens.cycles.max.  Resumable: every cycle's Jacobian is
%   saved beside its fingerprint truth and the chain's state after every
%   cycle; a rerun continues.  The environment variable S11_CHAIN
%   ('sens', 'perf', or 'both', the default) selects the chain, so the two
%   can run in two MATLAB sessions at once.
%
%   Gates as numbers, when both chains have run: the probed chain's final
%   contrast within P.sens.efc_ratio_max of the perfect chain's; the
%   probed chain's per-cycle estimate error at its fresh Jacobian under
%   P.sens.rel_err_max every cycle (the stage-10 control, generalized);
%   every cycle's end no brighter than its start; strokes under 50 nm.
%   Reported: the per-cycle table of both chains, the cycle at which each
%   stopped and why, the Jacobian wall time, the goal's distance.
%
%   Products: s11_{sens,perf}_state.mat, ctb_dm_jacobian_N512_e2e_{sens,perf}_c<k>.{mat,fp.json},
%   ctb_efc_e2e_{sens,perf}_c<k>.{mat,png}, s11_cycles.png, s11_report.txt, s11_run.mat
%   (the fingerprint truths are the committed record; the rest regenerates).
%
%   See also S10_SENSING, E2E_CYCLES, E2E_JACOBIAN, CTB_EFC, CTB_PROBE_ESTIMATE.

exdir = fileparts(mfilename('fullpath'));
root  = fullfile(exdir, '..', '..', '..');
run(fullfile(root, 'mmacos_setup.m'));
bench = fullfile(root, 'templates', '30_instruments', 'bench_ctb');
addpath(bench);  addpath(exdir);
P = e2e_params();  Q = P.sens;  C = Q.cycles;
cd(exdir);
t0 = tic;
say = @(varargin) fprintf('%s\n', sprintf(varargin{:}));
MODEL = 512;
which = getenv('S11_CHAIN');  if isempty(which), which = 'both'; end
assert(any(strcmp(which, {'sens', 'perf', 'both'})), 's11_cycles: S11_CHAIN is sens, perf or both');

%% -- [0] stage 9's model and stage 10's pick ----------------------------------------
need = {'s9_coupled.in', 's9_run.mat', 's10_run.mat'};
for k = 1:numel(need), assert(isfile(fullfile(exdir, need{k})), 's11_cycles: run s9_diffraction and s10_sensing first (%s missing)', need{k}); end
S9 = load(fullfile(exdir, 's9_run.mat'));  S10 = load(fullfile(exdir, 's10_run.mat'), 'pick_rel', 'probe_nm_pick');
e = S9.e;  li_pick = S9.S.coupled.lyot_pick;
r = ctb_dm_rx('rx_in', fullfile(exdir, 's9_coupled.in'), 'rx_out', fullfile(exdir, 's9_coupled_dm.in'));
chain_pick = {'elt', e, 'r_lyot_inner_frac', li_pick};
ch = ctb_chain('rx', r.rx_out, 'model_size', MODEL, chain_pick{:});
say('\n==================== e2e s11 -- recompute-and-restart cycles on the coupled telescope + CTB model (%s)', which);
say('[0] %s, Lyot blocker %.2f; the probe at %.1fx the default (%.1f nm at the static); cycles to %d, stop below x%.2f per cycle, goal %.1e', ...
    r.rx_out, li_pick, S10.pick_rel, S10.probe_nm_pick, C.max, C.stop_gain, C.goal);
base = struct('exdir', exdir, 'r', r, 'e', e, 'ch', ch, 'chain_pick', {chain_pick}, 'MODEL', MODEL, 'a0', {{}}, ...
              'niter', C.niter, 'alphas', Q.alphas, 'max_cycles', C.max, 'stop_gain', C.stop_gain, 'goal', C.goal, 'say', say);
cfgS = base;  cfgS.tag = 'sens';  cfgS.sensing = struct('npairs', Q.npairs, 'probe_rel', S10.pick_rel);  cfgS.reuse = {'e2e_coupled'};
cfgP = base;  cfgP.tag = 'perf';  cfgP.sensing = 'true';  cfgP.reuse = {'e2e_coupled', 'e2e_coupled_r1'};

%% -- [1] the chains ---------------------------------------------------------------------
R = struct();
if any(strcmp(which, {'sens', 'both'})), R.sens = e2e_cycles(cfgS); end
if any(strcmp(which, {'perf', 'both'})), R.perf = e2e_cycles(cfgP); end
for f = {'sens', 'perf'}                                        % the other chain's state, when it is on disk
    sf = fullfile(exdir, sprintf('s11_%s_state.mat', f{1}));
    if ~isfield(R, f{1}) && isfile(sf), R.(f{1}) = load(sf); end
end

%% -- [2] the record ----------------------------------------------------------------------
G = struct();  rep = {};
rep{end+1} = sprintf('e2e s11 -- recompute-and-restart cycles on the coupled telescope + CTB model (%s)', datestr(now, 'yyyy-mm-dd'));
rep{end+1} = sprintf('deck %s, Lyot blocker %.2f; probe %.1fx the default; cycles to %d, stop below x%.2f, goal %.1e; stage 9: static %.3e, one round %.3e', ...
    r.rx_out, li_pick, S10.pick_rel, C.max, C.stop_gain, C.goal, S9.efc_contrast(1), S9.efc_contrast_r1(end));
for f = fieldnames(R).'
    X = R.(f{1});
    say('[2] chain %s: %d cycles, %d iterations, %.3e -> %.3e, stop ''%s''', f{1}, numel(X.cycles), X.total_iters, X.cycles(1).c_start, X.c_final, X.stop_reason);
    rep{end+1} = sprintf('chain %s: %d cycles, %d iterations, %.3e -> %.3e, stop ''%s'' (x%.0f overall)', f{1}, numel(X.cycles), X.total_iters, X.cycles(1).c_start, X.c_final, X.stop_reason, X.cycles(1).c_start/X.c_final);
    for k = 1:numel(X.cycles)
        c = X.cycles(k);
        line = sprintf('   cycle %2d: Jacobian %-18s %s %5.1f min; %.3e -> %.3e in %2d iterations (x%.2f); strokes %s nm rms', ...
            c.k, c.jac_tag, tern_(c.jac_reused, 'reused  ', 'measured'), c.jac_time_s/60, c.c_start, c.c_end, c.iters, c.gain, mat2str(c.stroke_rms_nm, 3));
        if ~isnan(c.est_rel0), line = sprintf('%s; estimate at the start %.3e (floor %.1e)', line, c.est_rel0, c.est_floor0); end
        say('%s', line);  rep{end+1} = line;
    end
    gn = sprintf('monotone_%s', f{1});
    ok = all([X.cycles.c_end] <= [X.cycles.c_start] * (1 + 1e-9));
    G.(gn) = struct('v', max([X.cycles.c_end] ./ [X.cycles.c_start]), 'ok', ok, 'say', sprintf('chain %s: every cycle ends no brighter than it starts', f{1}));
    gn = sprintf('stroke_%s', f{1});  smax = max(cellfun(@max, {X.cycles.stroke_rms_nm}));
    G.(gn) = struct('v', smax, 'ok', smax < 50, 'say', sprintf('chain %s: DM strokes under 50 nm rms (max %.2f)', f{1}, smax));
end
if isfield(R, 'sens')
    rel = [R.sens.cycles.est_rel0];
    G.estimate = struct('v', max(rel), 'ok', all(rel < Q.rel_err_max), 'say', sprintf('the probed chain''s estimate at every fresh Jacobian within %.2f of the field (worst %.3e over %d cycles)', Q.rel_err_max, max(rel), numel(rel)));
end
if isfield(R, 'sens') && isfield(R, 'perf')
    G.vs_perfect = struct('v', R.sens.c_final/R.perf.c_final, 'ok', R.sens.c_final < Q.efc_ratio_max * R.perf.c_final, ...
        'say', sprintf('the probed chain ends at %.3e, %.2fx the perfect chain''s %.3e (gate %.1fx; %d vs %d cycles)', R.sens.c_final, R.sens.c_final/R.perf.c_final, R.perf.c_final, Q.efc_ratio_max, numel(R.sens.cycles), numel(R.perf.cycles)));
end
gn = fieldnames(G);  allok = true;
for k = 1:numel(gn), say('    [%s] %s', tern_(G.(gn{k}).ok, 'PASS', 'FAIL'), G.(gn{k}).say);  allok = allok && G.(gn{k}).ok;  rep{end+1} = sprintf('   [%s] %s', tern_(G.(gn{k}).ok, 'PASS', 'FAIL'), G.(gn{k}).say); end
try
    f = figure('Visible', 'off', 'Position', [0 0 1100 520], 'Color', 'w');
    st = struct('sens', 'b-s', 'perf', 'k-o');  lg = {};
    for fn = fieldnames(R).'
        X = R.(fn{1});  n0 = 0;
        for k = 1:numel(X.cycles)
            c = X.cycles(k);  nn = n0 + (0:numel(c.contrast)-1);
            semilogy(nn, c.contrast, st.(fn{1}), 'LineWidth', 1.5, 'HandleVisibility', tern_(k == 1, 'on', 'off'));  hold on
            if ~all(isnan(c.contrast_est)), semilogy(nn, c.contrast_est, [st.(fn{1})(1) ':'], 'LineWidth', 1.0, 'HandleVisibility', 'off'); end
            xline(n0, ':', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
            n0 = n0 + c.iters;
        end
        lg{end+1} = sprintf('%s: %d cycles, %.2e', fn{1}, numel(X.cycles), X.c_final); %#ok<SAGROW>
    end
    if C.goal > 0, yline(C.goal, 'r--', 'goal', 'HandleVisibility', 'off'); end
    grid on;  xlabel('EFC iteration (dotted verticals: a Jacobian re-measured about the state)');  ylabel('dark-zone mean contrast, 3-15 \lambda/D');
    legend(lg, 'Location', 'northeast');  title('e2e s11: recompute-and-restart cycles, probed (dotted: the estimate) and perfect sensing');
    print(f, fullfile(exdir, 's11_cycles.png'), '-dpng', '-r110');  close(f);
catch ME, say('    (figure: %s)', ME.message);
end
rep{end+1} = sprintf('products: s11_{sens,perf}_state.mat ctb_dm_jacobian_N512_e2e_{sens,perf}_c<k>.{mat,fp.json} ctb_efc_e2e_{sens,perf}_c<k>.{mat,png} s11_cycles.png s11_report.txt s11_run.mat; this run %.0f s', toc(t0));
fid = fopen(fullfile(exdir, 's11_report.txt'), 'w');  fprintf(fid, '%s\n', rep{:});  fclose(fid);
S11 = struct('G', G, 'R', R, 'pick_rel', S10.pick_rel, 'cycles_cfg', C);
save(fullfile(exdir, 's11_run.mat'), '-struct', 'S11');
say('[3] report s11_report.txt, s11_cycles.png, s11_run.mat; %s; %.0f s', tern_(allok, 'ALL GATES PASS', 'A GATE FAILED'), toc(t0));

function s = tern_(c, a, b)
    if c, s = a; else, s = b; end
end
