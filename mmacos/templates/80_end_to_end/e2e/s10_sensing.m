%S10_SENSING  e2e stage 10: pairwise probing on the coupled telescope + CTB model.
%
%   Consumes stage 9's products (s9_coupled_dm.in with its station map and
%   Lyot pick in s9_run.mat, the two Jacobians ctb_dm_jacobian_N512_e2e_coupled
%   {,_r1} and the two EFC records ctb_efc_e2e_coupled{,_r1}.mat) and does
%   on the telescope what the mmacos manual's Section 8 (ch08d) does on the
%   bench: the dark-zone field is ESTIMATED from intensity images taken
%   with DM probe pairs (templates/30_instruments/bench_ctb/ctb_probe_estimate,
%   Give'on, Kern & Shaklan 2011), scored against the engine's true field,
%   which the controller never sees, and EFC is closed on the estimate
%   (ctb_efc 'sensing') against stage 9's perfect-sensing record
%   (PLAN_CONSOLIDATION 3a, Dave 2026-10-28: the estimator on the CTB
%   first, then the fold, then the coupled model, then pairwise probing on
%   the coupled system -- this is the last of the four).
%
%   [1] the static state (the annular Lyot stop of stage 9's pick, flat
%       DMs) at the estimator's DEFAULT amplitude, the probe as bright as
%       the speckle: the estimation floor, the relative field error, the
%       per-pixel error histogram (no cut), the probe-intensity diagnostic
%       ratios (the model-mismatch meter) and the per-pixel conditioning.
%       The finding this stage records: the default rule, which puts a
%       0.23 rad probe on the bench's 3e-7 halo, puts a 0.9 rad one on the
%       telescope's 5e-7 halo (the secondary's shadow, s9 [3b]), and the
%       probe's second-order response leaves the estimate 47 % off the
%       field.  The error scales as the amplitude squared.
%   [2] the amplitude sweep over P.sens.amp_rel of the default, noiseless
%       and at P.sens.photons_efc photons per image; the PICK is the
%       largest multiple whose noiseless error is under P.sens.rel_err_max
%       (the bench's regime, 4 %), and the probe follows the field from
%       there ('probe_rel': the default's scaling by the speckle level,
%       times the pick).  Then the photon sweep over P.sens.photons at the
%       pick: the exposure the telescope's halo needs.  Measured on the
%       way: at 1e8 photons per image (the bench's count, 50 photons per
%       speckle pixel here) no linear probe estimates this halo -- the
%       small ones are photon-starved, the large one nonlinear.
%   [3] the dug state (stage 9's relinearized dark zone, 6.7e-8): the same
%       estimate at the pick with the Jacobian about the dug state, and
%       the photon sweep there -- what a lab would have to see.
%   [4] EFC closed on the estimate at the pick from flat mirrors, stage
%       9's settings (20 iterations, the alpha grid to 1): fixed G, then
%       the relinearization round on stage 9's second Jacobian from the
%       probed loop's own dug state (a Jacobian about a neighbouring
%       state: the two states differ by the sensing path only), then the
%       fixed-G loop at P.sens.photons_efc photons, and the counter-example,
%       the fixed-G loop at the default rule; every curve against stage
%       9's perfect-sensing curves.  The alpha line search scores trials
%       on the MEASURED contrast as ctb_efc always does, so the plant
%       stays honest; only the field the controller inverts is the estimate.
%   [5] products: s10_probe.png (the sweeps, the error histogram and
%       maps), s10_efc.png (the curves), ctb_efc_e2e_sens{,_r1,_ph,_def}.{mat,png},
%       s10_report.txt, s10_run.mat.  Gates as numbers (P.sens): the
%       noiseless estimate at the pick within rel_err_max of the field,
%       the smallest probe's within lin_err_max (the Jacobian predicts
%       the probes), the photon floors ordered, the
%       dug-state floor below dug_floor_frac of the field, the probed EFC
%       within efc_ratio_max of perfect sensing (both rounds), the
%       photon-noise EFC within photon_ratio_max of the noiseless one,
%       the strokes under 50 nm.  Reported, not gated: the default rule's
%       estimate and the EFC closed on it, and the probe-intensity
%       diagnostic, which on the telescope carries an amplitude-independent
%       22 % offset on the symmetric probe (the probe's second-order field
%       projected on the shadow's halo; the pairwise difference cancels it,
%       the sum diagnostic keeps it), so it is not a Jacobian meter there.
%
%   See also S9_DIFFRACTION, E2E_JACOBIAN, CTB_PROBE_ESTIMATE, CTB_EFC.

exdir = fileparts(mfilename('fullpath'));
root  = fullfile(exdir, '..', '..', '..');
run(fullfile(root, 'mmacos_setup.m'));
bench = fullfile(root, 'templates', '30_instruments', 'bench_ctb');
addpath(bench);  addpath(exdir);
P = e2e_params();  Q = P.sens;
cd(exdir);
t0 = tic;
say = @(varargin) fprintf('%s\n', sprintf(varargin{:}));
MODEL = 512;                                                   % stage 9's diffraction model size

%% -- [0] stage 9's products ---------------------------------------------------
need = {'s9_coupled.in', 's9_run.mat', 'ctb_efc_e2e_coupled.mat', 'ctb_efc_e2e_coupled_r1.mat'};
for k = 1:numel(need), assert(isfile(fullfile(exdir, need{k})), 's10_sensing: run s9_diffraction first (%s missing)', need{k}); end
S9 = load(fullfile(exdir, 's9_run.mat'));
e = S9.e;  li_pick = S9.S.coupled.lyot_pick;
names = regexp(fileread(fullfile(exdir, 's9_coupled.in')), '(?m)^\s*EltName=\s*(\S+)', 'tokens');  names = [names{:}];
for f = fieldnames(e).'
    assert(strcmp(names{e.(f{1})}, f{1}), 's10_sensing: station %s is element %d of s9_coupled.in, named %s', f{1}, e.(f{1}), names{e.(f{1})});
end
r = ctb_dm_rx('rx_in', fullfile(exdir, 's9_coupled.in'), 'rx_out', fullfile(exdir, 's9_coupled_dm.in'));   % the DM deck (and flat256.txt beside it), as stage 9 wrote it
chain_pick = {'elt', e, 'r_lyot_inner_frac', li_pick};
ch = ctb_chain('rx', r.rx_out, 'model_size', MODEL, chain_pick{:});   % the chain owns the session from here
o9  = load(fullfile(exdir, 'ctb_efc_e2e_coupled.mat'));     % stage 9's perfect-sensing EFC, fixed G
o9r = load(fullfile(exdir, 'ctb_efc_e2e_coupled_r1.mat'));  % ... and its relinearized round
say('\n==================== e2e s10 -- pairwise probing on the coupled telescope + CTB model');
say('[0] %s, Lyot blocker %.2f pupil radii; stage 9: static %.3e, EFC %.3e fixed-G -> %.3e relinearized', ...
    r.rx_out, li_pick, S9.efc_contrast(1), S9.efc_contrast(end), S9.efc_contrast_r1(end));
J  = e2e_jacobian(exdir, 'e2e_coupled',    r, e, ch, chain_pick, {},   MODEL, say);
J2 = e2e_jacobian(exdir, 'e2e_coupled_r1', r, e, ch, chain_pick, o9.a, MODEL, say);
G = struct();  lam_nm = ch.lambda_m * 1e9;
rad = @(nm) 2*pi * 2*nm / lam_nm;                               % a surface stroke's peak phase, rad (reflection: twice the sag)
est_ = @(Jx, a, varargin) ctb_probe_estimate(ch, Jx, a, 'npairs', Q.npairs, varargin{:});

%% -- [1] the default rule at the static state ----------------------------------------
t1 = tic;  e0 = est_(J, {});  dt1 = toc(t1);
q0 = prctile(e0.err_px, [50 90 99 100]);
say('[1] flat DMs, annular stop, the DEFAULT amplitude: true contrast %.3e (stage 9 %.3e), estimated %.3e; %d pairs at %.2f nm peak = %.2f rad (the bench: 9.27 nm, 0.23 rad)', ...
    e0.c_true, S9.efc_contrast(1), e0.c_est, size(e0.dE, 2), e0.probe_nm, rad(e0.probe_nm));
say('    estimation floor %.2e (%.1f %% of the field), relative field error %.3e; ratios %s; worst pixel conditioning %.1f; %d chain runs, %.2f s each', ...
    e0.floor, 100*e0.floor/e0.c_true, e0.rel_err, strjoin(compose('%.3f', e0.ratio), ' '), max(e0.cond2), e0.nruns, dt1/e0.nruns);
say('    per-pixel error |dE|^2/peak: median %.2e, 90%% %.2e, 99%% %.2e, max %.2e', q0);
G.static_match = struct('v', abs(e0.c_true/S9.efc_contrast(1) - 1), 'ok', abs(e0.c_true/S9.efc_contrast(1) - 1) < 1e-3, ...
    'say', sprintf('the chain reproduces stage 9''s static dark zone (%.4e vs %.4e)', e0.c_true, S9.efc_contrast(1)));
G.default_rule = struct('v', e0.rel_err, 'ok', true, 'say', sprintf(['REPORTED, not gated: the estimator''s default amplitude (probe intensity = the speckle level) is %.1f nm, ' ...
    '%.2f rad, on the telescope''s %.1e halo; the estimate is %.0f %% off the field (floor %.1e), ratios %s -- the probe''s second-order response, not the Jacobian'], ...
    e0.probe_nm, rad(e0.probe_nm), e0.c_true, 100*e0.rel_err, e0.floor, mat2str(e0.ratio, 3)));

%% -- [2] the amplitude sweep and the pick; the photon sweep at the pick --------------
sw1 = struct('rel', {}, 'nm', {}, 'floor', {}, 'floor_ph', {}, 'err', {}, 'err_ph', {}, 'ratio', {});
for f = Q.amp_rel
    es = est_(J, {}, 'probe_rel', f);  ep = est_(J, {}, 'probe_rel', f, 'photons', Q.photons_efc);
    sw1(end+1) = struct('rel', f, 'nm', es.probe_nm, 'floor', es.floor, 'floor_ph', ep.floor, 'err', es.rel_err, 'err_ph', ep.rel_err, 'ratio', es.ratio); %#ok<SAGROW>
    say('[2] amplitude %.1fx = %6.2f nm (%.2f rad): noiseless floor %.2e, relative error %.3e, ratios %s; at %.0e photons floor %.2e, error %.3e', ...
        f, es.probe_nm, rad(es.probe_nm), es.floor, es.rel_err, strjoin(compose('%.3f', es.ratio), ' '), Q.photons_efc, ep.floor, ep.rel_err);
end
kp = find([sw1.err] < Q.rel_err_max, 1, 'last');  assert(~isempty(kp), 's10_sensing: no amplitude in the sweep estimates the field within %.2f', Q.rel_err_max);
f_pick = sw1(kp).rel;                                           % the largest probe still in the linear regime (the bench's, 4 %)
[~, kph] = min([sw1.floor_ph]);
say('    PICK: %.1fx the default (%.2f nm, %.2f rad): the largest amplitude whose noiseless error is under %.2f; the probe follows the field from here', f_pick, sw1(kp).nm, rad(sw1(kp).nm), Q.rel_err_max);
say('    at %.0e photons per image the smallest floor is %.2e (%.1f %% of the field) at %.1fx: the halo''s exposure budget, see the photon sweep', Q.photons_efc, sw1(kph).floor_ph, 100*sw1(kph).floor_ph/e0.c_true, sw1(kph).rel);
e1 = est_(J, {}, 'probe_rel', f_pick);
q1 = prctile(e1.err_px, [50 90 99 100]);
say('    at the pick, noiseless: estimated %.3e (true %.3e); floor %.2e (%.2f %% of the field), relative error %.3e; ratios %s; conditioning %.1f', ...
    e1.c_est, e1.c_true, e1.floor, 100*e1.floor/e1.c_true, e1.rel_err, strjoin(compose('%.3f', e1.ratio), ' '), max(e1.cond2));
say('    per-pixel error: median %.2e, 90%% %.2e, 99%% %.2e, max %.2e', q1);
G.est = struct('v', e1.rel_err, 'ok', e1.rel_err < Q.rel_err_max, ...
    'say', sprintf('noiseless estimate at the pick (%.1fx, %.1f nm): relative field error %.3e (gate %.2f), floor %.2e on a field of %.3e', f_pick, e1.probe_nm, e1.rel_err, Q.rel_err_max, e1.floor, e1.c_true));
G.jacobian = struct('v', sw1(1).err, 'ok', sw1(1).err < Q.lin_err_max, ...
    'say', sprintf('the Jacobian predicts the probes: the smallest probe (%.1fx, %.1f nm) estimates the field to %.1e (gate %.0e); the error grows as the amplitude squared (%s over %s)', ...
    sw1(1).rel, sw1(1).nm, sw1(1).err, Q.lin_err_max, mat2str([sw1.err], 2), mat2str([sw1.rel], 2)));
G.knee = struct('v', f_pick, 'ok', f_pick < 1, 'say', sprintf('the pick is below the default rule (%.1fx): the brighter halo wants a smaller probe', f_pick));
sw2 = struct('ph', {}, 'floor', {}, 'rel', {});
for ph = Q.photons
    es = est_(J, {}, 'probe_rel', f_pick, 'photons', ph);
    sw2(end+1) = struct('ph', ph, 'floor', es.floor, 'rel', es.rel_err); %#ok<SAGROW>
    say('    photons %-8s at the bare peak: floor %.2e (%.2f %% of the field), relative error %.3e', phname_(ph), es.floor, 100*es.floor/e1.c_true, es.rel_err);
end
fl = [sw2.floor];  [~, order] = sort([sw2.ph], 'descend');  order = [find([sw2.ph] == 0), order([sw2(order).ph] > 0)];   % noiseless, then the most photons first
G.photons = struct('v', fl(order), 'ok', all(diff(fl(order)) > 0), 'say', sprintf('the photon floors at the pick are ordered: noiseless %.2e < %s', fl(order(1)), strjoin(arrayfun(@(k) sprintf('%s %.2e', phname_(sw2(k).ph), sw2(k).floor), order(2:end), 'UniformOutput', false), ' < ')));

%% -- [3] the dug state -----------------------------------------------------------------
a_dug = o9r.a;
t3 = tic;  ed = est_(J2, a_dug, 'probe_rel', f_pick);  dt3 = toc(t3);
qd = prctile(ed.err_px, [50 90 99 100]);
say('[3] the dug state (stage 9 relinearized) at the pick: true contrast %.3e (stage 9 %.3e), estimated %.3e; probes %.2f nm peak (%.2f rad)', ed.c_true, S9.efc_contrast_r1(end), ed.c_est, ed.probe_nm, rad(ed.probe_nm));
say('    estimation floor %.2e (%.2f %% of the field), relative field error %.3e; ratios %s; conditioning %.1f; %.2f s per run', ...
    ed.floor, 100*ed.floor/ed.c_true, ed.rel_err, strjoin(compose('%.3f', ed.ratio), ' '), max(ed.cond2), dt3/ed.nruns);
say('    per-pixel error: median %.2e, 90%% %.2e, 99%% %.2e, max %.2e', qd);
sw3 = struct('ph', {}, 'floor', {}, 'rel', {});
for ph = Q.photons
    es = est_(J2, a_dug, 'probe_rel', f_pick, 'photons', ph);
    sw3(end+1) = struct('ph', ph, 'floor', es.floor, 'rel', es.rel_err); %#ok<SAGROW>
    say('    photons %-8s: floor %.2e (%.2f %% of the dug field), relative error %.3e', phname_(ph), es.floor, 100*es.floor/ed.c_true, es.rel_err);
end
drift = max(abs(sw1(1).ratio - e0.ratio));
G.ratio = struct('v', max(abs(e1.ratio - 1)), 'ok', true, 'say', sprintf(['REPORTED, not gated: the probe-intensity diagnostic reads %s at the pick and moves only %.3f between %.1fx and the default -- ' ...
    'an amplitude-INDEPENDENT offset on the symmetric probe, the probe''s second-order field projected on the halo (2 Re(E0* d2) scales with the probe intensity as |dE|^2 does), ' ...
    'which the pairwise DIFFERENCE cancels and the sum diagnostic keeps; it is 2 %% on the bench''s halo and %.0f %% on the shadow''s, so on the telescope the ratio is not a Jacobian meter (the dug state reads %s)'], ...
    mat2str(e1.ratio, 3), drift, sw1(1).rel, 100*max(abs(e1.ratio - 1)), mat2str(ed.ratio, 3)));
% the control: the same estimate at the second Jacobian's OWN linearization point (stage 9's fixed-G dug
% state); the difference is what a Jacobian about a neighbouring state costs the estimate
ec = est_(J2, o9.a, 'probe_rel', f_pick);
say('    the control, at the Jacobian''s own linearization point (stage 9 fixed-G, %.3e): estimated %.3e, floor %.2e (%.2f %%), relative error %.3e, ratios %s', ...
    ec.c_true, ec.c_est, ec.floor, 100*ec.floor/ec.c_true, ec.rel_err, strjoin(compose('%.3f', ec.ratio), ' '));
G.dug = struct('v', ed.floor/ed.c_true, 'ok', ed.floor < Q.dug_floor_frac * ed.c_true, ...
    'say', sprintf('at the dug state the noiseless floor is %.2f %% of the %.3e field (gate %.0f %%); relative error %.3e with the Jacobian about the neighbouring state, %.3e at that Jacobian''s own point (%.3e)', ...
    100*ed.floor/ed.c_true, ed.c_true, 100*Q.dug_floor_frac, ed.rel_err, ec.rel_err, ec.c_true));

%% -- [4] EFC closed on the estimate ---------------------------------------------------
efc = {'outdir', exdir, 'niter', Q.niter, 'alphas', Q.alphas, 'visible', false};
sens = struct('npairs', Q.npairs, 'probe_rel', f_pick);
oE = ctb_efc('jac', J, 'tag', 'e2e_sens', efc{:}, 'sensing', sens);
say('[4] EFC on the noiseless estimate at the pick, fixed G: %s', mat2str(oE.contrast, 4));
say('    stage 9, perfect sensing:                        %s', mat2str(S9.efc_contrast, 4));
oE2 = ctb_efc('jac', J2, 'a0', oE.a, 'tag', 'e2e_sens_r1', efc{:}, 'sensing', sens);
say('    relinearized (stage 9''s second Jacobian, from the probed loop''s dug state): %s', mat2str(oE2.contrast, 4));
say('    stage 9, perfect sensing:                        %s', mat2str(S9.efc_contrast_r1, 4));
sens.photons = Q.photons_efc;
oP = ctb_efc('jac', J, 'tag', 'e2e_sens_ph', efc{:}, 'sensing', sens);
say('    on the estimate at %.0e photons, fixed G:         %s', Q.photons_efc, mat2str(oP.contrast, 4));
oD = ctb_efc('jac', J, 'tag', 'e2e_sens_def', efc{:}, 'sensing', struct('npairs', Q.npairs));
say('    the counter-example, the default rule, fixed G:  %s', mat2str(oD.contrast, 4));
say('    strokes nm rms: pick %s, relinearized %s, photons %s, default rule %s (stage 9 %s / %s)', mat2str(oE.stroke_rms_nm, 3), mat2str(oE2.stroke_rms_nm, 3), ...
    mat2str(oP.stroke_rms_nm, 3), mat2str(oD.stroke_rms_nm, 3), mat2str(S9.strokes_nm, 3), mat2str(S9.strokes_nm_r1, 3));
G.efc = struct('v', oE.c_after/S9.efc_contrast(end), 'ok', oE.c_after < Q.efc_ratio_max * S9.efc_contrast(end), ...
    'say', sprintf('EFC on the estimate, fixed G: %.3e -> %.3e in %d iterations, %.2fx perfect sensing''s %.3e (gate %.1fx)', oE.c_before, oE.c_after, numel(oE.contrast)-1, oE.c_after/S9.efc_contrast(end), S9.efc_contrast(end), Q.efc_ratio_max));
G.efc_r1 = struct('v', oE2.c_after/S9.efc_contrast_r1(end), 'ok', oE2.c_after < Q.efc_ratio_max * S9.efc_contrast_r1(end), ...
    'say', sprintf('relinearized: -> %.3e in %d more, %.2fx perfect sensing''s %.3e (gate %.1fx); overall x%.0f from the static', oE2.c_after, numel(oE2.contrast)-1, oE2.c_after/S9.efc_contrast_r1(end), S9.efc_contrast_r1(end), Q.efc_ratio_max, oE.c_before/oE2.c_after));
G.efc_ph = struct('v', oP.c_after/oE.c_after, 'ok', oP.c_after < Q.photon_ratio_max * oE.c_after, ...
    'say', sprintf('at %.0e photons per image: -> %.3e in %d iterations, %.2fx the noiseless estimate''s (gate %.1fx)', Q.photons_efc, oP.c_after, numel(oP.contrast)-1, oP.c_after/oE.c_after, Q.photon_ratio_max));
G.efc_default = struct('v', oD.c_after/S9.efc_contrast(end), 'ok', true, ...
    'say', sprintf('REPORTED, not gated: EFC on the default rule''s estimate reaches %.3e in %d iterations, %.2fx perfect sensing (the line search keeps the plant honest; the estimate does not)', oD.c_after, numel(oD.contrast)-1, oD.c_after/S9.efc_contrast(end)));
smax = max([oE2.stroke_rms_nm oP.stroke_rms_nm]);
G.stroke = struct('v', smax, 'ok', smax < 50, 'say', sprintf('DM strokes under 50 nm rms (max %.2f)', smax));

%% -- [5] products ---------------------------------------------------------------------
gn = fieldnames(G);  allok = true;
for k = 1:numel(gn), say('    [%s] %s', tern_(G.(gn{k}).ok, 'PASS', 'FAIL'), G.(gn{k}).say);  allok = allok && G.(gn{k}).ok; end
try
    f = figure('Visible', 'off', 'Position', [0 0 1500 800], 'Color', 'w');
    subplot(2, 3, 1);
    loglog([sw1.nm], [sw1.floor], 'o-', 'LineWidth', 1.5);  hold on;  loglog([sw1.nm], [sw1.floor_ph], 's-', 'LineWidth', 1.5);
    loglog(sw1(kp).nm, sw1(kp).floor_ph, 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y');
    yline(e0.c_true, 'k--', 'static dark zone', 'LabelHorizontalAlignment', 'left');
    grid on;  xlabel('probe amplitude, nm peak');  ylabel('estimation floor, mean |\DeltaE|^2 / peak');
    legend('noiseless', sprintf('%.0e photons', Q.photons_efc), 'the pick', 'Location', 'northwest');  title('Floor against probe amplitude (static state)');
    subplot(2, 3, 2);
    semilogx([sw1.nm], max(abs(reshape([sw1.ratio], [], numel(sw1)) - 1), [], 1), 'o-', 'LineWidth', 1.5);  hold on;  xline(e0.probe_nm, 'k--', 'default rule');
    grid on;  xlabel('probe amplitude, nm peak');  ylabel('worst |ratio - 1|');  title('Probe-intensity diagnostic');
    subplot(2, 3, 3);
    phx = [sw2.ph];  phx(phx == 0) = 1e12;
    loglog(phx, [sw2.floor], 'o-', 'LineWidth', 1.5);  hold on;  loglog(phx, [sw3.floor], 's-', 'LineWidth', 1.5);
    yline(e1.c_true, 'k--', 'static', 'LabelHorizontalAlignment', 'left');  yline(ed.c_true, 'k:', 'dug', 'LabelHorizontalAlignment', 'left');
    grid on;  set(gca, 'XDir', 'reverse');  xticks(sort(phx));  xticklabels(regexprep(compose('%.0e', sort(phx)), '1e\+12', 'none'));
    xlabel('photons per image at the bare peak');  ylabel('estimation floor');  legend('static state', 'dug state', 'Location', 'southwest');  title(sprintf('Floor against photon count (probe %.1fx the default)', f_pick));
    subplot(2, 3, 4);
    histogram(log10(e0.err_px), 60);  hold on;  histogram(log10(e1.err_px), 60);  histogram(log10(ed.err_px), 60);  grid on
    xlabel('log_{10} |\DeltaE|^2 / peak, per pixel');  ylabel('pixels');  legend('static, default rule', 'static, the pick', 'dug, the pick');  title('Per-pixel estimation error (no cut)');
    img = @(v) accum_(v, e1.dz_idx, J.N, J.center_px, J.lamD_px, J.outer_lamD);
    subplot(2, 3, 5);  [A, ax] = img(e1.amp_err);  imagesc(ax, ax, A.', 'AlphaData', ~isnan(A.'));  axis image xy;  colorbar;  clim([-0.2 0.2]);
    title('static, the pick: amplitude error, |E_{est}|/|E_{true}| - 1');  xlabel('\lambda/D');  ylabel('\lambda/D');
    subplot(2, 3, 6);  [A, ax] = img(e1.phase_err);  imagesc(ax, ax, A.', 'AlphaData', ~isnan(A.'));  axis image xy;  colorbar;  clim([-0.2 0.2]);
    title('static, the pick: phase error, rad');  xlabel('\lambda/D');  ylabel('\lambda/D');
    sgtitle(f, sprintf('e2e s10: pairwise probing on the telescope + CTB (model %d, %d pairs; default %.1f nm, the pick %.1fx)', MODEL, Q.npairs, e0.probe_nm, f_pick));
    print(f, fullfile(exdir, 's10_probe.png'), '-dpng', '-r110');  close(f);
    f = figure('Visible', 'off', 'Position', [0 0 900 520], 'Color', 'w');
    n = @(c) 0:numel(c)-1;  n9 = numel(S9.efc_contrast) - 1;  nE = numel(oE.contrast) - 1;
    semilogy(n(S9.efc_contrast), S9.efc_contrast, 'k-o', 'LineWidth', 1.5);  hold on
    semilogy(n9 + n(S9.efc_contrast_r1), S9.efc_contrast_r1, 'k-o', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    semilogy(n(oE.contrast), oE.contrast, 'b-s', 'LineWidth', 1.5);  semilogy(n(oE.contrast), oE.contrast_est, 'b:s', 'LineWidth', 1.2);
    semilogy(nE + n(oE2.contrast), oE2.contrast, 'b-s', 'LineWidth', 1.5, 'HandleVisibility', 'off');  semilogy(nE + n(oE2.contrast), oE2.contrast_est, 'b:s', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    semilogy(n(oP.contrast), oP.contrast, 'r-^', 'LineWidth', 1.5);  semilogy(n(oP.contrast), oP.contrast_est, 'r:^', 'LineWidth', 1.2);
    semilogy(n(oD.contrast), oD.contrast, '-d', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
    grid on;  xlabel('EFC iteration (the relinearized round continues the count)');  ylabel('dark-zone mean contrast, 3-15 \lambda/D');
    legend('perfect sensing (stage 9)', sprintf('probe estimate, %.1fx: true contrast', f_pick), 'probe estimate: estimated', sprintf('%.0e photons: true', Q.photons_efc), ...
        sprintf('%.0e photons: estimated', Q.photons_efc), 'the default rule: true', 'Location', 'northeast');
    title('e2e s10: EFC on the telescope + CTB closed on pairwise-probe estimates');
    print(f, fullfile(exdir, 's10_efc.png'), '-dpng', '-r110');  close(f);
catch ME, say('    (figure: %s)', ME.message);
end
rep = {};
rep{end+1} = sprintf('e2e s10 -- pairwise probing on the coupled telescope + CTB model (%s)', datestr(now, 'yyyy-mm-dd'));
rep{end+1} = sprintf('deck %s, Lyot blocker %.2f; Jacobians %s (%s) and %s (%s); %d pairs; %.2f s per chain run', r.rx_out, li_pick, J.tag, J.date, J2.tag, J2.date, Q.npairs, dt1/e0.nruns);
rep{end+1} = sprintf('static state, the default rule: true %.4e, estimated %.4e, probes %.2f nm peak (%.2f rad); floor %.3e (%.1f %%), relative error %.3e, ratios %s, conditioning %.1f', ...
    e0.c_true, e0.c_est, e0.probe_nm, rad(e0.probe_nm), e0.floor, 100*e0.floor/e0.c_true, e0.rel_err, mat2str(e0.ratio, 4), max(e0.cond2));
rep{end+1} = sprintf('   per-pixel error median %.2e, 90%% %.2e, 99%% %.2e, max %.2e', q0);
for k = 1:numel(sw1), rep{end+1} = sprintf('   amplitude %.1fx = %6.2f nm (%.2f rad): noiseless floor %.3e, error %.3e, ratios %s; at %.0e photons floor %.3e, error %.3e', ...
        sw1(k).rel, sw1(k).nm, rad(sw1(k).nm), sw1(k).floor, sw1(k).err, mat2str(sw1(k).ratio, 4), Q.photons_efc, sw1(k).floor_ph, sw1(k).err_ph); end
rep{end+1} = sprintf('the pick: %.1fx the default = %.2f nm (%.2f rad); noiseless: estimated %.4e, floor %.3e (%.2f %%), relative error %.3e, ratios %s, conditioning %.1f', ...
    f_pick, e1.probe_nm, rad(e1.probe_nm), e1.c_est, e1.floor, 100*e1.floor/e1.c_true, e1.rel_err, mat2str(e1.ratio, 4), max(e1.cond2));
rep{end+1} = sprintf('   per-pixel error median %.2e, 90%% %.2e, 99%% %.2e, max %.2e', q1);
for k = 1:numel(sw2), rep{end+1} = sprintf('   photons %-8s: floor %.3e, relative error %.3e', phname_(sw2(k).ph), sw2(k).floor, sw2(k).rel); end
rep{end+1} = sprintf('dug state at the pick: true %.4e, estimated %.4e, probes %.2f nm peak (%.2f rad); floor %.3e (%.2f %%), relative error %.3e, ratios %s, conditioning %.1f', ...
    ed.c_true, ed.c_est, ed.probe_nm, rad(ed.probe_nm), ed.floor, 100*ed.floor/ed.c_true, ed.rel_err, mat2str(ed.ratio, 4), max(ed.cond2));
rep{end+1} = sprintf('   per-pixel error median %.2e, 90%% %.2e, 99%% %.2e, max %.2e', qd);
rep{end+1} = sprintf('   the control at the Jacobian''s own point (%.4e): estimated %.4e, floor %.3e (%.2f %%), relative error %.3e, ratios %s', ec.c_true, ec.c_est, ec.floor, 100*ec.floor/ec.c_true, ec.rel_err, mat2str(ec.ratio, 4));
for k = 1:numel(sw3), rep{end+1} = sprintf('   photons %-8s: floor %.3e, relative error %.3e', phname_(sw3(k).ph), sw3(k).floor, sw3(k).rel); end
rep{end+1} = sprintf('EFC on the noiseless estimate at the pick, fixed G: %s', mat2str(oE.contrast, 4));
rep{end+1} = sprintf('   estimated contrast:                  %s', mat2str(oE.contrast_est(1:end-1), 4));
rep{end+1} = sprintf('   stage 9 perfect sensing:             %s', mat2str(S9.efc_contrast, 4));
rep{end+1} = sprintf('relinearized on the estimate:          %s', mat2str(oE2.contrast, 4));
rep{end+1} = sprintf('   stage 9 perfect sensing:             %s', mat2str(S9.efc_contrast_r1, 4));
rep{end+1} = sprintf('EFC at %.0e photons, fixed G:           %s', Q.photons_efc, mat2str(oP.contrast, 4));
rep{end+1} = sprintf('EFC on the default rule, fixed G:      %s', mat2str(oD.contrast, 4));
rep{end+1} = sprintf('   strokes nm rms: pick %s, relinearized %s, photons %s, default rule %s; stage 9 %s / %s', mat2str(oE.stroke_rms_nm, 3), mat2str(oE2.stroke_rms_nm, 3), mat2str(oP.stroke_rms_nm, 3), mat2str(oD.stroke_rms_nm, 3), mat2str(S9.strokes_nm, 3), mat2str(S9.strokes_nm_r1, 3));
rep{end+1} = sprintf('   bench reference (ch08d, plain stop, the default rule = 9.27 nm): perfect 8.96e-9, noiseless estimate 9.08e-9, 1e8 photons 9.37e-9 after 10 iterations');
for k = 1:numel(gn), rep{end+1} = sprintf('   [%s] %s', tern_(G.(gn{k}).ok, 'PASS', 'FAIL'), G.(gn{k}).say); end
rep{end+1} = sprintf('products: s10_probe.png s10_efc.png ctb_efc_e2e_sens{,_r1,_ph,_def}.{mat,png} s10_report.txt s10_run.mat; %.0f s', toc(t0));
fid = fopen(fullfile(exdir, 's10_report.txt'), 'w');  fprintf(fid, '%s\n', rep{:});  fclose(fid);
S10 = struct('G', G, 'e', e, 'lyot_pick', li_pick, 'probe_nm_default', e0.probe_nm, 'pick_rel', f_pick, 'probe_nm_pick', e1.probe_nm, ...
             'default', slim_(e0), 'static', slim_(e1), 'dug', slim_(ed), 'control', slim_(ec), 'amp', sw1, 'photons', sw2, 'photons_dug', sw3, ...
             'efc', oE.contrast, 'efc_est', oE.contrast_est, 'efc_r1', oE2.contrast, 'efc_r1_est', oE2.contrast_est, 'efc_ph', oP.contrast, 'efc_ph_est', oP.contrast_est, ...
             'efc_default', oD.contrast, 'efc_default_est', oD.contrast_est, ...
             'strokes_nm', oE.stroke_rms_nm, 'strokes_nm_r1', oE2.stroke_rms_nm, 'strokes_nm_ph', oP.stroke_rms_nm, 's9', struct('efc', S9.efc_contrast, 'efc_r1', S9.efc_contrast_r1));
save(fullfile(exdir, 's10_run.mat'), '-struct', 'S10');
say('[5] report s10_report.txt, s10_probe.png, s10_efc.png, s10_run.mat; %s; %.0f s', tern_(allok, 'ALL GATES PASS', 'A GATE FAILED'), toc(t0));

% =================================================================================================
function s = tern_(c, a, b)
    if c, s = a; else, s = b; end
end

function s = phname_(ph)
    if ph == 0, s = 'none'; else, s = sprintf('%.0e', ph); end
end

function s = slim_(est)
%SLIM_  The estimate's scalars, without its dark-zone vectors.
    s = rmfield(est, {'E_hat', 'E_true', 'err_px', 'amp_err', 'phase_err', 'cond2', 'I0', 'Ip', 'Im', 'dE', 'probes', 'dz_idx'});
end

function [A, ax] = accum_(v, idx, N, c, lamD, outer)
    F = nan(N);  F(idx) = v;
    w = ceil((outer + 1) * lamD);  ix = c-w : c+w;
    A = F(ix, ix);  ax = (ix - c) / lamD;
end
