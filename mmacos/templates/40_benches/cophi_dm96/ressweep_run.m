function out = ressweep_run(varargin)
%RESSWEEP_RUN  The per-gauge RESOLUTION SWEEP (BRIEF_to_resolution_sweep, 2026-10-13):
%   what pixel count per actuator buys and costs, gauge by gauge, on the bench of
%   record.  One runner (Dave's standing rule); edit the call line, rerun.
%
%   Usage (interactive; NO exit -- ressweep_run_batch is the -batch wrapper):
%     ressweep_run('gauge','ifo')                     % IFO snapshot, mirror rig of record
%     ressweep_run('gauge','het')                     % COPHI four-bucket camera (plain splitter)
%     ressweep_run('gauge','het_iq')                  % COPHI lock-in reading (cophi.iq)
%     ressweep_run('gauge','ifo','s',[3.277 1])       % a subset of samplings
%     ressweep_run('gauge','ifo','report',true)       % re-print the table from the saved runs
%
%   MECHANISM (the README says why).  s <= the model's own sampling (3.277 model px
%   per actuator at NGRID 385): the CAMERA'S PIXELS -- every detector frame is
%   area-integrated onto camera pixels of c = 3.277/s model px and replicated back
%   (dm_gauge_lib/dmg_camres, set through tg96_run's P.camres), so the mask, the
%   placement, the 64-state matrix and the battery rows all run THROUGH the
%   resampled reading.  s > 3.277: a finer source grid (NGRID scaled), c = 1.  The
%   anchor (3.277, c = 1) is the identity -- the record's code path bit for bit.
%   Per sampling the deck stage's hook (ressweep_hook) adds the noise rows on the
%   gauge's record camera, the +-0.1 pitch registration rows, the hold loop and
%   the data rate.
%
%   Writes runs/ressweep_<gauge>/<gauge>_s<NNN>/ (tg96_run's report + .mat per
%   sampling) and runs/ressweep_<gauge>/ressweep.txt + ressweep_<gauge>.mat.
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
addpath(exdir);
o = struct('gauge', 'ifo', 's', [3.277 2 1.5 1 4], 'ppa_model', 3.277, 'ngrid', 385, ...
    'report', false, 'nreal', 12, 'loop', true, 'reg_pitch', 0.1, 'model', 1024, ...
    'over', {{}}, 'gdir', '');   % over: dotted P overrides AFTER the gauge's sheet (smoke: matrix_states, K)
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
gdir = o.gdir;  if isempty(gdir), gdir = fullfile(exdir, 'runs', ['ressweep_' o.gauge]); end
if ~exist(gdir, 'dir'), mkdir(gdir); end
resf = fullfile(gdir, ['ressweep_' o.gauge '.mat']);

res = struct('s_target', {}, 'tag', {}, 'R', {}, 'P', {}, 'wall_min', {});
if isfile(resf), L = load(resf);  res = L.res; end
if ~o.report
    for s = o.s
        tag = sprintf('%s_s%03d', o.gauge, round(100*s));
        P = gauge_params_(o, exdir, fullfile(gdir, tag), tag);
        for q = 1:2:numel(o.over)
            parts = strsplit(o.over{q}, '.');  P = setfield(P, parts{:}, o.over{q+1}); %#ok<SFLD>
        end
        if s > o.ppa_model + 1e-6
            P.NGRID = round(o.ngrid * s / o.ppa_model);  P.camres = [];   % finer source grid
        elseif abs(s - o.ppa_model) < 1e-6
            P.camres = [];                                                % the anchor: the identity
        else
            P.camres = struct('c', o.ppa_model / s, 'offset', [0 0], 'center', []);
        end
        fprintf('\n===== ressweep %s: s %.3f px/act (NGRID %d, c %s) -> %s =====\n', o.gauge, s, P.NGRID, ...
            iff_(isempty(P.camres), '1', sprintf('%.4f', o.ppa_model/s)), P.outdir);
        t0 = tic;
        r = tg96_run(P);
        j = find(strcmp({res.tag}, tag), 1);  if isempty(j), j = numel(res) + 1; end
        Pk = P;  if isfield(Pk, 'ressweep'), Pk = rmfield(Pk, 'ressweep'); end
        res(j) = struct('s_target', s, 'tag', tag, 'R', r.deck.deck_hook, 'P', Pk, 'wall_min', toc(t0)/60);
        save(resf, 'res', '-v7.3');
        write_table_(o, res, gdir);                % after EVERY sampling: a killed run keeps its rows
    end
end
write_table_(o, res, gdir);
% the PRUNED record (runs/.gitignore commits *_results.mat): the rows and the hook's
% numbers, no parameter sheets -- what tests/tResolutionSweep reads on a clone
resp = res;  if isfield(resp, 'P'), resp = rmfield(resp, 'P'); end
save(fullfile(gdir, ['ressweep_' o.gauge '_results.mat']), 'resp', '-v7');
out = struct('opt', o, 'res', res);
end

% =====================================================================
function P = gauge_params_(o, exdir, outdir, tag)
% the bench of record for each gauge (the composite deck's rows)
switch o.gauge
    case 'ifo'
        % the IFO snapshot, mirror rig of record (tg_psi_dm96_oap runs camI_oap /
        % redo96_oapph: the redo tail, POL_IN source, AOIs 20/25, D_RC_L2 125)
        addpath(fullfile(exdir, '..', 'tg_psi_dm96_oap'));
        P = tg96_params();
        P.bench.optics = 'oap';  P.bench.POL_IN = 'source';  P.bench.D_RC_L2 = 125;
        P.oap.OAP1_AOI = 20;  P.oap.OAP2_AOI = 25;  P.oap.OAP1_SIDE = 1;  P.oap.OAP2_SIDE = -1;
        P.bench.tail_file = 'redo96_oap_tail.mat';
        cams = cam_('polcam', 'IMX250MZR class polarization camera (the record IFO camera row; well ~10 ke UNCONFIRMED, read 2.4 e- ASSUMED)', ...
            2048, 1e4, 2.4, 75, 1, 1, false, 450);
    case {'het', 'het_iq'}
        % COPHI, mirror rig of record, the PLAIN splitter (the priced layout), the
        % 'median' annulus fill (ruled 2026-10-12); cophi_run prepares the sheet
        args = {'bench.optics', 'oap', 'bench.polarizing', false, 'cophi.annulus_fill', 'median', ...
                'outdir', outdir, 'tag', tag, 'stages', {}};
        if strcmp(o.gauge, 'het_iq'), args = [args, {'cophi.iq', true}]; end
        r = cophi_run(args{:});  P = r.P;
        P.reading_ctx = 'cophi_ctx';
        if strcmp(o.gauge, 'het')
            cams = [cam_('photonfocus', 'Photonfocus MV1-D1024E-160 (200 ke, 220 e-)', 1024, 200e3, 220, 150, P.cophi.N, 0, false, 60), ...
                    cam_('baseline', 'Feng''s camera as proposed (1k, 60 ke, 30 fps, read 5 e- ASSUMED)', 1024, 60e3, 5, 30, P.cophi.N, 0, false, 120)];
        else
            cams = [cam_('lockin', 'heliCam C4 class lock-in (512 sites, 500 ke, 1 kHz I/Q; read 0)', 512/0.95, 500e3, 0, 1e3, 1, 0, true, 125), ...
                    cam_('lockin_r', 'the same, read 100 e- per sample ASSUMED', 512/0.95, 500e3, 100, 1e3, 1, 0, true, 125)];
        end
    otherwise
        error('ressweep_run: gauge must be ifo | het | het_iq (the ZWFS gauges need their own hook)');
end
P.tag = tag;  P.outdir = outdir;  P.MODEL = o.model;  P.NGRID = o.ngrid;
P.stages = {'bench', 'deck'};
P.battery.deck = true;  P.battery.noise = false;  P.pzt.step_err = 0;
P.battery.cap_ladder = P.battery.base_rms;  P.battery.recap_surf = [];
P.battery.est_mode = 'both';
if isfield(P.battery, 'noise_cam'), P.battery.noise_cam = []; end
P.battery.deck_hook = 'ressweep_hook';
P.ressweep = struct('cams', cams, 'nreal', o.nreal, 'loop', o.loop, 'reg_pitch', o.reg_pitch);
end

function c = cam_(tag, name, npix, well, read, rate, fpr, pol, iq, M)
c = struct('tag', tag, 'name', name, 'npix', npix, 'well', well, 'read', read, 'rate', rate, ...
    'fpr', fpr, 'pol', pol, 'iq', iq, 'M', M, 'fill', 0.5);
end

function write_table_(o, res, gdir)
if isempty(res), return; end
[~, ix] = sort([res.s_target], 'descend');  res = res(ix);
fn = fullfile(gdir, 'ressweep.txt');  f = fopen(fn, 'w');  cl = onCleanup(@() fclose(f));
pr = @(varargin) fprintf(f, varargin{:});
pr('ressweep %s -- %s\n', o.gauge, char(datetime('now')));
pr('px/act = camera pixels per actuator pitch (model 3.277 at NGRID 385; > 3.277 = finer source grid).\n');
pr('Battery (noise-free, the 30 nm working surface, matrix on it; one-step | converged): gain / err pm / floor pm\n');
pr('%7s %6s | %-24s | %-24s | %-24s\n', 'px/act', 'lit', 'single 10 nm', 'grid 1 nm', 'dense random 10 nm');
for j = 1:numel(res)
    R = res(j).R;  pr('%7.3f %6d |', R.s, R.nlit);
    for q = 1:3
        a = R.rows(q, :);  b = R.rows_conv(q, :);
        pr(' %6.4f/%6.4f %5.1f/%5.1f %3s|', a{2}, b{2}, a{3}, b{3}, f0_(a{4}));
    end
    pr('\n');
end
pr('\nRegistration (camera shifted +-%g pitch after calibration; worst of the four shifts): gain one-step / converged, largest neighbour / dev, noise-free floor pm, wrong actuator\n', o.reg_pitch);
for j = 1:numel(res)
    R = res(j).R;  g = R.reg(2:end, :);  [~, w] = max(abs(g(:, 3) - 1));
    pr('%7.3f | at 0: g %.4f, nbr %.4f, flr %.2f | worst (%+.2f,%+.2f px): g %.4f / %.4f, nbr %.4f, flr %.2f, wrong %d of 4\n', ...
        R.s, R.reg(1, 3), R.reg(1, 5), R.reg(1, 6), g(w, 1), g(w, 2), g(w, 3), g(w, 4), g(w, 5), g(w, 6), nnz(g(:, 8)));
end
pr('\nNoise (one reading; pm per reading = piston-free robust floor; s per pm = floor^2 / readings per s) and data:\n');
pr('%7s %-12s %-8s %8s %8s %9s %10s %9s %10s\n', 'px/act', 'camera', 'binning', 'read e-', 'pm/read', 's per pm', 'out px', 'Gb/s', 'pupil px');
for j = 1:numel(res)
    R = res(j).R;
    for q = 1:numel(R.noise)
        n = R.noise(q);
        pr('%7.3f %-12s %-8s %8.3g %8.3g %9.3g %10.3g %9.3g %10.3g\n', R.s, n.cam, n.binning, n.read_e, n.pm_per_reading, n.s_per_pm, n.npix_out, n.gbps, R.pupil_px);
    end
end
pr('\nLoop (linear instrument = this sampling''s matrix + the camera''s noise; g 0.5, K 60, seed 77; digital binning): hold ss pm (piston-free)\n');
for j = 1:numel(res)
    R = res(j).R;
    for q = 1:numel(R.loop)
        L = R.loop(q);
        pr('%7.3f %-12s M %4d (%.3g s) %-8s ss %.3g (pf %.3g)  sig_n %.3g  theory %.3g  gG %.3f%s\n', R.s, L.cam, L.M, L.cycle_s, L.drift, L.ss_pm, L.ss_pf_pm, L.sig_n_pm, L.theory_pm, L.gG, iff_(L.diverged, ' DIVERGED', ''));
    end
end
pr('\nWall: ');  for j = 1:numel(res), pr('%s %.0f min; ', res(j).tag, res(j).wall_min); end
pr('\n');
end

function s = f0_(x), if isnan(x), s = '-'; else, s = sprintf('%.0f', x); end, end
function s = iff_(c, a, b), if c, s = a; else, s = b; end, end
