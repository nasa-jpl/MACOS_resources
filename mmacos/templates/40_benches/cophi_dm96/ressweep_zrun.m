function out = ressweep_zrun(varargin)
%RESSWEEP_ZRUN  The resolution sweep for the ZERNIKE-family gauges -- stepped ZWFS S,
%   vector ZWFS V, pinhole P (one zwfs_run per sampling reads all three) -- the
%   twin of ressweep_run (IFO / COPHI).  BRIEF_to_resolution_sweep, 2026-10-13.
%
%   Usage (interactive; NO exit -- ressweep_zrun_batch is the -batch wrapper):
%     ressweep_zrun('s', [3.30 2 1.5 1 4])            % the model-2048 samplings
%     ressweep_zrun('s', 1.65)                        % the model-1024 one, in its OWN process
%
%   The zwfs bench of record (zwfs_dm96 runs redo96_oapsens / camS_oap: the mirror
%   rig, MASK_TRIM 0.632, the engine arm maps, the matrix on the 30 nm surface).
%   Its records sit at TWO samplings: the battery at NGRID 385 (3.30 model px per
%   actuator, model 2048, the trimmed size table) and the camera rows at NGRID 193
%   (1.65, model 1024).  So: 3.30 = the battery anchor (identity), and 2 / 1.5 / 1
%   through the camera's pixels on the 385 grid; 1.65 NATIVE at NGRID 193 (the
%   camera rows' anchor -- a point-sampled grid, set against the binned 1.65 the
%   sweep does not run); 4 = NGRID 467 at model 2048 (a finer source grid).
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
addpath(exdir);
zdir = fullfile(exdir, '..', 'zwfs_dm96');  addpath(zdir);
o = struct('s', [3.30 2 1.5 1 4], 'ppa_model', 3.30, 'nreal', 12, 'loop', true, 'reg_pitch', 0.1, ...
    'report', false, 'over', {{}}, 'gdir', '');
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
gdir = o.gdir;  if isempty(gdir), gdir = fullfile(exdir, 'runs', 'ressweep_zwfs'); end
if ~exist(gdir, 'dir'), mkdir(gdir); end
resf = fullfile(gdir, 'ressweep_zwfs.mat');
res = struct('s_target', {}, 'tag', {}, 'R', {}, 'rows', {}, 'wall_min', {});
if isfile(resf), L = load(resf);  res = L.res; end
% ONE engine model size per MATLAB process (zwfs_run's rule: a second macos.init at
% another size corrupts the heap -- the 3.30 -> 1.65 step of the first sweep died
% 'free(): invalid next size' on 2026-10-13): 1.65 runs at model 1024, every other
% sampling at 2048, so a call may not mix them
assert(~any(abs(o.s - 1.65) < 1e-6) || all(abs(o.s - 1.65) < 1e-6), ...
    'ressweep_zrun: run s = 1.65 (model 1024) in its OWN process, apart from the model-2048 samplings');
if ~o.report
    for s = o.s
        tag = sprintf('zwfs_s%03d', round(100*s));
        P = zwfs_params();
        % the bench of record (redo96_oapsens / camS_oap call lines)
        P.bench.optics = 'oap';  P.bench.OAP1_AOI = 20;  P.bench.OAP2_AOI = 25;
        P.bench.OAP1_SIDE = 1;  P.bench.OAP2_SIDE = -1;  P.bench.MASK_TRIM = 0.632;
        P.mask.v_arm = 'engine';  P.battery.calib_surface = 'base';
        P.readings = {'S', 'V', 'P'};  P.stages = {'bench', 'battery'};
        P.battery.rows = {'base/single', 'base/grid', 'base/rand'};
        P.battery.ladder = P.battery.base_rms;                 % one rung (the capture range is not this sweep's)
        P.dm = P.dm(1);                                        % the 96 x 96 DM only
        if abs(s - 1.65) < 1e-6
            P.MODEL = 1024;  P.NGRID = 193;  P.camres = [];    % the camera rows' native grid
        elseif s > o.ppa_model + 1e-6
            P.MODEL = 2048;  P.NGRID = round(385 * s / o.ppa_model);  P.camres = [];
            P.param_file = 'macos_param_2048.txt';
        else
            P.MODEL = 2048;  P.NGRID = 385;  P.param_file = 'macos_param_2048.txt';
            if abs(s - o.ppa_model) < 1e-6, P.camres = [];
            else, P.camres = struct('c', o.ppa_model / s, 'offset', [0 0], 'center', []); end
        end
        P.tag = tag;  P.outdir = fullfile(gdir, tag);
        P.battery.deck_hook = 'ressweep_zhook';
        mono = struct('tag', 'mono', 'name', 'IMX250 mono class (well ~10 ke UNCONFIRMED, read 2.4 e- ASSUMED)', ...
            'npix', 2048, 'well', 1e4, 'read', 2.4, 'rate', 75, 'fill', 0.5, 'pol', 0);
        pol = mono;  pol.tag = 'polcam';  pol.name = 'IMX250MZR class polarization camera';  pol.pol = 1;
        P.ressweep = struct('cams', struct('S', mono, 'V', pol, 'P', mono), 'nreal', o.nreal, 'loop', o.loop, ...
            'reg_pitch', o.reg_pitch, ...
            'M', struct('S', 330, 'V', 1435, 'P', 237));     % ~5.2 pm per cycle = a 3 pm hold at camS_oap's pm per reading (94.4 / 197 / 80)
        for q = 1:2:numel(o.over)
            parts = strsplit(o.over{q}, '.');  P = setfield(P, parts{:}, o.over{q+1}); %#ok<SFLD>
        end
        fprintf('\n===== ressweep ZWFS: s %.3f px/act (model %d, NGRID %d, c %s) -> %s =====\n', s, P.MODEL, P.NGRID, ...
            iff_(isempty(P.camres), '1', sprintf('%.4f', o.ppa_model/s)), P.outdir);
        t0 = tic;
        r = zwfs_run(P);
        j = find(strcmp({res.tag}, tag), 1);  if isempty(j), j = numel(res) + 1; end
        res(j) = struct('s_target', s, 'tag', tag, 'R', r.battery.deck_hook, 'rows', r.battery.n96.res, 'wall_min', toc(t0)/60);
        save(resf, 'res', '-v7.3');
        write_table_(res, gdir);
    end
end
write_table_(res, gdir);
% the PRUNED record (runs/.gitignore commits *_results.mat): the rows and the hook's
% numbers, no parameter sheets -- what tests/tResolutionSweep reads on a clone
resp = res;  if isfield(resp, 'P'), resp = rmfield(resp, 'P'); end
save(fullfile(gdir, 'ressweep_zwfs_results.mat'), 'resp', '-v7');
out = struct('opt', o, 'res', res);
end

function write_table_(res, gdir)
if isempty(res), return; end
[~, ix] = sort([res.s_target], 'descend');  res = res(ix);
f = fopen(fullfile(gdir, 'ressweep.txt'), 'w');  cl = onCleanup(@() fclose(f));
pr = @(varargin) fprintf(f, varargin{:});
pr('ressweep ZWFS family (S stepped, V vector, P pinhole) -- %s\n', char(datetime('now')));
pr('px/act = camera pixels per actuator (3.30 at NGRID 385, model 2048; 1.65 = NGRID 193 native, model 1024; > 3.30 finer source grid).\n');
pr('Battery (noise-free, 30 nm surface, matrix on it): gain one-step / converged, err pm one-step / converged, floor pm one-step\n');
for j = 1:numel(res)
    R = res(j).R;  rw = res(j).rows;
    for q = 1:numel(rw)
        rn = find(strcmp(R.rows_conv_names, rw(q).row));  k = find(strcmp(R.readings, rw(q).rd));
        cv = squeeze(R.rows_conv(rn, k, :));
        pr('%7.3f %-17s %-2s %7.4f / %7.4f  %7.1f / %7.1f  %5s\n', R.s, rw(q).row, rw(q).rd, rw(q).raw(1), cv(1), rw(q).raw(2), cv(2), f0_(rw(q).raw(3)));
    end
end
pr('\nRegistration (+-0.1 pitch after calibration; worst of four shifts per reading): gain one-step / converged, largest neighbour / dev, wrong of 4\n');
for j = 1:numel(res)
    R = res(j).R;
    for k = 1:numel(R.readings)
        g = squeeze(R.reg(2:end, k, :));  [~, w] = max(abs(g(:, 1) - 1));
        pr('%7.3f %-2s at 0: g %.4f nbr %.4f | worst: g %.4f / %.4f, nbr %.4f, flr %.2f, wrong %d of 4\n', R.s, R.readings{k}, ...
            R.reg(1, k, 1), R.reg(1, k, 3), g(w, 1), g(w, 2), g(w, 3), g(w, 4), nnz(g(:, 5)));
    end
end
pr('\nNoise (one reading) and data:\n');
for j = 1:numel(res)
    R = res(j).R;
    for q = 1:numel(R.noise)
        n = R.noise(q);
        pr('%7.3f %-2s %-7s %-7s read %7.3g  pm/read %8.3g  s per pm %9.3g  out px %9.3g  Gb/s %7.3g  pupil px %9.3g\n', R.s, n.rd, n.cam, n.binning, ...
            n.read_e, n.pm_per_reading, n.s_per_pm, n.npix_out, n.gbps, R.pupil_px);
    end
end
pr('\nLoop (linear instrument; digital binning):\n');
for j = 1:numel(res)
    R = res(j).R;
    for q = 1:numel(R.loop)
        L = R.loop(q);
        pr('%7.3f %-2s M %5d (%.3g s) %-8s ss %.3g (pf %.3g) sig_n %.3g theory %.3g%s\n', R.s, L.rd, L.M, L.cycle_s, L.drift, L.ss_pm, L.ss_pf_pm, L.sig_n_pm, L.theory_pm, iff_(L.diverged, ' DIVERGED', ''));
    end
end
pr('\nWall: ');  for j = 1:numel(res), pr('%s %.0f min; ', res(j).tag, res(j).wall_min); end
pr('\n');
end

function s = f0_(x), if isnan(x), s = '-'; else, s = sprintf('%.0f', x); end, end
function s = iff_(c, a, b), if c, s = a; else, s = b; end, end
