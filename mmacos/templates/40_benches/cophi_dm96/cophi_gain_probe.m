function R = cophi_gain_probe(optics)
%COPHI_GAIN_PROBE  Isolate the H reading's 0.5 % low gain (stage battery, 2026-10-11).
%   The battery's H rows read 0.994-0.997 on both rigs where the brief expected
%   1.000 +- the matrix bias.  Four discriminators (CC's list), one MATLAB:
%     #1 SYNTHETIC, no estimator: a 1.000 nm single-actuator poke at the hold-out
%        site; the H reading's pixel surface vs the ENGINE's own field change
%        (the test arm alone), least-squares slope over the poke's footprint
%     #2 the LO := the test arm's own flat-DM field (no placement) -> rows
%     #3 the scalar trace (polarization off) -> rows
%     #4 the fit on the TEST beam's support, the lit set forced to the
%        battery's (same surface) -> rows
%   Rows = the record's deck rows only (single / grid / rand on the 30 nm
%   surface, matrix on it); no ladders.  Writes runs/gainprobe_<optics>/.
%   NB the matrix is measured THROUGH the reading it inverts, so a uniform
%   reading gain cancels: #1 = 1.000 with rows still 0.995 says the 0.5 % is the
%   estimator's, and #2 then says whether ANY reading reads it.
if nargin < 1, optics = 'lens'; end
exdir = fileparts(mfilename('fullpath'));
if isempty(which('macos.init')), run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m')); end
addpath(exdir);
P0 = cophi_params();
run_tag = ['battery_' optics];
src = fullfile(exdir, 'runs', run_tag);
out = fullfile(exdir, 'runs', ['gainprobe_' optics]);  if ~exist(out, 'dir'), mkdir(out); end
fid = fopen(fullfile(out, ['gainprobe_' optics '_report.txt']), 'w');  cl = onCleanup(@() fclose(fid));
say = @(varargin) [fprintf(varargin{:}), fprintf(fid, varargin{:})];
say('=== COPHI gain probe (%s rig), %s ===\n', optics, char(datetime('now')));
B = load(fullfile(src, [run_tag '_cophi_battery_results.mat']));  DK = B.battery.deck;
say('battery record %s: H rows %s\n', run_tag, strjoin(cellfun(@(r, g) sprintf('%s %.4f', r, g), DK.rows(:,1), DK.rows(:,2), 'UniformOutput', false), ', '));
ov = {'tag', run_tag, 'bench.optics', optics};

% ---- #1 synthetic gain, no estimator --------------------------------
t1 = tic;
P = apply_(P0, ov{:});  P.outdir = src;
if strcmp(optics, 'oap')
    P.bench.POL_IN = 'source';  P.bench.D_RC_L2 = 125;
    P.oap.OAP1_AOI = 20;  P.oap.OAP2_AOI = 25;  P.oap.OAP1_SIDE = 1;  P.oap.OAP2_SIDE = -1;
end
P.bench.tail_file = P.cophi.record_tail.(optics);
S = load(fullfile(src, [run_tag '_cophi.mat']));  G = S.out.bench.G;
old = cd(src);  cd_back = onCleanup(@() cd(old));
macos.init(P.MODEL);
C = cophi_ctx(P, G, @(varargin) []);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
cfg = P.dm(1);  A = zeros(cfg.nact);  A(DK.ic(1), DK.ic(2)) = 1e-6;     % 1.000 nm (mm units)
M1 = dm_influence_map(P.grid.N_G, P.grid.DX_G, 'nact', cfg.nact, 'pitch', cfg.pitch, 'act', A);
h  = C.measf(M1);
E0 = C.fieldf(zeros(P.grid.N_G));  E1 = C.fieldf(M1);
dphi = angle(sum(E1 .* conj(E0), 3));
ht = dphi * P.LAM/(4*pi);
foot = abs(ht) > 0.3*max(abs(ht(:)));
g1 = ht(foot) \ h(foot);
pk = max(abs(h(:))) / max(abs(ht(:)));
say('\n#1 SYNTHETIC (1.000 nm at (%d,%d), no estimator): H pixel surface vs the engine''s own field change over the poke footprint (%d px):\n', DK.ic(1), DK.ic(2), nnz(foot));
say('   slope h_H / h_engine = %.6f (peak ratio %.6f); engine peak %.4f nm  [%.0f s]\n', g1, pk, 1e6*max(abs(ht(:))), toc(t1));
R.g1 = g1;  R.pk1 = pk;
clear cd_back

% ---- #2-#4: the rows under each variant ------------------------------
rowsonly = {'battery.cap_ladder', 30e-6, 'battery.recap_surf', []};
V = {'2_lo_self',   {'cophi.diag_lo', 'self'}; ...
     '3_scalar',    {'cophi.scalar', true}; ...
     '4_mask_test', {'cophi.diag_mask', 'test', 'place.lit_force', DK.lit}};
for v = 1:size(V, 1)
    tv = tic;
    tg = ['gp' V{v,1} '_' optics];
    args = [{'tag', tg, 'outdir', fullfile(out, tg), 'bench.optics', optics, 'stages', {'bench','battery'}}, rowsonly, V{v,2}];
    o = cophi_run(args{:});
    D = o.battery.deck;
    say('\n#%s: rows (gain / err pm / floor pm), lit %d  [%.1f min]\n', V{v,1}, D.nlit, toc(tv)/60);
    for r = 1:size(D.rows, 1)
        say('   %-18s %7.4f %8.1f %6s   (battery H %.4f)\n', D.rows{r,1}, D.rows{r,2}, D.rows{r,3}, num2str(D.rows{r,4}, '%.0f'), DK.rows{r,2});
    end
    R.(['v' V{v,1}]) = D.rows;
end
save(fullfile(out, ['gainprobe_' optics '_results.mat']), 'R');
end

function P = apply_(P, varargin)
    for k = 1:2:numel(varargin)
        parts = strsplit(varargin{k}, '.');  P = setfield(P, parts{:}, varargin{k+1}); %#ok<SFLD>
    end
end
