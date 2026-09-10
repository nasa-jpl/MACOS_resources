function out = zwfs_run(varargin)
%ZWFS_RUN  The ZWFS DM-gauge runner: build, gate, calibrate, measure.
%   out = zwfs_run()                      every default (zwfs_params): the S7 record
%   out = zwfs_run(P)                     a parameter struct from zwfs_params, edited
%   out = zwfs_run('NGRID',385, ...)      call-line overrides (top-level or dotted names,
%                                         e.g. 'mask.DIA_LAMD',3, 'stages',{'battery','figs'})
%   out = zwfs_run(P, 'tag','x', ...)     both
%
%   One entry point for the Zernike-wavefront-sensor gauge of a 96 mm
%   deformable mirror (templates/40_benches/zwfs_dm96/README.md).  It
%   builds the bench (the TG96 test arm with the dimple mask at its
%   internal focus), asserts the sampling budget and the sensor gates,
%   registers the camera to the DM by the two-poke doctrine, calibrates
%   each reading's response kernel and modal transfer, and measures the
%   differential rows and the working-state ladder -- in ACTUATOR space,
%   all errors in pm.  Optional stages add the multi-color combination and
%   the photon-noise pricing.  Every number goes to the report; every
%   gate prints its value and its threshold.
%
%   OUTPUTS (in <this dir>/runs/<tag>/ unless P.outdir is set):
%     <tag>_report.txt    the record (console is tee'd)
%     <tag>.mat           out = struct(P, bench, battery, color, noise)
%     <tag>_test.in       the emitted deck (+ per-color copies)
%     <tag>_*.png         figures (stage 'figs'; also zwfs_run_figs(out))
%
%   RULES OF THE ROAD: one engine model size per MATLAB process (a second
%   macos.init at another size corrupts the heap); run MODEL 1024 jobs
%   sequentially, memory-capped (README "Run it yourself" has the batch
%   line); readings and stages are subsets of zwfs_params' lists.
%
%   The scoring machinery is ../dm_gauge_lib (shared with the
%   interferometer); the measurement factory is dmg_zwfs_gauge.

P = parse_(varargin{:});
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
addpath(exdir);                                                     % zwfs_mask
addpath(fullfile(exdir, '..', 'dm_gauge_lib'));                     % dmg_*
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));  % dm_influence_map
if isempty(P.outdir), P.outdir = fullfile(exdir, 'runs', P.tag); end
if ~exist(P.outdir, 'dir'), mkdir(P.outdir); end
cd(P.outdir);
t_all = tic;
rep = fopen(sprintf('%s_report.txt', P.tag), 'w');
want = @(st) any(strcmp(P.stages, st));
out = struct('P', P);

dmg_say(rep, '=== ZWFS runner: tag %s  (%s) ===\n', P.tag, datestr(now, 'yyyy-mm-dd HH:MM'));
dmg_say(rep, 'model %d, NGRID %d, lambda %.1f nm, mask_prop %s, spot %.2f lam/D, etch %.1f nm, NITER %d\n', ...
    P.MODEL, P.NGRID, P.LAM*1e6, P.bench.mask_prop, P.mask.DIA_LAMD, P.mask.ETCH_MM*1e6, P.mask.NITER);
dmg_say(rep, 'stages: %s | readings: %s\n', strjoin(P.stages, ' '), strjoin(P.readings, ' '));
dmg_say(rep, 'readings: L frozen-linear (1 frame) | F exact frozen-b (1) | I exact iterated-b (1) | I+ = I with the base''s refined stepped branch prior (1; base 4 once) | S phase-stepped (4)\n');

S = stage_bench_(P, rep);
out.bench = S.summary;
if want('battery'), out.battery = stage_battery_(P, S, rep); end
if want('color'),   out.color   = stage_color_(P, S, rep);   end
if want('noise'),   out.noise   = stage_noise_(P, S, out, rep); end
dmg_say(rep, 'run complete in %.1f min\n', toc(t_all)/60);
fclose(rep);
save(sprintf('%s.mat', P.tag), 'out');
fprintf('wrote %s\n', fullfile(P.outdir, sprintf('%s_report.txt + %s.mat', P.tag, P.tag)));
if want('figs'), zwfs_run_figs(out); end
end

% =====================================================================
%  parameters
% =====================================================================
function P = parse_(varargin)
if ~isempty(varargin) && isstruct(varargin{1})
    P = varargin{1};  varargin(1) = [];
else
    P = zwfs_params();
end
assert(mod(numel(varargin), 2) == 0, 'zwfs_run: name/value pairs expected');
for i = 1:2:numel(varargin)
    parts = strsplit(varargin{i}, '.');
    P = setfield(P, parts{:}, varargin{i+1}); %#ok<SFLD>
end
if ~isempty(P.dm_use), P.dm = P.dm(P.dm_use); end
if ~isempty(P.hold), for i = 1:numel(P.dm), P.dm(i).hold = P.hold; end; end
ok = {'L','F','I','I+','S'};
assert(all(ismember(P.readings, ok)), 'zwfs_run: readings must be a subset of %s', strjoin(ok, ' '));
assert(all(ismember(P.stages, {'bench','battery','color','noise','figs'})), 'zwfs_run: unknown stage');
end

function n = index_(P, lam_mm)
% substrate index at lam_mm: Malitson 1965 fused silica, or a number
if ischar(P.mask.index) || isstring(P.mask.index)
    assert(strcmpi(P.mask.index, 'malitson'), 'zwfs_run: mask.index = ''malitson'' or a number');
    um = lam_mm*1e3;
    n = sqrt(1 + 0.6961663*um^2/(um^2-0.0684043^2) + 0.4079426*um^2/(um^2-0.1162414^2) ...
               + 0.8974794*um^2/(um^2-9.896161^2));
else
    n = P.mask.index;
end
end

function g = gauge_opt_(P, lam_mm)
% dmg_zwfs_gauge options at wavelength lam_mm: ONE physical mask (fixed
% etch, fixed dimple diameter), so phase, depth ladder and lam/D all move.
n = index_(P, lam_mm);  n0 = index_(P, P.LAM);
g = struct('LAM', lam_mm, 'F2', P.bench.F2, 'R_BEAM', P.bench.R_TO_AP, ...
    'DIA_LAMD', P.mask.DIA_LAMD * P.LAM/lam_mm, ...
    'PHI_M', 2*pi*(n-1)*P.mask.ETCH_MM/lam_mm, ...
    'PHIS', P.mask.PHIS_REC * (n-1)/(n0-1) * (P.LAM/lam_mm), ...
    'S_CONV', P.mask.S_CONV, 'NITER', P.mask.NITER);
end

function k = class_(rd)
% kernel / transfer class of a reading: 1 = linear map, 2 = exact map, 3 = stepped
switch rd
    case 'L',          k = 1;
    case {'F','I','I+'}, k = 2;
    case 'S',          k = 3;
end
end

function fn = color_deck_(src, nm)
% copy of an emitted deck with ONLY the header Wavelen= line changed
txt = fileread(src);
txt = regexprep(txt, 'Wavelen=\s*\S+', sprintf('Wavelen=  %.9E', nm*1e-6), 'once');
[p, b, e] = fileparts(src);
fn = fullfile(p, sprintf('%s_%gnm%s', b, nm, e));
fid = fopen(fn, 'w');  fwrite(fid, txt);  fclose(fid);
end

function budget_(P, rep, ok, fmt, varargin)
% a sampling-budget line: prints PASS / NOT MET, warns or errors per P.samp.enforce
dmg_say(rep, [fmt '  -> %s\n'], varargin{:}, ifelse_(ok, 'PASS', 'NOT MET'));
if ~ok
    msg = sprintf(fmt, varargin{:});
    if strcmp(P.samp.enforce, 'error'), error('zwfs_run:sampling', '%s', msg);
    else, warning('zwfs_run:sampling', '%s', msg); end
end
end

function y = ifelse_(c, a, b)
if c, y = a; else, y = b; end
end

function t = fmt0_(x)
if isnan(x), t = '-'; else, t = sprintf('%.0f', x); end
end

function t = fmt2_(x)
if isnan(x), t = '-'; else, t = sprintf('%.2f', x); end
end

% =====================================================================
%  stage: bench -- build, gates, sampling budget, frame, registration
% =====================================================================
function S = stage_bench_(P, rep)
t0 = tic;
dmg_say(rep, '\n---- bench ----\n');
if ~isempty(P.param_file)
    pf = P.param_file;
    if ~isfile(pf), pf = fullfile(fileparts(mfilename('fullpath')), P.param_file); end
    if ~isfile(pf), pf = which(P.param_file); end
    assert(~isempty(pf) && isfile(pf), 'zwfs_run: param_file %s not found', P.param_file);
    copyfile(pf, 'macos_param.txt');           % cwd = the run dir; the engine looks here first
    dmg_say(rep, 'engine size table: %s (copied into the run dir as macos_param.txt)\n', pf);
elseif exist('macos_param.txt', 'file')
    delete('macos_param.txt');                 % a stale copy from an earlier run must not win
end
macos.init(P.MODEL);
macos.write_grid_file(P.grid.flat_file, zeros(P.grid.N_G));
bf = fieldnames(P.bench);  bargs = cell(1, 2*numel(bf));
for i = 1:numel(bf), bargs{2*i-1} = bf{i};  bargs{2*i} = P.bench.(bf{i}); end
G = macos.design.twyman_green(bargs{:}, 'ngridpts', P.NGRID, ...
    'to_grid_file', P.grid.flat_file, 'to_grid_n', P.grid.N_G, 'to_grid_dx', P.grid.DX_G);
G.bt.wavelen = P.LAM;
deck = sprintf('%s_test.in', P.tag);
G.bt.emit(deck);
iTO = G.T.iTO;  iMASK = G.T.iMASK;  iDET = G.T.iDET;
macos.load_rx(deck);
wl = macos.get_src_wvl();
assert(abs(wl - P.LAM) < 1e-9*P.LAM, 'deck %s did not take Wavelen (%g vs %g)', deck, wl, P.LAM);
Z1 = G.bt.E(iMASK-1).zelt;  Z2 = G.bt.E(iMASK+1).zelt;
dmg_say(rep, 'deck %s: TO elt %d, FocalMask %d, Detector %d; mask sandwich spheres zElt %.3f / %.3f mm (%s)\n', ...
    deck, iTO, iMASK, iDET, Z1, Z2, ifelse_(abs(Z1-Z2) < 1e-9, 'SYMMETRIC', 'ASYMMETRIC -- Fresnel-defocused pupil'));

% ---- the measurement factory + sensor gates --------------------------
gopt = gauge_opt_(P, P.LAM);
if strcmp(P.bench.mask_prop, 'nf_legacy'), warning('off', 'dmg_zwfs_gauge:roundtrip'); end
ZW = dmg_zwfs_gauge(iTO, iMASK, iDET, gopt);
warning('on', 'dmg_zwfs_gauge:roundtrip');
msk = ZW.msk;  N_WF = ZW.N_WF;
dimple_px = ZW.dia_mm*1e-3 / abs(macos.dx_at(iMASK));
dmg_say(rep, 'mask: phase %.4f rad (|c| %.3f), dimple %.3f lam/D = %.4e mm = %.2f px at the mask plane; msk %d px\n', ...
    gopt.PHI_M, abs(ZW.cc), gopt.DIA_LAMD, ZW.dia_mm, dimple_px, nnz(msk));
dmg_say(rep, 'G1 mask-sandwich round trip (unmasked, entrance -> exit sphere): %.3e   (gate < 1e-12)\n', ZW.gate.roundtrip);
dmg_say(rep, 'G2 reference-wave surrogate T(D Ti(E0)) vs the engine''s Eb on msk: %.3e   (gate < 1e-10)\n', ZW.gate.bsur);
g1 = ZW.gate.roundtrip < 1e-12;  g2 = ZW.gate.bsur < 1e-10;
if ~strcmp(P.bench.mask_prop, 'nf_legacy'), assert(g1, 'G1 FAIL'); end
assert(g2, 'G2 FAIL');
% reference-wave profile: |Eb|/|E0| vs normalized pupil radius.  The dimple
% is a function of lam/D only, so this profile must not depend on the ray
% grid -- compare it across NGRID to test whether a coarsely sampled dimple
% (px across at the mask plane) still represents the mask.
[cy, cx] = find(msk);  c0 = [mean(cx) mean(cy)];  rr_px = hypot(cx-c0(1), cy-c0(2));
rn = rr_px / prctile(rr_px, 99);  ratio = abs(ZW.Eb0(msk))./abs(ZW.E0(msk));
edges = linspace(0, 1, 21);  bprof = nan(1, 20);
for ib = 1:20, sel = rn >= edges(ib) & rn < edges(ib+1);  if any(sel), bprof(ib) = mean(ratio(sel)); end; end
dmg_say(rep, 'reference wave |Eb|/|E0| vs pupil radius (20 bins, 0..1): %s\n', sprintf('%.4f ', bprof));

% ---- frame + sampling budget ----------------------------------------
[mag, dxd_mm] = dmg_frame(iTO, iDET);
xg = ((0:P.grid.N_G-1)-(P.grid.N_G-1)/2)*P.grid.DX_G;
[gxd, gyd] = meshgrid(xg, xg);
dmg_say(rep, 'frame: ray magnification %.4f DM-mm per detector-mm, detector px %.4e mm -> %.4f DM-mm per px\n', mag, dxd_mm, mag*dxd_mm);
dmg_say(rep, 'sampling budget (P.samp, enforce = %s):\n', P.samp.enforce);
budget_(P, rep, dimple_px >= P.samp.min_dimple_px, '  dimple %.2f px across at the mask plane (min %g)', dimple_px, P.samp.min_dimple_px);
ppa = zeros(1, numel(P.dm));
for ic = 1:numel(P.dm)
    ppa(ic) = P.dm(ic).pitch / (mag*dxd_mm);
    budget_(P, rep, ppa(ic) >= P.samp.min_px_per_act, '  DM %dx%d: %.2f detector px per actuator (min %g)', ...
        P.dm(ic).nact, P.dm(ic).nact, ppa(ic), P.samp.min_px_per_act);
end

% ---- G3: DM-conjugate pupil (amplitude invariant under a phase state) --
cfg = P.dm(1);
dmap = @(act) dm_influence_map(P.grid.N_G, P.grid.DX_G, 'nact', cfg.nact, 'pitch', cfg.pitch, 'act', act);
[axg, ayg] = meshgrid(((1:cfg.nact)-(cfg.nact+1)/2)*cfg.pitch);
lit = dmg_lit(msk, dxd_mm, mag, axg, ayg);
rng(P.battery.seed_base);  Ab = zeros(cfg.nact);  Ab(lit) = P.battery.base_rms*randn(nnz(lit),1);
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), dmap(Ab));
E1 = macos.complex_field(iDET);
rr = abs(E1(msk))./abs(ZW.E0(msk));
dmg_say(rep, 'G3 %.0f nm rms working state (DM %dx%d, lit %d): detector |E|/|E_flat| on msk std %.3e   (gate < 1e-12: A = |E0| is exact)\n', ...
    P.battery.base_rms*1e6, cfg.nact, cfg.nact, nnz(lit), std(rr));
g3 = std(rr) < 1e-12;
if ~strcmp(P.bench.mask_prop, 'nf_legacy'), assert(g3, 'G3 FAIL'); end

% ---- registration: two-poke doctrine on P.dm(1), linear reading -------
POKE = P.reg.POKE;
ic0 = cfg.nact/2;  Aa = zeros(cfg.nact);  Aa(ic0,ic0) = 1;  Ma = dmap(POKE*Aa);
hA = ZW.measL(Ma);
R = struct('P', P.reg.PARb, 'tax',0, 'tay',0, 'bx',0, 'by',0, 'dxd_mm',dxd_mm, 'mag',mag, ...
           'msk',msk, 'N_WF',N_WF, 'gxd',gxd, 'gyd',gyd);
[R.bx, R.by, R.tax, R.tay] = dmg_anchor(hA, Ma, msk, N_WF, xg);
dmg_say(rep, 'registration: center-poke anchor at detector px (%.2f, %.2f) <-> DM (%.2f, %.2f) mm\n', R.bx, R.by, R.tax, R.tay);
if strcmp(P.reg.mode, 'search')
    Ab_ = zeros(cfg.nact);  Ab_(cfg.hold(1), cfg.hold(2)) = 1;  Mb = dmap(POKE*Ab_);
    hB = ZW.measL(Mb);
    [PARb, sgn, cb, cn, ccs] = dmg_register(hB, Mb, R);
    dmg_say(rep, '  parity search on the off-center poke (%d,%d): winner [%s] sign %+d, |corr| %.3f, runner-up %.3f (gates >= %.2f, sep >= %.2f); all 8: %s\n', ...
        cfg.hold(1), cfg.hold(2), num2str(PARb), sgn, cb, cn, P.reg.min_corr, P.reg.min_sep, sprintf('%.2f ', ccs));
    assert(cb >= P.reg.min_corr && cb - cn >= P.reg.min_sep, 'registration selection gate FAIL');
    if ~isequal(PARb, P.reg.PARb) || sgn ~= P.reg.sgn
        warning('zwfs_run:registration', 'parity/sign [%s] %+d differ from P.reg record [%s] %+d -- deck-dependent by doctrine; the search result is used', ...
            num2str(PARb), sgn, num2str(P.reg.PARb), P.reg.sgn);
    end
else
    PARb = P.reg.PARb;  sgn = P.reg.sgn;
    dmg_say(rep, '  parity [%s] sign %+d taken from P.reg (mode record)\n', num2str(PARb), sgn);
end
R.P = PARb;
hAd = sgn*dmg_samp(hA, R);  hAd(isnan(hAd)) = 0;
cpm = corrcoef(hAd(:), Ma(:));
dmg_say(rep, '  center poke in the DM frame: raw peak gain %.4f, corr(map, truth) %.4f\n', max(hAd(:))/max(Ma(:)), cpm(1,2));
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), zeros(P.grid.N_G));
dmg_say(rep, 'bench stage %.1f min\n', toc(t0)/60);

S = struct('G',G, 'deck',deck, 'iTO',iTO, 'iMASK',iMASK, 'iDET',iDET, 'ZW',ZW, 'gopt',gopt, ...
    'mag',mag, 'dxd_mm',dxd_mm, 'xg',xg, 'gxd',gxd, 'gyd',gyd, 'PARb',PARb, 'sgn',sgn);
S.summary = struct('deck',deck, 'Z1',Z1, 'Z2',Z2, 'phi_m',gopt.PHI_M, 'dia_lamd',gopt.DIA_LAMD, ...
    'dimple_px',dimple_px, 'nmsk',nnz(msk), 'roundtrip',ZW.gate.roundtrip, 'bsur',ZW.gate.bsur, 'bprof',bprof, ...
    'ampmod_std',std(rr), 'mag',mag, 'dxd_mm',dxd_mm, 'px_per_act',ppa, 'PARb',PARb, 'sgn',sgn, ...
    'kernel_peak',max(hAd(:))/max(Ma(:)), 'kernel_corr',cpm(1,2), 'anchor',[R.bx R.by R.tax R.tay]);
end

% =====================================================================
%  frames and readings
% =====================================================================
function F = frames_(ZW, M, needS)
% capture the frames one DM state needs: Ia (the one masked frame), and
% when needS the stepped set Fr + its rank-2 retrieval X
F = struct('Ia', ZW.frameL(M), 'Fr', [], 'X', []);
if needS
    F.Fr = ZW.framesS(M);  F.X = ZW.reconS(F.Fr);
end
end

function h = readmap_(ZW, rd, F, plus, Xref)
% absolute height map (mm) of one reading from captured frames
switch rd
    case 'L',  h = ZW.reconL(F.Ia);
    case 'F',  h = ZW.reconI(F.Ia, [], [], [], 0);
    case 'I',  h = ZW.reconI(F.Ia);
    case 'I+', h = ZW.reconI(F.Ia, [], plus);
    case 'S',  h = ZW.stepdiff(F.X, Xref);
end
end

function d = diff_(ZW, rd, F1, F0, plus)
% differential height map between two captured states
if strcmp(rd, 'S'), d = ZW.stepdiff(F1.X, F0.X);
else, d = readmap_(ZW, rd, F1, plus, []) - readmap_(ZW, rd, F0, plus, []); end
end

function C = calibrate_(P, S, ZW, cfg, classes, lit)
% per-class registration anchor + measured response kernel + estimator
% (the S2/S3 recipe): class 1 linear map, 2 exact map, 3 stepped map.
% P.battery.calib_surface: 'flat' (the record) or 'base' -- the kernel and
%   the modal transfer are measured DIFFERENTIALLY on the working surface
%   (P.battery.base_rms, seed_base), the class-2 map read with the base's
%   refined sign map (I+).
% P.reg.kernel_site: 'center' (the record), 'hold' (cfg.hold) or [r c] --
%   where the response kernel is measured.  The registration ANCHOR always
%   comes from the centre poke (doctrine: translation from a symmetric
%   site); a second poke at the kernel site supplies the stencil.
N_G = P.grid.N_G;  xg = S.xg;
C.dmap = @(act) dm_influence_map(N_G, P.grid.DX_G, 'nact', cfg.nact, 'pitch', cfg.pitch, 'act', act);
[C.axg, C.ayg] = meshgrid(((1:cfg.nact)-(cfg.nact+1)/2)*cfg.pitch);
if nargin < 6 || isempty(lit), lit = dmg_lit(ZW.msk, S.dxd_mm, S.mag, C.axg, C.ayg); end
C.lit = lit;
POKE = P.reg.POKE;
ic = cfg.nact/2;  Aa = zeros(cfg.nact);  Aa(ic,ic) = 1;  C.Ma = C.dmap(POKE*Aa);
needS = any(classes == 3);
% ---- the calibration surface -----------------------------------------
C.surface = P.battery.calib_surface;
onbase = strcmp(C.surface, 'base');
if onbase
    rng(P.battery.seed_base);  Ab = zeros(cfg.nact);  Ab(lit) = P.battery.base_rms*randn(nnz(lit),1);
    C.Abase = Ab;  C.F0 = frames_(ZW, C.dmap(Ab), true);  C.plus = ZW.priorS(C.F0.Ia, C.F0.Fr);
    C.Fflat = frames_(ZW, zeros(N_G), needS);
else
    C.Abase = zeros(cfg.nact);  C.Fflat = frames_(ZW, zeros(N_G), needS);  C.F0 = C.Fflat;  C.plus = [];
end
% class maps of a state F relative to the calibration surface
if onbase
    C.cmap = @(F, k) cmap_base_(ZW, F, C.F0, C.plus, k);
else
    C.cmap = @(F, k) cmap_flat_(ZW, F, C.Fflat, k);
end
% ---- the kernel site ----------------------------------------------------
ks = P.reg.kernel_site;
if ischar(ks) || isstring(ks)
    switch char(ks)
        case 'center', ks = [ic ic];
        case 'hold',   ks = cfg.hold;
        otherwise,     error('zwfs_run: reg.kernel_site must be ''center'', ''hold'' or [r c]');
    end
end
C.kernel_site = ks;
Ak = zeros(cfg.nact);  Ak(ks(1), ks(2)) = 1;  C.Mk = C.dmap(POKE*Ak);
Fa = frames_(ZW, C.dmap(C.Abase + POKE*Aa), needS);           % centre poke: the anchor
if isequal(ks, [ic ic]), Fk = Fa;  else, Fk = frames_(ZW, C.dmap(C.Abase + POKE*Ak), needS); end
C.R = cell(1,3);  C.stn = cell(1,3);  C.est = cell(1,3);  C.kinfo = nan(3,3);
for k = classes(:).'
    R = struct('P', S.PARb, 'tax',0, 'tay',0, 'bx',0, 'by',0, 'dxd_mm',S.dxd_mm, 'mag',S.mag, ...
               'msk',ZW.msk, 'N_WF',ZW.N_WF, 'gxd',S.gxd, 'gyd',S.gyd);
    hA = C.cmap(Fa, k);
    [R.bx, R.by, R.tax, R.tay] = dmg_anchor(hA, C.Ma, ZW.msk, ZW.N_WF, xg);
    if isequal(ks, [ic ic])
        hK = hA;  tax = R.tax;  tay = R.tay;  Mtruth = C.Ma;
    else
        hK = C.cmap(Fk, k);
        [~, ~, tax, tay] = dmg_anchor(hK, C.Mk, ZW.msk, ZW.N_WF, xg);   % the kernel site's truth peak
        Mtruth = C.Mk;
    end
    if strcmp(P.reg.stencil_site, 'lattice')
        % sample the stencil about the EXACT actuator centre (the lattice point
        % nearest the poke's peak) instead of the map-grid point dmg_anchor
        % returns (up to half a grid pitch, 0.14 mm, off the centre the fit
        % samples at)
        lat = ((1:cfg.nact)-(cfg.nact+1)/2)*cfg.pitch;
        [~, i1] = min(abs(lat - tax));  tax = lat(i1);
        [~, i2] = min(abs(lat - tay));  tay = lat(i2);
    end
    hd = S.sgn*dmg_samp(hK, R);  hd(isnan(hd)) = 0;
    stn = dmg_stencil(hd, xg, tax, tay, cfg.pitch, P.reg.hw) / POKE;
    cpm = corrcoef(hd(:), Mtruth(:));
    C.kinfo(k,:) = [max(hd(:))/max(Mtruth(:)), min(stn(:))/max(stn(:)), cpm(1,2)];
    C.R{k} = R;  C.stn{k} = stn;
    C.est{k} = @(h) dmg_act_fit(S.sgn*dmg_samp(h, R), xg, C.axg, C.ayg, stn, lit, P.battery.act_lam);
end
end

function h = cmap_flat_(ZW, F, Fflat, k)
% absolute class map on the flat (the record's form)
switch k
    case 1, h = ZW.reconL(F.Ia);
    case 2, h = ZW.reconI(F.Ia);
    case 3, h = ZW.stepdiff(F.X, Fflat.X);
end
end

function h = cmap_base_(ZW, F, F0, plus, k)
% class map DIFFERENTIAL to the working surface; class 2 = I+ on the base
switch k
    case 1, h = ZW.reconL(F.Ia) - ZW.reconL(F0.Ia);
    case 2, h = ZW.reconI(F.Ia, [], plus) - ZW.reconI(F0.Ia, [], plus);
    case 3, h = ZW.stepdiff(F.X, F0.X);
end
end

function gk = transfer_(P, ZW, C, cfg, classes, AMPM)
% modal transfer through each class's estimator on the probes cfg.PQ,
% measured on the calibration surface (flat: absolute; base: differential)
nm = size(cfg.PQ, 1);  gk = nan(nm, 3);
[ii, jj] = meshgrid((0.5:cfg.nact)/cfg.nact);
needS = any(classes == 3);
for m = 1:nm
    p = cfg.PQ(m,1);  q = cfg.PQ(m,2);
    Ak = cos(pi*p*ii).*cos(pi*q*jj);
    F = frames_(ZW, C.dmap(C.Abase + AMPM*Ak), needS);
    for k = classes(:).'
        a = C.est{k}(C.cmap(F, k));  gk(m,k) = (AMPM*Ak(C.lit)) \ a(C.lit);
    end
end
end

function [g, e, fl, snr] = score_(a, Ad, lit)
% actuator-space score: gain, rms error (pm), unpoked floor (pm), SNR
g = Ad(lit) \ a(lit);
e = sqrt(mean((a(lit) - Ad(lit)).^2))*1e9;
pk = (Ad ~= 0) & lit;  un = lit & ~pk;
if nnz(pk) < nnz(lit)/4
    fl = std(a(un))*1e9;  snr = mean(a(pk)) / max(std(a(un)), eps);
else
    fl = NaN;  snr = NaN;
end
end

function ROWS = rows_(P, cfg, lit, names)
% the differential rows: {label, base command, deviation command}
NACT = cfg.nact;
rng(P.battery.seed_base);  Ab = zeros(NACT);  Ab(lit) = P.battery.base_rms*randn(nnz(lit),1);
rng(P.battery.seed_dev);   Ar = zeros(NACT);  Ar(lit) = P.battery.dev_rand*randn(nnz(lit),1);
Ah = zeros(NACT);  Ah(cfg.hold(1), cfg.hold(2)) = 1;
st = P.battery.grid_step;  Pg = zeros(NACT);  Pg(st:st:NACT, st:st:NACT) = 1;  Pg = Pg .* lit;
b = P.battery.base_rms*1e6;
ALL = {'flat/hold',   sprintf('flat/hold%g', P.reg.POKE*1e6),          zeros(NACT), P.reg.POKE*Ah; ...
       'flat/rand',   sprintf('flat/rand%g', P.battery.dev_rand*1e6),    zeros(NACT), Ar; ...
       'base/single', sprintf('rand%g/single%g', b, P.battery.dev_single*1e6), Ab, P.battery.dev_single*Ah; ...
       'base/grid',   sprintf('rand%g/grid@%gnm', b, P.battery.grid_amp*1e6), Ab, P.battery.grid_amp*Pg; ...
       'base/rand',   sprintf('rand%g/rand%g', b, P.battery.dev_rand*1e6), Ab, Ar};
keep = ismember(ALL(:,1), names);
ROWS = ALL(keep, 2:4);
end

% =====================================================================
%  stage: battery -- kernels, transfer, rows, ladder, per DM config
% =====================================================================
function B = stage_battery_(P, S, rep)
ZW = S.ZW;  msk = ZW.msk;
RD = P.readings;  KC = cellfun(@class_, RD);  classes = unique(KC);
BETA = P.battery.BETA;
B = struct();
for icfg = 1:numel(P.dm)
    t0 = tic;  cfg = P.dm(icfg);  NACT = cfg.nact;
    dmg_say(rep, '\n---- battery: DM %dx%d, pitch %.1f mm ----\n', NACT, NACT, cfg.pitch);
    C = calibrate_(P, S, ZW, cfg, classes);
    lit = C.lit;
    kk = find(~isnan(C.kinfo(:,1))).';
    dmg_say(rep, 'calibration: surface %s%s, kernel measured at actuator (%d,%d); test actuator (%d,%d)\n', C.surface, ifelse_(strcmp(C.surface,'base'), sprintf(' (%g nm rms, seed %d)', P.battery.base_rms*1e6, P.battery.seed_base), ''), C.kernel_site(1), C.kernel_site(2), cfg.hold(1), cfg.hold(2));
    dmg_say(rep, 'lit actuators %d; kernel per class [raw peak gain, ring min/peak, corr]:', nnz(lit));
    cn = {'L', 'I', 'S'};
    for k = kk, dmg_say(rep, '  %s [%.3f %.3f %.3f]', cn{k}, C.kinfo(k,:)); end
    dmg_say(rep, '\n');
    if any(C.kinfo(kk,3) < 0.9)
        dmg_say(rep, 'NOTE: a kernel/truth correlation is below 0.9 (registration sanity line; on a working surface the differential kernel carries the surface''s crosstalk) -- read the rows with that in mind\n');
        if strcmp(C.surface, 'flat'), error('zwfs_run:registration', 'kernel/truth correlation < 0.9 on the flat: registration is suspect'); end
    end
    % ---- modal transfer -------------------------------------------------
    gk = transfer_(P, ZW, C, cfg, classes, P.battery.AMPM);
    is1d = cfg.PQ(:,2) == 0;  pk1 = cfg.PQ(is1d,1);
    dmg_say(rep, 'modal transfer (actuator-space estimator, RAW):\n%9s %7s |', 'mode', 'cyc/ap');
    for k = kk, dmg_say(rep, ' %7s', cn{k}); end
    dmg_say(rep, '\n');
    for m = 1:size(cfg.PQ,1)
        dmg_say(rep, '  (%2d,%2d) %7.1f |', cfg.PQ(m,1), cfg.PQ(m,2), hypot(cfg.PQ(m,1), cfg.PQ(m,2))/2);
        for k = kk, dmg_say(rep, ' %7.4f', gk(m,k)); end
        dmg_say(rep, '%s\n', ifelse_(is1d(m), '', '   (separability row)'));
    end
    dmg_say(rep, 'min |g| over the (p,0) rows:');
    for k = kk, dmg_say(rep, '  %s %.3f', cn{k}, min(abs(gk(is1d,k)))); end
    dmg_say(rep, '\n');
    corrk = @(a, k) dmg_modal_corr(a, 'separable', pk1, gk(is1d,k), BETA, NACT);
    % ---- the rows ------------------------------------------------------
    ROWS = rows_(P, cfg, lit, P.battery.rows);
    needS = any(KC == 3) || any(strcmp(RD, 'I+'));
    dmg_say(rep, 'rows (differential, actuator space).  g = gain, e = rms err over lit (pm), flr = rms of unpoked lit (pm), SNR = mean(poked)/flr\n');
    dmg_say(rep, '%-17s %-3s | %7s %8s %8s %7s | %7s %8s %8s %7s\n', 'row', 'rd', 'g_raw', 'e_raw', 'flr_raw', 'SNRraw', 'g_cor', 'e_cor', 'flr_cor', 'SNRcor');
    res = struct('row',{},'rd',{},'raw',{},'cor',{},'fold',{});
    plusb = [];  pinf = struct('frac', NaN);
    for r = 1:size(ROWS,1)
        base = ROWS{r,2};  dev = ROWS{r,3};
        if r == 1 || ~isequal(base, ROWS{r-1,2})
            F0 = frames_(ZW, C.dmap(base), needS);
            if any(strcmp(RD, 'I+')), [plusb, pinf] = ZW.priorS(F0.Ia, F0.Fr); end
        end
        F1 = frames_(ZW, C.dmap(base + dev), needS);
        if any(strcmp(RD, 'I+')) && any(base(:) ~= 0)
            % fold-crossing diagnostic: pixels whose side of the quarter-wave fold
            % differs between the base and base+change (the base's sign map is
            % wrong there for the one-frame I+ reading of the changed state)
            plus1 = ZW.priorS(F1.Ia, F1.Fr);
            dmg_say(rep, '  [%s] pixels that cross the fold under the change: %d of %d beyond-fold (%.2f%% of msk)\n', ...
                ROWS{r,1}, nnz(plus1 ~= plusb), nnz(plusb), 100*nnz(plus1 ~= plusb)/nnz(msk));
        end
        for k = 1:numel(RD)
            araw = C.est{KC(k)}(diff_(ZW, RD{k}, F1, F0, plusb));  acor = corrk(araw, KC(k));
            [gr, er, fr, sr] = score_(araw, dev, lit);  [gc, ec, fc, sc] = score_(acor, dev, lit);
            dmg_say(rep, '%-17s %-3s | %7.4f %8.0f %8s %7s | %7.4f %8.0f %8s %7s\n', ifelse_(k==1, ROWS{r,1}, ''), RD{k}, ...
                gr, er, fmt0_(fr), fmt2_(sr), gc, ec, fmt0_(fc), fmt2_(sc));
            res(end+1) = struct('row',ROWS{r,1}, 'rd',RD{k}, 'raw',[gr er fr sr], 'cor',[gc ec fc sc], ...
                                'fold',ifelse_(isempty(plusb), NaN, mean(plusb(msk)))); %#ok<AGROW>
        end
    end
    if any(strcmp(RD, 'I+'))
        dmg_say(rep, '(beyond-fold fraction of msk on the working-state base, prior passes: %s)\n', sprintf('%.4f ', pinf.frac));
    end
    % ---- break scale ----------------------------------------------------
    switch P.battery.ladder_sites
        case 'hold',  Asng = zeros(NACT);  Asng(cfg.hold(1), cfg.hold(2)) = P.battery.dev_single;  site = sprintf('the hold-out actuator (%d,%d)', cfg.hold(1), cfg.hold(2));
        case 'grid',  st = P.battery.grid_step;  Pg = zeros(NACT);  Pg(st:st:NACT, st:st:NACT) = 1;  Asng = P.battery.dev_single*(Pg .* lit);  site = sprintf('%d grid sites (every %d actuators)', nnz(Asng), st);
        otherwise,    error('zwfs_run: battery.ladder_sites must be ''hold'' or ''grid''');
    end
    dmg_say(rep, 'break scale (%g nm differential on %s, on a growing base; corrected estimates; g = gain over the poked site(s), floor pm over the unpoked, SNR; fold0/fold = beyond-fold fraction from the plain / refined stepped prior; past the fold quote GAIN, not SNR):\n', P.battery.dev_single*1e6, site);
    dmg_say(rep, '%8s %6s %6s |', 'base', 'fold0', 'fold');
    for k = 1:numel(RD), dmg_say(rep, ' %-22s|', sprintf('%s: g flr SNR', RD{k})); end
    dmg_say(rep, '\n');
    rng(P.battery.seed_base);  Bfield = zeros(NACT);  Bfield(lit) = randn(nnz(lit),1);  Bfield = Bfield / std(Bfield(lit));
    lad = struct('amp',{},'fold',{},'fold0',{},'g',{},'flr',{},'snr',{});
    for amp = P.battery.ladder
        Ab = amp*Bfield;
        F0 = frames_(ZW, C.dmap(Ab), needS);
        if any(strcmp(RD, 'I+')), [plusb, pinf] = ZW.priorS(F0.Ia, F0.Fr); else, pinf = struct('frac', [NaN NaN]); end
        F1 = frames_(ZW, C.dmap(Ab + Asng), needS);
        g = nan(1,numel(RD));  fl = g;  sn = g;
        for k = 1:numel(RD)
            a = corrk(C.est{KC(k)}(diff_(ZW, RD{k}, F1, F0, plusb)), KC(k));
            [g(k), ~, fl(k), sn(k)] = score_(a, Asng, lit);
        end
        dmg_say(rep, '%5.0f nm %6.4f %6.4f |', amp*1e6, pinf.frac(1), pinf.frac(end));
        for k = 1:numel(RD), dmg_say(rep, ' %7.4f %6.0f %7.1f|', g(k), fl(k), sn(k)); end
        dmg_say(rep, '\n');
        lad(end+1) = struct('amp',amp, 'fold',pinf.frac(end), 'fold0',pinf.frac(1), 'g',g, 'flr',fl, 'snr',sn); %#ok<AGROW>
    end
    % ---- verdict lines ----------------------------------------------------
    for k = find(strcmp(RD, 'I') | strcmp(RD, 'I+'))
        i = find(strcmp({res.row}, ROWS{1,1}) & strcmp({res.rd}, RD{k}), 1);
        if ~isempty(i) && strncmp(ROWS{1,1}, 'flat/hold', 9)
            dmg_say(rep, 'SPEC: flat hold-out RAW gain, reading %-2s (one frame, no Wiener): %.4f -> %s (within 3%% of 1)\n', ...
                RD{k}, res(i).raw(1), ifelse_(abs(res(i).raw(1)-1) <= 0.03, 'PASS', 'not met'));
        end
    end
    ig = find(strncmp({res.row}, 'rand', 4) & contains({res.row}, 'grid'));
    for i = ig
        dmg_say(rep, 'SPEC: grid-on-base SNR, reading %-2s: raw %6.2f / corrected %6.2f -> %s\n', res(i).rd, res(i).raw(4), res(i).cor(4), ...
            ifelse_(max(res(i).raw(4), res(i).cor(4)) >= 5, 'DETECTED (>= 5)', 'below 5'));
    end
    dmg_say(rep, 'DM %dx%d battery %.1f min\n', NACT, NACT, toc(t0)/60);
    B.(sprintf('n%d', NACT)) = struct('cfg',cfg, 'lit',lit, 'kinfo',C.kinfo, 'PQ',cfg.PQ, 'gk',gk, ...
        'readings',{RD}, 'rows',{ROWS(:,1)}, 'res',res, 'ladder',lad);
end
macos.set_elt_grid(S.iTO, macos.get_elt_grid_spacing(S.iTO), zeros(P.grid.N_G));
end

% =====================================================================
%  stage: color -- one physical mask at K wavelengths, combined
% =====================================================================
function CO = stage_color_(P, S, rep)
t0 = tic;
cfg = P.dm(1);  NACT = cfg.nact;
RD = P.color.readings;  assert(all(ismember(RD, P.readings)), 'color.readings must be a subset of P.readings');
KC = cellfun(@class_, RD);  classes = unique(KC);
LAMS = P.color.lams_nm;  K = numel(LAMS);  BETA = P.color.BETA;
assert(abs(LAMS(1)*1e-6 - P.LAM) < 1e-12, 'color.lams_nm(1) must be P.LAM (the record color carries lit + bases)');
dmg_say(rep, '\n---- color: %s nm, DM %dx%d, readings %s ----\n', num2str(LAMS), NACT, NACT, strjoin(RD, ' '));
dmg_say(rep, 'ONE physical mask (%.1f nm etch, %.2f lam/D at %.1f nm); per-color calibration (references, anchor, kernel, transfer); lit + bases from the first color; beta %.2f; combiner DC %s\n', ...
    P.mask.ETCH_MM*1e6, P.mask.DIA_LAMD, P.LAM*1e6, BETA, P.color.dc);
decks = cell(1,K);  for k = 1:K, decks{k} = color_deck_(S.deck, LAMS(k)); end
is1d = cfg.PQ(:,2) == 0;  pk1 = cfg.PQ(is1d,1);
gk = nan(size(cfg.PQ,1), 3, K);
col = struct('nm',{},'phi',{},'absc',{},'dia_lamd',{},'dimple_px',{},'nmsk',{},'roundtrip',{},'bsur',{},'kinfo',{},'tmin',{});
RAW = cell(0, K);  lit = [];  ROWS = {};
for k = 1:K
    tk = tic;
    macos.load_rx(decks{k});
    LAM = LAMS(k)*1e-6;  wl = macos.get_src_wvl();
    assert(abs(wl - LAM) < 1e-9*LAM, 'deck %s did not take Wavelen (%g)', decks{k}, wl);
    gopt = gauge_opt_(P, LAM);
    ZW = dmg_zwfs_gauge(S.iTO, S.iMASK, S.iDET, gopt);
    assert(ZW.gate.bsur < 1e-10, 'color %g nm: surrogate gate FAIL (%.2e)', LAMS(k), ZW.gate.bsur);
    dimple_px = ZW.dia_mm*1e-3 / abs(macos.dx_at(S.iMASK));
    C = calibrate_(P, S, ZW, cfg, classes, lit);
    if k == 1
        lit = C.lit;
        ROWS = rows_(P, cfg, lit, P.color.rows);
        RAW = cell(size(ROWS,1), K);
        dmg_say(rep, 'lit actuators %d; hold-out (%d,%d)\n', nnz(lit), cfg.hold(1), cfg.hold(2));
        dmg_say(rep, '%6s | %7s %6s %8s %9s %7s %9s %9s | %s\n', 'nm', 'phi_m', '|c|', 'dia_l/D', 'dimplePx', 'msk_px', 'roundtrip', 'bsur', 'kernel peak per class L I S');
    end
    kk = find(~isnan(C.kinfo(:,1))).';
    dmg_say(rep, '%6g | %7.4f %6.3f %8.3f %9.2f %7d %9.1e %9.1e |', LAMS(k), gopt.PHI_M, abs(ZW.cc), gopt.DIA_LAMD, dimple_px, nnz(ZW.msk), ZW.gate.roundtrip, ZW.gate.bsur);
    for c = 1:3, if any(kk == c), dmg_say(rep, ' %.3f', C.kinfo(c,1)); else, dmg_say(rep, '     -'); end; end
    dmg_say(rep, '\n');
    gk(:,:,k) = transfer_(P, ZW, C, cfg, classes, P.battery.AMPM);
    needS = any(KC == 3) || any(strcmp(RD, 'I+'));
    plusb = [];
    for r = 1:size(ROWS,1)
        base = ROWS{r,2};  dev = ROWS{r,3};
        if r == 1 || ~isequal(base, ROWS{r-1,2})
            F0 = frames_(ZW, C.dmap(base), needS);
            if any(strcmp(RD, 'I+')), plusb = ZW.priorS(F0.Ia, F0.Fr); end
        end
        F1 = frames_(ZW, C.dmap(base + dev), needS);
        A = struct();
        for j = 1:numel(RD)
            A.(fld_(RD{j})) = C.est{KC(j)}(diff_(ZW, RD{j}, F1, F0, plusb));
        end
        RAW{r,k} = A;
    end
    col(end+1) = struct('nm',LAMS(k), 'phi',gopt.PHI_M, 'absc',abs(ZW.cc), 'dia_lamd',gopt.DIA_LAMD, ...
        'dimple_px',dimple_px, 'nmsk',nnz(ZW.msk), 'roundtrip',ZW.gate.roundtrip, 'bsur',ZW.gate.bsur, ...
        'kinfo',C.kinfo, 'tmin',toc(tk)/60); %#ok<AGROW>
    fprintf('color %g nm done in %.1f min\n', LAMS(k), toc(tk)/60);
end
% ---- transfer tables per class + the combination's transfer ------------
cn = {'L', 'I', 'S'};
Gc = nan(nnz(is1d), 3);  G1 = nan(nnz(is1d), 3, K);
for c = classes(:).'
    g1 = squeeze(gk(is1d, c, :));  if K == 1, g1 = g1(:); end
    G1(:,c,:) = reshape(g1.^2 ./ (g1.^2 + BETA^2), [], 1, K);
    Gc(:,c) = sum(g1.^2, 2) ./ (sum(g1.^2, 2) + BETA^2);
    dmg_say(rep, '\nmodal transfer (p,0) rows, class %s, per color; Gcomb = the K-color combination''s transfer on that row\n%9s %7s |', cn{c}, 'mode', 'cyc/ap');
    for k = 1:K, dmg_say(rep, ' %7g', LAMS(k)); end
    dmg_say(rep, ' | %7s %7s\n', sprintf('G%g', LAMS(1)), 'Gcomb');
    i1 = 0;
    for m = find(is1d).'
        i1 = i1 + 1;
        dmg_say(rep, '  (%2d, 0) %7.1f |', cfg.PQ(m,1), cfg.PQ(m,1)/2);
        for k = 1:K, dmg_say(rep, ' %7.4f', gk(m,c,k)); end
        dmg_say(rep, ' | %7.4f %7.4f\n', G1(i1,c,1), Gc(i1,c));
    end
    dmg_say(rep, '  min |g|:');  for k = 1:K, dmg_say(rep, ' %g:%.3f', LAMS(k), min(abs(gk(is1d,c,k)))); end
    dmg_say(rep, '\n  min Geff (after Wiener, beta %.2f):', BETA);
    for k = 1:K, dmg_say(rep, ' %g:%.3f', LAMS(k), min(G1(:,c,k))); end
    dmg_say(rep, ' | K-comb %.3f\n', min(Gc(:,c)));
end
% ---- rows: per color, mean, K-comb, best pair; per reading ---------------
dmg_say(rep, '\nrows (differential, actuator space).  g = gain, e = rms error over lit (pm), flr = rms of unpoked lit (pm), SNR = mean(poked)/flr\n');
dmg_say(rep, 'columns: each single color (joint Wiener, that color''s transfer) | mean of the singles | K-color comb | best pair\n');
pairs = nchoosek(1:K, 2);
res = struct('row',{},'rd',{},'g',{},'e',{},'fl',{},'snr',{},'best_pair',{});
for j = 1:numel(RD)
    c = KC(j);
    gcell = @(ks) arrayfun(@(k) gk(is1d, c, k), ks, 'UniformOutput', false);
    dmg_say(rep, '\n-- reading %s --\n', RD{j});
    for r = 1:size(ROWS,1)
        Ad = ROWS{r,3};
        A = cellfun(@(q) q.(fld_(RD{j})), RAW(r,:), 'UniformOutput', false);
        gs = zeros(1,K);  es = gs;  fls = gs;  sns = gs;  singles = cell(1,K);
        for k = 1:K
            singles{k} = dmg_color_comb(A(k), 'separable', pk1, gcell(k), BETA, NACT, P.color.dc);
            [gs(k), es(k), fls(k), sns(k)] = score_(singles{k}, Ad, lit);
        end
        amean = singles{1};  for k = 2:K, amean = amean + singles{k}; end
        amean = amean / K;
        [gm, em, flm, snm] = score_(amean, Ad, lit);
        acomb = dmg_color_comb(A, 'separable', pk1, gcell(1:K), BETA, NACT, P.color.dc);
        [gc, ec, flc, snc] = score_(acomb, Ad, lit);
        bp = [NaN NaN];  bscore = -Inf;  bg = NaN;  be = NaN;  bfl = NaN;  bsn = NaN;
        for ip = 1:size(pairs,1)
            ks = pairs(ip,:);
            ap = dmg_color_comb(A(ks), 'separable', pk1, gcell(ks), BETA, NACT, P.color.dc);
            [gp, ep, flp, snp] = score_(ap, Ad, lit);
            sc = ifelse_(isnan(snp), -ep, abs(snp));
            if sc > bscore, bscore = sc;  bp = LAMS(ks);  bg = gp;  be = ep;  bfl = flp;  bsn = snp; end
        end
        dmg_say(rep, '%-18s g   :', ROWS{r,1});  for k = 1:K, dmg_say(rep, ' %8.4f', gs(k)); end
        dmg_say(rep, ' | %8.4f %8.4f | %8.4f (%g+%g)\n', gm, gc, bg, bp(1), bp(2));
        dmg_say(rep, '%-18s e pm:', '');  for k = 1:K, dmg_say(rep, ' %8.0f', es(k)); end
        dmg_say(rep, ' | %8.0f %8.0f | %8.0f\n', em, ec, be);
        if ~isnan(sns(1))
            dmg_say(rep, '%-18s flr :', '');  for k = 1:K, dmg_say(rep, ' %8.0f', fls(k)); end
            dmg_say(rep, ' | %8.0f %8.0f | %8.0f\n', flm, flc, bfl);
            dmg_say(rep, '%-18s SNR :', '');  for k = 1:K, dmg_say(rep, ' %8.2f', sns(k)); end
            dmg_say(rep, ' | %8.2f %8.2f | %8.2f\n', snm, snc, bsn);
        end
        res(end+1) = struct('row',ROWS{r,1}, 'rd',RD{j}, 'g',[gs gm gc bg], 'e',[es em ec be], ...
            'fl',[fls flm flc bfl], 'snr',[sns snm snc bsn], 'best_pair',bp); %#ok<AGROW>
    end
end
dmg_say(rep, 'color stage %.1f min\n', toc(t0)/60);
macos.load_rx(S.deck);                                  % back to the record color
CO = struct('lams_nm',LAMS, 'readings',{RD}, 'PQ',cfg.PQ, 'gk',gk, 'Gc',Gc, 'G1',G1, 'col',col, ...
    'rows',{ROWS(:,1)}, 'RAW',{RAW}, 'lit',lit, 'res',res, 'beta',BETA);
end

function f = fld_(rd)
f = strrep(rd, '+', 'p');
end

% =====================================================================
%  stage: noise -- photon-shot pricing of every reading
% =====================================================================
function NO = stage_noise_(P, S, out, rep)
t0 = tic;
ZW = S.ZW;  cfg = P.dm(1);  NACT = cfg.nact;
RD = P.noise.readings;  assert(all(ismember(RD, P.readings)), 'noise.readings must be a subset of P.readings');
KC = cellfun(@class_, RD);  classes = unique(KC);
dmg_say(rep, '\n---- noise: DM %dx%d, readings %s ----\n', NACT, NACT, strjoin(RD, ' '));
dmg_say(rep, 'scenario: single act (%d,%d) %g nm differential on the %g nm rms working state; axis = photons per DM STATE (a reading''s frames share it: L/F/I/I+ 1 frame, S 4)\n', ...
    cfg.hold(1), cfg.hold(2), P.battery.dev_single*1e6, P.battery.base_rms*1e6);
dmg_say(rep, 'I+ prior frames (the base''s 4 stepped frames, taken once): %s\n', strjoin(P.noise.prior, ' / '));
fn = sprintf('n%d', NACT);
if isfield(out, 'battery') && isfield(out.battery, fn) && all(~isnan(out.battery.(fn).gk(1, classes)))
    C = calibrate_(P, S, ZW, cfg, classes);  gk = out.battery.(fn).gk;
    dmg_say(rep, 'calibration: kernels re-measured, modal transfer from this run''s battery\n');
else
    C = calibrate_(P, S, ZW, cfg, classes);  gk = transfer_(P, ZW, C, cfg, classes, P.battery.AMPM);
    dmg_say(rep, 'calibration: kernels + modal transfer measured here\n');
end
lit = C.lit;  is1d = cfg.PQ(:,2) == 0;  pk1 = cfg.PQ(is1d,1);
corrk = @(a, k) dmg_modal_corr(a, 'separable', pk1, gk(is1d,k), P.battery.BETA, NACT);
rng(P.battery.seed_base);  Ab = zeros(NACT);  Ab(lit) = P.battery.base_rms*randn(nnz(lit),1);
Asng = zeros(NACT);  Asng(cfg.hold(1), cfg.hold(2)) = P.battery.dev_single;
F0 = frames_(ZW, C.dmap(Ab), true);  F1 = frames_(ZW, C.dmap(Ab + Asng), true);
dmg_say(rep, 'frames captured (%.1f min); the Monte-Carlo is trace-free\n', toc(t0)/60);
noisy = @(I, nph) I .* (1 + randn(size(I)) ./ sqrt(max(I / sum(I(:)) * nph, 1)));
un = lit;  un(cfg.hold(1), cfg.hold(2)) = false;
% columns: one per (reading [, prior treatment])
cols = {};
for j = 1:numel(RD)
    if strcmp(RD{j}, 'I+')
        for pr = P.noise.prior, cols(end+1,:) = {RD{j}, pr{1}}; end %#ok<AGROW>
    else
        cols(end+1,:) = {RD{j}, ''}; %#ok<AGROW>
    end
end
nc = size(cols,1);
lab = cellfun(@(r, p) ifelse_(isempty(p), r, sprintf('%s(%s)', r, p)), cols(:,1), cols(:,2), 'UniformOutput', false);
dmg_say(rep, '%10s |', 'N/state');
for c = 1:nc, dmg_say(rep, ' %-24s|', sprintf('%s: sig flr g', lab{c})); end
dmg_say(rep, '\n');
NS = P.noise.nstates;  NR = P.noise.nreal;
sig = nan(numel(NS), nc);  flr = sig;  gm = sig;
for in = 1:numel(NS)
    n = NS(in);
    pk = zeros(NR, nc);  fl = zeros(NR, nc);
    for r = 1:NR
        rng(P.noise.seed + r + in*100);
        Ia0 = noisy(F0.Ia, n);  Ia1 = noisy(F1.Ia, n);
        Fr0q = F0.Fr;  Fr1q = F1.Fr;  Fr0f = F0.Fr;          % stepped at N/4 per frame; prior at N per frame
        for k = 1:4
            Fr0q(:,:,k) = noisy(F0.Fr(:,:,k), n/4);  Fr1q(:,:,k) = noisy(F1.Fr(:,:,k), n/4);
            Fr0f(:,:,k) = noisy(F0.Fr(:,:,k), n);
        end
        for c = 1:nc
            rd = cols{c,1};  kc = class_(rd);
            switch rd
                case 'L',  d = ZW.reconL(Ia1) - ZW.reconL(Ia0);
                case 'F',  d = ZW.reconI(Ia1, [], [], [], 0) - ZW.reconI(Ia0, [], [], [], 0);
                case 'I',  d = ZW.reconI(Ia1) - ZW.reconI(Ia0);
                case 'I+'
                    switch cols{c,2}
                        case 'split',     plus = ZW.priorS(Ia0, Fr0q);
                        case 'full',      plus = ZW.priorS(Ia0, Fr0f);
                        case 'noiseless', plus = ZW.priorS(F0.Ia, F0.Fr);
                    end
                    d = ZW.reconI(Ia1, [], plus) - ZW.reconI(Ia0, [], plus);
                case 'S',  d = ZW.stepdiff(ZW.reconS(Fr1q), ZW.reconS(Fr0q));
            end
            a = corrk(C.est{kc}(d), kc);
            pk(r,c) = a(cfg.hold(1), cfg.hold(2));  fl(r,c) = std(a(un));
        end
    end
    sig(in,:) = std(pk, 0, 1)*1e9;  flr(in,:) = mean(fl, 1)*1e9;  gm(in,:) = mean(pk, 1)/P.battery.dev_single;
    dmg_say(rep, '%10.1e |', n);
    for c = 1:nc, dmg_say(rep, ' %8.1f %7.0f %7.3f|', sig(in,c), flr(in,c), gm(in,c)); end
    dmg_say(rep, '\n');
end
% 1/sqrt(N) pricing from the shot-noise-dominated points
n1pm = nan(1,nc);
for c = 1:nc
    use = sig(:,c) > 0.5;
    if nnz(use) >= 2
        cc = exp(mean(log(sig(use,c)) + 0.5*log(NS(use).')));
        n1pm(c) = cc^2;
        dmg_say(rep, '%-14s sigma ~ %.3g/sqrt(N) pm  ->  N(1 pm) ~ %.2e photons/state\n', lab{c}, cc, cc^2);
    end
end
dmg_say(rep, 'noise stage %.1f min\n', toc(t0)/60);
NO = struct('cols',{lab}, 'nstates',NS, 'nreal',NR, 'sig_pm',sig, 'flr_pm',flr, 'g',gm, 'n_1pm',n1pm);
end
