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
%   all errors in pm.  Optional stages add the multi-color combination,
%   the photon-noise pricing and the closed-loop HOLD metric (stage
%   'loop': the DM held at the working surface by a servo closed through
%   one reading, scored as the steady-state hold error vs photons per
%   cycle -- dm_gauge_lib/dmg_loop, shared with the interferometer).
%   Every number goes to the report; every gate prints its value and its
%   threshold.
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
dmg_say(rep, 'readings: L frozen-linear (1 frame) | F exact frozen-b (1) | I exact iterated-b (1) | I+ = I with the base''s refined stepped branch prior (1; base 4 once) | S phase-stepped (4) | V vector pair: +phi and -phi dimple images at once, exact, no fold (2 simultaneous frames) | P point-diffraction, stepped pinhole at the FocalMask, exact four-step (K frames) | PF point-diffraction, fiber reference + photonic phase shifter, exact (K)\n');

S = stage_bench_(P, rep);
out.bench = S.summary;
if want('battery'), out.battery = stage_battery_(P, S, rep); end
if want('color'),   out.color   = stage_color_(P, S, rep);   end
if want('noise'),   out.noise   = stage_noise_(P, S, out, rep); end
if want('loop'),    out.loop    = stage_loop_(P, S, rep);  end
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
% a parameter struct may sit anywhere in the argument list (the batch
% wrapper puts 'tag', TAG first: zwfs_batch.sh TAG "pdi_params, ...")
is = find(cellfun(@isstruct, varargin), 1);
if ~isempty(is)
    P = varargin{is};  varargin(is) = [];
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
ok = {'L','F','I','I+','S','V','P','PF'};
assert(all(ismember(P.readings, ok)), 'zwfs_run: readings must be a subset of %s', strjoin(ok, ' '));
assert(all(ismember(P.stages, {'bench','battery','color','noise','loop','figs'})), 'zwfs_run: unknown stage');
% knobs added with the gauge-deck slice (2026-09-13).  They default HERE so
% an unedited zwfs_params sheet still runs every stage; pdi_params.m carries
% their documentation and the PDI record's values.
P = setdef_(P, 'loop', 'start_rms',   []);      % descent: the DM's initial figure, mm rms
P = setdef_(P, 'loop', 'recal_every', 0);       % cycles between on-surface re-calibrations (0 = never)
P = setdef_(P, 'loop', 'recal_list',  []);      % descent: the recal_every values to compare ([] = [recal_every])
P = setdef_(P, 'loop', 'intra',       0);       % fraction of the cycle's drift developing WITHIN a scan
P = setdef_(P, 'loop', 'reach', [10e-6 3e-9]);  % descent: the levels whose cycle count is reported
P = setdef_(P, 'battery', 'unwrap', false);     % unwrap every WRAPPED differential (S V P PF) before
                                                % the estimator (dm_gauge_lib/dmg_unwrap); default OFF
                                                % so every record taken before 2026-09-13 reproduces
P = setdef_(P, 'loop', 'unwrap', 'auto');       % the loop stage's own setting: 'auto' = battery.unwrap,
                                                % or ON whenever loop.start_rms is set (a descent is the
                                                % case the unwrapper exists for); true / false force it
P = setdef_(P, 'pdi',  'bench',  'zwfs');       % 'zwfs' = the ZWFS test arm; 'psri' = the two P/SRI decks
P = setdef_(P, 'pdi',  'ref_frozen', false);    % pdi.bench 'psri': hold the traced reference at
                                               % the flat state (the control run)
P = setdef_(P, 'pdi',  'ref_walk', 0);          % PF: reference-arm phase walk, rad per cycle rms
P = setdef_(P, 'pdi',  'ref_seed', 0);          % its stream (0 = dmg_loop's default)
P = setdef_(P, 'pdi',  'psri', struct());       % macos.design.psri_bench overrides (pdi.bench 'psri')
end

function P = setdef_(P, grp, fld, val)
if ~isfield(P, grp) || ~isfield(P.(grp), fld), P.(grp).(fld) = val; end
end

function tf = wrapped_(rd)
% the readings whose DIFFERENTIAL is a wrapped phase difference.  L is a
% linear map and has no wrap; F / I / I+ solve an ABSOLUTE phase and are
% differenced afterwards, which the unwrapper cannot mend, so they are
% left alone too (Dave / CCL 2026-09-13).
tf = any(strcmp(rd, {'S', 'V', 'P', 'PF'}));
end

function uw = unwrap_fn_(P, ZW)
%UNWRAP_FN_  [] when off, else a handle taking a WRAPPED height map (mm)
%   and returning the unwrapped one.  The readings return height, not
%   phase, so the factor is undone and redone around dmg_unwrap:
%   h = S_CONV * phi * lambda / (4 pi).
uw = [];
if ~P.battery.unwrap, return; end
hpr = P.mask.S_CONV * P.LAM / (4*pi);            % height per radian
msk = ZW.msk;
uw = @(d) hpr * dmg_unwrap(d/hpr, msk);
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
    'S_CONV', P.mask.S_CONV, 'NITER', P.mask.NITER, ...
    'V_RET_ERR', P.mask.v_ret_err, 'V_LEAK_PHASE', P.mask.v_leak_phase, 'V_CAL', P.mask.v_cal, ...
    'V_ARM', P.mask.v_arm, 'V_LASER_DEG', P.mask.v_laser_deg, ...
    'V_ARM_DPHASE', P.mask.v_arm_dphase, 'V_ARM_DAMP', P.mask.v_arm_damp);
if isstruct(P.mask.v_analyzer), g.V_ANALYZER = P.mask.v_analyzer; end   % V4 (resolved from 'engine' in the bench stage)
g.V_CLEAR = P.mask.v_clear;                                              % V5
end

function bargs = bench_args_(P)
% P.bench as twyman_green name/value pairs, minus the runner-side fields the
% builder does not take: the OAP coating (applied after each load, coat_oap_)
skip = {'coat_oap', 'coat_bareAl', 'coat_protectedAl'};
bf = setdiff(fieldnames(P.bench), skip, 'stable');  bargs = cell(1, 2*numel(bf));
for i = 1:numel(bf), bargs{2*i-1} = bf{i};  bargs{2*i} = P.bench.(bf{i}); end
end

function coat_oap_(P, bt, rep)
% the OAP rig's mirror coating (CCMac's tg96_run item B, same stacks): a
% Model-A thin-film stack on the elements named L1 and L2 (the OAPs), applied
% AFTER a deck load (a load clears it).  Active only under polarization, i.e.
% for the vector reading's arm maps (mask.v_arm 'engine'); the scalar traces
% never see a coating.  bench.coat_oap 'none' | 'bareAl' | 'protectedAl'.
if ~isfield(P.bench, 'coat_oap') || any(strcmp(P.bench.coat_oap, {'none', ''})), return; end
assert(isfield(P.bench, 'optics') && strcmp(P.bench.optics, 'oap'), 'zwfs_run: bench.coat_oap needs bench.optics ''oap''');
switch P.bench.coat_oap
    case 'bareAl',      cs = P.bench.coat_bareAl;
    case 'protectedAl', cs = P.bench.coat_protectedAl;
    otherwise, error('zwfs_run: bench.coat_oap must be none | bareAl | protectedAl');
end
nm = {bt.E.name};  iL = [find(strcmp(nm, 'L1'), 1), find(strcmp(nm, 'L2'), 1)];
assert(numel(iL) == 2, 'zwfs_run: elements L1 and L2 not found for the OAP coating');
for j = 1:2
    macos.coating(iL(j), 'index', cs.index, 'extinc', cs.extinc, 'thickness', cs.thickness);
end
if ~isempty(rep)
    dmg_say(rep, 'OAP coating: %s on L1 (elt %d) and L2 (elt %d), %d layer(s); active under polarization (the vector reading''s arm maps)\n', ...
        P.bench.coat_oap, iL(1), iL(2), numel(cs.index));
end
end

function po = pdi_opt_(P, lam_mm)
% dmg_pdi_gauge options at wavelength lam_mm: ONE physical pinhole (fixed
% diameter, so lam/D moves with color); the phase steps are the photonic /
% stepped-substrate values at every color (a substrate step would scale as
% the dimple ladder does -- not modeled: the steps are taken as set)
po = struct('LAM', lam_mm, 'F2', P.bench.F2, 'R_BEAM', P.bench.R_TO_AP, ...
    'DIA_LAMD', P.pdi.DIA_LAMD * P.LAM/lam_mm, 'THETAS', P.pdi.thetas, ...
    'T_SURR', P.pdi.t_surr, 'B2', P.pdi.b2, 'NITER', P.pdi.NITER, ...
    'PICKOFF', P.pdi.pickoff, 'A_REF', P.pdi.a_ref, 'S_CONV', P.mask.S_CONV, ...
    'REF_SHAPE', P.pdi.ref_shape, 'FIB_V', P.pdi.fib_V, 'FIB_B', P.pdi.fib_b, 'FIB_A_LAMD', P.pdi.fib_a_lamd, ...
    'SCHEME', P.pdi.scheme, 'STEP_ERR', P.pdi.step_err);
end

function po = pdi_opt_deck_(P, lam_mm, dk)
% pdi_opt_ with the reference arm TRACED through the two P/SRI decks
po = pdi_opt_(P, lam_mm);  po.MODE = 'fiber';  po.REF_SHAPE = 'deck';  po.DECKS = dk;
po.REF_FROZEN = P.pdi.ref_frozen;
end

function PD = pdi_build_(P, lam_mm, iTO, iMASK, iDET)
% the point-diffraction factories the run asks for (fields P / PF), built on
% the FLAT DM (call with the flat loaded)
PD = struct();
if any(strcmp(P.readings, 'P')),  po = pdi_opt_(P, lam_mm);  po.MODE = 'pinhole';  PD.P  = dmg_pdi_gauge(iTO, iMASK, iDET, po); end
if any(strcmp(P.readings, 'PF')), po = pdi_opt_(P, lam_mm);  po.MODE = 'fiber';    PD.PF = dmg_pdi_gauge(iTO, iMASK, iDET, po); end
end

function k = class_(rd)
% kernel / transfer class of a reading: 1 = linear map, 2 = exact map, 3 = stepped, 4 = vector pair,
% 5 = point-diffraction pinhole, 6 = point-diffraction fiber reference
switch rd
    case 'L',          k = 1;
    case {'F','I','I+'}, k = 2;
    case 'S',          k = 3;
    case 'V',          k = 4;
    case 'P',          k = 5;
    case 'PF',         k = 6;
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
if strcmp(P.pdi.bench, 'psri'), S = stage_bench_psri_(P, rep);  return; end
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
bargs = bench_args_(P);
G = macos.design.twyman_green(bargs{:}, 'ngridpts', P.NGRID, ...
    'to_grid_file', P.grid.flat_file, 'to_grid_n', P.grid.N_G, 'to_grid_dx', P.grid.DX_G);
G.bt.wavelen = P.LAM;
deck = sprintf('%s_test.in', P.tag);
G.bt.emit(deck);
iTO = G.T.iTO;  iMASK = G.T.iMASK;  iDET = G.T.iDET;
macos.load_rx(deck);
coat_oap_(P, G.bt, rep);                                   % the OAP rig's mirror coating (bench.coat_oap), if any
wl = macos.get_src_wvl();
assert(abs(wl - P.LAM) < 1e-9*P.LAM, 'deck %s did not take Wavelen (%g vs %g)', deck, wl, P.LAM);
Z1 = G.bt.E(iMASK-1).zelt;  Z2 = G.bt.E(iMASK+1).zelt;
dmg_say(rep, 'deck %s: TO elt %d, FocalMask %d, Detector %d; mask sandwich spheres zElt %.3f / %.3f mm (%s)\n', ...
    deck, iTO, iMASK, iDET, Z1, Z2, ifelse_(abs(Z1-Z2) < 1e-9, 'SYMMETRIC', 'ASYMMETRIC -- Fresnel-defocused pupil'));

% ---- V3: AR coats on the refracting faces (polarization-mode physics only:
% the scalar traces never see a coating; the arm maps do)
if ischar(P.mask.v_analyzer) && strcmp(P.mask.v_analyzer, 'engine')     % V4: the analyzer's leak from the engine
    ana = dmg_analyzer_maps(P, 'qwp_err', P.mask.v_qwp_err, 'qwp_az', P.mask.v_qwp_az, 'NGRID', P.NGRID, 'MODEL', P.MODEL, 'deck_dir', pwd);   % the run's own size: no model-size transition in this process
    P.mask.v_analyzer = ana;  analyzer = ana;
    macos.load_rx(deck);  coat_oap_(P, G.bt, rep);                                   % back to the record deck
else
    analyzer = [];
end
if P.mask.v_arm_ar
    nar = 0;
    for k = 1:macos.num_elt()
        ei = macos.get_elt_info(k);
        if strcmp(ei.type, 'Refractor')
            macos.coating(k, 'index', P.mask.v_ar_n, 'extinc', 0, 'thickness', P.LAM/(4*P.mask.v_ar_n));  nar = nar + 1;
        end
    end
    dmg_say(rep, 'V3 arm: quarter-wave AR (n %.3f, %.1f nm) on %d refracting faces\n', P.mask.v_ar_n, P.LAM/(4*P.mask.v_ar_n)*1e6, nar);
end
% ---- the measurement factory + sensor gates --------------------------
gopt = gauge_opt_(P, P.LAM);
if strcmp(P.bench.mask_prop, 'nf_legacy'), warning('off', 'dmg_zwfs_gauge:roundtrip'); end
ZW = dmg_zwfs_gauge(iTO, iMASK, iDET, gopt);
warning('on', 'dmg_zwfs_gauge:roundtrip');
ZW.need_scalar = any(ismember(P.readings, {'L', 'F', 'I', 'I+'}));   % frames_: re-capture Ia when V's model is not the scalar sensor
ZW.PD = pdi_build_(P, P.LAM, iTO, iMASK, iDET);          % the point-diffraction readings (flat DM)
msk = ZW.msk;  N_WF = ZW.N_WF;
dimple_px = ZW.dia_mm*1e-3 / abs(macos.dx_at(iMASK));
dmg_say(rep, 'mask: phase %.4f rad (|c| %.3f), dimple %.3f lam/D = %.4e mm = %.2f px at the mask plane; msk %d px\n', ...
    gopt.PHI_M, abs(ZW.cc), gopt.DIA_LAMD, ZW.dia_mm, dimple_px, nnz(msk));
% the focal spot and the mask as the run samples them (flat DM; for the
% figure <tag>_mask.png -- the record's own resolution, not stage 1's)
dxm = abs(macos.dx_at(iMASK));  If = abs(macos.complex_field(iMASK)).^2;
hw = 24;  zc = round(ZW.ctr(1));  zr = round(ZW.ctr(2));
rz = max(1,zr-hw):min(N_WF,zr+hw);  cz = max(1,zc-hw):min(N_WF,zc+hw);
enc = sum(sum(If .* ZW.D)) / sum(If(:));                 % the light the dimple encloses (area-weighted disk)
maskfig = struct('If', If(rz, cz)/max(If(:)), 'mask_phase', gopt.PHI_M*ZW.D(rz, cz), ...
    'ctr', [ZW.ctr(1)-cz(1)+1, ZW.ctr(2)-rz(1)+1], 'dia_px', dimple_px, 'px_um', dxm*1e6, ...
    'px_per_lamd', dimple_px/gopt.DIA_LAMD, 'dia_lamd', gopt.DIA_LAMD, 'phi_m', gopt.PHI_M, ...
    'etch_nm', P.mask.ETCH_MM*1e6, 'enclosed', enc);
dmg_say(rep, 'focal spot: %.3f um per px at the mask plane, %.2f px per lam F/D; the dimple encloses %.1f%% of the flat DM''s focal-plane light\n', ...
    dxm*1e6, dimple_px/gopt.DIA_LAMD, 100*enc);
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
pin_px = NaN;
if ~isempty(fieldnames(ZW.PD))
    pdn = fieldnames(ZW.PD);  pd1 = ZW.PD.(pdn{1});
    pin_px = pd1.dia_mm*1e-3 / abs(macos.dx_at(iMASK));
    budget_(P, rep, pin_px >= P.samp.min_dimple_px, '  pinhole %.2f px across at the mask plane (min %g; the dimple''s rule)', pin_px, P.samp.min_dimple_px);
end
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
% ---- G4 (vector reading): exact beyond the one-frame fold, no branch ---
% Sparse-grid single-actuator pokes of P.mask.v_gate_nm (every 8th
% actuator): the reference wave barely moves (the core is intact) while
% the poked pixels sit beyond the quarter-wave sensor's -pi/4 fold, so
% the +phi/-phi pair must solve them exactly and the single +phi frame's
% principal branch must not.  Truth = the engine's own unmasked detector
% field phase; errors with the mean over msk removed (piston is the one
% direction the sensor cannot see).  A whole-pupil figure of that height
% is NOT a fold test: the core collapses (the 30-40 nm cliff, S7).
gV = struct('eV', NaN, 'eI', NaN, 'beyond', NaN, 'rmsfig', NaN);
if any(strcmp(P.readings, 'V'))
    if ZW.leak.eta < 1 || ~strcmp(P.mask.v_cal, 'ideal')
        if isfield(ZW, 'calV_info'), fitnote = sprintf(' (fit on the flat, resid %.1e)', ZW.calV_info.resid); else, fitnote = ''; end
        kP = mean(ZW.vcal.kapP(:));  kM = mean(ZW.vcal.kapM(:));   % maps in 'map' mode: their means
        dmg_say(rep, 'V2 metasurface: retardance error %.3f rad -> converts eta %.5f, leaks %.4f amplitude at phase %.2f rad (kappa_true %.5f %+.5fi); solver %s: kappa+ %.5f %+.5fi, kappa- %.5f %+.5fi, eta %.5f%s\n', ...
            P.mask.v_ret_err, ZW.leak.eta, sqrt(1-ZW.leak.eta), P.mask.v_leak_phase, real(ZW.vcal.kap_true), imag(ZW.vcal.kap_true), ...
            ZW.vcal.mode, real(kP), imag(kP), real(kM), imag(kM), ZW.vcal.eta, fitnote);
    end
    % ---- V3: the arm's per-channel maps ---------------------------------
    if ~strcmp(ZW.arm.mode, 'none')
        ai = ZW.arm.info;
        switch ZW.arm.mode
            case 'engine'
                st = ai.stats;
                dmg_say(rep, 'V3 arm (engine Jones pupil at elt %d, the mask sandwich''s entrance sphere; laser %.1f deg from the source x): exit axis [%.4f %.4f %.4f], beam half-cone %.1f deg, longitudinal residual %.1e, common-phase slope vs the pupil phase %+.4f (resid %.1e; -2 = the vector trace is the scalar''s conjugate), transmittance %.4f\n', ...
                    ai.iSRF, ai.laser_deg, ai.axis, ai.cone_deg, ai.leak, ai.common_slope, ai.common_resid, ai.T);
                dmg_say(rep, '  diattenuation mean %.2e, rms %.2e, max %.2e; retardance mean %.2e rad, rms %.2e, max %.2e\n', ...
                    st.D_mean, st.D_rms, st.D_max, st.ret_mean, st.ret_rms, st.ret_max);
                dmg_say(rep, '  per channel: |qL| %.5f (rms %.1e), |qR| %.5f (rms %.1e); phase about mean L %.2e rad rms, R %.2e; channel DIFFERENCE: phase %.2e rad rms (PV %.2e), amplitude ratio %.1e rms (PV %.1e)\n', ...
                    st.aL_mean, st.aL_rms, st.aR_mean, st.aR_rms, st.pL_rms, st.pR_rms, st.dphase_rms, st.dphase_pv, st.damp_rms, st.damp_pv);
            case 'synthetic'
                dmg_say(rep, 'V3 arm (synthetic astigmatic maps): channel differential phase %.4f rad rms, differential amplitude %.4f rms over msk\n', ai.dphase_rms, ai.damp_rms);
            otherwise
                dmg_say(rep, 'V3 arm: given maps\n');
        end
        dmg_say(rep, 'G8 chained pupil-map + dimple apodization: unit map vs the plain frame %.1e (gate < 1e-12); channel map vs the surrogate |qE0 + c b(qE0)|^2 on msk %.1e (gate < 1e-10)   -> %s\n', ...
            ZW.gate.chain, ZW.gate.chain_sur, ifelse_(ZW.gate.chain < 1e-12 && ZW.gate.chain_sur < 1e-10, 'PASS', 'FAIL'));
        assert(ZW.gate.chain < 1e-12 && ZW.gate.chain_sur < 1e-10, 'G8 FAIL');
        switch P.mask.v_cal
            case 'map',  armnote = 'the true maps (polarimetrically calibrated bench)';
            case 'fit',  armnote = 'per-channel unmasked amplitude maps + constants fitted on the flat''s masked images';
            case 'amp',  armnote = 'per-channel unmasked amplitude maps (the reference frames every bench takes; polarization phases unknown)';
            otherwise,   armnote = 'ideal (knows nothing of the arm: the raw size of the term)';
        end
        dmg_say(rep, '  solver arm model: %s\n', armnote);
    end
    if ~strcmp(ZW.ana.mode, 'none')
        if ~isempty(analyzer)
            sa = analyzer.stats;
            dmg_say(rep, 'V4 analyzer (engine: quarter-wave plate retardance error %.4f waves, azimuth error %.2f deg, MacNeille cube; cone at the pupil image %.2f deg): camera A main %s %.4f, incoherent leak %.2e (rms %.1e, max %.1e), coherent %.2e at %+.2f rad; camera B main %s %.4f, incoherent %.2e (rms %.1e, max %.1e), coherent %.2e at %+.2f rad; split A/B %.4f\n', ...
                analyzer.qwp_err, analyzer.qwp_az, analyzer.cone_deg, ...
                sa.A.main, sa.A.Pmain, sa.A.leak_mean, sa.A.leak_rms, sa.A.leak_max, abs(ZW.ana.cA), angle(ZW.ana.cA), ...
                sa.B.main, sa.B.Pmain, sa.B.leak_mean, sa.B.leak_rms, sa.B.leak_max, abs(ZW.ana.cB), angle(ZW.ana.cB), analyzer.split);
        else
            dmg_say(rep, 'V4 analyzer (given): lA %.2e, cA %.2e at %+.2f rad; lB %.2e, cB %.2e at %+.2f rad\n', ZW.ana.lA, abs(ZW.ana.cA), angle(ZW.ana.cA), ZW.ana.lB, abs(ZW.ana.cB), angle(ZW.ana.cB));
        end
        dmg_say(rep, '  the solver knows nothing of the analyzer: the V error below is its price\n');
    end
    Afig = zeros(cfg.nact);  Afig(4:8:end, 4:8:end) = P.mask.v_gate_nm*1e-6;
    macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), dmap(Afig));
    Et = macos.complex_field(iDET);
    phi_t = angle(Et .* conj(ZW.E0));  h_t = P.mask.S_CONV*phi_t*P.LAM/(4*pi);
    [Ip, Im] = ZW.frameV(dmap(Afig));
    hV = ZW.reconV(Ip, Im);  hI = ZW.reconI(Ip);
    pm_ = @(x) x(msk) - mean(x(msk));
    gV.rmsfig = std(h_t(msk))*1e9;
    gV.eV = sqrt(mean((pm_(hV) - pm_(h_t)).^2))*1e9;  gV.eI = sqrt(mean((pm_(hI) - pm_(h_t)).^2))*1e9;
    gV.beyond = mean(phi_t(msk) < -pi/4 | phi_t(msk) > 3*pi/4);
    dmg_say(rep, 'G4 vector pair on %g nm single-actuator pokes every 8th actuator (%.0f pm rms on msk, peak %.2f rad; %.2f%% of msk beyond the one-frame fold): V rms error %.3f pm (gate < 0.1%% of the figure), one-frame exact I %.0f pm (non-vacuity: must exceed 10x)   -> %s\n', ...
        P.mask.v_gate_nm, gV.rmsfig, max(abs(phi_t(msk))), 100*gV.beyond, gV.eV, gV.eI, ifelse_(gV.eV < 1e-3*gV.rmsfig && gV.eI > 10*gV.eV, 'PASS', 'FAIL'));
    priced = (ZW.leak.eta < 1 && strcmp(P.mask.v_cal, 'ideal')) || ...
             (~strcmp(ZW.arm.mode, 'none') && ~strcmp(P.mask.v_cal, 'map')) || ...
             ~strcmp(ZW.ana.mode, 'none');
    % ---- G9 (V5, plan 11.2): the pair read through a pupil AMPLITUDE dip.
    % The dip multiplies the unmasked poke field at the pupil image (a
    % Gaussian well of depth d over a quarter of the pupil, off center); the
    % frames are the surrogate's (== the engine's chained frames, G8).  The
    % phase-only solve with the FLAT's amplitude misreads by the dip; with
    % the STATE's clear frame (I0) it must not; the pair-only complex solve
    % (solveVA_) is printed for the record: ambiguous where A cos(phi)
    % crosses the reference wave, which 100 nm pokes do.
    if ~isempty(P.mask.v_dip)
        [ii, jj] = ndgrid(1:N_WF, 1:N_WF);  rp = sqrt(nnz(msk)/pi);
        cy = ZW.ctr(1) + 0.45*rp;  cx = ZW.ctr(2) + 0.30*rp;  sg = 0.25*rp;
        well = exp(-((ii-cy).^2 + (jj-cx).^2)/(2*sg^2));
        A0m = abs(ZW.E0);
        for d = P.mask.v_dip(:)'
            Adip = 1 - d*well;
            Ed = Adip .* Et;                                      % the G4 poke field with the dip
            [Ipd, Imd] = ZW.frameV_sur(Ed);
            hF = ZW.reconV(Ipd, Imd);                             % the flat's amplitude
            hC = ZW.reconV(Ipd, Imd, abs(Ed).^2);                 % the state's clear frame
            [phA, ia] = ZW.solveVA(Ipd, Imd, [], 0);  hA = P.mask.S_CONV*phA*P.LAM/(4*pi);   % the pair alone, one pass
            eF = sqrt(mean((pm_(hF) - pm_(h_t)).^2))*1e9;  eC = sqrt(mean((pm_(hC) - pm_(h_t)).^2))*1e9;
            eA = sqrt(mean((pm_(hA) - pm_(h_t)).^2))*1e9;  eAmp = sqrt(mean((ia.A(msk)./A0m(msk) - Adip(msk)).^2));
            ok = eC < 1e-3*gV.rmsfig && eF > 10*eC;
            dmg_say(rep, 'G9 vector pair through a %.0f%% pupil amplitude dip (Gaussian, sigma 0.25 of the pupil radius, off center): the flat''s amplitude %.2f pm; the state''s clear frame %.3f pm (gate < 0.1%% of the figure; non-vacuity: the flat''s > 10x); the pair alone %.0f pm, its amplitude %.1e rms off (ambiguous: A cos phi crosses b at these pokes)   -> %s\n', ...
                100*d, eF, eC, eA, eAmp, ifelse_(ok, 'PASS', 'FAIL'));
            assert(ok, 'G9 FAIL at a %.0f%% dip', 100*d);
        end
    end
    if ~priced
        assert(gV.eV < 1e-3*gV.rmsfig, 'G4 FAIL: the vector pair does not reproduce the figure');
        assert(gV.beyond > 0.005 && gV.eI > 10*gV.eV, 'G4 is vacuous: the single frame passes too -- raise mask.v_gate_nm');
    else
        dmg_say(rep, '  (G4 not asserted: an uncalibrated metasurface / arm / analyzer error is being priced -- the V error above IS the number)\n');
    end
end
% ---- the station walk-through figures (Dave 2026-09-15: the key signals
% along the train, the Keysight deck's station-by-station model): for each
% of S / V / P present, two rows -- the flat DM and the 30 nm working
% surface -- across the mirror command, the focal spot with the mask's
% footprint, the mask, the reference wave, two camera frames, the
% recovered surface and its residual against the engine's own field.
if ~isfield(P, 'figs') || ~isfield(P.figs, 'stations') || P.figs.stations
    for rd_ = intersect({'S', 'V', 'P'}, P.readings, 'stable')
        stations_fig_(rd_{1}, P, rep, ZW, dmap, Ab, iTO, iMASK, iDET, msk, gopt, rz, cz);
    end
end
% ---- G5-G7 (point-diffraction readings) ---------------------------------
% G5: exact beyond the fold on the SAME sparse pokes as G4 (truth = the
% engine's unmasked field phase); G6: the reference's motion under the
% working state vs pinhole diameter (the PDI's argument, measured against
% the dimple); G7: with t = 1 at the dimple's diameter the pinhole reading
% IS the stepped Zernike reading S (structural identity, frozen reference).
gP = struct();
if ~isempty(fieldnames(ZW.PD))
    if ~exist('h_t', 'var')
        Afig = zeros(cfg.nact);  Afig(4:8:end, 4:8:end) = P.mask.v_gate_nm*1e-6;
        macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), dmap(Afig));
        Et = macos.complex_field(iDET);
        phi_t = angle(Et .* conj(ZW.E0));  h_t = P.mask.S_CONV*phi_t*P.LAM/(4*pi);
        pm_ = @(x) x(msk) - mean(x(msk));
    end
    rmsfig = std(h_t(msk))*1e9;  beyond = mean(phi_t(msk) < -pi/4 | phi_t(msk) > 3*pi/4);
    for pdn = fieldnames(ZW.PD).'
        rd = pdn{1};  pd = ZW.PD.(rd);
        switch pd.mode
            case 'pinhole', desc = sprintf('pinhole %.2f lam/D, surround t %.4f, %d steps, |b|^2 %s, NITER %d', P.pdi.DIA_LAMD, pd.t, pd.K, pd.b2mode, pd.NITER);
            case 'fiber'
                switch pd.refshape
                    case 'fiber',   shp = sprintf('LP01 mode V %.2f b %.2f core radius %.2f lam/D, coupling eta_c %.4f on the flat', P.pdi.fib_V, P.pdi.fib_b, P.pdi.fib_a_lamd, pd.eta_c);
                    case 'pinhole', shp = sprintf('pinhole %.2f lam/D shape', P.pdi.DIA_LAMD);
                end
                desc = sprintf('P/SRI, reference = %s; pickoff %.2f, a %.4g (match %.4g, budget %.4g), %d steps (%s%s), |kappa| %s', ...
                    shp, pd.f, pd.a, pd.a_match, pd.a_budget, pd.K, pd.scheme, ifelse_(pd.step_err ~= 0, sprintf(', step error %+.3f', pd.step_err), ''), pd.b2mode);
        end
        dmg_say(rep, '%s: %s; frames per measurement %d; eta_pin %.4f; throughput (detected/incident, flat) %.4f; visibility on the flat %.4f; surrogate vs engine reference %.2e; the flat reads %.2e rad rms, amplitude %.2e rel\n', ...
            rd, desc, pd.nframes, pd.eta_pin, pd.throughput, pd.vis, pd.gate.bsur, pd.gate.flat, pd.gate.amp);
        assert(pd.gate.bsur < 1e-10, '%s: reference surrogate gate FAIL (%.2e)', rd, pd.gate.bsur);
        priced = pd.step_err ~= 0;          % a deliberate step error is being PRICED: the gates print, they do not assert
        if ~priced, assert(pd.gate.flat < 1e-9, '%s: the flat does not read zero (%.2e rad rms)', rd, pd.gate.flat); end
        hP = pd.height(pd.frames(dmap(Afig)));
        e5 = sqrt(mean((pm_(hP) - pm_(h_t)).^2))*1e9;
        dmg_say(rep, 'G5 %s on %g nm single-actuator pokes every 8th actuator (%.0f pm rms on msk, %.2f%% of msk beyond the one-frame fold): rms error %.3f pm (gate < 0.1%% of the figure)   -> %s%s\n', ...
            rd, P.mask.v_gate_nm, rmsfig, 100*beyond, e5, ifelse_(e5 < 1e-3*rmsfig, 'PASS', 'FAIL'), ifelse_(priced, sprintf(' (not asserted: a %+.3f step error is being priced -- the flat reads %.2e rad rms, this error IS the number)', pd.step_err, pd.gate.flat), ''));
        if ~priced, assert(e5 < 1e-3*rmsfig, 'G5 FAIL: %s does not reproduce the figure', rd); end
        gP.(rd) = struct('e5', e5, 'throughput', pd.throughput, 'vis', pd.vis, 'nframes', pd.nframes);
    end
    % G6: reference motion under the working state (E1 traced at G3), by diameter
    % (the pinhole-shaped reference: from the P factory, else a pinhole-shaped PF)
    pd1 = [];
    for pdn2 = fieldnames(ZW.PD).', if strcmp(ZW.PD.(pdn2{1}).refshape, 'pinhole'), pd1 = ZW.PD.(pdn2{1}); end, end
    if isfield(ZW.PD, 'PF') && strcmp(ZW.PD.PF.refshape, 'fiber')
        [~, ~, kap] = ZW.PD.PF.refstab(E1);
        dmg_say(rep, 'G6 P/SRI fiber reference under the %.0f nm rms working state: shape FIXED by construction (the mode''s); coupling relative to the flat |kappa| %.5f, arg %+.4f rad (an amplitude scale + a piston; the solver takes kappa = 1 in |kappa| ''flat'' mode)\n', ...
            P.battery.base_rms*1e6, abs(kap), angle(kap));
        gP.kappa = kap;
    end
    if isempty(pd1), pd1 = dmg_pdi_gauge(iTO, iMASK, iDET, setfield(setfield(pdi_opt_(P, P.LAM), 'MODE', 'pinhole'), 'NITER', 0)); macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), zeros(P.grid.N_G)); end %#ok<SFLD>
    dias = P.pdi.refstab_dia;
    [rs, rsh, scl] = pd1.refstab(E1, dias);
    [rd_, rdh, sdl] = pd1.refstab(E1, P.mask.DIA_LAMD);
    dmg_say(rep, 'G6 reference motion under the %.0f nm rms working state (pinhole-diffracted reference, flat -> state, on msk), by pinhole diameter:\n%8s %8s %8s %8s   %s\n', ...
        P.battery.base_rms*1e6, 'lam/D', 'total', '|scale|', 'SHAPE', '(total = |b1-b0|/|b0|; scale = best complex factor, a Strehl-class amplitude drop the |b|^2 frame / iteration absorbs; shape = the rest, what the solve must iterate out)');
    for i = 1:numel(dias), dmg_say(rep, '%8.2f %8.4f %8.4f %8.4f\n', dias(i), rs(i), abs(scl(i)), rsh(i)); end
    dmg_say(rep, '%8.2f %8.4f %8.4f %8.4f   <- the ZWFS dimple\n', P.mask.DIA_LAMD, rd_, abs(sdl), rdh);
    gP.refstab = rs;  gP.refstab_shape = rsh;  gP.refstab_scale = scl;  gP.refstab_dia = dias;
    % G7: t = 1 at the dimple diameter, frozen reference == the stepped reading S
    if isfield(ZW.PD, 'P')
        po = pdi_opt_(P, P.LAM);  po.MODE = 'pinhole';  po.DIA_LAMD = P.mask.DIA_LAMD;  po.T_SURR = 1;  po.NITER = 0;  po.B2 = 'flat';
        po.SCHEME = 'ls';  po.THETAS = [0 pi/2 pi 3*pi/2];  po.STEP_ERR = 0;      % the identity is tested with S's own steps, no priced error
        macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), zeros(P.grid.N_G));
        pdS = dmg_pdi_gauge(iTO, iMASK, iDET, po);
        Aq = zeros(cfg.nact);  Aq(4:8:end, 4:8:end) = P.reg.POKE;
        FrS = ZW.framesS(dmap(Aq));  hS = ZW.stepdiff(ZW.reconS(FrS), ZW.reconS(ZW.framesS(zeros(P.grid.N_G))));
        hQ = pdS.height(pdS.frames(dmap(Aq)));
        e7 = norm(pm_(hQ) - pm_(hS)) / norm(pm_(hS));
        dmg_say(rep, 'G7 the pinhole reading at t = 1, %.2f lam/D, frozen reference vs the stepped Zernike reading S on %g nm sparse pokes: rel rms diff %.2e   (gate < 1e-2: same instrument, same frames)\n', P.mask.DIA_LAMD, P.reg.POKE*1e6, e7);
        assert(e7 < 1e-2, 'G7 FAIL: the pinhole reading does not reduce to S at t = 1');
        gP.e7 = e7;
    end
    % ---- figure data for zwfs_run_figs (<tag>_pdi.png): the focal plane with
    % the pinhole, the dimple and the waveguide mode; the reference amplitudes
    % across the pupil; the visibility maps on the flat (4x decimated)
    fd = struct();
    wf = 24;  r0 = round(pd1.ctr(2));  c0 = round(pd1.ctr(1));
    rr = max(1, r0-wf):min(N_WF, r0+wf);  cc = max(1, c0-wf):min(N_WF, c0+wf);
    Efoc = pd1.C.Ti(ZW.E0);
    fd.px_per_lamd = (P.LAM*P.bench.F2/(2*P.bench.R_TO_AP)) / abs(macos.dx_at(iMASK)*1e3);
    fd.spot = abs(Efoc(rr, cc));  fd.pinhole = pd1.D(rr, cc);  fd.dimple = ZW.D(rr, cc);
    fd.pin_dia_lamd = P.pdi.DIA_LAMD;  fd.dimple_dia_lamd = P.mask.DIA_LAMD;
    if isfield(ZW.PD, 'PF') && ~isempty(ZW.PD.PF.mode_f), fd.mode = abs(ZW.PD.PF.mode_f(rr, cc)); else, fd.mode = []; end
    dec = @(x) x(1:4:end, 1:4:end);
    fd.E0 = dec(abs(ZW.E0));  fd.Eb = dec(abs(pd1.Eb0));  fd.msk = dec(msk);
    if isfield(ZW.PD, 'PF'), fd.Rfib = dec(abs(ZW.PD.PF.R)); else, fd.Rfib = []; end
    for pdn = fieldnames(ZW.PD).'
        pd = ZW.PD.(pdn{1});  Fr0 = pd.frames0(:,:,1:pd.K);
        Imax = max(Fr0, [], 3);  Imin = min(Fr0, [], 3);
        fd.(['vis_' pdn{1}]) = dec((Imax - Imin) ./ max(Imax + Imin, realmin));
        fd.(['frame0_' pdn{1}]) = dec(Fr0(:,:,1));
    end
    gP.figdata = fd;
end
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
    'ampmod_std',std(rr), 'g4',gV, 'g567',gP, 'pin_px',pin_px, 'mag',mag, 'dxd_mm',dxd_mm, 'px_per_act',ppa, 'PARb',PARb, 'sgn',sgn, ...
    'kernel_peak',max(hAd(:))/max(Ma(:)), 'kernel_corr',cpm(1,2), 'anchor',[R.bx R.by R.tax R.tay]);
S.summary.maskfig = maskfig;                     % the focal spot + mask windows for <tag>_mask.png
S.summary.analyzer = analyzer;                   % V4: the analyzer's leak maps and stats ([] when 'none')
end

% =====================================================================
%  stage: bench -- the P/SRI's own bench (pdi.bench 'psri', 2026-09-13)
% =====================================================================
function S = stage_bench_psri_(P, rep)
%STAGE_BENCH_PSRI_  The buildable P/SRI (macos.design.psri_bench): the TG96
%   front end, then a Mach-Zehnder whose REFERENCE arm carries the pinhole
%   and the phase shifter.  The reading is PF with its reference physically
%   traced through the reference deck -- no synthesized mode, no surrogate.
%   Only PF lives here: the test arm's own seat is empty (the P/SRI filters
%   in the other arm), so the dimple readings (L F I I+ S V) and the
%   common-path pinhole P have no mask to sit in and are refused.
t0 = tic;
dmg_say(rep, '\n---- bench: the P/SRI, two traced arms ----\n');
assert(isequal(P.readings, {'PF'}), ...
    'zwfs_run: pdi.bench ''psri'' carries the PF reading only (the test arm''s seat is empty); readings = {''PF''}');
if ~isempty(P.param_file)
    pf = P.param_file;
    if ~isfile(pf), pf = fullfile(fileparts(mfilename('fullpath')), P.param_file); end
    if ~isfile(pf), pf = which(P.param_file); end
    assert(~isempty(pf) && isfile(pf), 'zwfs_run: param_file %s not found', P.param_file);
    copyfile(pf, 'macos_param.txt');
    dmg_say(rep, 'engine size table: %s (copied into the run dir as macos_param.txt)\n', pf);
elseif exist('macos_param.txt', 'file')
    delete('macos_param.txt');
end
macos.init(P.MODEL);
macos.write_grid_file(P.grid.flat_file, zeros(P.grid.N_G));
% the bench: the front end's values are the ZWFS record's (P.bench, minus the
% keys twyman_green owns and psri_bench does not), plus P.pdi.psri
bf = intersect(fieldnames(P.bench), fieldnames(psri_opts_()));
bargs = {};
for i = 1:numel(bf), bargs = [bargs, {bf{i}, P.bench.(bf{i})}]; end %#ok<AGROW>
pf2 = fieldnames(P.pdi.psri);
for i = 1:numel(pf2), bargs = [bargs, {pf2{i}, P.pdi.psri.(pf2{i})}]; end %#ok<AGROW>
G = macos.design.psri_bench(bargs{:}, 'ngridpts', P.NGRID, ...
    'to_grid_file', P.grid.flat_file, 'to_grid_n', P.grid.N_G, 'to_grid_dx', P.grid.DX_G);
G.bt.wavelen = P.LAM;  G.br.wavelen = P.LAM;
deck = sprintf('%s_test.in', P.tag);  rdeck = sprintf('%s_ref.in', P.tag);
G.bt.emit(deck);  G.br.emit(rdeck);
iTO = G.T.iTO;  iDET = G.T.iDET;  iMASK = G.T.iSEAT;      % the seat: empty on this arm
macos.load_rx(deck);
wl = macos.get_src_wvl();
assert(abs(wl - P.LAM) < 1e-9*P.LAM, 'deck %s did not take Wavelen (%g vs %g)', deck, wl, P.LAM);
B = G.balance;
dmg_say(rep, 'decks %s / %s: TO elt %d, Detector %d (test); Pinhole %d, Detector %d (reference)\n', ...
    deck, rdeck, iTO, iDET, G.R.iPIN, G.R.iDET);
dmg_say(rep, 'balance: chief optical path test %.4f mm, reference %.4f mm (difference %.2e); compensator %.3f mm of n = %.2f; exit chiefs %.2e mm apart; camera planes %.2e mm apart\n', ...
    B.opl_test_mm, B.opl_ref_mm, B.dopl_mm, B.t_comp_mm, G.P.N_GLASS, B.exit_offset_mm, B.det_offset_mm);
dmg_say(rep, 'reference arm: Lr1 f %.1f mm (F/%.2f on the %.1f mm beam), conic %.4f; pinhole seat trim %+.3f mm; Lr2 = Lr1 mirrored (conic %.4f)\n', ...
    G.P.F_REF, G.P.F_REF/(2*G.P.R_TO_AP), 2*G.P.R_TO_AP, G.P.LR1_Kc, G.P.REF_TRIM, G.P.LR2_Kc);
% ---- the PF factory with BOTH arms traced ---------------------------------
dk = struct('test', fullfile(pwd, deck), 'ref', fullfile(pwd, rdeck), 'iTO', iTO, ...
            'iDET_test', iDET, 'iPIN', G.R.iPIN, 'iDET_ref', G.R.iDET, ...
            'F_PIN', G.P.F_REF, 'R_PIN', G.P.R_TO_AP);
pd = dmg_pdi_gauge(iTO, iMASK, iDET, pdi_opt_deck_(P, P.LAM, dk));
msk = pd.msk;  N_WF = pd.N_WF;
ZW = struct('msk', msk, 'N_WF', N_WF, 'E0', pd.E0, 'Eb0', pd.Eb0, 'D', pd.D, ...
            'has_scalar', false, 'has_step', false, 'need_scalar', false, 'v_scalar_equiv', false, ...
            'dia_mm', pd.dia_mm, 'ctr', pd.ctr, 'gate', pd.gate, 'PD', struct('PF', pd));
ZW.measL = @(M) pd.height(pd.frames(M));                  % the reading is its own registration map
pin_px = pd.dia_mm / pd.dx_mask_mm;
dmg_say(rep, 'PF: pinhole %.3f lam/D of the reference lens = %.4e mm = %.2f px at its focus (%.3f um per px); coupling eta %.4f; reference amplitude a %.4g (match %.4g, budget %.4g); %d steps (%s); frames per measurement %d\n', ...
    P.pdi.DIA_LAMD, pd.dia_mm, pin_px, pd.dx_mask_mm*1e3, pd.eta_pin, pd.a, pd.a_match, pd.a_budget, pd.K, pd.scheme, pd.nframes);
dmg_say(rep, 'PF: throughput (detected/incident, flat) %.4f; visibility on the flat %.4f; msk %d px; the flat reads %.2e rad rms, amplitude %.2e rel\n', ...
    pd.throughput, pd.vis, nnz(msk), pd.gate.flat, pd.gate.amp);
if pd.frozen
    dmg_say(rep, 'PF: REFERENCE FROZEN at the flat state (pdi.ref_frozen) -- the traced arm is captured once and reused, so this run carries the real arm''s SHAPE but not its MOTION: the control that separates the two\n');
end
priced = P.pdi.step_err ~= 0;
if ~priced, assert(pd.gate.flat < 1e-9, 'PF: the flat does not read zero (%.2e rad rms)', pd.gate.flat); end
budget_(P, rep, pin_px >= P.samp.min_dimple_px, '  pinhole %.2f px across at the reference focus (min %g; the dimple''s rule)', pin_px, P.samp.min_dimple_px);
% ---- frame + sampling budget ----------------------------------------------
[mag, dxd_mm] = dmg_frame(iTO, iDET);
xg = ((0:P.grid.N_G-1)-(P.grid.N_G-1)/2)*P.grid.DX_G;
[gxd, gyd] = meshgrid(xg, xg);
dmg_say(rep, 'frame: ray magnification %.4f DM-mm per detector-mm, detector px %.4e mm -> %.4f DM-mm per px\n', mag, dxd_mm, mag*dxd_mm);
ppa = zeros(1, numel(P.dm));
for ic = 1:numel(P.dm)
    ppa(ic) = P.dm(ic).pitch / (mag*dxd_mm);
    budget_(P, rep, ppa(ic) >= P.samp.min_px_per_act, '  DM %dx%d: %.2f detector px per actuator (min %g)', ...
        P.dm(ic).nact, P.dm(ic).nact, ppa(ic), P.samp.min_px_per_act);
end
% ---- G3 / G5 / G6 ----------------------------------------------------------
cfg = P.dm(1);
dmap = @(act) dm_influence_map(P.grid.N_G, P.grid.DX_G, 'nact', cfg.nact, 'pitch', cfg.pitch, 'act', act);
[axg, ayg] = meshgrid(((1:cfg.nact)-(cfg.nact+1)/2)*cfg.pitch);
lit = dmg_lit(msk, dxd_mm, mag, axg, ayg);
rng(P.battery.seed_base);  Ab = zeros(cfg.nact);  Ab(lit) = P.battery.base_rms*randn(nnz(lit),1);
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), dmap(Ab));
E1 = macos.complex_field(iDET);
rr = abs(E1(msk))./abs(pd.E0(msk));
dmg_say(rep, 'G3 %g nm rms working state (DM %dx%d, lit %d): detector |E|/|E_flat| on msk std %.3e   (gate < 1e-12)\n', ...
    P.battery.base_rms*1e6, cfg.nact, cfg.nact, nnz(lit), std(rr));
Afig = zeros(cfg.nact);  Afig(4:8:end, 4:8:end) = P.mask.v_gate_nm*1e-6;
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), dmap(Afig));
Et = macos.complex_field(iDET);
phi_t = angle(Et .* conj(pd.E0));  h_t = P.mask.S_CONV*phi_t*P.LAM/(4*pi);
pm_ = @(x) x(msk) - mean(x(msk));
rmsfig = std(h_t(msk))*1e9;  beyond = mean(phi_t(msk) < -pi/4 | phi_t(msk) > 3*pi/4);
hP = pd.height(pd.frames(dmap(Afig)));
e5 = sqrt(mean((pm_(hP) - pm_(h_t)).^2))*1e9;
dmg_say(rep, 'G5 PF on %g nm single-actuator pokes every 8th actuator (%.0f pm rms on msk, %.2f%% beyond the one-frame fold): rms error %.3f pm = %.3f%% of the figure   -> %s\n', ...
    P.mask.v_gate_nm, rmsfig, 100*beyond, e5, 100*e5/rmsfig, ifelse_(e5 < 1e-3*rmsfig, 'within the 0.1% ideal-reference line', 'ABOVE the 0.1% ideal-reference line'));
dmg_say(rep, '     (on this bench G5 is a MEASUREMENT, not a gate: the reference arm is TRACED, so it moves with the state while the solver uses the flat''s R0 -- the synthesized reference reads 0.000 pm here by construction.  The assert is the 1%% sanity line)\n');
if ~priced, assert(e5 < 1e-2*rmsfig, 'G5 FAIL: PF does not reproduce the figure at all (%.1f%%)', 100*e5/rmsfig); end
[r6, rsh6, scl6] = pd.refstab(dmap(Ab));
dmg_say(rep, 'G6 the TRACED reference arm under the %g nm rms working state: total change %.4f, best complex scale |%.5f| arg %+.4f rad (the solver absorbs it through the shutter frame + a piston), SHAPE change %.4f -- the part nothing absorbs\n', ...
    P.battery.base_rms*1e6, r6, abs(scl6), angle(scl6), rsh6);
gP = struct('PF', struct('e5', e5, 'throughput', pd.throughput, 'vis', pd.vis, 'nframes', pd.nframes), ...
            'kappa', scl6, 'refstab_deck', [r6 rsh6], 'pin_px', pin_px, 'balance', B);
% ---- registration ---------------------------------------------------------
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
else
    PARb = P.reg.PARb;  sgn = P.reg.sgn;
    dmg_say(rep, '  parity [%s] sign %+d taken from P.reg (mode record)\n', num2str(PARb), sgn);
end
R.P = PARb;
hAd = sgn*dmg_samp(hA, R);  hAd(isnan(hAd)) = 0;
cpm = corrcoef(hAd(:), Ma(:));
dmg_say(rep, '  center poke in the DM frame: raw peak gain %.4f, corr(map, truth) %.4f\n', max(hAd(:))/max(Ma(:)), cpm(1,2));
macos.load_rx(deck);  coat_oap_(P, G.bt, []);
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), zeros(P.grid.N_G));
dmg_say(rep, 'bench stage %.1f min\n', toc(t0)/60);
S = struct('G',G, 'deck',deck, 'rdeck',rdeck, 'dk',dk, 'iTO',iTO, 'iMASK',iMASK, 'iDET',iDET, 'ZW',ZW, ...
    'gopt',struct(), 'mag',mag, 'dxd_mm',dxd_mm, 'xg',xg, 'gxd',gxd, 'gyd',gyd, 'PARb',PARb, 'sgn',sgn);
S.summary = struct('deck',deck, 'rdeck',rdeck, 'bench','psri', 'nmsk',nnz(msk), 'g567',gP, ...
    'pin_px',pin_px, 'mag',mag, 'dxd_mm',dxd_mm, 'px_per_act',ppa, 'PARb',PARb, 'sgn',sgn, ...
    'ampmod_std',std(rr), 'balance',B, 'kernel_peak',max(hAd(:))/max(Ma(:)), 'kernel_corr',cpm(1,2), ...
    'anchor',[R.bx R.by R.tax R.tay]);
end

function o = psri_opts_()
% the psri_bench option names the ZWFS bench sheet shares (front end + tail)
o = struct('F1',0, 'F2',0, 'BS_AOI',0, 'D_LENS',0, 'N_GLASS',0, 'R_BAFFLE',0, 'D_SB',0, 'FILL',0, ...
           'BS_T',0, 'D_L1_BS',0, 'D_BS_TO',0, 'D_BS_CMP',0, 'D_RECOMB',0, 'R_TO_AP',0, ...
           'L1_Kr',0, 'L1_Kc',0, 'L2_Kr',0, 'L2_Kc',0, 'to_Kr',0, ...
           'MASK_TRIM',0, 'FL_F',0, 'FL_Kc',0, 'FL_D',0, 'D_MASK_FL',0, 'DET_TRIM',0);
end

% =====================================================================
%  frames and readings
% =====================================================================
function F = frames_(ZW, M, needS, needV, needP, aux)
% capture the frames one DM state needs: Ia (the one masked frame of the
% scalar readings), when needS the stepped set Fr + its rank-2 retrieval
% X, when needV the vector pair Ip / Im (+phi / -phi images; Ip IS the
% scalar frame when the V model is the scalar sensor -- ideal metasurface,
% no arm maps -- and Ia is then not re-captured), and when needP = [P PF]
% the point-diffraction frame sets FP (pinhole) / FF (fiber)
if nargin < 4, needV = false; end
if nargin < 5, needP = [false false]; end
if nargin < 6, aux = []; end
has_scalar = ~isfield(ZW, 'has_scalar') || ZW.has_scalar;    % the P/SRI bench has neither the
has_step   = ~isfield(ZW, 'has_step')   || ZW.has_step;      % dimple's frame nor its stepped set
ds = [];  if ~isempty(aux) && isfield(aux, 'dstep') && any(aux.dstep(:) ~= 0), ds = aux.dstep; end
if needV
    [Ip, Im] = ZW.frameV(M);                                 % simultaneous: no within-scan drift
    if ZW.v_scalar_equiv || ~ZW.need_scalar, Ia = Ip; else, Ia = ZW.frameL(M); end
elseif has_scalar
    Ia = ZW.frameL(M);  Ip = [];  Im = [];
else
    Ia = [];  Ip = [];  Im = [];
end
F = struct('Ia', Ia, 'Ip', Ip, 'Im', Im, 'Fr', [], 'X', [], 'FP', [], 'FF', []);
if needS && has_step
    if isempty(ds)
        F.Fr = ZW.framesS(M);
    else
        % the DM advancing across the scan: frame j at its own state.  The
        % gauge captures a whole set per state and we keep the j-th, so the
        % four frames are the four states' -- what a stepped reading on a
        % drifting DM actually gets (cost: 4 captures, 4 traces).
        F.Fr = ZW.framesS(M);
        for j = 2:4
            Fj = ZW.framesS(M + (j-1)/3 * ds);  F.Fr(:,:,j) = Fj(:,:,j);
        end
    end
    F.X = ZW.reconS(F.Fr);
end
if needP(1), F.FP = ZW.PD.P.frames(M, aux); end
if needP(2), F.FF = ZW.PD.PF.frames(M, aux); end
end

function h = readmap_(ZW, rd, F, plus, Xref)
% absolute height map (mm) of one reading from captured frames
switch rd
    case 'L',  h = ZW.reconL(F.Ia);
    case 'F',  h = ZW.reconI(F.Ia, [], [], [], 0);
    case 'I',  h = ZW.reconI(F.Ia);
    case 'I+', h = ZW.reconI(F.Ia, [], plus);
    case 'S',  h = ZW.stepdiff(F.X, Xref);
    case 'V',  h = ZW.reconV(F.Ip, F.Im);
    case 'P',  h = ZW.PD.P.height(F.FP);
    case 'PF', h = ZW.PD.PF.height(F.FF);
end
end

function d = diff_(ZW, rd, F1, F0, plus, uw)
% differential height map between two captured states.  uw (optional) is
% the unwrapper from unwrap_fn_: the four wrapped readings get it, the
% others do not (wrapped_).
if nargin < 6, uw = []; end
switch rd
    case 'S', d = ZW.stepdiff(F1.X, F0.X);
    case 'V', d = ZW.diffV(F1.Ip, F1.Im, F0.Ip, F0.Im);          % wrapped phase difference
    case 'P',  d = ZW.PD.P.diff(F1.FP, F0.FP);                     % wrapped phase difference
    case 'PF', d = ZW.PD.PF.diff(F1.FF, F0.FF);
    otherwise, d = readmap_(ZW, rd, F1, plus, []) - readmap_(ZW, rd, F0, plus, []);
end
if ~isempty(uw) && wrapped_(rd), d = uw(d); end
end

function C = calibrate_(P, S, ZW, cfg, classes, lit, Abase)
% per-class registration anchor + measured response kernel + estimator.
% P.battery.calib_mode 'matrix' (Dave 2026-09-10) replaces the single-site
%   kernel by the MEASURED response matrix dw/da: see calib_matrix_.
if nargin < 6, lit = []; end
if nargin < 7, Abase = []; end
if strcmp(P.battery.calib_mode, 'matrix'), C = calib_matrix_(P, S, ZW, cfg, classes, lit, Abase); return; end
assert(isempty(Abase), 'zwfs_run: an explicit calibration surface needs battery.calib_mode ''matrix''');
assert(~any(classes >= 4), 'zwfs_run: the vector reading V and the point-diffraction readings P / PF are supported in battery.calib_mode ''matrix'' only');
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
C.lit = lit;  C.uw = unwrap_fn_(P, ZW);
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
    C.cmap = @(F, k) cmap_base_(ZW, F, C.F0, C.plus, k, C.uw);
else
    C.cmap = @(F, k) cmap_flat_(ZW, F, C.Fflat, k, C.uw);
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
C.R = cell(1,6);  C.stn = cell(1,6);  C.est = cell(1,6);  C.kinfo = nan(6,3);
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

function C = calib_matrix_(P, S, ZW, cfg, classes, lit, Abase)
%CALIB_MATRIX_  The measured response matrix dw/da (Dave 2026-09-10).
%   Poke every STEP-th actuator in a sparse grid so no two responses
%   overlap, step through the STEP^2 grid offsets so every lit actuator is
%   poked once (STEP^2 states instead of one per actuator), and cut each
%   actuator's response out of its own window in DETECTOR pixels.  The
%   columns form J (detector px x lit actuators) per reading class; the
%   estimator solves min |J a - m|^2 + l2 |a|^2 by a sparse direct solve.
%   No single-site kernel, no shift-invariance, no frequency correction;
%   registration only PLACES the windows (half a grid step wide, so a
%   few-pixel error is harmless) -- the columns carry the actual response
%   wherever it lands, so position dependence and irregularities of a real
%   DM are in the calibration by construction.  On a working surface
%   (calib_surface 'base') the pokes ride on the base and the class maps
%   are differential, exactly as in kernel mode.
N_G = P.grid.N_G;  xg = S.xg;  nact = cfg.nact;
C.dmap = @(act) dm_influence_map(N_G, P.grid.DX_G, 'nact', nact, 'pitch', cfg.pitch, 'act', act);
[C.axg, C.ayg] = meshgrid(((1:nact)-(nact+1)/2)*cfg.pitch);
if isempty(lit), lit = dmg_lit(ZW.msk, S.dxd_mm, S.mag, C.axg, C.ayg); end
C.lit = lit;  C.mode = 'matrix';  C.uw = unwrap_fn_(P, ZW);
POKE = P.reg.POKE;  step = P.battery.matrix_step;
ic = nact/2;  Aa = zeros(nact);  Aa(ic,ic) = 1;  C.Ma = C.dmap(POKE*Aa);
needS = any(classes == 3);  needV = any(classes == 4);  needP = [any(classes == 5) any(classes == 6)];
C.surface = P.battery.calib_surface;  onbase = strcmp(C.surface, 'base');
if nargin >= 7 && ~isempty(Abase), onbase = true;  C.surface = 'current'; end
has_step = ~isfield(ZW, 'has_step') || ZW.has_step;     % the P/SRI bench has no stepped set
needS0 = has_step;                                      % (the prior the I+ reading needs)
if onbase
    if nargin >= 7 && ~isempty(Abase)
        Ab = Abase;                                     % the surface the loop holds NOW (ins.recal)
    else
        rng(P.battery.seed_base);  Ab = zeros(nact);  Ab(lit) = P.battery.base_rms*randn(nnz(lit),1);
    end
    C.Abase = Ab;  C.F0 = frames_(ZW, C.dmap(Ab), needS0, needV, needP);
    if has_step, C.plus = ZW.priorS(C.F0.Ia, C.F0.Fr); else, C.plus = []; end
    C.Fflat = frames_(ZW, zeros(N_G), needS, needV, needP);
    C.cmap = @(F, k) cmap_base_(ZW, F, C.F0, C.plus, k, C.uw);
else
    C.Abase = zeros(nact);  C.Fflat = frames_(ZW, zeros(N_G), needS, needV, needP);  C.F0 = C.Fflat;  C.plus = [];
    C.cmap = @(F, k) cmap_flat_(ZW, F, C.Fflat, k, C.uw);
end
% ---- window placement from the bench registration (anchor from the
% centre poke, parity/sign from the bench stage): DM lattice -> detector px
Fa = frames_(ZW, C.dmap(C.Abase + POKE*Aa), needS, needV, needP);
k1 = classes(1);
hA = C.cmap(Fa, k1);
[bx, by, tax, tay] = dmg_anchor(hA, C.Ma, ZW.msk, ZW.N_WF, xg);
lat = ((1:nact)-(nact+1)/2)*cfg.pitch;
Pp = S.PARb;  sc = S.dxd_mm*S.mag;
[cc_, rr_] = meshgrid(1:nact, 1:nact);            % actuator (r,c): x = lat(c), y = lat(r)
off = {lat(cc_) - tax, lat(rr_) - tay};
U = Pp(3)*off{Pp(1)}/sc + bx;  V = Pp(4)*off{Pp(2)}/sc + by;   % detector (col,row) of every actuator
hw_px = floor(0.5*step*cfg.pitch/sc);                            % half a grid step, px
C.win = struct('U',U, 'V',V, 'hw_px',hw_px, 'step',step);
% ---- the multiplexed poke sets ------------------------------------------
N = ZW.N_WF;  ilit = find(lit);  nlit = numel(ilit);  col_of = zeros(nact);  col_of(ilit) = 1:nlit;
I = cell(1,6);  Jc = cell(1,6);  V3 = cell(1,6);
for k = classes(:).', I{k} = {};  Jc{k} = {};  V3{k} = {}; end
nstates = 0;  npoked = 0;  pk = zeros(1,6);  nclip = 0;
t0 = tic;
for ox = 1:step
    for oy = 1:step
        A = zeros(nact);  A(ox:step:nact, oy:step:nact) = 1;  A = A .* lit;
        if strcmp(P.battery.matrix_sign, 'alternate')
            % checkerboard of +/- pokes over the grid sites (Dave 2026-09-10): the
            % multiplexed pattern is zero-mean, so the sensor's piston null puts
            % no shared pedestal into the frame and the halos cancel pairwise
            [rr2, cc2] = find(A);
            sgnA = 1 - 2*mod((rr2-ox)/step + (cc2-oy)/step, 2);
            A(sub2ind([nact nact], rr2, cc2)) = sgnA;
        end
        if ~any(A(:)), continue; end
        F = frames_(ZW, C.dmap(C.Abase + POKE*A), needS, needV, needP);  nstates = nstates + 1;
        [pr, pc] = find(A);
        for k = classes(:).'
            h = C.cmap(F, k) / POKE;                            % response per unit command
            % THE SENSOR CANNOT SEE PISTON: every reading is mean-referenced over
            % the pupil, so the multiplexed frame carries the pokes' shared
            % negative pedestal (-sum of blob volumes / mask area).  Remove it
            % before cutting (the median over the mask: the blobs cover ~10% of
            % the pixels), and give each column its OWN pedestal spread over the
            % whole mask below (v = column volume; a rank-one term in J'J).
            h = h - median(h(ZW.msk));
            for q = 1:numel(pr)
                r = pr(q);  c = pc(q);
                u0 = round(U(r,c));  v0 = round(V(r,c));
                rows = max(1, v0-hw_px):min(N, v0+hw_px);  cols = max(1, u0-hw_px):min(N, u0+hw_px);
                if numel(rows) < 2*hw_px+1 || numel(cols) < 2*hw_px+1, nclip = nclip + 1; end
                [CC, RR] = meshgrid(cols, rows);
                blk = h(rows, cols) * A(r,c);  blk(~ZW.msk(rows, cols)) = 0;   % per unit +command
                I{k}{end+1} = RR(:) + (CC(:)-1)*N;  Jc{k}{end+1} = col_of(r,c)*ones(numel(blk),1);  V3{k}{end+1} = blk(:);
                if k == classes(1), npoked = npoked + 1; end
                pk(k) = max(pk(k), max(abs(blk(:))));
            end
        end
    end
end
C.J = cell(1,6);  C.JtJ = cell(1,6);  C.est = cell(1,6);  C.kinfo = nan(6,3);
Am = nnz(ZW.msk);
for k = classes(:).'
    J = sparse(vertcat(I{k}{:}), vertcat(Jc{k}{:}), vertcat(V3{k}{:}), N*N, nlit);
    v = full(sum(J, 1)).';                                  % column volumes
    % full column = local window - v/Am over the mask (the piston null), so
    % J'J = Jl'Jl - v v'/Am  (rank one; the uniform command is nulled, as the
    % sensor nulls it -- the regularization carries that direction)
    JtJ = full(J.'*J) - (v*v.')/Am;  d = diag(JtJ);
    l2 = P.battery.matrix_lam * median(d(d > 0));
    Rf = chol(JtJ + l2*eye(nlit));
    C.J{k} = J;  C.JtJ{k} = JtJ;
    C.est{k} = @(h) est_matrix_(h, J, v, Am, Rf, ilit, nact, ZW.msk);
    % kinfo: [mean column peak (per unit command), fraction of columns clipped, column-norm spread]
    cn = sqrt(d);
    C.kinfo(k,:) = [mean(full(max(abs(J), [], 1))), nclip/max(npoked,1), std(cn)/mean(cn)];
end
C.matrix = struct('nstates',nstates, 'npoked',npoked, 'nlit',nlit, 'hw_px',hw_px, ...
                  'step',step, 'tmin',toc(t0)/60, 'nclip',nclip);
C.R = {};  C.stn = {};  C.kernel_site = [NaN NaN];
end

function a = est_matrix_(h, J, v, Am, Rf, ilit, nact, msk)
% actuator commands from a detector-space map by the measured matrix
% (columns = local window - v/Am over the mask; J'm = Jl'm - v (1'm)/Am)
h(~msk) = 0;
b = J.' * h(:) - v * (sum(h(msk))/Am);
x = Rf \ (Rf.' \ b);
a = zeros(nact);  a(ilit) = x;
end

function h = cmap_flat_(ZW, F, Fflat, k, uw)
% absolute class map on the flat (the record's form)
if nargin < 5, uw = []; end
switch k
    case 1, h = ZW.reconL(F.Ia);
    case 2, h = ZW.reconI(F.Ia);
    case 3, h = ZW.stepdiff(F.X, Fflat.X);
    case 4, h = ZW.reconV(F.Ip, F.Im);
    case 5, h = ZW.PD.P.height(F.FP);
    case 6, h = ZW.PD.PF.height(F.FF);
end
if ~isempty(uw) && k >= 3, h = uw(h); end
end

function h = cmap_base_(ZW, F, F0, plus, k, uw)
% class map DIFFERENTIAL to the working surface; class 2 = I+ on the base
if nargin < 6, uw = []; end
switch k
    case 1, h = ZW.reconL(F.Ia) - ZW.reconL(F0.Ia);
    case 2, h = ZW.reconI(F.Ia, [], plus) - ZW.reconI(F0.Ia, [], plus);
    case 3, h = ZW.stepdiff(F.X, F0.X);
    case 4, h = ZW.diffV(F.Ip, F.Im, F0.Ip, F0.Im);
    case 5, h = ZW.PD.P.diff(F.FP, F0.FP);
    case 6, h = ZW.PD.PF.diff(F.FF, F0.FF);
end
if ~isempty(uw) && k >= 3, h = uw(h); end
end

function gk = transfer_(P, ZW, C, cfg, classes, AMPM)
% modal transfer through each class's estimator on the probes cfg.PQ,
% measured on the calibration surface (flat: absolute; base: differential)
nm = size(cfg.PQ, 1);  gk = nan(nm, 6);
[ii, jj] = meshgrid((0.5:cfg.nact)/cfg.nact);
needS = any(classes == 3);  needV = any(classes == 4);  needP = [any(classes == 5) any(classes == 6)];
for m = 1:nm
    p = cfg.PQ(m,1);  q = cfg.PQ(m,2);
    Ak = cos(pi*p*ii).*cos(pi*q*jj);
    F = frames_(ZW, C.dmap(C.Abase + AMPM*Ak), needS, needV, needP);
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
    if strcmp(P.battery.calib_mode, 'matrix')
        dmg_say(rep, 'MATRIX calibration: %d multiplexed states (grid step %d), %d actuators poked once each, windows +/-%d px; J built in %.1f min; %d clipped windows\n', ...
            C.matrix.nstates, C.matrix.step, C.matrix.npoked, C.matrix.hw_px, C.matrix.tmin, C.matrix.nclip);
        dmg_say(rep, 'lit actuators %d; per class [mean column peak per unit command, clipped fraction, column-norm spread]:', nnz(lit));
    else
    dmg_say(rep, 'lit actuators %d; kernel per class [raw peak gain, ring min/peak, corr]:', nnz(lit));
    end
    cn = {'L', 'I', 'S', 'V', 'P', 'PF'};
    for k = kk, dmg_say(rep, '  %s [%.3f %.3f %.3f]', cn{k}, C.kinfo(k,:)); end
    dmg_say(rep, '\n');
    if ~strcmp(P.battery.calib_mode, 'matrix') && any(C.kinfo(kk,3) < 0.9)
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
    if strcmp(P.battery.calib_mode, 'matrix')
        % the measured matrix carries the response at every frequency (S10);
        % the kernel-era Wiener correction would only re-shape it (a reading
        % whose transfer exceeds 1 gets penalized), so matrix mode reports
        % RAW estimates: the cor columns and the ladder equal raw
        corrk = @(a, k) a;
        dmg_say(rep, 'matrix mode: no modal correction applied (cor columns and the ladder = raw estimates)\n');
    end
    % ---- the rows ------------------------------------------------------
    ROWS = rows_(P, cfg, lit, P.battery.rows);
    needS = any(KC == 3) || any(strcmp(RD, 'I+'));  needV = any(KC == 4);  needP = [any(KC == 5) any(KC == 6)];
    dmg_say(rep, 'rows (differential, actuator space).  g = gain, e = rms err over lit (pm), flr = rms of unpoked lit (pm), SNR = mean(poked)/flr\n');
    dmg_say(rep, '%-17s %-3s | %7s %8s %8s %7s | %7s %8s %8s %7s\n', 'row', 'rd', 'g_raw', 'e_raw', 'flr_raw', 'SNRraw', 'g_cor', 'e_cor', 'flr_cor', 'SNRcor');
    res = struct('row',{},'rd',{},'raw',{},'cor',{},'fold',{});
    plusb = [];  pinf = struct('frac', NaN);
    for r = 1:size(ROWS,1)
        base = ROWS{r,2};  dev = ROWS{r,3};
        if r == 1 || ~isequal(base, ROWS{r-1,2})
            F0 = frames_(ZW, C.dmap(base), needS, needV, needP);
            if any(strcmp(RD, 'I+')), [plusb, pinf] = ZW.priorS(F0.Ia, F0.Fr); end
        end
        F1 = frames_(ZW, C.dmap(base + dev), needS, needV, needP);
        if any(strcmp(RD, 'I+')) && any(base(:) ~= 0)
            % fold-crossing diagnostic: pixels whose side of the quarter-wave fold
            % differs between the base and base+change (the base's sign map is
            % wrong there for the one-frame I+ reading of the changed state)
            plus1 = ZW.priorS(F1.Ia, F1.Fr);
            dmg_say(rep, '  [%s] pixels that cross the fold under the change: %d of %d beyond-fold (%.2f%% of msk)\n', ...
                ROWS{r,1}, nnz(plus1 ~= plusb), nnz(plusb), 100*nnz(plus1 ~= plusb)/nnz(msk));
        end
        if any(base(:) ~= 0) && nnz(dev) == 1
            % map-space diagnostic: the same change on the FLAT, read the same way,
            % vs its differential on the working surface, over the changed
            % actuator's own window (the matrix mode's window; +/-4 actuators)
            Ff = frames_(ZW, C.dmap(dev), needS, needV, needP);  Fz = frames_(ZW, zeros(P.grid.N_G), needS, needV, needP);
            [rd_, cd_] = find(dev);
            if isfield(C, 'win'), u0 = round(C.win.U(rd_, cd_));  v0 = round(C.win.V(rd_, cd_));  hwp = C.win.hw_px;
            else, u0 = round(ZW.N_WF/2);  v0 = u0;  hwp = round(4*cfg.pitch/(S.dxd_mm*S.mag)); end
            rws = max(1, v0-hwp):min(ZW.N_WF, v0+hwp);  cls = max(1, u0-hwp):min(ZW.N_WF, u0+hwp);
            dmg_say(rep, '  [%s] differential map on the surface vs the same change on the flat, over the changed actuator''s window (rel rms diff / amplitude ratio):', ROWS{r,1});
            for k = 1:numel(RD)
                if strcmp(RD{k}, 'I+'), pz = false(ZW.N_WF); else, pz = []; end
                db = diff_(ZW, RD{k}, F1, F0, plusb, C.uw);  df = diff_(ZW, RD{k}, Ff, Fz, pz, C.uw);
                wb = db(rws, cls);  wf = df(rws, cls);
                dmg_say(rep, '  %s %.3f / %.3f', RD{k}, norm(wb(:)-wf(:))/max(norm(wf(:)), eps), (wf(:).'*wb(:))/max(wf(:).'*wf(:), eps));
            end
            dmg_say(rep, '\n');
        end
        for k = 1:numel(RD)
            araw = C.est{KC(k)}(diff_(ZW, RD{k}, F1, F0, plusb, C.uw));  acor = corrk(araw, KC(k));
            [gr, er, fr, sr] = score_(araw, dev, lit);  [gc, ec, fc, sc] = score_(acor, dev, lit);
            dmg_say(rep, '%-17s %-3s | %7.4f %8.0f %8s %7s | %7.4f %8.0f %8s %7s\n', ifelse_(k==1, ROWS{r,1}, ''), RD{k}, ...
                gr, er, fmt0_(fr), fmt2_(sr), gc, ec, fmt0_(fc), fmt2_(sc));
            if isempty(plusb), foldv = NaN; else, foldv = mean(plusb(msk)); end   % (ifelse_ evaluates both arms)
            res(end+1) = struct('row',ROWS{r,1}, 'rd',RD{k}, 'raw',[gr er fr sr], 'cor',[gc ec fc sc], 'fold',foldv); %#ok<AGROW>
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
        F0 = frames_(ZW, C.dmap(Ab), needS, needV, needP);
        if any(strcmp(RD, 'I+')), [plusb, pinf] = ZW.priorS(F0.Ia, F0.Fr); else, pinf = struct('frac', [NaN NaN]); end
        F1 = frames_(ZW, C.dmap(Ab + Asng), needS, needV, needP);
        g = nan(1,numel(RD));  fl = g;  sn = g;
        for k = 1:numel(RD)
            a = corrk(C.est{KC(k)}(diff_(ZW, RD{k}, F1, F0, plusb, C.uw)), KC(k));
            [g(k), ~, fl(k), sn(k)] = score_(a, Asng, lit);
        end
        dmg_say(rep, '%5.0f nm %6.4f %6.4f |', amp*1e6, pinf.frac(1), pinf.frac(end));
        for k = 1:numel(RD), dmg_say(rep, ' %7.4f %6.0f %7.1f|', g(k), fl(k), sn(k)); end
        dmg_say(rep, '\n');
        lad(end+1) = struct('amp',amp, 'fold',pinf.frac(end), 'fold0',pinf.frac(1), 'g',g, 'flr',fl, 'snr',sn); %#ok<AGROW>
    end
    % ---- capture range: the working surface a reading holds to 10% ----------
    % (Dave 2026-09-12: the devices do not operate at null) -- the largest
    % ladder rung with |g - 1| <= 0.1 and, when the next rung is beyond it,
    % the crossing interpolated in log(rms); 'beyond' when the last rung holds.
    if numel(lad) >= 2
        amps = [lad.amp];  G = reshape([lad.g], numel(RD), []).';
        dmg_say(rep, 'capture range to 10%% (largest base rms with the gain within 0.9..1.1; matrix on %s; log-interpolated crossing):', ...
            ifelse_(strcmp(C.surface, 'base'), sprintf('the %.0f nm surface', P.battery.base_rms*1e6), 'the flat'));
        for k = 1:numel(RD)
            ok = abs(G(:,k) - 1) <= 0.1;  r10 = NaN;  note = '';
            if ok(1)
                j = find(~ok, 1);
                if isempty(j)
                    r10 = amps(end);  note = '+ (holds at the last rung)';
                else
                    e0 = abs(G(j-1,k) - 1);  e1 = abs(G(j,k) - 1);
                    r10 = exp(log(amps(j-1)) + (0.1 - e0)/(e1 - e0) * (log(amps(j)) - log(amps(j-1))));
                end
            else
                note = ' (outside 10% at the first rung)';
            end
            dmg_say(rep, '  %s %.0f nm%s |', RD{k}, r10*1e6, note);
        end
        dmg_say(rep, '\n');
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
gk = nan(size(cfg.PQ,1), 6, K);
col = struct('nm',{},'phi',{},'absc',{},'dia_lamd',{},'dimple_px',{},'nmsk',{},'roundtrip',{},'bsur',{},'kinfo',{},'tmin',{});
RAW = cell(0, K);  lit = [];  ROWS = {};
for k = 1:K
    tk = tic;
    macos.load_rx(decks{k});  coat_oap_(P, S.G.bt, []);
    LAM = LAMS(k)*1e-6;  wl = macos.get_src_wvl();
    assert(abs(wl - LAM) < 1e-9*LAM, 'deck %s did not take Wavelen (%g)', decks{k}, wl);
    gopt = gauge_opt_(P, LAM);
    ZW = dmg_zwfs_gauge(S.iTO, S.iMASK, S.iDET, gopt);
    assert(ZW.gate.bsur < 1e-10, 'color %g nm: surrogate gate FAIL (%.2e)', LAMS(k), ZW.gate.bsur);
    ZW.PD = pdi_build_(P, LAM, S.iTO, S.iMASK, S.iDET);
    dimple_px = ZW.dia_mm*1e-3 / abs(macos.dx_at(S.iMASK));
    C = calibrate_(P, S, ZW, cfg, classes, lit);
    if k == 1
        lit = C.lit;
        ROWS = rows_(P, cfg, lit, P.color.rows);
        RAW = cell(size(ROWS,1), K);
        dmg_say(rep, 'lit actuators %d; hold-out (%d,%d)\n', nnz(lit), cfg.hold(1), cfg.hold(2));
        dmg_say(rep, '%6s | %7s %6s %8s %9s %7s %9s %9s | %s\n', 'nm', 'phi_m', '|c|', 'dia_l/D', 'dimplePx', 'msk_px', 'roundtrip', 'bsur', 'kernel peak per class L I S V P PF');
    end
    kk = find(~isnan(C.kinfo(:,1))).';
    dmg_say(rep, '%6g | %7.4f %6.3f %8.3f %9.2f %7d %9.1e %9.1e |', LAMS(k), gopt.PHI_M, abs(ZW.cc), gopt.DIA_LAMD, dimple_px, nnz(ZW.msk), ZW.gate.roundtrip, ZW.gate.bsur);
    for c = 1:6, if any(kk == c), dmg_say(rep, ' %.3f', C.kinfo(c,1)); else, dmg_say(rep, '     -'); end; end
    dmg_say(rep, '\n');
    gk(:,:,k) = transfer_(P, ZW, C, cfg, classes, P.battery.AMPM);
    needS = any(KC == 3) || any(strcmp(RD, 'I+'));  needV = any(KC == 4);  needP = [any(KC == 5) any(KC == 6)];
    plusb = [];
    for r = 1:size(ROWS,1)
        base = ROWS{r,2};  dev = ROWS{r,3};
        if r == 1 || ~isequal(base, ROWS{r-1,2})
            F0 = frames_(ZW, C.dmap(base), needS, needV, needP);
            if any(strcmp(RD, 'I+')), plusb = ZW.priorS(F0.Ia, F0.Fr); end
        end
        F1 = frames_(ZW, C.dmap(base + dev), needS, needV, needP);
        A = struct();
        for j = 1:numel(RD)
            A.(fld_(RD{j})) = C.est{KC(j)}(diff_(ZW, RD{j}, F1, F0, plusb, C.uw));
        end
        RAW{r,k} = A;
    end
    col(end+1) = struct('nm',LAMS(k), 'phi',gopt.PHI_M, 'absc',abs(ZW.cc), 'dia_lamd',gopt.DIA_LAMD, ...
        'dimple_px',dimple_px, 'nmsk',nnz(ZW.msk), 'roundtrip',ZW.gate.roundtrip, 'bsur',ZW.gate.bsur, ...
        'kinfo',C.kinfo, 'tmin',toc(tk)/60); %#ok<AGROW>
    fprintf('color %g nm done in %.1f min\n', LAMS(k), toc(tk)/60);
end
% ---- transfer tables per class + the combination's transfer ------------
cn = {'L', 'I', 'S', 'V', 'P', 'PF'};
Gc = nan(nnz(is1d), 6);  G1 = nan(nnz(is1d), 6, K);
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
macos.load_rx(S.deck);  coat_oap_(P, S.G.bt, []);       % back to the record color
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
RD = P.noise.readings(ismember(P.noise.readings, P.readings));    % the stage prices the readings run
assert(~isempty(RD), 'noise.readings has nothing in common with P.readings');
KC = cellfun(@class_, RD);  classes = unique(KC);
dmg_say(rep, '\n---- noise: DM %dx%d, readings %s ----\n', NACT, NACT, strjoin(RD, ' '));
dmg_say(rep, 'scenario: single act (%d,%d) %g nm differential on the %g nm rms working state; axis = photons per MEASUREMENT (one DM shape measured once; a reading''s frames share it: L/F/I/I+ 1 frame, S 4, V 2, P/PF K = %d%s; PDI throughput is printed at the bench stage -- divide by it for incident photons)\n', ...
    cfg.hold(1), cfg.hold(2), P.battery.dev_single*1e6, P.battery.base_rms*1e6, numel(P.pdi.thetas), ifelse_(strcmp(P.pdi.b2, 'state'), ' (+1 for P)', ''));
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
needV = any(KC == 4);  needP = [any(KC == 5) any(KC == 6)];
has_scalar = ~isfield(ZW, 'has_scalar') || ZW.has_scalar;
has_step   = ~isfield(ZW, 'has_step')   || ZW.has_step;
F0 = frames_(ZW, C.dmap(Ab), has_step, needV, needP);  F1 = frames_(ZW, C.dmap(Ab + Asng), has_step, needV, needP);
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
dmg_say(rep, '%10s |', 'N/meas');
for c = 1:nc, dmg_say(rep, ' %-24s|', sprintf('%s: sig flr g', lab{c})); end
dmg_say(rep, '\n');
NS = P.noise.nstates;  NR = P.noise.nreal;
sig = nan(numel(NS), nc);  flr = sig;  gm = sig;
for in = 1:numel(NS)
    n = NS(in);
    pk = zeros(NR, nc);  fl = zeros(NR, nc);
    for r = 1:NR
        rng(P.noise.seed + r + in*100);
        if has_scalar, Ia0 = noisy(F0.Ia, n);  Ia1 = noisy(F1.Ia, n); end
        if needV                                              % vector pair: N/2 per image
            Ip0 = noisy(F0.Ip, n/2);  Im0 = noisy(F0.Im, n/2);  Ip1 = noisy(F1.Ip, n/2);  Im1 = noisy(F1.Im, n/2);
        end
        if needP(1), nfP = size(F0.FP, 3);  FP0 = F0.FP;  FP1 = F1.FP;  for k = 1:nfP, FP0(:,:,k) = noisy(F0.FP(:,:,k), n/nfP);  FP1(:,:,k) = noisy(F1.FP(:,:,k), n/nfP); end; end
        if needP(2), nfF = size(F0.FF, 3);  FF0 = F0.FF;  FF1 = F1.FF;  for k = 1:nfF, FF0(:,:,k) = noisy(F0.FF(:,:,k), n/nfF);  FF1(:,:,k) = noisy(F1.FF(:,:,k), n/nfF); end; end
        Fr0q = F0.Fr;  Fr1q = F1.Fr;  Fr0f = F0.Fr;          % stepped at N/4 per frame; prior at N per frame
        if has_step
            for k = 1:4
                Fr0q(:,:,k) = noisy(F0.Fr(:,:,k), n/4);  Fr1q(:,:,k) = noisy(F1.Fr(:,:,k), n/4);
                Fr0f(:,:,k) = noisy(F0.Fr(:,:,k), n);
            end
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
                case 'V',  d = ZW.diffV(Ip1, Im1, Ip0, Im0);
                case 'P',  d = ZW.PD.P.diff(FP1, FP0);
                case 'PF', d = ZW.PD.PF.diff(FF1, FF0);
            end
            if ~isempty(C.uw) && wrapped_(rd), d = C.uw(d); end
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
        dmg_say(rep, '%-14s sigma ~ %.3g/sqrt(N) pm  ->  N(1 pm) ~ %.2e photons/measurement\n', lab{c}, cc, cc^2);
    end
end
dmg_say(rep, 'noise stage %.1f min\n', toc(t0)/60);
NO = struct('cols',{lab}, 'nstates',NS, 'nreal',NR, 'sig_pm',sig, 'flr_pm',flr, 'g',gm, 'n_1pm',n1pm);
end

% =====================================================================
%  stage: loop -- the closed-loop hold metric (Dave 2026-09-11)
% =====================================================================
function LO = stage_loop_(P, S, rep)
t0 = tic;
ZW = S.ZW;  cfg = P.dm(1);  NACT = cfg.nact;
RD = P.loop.readings;  assert(all(ismember(RD, P.readings)), 'loop.readings must be a subset of P.readings');
KC = cellfun(@class_, RD);  classes = unique(KC);
g = P.loop.g;  K = P.loop.K;  NPH = P.loop.nph(:).';  DR = P.loop.drifts;
dmg_say(rep, '\n---- loop: closed-loop hold, DM %dx%d, readings %s ----\n', NACT, NACT, strjoin(RD, ' '));
dmg_say(rep, 'loop: gain %.2f, %d cycles (steady state = last %d), set point = %s, reference frames %s, drift seed %d; each cycle = ONE measurement (the DM shape traced once, the reading''s frames with N photons), differential to the set point through the measured matrix\n', ...
    g, K, floor(K/2), P.loop.surface, P.loop.ref, P.loop.seed);
dmg_say(rep, 'dynamics: r(k+1) = (1 - gG) r(k) - gG e(k) + d(k+1); noise-only rms = sig_n sqrt(gG/(2-gG)); walk rms^2 = (sig_d^2 + g^2G^2 sig_n^2)/(gG(2-gG)); ramp lag = rate/(gG)\n');
% ---- calibration ON the set point --------------------------------------
SR = P.loop.start_rms(:).';
descent = ~isempty(SR) && any(SR > 0);  SR = SR(SR > 0);
Pl = P;  Pl.battery.calib_surface = ifelse_(strcmp(P.loop.surface, 'base'), 'base', 'flat');
if ischar(P.loop.unwrap) || isstring(P.loop.unwrap)
    assert(strcmpi(P.loop.unwrap, 'auto'), 'zwfs_run: loop.unwrap must be true, false or ''auto''');
    Pl.battery.unwrap = P.battery.unwrap || descent;   % a descent is what the unwrapper is for
else
    Pl.battery.unwrap = logical(P.loop.unwrap);
end
C = calibrate_(Pl, S, ZW, cfg, classes);
lit = C.lit;  A0 = C.Abase;
dmg_say(rep, 'calibration: %s on the %s (%s); lit actuators %d\n', ifelse_(strcmp(P.battery.calib_mode,'matrix'), 'measured response matrix', 'kernel'), ...
    Pl.battery.calib_surface, ifelse_(strcmp(Pl.battery.calib_surface,'base'), sprintf('%g nm rms, seed %d', P.battery.base_rms*1e6, P.battery.seed_base), 'flat DM'), nnz(lit));
if ~strcmp(P.battery.calib_mode, 'matrix')
    dmg_say(rep, 'NOTE: kernel calibration -- the loop stage is specified for the measured matrix (battery.calib_mode ''matrix''); the modal correction is NOT applied here\n');
end
% the set point's stepped frames (once): the I+ prior for every cycle
has_step = ~isfield(ZW, 'has_step') || ZW.has_step;
plusb = [];
if has_step
    F0ref = frames_(ZW, C.dmap(A0), true, false);
    if any(strcmp(RD, 'I+')), plusb = ZW.priorS(F0ref.Ia, F0ref.Fr); end
end
% the knobs this slice added (Dave 2026-09-13): the DM's initial figure, the
% on-surface re-calibration, the drift developing WITHIN a stepped scan, and
% the P/SRI's own reference-arm walk
RECL = P.loop.recal_list;  if isempty(RECL), RECL = P.loop.recal_every; end
dmg_say(rep, 'wrapped-differential UNWRAPPING (dm_gauge_lib/dmg_unwrap, least squares on the lit mask): %s for the readings whose differential is a wrapped phase difference (S, V, P, PF); L, F, I and I+ are untouched\n', ...
    ifelse_(Pl.battery.unwrap, 'ON', 'off'));
if P.loop.intra > 0
    dmg_say(rep, 'WITHIN-MEASUREMENT DRIFT: %.0f%% of each cycle''s drift increment develops ACROSS one measurement''s scan -- frame j of nf at (j-1)/(nf-1) of it.  The stepped readings (S, P, PF) capture their frames one at a time and pay for it; the single-frame (L, I+) and simultaneous (V) readings see one instant and do not\n', 100*P.loop.intra);
end
if P.pdi.ref_walk > 0
    dmg_say(rep, 'REFERENCE-ARM WALK: the P/SRI''s reference phase random-walks %.3g rad per cycle relative to the test arm (the non-common-path term).  Common-path readings (L, I+, S, V, P) do not have this arm and ignore it\n', P.pdi.ref_walk);
end
Astart = {};
if descent
    assert(numel(P.loop.reach) == 2, 'zwfs_run: loop.reach must name exactly two levels (the descent table''s columns)');
    assert(strcmp(P.loop.surface, 'base') && P.battery.base_rms > 0, ...
        'zwfs_run: a descent needs loop.surface ''base'' (the starting surface is the set point''s own field, rescaled)');
    dmg_say(rep, 'DESCENT: the DM starts at a surface of %s nm rms (the set point''s field, rescaled; %s nm WFE) and must reach the hold regime.  The response matrix is measured THERE, on each starting surface -- not on the set point the loop has yet to reach -- and then re-measured on the loop''s CURRENT surface every %s cycles (0 = never).  Reach levels %s nm\n', ...
        mat2str(SR*1e6), mat2str(2*SR*1e6), mat2str(RECL), mat2str(P.loop.reach*1e6));
    % ONE start matrix per starting surface, covering every class at once --
    % the calibration a bench would make in place, and 1/numel(RD) of the
    % cost of building it per reading.  Built ONE AT A TIME in the descent
    % block below and dropped after use: each carries a Cholesky factor and
    % a sparse J PER CLASS (~750 MB for five readings at NGRID 193), so
    % holding the whole ladder's worth at once costs GB and took the box
    % out with an OOM kill on 2026-09-13.
    a0r = sqrt(mean(A0(lit).^2));
    for q = 1:numel(SR), Astart{q} = A0 * (SR(q)/a0r); end %#ok<AGROW>
    % what the OPENING differential looks like to each reading: the wrapped
    % phase difference between the starting surface and the set point, and
    % how much of it the unwrapper can make consistent
    hpr = P.mask.S_CONV*P.LAM/(4*pi);
    F0d = frames_(ZW, C.dmap(A0), has_step, any(KC == 4), [any(KC == 5) any(KC == 6)]);
    dmg_say(rep, 'the OPENING differential (the starting surface against the set point), per reading: rms of the WRAPPED map, 2 pi residues inside the mask, largest wrapped gradient (rad per px), and the rms the unwrapper returns -- the truth is %s nm\n', mat2str(round((SR - a0r)*1e6)));
    for q = 1:numel(SR)
        F1d = frames_(ZW, C.dmap(Astart{q}), has_step, any(KC == 4), [any(KC == 5) any(KC == 6)]);
        dmg_say(rep, '  start %3.0f nm |', SR(q)*1e6);
        for j = 1:numel(RD)
            dw = diff_(ZW, RD{j}, F1d, F0d, plusb, []);
            if wrapped_(RD{j})
                [du, iu] = dmg_unwrap(dw/hpr, ZW.msk);
                dmg_say(rep, ' %s: %6.1f nm wrapped, %5d res, grad %.2f, %7.1f nm unwrapped |', ...
                    RD{j}, std(dw(ZW.msk))*1e6, iu.nres, iu.maxgrad, std(du(ZW.msk)*hpr)*1e6);
            else
                dmg_say(rep, ' %s: %6.1f nm (no wrap) |', RD{j}, std(dw(ZW.msk))*1e6);
            end
        end
        dmg_say(rep, '\n');
    end
end
% ---- the runs ---------------------------------------------------------------
res = struct('rd',{}, 'drift',{}, 'nph',{}, 'amp',{}, 'L',{}, 'start',{});
nrun = numel(RD) * (numel(P.loop.steps) + numel(NPH)*(P.loop.floor + numel(DR) + descent*numel(RECL)*numel(SR)));
dmg_say(rep, '%d loop runs of %d states each (%d traced states)\n', nrun, K+1, nrun*(K+1));
irun = 0;
for j = 1:numel(RD)
    rd = RD{j};  kc = KC(j);
    ins = mk_ins_(P, Pl, S, ZW, C, cfg, lit, plusb, rd, kc);
    base = mk_base_(P, A0, g, K, rd);
    % noiseless steps: time constant + dynamic range
    for amp = P.loop.steps
        o = base;  o.nph = Inf;  o.drift = struct('kind', 'step', 'amp', amp);
        L = dmg_loop(ins, o);  irun = irun + 1;
        res(end+1) = struct('rd',rd, 'drift','step', 'nph',Inf, 'amp',amp, 'L',L, 'start',0); %#ok<AGROW>
        fprintf('[loop %d/%d] %s step %g nm: rho %.3f, residual at K %.2f pm%s (%.1f min)\n', irun, nrun, rd, amp*1e6, L.rho, L.rms(L.k_end)*1e9, div_(L), toc(t0)/60);
    end
    for nph = NPH
        kinds = DR;  if P.loop.floor, kinds = [{'none'} DR]; end
        for kd = 1:numel(kinds)
            o = base;  o.nph = nph;
            switch kinds{kd}
                case 'none',    o.drift = struct('kind', 'none');  amp = 0;
                case 'walk',    o.drift = struct('kind', 'walk', 'sigma', P.loop.walk_sigma);  amp = P.loop.walk_sigma;
                case 'thermal', o.drift = struct('kind', 'thermal', 'rate', P.loop.thermal_rate);  amp = P.loop.thermal_rate;
                case 'cam',     o.drift = struct('kind', 'none');  o.cam.walk = P.loop.cam_walk;  amp = P.loop.cam_walk;
                otherwise,      error('zwfs_run: loop.drifts must be a subset of walk | thermal | cam');
            end
            L = dmg_loop(ins, o);  irun = irun + 1;
            res(end+1) = struct('rd',rd, 'drift',kinds{kd}, 'nph',nph, 'amp',amp, 'L',L, 'start',0); %#ok<AGROW>
            fprintf('[loop %d/%d] %s %s @ %.0e photons: ss %.2f pm, bias %.2f pm, sig_n %.2f pm%s (%.1f min)\n', irun, nrun, rd, kinds{kd}, nph, L.ss*1e9, L.bias*1e9, L.sig_n*1e9, div_(L), toc(t0)/60);
        end
    end
end
% ---- the descent ladder: from the DM's initial figure down to the hold
% regime.  The START loop is OUTSIDE the reading loop on purpose: the
% matrix measured on a starting surface covers every class at once, so it
% is built once per start and every reading uses it -- and then it is
% DROPPED before the next start is built, which is what keeps the ladder
% inside the box's memory.
if descent
    for q = 1:numel(SR)
        ts = tic;
        Pq = Pl;  Pq.battery.calib_surface = 'base';
        Cq = calib_matrix_(Pq, S, ZW, cfg, classes, lit, Astart{q});
        Cq.JtJ = {};  Cq.F0 = [];  Cq.Fflat = [];  Cq.Fa = [];   % keep only what est needs
        dmg_say(rep, 'descent: the starting matrix on the %.0f nm surface -- %d states, %.1f min\n', ...
            SR(q)*1e6, Cq.matrix.nstates, toc(ts)/60);
        for j = 1:numel(RD)
            rd = RD{j};  kc = KC(j);
            insd = mk_ins_(P, Pl, S, ZW, C, cfg, lit, plusb, rd, kc);
            insd.est = Cq.est{kc};                     % the matrix measured on THAT start
            base = mk_base_(P, A0, g, K, rd);
            for rc = RECL
                for nph = NPH
                    o = base;  o.nph = nph;  o.drift = struct('kind', 'none');
                    o.start_rms = SR(q);  o.recal_every = rc;
                    L = dmg_loop(insd, o);  irun = irun + 1;
                    res(end+1) = struct('rd',rd, 'drift','descent', 'nph',nph, 'amp',rc, 'L',L, 'start',SR(q)); %#ok<AGROW>
                    fprintf('[loop %d/%d] %s descent from %.0f nm (recal %d) @ %.0e photons: r(1) %.1f nm -> r(K) %.2f pm, %d recals%s (%.1f min)\n', ...
                        irun, nrun, rd, SR(q)*1e6, rc, nph, L.rms(1)*1e6, L.rms(L.k_end)*1e9, L.n_recal, div_(L), toc(t0)/60);
                end
            end
        end
        clear Cq insd                                  % before the next start is built
    end
end
% ---- tables -------------------------------------------------------------------
pm = @(x) x*1e9;
dmg_say(rep, '\nstep response (noiseless; a step of the given rms at cycle 1).  rho = fitted per-cycle contraction (1 - gG), tau = cycles to 1/e, k1e = first cycle below 1/e, r(K/2) and r(K) = residual (pm) at cycles %d and %d -- a residual that stops falling is the reading''s noiseless bias floor on this surface\n', floor(K/2), K);
dmg_say(rep, '%8s |', 'step');
for j = 1:numel(RD), dmg_say(rep, ' %-38s|', sprintf('%s: rho tau k1e r(K/2) r(K)', RD{j})); end
dmg_say(rep, '\n');
for amp = P.loop.steps
    dmg_say(rep, '%5.0f nm |', amp*1e6);
    for j = 1:numel(RD)
        i = find(strcmp({res.rd}, RD{j}) & strcmp({res.drift}, 'step') & [res.amp] == amp, 1);  L = res(i).L;
        if L.diverged, dmg_say(rep, ' %-38s|', sprintf('DIVERGED at cycle %d (%.0f pm)', L.k_end, pm(L.rms(L.k_end))));
        else, dmg_say(rep, ' %6.3f %6.1f %4s %10.3f %10.3f |', L.rho, L.tau, fmt0_(L.k_1e), pm(L.rms(floor(K/2))), pm(L.rms(end))); end
    end
    dmg_say(rep, '\n');
end
kinds = DR;  if P.loop.floor, kinds = [{'none'} DR]; end
for kd = 1:numel(kinds)
    switch kinds{kd}
        case 'none',    lab = 'noise only (drift 0): the G2 line, ss vs sig_n sqrt(g/(2-g))';
        case 'walk',    lab = sprintf('random walk, %g pm per actuator per cycle', P.loop.walk_sigma*1e9);
        case 'thermal', lab = sprintf('thermal ramp, %g pm rms per cycle (defocus + astigmatism)', P.loop.thermal_rate*1e9);
        case 'cam',     lab = sprintf('CAMERA drift, no DM drift: a per-pixel offset random-walking %g %s per cycle (%.0f%% of each step within the scan); zero-sum readings (S, P, PF) subtract a within-scan-constant offset exactly, single-frame readings (L, I+, V) imprint o_k - o_0 on the DM', ...
                            P.loop.cam_walk, ifelse_(strcmp(P.loop.cam_unit, 'rel'), 'x the mean photons per lit pixel per FRAME (the relative form: a bias / gain drift scaled to the signal)', 'electrons per pixel'), 100*P.loop.cam_intra);
    end
    dmg_say(rep, '\nhold error vs photons per cycle (= per measurement, one per cycle) -- %s.  ss = steady-state rms over lit (pm), bias = rms of the mean residual (noise averaged out), sig_n = single-shot estimate noise (pm), th = the theory line from sig_n\n', lab);
    dmg_say(rep, '%9s |', 'N/cycle');
    for j = 1:numel(RD), dmg_say(rep, ' %-30s|', sprintf('%s: ss bias sig_n th', RD{j})); end
    dmg_say(rep, '\n');
    for nph = NPH
        dmg_say(rep, '%9.1e |', nph);
        for j = 1:numel(RD)
            i = find(strcmp({res.rd}, RD{j}) & strcmp({res.drift}, kinds{kd}) & [res.nph] == nph, 1);  L = res(i).L;
            switch kinds{kd}
                case 'none',    th = L.theory.ss_noise;
                case 'walk',    th = L.theory.ss_walk;
                case 'thermal', th = hypot(L.theory.lag_ramp, L.theory.ss_noise);
                case 'cam',     th = L.theory.ss_noise;          % the immune reading's line; the excess is the camera's
            end
            if L.diverged, dmg_say(rep, ' %-30s|', sprintf('DIVERGED at cycle %d', L.k_end));
            else, dmg_say(rep, ' %7.2f %6.2f %6.2f %6.2f |', pm(L.ss), pm(L.bias), pm(L.sig_n), pm(th)); end
        end
        dmg_say(rep, '\n');
    end
end
% ---- the one number: photons per cycle to hold the spec ------------------------
spec = P.loop.hold_spec;
dmg_say(rep, '\nphotons per cycle to hold %.1f pm rms (log-log interpolation of ss over N; ''floor x'' = not reached: the noise-free residual sits at x pm):\n', spec*1e9);
dmg_say(rep, '%-8s |', 'drift');
for j = 1:numel(RD), dmg_say(rep, ' %-14s|', RD{j}); end
dmg_say(rep, '\n');
n_hold = nan(numel(kinds), numel(RD));
for kd = 1:numel(kinds)
    dmg_say(rep, '%-8s |', kinds{kd});
    for j = 1:numel(RD)
        ss = nan(1, numel(NPH));  bias = ss;
        for q = 1:numel(NPH)
            i = find(strcmp({res.rd}, RD{j}) & strcmp({res.drift}, kinds{kd}) & [res.nph] == NPH(q), 1);
            ss(q) = res(i).L.ss;  bias(q) = res(i).L.bias;
        end
        dv = false(1, numel(NPH));
        for q = 1:numel(NPH)
            i = find(strcmp({res.rd}, RD{j}) & strcmp({res.drift}, kinds{kd}) & [res.nph] == NPH(q), 1);  dv(q) = res(i).L.diverged;
        end
        if all(dv), txt = 'DIVERGED';  n_hold(kd,j) = NaN;
        else, [n_hold(kd,j), txt] = hold_photons_(NPH, ss, bias, spec); end
        dmg_say(rep, ' %-14s|', txt);
    end
    dmg_say(rep, '\n');
end
% ---- the descent table ---------------------------------------------------------
if descent
    dmg_say(rep, '\nDESCENT LADDER to the set point, the matrix measured at each start, unwrapping %s.  start = the DM''s initial surface rms (its WFE is twice that); r(1) = the residual the loop opens with (nm); k(10 nm) / k(3 pm) = the first cycle at or below those levels (- = never within %d cycles); r(K) = the residual at cycle %d (pm); rho = the fitted per-cycle contraction; recals = on-surface re-calibrations run\n', ...
        ifelse_(Pl.battery.unwrap, 'ON', 'OFF'), K, K);
    dmg_say(rep, '%-4s %6s %9s %6s | %9s %8s %8s %13s %6s %6s\n', 'rd', 'start', 'N/cycle', 'recal', 'r(1) nm', 'k(10nm)', 'k(3pm)', 'r(K) pm', 'rho', 'recals');
    for j = 1:numel(RD)
        for q = 1:numel(SR)
            for rc = RECL
                for nph = NPH
                    i = find(strcmp({res.rd}, RD{j}) & strcmp({res.drift}, 'descent') & [res.nph] == nph & [res.amp] == rc & [res.start] == SR(q), 1);
                    if isempty(i), continue; end
                    L = res(i).L;
                    if L.diverged
                        dmg_say(rep, '%-4s %5.0f %9.1e %6s | %9.2f %8s %8s %13s %6s %6d\n', RD{j}, SR(q)*1e6, nph, ifelse_(rc == 0, 'never', sprintf('%d', rc)), ...
                            L.rms(1)*1e6, '-', '-', sprintf('DIVERGED@%d', L.k_end), '-', L.n_recal);
                    else
                        dmg_say(rep, '%-4s %5.0f %9.1e %6s | %9.2f %8s %8s %13.3f %6.3f %6d\n', RD{j}, SR(q)*1e6, nph, ifelse_(rc == 0, 'never', sprintf('%d', rc)), ...
                            L.rms(1)*1e6, fmt0_(L.k_reach(1)), fmt0_(L.k_reach(2)), L.rms(L.k_end)*1e9, L.rho, L.n_recal);
                    end
                end
            end
        end
    end
    % the one number the capture slide needs: the largest start that got there
    dmg_say(rep, '\nthe largest start that CONVERGES (reaches %g pm within %d cycles), per reading and setting:\n%-4s |', P.loop.reach(2)*1e9, K, 'rd');
    for rc = RECL, for nph = NPH, dmg_say(rep, ' recal %-5s @ %-7.0e|', ifelse_(rc == 0, 'never', sprintf('%d', rc)), nph); end, end
    dmg_say(rep, '\n');
    for j = 1:numel(RD)
        dmg_say(rep, '%-4s |', RD{j});
        for rc = RECL
            for nph = NPH
                best = NaN;
                for q = 1:numel(SR)
                    i = find(strcmp({res.rd}, RD{j}) & strcmp({res.drift}, 'descent') & [res.nph] == nph & [res.amp] == rc & [res.start] == SR(q), 1);
                    if ~isempty(i) && ~res(i).L.diverged && ~isnan(res(i).L.k_reach(2)), best = SR(q); end
                end
                dmg_say(rep, ' %18s|', ifelse_(isnan(best), 'none', sprintf('%.0f nm', best*1e6)));
            end
        end
        dmg_say(rep, '\n');
    end
end
% ---- spectrum of the held residual at the highest photon level ------------------
dmg_say(rep, '\nspectrum of the held residual at %.0e photons per cycle: rms (pm) in [< 4, 4-12, > 12] cycles per aperture\n', NPH(end));
for kd = 1:numel(kinds)
    dmg_say(rep, '%-8s |', kinds{kd});
    for j = 1:numel(RD)
        i = find(strcmp({res.rd}, RD{j}) & strcmp({res.drift}, kinds{kd}) & [res.nph] == NPH(end), 1);
        dmg_say(rep, ' %s: %5.2f %5.2f %5.2f |', RD{j}, pm(res(i).L.spec.band));
    end
    dmg_say(rep, '\n');
end
dmg_say(rep, 'loop stage %.1f min (%d traced states)\n', toc(t0)/60, nrun*(K+1));
macos.set_elt_grid(S.iTO, macos.get_elt_grid_spacing(S.iTO), zeros(P.grid.N_G));
LO = struct('readings',{RD}, 'drifts',{kinds}, 'nph',NPH, 'steps',P.loop.steps, 'g',g, 'K',K, ...
    'surface',P.loop.surface, 'hold_spec',spec, 'n_hold',n_hold, 'lit',lit, 'A0',A0, 'res',res, ...
    'start_rms',SR, 'recal_list',RECL, 'intra',P.loop.intra, 'ref_walk',P.pdi.ref_walk, 'unwrap',Pl.battery.unwrap);
end

function t = div_(L)
if L.diverged, t = sprintf(' DIVERGED at cycle %d', L.k_end); else, t = ''; end
end

function F = meas_loop_(ZW, C, cmd, nd, varargin)
% one cycle's capture.  dmg_loop hands the within-scan drift in ACTUATOR
% space; the influence map is linear in the commands, so it converts here
% and frames_ adds the per-frame fraction of it.
aux = [];
if ~isempty(varargin) && ~isempty(varargin{1})
    aux = varargin{1};
    if isfield(aux, 'dstep') && ~isempty(aux.dstep) && any(aux.dstep(:) ~= 0)
        aux.dstep = C.dmap(aux.dstep);
    else
        aux.dstep = [];
    end
end
F = frames_(ZW, C.dmap(cmd), nd(1), nd(2), nd(3:4), aux);
end

function ins = mk_ins_(P, Pl, S, ZW, C, cfg, lit, plusb, rd, kc)
% the instrument one reading presents to dmg_loop.  Its estimator is the
% SET POINT's by default; the descent replaces it with the matrix measured
% on the starting surface.  Pl, not P, for the re-calibration: a mid-run
% re-calibration must use the SAME unwrap setting as the measurements it
% will be applied to.
nd = [strcmp(rd, 'S'), strcmp(rd, 'V'), strcmp(rd, 'P'), strcmp(rd, 'PF')];
ins = struct('lit', lit, 'npix', ZW.N_WF, 'cam_unit', P.loop.cam_unit, ...
    'measure', @(cmd, varargin) meas_loop_(ZW, C, cmd, nd, varargin{:}), ...
    'noisy',   @(F, nph, seed, varargin) noisy_frames_(ZW, F, nph, seed, rd, P.loop.cam_unit, varargin{:}), ...
    'diff',    @(F1, F0) diff_(ZW, rd, F1, F0, plusb, C.uw), ...
    'est',     C.est{kc}, ...
    'recal',   @(cmd) recal_loop_(Pl, S, ZW, cfg, kc, cmd, lit));
end

function base = mk_base_(P, A0, g, K, rd)
% the loop options every run of one reading shares
base = struct('A0', A0, 'g', g, 'K', K, 'seed', P.loop.seed, 'ref', P.loop.ref, 'rmax', P.loop.rmax, ...
              'cam', struct('walk', 0, 'intra', P.loop.cam_intra), ...
              'intra', P.loop.intra, 'ref_walk', ifelse_(strcmp(rd, 'PF'), P.pdi.ref_walk, 0), ...
              'reach', P.loop.reach);
if P.pdi.ref_seed > 0, base.ref_seed = P.pdi.ref_seed; end
end

function rc = recal_loop_(P, S, ZW, cfg, kc, cmd, lit)
% re-measure the response matrix ON the surface the loop is holding now --
% the instrument's own calibration, run in place (Dave 2026-09-13: how a
% device that cannot be taken to null gets from capture to hold)
Pr = P;  Pr.battery.calib_surface = 'base';
Cr = calib_matrix_(Pr, S, ZW, cfg, kc, lit, cmd);
rc = struct('est', Cr.est{kc}, 'nstates', Cr.matrix.nstates + 2);
end

function Fn = noisy_frames_(ZW, F, nph, seed, rd, unit, cam)
% photon noise on a captured state's frames: nph photons per MEASUREMENT (one
% DM shape measured once), split over the reading's frames (L / I+ one frame
% at nph; S four at nph/4; V two at nph/2; P / PF nf at nph/nf), the S5 model;
% the stepped retrieval X is redone from the noisy frames.  With cam (from
% dmg_loop's camera drift) the j-th of nf frames also gets the detector
% offset o + (j-1)/(nf-1) d, electrons per pixel, converted to frame units by
% that frame's photon scale (sum(I)/photons per frame) when unit is 'e', or
% as a FRACTION of the SCAN's mean photons per lit pixel per frame when unit
% is 'rel' (a bias drift scaled to the signal level: at >= 1e13 photons per
% measurement a pixel holds ~1e8 photons per frame and an electron-class
% offset is 1e-4 of the shot noise -- runs/pcam193).  The scale is ONE
% number per scan (the mean over the reading's frames), so the offset is the
% same electrons on every frame of a scan, as a camera bias is -- scaling by
% each frame's own mean (runs/pcam193r_perframe) gave frames of one scan
% different offsets and broke the zero-sum readings' exact immunity (S / P /
% PF read 32 / 34 / 130 pm floors instead of their noise-only values).  A
% frame is in the scan order the reading captures it (S: clear then the
% three depths; V: the two images at once, so both get o; P / PF: the steps
% in order).
Fn = F;
if ~isfinite(nph), return; end
if nargin < 6 || isempty(unit), unit = 'e'; end
if nargin < 7, cam = []; end
rs = RandStream('mt19937ar', 'Seed', seed);
shot = @(I, n) I .* (1 + randn(rs, size(I)) ./ sqrt(max(I / sum(I(:)) * n, 1)));
switch rd
    case 'S',  scanI = F.Fr;
    case 'V',  scanI = cat(3, F.Ia, F.Im);
    case 'P',  scanI = F.FP;
    case 'PF', scanI = F.FF;
    otherwise, scanI = F.Ia;
end
m3 = repmat(ZW.msk, [1 1 size(scanI, 3)]);
scan_mean = mean(scanI(m3));                                  % the scan's mean over the lit pixels, all frames
off = @(I, n, j, nf) offset_(I, n, j, nf, cam, unit, scan_mean);
switch rd
    case 'S'
        nf = size(F.Fr, 3);
        for k = 1:nf, Fn.Fr(:,:,k) = shot(F.Fr(:,:,k), nph/nf) + off(F.Fr(:,:,k), nph/nf, k, nf); end
        Fn.X = ZW.reconS(Fn.Fr);
    case 'V'
        Fn.Ip = shot(F.Ip, nph/2) + off(F.Ip, nph/2, 1, 1);  Fn.Im = shot(F.Im, nph/2) + off(F.Im, nph/2, 1, 1);
    case 'P'
        nf = size(F.FP, 3);  for k = 1:nf, Fn.FP(:,:,k) = shot(F.FP(:,:,k), nph/nf) + off(F.FP(:,:,k), nph/nf, k, nf); end
    case 'PF'
        nf = size(F.FF, 3);  for k = 1:nf, Fn.FF(:,:,k) = shot(F.FF(:,:,k), nph/nf) + off(F.FF(:,:,k), nph/nf, k, nf); end
    otherwise
        Fn.Ia = shot(F.Ia, nph) + off(F.Ia, nph, 1, 1);
end
end

function O = offset_(I, n, j, nf, cam, unit, scan_mean)
% the camera offset of frame j of nf in this frame's units: 'e' = electrons
% per pixel x (frame units per photon = sum(I)/n); 'rel' = a fraction of the
% SCAN's mean over the lit pixels (one scale for every frame of the scan)
if isempty(cam), O = 0;  return; end
w = 0;  if nf > 1, w = (j-1)/(nf-1); end
switch unit
    case 'e',   sc = sum(I(:))/n;
    case 'rel', sc = scan_mean;
    otherwise,  error('zwfs_run: loop.cam_unit must be ''e'' or ''rel''');
end
O = (cam.o + w*cam.d) * sc;
end

function [n, txt] = hold_photons_(NPH, ss, bias, spec)
% photons per cycle at which the steady-state rms crosses spec (log-log
% interpolation); NaN + a reason when the curve never crosses
n = NaN;
if all(ss > spec)
    txt = sprintf('floor %.1f', min(ss)*1e9);
    if ss(end) > spec && bias(end) < spec, txt = sprintf('> %.0e', NPH(end)); end
    return
end
if ss(1) <= spec, n = NPH(1);  txt = sprintf('< %.0e', NPH(1));  return; end
q = find(ss <= spec, 1);                                     % first point at or under spec
x = log(NPH(q-1:q));  y = log(ss(q-1:q));
n = exp(x(1) + (log(spec) - y(1)) * (x(2)-x(1)) / (y(2)-y(1)));
txt = sprintf('%.1e', n);
end

function stations_fig_(rd, P, rep, ZW, dmap, Ab, iTO, iMASK, iDET, msk, gopt, rz, cz)
%STATIONS_FIG_  The key signals along the train for one reading, two states.
%   Row 1 the flat DM, row 2 the working surface (P.battery.base_rms rms,
%   seed_base).  Columns: the mirror command (nm), the focal spot at the
%   mask (log10, the mask's footprint drawn), the mask (phase or
%   transmission), the reference wave |b| at the detector, two camera
%   frames, the recovered surface (nm) and its residual against the
%   engine's field (pm).  Frames from the gauge's own handles; the truth
%   from macos.complex_field at the detector.  Writes <tag>_stations_<rd>.png.
N = ZW.N_WF;  cvt = @(phi) P.mask.S_CONV*phi*P.LAM/(4*pi);   % rad of phase -> mm of surface
[mr, mc] = find(msk);  pw = 4;                                  % the pupil's box on the detector grid
pr = max(1, min(mr)-pw):min(N, max(mr)+pw);  pc = max(1, min(mc)-pw):min(N, max(mc)+pw);
states = {zeros(size(Ab)), Ab};  rown = {'flat DM', sprintf('%.0f nm rms working surface', P.battery.base_rms*1e6)};
E0 = ZW.E0;  th0 = angle(E0);
f = figure('Color', 'w', 'Position', [40 40 2000 640], 'Visible', 'off');
tl = tiledlayout(f, 2, 8, 'Padding', 'compact', 'TileSpacing', 'tight');
ink = [11 11 11]/255;
switch rd
    case 'S', fr_names = {'clear frame', 'first depth frame'};  mask_name = 'dimple phase, rad';
    case 'V', fr_names = {'camera A: +phi image', 'camera B: -phi image'};  mask_name = 'metasurface: +phi (and -phi), rad';
    case 'P', fr_names = {'first step frame', 'shutter frame (pinhole only)'};  mask_name = 'pinhole transmission';
end
resid_pm = [NaN NaN];
for r = 1:2
    A = states{r};  M = dmap(A);
    macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
    Et = macos.complex_field(iDET);                            % the state's unmasked field: the truth
    h_t = cvt(atan2(sin(angle(Et) - th0), cos(angle(Et) - th0)));
    If = abs(macos.complex_field(iMASK)).^2;  If = If(rz, cz)/max(If(:));
    b = ZW.bsur(Et);                                           % the reference wave (gate G2: == the engine's)
    switch rd
        case 'S'
            Fr = ZW.framesS(M);  Fr0 = ZW.framesS(zeros(size(M)));
            fr = {Fr(:,:,1), Fr(:,:,2)};
            h = ZW.stepdiff(ZW.reconS(Fr), ZW.reconS(Fr0));
            mk = gopt.PHI_M*ZW.D(rz, cz);
        case 'V'
            [Ip, Im] = ZW.frameV(M);  fr = {Ip, Im};
            h = ZW.reconV(Ip, Im);
            mk = gopt.PHI_M*ZW.D(rz, cz);   % the +phi image's dimple; the -phi image sees its negative
        case 'P'
            pd = ZW.PD.P;  Fr = pd.frames(M);  fr = {Fr(:,:,1), Fr(:,:,end)};
            h = pd.diff(Fr, pd.frames0);
            mk = abs(pd.D(rz, cz));
    end
    res = (h - h_t);  res = res - mean(res(msk));  resid_pm(r) = std(res(msk))*1e9;
    hm = h - mean(h(msk));  ht = h_t - mean(h_t(msk));
    panels = {M*1e6, 'mirror command, nm', 'lin'; ...
              If, 'focal spot at the mask, log', 'log'; ...
              mk, mask_name, 'lin'; ...
              abs(b)/max(abs(E0(:))), 'reference wave |b|', 'lin'; ...
              fr{1}/max(fr{1}(:)), fr_names{1}, 'lin'; ...
              fr{2}/max(fr{2}(:)), fr_names{2}, 'lin'; ...
              hm*1e6, 'recovered surface, nm', 'map'; ...
              res*1e9, sprintf('raw map minus the engine, pm: %.0f rms', resid_pm(r)), 'map'};
    for c = 1:8
        ax = nexttile(tl, (r-1)*8 + c);
        Z = panels{c,1};
        if strcmp(panels{c,3}, 'log'), Z = log10(max(Z, 1e-10)); end
        if c >= 4 && c <= 8, Z(~msk) = NaN;  Z = Z(pr, pc); end   % the pupil only
        imagesc(ax, Z);  axis(ax, 'image', 'off');
        if strcmp(panels{c,3}, 'map'), colormap(ax, 'parula'); else, colormap(ax, 'gray'); end
        if strcmp(panels{c,3}, 'log'), colormap(ax, 'parula'); caxis(ax, [-6 0]); end
        if c == 2   % the mask's footprint on the spot
            hold(ax, 'on');  t = linspace(0, 2*pi, 90);  rr = ZW.dia_mm*1e-3/abs(macos.dx_at(iMASK))/2;
            cx = ZW.ctr(1)-cz(1)+1;  cy = ZW.ctr(2)-rz(1)+1;
            plot(ax, cx + rr*cos(t), cy + rr*sin(t), 'w-', 'LineWidth', 1.2);
        end
        if strcmp(panels{c,3}, 'map') || c == 2 || c == 4, cb = colorbar(ax); cb.FontSize = 11; end
        title(ax, panels{c,2}, 'FontSize', 13, 'FontWeight', 'normal', 'Color', ink);
        if c == 1, ylabel(ax, rown{r}, 'FontSize', 14, 'FontWeight', 'bold', 'Visible', 'on'); end
    end
end
desc = struct('S', 'the stepped Zernike dimple, four frames', 'V', 'the polarized dimple, two frames at once', 'P', 'the stepped pinhole with a shutter frame');
title(tl, sprintf('%s: reading %s (%s), station by station: the mirror, the focus, the mask, the reference, two frames, the recovered surface, the residual', P.tag, rd, desc.(rd)), 'FontSize', 15, 'Color', ink, 'Interpreter', 'none');
out = sprintf('%s_stations_%s.png', P.tag, rd);
print(f, out, '-dpng', '-r130');  close(f);
dmg_say(rep, 'stations figure %s: residual vs the engine''s field on msk %.2f pm (flat), %.2f pm (%s)\n', out, resid_pm(1), resid_pm(2), rown{2});
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), dmap(zeros(size(Ab))));   % leave the DM flat
end
