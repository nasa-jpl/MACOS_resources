function out = ifo_l2_metric(tg_args, opts)
%IFO_L2_METRIC  Metric harness for the detector-leg (L2) redesign trade.
%   OUT = IFO_L2_METRIC(TG_ARGS, ...) evaluates one candidate rig built by
%   macos.design.twyman_green(TG_ARGS{:}) against the phase-1 DM-metrology
%   yardstick (PLAN_IFO_PUPIL_RELAY.md sec.4).  TG_ARGS is a cell of extra
%   name-value pairs forwarded to twyman_green ({} = the unmodified singlet
%   baseline).  The DM / grid options are supplied by the harness.
%
%   M1 (primary): physical-instrument vs-truth residual.  Poked and baseline
%   test arms are built exactly as the phase-1 driver does; the differential
%   phase comes from DIRECT complex fields at the detector,
%       phi = angle(Ed_pk .* conj(Ed_b)),   h = +phi*LAM/(4*pi),
%   which equals the 14-frame PSI differential to 3e-5 pm (phase-1 finding;
%   no PSI frames needed -> ~10x cheaper).  The recovered surface is compared
%   to the PROPER truth through the EMPIRICAL per-ray pupil map (orientation
%   scan + similarity-only refine + spline truth resample) -- the phase-1 5b
%   machinery lifted verbatim.  SIGN CONVENTIONS ARE LOAD-BEARING: the engine
%   field phase ADVANCES as optical path SHORTENS; do not "fix" any sign.
%
%   M2: pupil-map report -- magnification, anamorphism, rotation, nonlinear
%   distortion from the per-ray affine fit.
%
%   Guards (all must pass for a candidate to count):
%     rays   no additional ray loss through the train (<=0.1% lost at DET);
%     spot   focal-mask rms transverse spot <= 1 um on the BASELINE arm
%            (the poked arm legitimately spreads ~2*slope*F2 -- that is the
%            signal, not a defect);
%     conj   DM-tilt test: tilt at a pupil => no chief translation at its
%            conjugate (Stage-C pattern from example_bench_layout);
%     null   zero-grid differential repeat null < 1e-8 rad rms (verifies the
%            common-path cancellation the differential protocol relies on).
%   The inter-arm static phase (test-vs-ref at the detector) is REPORTED,
%   not gated -- the tail is common-path so architecture changes should
%   leave it unchanged; a jump flags a broken emit.
%
%   Options:
%     'poke_nm'  |command| bound, nm         (default 50)
%     'seed'     PROPER DM RNG seed          (default 7)
%     'pattern'  'checker' | 'random'        (default 'checker')
%     'workdir'  scratch dir for grid files + emitted Rx (default: this dir)
%     'tilt_rad' DM tilt for the conjugate guard (default 1e-5)
%     'verbose'  print the report            (default true)
%
%   OUT fields: m1_resid_nm, m1_corr, truth_rms_nm, rec_rms_nm, axes,
%   map (mag/anam_pct/rot_mrad/nl_rms_mm/nl_pct_beam/r_beam_mm),
%   guards (.rays .spot .conj .null each with .pass + numbers,
%   .static_rad_rms report), pass (all guards), reg (refine params),
%   dbg (h, msk, ht, xy_to, xy_d, nl, Aaf, dxp, ax, Mdm) for the
%   mechanism analysis, params (echo).
%
%   Gate 0: ifo_l2_metric({}) must reproduce the phase-1 committed numbers
%   (resid 6.76 +/- 0.1 nm, mag 0.8101, rot 180.000 deg, nonlinear
%   0.0205 mm rms) -- see run_gate0.m.

arguments
    tg_args cell = {}
    opts.poke_nm  (1,1) double = 50
    opts.seed     (1,1) double = 7
    opts.pattern  (1,:) char   = 'checker'
    opts.nact     (1,1) double = 16
    opts.spacing  (1,1) double = 3.5
    opts.n_g      (1,1) double = 256
    opts.dx_g     (1,1) double = 0.35
    opts.workdir  (1,:) char   = ''
    opts.tilt_rad (1,1) double = 1e-5
    opts.verbose  (1,1) logical = true
    opts.plane    (1,:) char {mustBeMember(opts.plane, {'det','rc'})} = 'det'
end
% 'plane','rc' measures the differential phase AT THE RECOMB PLANE (before
% the tail) through the same machinery -- the tail-immune gauge.  The
% det-vs-rc difference isolates the detector-leg retrace term (mechanism
% analysis, PLAN sec.5).  Guards always run on the detector-leg instrument.

LAM = 6.328e-4;                                    % mm (rig wavelength)
if isempty(which('macos.init'))
    addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/src'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
here = fileparts(mfilename('fullpath'));
if isempty(opts.workdir), opts.workdir = here; end
oldwd = cd(opts.workdir);                          % GridFile= resolves from cwd
restore = onCleanup(@() cd(oldwd));

persistent booted
if isempty(booted), macos.init(256); booted = true; end

% ---- 1. DM surface from PROPER -> GridFile (cached per parameter set) ---
PYBIN = fullfile(getenv('HOME'), 'dev/MACOS_resources/pymacos/.venv/bin/python');
gen   = fullfile(here, '..', 'dm_map_gen.py');
% NOTE: the engine's GridFile field truncates at 24 characters (GridInit
% then reports "does not exist" and traces a FLAT surface) -- keep these
% names short.  The assert after the field difference below is the backstop.
gf = sprintf('dm_p%g_s%d_%c.txt', opts.poke_nm, opts.seed, opts.pattern(1));
if ~isfile(gf)
    cmdline = sprintf('%s %s %s %d %g %d %g %g %d %s', PYBIN, gen, gf, ...
        opts.n_g, opts.dx_g, opts.nact, opts.spacing, opts.poke_nm, ...
        opts.seed, opts.pattern);
    [st, msg] = system(cmdline);
    assert(st == 0, 'PROPER DM generation failed: %s', msg);
end
zf = 'dm_zero.txt';
if ~isfile(zf), macos.write_grid_file(zf, zeros(opts.n_g)); end
Mdm = macos.read_grid_file(gf);                    % engine convention, mm

% ---- 2. build + emit the candidate rig ----------------------------------
G  = macos.design.twyman_green('to_grid_file', gf, 'to_grid_n', opts.n_g, ...
        'to_grid_dx', opts.dx_g, tg_args{:});
G0 = macos.design.twyman_green('to_grid_file', zf, 'to_grid_n', opts.n_g, ...
        'to_grid_dx', opts.dx_g, tg_args{:});
rx_test = fullfile(opts.workdir, 'l2m_test_arm.in');  G.bt.emit(rx_test);
rx_base = fullfile(opts.workdir, 'l2m_base_arm.in');  G0.bt.emit(rx_base);
rx_ref  = fullfile(opts.workdir, 'l2m_ref_arm.in');   G.br.emit(rx_ref);
T = G.T;  R = G.R;
if strcmp(opts.plane, 'rc'), iMEAS = T.iRC;  rMEAS = R.iRC;
else,                        iMEAS = T.iDET; rMEAS = R.iDET; end

% ---- 3. direct differential fields (M1 numerator) -----------------------
macos.load_rx(rx_test);  Ed_pk = macos.complex_field(iMEAS);
macos.load_rx(rx_base);  Ed_b  = macos.complex_field(iMEAS);
msk = abs(Ed_pk) > 0.1*max(abs(Ed_pk(:)));
phi = angle(Ed_pk .* conj(Ed_b));                  % differential, wrap-free
assert(std(phi(msk)) > 1e-6, ['ifo_l2_metric: poked and baseline fields ' ...
    'are identical -- GridFile not loaded (silent flat-DM fallback)?']);
h   = phi * LAM/(4*pi);                            % surface height, mm (+ sign:
                                                   %  engine phase advances as
                                                   %  path shortens)

% null guard: repeat the baseline field, differential must vanish
macos.load_rx(rx_base);  Ed_b2 = macos.complex_field(iMEAS);
null_rms = std(angle(Ed_b2(msk) .* conj(Ed_b(msk))));

% inter-arm static (report only; complex-domain centering avoids the +/-pi
% branch artifact -- never std() wrapped angles directly)
macos.load_rx(rx_ref);   Er = macos.complex_field(rMEAS);
ang = angle(Ed_b .* conj(Er));
ang = angle(exp(1i*(ang - median(ang(msk)))));
static_rms = std(ang(msk));

% ---- 4. per-ray pupil map on the poked test arm (M2 + truth resample) ---
macos.load_rx(rx_test);
s  = macos.trace(T.iTO);   ito  = macos.get_ray_info(s.nRays);
s2 = macos.trace(iMEAS);   idet = macos.get_ray_info(s2.nRays);
okr = ito.ok_trace(:) & ito.ok_pass(:) & idet.ok_trace(:) & idet.ok_pass(:);
dxp = macos.dx_at(iMEAS, 'mm');
% DM-plane coords in the grid frame the Rx declares (pData=vpt,
% xData=perp(psi), yData=psi x xData) -- the frame Mdm lives in
psi1 = macos.get_elt_psi(T.iTO);  vpt1 = macos.get_elt_vpt(T.iTO);
u1 = macos.design.Bench.perp(psi1);  v1 = cross(psi1, u1);
xy_to = [u1.'; v1.'] * (ito.pos - vpt1);
% measurement-plane coords about the chief
psi2 = macos.get_elt_psi(iMEAS);
u2 = macos.design.Bench.perp(psi2);  v2 = cross(psi2, u2);
xy_d = [u2.'; v2.'] * (idet.pos - idet.pos(:,1));
xy_to = xy_to(:,okr);  xy_d = xy_d(:,okr);
% beam radius at the DM (physical scale)
d = xy_to - mean(xy_to, 2);
r_beam = max(sqrt(sum(d.^2,1)));
% affine part = the classical distortion numbers
Aaf = [xy_d.' ones(nnz(okr),1)] \ xy_to.';
Lm  = Aaf(1:2,:).';
[Us,Ss,Vs] = svd(Lm);
smag = diag(Ss);  Rrot = Us*Vs.';
xy_fit = Lm*xy_d + Aaf(3,:).';
nl = xy_to - xy_fit;
nl_rms = sqrt(mean(sum(nl.^2,1)));
Fx = scatteredInterpolant(xy_d(1,:).', xy_d(2,:).', xy_to(1,:).', 'linear', 'linear');
Fy = scatteredInterpolant(xy_d(1,:).', xy_d(2,:).', xy_to(2,:).', 'linear', 'linear');

% ---- 5. orientation scan + similarity refine -> M1 ----------------------
N = size(h,1);
[colg, rowg] = meshgrid(1:N, 1:N);
cx = sum(colg(msk))/nnz(msk);  cy = sum(rowg(msk))/nnz(msk);
a1 = (colg - cx)*dxp;  a2 = (rowg - cy)*dxp;
c_d = mean(xy_d, 2);
ax = ((1:opts.n_g) - (opts.n_g+1)/2)*opts.dx_g;
hm = h - mean(h(msk));
cands = {a1,a2,'x=+col,y=+row'; a1,-a2,'x=+col,y=-row'; ...
         -a1,a2,'x=-col,y=+row'; -a1,-a2,'x=-col,y=-row'; ...
         a2,a1,'x=+row,y=+col'; a2,-a1,'x=+row,y=-col'; ...
         -a2,a1,'x=-row,y=+col'; -a2,-a1,'x=-row,y=-col'};
best = struct('c', -inf, 'i', 1);
for c = 1:size(cands,1)
    Xt = Fx(cands{c,1}+c_d(1), cands{c,2}+c_d(2));
    Yt = Fy(cands{c,1}+c_d(1), cands{c,2}+c_d(2));
    ht = interpn(ax, ax, Mdm, Xt, Yt, 'linear', 0);
    ht = ht - mean(ht(msk));
    cc = corrcoef(hm(msk), ht(msk));  cc = cc(1,2);
    if isfinite(cc) && cc > best.c
        best = struct('c',cc, 'ht',ht, 'name',cands{c,3}, 'i',c);
    end
end
assert(isfield(best, 'ht'), ['ifo_l2_metric: orientation scan found no ' ...
    'finite correlation -- truth resample degenerate (ray map / mask?)']);
% similarity-only refine (offset + rot + scale) -- deliberately NOT extended
% with higher-order terms: the retrace structure must stay in the residual
A1 = cands{best.i,1};  A2 = cands{best.i,2};
regc = @(p) -reg_corr(p, A1, A2, c_d, Fx, Fy, ax, Mdm, hm, msk);
p_reg = fminsearch(regc, [0 0 0 0], optimset('TolX',1e-7,'TolFun',1e-10,'Display','off'));
[c_ref, ht_ref, Xt_ref, Yt_ref] = reg_corr(p_reg, A1, A2, c_d, Fx, Fy, ax, Mdm, hm, msk);
corr_scan = best.c;
if c_ref > best.c
    best.c = c_ref;  best.ht = ht_ref;
else
    p_reg = zeros(1,4);
    [~, ~, Xt_ref, Yt_ref] = reg_corr(p_reg, A1, A2, c_d, Fx, Fy, ax, Mdm, hm, msk);
end
res = hm(msk) - best.ht(msk);

% ---- 6. guards ----------------------------------------------------------
guards = struct();
% rays: loss fraction at the detector, both arms
nlost_test = s2.nRays - nnz(idet.ok_trace(:) & idet.ok_pass(:));
macos.load_rx(rx_base);
sb = macos.trace(T.iDET);  ib = macos.get_ray_info(sb.nRays);
nlost_base = sb.nRays - nnz(ib.ok_trace(:) & ib.ok_pass(:));
guards.rays = struct('n', sb.nRays, 'lost_test', nlost_test, ...
    'lost_base', nlost_base, ...
    'pass', max(nlost_test, nlost_base) <= 0.001*sb.nRays);
% spot: focal-mask rms transverse spread, BASELINE arm
macos.load_rx(rx_base);
sm = macos.trace(T.iMASK);  im = macos.get_ray_info(sm.nRays);
okm = im.ok_trace(:) & im.ok_pass(:);
pch = im.pos(:,1);  dch = im.dir(:,1)/norm(im.dir(:,1));
dm_ = im.pos(:,okm) - pch;  dm_ = dm_ - dch*(dch.'*dm_);
spot_um = 1e3*sqrt(mean(sum(dm_.^2,1)));
guards.spot = struct('rms_um', spot_um, 'pass', spot_um <= 1.0);
% conjugate: tilt the DM, chief must not translate at the detector
% (Stage-C pattern; the tilted focal spot at the MASK is the sanity check
% that the tilt took effect: expect ~2*tilt*F2 there)
t = opts.tilt_rad;
K = [0 -psi1(3) psi1(2); psi1(3) 0 -psi1(1); -psi1(2) psi1(1) 0]; %#ok<NASGU>
Ku = [0 -u1(3) u1(2); u1(3) 0 -u1(1); -u1(2) u1(1) 0];
Rt = eye(3) + sin(t)*Ku + (1-cos(t))*(Ku*Ku);      % Rodrigues about u1
vdet0 = macos.get_elt_vpt(T.iDET);
cdet  = macos.get_elt_psi(T.iDET);  cdet = cdet/norm(cdet);
smk0 = macos.trace(T.iMASK);  imk0 = macos.get_ray_info(smk0.nRays);
mask_p0 = imk0.pos(:,1);
macos.load_rx(rx_base);
macos.set_elt_psi(T.iTO, Rt*psi1);
smk = macos.trace(T.iMASK);  imk = macos.get_ray_info(smk.nRays);
mask_shift_um = 1e3*norm(imk.pos(:,1) - mask_p0);
sdt = macos.trace(T.iDET);  idt = macos.get_ray_info(sdt.nRays);
dd = idt.pos(:,1) - vdet0;  dd = dd - cdet*(cdet.'*dd);
conj_shift_um = 1e3*norm(dd);
guards.conj = struct('tilt_rad', t, 'shift_um', conj_shift_um, ...
    'mask_shift_um', mask_shift_um, 'pass', conj_shift_um <= 1.0);
% null
guards.null = struct('rad_rms', null_rms, 'pass', null_rms < 1e-8);
guards.static_rad_rms = static_rms;
gpass = guards.rays.pass && guards.spot.pass && guards.conj.pass && guards.null.pass;

% ---- 7. pack + report ---------------------------------------------------
out = struct( ...
    'm1_resid_nm', 1e6*std(res), 'm1_corr', best.c, ...
    'truth_rms_nm', 1e6*std(best.ht(msk)), 'rec_rms_nm', 1e6*std(hm(msk)), ...
    'axes', best.name, 'corr_scan', corr_scan, ...
    'map', struct('mag', sqrt(abs(det(Lm))), ...
        'anam_pct', 100*(smag(1)/smag(2)-1), ...
        'rot_mrad', 1e3*atan2(Rrot(2,1),Rrot(1,1)), ...
        'nl_rms_mm', nl_rms, 'nl_pct_beam', 100*nl_rms/r_beam, ...
        'r_beam_mm', r_beam), ...
    'guards', guards, 'pass', gpass, ...
    'reg', struct('p', p_reg, 'offset_mm', p_reg(1:2), ...
        'rot_mrad', 1e3*p_reg(3), 'scale', expm1(p_reg(4))), ...
    'params', opts, ...
    'dbg', struct('h', h, 'hm', hm, 'msk', msk, 'ht', best.ht, ...
        'Xt', Xt_ref, 'Yt', Yt_ref, 'xy_to', xy_to, ...
        'xy_d', xy_d, 'nl', nl, 'Aaf', Aaf, 'dxp', dxp, 'ax', ax, ...
        'Mdm', Mdm, 'okr', okr, 'r_beam', r_beam));

if opts.verbose
    fprintf('\n=== ifo_l2_metric (poke %g nm, %s, seed %d) ===\n', ...
        opts.poke_nm, opts.pattern, opts.seed);
    fprintf('  M1: residual %.3f nm rms  (corr %.6f; scan %.6f; truth %.2f, recovered %.2f nm rms)\n', ...
        out.m1_resid_nm, out.m1_corr, corr_scan, out.truth_rms_nm, out.rec_rms_nm);
    fprintf('  M2: mag %.4f, anam %.3f%%, rot %.3f mrad, nonlinear %.4f mm rms (%.2f%% of beam)\n', ...
        out.map.mag, out.map.anam_pct, out.map.rot_mrad, out.map.nl_rms_mm, out.map.nl_pct_beam);
    fprintf('  guards: rays %d/%d lost [%s], spot %.3f um [%s], conj %.3f um @%g rad (mask sanity %.1f um) [%s], null %.2e rad [%s]\n', ...
        max(guards.rays.lost_test, guards.rays.lost_base), guards.rays.n, pf(guards.rays.pass), ...
        guards.spot.rms_um, pf(guards.spot.pass), ...
        guards.conj.shift_um, t, guards.conj.mask_shift_um, pf(guards.conj.pass), ...
        guards.null.rad_rms, pf(guards.null.pass));
    fprintf('  inter-arm static (report): %.3e rad rms;  overall %s\n', ...
        static_rms, pf(gpass));
end
end

% -------------------------------------------------------------------------
function s = pf(p)
    if p, s = 'PASS'; else, s = 'FAIL'; end
end

function [c, ht, Xt, Yt] = reg_corr(p, A1, A2, c_d, Fx, Fy, ax, Mdm, hm, msk)
%REG_CORR  Truth-vs-recovery correlation under a similarity adjustment
%   P = [dx dy rot log_scale] on the pixel->detector coordinates; truth
%   resampled through the empirical ray map with SPLINE interpolation
%   (bilinear costs 100s of pm at actuator-scale structure).  Lifted from
%   example_bench_ifo_dm.m section 5b -- keep the two in sync.
    s = exp(p(4));  ct = cos(p(3));  st = sin(p(3));
    X = s*(ct*A1 - st*A2) + c_d(1) + p(1);
    Y = s*(st*A1 + ct*A2) + c_d(2) + p(2);
    Xt = Fx(X, Y);  Yt = Fy(X, Y);
    ht = interpn(ax, ax, Mdm, Xt, Yt, 'spline', 0);
    ht = ht - mean(ht(msk));
    cc = corrcoef(hm(msk), ht(msk));
    c = cc(1,2);
end
