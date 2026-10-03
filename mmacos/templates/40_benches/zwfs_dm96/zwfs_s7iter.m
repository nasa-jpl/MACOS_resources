function out = zwfs_s7iter()
%ZWFS_S7ITER  Stage 7: the sensor MODEL CORRECTION + the ITERATED-REFERENCE
%   exact reading (literature import #1, macos/REPORT_zwfs_lit_scan.md).
%
%   MODEL CORRECTION (found building the reconstructor, 2026-09-09): the
%   twyman_green 'mask_prop','nf' sandwich emitted the exit reference
%   sphere with zElt/Kr = 0.6*D_MASK_FL (23.86 mm) against the entrance
%   sphere's 352.7 mm.  The engine's SPH2PL (NF1) multiplies the focal
%   field by exp(i*S*(m^2+n^2)), S ~ (Z2-Z1)*Z1/Z2, and PL2SPH (NF2) is a
%   plain shifted FFT, so the unmasked round trip was a Fresnel DEFOCUS of
%   the pupil by z_eff = Z1*(Z1-Z2)/Z2 = 4.86 m (entrance-sphere scale)
%   instead of the identity the ctb_dcr.in precedent gets with EQUAL
%   radii.  Block A puts the legacy numbers on record (round trip 0.159,
%   29% rms amplitude modulation under the 30 nm state, the ringed poke
%   kernel, the S2 tie-in 0.903/132 pm); the corrected emission ('nf' =
%   symmetric; 'nf_legacy' = the old one) is gated in Block B.  The
%   S1-S6 ZWFS record stands as the LEGACY-model record.
%
%   READING #3 (dmg_zwfs_gauge measI/reconI): per-pixel exact solve
%   (Ruane 2020 eq 36-37 / N'Diaye 2013 eq 6-7) with the reference wave b
%   re-propagated from the estimate through the FFT surrogate of the mask
%   model (gated against the engine's Eb), NITER iterations; ONE masked
%   frame, A = |E0| (exact on the now-conjugate pupil).  Its one physical
%   limit is the quarter-wave sensor's branch: (phi - Theta) in [-pi, 0]
%   per pixel; on a 30 nm working state ~8% of pixels sit beyond the fold.
%   'I+' resolves them with a ONE-TIME stepped retrieval of the base as a
%   branch prior (the base costs 4 frames once; every differential frame
%   is still one), REFINED by re-solving that retrieval with the iterated
%   reading's own |b|^2 (dmg_zwfs_gauge priorS; two passes reach the true
%   branch, where the plain stepped prior misses ~3% of pixels).  Readings side by side: L (frozen linear, 1 frame),
%   F (exact, frozen b, 1 frame), I (exact, iterated b, 1 frame), I+ (I +
%   base branch prior), S (phase-stepped, 4 frames).
%
%   BATTERY (Block C, 96x96 @ 1 mm and 48x48 @ 2 mm): per reading its own
%   measured kernel + modal transfer (the S3/S6 (p,0) rows) + separable
%   Wiener; the S6 five rows RAW and corrected; the S4 break-scale ladder.
%   Success (lit-scan spec): grid-on-base SNR >= 5 from ONE frame; the
%   flat hold-out RAW gain within 3% of 1 without the Wiener.
%   Run:  cd <this dir>;  matlab -batch "zwfs_s7iter; exit(0)"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
addpath(fullfile(exdir, '..', 'dm_gauge_lib'));
t_all = tic;
rep = fopen('zwfs_s7iter_report.txt', 'w');

s = 96/56;  LAM = 6.328e-4;
MODEL = 1024;  NGRID = 193;  N_G = 384;  DX_G = 0.28;
AOI = 7;  D_BS_TO = 700;  R_BEAM = s*30;  F2 = s*250;
T_FL_F = 42.5325;  T_FL_Kc = -2.58764;  T_DMF = 39.7694;  T_TRIM = -1.2473;
MASK_TRIM = -5.582;
N_FS = 1.45702;  ETCH_MM = 346.2e-6;
PHI_M = 2*pi*(N_FS-1)*ETCH_MM/LAM;
DIA_LAMD = 2.0;  S_CONV = -1;
PHIS = [pi/2, pi, 3*pi/2];
BETA = 0.1;  NITER = 5;
LADDER = [30 40 50 60 120 240 480]*1e-6;       % base rms, mm (40/50 locate the exact readings' cliff)
POKE = 20e-6;  AMPM = 10e-6;  AMPG = 1e-6;
nrm = @(x) norm(x(:));  wrap = @(p) atan2(sin(p), cos(p));
gopt = struct('LAM',LAM, 'F2',F2, 'R_BEAM',R_BEAM, 'DIA_LAMD',DIA_LAMD, ...
              'PHI_M',PHI_M, 'PHIS',PHIS, 'S_CONV',S_CONV, 'NITER',NITER);
RD = {'L', 'F', 'I', 'I+', 'S'};               % the readings, in report order
KC = [1 2 2 2 3];                              % kernel/transfer class: 1=L 2=I 3=S

dmg_say(rep, '=== ZWFS S7: model correction (symmetric NF sandwich) + iterated-reference exact reading (pm) ===\n');
dmg_say(rep, 'model %d, NGRID %d, spot %.2f lam/D, NITER %d, beta %.2f\n', MODEL, NGRID, DIA_LAMD, NITER, BETA);
dmg_say(rep, 'readings: L frozen-linear (1 frame) | F exact frozen-b (1) | I exact iterated-b (1) | I+ = I with the base''s stepped branch prior (1; base 4 once) | S phase-stepped (4)\n');

% ---- bench: ONE build, two emissions differing only in the exit sphere
macos.init(MODEL);
macos.write_grid_file('zwfs_flat.txt', zeros(N_G));
bargs = {'polarizing',false, 'ngridpts',NGRID, 'BS_AOI',AOI, ...
    'F1',s*500, 'F2',F2, 'D_LENS',s*60, 'R_BAFFLE',s*12.5, 'D_SB',s*250, ...
    'BS_T',s*1.5, 'D_L1_BS',s*150, 'D_BS_TO',D_BS_TO, 'D_BS_CMP',s*100, ...
    'R_TO_AP',s*30, 'L1_Kr',s*236.866, 'L1_Kc',-0.5829, ...
    'L2_Kr',-s*124.076, 'L2_Kc',-0.5826, ...
    'to_grid_file','zwfs_flat.txt', 'to_grid_n',N_G, 'to_grid_dx',DX_G, ...
    'tail_arch','fieldlens', 'MASK_TRIM',MASK_TRIM, ...
    'FL_F',T_FL_F, 'FL_Kc',T_FL_Kc, ...
    'FL_D',s*12, 'D_MASK_FL',T_DMF, 'DET_TRIM',T_TRIM};
G = macos.design.twyman_green(bargs{:}, 'mask_prop','nf');
G.bt.emit('zwfs_test.in');
Gl = macos.design.twyman_green(bargs{:}, 'mask_prop','nf_legacy');
Gl.bt.emit('zwfs_test_legacy.in');
iTO = G.T.iTO;  iMASK = G.T.iMASK;  iDET = G.T.iDET;
assert(isequal([Gl.T.iTO Gl.T.iMASK Gl.T.iDET], [iTO iMASK iDET]));
Z1 = Gl.bt.E(iMASK-1).zelt;  Z2l = Gl.bt.E(iMASK+1).zelt;  Z2s = G.bt.E(iMASK+1).zelt;
xg = ((0:N_G-1)-(N_G-1)/2)*DX_G;  [gxd, gyd] = meshgrid(xg, xg);
PARb = [1 2 1 1];  sgn = +1;                   % S2 registration, this deck
T = @(x) fftshift(fft2(fftshift(x)))/MODEL;
NACT0 = 96;  PITCH0 = 1.0;
[axg0, ayg0] = meshgrid(((1:NACT0)-(NACT0+1)/2)*PITCH0);
dmap0 = @(act) dm_influence_map(N_G, DX_G, 'nact',NACT0, 'pitch',PITCH0, 'act', act);
ic0 = NACT0/2;  Aa0 = zeros(NACT0);  Aa0(ic0,ic0) = 1;  Ma0 = dmap0(POKE*Aa0);
Ah0 = zeros(NACT0);  Ah0(60,40) = 1;

% ================= Block A: the legacy model, on record ===============
dmg_say(rep, '\n---- Block A: LEGACY model (nf_legacy: exit sphere zElt %.3f vs entrance %.3f mm) ----\n', Z2l, Z1);
macos.load_rx('zwfs_test_legacy.in');
E_in = macos.complex_field(iMASK-1);  E_mask = macos.complex_field(iMASK);
E_out = macos.complex_field(iMASK+1);  E_det = macos.complex_field(iDET);
dx1 = abs(macos.dx_at(iMASK-1))*1e3;
S_pred = -pi*LAM*(Z2l-Z1)*(Z1/Z2l)/(dx1*MODEL)^2;            % SPH2PL's factor, rad/px^2
% SPH2PL identity: E_mask == exp(i*S*(m^2+n^2)) .* [shifted FFT](E_in), with S
% the engine's factor -- test both FFT sign conventions and both signs of S,
% keep the best (the pin is the RELATIVE ERROR of the identity, not a fit).
[mm, nn] = meshgrid((1:MODEL) - MODEL/2 - 1);  r2 = mm.^2 + nn.^2;
Tf = @(x) fftshift(fft2(fftshift(x)))/MODEL;  Tb = @(x) fftshift(ifft2(fftshift(x)))*MODEL;
cand = {'fft2, +S', Tf(E_in).*exp(1i*S_pred*r2); 'fft2, -S', Tf(E_in).*exp(-1i*S_pred*r2); ...
        'ifft2, +S', Tb(E_in).*exp(1i*S_pred*r2); 'ifft2, -S', Tb(E_in).*exp(-1i*S_pred*r2)};
cerr = cellfun(@(c) nrm(c - E_mask)/nrm(E_mask), cand(:,2));
[S_err, ib] = min(cerr);
q = E_mask ./ Tf(E_in);  Tq = Tf(E_in);  spq = abs(Tq) > 1e-2*max(abs(Tq(:)));
z_eff = Z1*(Z1-Z2l)/Z2l;  D_in = 2*R_BEAM*Z1/F2;
f_null = D_in*sqrt(1/(2*LAM*z_eff));
dmg_say(rep, 'unmasked round trip |E_out-E_in|/|E_in| = %.3e; |E_det-E_out| = %.1e (tail = identity on the grid)\n', ...
    nrm(E_out-E_in)/nrm(E_in), nrm(E_det-E_out)/nrm(E_out));
dmg_say(rep, 'focal quadratic factor: SPH2PL S = -pi*lam*(Z2-Z1)*(Z1/Z2)/(dx1*N)^2 = %.4e rad/px^2; identity E_mask == exp(i S r^2) FFT(E_in) holds to %.2e (%s); phase spread over the spot core %.3f rad\n', ...
    S_pred, S_err, cand{ib,1}, std(angle(q(spq))));
dmg_say(rep, 'equivalent pupil Fresnel defocus z_eff = Z1(Z1-Z2)/Z2 = %.0f mm -> phase->amplitude null predicted at %.1f cyc/ap (S3 record: (56,0) 0.20 at 28, (64,0) -0.52 at 32)\n', ...
    z_eff, f_null);
warning('off', 'dmg_zwfs_gauge:roundtrip');
ZWl = dmg_zwfs_gauge(iTO, iMASK, iDET, gopt);
warning('on', 'dmg_zwfs_gauge:roundtrip');
[magl, dxdl] = dmg_frame(iTO, iDET);
litl = dmg_lit(ZWl.msk, dxdl, magl, axg0, ayg0);
Rl = struct('P',PARb, 'tax',0, 'tay',0, 'bx',0, 'by',0, 'dxd_mm',dxdl, 'mag',magl, ...
            'msk',ZWl.msk, 'N_WF',ZWl.N_WF, 'gxd',gxd, 'gyd',gyd);
hA = ZWl.measL(Ma0);
[Rl.bx, Rl.by, Rl.tax, Rl.tay] = dmg_anchor(hA, Ma0, ZWl.msk, ZWl.N_WF, xg);
hAd = sgn*dmg_samp(hA, Rl);  hAd(isnan(hAd)) = 0;
stnl = dmg_stencil(hAd, xg, Rl.tax, Rl.tay, PITCH0, 6) / POKE;
cpm = corrcoef(hAd(:), Ma0(:));
estl = @(h) dmg_act_fit(sgn*dmg_samp(h, Rl), xg, axg0, ayg0, stnl, litl);
aH = estl(ZWl.measL(dmap0(POKE*Ah0)));
rng(7);  Ab30l = zeros(NACT0);  Ab30l(litl) = 30e-6*randn(nnz(litl),1);
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), dmap0(Ab30l));
E_det1 = macos.complex_field(iDET);
rr = abs(E_det1(ZWl.msk))./abs(E_det(ZWl.msk));
dmg_say(rep, 'linear reading, center poke 20 nm: raw kernel peak gain %.4f, ring min/peak %.3f, corr(map, truth) %.3f; hold-out (60,40) through the measured-kernel fit: gain %.4f, err %.0f pm  [S2 record 0.903 / 132]\n', ...
    max(hAd(:))/max(Ma0(:)), min(stnl(:))/max(stnl(:)), cpm(1,2), aH(60,40)/POKE, ...
    sqrt(mean((aH(litl)-POKE*Ah0(litl)).^2))*1e9);
dmg_say(rep, '30 nm working state: detector amplitude |E|/|E_flat| on msk: mean %.4f, std %.3f, range %.3f..%.3f  (a phase-only DM state must leave |E| = 1)\n', ...
    mean(rr), std(rr), min(rr), max(rr));
legacy = struct('roundtrip',nrm(E_out-E_in)/nrm(E_in), 'S_pred',S_pred, 'S_err',S_err, ...
    'z_eff',z_eff, 'f_null',f_null, 'kpeak',max(hAd(:))/max(Ma0(:)), 'kmin',min(stnl(:))/max(stnl(:)), ...
    'kcorr',cpm(1,2), 'hold_g',aH(60,40)/POKE, 'ampmod_std',std(rr), 'lit',nnz(litl), 'msk',nnz(ZWl.msk));

% ================= Block B: corrected model, gates ===================
dmg_say(rep, '\n---- Block B: CORRECTED model (nf: exit sphere zElt %.3f == entrance) -- gates ----\n', Z2s);
macos.load_rx('zwfs_test.in');
ZW = dmg_zwfs_gauge(iTO, iMASK, iDET, gopt);
msk = ZW.msk;  N_WF = ZW.N_WF;  E0 = ZW.E0;  th0 = angle(E0);  cc = ZW.cc;
dmg_say(rep, 'G1 sandwich round trip (unmasked, entrance->exit sphere): %.3e   (gate < 1e-12)\n', ZW.gate.roundtrip);
assert(ZW.gate.roundtrip < 1e-12, 'G1 FAIL');
dmg_say(rep, 'G2 reference-wave surrogate T(D Ti(E0)) vs the engine''s Eb on msk: %.3e   (gate < 1e-10)\n', ZW.gate.bsur);
assert(ZW.gate.bsur < 1e-10, 'G2 FAIL');
dmg_say(rep, 'G1 G2 PASS.  dimple %.2f px, msk %d px (legacy %d)\n', ZW.dia_mm*1e-3/abs(macos.dx_at(iMASK)), nnz(msk), legacy.msk);
[mag, dxd_mm] = dmg_frame(iTO, iDET);
lit0 = dmg_lit(msk, dxd_mm, mag, axg0, ayg0);
dmg_say(rep, 'frame: ray mag %.4f, det px %.4e mm (legacy %.4f / %.4e); lit %d (legacy %d)\n', mag, dxd_mm, magl, dxdl, nnz(lit0), legacy.lit);
rng(7);  Ab30 = zeros(NACT0);  Ab30(lit0) = 30e-6*randn(nnz(lit0),1);
M30 = dmap0(Ab30);
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M30);
E_det1 = macos.complex_field(iDET);
rr = abs(E_det1(msk))./abs(E0(msk));
dmg_say(rep, 'G3 30 nm state, |E|/|E_flat| on msk: std %.3e   (gate < 1e-12: the pupil is DM-conjugate, A = |E0| is exact)\n', std(rr));
assert(std(rr) < 1e-12, 'G3 FAIL');
macos.intensity(iMASK);  macos.apodize_complex(iMASK, ZW.D);
Eb1 = macos.complex_field(iDET, 'reset_trace', false);
I1 = ZW.frameL(M30);
phi_true = angle(E_det1./E0);
Th1 = angle(cc) + angle(Eb1) - th0;
plus_true = msk & (wrap(phi_true - Th1) > 0);
rmsm = @(x, m) sqrt(mean(x(m).^2));
rmsm0 = @(x, m) sqrt(mean((x(m) - mean(x(m))).^2));      % piston removed: |E + c b|^2 is
                                                          % invariant under a common phase on
                                                          % E and b, so the iterated reading is
                                                          % exact only UP TO PISTON (the ZWFS
                                                          % piston null, S3) -- the oracle b
                                                          % carries the true piston, an iterated
                                                          % b cannot
[p_or, ~] = ZW.solveI(I1, [], [], Eb1, 0);
[p_orp, ~] = ZW.solveI(I1, [], plus_true, Eb1, 0);
dmg_say(rep, 'G4 exact solve with the ORACLE b (the engine''s Eb of the state), 30 nm base (phi rms %.3f rad): err on principal-branch px %.2e rad, on ALL msk px with the true branch %.2e rad; beyond-fold fraction %.4f (n=%d)   (gate < 1e-10)\n', ...
    rmsm(phi_true, msk), rmsm(wrap(p_or-phi_true), msk & ~plus_true), rmsm(wrap(p_orp-phi_true), msk), ...
    mean(plus_true(msk)), nnz(plus_true));
assert(rmsm(wrap(p_orp-phi_true), msk) < 1e-10, 'G4 FAIL');
[p_I, infI] = ZW.solveI(I1);
Fr0 = ZW.framesS(M30);  X30 = ZW.reconS(Fr0);  plus30 = ZW.plusFromX(X30);
[p_Ip0, ~] = ZW.solveI(I1, [], plus30);
[plus30r, pinf] = ZW.priorS(I1, Fr0);
[p_Ip, infIp] = ZW.solveI(I1, [], plus30r);
p_F = ZW.solveI(I1, [], [], [], 0);
p_L = zeros(N_WF);  p_L(msk) = (I1(msk) - ZW.I_flat(msk))./ZW.den(msk);
dmg_say(rep, 'G5 30 nm base, the readings in map space (rad rms err vs the true detector phase, raw / PISTON REMOVED): L %.3e / %.3e | F %.3e / %.3e | I %.3e / %.3e | I+ plain stepped prior %.3e / %.3e | I+ REFINED prior %.3e / %.3e\n', ...
    rmsm(wrap(p_L-phi_true), msk), rmsm0(wrap(p_L-phi_true), msk), rmsm(wrap(p_F-phi_true), msk), rmsm0(wrap(p_F-phi_true), msk), ...
    rmsm(wrap(p_I-phi_true), msk), rmsm0(wrap(p_I-phi_true), msk), rmsm(wrap(p_Ip0-phi_true), msk), rmsm0(wrap(p_Ip0-phi_true), msk), ...
    rmsm(wrap(p_Ip-phi_true), msk), rmsm0(wrap(p_Ip-phi_true), msk));
dmg_say(rep, '   (the refined-prior residual is a PISTON: mean err %.3e rad = |b_iter - Eb1|/|Eb1| %.3e; the intensity is invariant under a common phase on E and b -- the sensor''s piston null, S3 -- so an iterated b cannot recover it and the oracle b (G4) can)\n', ...
    mean(wrap(p_Ip(msk)-phi_true(msk))), nrm((infIp.b(msk)-Eb1(msk)))/nrm(Eb1(msk)));
dmg_say(rep, '   branch prior vs the true branch on msk: plain stepped agrees on %.4f (fold fraction %.4f), refined on %.4f (fold %s; true %.4f)\n', ...
    mean(plus30(msk) == plus_true(msk)), mean(plus30(msk)), mean(plus30r(msk) == plus_true(msk)), ...
    sprintf('%.4f ', pinf.frac), mean(plus_true(msk)));
dmg_say(rep, '   I  iteration dphi: %s\n   I+ iteration dphi: %s\n', sprintf('%.2e ', infI.dphi), sprintf('%.2e ', infIp.dphi));
MH = dmap0(POKE*Ah0);
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), MH);
E_detH = macos.complex_field(iDET);  phH = angle(E_detH./E0);
IH = ZW.frameL(MH);
[pH_I, infH] = ZW.solveI(IH);  pH_F = ZW.solveI(IH, [], [], [], 0);
pH_L = zeros(N_WF);  pH_L(msk) = (IH(msk) - ZW.I_flat(msk))./ZW.den(msk);
eL = rmsm0(wrap(pH_L-phH), msk);  eF = rmsm0(wrap(pH_F-phH), msk);  eI = rmsm0(wrap(pH_I-phH), msk);
dmg_say(rep, 'G6 flat hold-out 20 nm (phi rms %.2e, max %.2e rad): map err, piston removed: L %.2e | F %.2e | I %.2e rad (I/L = %.2f; raw %.2e / %.2e / %.2e); I dphi %s   (non-vacuity gate: I < L/2)\n', ...
    rmsm(phH, msk), max(abs(phH(msk))), eL, eF, eI, eI/eL, rmsm(wrap(pH_L-phH), msk), rmsm(wrap(pH_F-phH), msk), rmsm(wrap(pH_I-phH), msk), sprintf('%.1e ', infH.dphi));
assert(eI < eL/2, 'G6 FAIL');
dmg_say(rep, 'G3-G6 PASS\n');
gates = struct('roundtrip',ZW.gate.roundtrip, 'bsur',ZW.gate.bsur, 'ampmod_std',std(rr), ...
    'fold_frac30',mean(plus_true(msk)), 'map_err30',[rmsm(wrap(p_L-phi_true), msk) rmsm(wrap(p_F-phi_true), msk) ...
    rmsm(wrap(p_I-phi_true), msk) rmsm(wrap(p_Ip-phi_true), msk)], ...
    'map_err30_nopiston',[rmsm0(wrap(p_L-phi_true), msk) rmsm0(wrap(p_F-phi_true), msk) ...
    rmsm0(wrap(p_I-phi_true), msk) rmsm0(wrap(p_Ip0-phi_true), msk) rmsm0(wrap(p_Ip-phi_true), msk)], 'map_errH',[eL eF eI], ...
    'prior_agree',[mean(plus30(msk) == plus_true(msk)) mean(plus30r(msk) == plus_true(msk))]);

% ================= Block C: the battery, both DM sizes ===============
PQ96 = [1 0;2 0;4 0;8 0;16 0;24 0;32 0;40 0;48 0;56 0;64 0;72 0;80 0;8 8;24 24];
PQ48 = [1 0;2 0;4 0;8 0;12 0;16 0;24 0;32 0;40 0;8 8;16 16];
CFG = struct('nact', {96, 48}, 'pitch', {1.0, 2.0}, 'pq', {PQ96, PQ48}, 'hold', {[60 40], [30 20]});
out = struct('legacy',legacy, 'gates',gates, 'readings',{RD});
for icfg = 1:2
    NACT = CFG(icfg).nact;  PITCH = CFG(icfg).pitch;  PQ = CFG(icfg).pq;  hij = CFG(icfg).hold;
    is1d = PQ(:,2) == 0;  pk1 = PQ(is1d,1);  nm_modes = size(PQ,1);
    dmg_say(rep, '\n---- Block C: DM %dx%d, pitch %.1f mm ----\n', NACT, NACT, PITCH);
    dmap = @(act) dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', act);
    [axg, ayg] = meshgrid(((1:NACT)-(NACT+1)/2)*PITCH);
    lit = dmg_lit(msk, dxd_mm, mag, axg, ayg);
    ic = NACT/2;  Aa = zeros(NACT);  Aa(ic,ic) = 1;  Ma = dmap(POKE*Aa);
    Ah = zeros(NACT);  Ah(hij(1),hij(2)) = 1;
    % flat references for the stepped reading + the null of each reading
    Xflat = ZW.reconS(ZW.framesS(zeros(N_G)));
    Iaf = ZW.frameL(zeros(N_G));
    hnull = {ZW.reconL(Iaf), ZW.reconI(Iaf, [], [], [], 0), ZW.reconI(Iaf), ZW.reconI(Iaf), ZW.stepdiff(ZW.reconS(ZW.framesS(zeros(N_G))), Xflat)};
    % per-class kernels (the S2/S3 recipe, through each reading)
    Ia = ZW.frameL(Ma);  Fr = ZW.framesS(Ma);
    hK = {ZW.reconL(Ia), ZW.reconI(Ia), ZW.stepdiff(ZW.reconS(Fr), Xflat)};
    Rk = cell(1,3);  stn = cell(1,3);  est = cell(1,3);  kinfo = zeros(3,3);
    for k = 1:3
        R = struct('P',PARb, 'tax',0, 'tay',0, 'bx',0, 'by',0, 'dxd_mm',dxd_mm, 'mag',mag, ...
                   'msk',msk, 'N_WF',N_WF, 'gxd',gxd, 'gyd',gyd);
        [R.bx, R.by, R.tax, R.tay] = dmg_anchor(hK{k}, Ma, msk, N_WF, xg);
        hd = sgn*dmg_samp(hK{k}, R);  hd(isnan(hd)) = 0;
        stn{k} = dmg_stencil(hd, xg, R.tax, R.tay, PITCH, 6) / POKE;
        cpm = corrcoef(hd(:), Ma(:));
        kinfo(k,:) = [max(hd(:))/max(Ma(:)), min(stn{k}(:))/max(stn{k}(:)), cpm(1,2)];
        Rk{k} = R;
        est{k} = @(h) dmg_act_fit(sgn*dmg_samp(h, Rk{k}), xg, axg, ayg, stn{k}, lit);
    end
    assert(all(kinfo(:,3) > 0.9), 'registration sanity: kernel/truth correlation < 0.9');
    dmg_say(rep, 'lit actuators %d; hold-out (%d,%d); center-poke kernel per class [raw peak gain, ring min/peak, corr]: L [%.3f %.3f %.3f]  I [%.3f %.3f %.3f]  S [%.3f %.3f %.3f]\n', ...
        nnz(lit), hij(1), hij(2), kinfo(1,:), kinfo(2,:), kinfo(3,:));
    nullpm = @(a) std(a(lit))*1e9;
    dmg_say(rep, 'null (estimator on the flat, pm rms over lit): L %.2f  F %.2f  I %.2f  S %.2f\n', ...
        nullpm(est{1}(hnull{1})), nullpm(est{2}(hnull{2})), nullpm(est{2}(hnull{3})), nullpm(est{3}(hnull{5})));
    % modal transfer through each class's estimator (L, I, S)
    [ii, jj] = meshgrid((0.5:NACT)/NACT);
    gk = zeros(nm_modes, 3);
    for m = 1:nm_modes
        p = PQ(m,1);  q = PQ(m,2);
        Ak = cos(pi*p*ii).*cos(pi*q*jj);  M = dmap(AMPM*Ak);
        Ia = ZW.frameL(M);  Fr = ZW.framesS(M);
        hm = {ZW.reconL(Ia), ZW.reconI(Ia), ZW.stepdiff(ZW.reconS(Fr), Xflat)};
        for k = 1:3
            a = est{k}(hm{k});  gk(m,k) = (AMPM*Ak(lit)) \ a(lit);
        end
    end
    dmg_say(rep, 'modal transfer (actuator-space estimator, RAW):\n%9s %7s | %7s %7s %7s\n', 'mode', 'cyc/ap', 'L', 'I', 'S');
    for m = 1:nm_modes
        dmg_say(rep, '  (%2d,%2d) %7.1f | %7.4f %7.4f %7.4f%s\n', PQ(m,1), PQ(m,2), hypot(PQ(m,1),PQ(m,2))/2, ...
            gk(m,1), gk(m,2), gk(m,3), ifelse_(is1d(m), '', '   (separability row)'));
    end
    gk1 = gk(is1d,:);
    dmg_say(rep, 'min |g| over the (p,0) rows: L %.3f  I %.3f  S %.3f   [S3 legacy L: 0.50 at 1 cyc/ap, -0.52 at 32]\n', min(abs(gk1)));
    corrk = @(a, k) dmg_modal_corr(a, 'separable', pk1, gk1(:,k), BETA, NACT);
    % ---- the five rows (S6 set), RAW and corrected, five readings ------
    rng(7);   Ab30 = zeros(NACT);  Ab30(lit) = 30e-6*randn(nnz(lit),1);
    rng(23);  Arnd = zeros(NACT);  Arnd(lit) = 10e-6*randn(nnz(lit),1);
    Pg = zeros(NACT);  Pg(8:8:NACT, 8:8:NACT) = 1;  Pg = Pg .* lit;
    ROWS = {'flat/hold20',       zeros(NACT), POKE*Ah; ...
            'flat/rand10',       zeros(NACT), Arnd; ...
            'rand30/single10',   Ab30,        10e-6*Ah; ...
            'rand30/grid@1nm',   Ab30,        AMPG*Pg; ...
            'rand30/rand10',     Ab30,        Arnd};
    NR = size(ROWS,1);
    dmg_say(rep, 'rows (differential, actuator space; grid sites %d).  g = gain, e = rms err over lit (pm), flr = rms of unpoked lit (pm), SNR = mean(poked)/flr\n', nnz(Pg));
    dmg_say(rep, '%-17s %-3s | %7s %8s %8s %7s | %7s %8s %8s %7s\n', 'row', 'rd', 'g_raw', 'e_raw', 'flr_raw', 'SNRraw', 'g_cor', 'e_cor', 'flr_cor', 'SNRcor');
    res = struct('row',{},'rd',{},'raw',{},'cor',{},'fold',{});
    for r = 1:NR
        base = ROWS{r,2};  dev = ROWS{r,3};
        if r == 1 || ~isequal(base, ROWS{r-1,2})
            Mb = dmap(base);  Iab = ZW.frameL(Mb);  Frb = ZW.framesS(Mb);  Xb = ZW.reconS(Frb);
            [plusb, pinf] = ZW.priorS(Iab, Frb);
            h0 = {ZW.reconL(Iab), ZW.reconI(Iab, [], [], [], 0), ZW.reconI(Iab), ZW.reconI(Iab, [], plusb)};
        end
        M1 = dmap(base + dev);  Ia1 = ZW.frameL(M1);  Fr1 = ZW.framesS(M1);
        d = {ZW.reconL(Ia1) - h0{1}, ZW.reconI(Ia1, [], [], [], 0) - h0{2}, ZW.reconI(Ia1) - h0{3}, ...
             ZW.reconI(Ia1, [], plusb) - h0{4}, ZW.stepdiff(ZW.reconS(Fr1), Xb)};
        for k = 1:5
            araw = est{KC(k)}(d{k});  acor = corrk(araw, KC(k));
            [gr, er, fr, sr] = score_(araw, dev, lit);  [gc, ec, fc, sc] = score_(acor, dev, lit);
            dmg_say(rep, '%-17s %-3s | %7.4f %8.0f %8s %7s | %7.4f %8.0f %8s %7s\n', ifelse_(k==1, ROWS{r,1}, ''), RD{k}, ...
                gr, er, fmt0_(fr), fmt2_(sr), gc, ec, fmt0_(fc), fmt2_(sc));
            res(end+1) = struct('row',ROWS{r,1}, 'rd',RD{k}, 'raw',[gr er fr sr], 'cor',[gc ec fc sc], ...
                                'fold',mean(plusb(msk))); %#ok<AGROW>
        end
    end
    dmg_say(rep, '(beyond-fold fraction of msk on the rand30 base, prior passes: %s)\n', sprintf('%.4f ', pinf.frac));
    % ---- break scale: grow the working state ---------------------------
    dmg_say(rep, 'break scale (single act 10 nm differential on a growing base; corrected estimates; g at the poked site, floor pm, SNR; fold0/fold = beyond-fold fraction from the plain / refined stepped prior; past the fold quote GAIN, not SNR -- aliased recoveries still clear a formal SNR):\n');
    dmg_say(rep, '%8s %6s %6s |', 'base', 'fold0', 'fold');
    for k = 1:5, dmg_say(rep, ' %-22s|', sprintf('%s: g flr SNR', RD{k})); end
    dmg_say(rep, '\n');
    rng(7);  Bfield = zeros(NACT);  Bfield(lit) = randn(nnz(lit),1);  Bfield = Bfield / std(Bfield(lit));
    Asng = 10e-6*Ah;  pkm = Asng > 0;  un = lit & ~pkm;
    lad = struct('amp',{},'fold',{},'fold0',{},'g',{},'flr',{},'snr',{});
    for amp = LADDER
        Ab = amp*Bfield;  Mb = dmap(Ab);  Iab = ZW.frameL(Mb);  Frb = ZW.framesS(Mb);  Xb = ZW.reconS(Frb);
        [plusb, pinf] = ZW.priorS(Iab, Frb);
        M1 = dmap(Ab + Asng);  Ia1 = ZW.frameL(M1);  Fr1 = ZW.framesS(M1);
        d = {ZW.reconL(Ia1) - ZW.reconL(Iab), ZW.reconI(Ia1, [], [], [], 0) - ZW.reconI(Iab, [], [], [], 0), ...
             ZW.reconI(Ia1) - ZW.reconI(Iab), ...
             ZW.reconI(Ia1, [], plusb) - ZW.reconI(Iab, [], plusb), ZW.stepdiff(ZW.reconS(Fr1), Xb)};
        g = nan(1,5);  fl = g;  sn = g;
        for k = 1:5
            a = corrk(est{KC(k)}(d{k}), KC(k));
            g(k) = a(hij(1),hij(2))/10e-6;  fl(k) = std(a(un))*1e9;  sn(k) = abs(g(k))*1e4/max(fl(k), eps);
        end
        dmg_say(rep, '%5.0f nm %6.4f %6.4f |', amp*1e6, pinf.frac(1), pinf.frac(end));
        for k = 1:5, dmg_say(rep, ' %7.4f %6.0f %7.1f|', g(k), fl(k), sn(k)); end
        dmg_say(rep, '\n');
        lad(end+1) = struct('amp',amp, 'fold',pinf.frac(end), 'fold0',pinf.frac(1), 'g',g, 'flr',fl, 'snr',sn); %#ok<AGROW>
    end
    % ---- verdict lines against the lit-scan spec -----------------------
    iH = find(strcmp({res.row}, 'flat/hold20') & strcmp({res.rd}, 'I'), 1);
    iG = find(strcmp({res.row}, 'rand30/grid@1nm'));
    gI = res(iH).raw(1);
    dmg_say(rep, 'SPEC: flat hold-out RAW gain, reading I (one frame, no Wiener): %.4f -> %s (within 3%% of 1)\n', gI, ifelse_(abs(gI-1) <= 0.03, 'PASS', 'not met'));
    for k = iG
        dmg_say(rep, 'SPEC: grid-on-30nm-base SNR, reading %-2s: raw %5.2f / corrected %5.2f -> %s\n', res(k).rd, res(k).raw(4), res(k).cor(4), ...
            ifelse_(max(res(k).raw(4), res(k).cor(4)) >= 5, 'DETECTED (>= 5)', 'below 5'));
    end
    out.(sprintf('n%d', NACT)) = struct('lit',lit, 'kinfo',kinfo, 'PQ',PQ, 'gk',gk, 'rows',{ROWS(:,1)}, ...
                                        'res',res, 'ladder',lad, 'spec_hold_gI',gI);
end
dmg_say(rep, 'S7 complete in %.1f min\n', toc(t_all)/60);
fclose(rep);
save('zwfs_s7iter.mat', 'out');
fprintf('wrote zwfs_s7iter_report.txt + zwfs_s7iter.mat\n');
end

% ---- helpers ----------------------------------------------------------
function [g, e, fl, snr] = score_(a, Ad, lit)
g = Ad(lit) \ a(lit);
e = sqrt(mean((a(lit) - Ad(lit)).^2))*1e9;
pk = (Ad ~= 0) & lit;  un = lit & ~pk;
if nnz(pk) < nnz(lit)/4
    fl = std(a(un))*1e9;  snr = mean(a(pk)) / max(std(a(un)), eps);
else
    fl = NaN;  snr = NaN;
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
