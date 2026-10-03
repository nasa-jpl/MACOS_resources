% example_bench_ifo.m
% ===================================================================
%  MACOS DESIGN LAYER -- TWYMAN-GREEN INTERFEROMETER ON A BENCH,
%  WITH PHASE-SHIFTING (de GROOT) DATA PROCESSING
% ===================================================================
%  A generic Twyman-Green IFO built with the macos.design.Bench
%  add-optic utilities, then RUN as an instrument: simulate 7
%  phase-shifted interferograms and recover the test-optic surface
%  with windowed phase-shifting-interferometry (PSI) processing --
%  the full simulate -> detect -> process -> compare-to-truth loop.
%
%  LAYOUT (two sequential prescriptions sharing one source + output):
%
%    TEST ARM   source -> baffle -> L1 (collimator) -> BS reflect
%               (45 deg AOI, front-coated 1.5 mm plate) -> COMPENSATOR
%               plate (double-passed) -> TEST OPTIC (retro; carries a
%               known weak-sphere figure + tilt = the "unknown") ->
%               back through the compensator -> BS transmit -> Recomb
%               plane -> L2 -> focal mask (focus) -> detector at the
%               TEST-OPTIC PUPIL IMAGE (fringes imaged there).
%    REF ARM    same source/L1 -> BS transmit -> PZT MIRROR (retro;
%               the phase shifter, BS->PZT = BS->test optic) -> BS
%               return-reflect (real glass path: back-face in,
%               internal reflect at the coating, back-face out) ->
%               the SAME Recomb plane -> same output train.
%
%  PLATE BALANCE (the classic Twyman-Green compensator): with a
%  front-coated plate BS, the reference arm crosses glass 3x (forward
%  transit + the internal V on return) while the bare test arm would
%  cross only 1x (output transit).  The COMPENSATOR -- an identical
%  plate at the identical 45-deg orientation, double-passed in the
%  test leg -- brings the test arm to 3 equal transits: glass paths
%  balance EXACTLY (same plate, same internal angles), so plate
%  piston/shear/aberration cancel between the arms.  (A pellicle
%  would dodge the bookkeeping but is aberration-challenged in real
%  instruments -- modeled the honest way instead.)
%
%  Both arms are built from ONE parameter set; the BS and compensator
%  are shared geometric plate TOKENS, so both prescriptions reference
%  the same physical planes.  MACOS is a sequential tracer, so the two
%  arms are two Rx; interference is a coherent COMPLEX ADD of their
%  fields.  Per the spec the add happens at the recombination plane
%  just after the BS and the sum is then propagated down the test-arm
%  continuation: implemented by injecting the sum into the test arm at
%  the Recomb plane via macos.apodize_complex (multiply E_test by
%  M = 1 + E_ref/E_test) and propagating on.  The driver first PROBES
%  that the injection is active; if the engine's (geometric)
%  propagation rebuilds the field from ray OPD and ignores it, it
%  falls back to superposing the two arms' fields AT the detector --
%  wave propagation is linear, so with a COMMON output train the two
%  are the same field.  The run prints which path ran.
%
%  PHASE SHIFTING + PROCESSING:
%  - The PZT mirror steps along its normal by lambda/8 per frame
%    (double pass -> pi/2 of fringe phase), 7 frames k = -3..3.
%  - Primary estimator: de Groot windowed PSI (P. de Groot, "Derivation
%    of algorithms for phase-shifting interferometry using the concept
%    of a data-sampling window", Appl. Opt. 34, 4723 (1995)): windowed
%    synchronous detection with the triangular 7-frame window
%    [1 2 3 4 3 2 1].  Exact identity at pi/2 steps:
%      sum(w I sin th) = -8 b sin(phi),  sum(w I cos th) = 8 b cos(phi).
%  - Comparison estimator: the Hariharan/Schwider 5-frame.
%  - Surface recovery: h = phi * lambda / (4*pi)  (double pass), then a
%    least-squares fit of {piston, tip, tilt, defocus} in test-optic
%    coordinates: the defocus coefficient recovers the sphere radius
%    (h = r^2/(2R)), the tilt term the injected tilt, and the post-fit
%    residual is the end-to-end processing error.
%
%  Run:  matlab -batch "run('.../example_bench_ifo.m')"   (needs MACOS_HOME)
% ===================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));
if isempty(exdir), exdir = pwd; end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');

MODEL = 256;
macos.init(MODEL);

% ---- parameters (mm) ----------------------------------------------
P = struct();
P.F1       = 500;    P.F2 = 250;   P.D_LENS = 60;  P.N_GLASS = 1.5;
P.R_BAFFLE = 12.5;   P.D_SB = 250; P.FILL = 0.95;
P.BS_T     = 1.5;                  % front-coated plate (compensated below)
P.D_L1_BS  = 150;
P.D_BS_TO  = 250;                  % BS -> test optic  (= BS -> PZT, symmetric)
P.D_BS_CMP = 100;                  % BS -> compensator plate (test leg)
P.D_RECOMB = 5;                    % BS exit -> recombination plane
P.D_RC_L2  = 200;                  % recomb -> L2
P.R_TO_AP  = 30;                   % test-optic aperture radius
% L1/L2 conics: bench_layout optimized values (same conjugates)
P.L1_Kr = 236.866;  P.L1_Kc = -0.5829;
P.L2_Kr = -124.076; P.L2_Kc = -0.5826;
% the "unknown" figure on the test optic (truth to recover):
%   A weak SPHERE, sized so the double-pass phase stays within +/-pi
%   across the pupil -- the PSI arctangent output is WRAPPED, and
%   keeping the truth inside one branch keeps this example free of 2-D
%   phase unwrapping.  (A full-scale figure -- many fringes -- works
%   identically through the interferometer; you then feed the wrapped
%   phi into a real 2-D unwrapper before the surface fit.)
%   Two calibration facts established while building this example:
%   - NULL: with a FLAT test optic the two arms match to ~2.5e-9 rad
%     -- the compensated-plate balance is exact, so what is recovered
%     below is genuinely the injected truth.
%   - TILT: a rigid tilt of the test optic is measured too, but its
%     phase signature through this folded 45-deg-plate double-pass is
%     NOT the naive 2*theta*x: it is ~2x larger and carries geometric
%     cross terms (pupil tilt plus an even astigmatism-like term), so
%     a tilt big enough for visible fringes wraps the phase.  The
%     demo therefore injects SPHERE ONLY and uses the fitted tilt
%     term as a null check; tilt metrology through this geometry (with
%     an unwrapper + a calibrated tilt model) is a natural extension.
P.R_FIG    = -2.0e7;               % weak sphere: sag ~ 13 nm at the edge
P.TILT     = 0;                    % rad (see TILT note; 0 = null check)
LAM        = 6.328e-4;             % mm
NFRAME     = 7;

% ---- build both arms with the Bench builder ------------------------
[bt, T, bs, det_leg] = build_test_arm(P);
[br, R] = build_ref_arm(P, bs, bt.E(T.iRC).vpt, det_leg);
fprintf('--- test arm ---\n'); bt.print_chain();
fprintf('--- ref arm ----\n'); br.print_chain();
% glass-balance audit: equal in-glass path in both arms (compensation)
fprintf('glass path: test %.4f mm, ref %.4f mm (diff %.2g)\n', ...
    glass_path(bt), glass_path(br), glass_path(bt) - glass_path(br));
rx_test = fullfile(exdir, 'ifo_test_arm.in');   bt.emit(rx_test);
rx_ref  = fullfile(exdir, 'ifo_ref_arm.in');    br.emit(rx_ref);

% input-parameter schematics for both arms
fs = bt.sketch('title', ...
    'IFO test arm -- BS reflect -> compensator (2x) -> test optic -> BS transmit out');
print(fs, fullfile(exdir,'ifo_test_arm_params.png'), '-dpng', '-r150');
fs = br.sketch('title', ...
    'IFO reference arm -- BS transmit -> PZT (phase shifter) -> BS internal-reflect out');
print(fs, fullfile(exdir,'ifo_ref_arm_params.png'), '-dpng', '-r150');

% ---- apply the tilt part of the "unknown" (sphere is in the Rx) -----
if P.TILT ~= 0
    macos.load_rx(rx_test);
    psi0 = macos.get_elt_psi(T.iTO);
    Rz = [cos(P.TILT) -sin(P.TILT) 0; sin(P.TILT) cos(P.TILT) 0; 0 0 1];
    macos.set_elt_psi(T.iTO, Rz*psi0);
    rx_testf = fullfile(exdir, 'ifo_test_arm_fig.in');
    macos.save_rx(rx_testf);
else
    rx_testf = rx_test;
end

% beam radius at the test optic (physical scale for the truth fit)
macos.load_rx(rx_testf);
s = macos.trace(T.iTO);  info = macos.get_ray_info(s.nRays);
ok = info.ok_trace(:) & info.ok_pass(:);
d = info.pos(:,ok) - info.pos(:,1);
dch = info.dir(:,1)/norm(info.dir(:,1));  d = d - dch*(dch.'*d);
r_beam = max(sqrt(sum(d.^2,1)));
fprintf('beam radius at test optic: %.2f mm\n', r_beam);

% ---- reference-arm fields at the recomb plane, 7 PZT steps ---------
macos.load_rx(rx_ref);
pzt_psi = macos.get_elt_psi(R.iPZT);
pzt_vpt = macos.get_elt_vpt(R.iPZT);
Er = cell(1, NFRAME);
for k = 1:NFRAME
    macos.load_rx(rx_ref);
    dz = (k - (NFRAME+1)/2) * LAM/8;          % lambda/8 -> pi/2 double-pass
    macos.set_elt_vpt(R.iPZT, pzt_vpt + dz*pzt_psi);
    Er{k} = macos.complex_field(R.iRC);
end
% self-check: realized phase step between consecutive frames
msk0 = abs(Er{4}) > 0.1*max(abs(Er{4}(:)));
step = zeros(1, NFRAME-1);
for k = 1:NFRAME-1
    q = Er{k+1}(msk0) .* conj(Er{k}(msk0));
    step(k) = angle(mean(q));
end
fprintf('realized PZT phase steps (want |step| = pi/2 = 1.571): %s rad\n', ...
    mat2str(abs(step), 4));

% ---- test-arm field at the recomb plane ----------------------------
macos.load_rx(rx_testf);
Et = macos.complex_field(T.iRC);
msk = abs(Et) > 0.1*max(abs(Et(:)));

% ---- interferograms: complex add at Recomb, propagate to detector --
%  Probe whether the recomb-plane injection reaches the detector (a
%  pure phase mask must change the detector field).
macos.load_rx(rx_testf);
E0  = macos.complex_field(T.iDET);
macos.load_rx(rx_testf);
macos.complex_field(T.iRC);
macos.apodize_complex(T.iRC, exp(1i*ones(size(Et))));
E1  = macos.complex_field(T.iDET, 'reset_trace', false);
inject_works = max(abs(E1(:) - E0(:))) > 1e-6*max(abs(E0(:)));
fprintf('recomb-plane injection %s\n', ternstr(inject_works, ...
    'ACTIVE: complex add at the BS, then propagate', ...
    'inert under geometric propagation -> superposing at the detector (equal by linearity)'));

I = cell(1, NFRAME);
if inject_works
    for k = 1:NFRAME
        macos.load_rx(rx_testf);
        Etk = macos.complex_field(T.iRC);
        M = ones(size(Etk));
        M(msk) = 1 + Er{k}(msk) ./ Etk(msk);
        macos.apodize_complex(T.iRC, M);
        Ed = macos.complex_field(T.iDET, 'reset_trace', false);
        I{k} = abs(Ed).^2;
    end
else
    macos.load_rx(rx_testf);
    Etd = macos.complex_field(T.iDET);
    for k = 1:NFRAME
        macos.load_rx(rx_ref);
        dz = (k - (NFRAME+1)/2) * LAM/8;
        macos.set_elt_vpt(R.iPZT, pzt_vpt + dz*pzt_psi);
        Erd = macos.complex_field(R.iDET);
        I{k} = abs(Etd + Erd).^2;
    end
end

% fringe visibility in the pupil
Imax = max(cat(3, I{:}), [], 3);  Imin = min(cat(3, I{:}), [], 3);
vis = (Imax(msk) - Imin(msk)) ./ max(Imax(msk) + Imin(msk), eps);
fprintf('fringe visibility (median in pupil): %.3f\n', median(vis));

% ---- PSI processing ------------------------------------------------
phi7 = psi_degroot7(I);                 % primary (windowed, 7-frame)
phi5 = psi_hariharan5(I(2:6));          % comparison (5-frame)
dphi = angle(exp(1i*(phi7 - phi5)));
fprintf('deGroot-7 vs Hariharan-5 agreement: %.3g rad rms (in pupil)\n', ...
    std(dphi(msk)));

% surface height: OPD = 2 h  ->  h = phi * lam / (4 pi).  The injected
% truth keeps phi within one branch of the arctangent, so no 2-D
% unwrapping is needed (see the R_FIG/TILT sizing note above).
h = phi7 * LAM / (4*pi);                % mm (surface height, single pass)

% ---- fit the truth model in test-optic coordinates -----------------
%  basis {1, X, Y, X^2+Y^2}; pixel -> mm scale from the measured pupil
%  radius (pixels) vs the traced beam radius at the test optic (mm).
N = size(h,1);
[xx, yy] = meshgrid(1:N, 1:N);
cx = sum(xx(msk))/nnz(msk);  cy = sum(yy(msk))/nnz(msk);
r_pix = max(sqrt((xx(msk)-cx).^2 + (yy(msk)-cy).^2));
sc = r_beam / r_pix;                    % mm per pixel, at the test optic
X = (xx - cx)*sc;  Y = (yy - cy)*sc;  R2 = X.^2 + Y.^2;
A = [ones(nnz(msk),1), X(msk), Y(msk), R2(msk)];
c = A \ h(msk);
h_res = h(msk) - A*c;
R_hat = 1/(2*c(4));                     % h = r^2/(2R)  ->  R = 1/(2 c4)
tilt_hat = hypot(c(2), c(3));           % surface tilt magnitude
fprintf('\n=== recovered test-optic figure (deGroot-7) ===\n');
fprintf('  sphere radius: |R_hat| = %.4g mm   (truth |R| = %.4g; %.2f%% error)\n', ...
    abs(R_hat), abs(P.R_FIG), 100*abs((abs(R_hat)-abs(P.R_FIG))/abs(P.R_FIG)));
fprintf('  tilt term: %.4g rad   (injected %.4g -- null check, expect ~0)\n', ...
    tilt_hat, P.TILT);
fprintf('  post-fit residual: %.3g nm rms surface (%.3g waves)\n', ...
    1e6*std(h_res), std(h_res)/LAM);

out = struct('params', P, 'lambda', LAM, 'r_beam', r_beam, ...
    'visibility', median(vis), 'steps', step, 'R_hat', R_hat, ...
    'tilt_hat', tilt_hat, 'resid_nm', 1e6*std(h_res), ...
    'inject_works', inject_works, 'rx_test', rx_testf, 'rx_ref', rx_ref);

% ---- figures -------------------------------------------------------
f1 = figure('Color','w', 'Position',[100 100 1500 500]);
subplot(1,3,1); imagesc(I{4});  axis image; colorbar;
title('interferogram, frame 4 (\delta = 0)');
subplot(1,3,2); pw = nan(N); pw(msk) = phi7(msk);
imagesc(pw, 'AlphaData', ~isnan(pw)); axis image; colorbar;
title('recovered phase (deGroot-7, wrapped, rad)');
subplot(1,3,3); hm = nan(N); hm(msk) = 1e6*(h(msk) - c(1));
imagesc(hm, 'AlphaData', ~isnan(hm)); axis image; colorbar;
title('recovered surface, piston removed (nm)');
print(f1, fullfile(exdir,'ifo_frames_and_phase.png'), '-dpng', '-r150');

f2 = figure('Color','w', 'Position',[100 100 1200 500]);
subplot(1,2,1); hf = nan(N); hf(msk) = 1e6*(A*c - c(1));
imagesc(hf, 'AlphaData', ~isnan(hf)); axis image; colorbar;
title('fitted truth model: tilt + sphere (nm)');
subplot(1,2,2); rmap = nan(N); rmap(msk) = 1e6*h_res;
imagesc(rmap, 'AlphaData', ~isnan(rmap)); axis image; colorbar;
title(sprintf('residual after truth fit: %.2g nm rms', 1e6*std(h_res)));
print(f2, fullfile(exdir,'ifo_surface_recovery.png'), '-dpng', '-r150');

save(fullfile(exdir,'bench_ifo.mat'), 'out');
fprintf('\nDONE.  Interferometer example artifacts in %s\n', exdir);

% ===================================================================
%  LOCAL FUNCTIONS
% ===================================================================
function [b, ix, bs, det_leg] = build_test_arm(P)
%BUILD_TEST_ARM  Probe leg: BS reflect -> compensator (down) -> test
%   optic -> compensator (up) -> BS transmit, then the output train.
%   Returns the shared BS token and the detector leg length so the
%   reference arm lands on the SAME planes.
    b = common_front(P, 'ifo_test');
    [~, bs] = b.add_bs_reflect(P.D_L1_BS, [0;-1;0], ...
                'thickness',P.BS_T, 'n',P.N_GLASS);
    cmp = b.plate(P.D_BS_CMP, bs.psi, 'thickness',P.BS_T, ...
                'n',P.N_GLASS, 'name','Comp');
    b.add_bs_transmit(cmp, 'tag','d');          % down toward the test optic
    ix.iTO = b.add_mirror(P.D_BS_TO - P.D_BS_CMP - P.BS_T, 'name','TestOptic', ...
                'aprad',P.R_TO_AP, 'Kr',P.R_FIG);
    b.add_bs_transmit(cmp, 'tag','u');          % back up through the comp
    b.add_bs_transmit(bs, 'tag','o');           % out through the BS
    ix.iRC = b.add_reference(P.D_RECOMB, 'Recomb');
    [ox, det_leg] = output_train(b, P, ix.iTO, []);
    ix = merge_ix(ix, ox);
end

function [b, ix] = build_ref_arm(P, bs, rc_vpt, det_leg)
%BUILD_REF_ARM  Reference leg: transmit through the SAME plate (shared
%   token), PZT retro symmetric to the test optic, internal-reflect
%   return off the coating into the output port, recombine on the SAME
%   plane as the test arm, then the same output train to the same
%   detector plane.
    b = common_front(P, 'ifo_ref');
    b.add_bs_transmit(bs, 'tag','f');           % straight through, +x
    ix.iPZT = b.add_mirror(P.D_BS_TO, 'name','PZT');   % retro phase shifter
    b.add_bs_reflect_return(bs);                % coating reflect -> +y out
    d_rc = dot(rc_vpt - b.pos, b.dir);          % land on the test arm's plane
    assert(d_rc > 0, 'ref arm: recomb plane is behind the return beam');
    ix.iRC = b.add_reference(d_rc, 'Recomb');
    [ox, ~] = output_train(b, P, [], det_leg);
    ix = merge_ix(ix, ox);
end

function b = common_front(P, name)
%COMMON_FRONT  Shared source + baffle + collimator (identical in both
%   arms; identical source grid => pixel-registered fields).
    AP = 2*atan(P.R_BAFFLE/P.D_SB)*P.FILL;
    b = macos.design.Bench(name, 'aperture', AP, 'ngridpts', 63);
    b.add_baffle(P.D_SB, P.R_BAFFLE);
    L1 = b.add_lens(P.F1 - P.D_SB, P.F1, P.D_LENS, 'mode','collimate', ...
                    'n',P.N_GLASS, 'name','L1');
    b.E(L1.i_pow).Kr = P.L1_Kr;  b.E(L1.i_pow).Kc = P.L1_Kc;
end

function [ix, det_leg] = output_train(b, P, conj_elt, det_leg)
%OUTPUT_TRAIN  Recomb -> L2 -> focal mask -> detector.  If CONJ_ELT is
%   given (test arm), the detector leg is computed so the detector is
%   CONJUGATE to that element through L2 (thin lens); the returned
%   det_leg is then reused verbatim by the reference arm so both
%   prescriptions share the detector plane.
    L2 = b.add_lens(P.D_RC_L2, P.F2, P.D_LENS, 'mode','focus', ...
                    'n',P.N_GLASS, 'name','L2');
    b.E(L2.i_pow).Kr = P.L2_Kr;  b.E(L2.i_pow).Kc = P.L2_Kc;
    ix.iMASK = b.add_reference(P.F2 - L2.thickness, 'FocalMask');
    if ~isempty(conj_elt)
        s_o = b.E(L2.i_pow).s - b.E(conj_elt).s;
        s_i = 1/(1/P.F2 - 1/s_o);
        det_leg = s_i - (b.E(ix.iMASK).s - b.E(L2.i_pow).s);
    end
    ix.iDET = b.add_detector(det_leg, 'Detector');
end

function g = glass_path(b)
%GLASS_PATH  Total chief path spent inside glass (legs whose PRECEDING
%   element left the ray in a medium with index > 1).
    g = 0;  prev_s = 0;  n_now = 1;
    for k = 1:numel(b.E)
        leg = b.E(k).s - prev_s;  prev_s = b.E(k).s;
        if n_now > 1, g = g + leg; end
        n_now = b.E(k).indref;
    end
end

function a = merge_ix(a, b)
    for f = fieldnames(b).', a.(f{1}) = b.(f{1}); end
end

function phi = psi_degroot7(I)
%PSI_DEGROOT7  de Groot windowed 7-frame PSI, pi/2 steps, triangular
%   window w = [1 2 3 4 3 2 1] (Appl. Opt. 34, 4723 (1995)).  With
%   theta_k = (k-4)*pi/2:  sum(w I sin) = -8 b sin(phi),
%   sum(w I cos) = 8 b cos(phi)  (exact; the window kills the DC term).
    w  = [1 2 3 4 3 2 1];
    th = ((1:7) - 4) * pi/2;
    num = zeros(size(I{1}));  den = zeros(size(I{1}));
    for k = 1:7
        num = num + w(k)*I{k}*sin(th(k));
        den = den + w(k)*I{k}*cos(th(k));
    end
    phi = atan2(-num, den);
end

function phi = psi_hariharan5(I)
%PSI_HARIHARAN5  Hariharan/Schwider 5-frame PSI, pi/2 steps
%   (frames at -pi..pi):  tan(phi) = 2(I2-I4) / (2 I3 - I1 - I5).
    phi = atan2(2*(I{2} - I{4}), 2*I{3} - I{1} - I{5});
end

function s = ternstr(c, a, b)
    if c, s = a; else, s = b; end
end
