function C = cophi_ctx(P, G, say)
%COPHI_CTX  The heterodyne (COPHI) reading on the LOADED-RUN bench, as a tg96_run
%   reading context: tg96_run calls it from arm_setup_ when P.reading_ctx is
%   'cophi_ctx', so its placement, matrix build and deck rows run unchanged on
%   the H reading (the record's code path, seeds and lit set).  cophi_run's
%   bench stage calls it directly and prints the registration (C.reg).
%
%   What it does, once per call (run dir = cwd, decks <tag>_test.in / _ref.in):
%   solves the test laser's launch at the input polarizer and the analyzer that
%   passes the test arm, solves the LO's PM-fiber launch, places the LO's field
%   on the test arm's pixels BY CAMERA POSITION (the geometric tail places by
%   ray index; see the README's stage bench), scales it to P.cophi.lo_ratio,
%   and returns
%     C.measf(M)  surface map (mm) of the grid M on the DM, about the flat null
%     C.phasef(M) the raw reconstructed phase (rad) -- dmg_cophi_frames(Et,
%                 Elo, P.cophi) -> dmg_cophi_recon, noise-free
%     C.p_null, C.msk, C.AT (.rx/.b/.iTO/.iDET), C.reading = 'H', C.reg
%   plus the fields tg96_run's stages read (N_G, DX_G, LAM, QWP, THETAS, NACT,
%   PITCH, rxT, rxR).
if nargin < 3 || isempty(say), say = @(varargin) fprintf(varargin{:}); end
rxT = [P.tag '_test.in'];  rxL = [P.tag '_ref.in'];
macos.init(P.MODEL);
iDET = G.T.iDET;  assert(G.R.iDET == numel(G.br.E) && iDET == numel(G.bt.E), 'a detector is not its deck''s last element');
say('LO arm: fiber tip at [%s] mm, collimator powered face AT the flat''s seat [%s] mm (elt %d of %d)\n', ...
    num2str(G.R.src_lo(:).', '%.2f '), num2str(G.R.seat(:).', '%.2f '), G.R.iLO, numel(G.br.E));

% ---- the test arm's launch: the laser's polarization at the input polarizer
% The record's Ex = Ey launch is defined in the SOURCE frame; on the mirror rig
% the OAP fold turns that frame against the input polarizer's, and the launch
% arrives nearly CROSSED (measured 2026-10-11: 1e-6 of the lens rig's power, the
% rest leakage).  So solve it as the LO's is solved: the launch that puts the
% most power through the input polarizer with the analyzer OPEN (0 + 90 deg),
% i.e. the laser rotated to the polarizer, which is what a bench does.
M = zeros(2);
for th = [0 90]
    A1 = arm_field_(rxT, G.bt, th, P, [1 0]);  A2 = arm_field_(rxT, G.bt, th, P, [0 1]);
    M = M + [A1(:) A2(:)]' * [A1(:) A2(:)];

end
test_launch = top_eig_(M);
say('Test-arm launch (laser at the input polarizer): Ex %.4f%+.4fi, Ey %.4f%+.4fi (open-analyzer power Ex-only %.4g, Ey-only %.4g, solved %.4g)\n', ...
    real(test_launch(1)), imag(test_launch(1)), real(test_launch(2)), imag(test_launch(2)), ...
    real(M(1,1)), real(M(2,2)), real(test_launch*M*test_launch'));
% ---- the analyzer: solve the angle that passes the test arm -----------
Tth = zeros(1, 3);  th3 = [0 45 90];
for k = 1:3
    E = arm_field_(rxT, G.bt, th3(k), P, test_launch);  Tth(k) = sum(abs(E(:)).^2);
end
a = (Tth(1) + Tth(3))/2;  b = (Tth(1) - Tth(3))/2;  c = Tth(2) - a;
th_pass = 0.5*atan2d(c, b);
Et = arm_field_(rxT, G.bt, th_pass, P, test_launch);
camT = cam_map_(iDET, size(Et, 1));               % the test arm's pixel -> camera position map
Tmax = sum(abs(Et(:)).^2);
say('Analyzer: test-arm power at 0/45/90 deg = %.6g / %.6g / %.6g -> passes at %.3f deg (%.6g; extinction %.2e)\n', ...
    Tth, th_pass, Tmax, (a - hypot(b, c))/(a + hypot(b, c)));
[~, ~, frT] = dmg_frame(G.T.iTO, iDET);  dxT = abs(macos.dx_at(iDET, 'mm'));

% ---- the LO's launch state: the PM fiber's orientation ----------------
% The engine is linear in the launch (Ex0, Ey0), so trace the two basis states
% through the analyzer and take the top eigenvector of their 2x2 Gram matrix:
% the launch that puts the most LO power through the analyzer -- what aligning
% a PM fiber's axis does on the bench.  Verified by a third trace AT that state.
Ex1 = arm_field_(rxL, G.br, th_pass, P, [1 0]);
Ey1 = arm_field_(rxL, G.br, th_pass, P, [0 1]);
Mg  = [Ex1(:) Ey1(:)]' * [Ex1(:) Ey1(:)];
if ~any(strcmp({G.br.E.name}, 'Analyzer'))
    % the PLAIN splitter: no analyzer projects the two arms onto one state, so
    % the PM fiber is aligned for the COHERENT overlap with the test field --
    % maximize sum_pix |sum_c conj(Et_c) Elo_c|^2 over the launch (Q = w w^H)
    w1 = sum(conj(Et) .* Ex1, 3);  w2 = sum(conj(Et) .* Ey1, 3);
    Mg = [w1(:) w2(:)]' * [w1(:) w2(:)];
    say('Plain splitter: the LO launch maximizes the coherent overlap with the test field (no analyzer)\n');
end
lo_launch = top_eig_(Mg);  v = lo_launch(:);
Elo = arm_field_(rxL, G.br, th_pass, P, lo_launch);
camL = cam_map_(G.R.iDET, size(Elo, 1));
lin_err = norm(Elo(:) - (v(1)*Ex1(:) + v(2)*Ey1(:))) / norm(Elo(:));
say('LO launch (PM fiber axis): Ex %.4f%+.4fi, Ey %.4f%+.4fi (power through the analyzer: Ex-only %.4g, Ey-only %.4g, solved %.4g; linearity check %.1e)\n', ...
    real(v(1)), imag(v(1)), real(v(2)), imag(v(2)), real(Mg(1,1)), real(Mg(2,2)), sum(abs(Elo(:)).^2), lin_err);
[~, ~, frL] = dmg_frame(G.R.iLO, G.R.iDET);  dxL = abs(macos.dx_at(G.R.iDET, 'mm'));   % the LO deck's OWN detector index
% the LO without the analyzer's projection (its power before the clean-up)
Elo_open = arm_field_(rxL, G.br, NaN, P, lo_launch);
lo_pass = sum(abs(Elo(:)).^2) / sum(abs(Elo_open(:)).^2);
% ---- the LO onto the test arm's pixels, BY CAMERA POSITION -------------
% The geometric tail places each ray's field on the detector grid BY RAY INDEX
% (verified 2026-10-11: field pixel = OPDMat index + the centring offset, phase
% = +k W, residual 1e-9 rad), not by where the ray lands.  The record's IFO
% never sees it: its reference arm shares the test arm's collimator, so both
% decks carry the SAME index -> position map.  The LO has its own collimator,
% and on the mirror rig the test arm's off-axis parabola maps the source grid
% differently from the LO's lens: the same pixel then holds two different
% camera points, and the image plane's many-wave fold aberration does not
% cancel (measured: 681 nm rms per index vs 0.45 nm by position).  So the LO
% is placed on the test arm's pixels by PHYSICAL position: its field is split
% into the smooth part E exp(-i k W) (amplitude + polarization) and the path
% W (smooth, many waves), both interpolated at the test rays' camera points,
% and recombined.  On the lens rig the two maps agree and this is a no-op to
% round-off; it is applied on both rigs, one code path.
Elo_native = Elo;
[Elo, sg_lo, sp_lo] = lo_on_test_(Elo_native, camL, camT, P.LAM);
say('LO placement: the LO field''s path phase measured as %+d k W (residual phase rms %.3g rad; the other sign %.3g rad)\n', ...
    sg_lo, min(sp_lo), max(sp_lo));
% ---- the grids -------------------------------------------------------
It = sum(abs(Et).^2, 3);  Il = sum(abs(Elo).^2, 3);
mskT = It > 0.1*max(It(:));
IlN = sum(abs(Elo_native).^2, 3);  mskL = IlN > 0.1*max(IlN(:));   % the LO's OWN footprint, its own grid
N = size(It, 1);
[cT, rT] = centroid_(mskT);  [cL, rL] = centroid_(mskL);
% both footprints in camera mm about each deck's chief ray, then in test pixels
cT_mm = (cT - frT.cen)*dxT;  cL_mm = (cL - frL.cen)*dxL;  rT_mm = rT*dxT;  rL_mm = rL*dxL;
cL = cL_mm/dxT + frT.cen;  rL = rL_mm/dxT;      % LO footprint, expressed on the test grid
[camT.idxdiff_rms, camT.idxdiff_max] = idx_diff_(camT, camL);
say(['\nDetector grid: N %d both decks; engine pitch test %.9g mm, LO %.9g mm (ratio - 1 = %.2e)\n' ...
     '  -> the LO field is placed on the test arm''s pixels BY CAMERA POSITION (the rows below are on that one grid);\n' ...
     '     the two decks'' same-index camera points differ by %.3f px rms (%.3f max) -- the index map the placement removes\n'], ...
    N, dxT, dxL, dxL/dxT - 1, camT.idxdiff_rms/dxT, camT.idxdiff_max/dxT);
chL = (frL.chief - frL.cen)*dxL/dxT + frT.cen;     % the LO's chief pixel on the test grid
dch = chL - frT.chief;
say('Chief-ray pixel [col row]: test [%.3f %.3f], LO [%.3f %.3f] -> LO - test = [%+.3f %+.3f] px\n', ...
    frT.chief, chL, dch);
magT = sqrt(abs(det(frT.Lm)));  magL = sqrt(abs(det(frL.Lm)));
say('Magnification (object mm per detector mm): DM->camera %.6f, seat->camera %.6f (ratio %.6f)\n', ...
    magT, magL, magL/magT);
say('Footprints (camera position, test pixels): test pupil image centroid [%.3f %.3f] px, radius %.2f px; LO [%.3f %.3f] px, radius %.2f px\n', ...
    cT, rT, cL, rL);
say(['            centroid LO - test = [%+.3f %+.3f] px; LO/test radius %.4f -- the LO overfills by %.1f px, so an\n' ...
     '            offset under that is a fill question, not a registration one (the test pupil is the DM STOP''s image)\n'], ...
    cL - cT, rL/rT, rL - rT);

% ---- the LO over the test pupil ---------------------------------------
w = Il(mskT);
cover = mean(w > 0.5*max(Il(:)));
say('LO over the test pupil (%d px): covered %.4f (I_LO > 0.5 max), I_LO rms/mean %.4f, min/max %.4f\n', ...
    nnz(mskT), cover, std(w)/mean(w), min(w)/max(w));
lam_nm = P.LAM*1e6;
% (no 'LO wavefront' row: the field at the camera carries the image plane's
% many-wave curvature, so a single arm's phase there is fully wrapped -- its rms
% reads lambda/sqrt(12) = 183 nm whatever the optics.  What the LO's wavefront
% does to the reading is the NULL below: LO minus test, on one grid.)

% ---- the null on the flat DM: the interference phase itself ------------
X = sum(Et .* conj(Elo), 3);
ph0 = angle(X .* conj(mean(X(mskT))));         % piston removed
z0 = lowfit_(ph0, mskT);
g  = 2*abs(X)./(It + Il);
say('Null (flat DM): interference phase %.3f nm rms of SURFACE after piston (h = phi lambda/4pi); tilt %.3f, defocus %.3f, rest %.3f\n', ...
    z0.rms_p*lam_nm/(4*pi), z0.tilt*lam_nm/(4*pi), z0.defocus*lam_nm/(4*pi), z0.rest*lam_nm/(4*pi));
say('Fringe contrast g over the test pupil: median %.4f, min %.4f (lo_ratio at the camera: median I_LO/I_t %.4f)\n', ...
    median(g(mskT)), min(g(mskT)), median(Il(mskT)./It(mskT)));
say('LO power through the analyzer at the solved launch: %.4f of the LO at the camera without it\n', lo_pass);

% ---- the verdict (BRIEF_to_cophi sec. 1: to a pixel) --------------------
% the verdict: the chief rays agree to a pixel at the camera, the LO's footprint
% holds the test pupil inside its overfill and covers it, and the null on the
% flat DM is the surface floor the engine's own common path leaves (< 1 nm)
nul_nm = z0.rms_p*lam_nm/(4*pi);
ok = max(abs(dch)) < 1 && norm(cL - cT) < (rL - rT) && cover > 0.99 && nul_nm < 1;
say(['\nREGISTRATION %s: chief %.3f px, LO footprint offset %.3f px (inside its %.1f px overfill), LO coverage %.4f,\n' ...
     '  null %.3f nm (spec: chief < 1 px, offset < overfill, coverage > 0.99, null < 1 nm)\n'], ...
    iff_(ok, 'PASS', 'FAIL'), max(abs(dch)), norm(cL - cT), rL - rT, cover, nul_nm);

R = struct('th_pass', th_pass, 'T_analyzer', Tth, 'dxT', dxT, 'dxL', dxL, 'N', N, ...
    'chief_T', frT.chief, 'chief_L', frL.chief, 'magT', magT, 'magL', magL, ...
    'cen_T', cT, 'cen_L', cL, 'rad_T', rT, 'rad_L', rL, 'cover', cover, ...
    'lo_phase_sign', sg_lo, 'null', z0, 'g_med', median(g(mskT)), 'lo_pass', lo_pass, 'lo_launch', lo_launch, 'test_launch', test_launch, ...
    'ok', ok, 'frT', frT, 'frL', frL);
R.Et = single(Et);  R.Elo = single(Elo);  R.Elo_native = single(Elo_native);  R.mskT = mskT;  R.camT = camT;  R.camL = camL;
% ---- the reading context (tg96_run's arm_setup_ contract) -----------------
C = struct('N_G', P.grid.N_G, 'DX_G', P.grid.DX_G, 'LAM', P.LAM, 'QWP', 0, 'THETAS', P.THETAS, ...
    'NACT', P.dm(1).nact, 'PITCH', P.dm(1).pitch, 'rxT', rxT, 'rxR', rxL, 'reading', 'H');
C.AT = struct('rx', rxT, 'b', G.bt, 'iTO', G.T.iTO, 'iDET', iDET);
r = 1;  if isfield(P, 'cophi') && isfield(P.cophi, 'lo_ratio') && ~isempty(P.cophi.lo_ratio), r = P.cophi.lo_ratio; end
Elo_r = Elo * sqrt(r / median(Il(mskT)./It(mskT)));     % LO power at the camera = lo_ratio x the test arm's (median over its pupil)
lo_mode = 'placed';  if isfield(P.cophi, 'diag_lo') && ~isempty(P.cophi.diag_lo), lo_mode = P.cophi.diag_lo; end
if strcmp(lo_mode, 'self')
    % DIAGNOSTIC (gain probe #2): the LO := the test arm's OWN flat-DM field, on
    % its own pixels (no placement, no interpolation) -- the engine-truth
    % interferometer through the identical estimator
    Elo_r = Et * sqrt(r);
    say('DIAGNOSTIC: LO = the test arm''s own flat-DM field (diag_lo ''self'')\n');
end
C.Elo = Elo_r;
% the mask: the record's rule is the INTERFERENCE frame's support, i.e. the wider
% (reference / LO) beam's footprint -- dmg_lit reads the lit set from it and the
% aperture cap (place.lit_margin_mm) then trims it to the actuators the TEST beam
% reaches.  The placed LO lives only on the test arm's pixels (no ray of the test
% arm lands outside its pupil), so the LO's footprint is carried as the disc it
% occupies on the camera, in test pixels (the bench stage's registration numbers).
[xx, yy] = meshgrid(1:N);
C.msk = hypot(xx - cL(1), yy - cL(2)) <= rL;
if isfield(P.cophi, 'diag_mask') && strcmp(P.cophi.diag_mask, 'test')
    % DIAGNOSTIC (gain probe #4): fit on the TEST beam's support only (pair with
    % place.lit_force so the lit set -- and the surface -- stays the record's)
    C.msk = mskT;
    say('DIAGNOSTIC: mask = the test beam''s support (diag_mask ''test'', %d px)\n', nnz(mskT));
end
opt = struct('N', 4);
if isfield(P, 'cophi'), opt.N = P.cophi.N; end
C.phasef = @(M) cophi_phase_(rxT, G.bt, th_pass, P, test_launch, G.T.iTO, M, Elo_r, opt);
C.fieldf = @(M) arm_field_(rxT, G.bt, th_pass, P, test_launch, G.T.iTO, M);   % the test arm's field (probe #1)
C.framesf = @(M, eo) cophi_frames_(C.fieldf(M), Elo_r, opt, eo);        % the N heterodyne frames (+ electronics terms eo)
C.opdf = @(M) opd_on_pixels_(rxT, G.bt, th_pass, P, test_launch, G.T.iTO, M, iDET, size(Et, 1));   % the test arm's OPD (mm) per pixel: the capture stage's TRUTH
C.rT = rT;  C.mskT = mskT;  C.T_test = Tmax;  C.T_lo = sum(abs(Elo(:)).^2);   % pupil radius (test px); powers at the camera (source flux 1)
C.p_null = C.phasef(zeros(P.grid.N_G));
C.measf  = @(M) angle(exp(1i*(C.phasef(M) - C.p_null))) * P.LAM/(4*pi);
C.reg = R;
Il_r = sum(abs(Elo_r).^2, 3);  Xr = sum(Et .* conj(Elo_r), 3);  gmap = 2*abs(Xr) ./ max(It + Il_r, eps);
C.N_ = opt.N;  C.gb_ = median(gmap(mskT)) * sin(pi/opt.N)/(pi/opt.N);   % the BUCKET contrast g' = g sinc(pi/N) (0.9003 g at N = 4)
C.magT = R.magT;  C.frT = R.frT;  C.dxT = R.dxT;   % camera mm -> DM mm, the test grid (the LO drift's phase)
C.read_frames = @(Fr, al) cophi_read_(Fr, al, mskT, P);   % the reading of an arbitrary frame set (the gate's red legs)
C.noise_hook = @(S) cophi_noise_(C, P, S);   % LAST: the closure captures C as it stands
C.loop_hook  = @(S) cophi_loop_(C, P, S);
C.test_launch = test_launch;  C.th_pass = th_pass;  C.lo_launch = lo_launch;
end

% ---------------------------------------------------------------------
function W = opd_on_pixels_(rx, b, th, P, launch, iTO, M, iDET, N)
%OPD_ON_PIXELS_  the test arm's OPD (mm) on its own detector pixels (cam_map_'s
%   index map), unwrapped by construction -- a ray's path, not a phase.
    arm_field_(rx, b, th, P, launch, iTO, M);
    Cm = cam_map_(iDET, N);  W = nan(N);  W(Cm.pix) = Cm.W;
end

function [Fr, alpha] = cophi_frames_(Et, Elo, opt, eo)
%COPHI_FRAMES_  dmg_cophi_frames + the ELECTRONICS terms of the campaign brief
%   sec. 9.1 (Dave 2026-10-11): eo.jitter (rad of beat phase per frame: the AOM
%   drivers' DDS phase noise and the camera trigger -- dmg_cophi_frames' own
%   knob), eo.walk (rad/frame differential fiber phase -- its own knob), and
%   eo.ripple (rms fractional AOM drive AMPLITUDE ripple per frame: the LO's
%   amplitude scaled by (1 + e_k) in frame k -- a contrast AND a DC modulation,
%   built here frame by frame so dmg_cophi_* is unchanged).  eo.seed sets the
%   draws; the same seed gives the same jitter/walk in every frame's call.
    if nargin < 4 || isempty(eo), eo = struct(); end
    o = opt;
    for f = {'jitter', 'walk', 'seed'}
        if isfield(eo, f{1}), o.(f{1}) = eo.(f{1}); end
    end
    rip = 0;  if isfield(eo, 'ripple'), rip = eo.ripple; end
    if rip == 0
        [Fr, alpha] = dmg_cophi_frames(Et, Elo, o);  return
    end
    sd = 0;  if isfield(eo, 'seed'), sd = eo.seed; end
    rs = RandStream('mt19937ar', 'Seed', 1e6 + sd);  ek = rip * randn(rs, 1, o.N);
    Fr = [];
    for k = 1:o.N
        [Fk, alpha] = dmg_cophi_frames(Et, Elo * (1 + ek(k)), o);
        if isempty(Fr), Fr = zeros(size(Fk)); end
        Fr(:,:,k) = Fk(:,:,k);
    end
end

function NZ = cophi_noise_(C, P, S)
%COPHI_NOISE_  the noise stage's rows, on the battery's own estimator (tg96_run
%   stage_deck_ calls it with S = .est .litmask .ic .msk .dmap .A0 .say .LAM).
%   PER CAMERA (P.cophi.cameras, by name; Dave's ruling 3: read noise in every
%   row): the camera is mapped onto the model grid by BINNING -- k = camera
%   pixels across the pupil / model pixels across the pupil, so a model pixel
%   holds k^2 camera pixels: well x k^2, read noise x k (quadrature) -- and the
%   frame mean at the brightest pixel sits at fill x well.  The single 10 nm
%   differential at the hold-out actuator (the S5 form of the record) is read
%   NR times with independent shot + read noise in both readings; sigma of ONE
%   reading = sigma(differential)/sqrt(2).  Cadence: the camera at its full-frame
%   rate, a reading every N frames (the beat = rate/N); seconds for 1 pm =
%   (sigma/1 pm)^2 readings averaged.  The per-pixel phase noise is checked
%   against the formula tCophi pinned, sigma_phi^2 = 2 (I0 + r^2)/(N I0^2 g'^2).
%   ELECTRONICS (campaign sec. 9.1): each term alone, noise-free, the reading's
%   error at the hold-out and over the unpoked lit set.
    say = S.say;  N = C.N_;  ic = S.ic;  lit = S.litmask;  lam = S.LAM;
    nr = 6;  if isfield(P.cophi, 'noise_nreal'), nr = P.cophi.noise_nreal; end
    pf = 0.95;  if isfield(P.cophi, 'pupil_fill'), pf = P.cophi.pupil_fill; end
    dev = zeros(size(S.A0));  dev(ic(1), ic(2)) = P.battery.dev_single;
    M0 = S.dmap(S.A0);  M1 = S.dmap(S.A0 + dev);
    E0f = C.fieldf(M0);  E1f = C.fieldf(M1);          % trace the two DM states ONCE
    fr = @(E, eo) cophi_frames_(E, C.Elo, struct('N', C.N_), eo);
    [F0, al] = fr(E0f, struct());  F1 = fr(E1f, struct());
    msk = S.msk;  un = lit;  un(ic(1), ic(2)) = false;
    rd = @(F) dmg_cophi_recon(F, al);
    ad = @(p1, p0) S.est(meanref_(angle(exp(1i*(p1 - p0))) * lam/(4*pi), msk));
    % the CONTRAST GATE (brief sec. 3: no fringe -> refused by name, not read):
    % a pixel whose fringe amplitude / dc is below gfrac x the pupil median in
    % EITHER reading reads 0.  Without it, the LO-only annulus inside the mask
    % (no test light) reads a uniformly random phase under noise, and the
    % estimator's mean-reference term (sum of h over the mask) carries it into
    % every actuator as a seed-dependent piston-like pattern (measured
    % 2026-10-11: floors moving 40 % between two otherwise identical benches).
    gfrac = 0.2;  if isfield(P.cophi, 'gate_frac'), gfrac = P.cophi.gate_frac; end
    adg = @(G1, G0) gated_(G1, G0, al, gfrac, C.mskT, lam, msk, S.est);
    a_clean = ad(rd(F1), rd(F0));
    nmc = fieldnames(P.cophi.cameras);
    say('\n---- noise: the H reading on each camera (single %g nm at (%d,%d), matrix on the surface; %d realizations; model pupil %.0f px across) ----\n', ...
        P.battery.dev_single*1e6, ic(1), ic(2), nr, 2*C.rT);
    say('  %-12s %6s %8s %8s %6s | %10s %9s %9s | %9s %8s %9s %10s\n', 'camera', 'px/pup', 'well', 'read e-', 'fps', ...
        'e-/frame', 'ph/read', 'sigma_phi', 'pm@hold', 'pm/read', 'reads/s', 's per pm');
    rows = {};
    for c = 1:numel(nmc)
        cam = P.cophi.cameras.(nmc{c});
        k = pf*cam.npix / (2*C.rT);
        ce = struct('well', cam.well*k^2, 'read', cam.read*k, 'fill', P.cophi.fill);
        pk = zeros(nr, 1);  fl = zeros(nr, 1);  fm = zeros(nr, 1);  sph = zeros(nr, 1);
        for r = 1:nr
            [G0, sc] = dmg_cophi_camera(F0, ce, 1000*c + 2*r);
            G1 = dmg_cophi_camera(F1, ce, 1000*c + 2*r + 1);
            p1 = rd(G1);  a = adg(G1, G0) - a_clean;
            a = a - mean(a(lit));              % remove the reading's PISTON (unobservable, inert on a DM)
            pk(r) = a(ic(1), ic(2));  fl(r) = std(a(un));  fm(r) = 1.4826*median(abs(a(un) - median(a(un))));
            e1 = angle(exp(1i*(p1 - rd(F1))));  sph(r) = sqrt(mean(e1(C.mskT).^2));
            if c == 1 && isfield(P.cophi, 'noise_map') && P.cophi.noise_map, AM(:, :, r) = a; end   % diagnostic maps (piston removed)
        end
        sig = std(pk)/sqrt(2) * 1e9;  flr = median(fm)/sqrt(2) * 1e9;  % pm, ONE reading: hold-out (piston removed) / ROBUST floor
        flr_std = mean(fl)/sqrt(2)*1e9;  frng = [min(fl) max(fl)]/sqrt(2)*1e9;
        % the HEADLINE is the floor (rms over every unpoked lit actuator, ~6900 x
        % nr samples); the hold-out sigma from nr draws is the check (+-30 % at 6)
        epf = P.cophi.fill * cam.well;                                  % e- per CAMERA pixel per frame (brightest)
        phr = N * sum(sum(mean(F0, 3))) * sc;                           % e- per reading, all pixels, both arms
        % the formula, at the pupil's median model pixel
        I0 = mean(F0, 3) * sc;  i0 = median(I0(C.mskT));
        g = C.gb_;  sform = sqrt(2*(i0 + ce.read^2) / (N * i0^2 * g^2));
        rate = cam.rate / N;  spm = max(flr^2, 1) / rate;      % seconds for 1 pm, from the FLOOR
        say('  %-12s %6.0f %8.3g %8.3g %6.0f | %10.3g %9.3g %9.3g | %9.3g %8.3g %9.3g %10.3g\n', nmc{c}, pf*cam.npix, cam.well, cam.read, cam.rate, ...
            epf, phr, mean(sph), sig, flr, rate, spm);
        say('  %-12s   sigma_phi measured / formula = %.3f (median pupil pixel, I0 %.3g e- per model pixel per frame, g'' %.4f)\n', '', mean(sph)/sform, i0, g);
        say('  %-12s   floor: robust (1.48 MAD, median of draws) %.3g pm; plain std %.3g pm, range over draws %.3g-%.3g pm\n', '', flr, flr_std, frng);
        rows(end+1, :) = {nmc{c}, cam.name, k, epf, phr, mean(sph), sform, sig, flr, rate, spm}; %#ok<AGROW>
    end
    % ---- electronics, EXACT expectation (2026-10-12): each term is linear and
    % enters as a per-FRAME offset -- a phase offset of the interference (beat /
    % trigger jitter: iid per frame; differential fiber walk: cumulative over the
    % scan, theta_k = sum_{j<k} g_j) or an amplitude factor on the LO (AOM drive
    % ripple: iid).  So each frame's response pattern through the estimator is
    % computed ONCE (a small offset, divided out), and the piston-free error of one
    % reading follows from the term's covariance: E[var_lit] = sum over independent
    % draws of var_lit(their pattern).  The Monte Carlo rows below are kept for the
    % record but are UNDER-SAMPLED: a reading's whole walk is three random numbers,
    % and twelve-draw means span +-50 % (measured; the first noise records' walk
    % row read 2.9x low on its seed set).
    Np = C.N_;  dlt = 1e-3;  Qp = cell(1, Np);  Qa = cell(1, Np);
    for kf = 1:Np
        th = zeros(1, Np);  th(kf) = dlt;
        Qp{kf} = (det_(C, E0f, th, zeros(1, Np), rd, ad, F0) ) / dlt;     % per unit phase in frame kf
        ep = zeros(1, Np);  ep(kf) = dlt;
        Qa{kf} = (det_(C, E0f, zeros(1, Np), ep, rd, ad, F0) ) / dlt;     % per unit amplitude in frame kf
    end
    pfv = @(Q) var(Q(un) - mean(Q(un)), 1);
    Vj = sum(cellfun(pfv, Qp));                                  % jitter: iid phase per frame
    Vr = sum(cellfun(pfv, Qa));                                  % ripple: iid amplitude per frame
    Vw = 0;  for j = 1:Np-1, Vw = Vw + pfv(sum(cat(3, Qp{j+1:Np}), 3)); end   % walk: increment j shifts frames j+1..N
    ex = struct('jitter', sqrt(Vj), 'ripple', sqrt(Vr), 'walk', sqrt(Vw));    % mm per unit term, ONE reading
    say('\n  electronics, EXACT expectation per reading (piston-free rms over unpoked lit, per UNIT term): jitter %.4g pm/rad, ripple %.4g pm per unit, walk %.4g pm per rad/frame\n', ...
        1e9*ex.jitter, 1e9*ex.ripple, 1e9*ex.walk);
    ph_floor = NaN;  if ~isempty(rows), ph_floor = rows{1, 9}; end                % the first camera's photon floor, pm
    for f = {'jitter', 'ripple', 'walk'}
        say('    %-6s budget (= the %s camera''s photon floor %.3g pm per reading): %.3g %s\n', f{1}, iff_(isempty(rows), '-', rows{1,1}), ph_floor, ...
            ph_floor/(1e9*ex.(f{1})), iff_(strcmp(f{1}, 'ripple'), 'rms fraction per frame', 'rad per frame'));
    end
    % ---- electronics: each term alone, noise-free (Monte Carlo, under-sampled; see above) ---
    el = struct('jitter', [1e-3 1e-2 3e-2], 'ripple', [1e-4 1e-3 1e-2], 'walk', [1e-2 5e-2]);
    if isfield(P.cophi, 'elec') && ~isempty(P.cophi.elec), el = P.cophi.elec; end
    say('\n  electronics (campaign sec. 9.1), each term ALONE, noise-free; the error of ONE reading (rms over %d draws):\n', nr);
    say('  %-34s %10s %12s\n', 'term', 'pm @ (ic)', 'pm over lit');
    erow = {};
    for f = fieldnames(el).'
        for v = el.(f{1})
            e1 = zeros(nr, 1);  e2 = zeros(nr, 1);
            for r = 1:nr
                eo = struct('seed', 7000 + r);  eo.(f{1}) = v;
                H1 = fr(E1f, eo);  eo.seed = 8000 + r;  H0 = fr(E0f, eo);
                a = ad(rd(H1), rd(H0)) - a_clean;
                e1(r) = a(ic(1), ic(2));  e2(r) = std(a(un));
            end
            lab = sprintf('%s %g %s', f{1}, v, iff_(strcmp(f{1}, 'ripple'), '(rms fraction/frame)', '(rad/frame)'));
            say('  %-34s %10.3g %12.3g\n', lab, sqrt(mean(e1.^2))/sqrt(2)*1e9, mean(e2)/sqrt(2)*1e9);
            erow(end+1, :) = {f{1}, v, sqrt(mean(e1.^2))/sqrt(2)*1e9, mean(e2)/sqrt(2)*1e9}; %#ok<AGROW>
        end
    end
    if exist('AM', 'var')
        sm = std(AM, 0, 3)/sqrt(2)*1e9;  v = sm(un);             % per-actuator sigma of ONE reading, pm
        [ri, ci] = find(un);  rr = hypot(ri - mean(ri), ci - mean(ci));  rmax = max(rr);
        say('\n  NOISE MAP (%s, %d draws): per-actuator sigma of one reading over the unpoked lit set: median %.3g, mean %.3g, rms %.3g, p90 %.3g, p99 %.3g, max %.3g pm; hold-out %.3g pm (percentile %.0f)\n', ...
            nmc{1}, nr, median(v), mean(v), sqrt(mean(v.^2)), prctile(v, 90), prctile(v, 99), max(v), sm(ic(1), ic(2)), 100*mean(v < sm(ic(1), ic(2))));
        ed = 0:0.1:1;  say('  radial (r / r_max of the lit set):');
        for q = 1:numel(ed)-1
            in = rr/rmax >= ed(q) & rr/rmax < ed(q+1) + (q == numel(ed)-1);
            say(' %.1f-%.1f %.3g', ed(q), ed(q+1), median(v(in)));
        end
        say('\n  hold-out at r/r_max = %.2f\n', hypot(ic(1) - mean(ri), ic(2) - mean(ci))/rmax);
        NMAP = sm;
    else
        NMAP = [];
    end
    % the LOOP's form of an electronics term (2026-10-11 cross-check): ONE reading of
    % the set point with the term against the NOISELESS reading of the same surface
    % (dmg_loop's e = a - a0), vs the differential form above (two readings, /sqrt 2)
    if isfield(P.cophi, 'elec_single') && ~isempty(P.cophi.elec_single)
        say('\n  the loop''s form: ONE reading with the term vs the noiseless reading of the same surface (piston-free std over unpoked lit; %d draws):\n', nr);
        es = P.cophi.elec_single;
        for f = fieldnames(es).'
            for v = es.(f{1})
                e2 = zeros(nr, 1);  e2d = zeros(nr, 1);
                for r = 1:nr
                    eo = struct('seed', 9000 + r);  eo.(f{1}) = v;
                    H0w = fr(E0f, eo);
                    a = S.est(meanref_(angle(exp(1i*(rd(H0w) - rd(F0)))) * lam/(4*pi), msk));
                    e2(r) = std(a(un));
                    eo.seed = 9500 + r;  H1w = fr(E1f, eo);
                    ad2 = ad(rd(H1w), rd(H0w)) - a_clean;  e2d(r) = std(ad2(un))/sqrt(2);
                end
                say('    %-8s %-9g single-vs-noiseless %8.3g pm (rms of draws %8.3g) | differential/sqrt2 %8.3g pm (rms %8.3g)\n', ...
                    f{1}, v, mean(e2)*1e9, sqrt(mean(e2.^2))*1e9, mean(e2d)*1e9, sqrt(mean(e2d.^2))*1e9);
            end
        end
    end
    % ---- the gate's LO-tilt probe (brief sec. 3): a 1 nrad tilt of the LO (DM
    % plane) read through the matrix -- how much of it the estimate carries as a
    % DM tilt plane (the 'absorption'), and what is left after the plane
    lot = struct();
    if isfield(P.cophi, 'gate_probes') && P.cophi.gate_probes
        [xx, yy] = meshgrid(1:size(msk,1));
        xd = (xx - C.frT.cen) * C.dxT * C.magT;
        th = 1e-9;  ElT = C.Elo .* exp(1i*2*pi/lam*th*xd);
        [Ft, al2] = dmg_cophi_frames(E0f, ElT, struct('N', C.N_));
        a = S.est(meanref_(angle(exp(1i*(rd(Ft) - rd(F0)))) * lam/(4*pi), msk));
        % compare like with like: the estimate is in actuator COMMANDS, the LO tilt a
        % SURFACE slope -- a smooth surface plane needs commands smaller by the sum
        % of the overlapping influence functions -- so build the DM SURFACE from the
        % estimated commands and fit ITS slope (over the lit disc) against theta/2
        hs = S.dmap(a);  ng = size(hs, 1);
        xg = ((1:ng) - (ng+1)/2) * P.grid.DX_G;  [gx, gy] = meshgrid(xg);
        inn = hypot(gx, gy) <= P.capture.lit_r - 2;            % inside the control set, clear of its edge
        B = [ones(nnz(inn),1) gx(inn) gy(inn)];  cf = B \ hs(inn);  res = hs(inn) - B*cf;
        absb = hypot(cf(2), cf(3)) / (th/2);
        lat = ((1:size(lit,1)) - (size(lit,1)+1)/2) * P.dm(1).pitch;  [ax, ay] = meshgrid(lat);
        Bc = [ones(nnz(lit),1) ax(lit) ay(lit)];  cc = Bc \ a(lit);
        lot = struct('theta', th, 'slope', cf(2:3), 'absorption', absb, 'resid_pm', 1e9*std(res), 'plane_pm', 1e9*std(B(:,2:3)*cf(2:3)), ...
            'cmd_slope_ratio', hypot(cc(2), cc(3))/(th/2));
        say(['\n  LO-TILT probe: %.1g rad of LO tilt reads as a DM SURFACE tilt plane of %.4g x theta/2 (the absorption), plane %.3g pm rms,\n' ...
             '    %.3g pm rms left after the plane; in COMMAND units the plane is %.4g x theta/2 (the influence functions'' overlap)\n'], ...
            th, absb, lot.plane_pm, lot.resid_pm, lot.cmd_slope_ratio);
    end
    NZ = struct('cams', {rows}, 'elec', {erow}, 'elec_exact', ex, 'lo_tilt', lot, 'noise_map', NMAP, 'T_test', C.T_test, 'T_lo', C.T_lo, 'nreal', nr, 'pupil_fill', pf);
end

function a = gated_(G1, G0, al, gfrac, mskT, lam, msk, est)
%GATED_  the differential of two noisy readings with the contrast gate.
    [p1, a1, d1] = dmg_cophi_recon(G1, al);  [p0, a0, d0] = dmg_cophi_recon(G0, al);
    c1 = a1 ./ max(d1, eps);  c0 = a0 ./ max(d0, eps);
    ok = c1 > gfrac*median(c1(mskT)) & c0 > gfrac*median(c0(mskT));
    h = angle(exp(1i*(p1 - p0))) * lam/(4*pi);  h(~ok) = 0;
    a = est(meanref_(h, msk));
end

function LP = cophi_loop_(C, P, S)
%COPHI_LOOP_  the hold servo on the H reading: the shared dmg_loop, the record's
%   loop (P.loop: g, K, seed, walk / thermal / cam), each cycle averaging M
%   readings on a named camera (equivalent camera: well x M, read x sqrt M; the
%   intra-scan fiber walk and beat jitter enter scaled 1/sqrt M -- each reading's
%   error is linear in them and independent), the estimator one-step AND
%   converged (S.est30 / S.est30c at the cycle's floor).  Rows: noise only, walk,
%   thermal, cam (camera offset walk), lo (the LO's tilt / defocus per cycle, a
%   phase on the LO at the camera), fiber walk at the budget and 3x, and one
%   cycle of SENSOR DELAY (dmg_loop o.delay).  Units: mm in, pm out.
    say = S.say;  lam = S.LAM;  msk = S.msk;  lit = S.litmask;  N = size(msk, 1);
    LPc = P.cophi.loop;
    cnt = cophi_counter();
    % camera-plane coordinates of the test grid, in DM mm (for the LO drift phase)
    [xx, yy] = meshgrid(1:N);
    xd = (xx - C.frT.cen) * C.dxT * C.magT;  yd = (yy - C.frT.cen) * C.dxT * C.magT;
    rr = hypot(xd, yd);  r0 = max(rr(C.mskT));  zdef = sqrt(3)*(2*(rr/r0).^2 - 1);
    frq = @(E, Elo, eo) cophi_frames_(E, Elo, struct('N', C.N_), eo);
    rd  = @(F) dmg_cophi_recon(F, alpha_(C.N_));
    base = struct('A0', S.A0, 'g', P.loop.g, 'K', P.loop.K, 'seed', P.loop.seed, 'ref', 'noiseless', 'rmax', P.loop.rmax);
    rows = {};
    say('\n---- loop: the hold servo on the H reading (g %.2f, K %d, seed %d; drifts per CYCLE as the record) ----\n', base.g, base.K, base.seed);
    for ic = 1:numel(LPc.cams)
        cname = LPc.cams{ic};  cam = P.cophi.cameras.(cname);  M = LPc.M.(cname);
        pf = 0.95;  if isfield(P.cophi, 'pupil_fill'), pf = P.cophi.pupil_fill; end
        k = pf*cam.npix / (2*C.rT);
        ce = struct('well', cam.well*k^2*M, 'read', cam.read*k*sqrt(M), 'fill', P.cophi.fill);
        tcyc = M * C.N_ / cam.rate;
        say('  camera %s: %d readings per cycle -> a cycle every %.2f s (%d frames at %g fps)\n', cname, M, tcyc, M*C.N_, cam.rate);
        for spec = LPc.rows
            sp = spec{1};  o = base;  eo = struct();  lo = struct('tilt', 0, 'defocus', 0);
            switch sp
                case 'none'
                case 'walk',    o.drift = struct('kind', 'walk', 'sigma', P.loop.walk_sigma);
                case 'thermal', o.drift = struct('kind', 'thermal', 'rate', P.loop.thermal_rate);
                case 'cam',     o.cam = struct('walk', P.loop.cam_walk, 'intra', P.loop.cam_intra);
                case 'lo',      lo = LPc.lo;
                case 'fiber1',  eo.walk = LPc.fiber;          % the budget, rad/frame
                case 'fiber3',  eo.walk = 3*LPc.fiber;
                case 'delay',   o.delay = 1;
                otherwise, error('cophi_loop_: unknown row %s', sp);
            end
            if isfield(eo, 'walk'), eo.walk = eo.walk/sqrt(M); end
            for em = {'onestep', 'converged'}
                cnt.n = 0;
                ins = struct('lit', lit, 'npix', N);
                ins.measure = @(cmd, varargin) meas_(C, S, cmd, cnt, lo, lam, xd, zdef);
                ins.noisy   = @(F, nph, seed, varargin) noisy_(F, nph, seed, ce, eo, frq, C, P, varargin{:});
                ins.diff    = @(F1, F0) diff_(F1, F0, frq, rd, C, lam, msk);
                if strcmp(em{1}, 'onestep'), ins.est = S.est30;
                else, ins.est = @(h) conv_est_(S, h, LPc.conv_tol); end
                o.nph = 1;                                      % finite: the camera noise is on
                L = dmg_loop(ins, o);
                rows(end+1, :) = {cname, sp, em{1}, L.ss*1e9, L.bias*1e9, L.sig_n*1e9, L.diverged, tcyc, ...
                    L.ss_pf*1e9, L.bias_pf*1e9, L.sig_n_pf*1e9}; %#ok<AGROW>
                say('  %-11s %-8s %-9s ss %7.2f pm  bias %6.2f pm  sig_n %7.2f pm | piston-free: ss %7.2f  bias %6.2f  sig_n %7.2f pm%s\n', ...
                    cname, sp, em{1}, L.ss*1e9, L.bias*1e9, L.sig_n*1e9, L.ss_pf*1e9, L.bias_pf*1e9, L.sig_n_pf*1e9, iff_(L.diverged, '  DIVERGED', ''));
            end
        end
    end
    LP = struct('rows', {rows});
end

function F = meas_(C, S, cmd, cnt, lo, lam, xd, zdef)
%MEAS_  the test field at the DM command (traced) + the LO as it stands at this
%   measurement (the n-th: the reference is n = 0): tilt (rad, DM-plane) and
%   defocus (nm rms OPD) advance per measurement.
    n = cnt.n;  cnt.n = cnt.n + 1;
    opd = n*(lo.tilt*xd + lo.defocus*1e-6*zdef);       % mm
    F = struct('E', C.fieldf(S.dmap(cmd)), 'Elo', C.Elo .* exp(1i*2*pi/lam*opd));
end

function G = noisy_(F, nph, seed, ce, eo, frq, C, P, cam)
%NOISY_  the frames of one cycle on the camera (+ the electronics terms eo and
%   the camera offset walk); nph = Inf -> the noiseless frames.
    e = eo;  e.seed = seed;
    G = frq(F.E, F.Elo, e);
    if ~isfinite(nph), return; end
    m = mean(G, 3);  sc = ce.fill * ce.well / max(m(:));
    Fe = G * sc;
    if nargin >= 9 && ~isempty(cam)                     % dmg_loop's offset o + ramp d over the scan, 'rel' units
        unit = mean(m(C.mskT)) * sc;  nf = size(Fe, 3);
        for j = 1:nf, Fe(:,:,j) = Fe(:,:,j) + unit*(cam.o + (j-1)/max(nf-1,1)*cam.d); end
    end
    rs = RandStream('mt19937ar', 'Seed', 3e5 + seed);
    Fe = Fe + sqrt(max(Fe, 0)) .* randn(rs, size(Fe)) + ce.read * randn(rs, size(Fe));
    G = Fe / sc;
end

function h = diff_(F1, F0, frq, rd, C, lam, msk)
%DIFF_  the H reading's wrapped differential (contrast-gated), height in mm.
    if isstruct(F1), F1 = frq(F1.E, F1.Elo, struct()); end
    if isstruct(F0), F0 = frq(F0.E, F0.Elo, struct()); end
    [p1, a1, d1] = rd(F1);  [p0, a0, d0] = rd(F0);
    c1 = a1 ./ max(d1, eps);  c0 = a0 ./ max(d0, eps);
    ok = c1 > 0.2*median(c1(C.mskT)) & c0 > 0.2*median(c0(C.mskT));
    h = angle(exp(1i*(p1 - p0))) * lam/(4*pi);  h(~ok) = 0;
    h = meanref_(h, msk);
end

function a = conv_est_(S, h, tol)
    [a, ~] = S.est30c_tol(h, tol);
end

function al = alpha_(N)
    D = 2*pi/N;  al = (0:N-1)*D + D/2;
end

function a = det_(C, E, th, ep, rd, ad, F0)
%DET_  the reading's actuator error for DETERMINISTIC per-frame offsets: phase
%   th(k) on the interference and amplitude (1 + ep(k)) on the LO in frame k,
%   against the clean reading F0 of the same surface.
    Np = numel(th);  Fr = [];  al = [];
    for k = 1:Np
        [Fk, al] = dmg_cophi_frames(E, C.Elo * (1 + ep(k)) * exp(-1i*th(k)), struct('N', Np));
        if isempty(Fr), Fr = zeros(size(Fk)); end
        Fr(:,:,k) = Fk(:,:,k);
    end
    a = ad(rd(Fr), rd(F0));
end

function h = meanref_(h, msk)
    h(~msk) = 0;  h = h - median(h(msk));
end

% ---------------------------------------------------------------------
function phi = cophi_phase_(rx, b, th, P, launch, iTO, M, Elo, opt)
%COPHI_PHASE_  the heterodyne phase of the test arm with grid M on the DM.
    E = arm_field_(rx, b, th, P, launch, iTO, M);
    [Fr, alpha] = dmg_cophi_frames(E, Elo, opt);
    phi = cophi_read_(Fr, alpha, [], P);
end

function phi = cophi_read_(Fr, alpha, mskT, P)
%COPHI_READ_  the H reading of a frame set, REFUSED BY NAME when there is no
%   fringe (brief sec. 3): the median fringe amplitude / dc over the pupil
%   (mskT, or the lit frames when empty) below P.cophi.contrast_floor (default
%   0.05) -> error cophi:noFringe.  An incoherent LO, a crossed polarization, or
%   a dark camera (well 0: amp = dc = 0) all land here instead of reading a
%   plausible-looking zero phase.
    [phi, amp, dc] = dmg_cophi_recon(Fr, alpha);
    c = amp ./ max(dc, eps);
    if isempty(mskT), m = dc > 0.1*max(dc(:)); if ~any(m(:)), m = true(size(dc)); end, else, m = mskT; end
    flo = 0.05;  if isfield(P, 'cophi') && isfield(P.cophi, 'contrast_floor'), flo = P.cophi.contrast_floor; end
    cm = median(c(m));
    if ~(cm >= flo)
        error('cophi:noFringe', 'COPHI reading refused: no fringe (median amp/dc %.3g over the pupil < the contrast floor %.3g)', cm, flo);
    end
end

% ---------------------------------------------------------------------
function E = arm_field_(rx, b, an_deg, P, launch, iTO, M)
%ARM_FIELD_  One arm's three-plane field at the camera, COPHI settings.
%   an_deg NaN leaves the analyzer OPEN (axis along the field: measured
%   separately as the LO's pass fraction).  The arm QWPs and the output QWP
%   are zero-retardance windows; the input polarizer (test arm only) at the
%   record's 45 deg.
    ws = warning('off', 'mmacos:modesInForce');  cw = onCleanup(@() warning(ws));   % polarization stays ON across these loads, deliberately
    macos.load_rx(rx);
    if isfield(P.cophi, 'lam_trace') && ~isempty(P.cophi.lam_trace)
        macos.set_src_wvl(P.cophi.lam_trace);          % the capture stage's second colour (WaveUnits mm)
    end
    if nargin >= 7 && ~isempty(M)          % the DM state (the deck's own grid is the flat)
        macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
    end
    nm = {b.E.name};
    iPol = find(strcmp(nm, 'PolIn'), 1);
    if ~isempty(iPol), macos.polarizer(iPol, 'axis', lax_(b.E(iPol).psi, 45)); end
    for j = find(contains(nm, 'QWP'))
        macos.waveplate(j, 'axis', lax_(b.E(j).psi, 0), 'retardance', 0);
    end
    iAn = find(strcmp(nm, 'Analyzer'), 1);
    % the mirror rig's OAP coating, as the record applies it (tg96_run load_arm):
    % every element named L1 / L2 that is an OAP (the LO's own collimator is a
    % lens, LOColl, and is not coated)
    cs = oap_coat_(P);
    if ~isempty(cs)
        for j = find(strcmp(nm, 'L1') | strcmp(nm, 'L2'))
            macos.coating(j, 'index', cs.index, 'extinc', cs.extinc, 'thickness', cs.thickness);
        end
    end
    if nargin < 5 || isempty(launch), launch = [1 1]/sqrt(2); end   % the record's Ex = Ey
    if isfield(P.cophi, 'scalar') && P.cophi.scalar
        % DIAGNOSTIC (gain probe #3): the scalar trace -- no polarization, the
        % polarizing elements are then Reference surfaces; one field plane
        macos.polarization('off');  macos.vector_diffraction(false);
        E = macos.complex_field(numel(b.E));  return
    end
    macos.polarization('on', 'Ex', [real(launch(1)) imag(launch(1))], ...
                             'Ey', [real(launch(2)) imag(launch(2))]);
    macos.vector_diffraction(true);
    if isempty(iAn)
        % the PLAIN-SPLITTER train (bench.polarizing false: Feng's layout, no
        % polarizing optics) -- no analyzer to set; 'open' and any angle coincide
        E = field3_(b);
    elseif isnan(an_deg)
        % open: two orthogonal analyzer settings, powers added (coherence across
        % them does not matter for the POWER, which is all this is used for)
        macos.polarizer(iAn, 'axis', lax_(b.E(iAn).psi, 0));    E1 = field3_(b);
        macos.polarizer(iAn, 'axis', lax_(b.E(iAn).psi, 90));   E2 = field3_(b);
        E = cat(3, E1, E2);
    else
        macos.polarizer(iAn, 'axis', lax_(b.E(iAn).psi, an_deg));
        E = field3_(b);
    end
end
function cs = oap_coat_(P)
%OAP_COAT_  the record's OAP coating stack (tg96_run oap_coat_stack_), [] if none.
    cs = [];
    if ~strcmp(P.bench.optics, 'oap') || ~isfield(P.bench, 'coat_oap'), return; end
    switch P.bench.coat_oap
        case {'none', ''}, cs = [];
        otherwise,         cs = P.bench.(['coat_' P.bench.coat_oap]);
    end
end
function v = top_eig_(M)
%TOP_EIG_  the unit launch [Ex Ey] maximizing v M v' (M the 2x2 launch Gram), real-dominant.
    M = (M + M')/2;  [V, D] = eig(M);  [~, im] = max(real(diag(D)));  v = V(:, im);
    k = find(abs(v) == max(abs(v)), 1);  v = v * exp(-1i*angle(v(k)));  v = v(:).';
end
function C = cam_map_(n, N)
%CAM_MAP_  pixel -> camera-position map of the LOADED deck at its detector n:
%   the passing rays 2..end land on the OPD mask's pixels in column-major order
%   (OPDMat index + the centring offset in the N x N field grid).  uv in mm in
%   the detector's (perp(psi), psi x perp(psi)) frame about its vertex.
    s = macos.trace(n);  ri = macos.get_ray_info(s.nRays);  m = macos.opd_mask();
    ok = ri.ok_trace(:) & ri.ok_pass(:);  ok(1) = false;
    assert(nnz(ok) == nnz(m), 'cam_map_: %d passing rays vs %d mask pixels', nnz(ok), nnz(m));
    psi = macos.get_elt_psi(n);  u = macos.design.Bench.perp(psi);  v = cross(psi, u);
    uv = [u.'; v.'] * (ri.pos(:, ok) - macos.get_elt_vpt(n));
    ng = size(m, 1);  off = (N - ng + mod(ng, 2))/2;
    [i, j] = find(m);  pix = sub2ind([N N], i + off, j + off);
    W = macos.opd();
    C = struct('pix', pix, 'uv', uv, 'k', find(ok), 'W', W(m));
end
function [r, mx] = idx_diff_(A, B)
%IDX_DIFF_  rms / max camera-position difference of the same ray index in two decks (mm).
    [kk, ia, ib] = intersect(A.k, B.k);  %#ok<ASGLU>
    d = A.uv(:, ia) - B.uv(:, ib);  dd = hypot(d(1,:), d(2,:));
    r = sqrt(mean(dd.^2));  mx = max(dd);
end
function [Eo, sg, sp] = lo_on_test_(El, CL, CT, lam)
%LO_ON_TEST_  the LO field at the TEST arm's pixels, by camera position.
%   El(pix) = Q(pix) exp(i k W(pix)) with W the LO's per-pixel path (the phase
%   of its dominant component, unwrapped along the ray grid by the engine's
%   own OPD) and Q smooth; Q (each component) and W are interpolated
%   (scatteredInterpolant, natural) at the test rays' camera points.
    N = size(El, 1);  nc = size(El, 3);  k = 2*pi/lam;
    W = macos_opd_at_(CL, N);                          % mm, per LO pixel (from the last LO trace's OPD)
    % the SIGN of the field's path phase is MEASURED, not assumed: the scalar
    % field carries +k W, but a vector-mode field (RayE) on a plain trace-to-
    % detector train is CONJUGATE to the OPD (macos_f90/CLAUDE.md, Phase 3a,
    % "TRAIN-DEPENDENT").  The right sign leaves E exp(-i sg k W) flat in phase.
    [~, cd] = max(squeeze(sum(sum(abs(El).^2, 1), 2)));
    Ed = El(:,:,cd);  Ed = Ed(CL.pix);  sp = zeros(1, 2);  sgs = [1 -1];
    for t = 1:2
        q = Ed .* exp(-1i*sgs(t)*k*W(CL.pix));  q = q .* conj(sum(q))/abs(sum(q));
        sp(t) = std(angle(q));
    end
    [~, it] = min(sp);  sg = sgs(it);
    assert(sp(it) < 0.5 && sp(3-it) > 4*sp(it), ...
        'lo_on_test_: field phase is neither +kW nor -kW (residual phase rms %.3g / %.3g rad)', sp(1), sp(2));
    Eo = zeros(size(El));
    FW = scatteredInterpolant(CL.uv(1,:).', CL.uv(2,:).', W(CL.pix), 'natural', 'none');
    Wt = FW(CT.uv(1,:).', CT.uv(2,:).');
    for c = 1:nc
        Ec = El(:,:,c);  Q = Ec(CL.pix) .* exp(-1i*sg*k*W(CL.pix));
        FQ = scatteredInterpolant(CL.uv(1,:).', CL.uv(2,:).', Q, 'natural', 'none');
        q = FQ(CT.uv(1,:).', CT.uv(2,:).');
        e = q .* exp(1i*sg*k*Wt);  e(isnan(e)) = 0;
        Ec2 = zeros(N);  Ec2(CT.pix) = e;  Eo(:,:,c) = Ec2;
    end
end
function W = macos_opd_at_(C, N)
%MACOS_OPD_AT_  the per-pixel OPD (mm) stored with a camera map (see cam_map_).
    W = zeros(N);  W(C.pix) = C.W;
end
function E = field3_(b)
    iD = numel(b.E);
    E = cat(3, macos.complex_field(iD,'plane',1), macos.complex_field(iD,'plane',2), ...
               macos.complex_field(iD,'plane',3));
end
function a = lax_(psi, deg)
    u1 = macos.design.Bench.perp(psi(:));  u2 = cross(psi(:), u1);
    a = cosd(deg)*u1 + sind(deg)*u2;  a = a(:).';
end
function ph = phase_dom_(E)
%PHASE_DOM_  the phase of the field's dominant Cartesian component.
    [~, k] = max(squeeze(sum(sum(abs(E).^2, 1), 2)));
    ph = angle(E(:,:,k));
end
function z = lowfit_(ph, msk)
%LOWFIT_  piston / tilt / defocus fit of a phase map over msk (rad).
    [n1, n2] = size(ph);  [x, y] = meshgrid(1:n2, 1:n1);
    xm = x(msk);  ym = y(msk);  r0 = max(hypot(xm - mean(xm), ym - mean(ym)));
    u = (xm - mean(xm))/r0;  v = (ym - mean(ym))/r0;
    p = ph(msk);  p = angle(exp(1i*(p - angle(mean(exp(1i*p))))));   % piston out, then small
    B = [ones(size(u)) u v 2*(u.^2 + v.^2) - 1];
    cf = B \ p;  res = p - B*cf;
    z = struct('rms_p', std(p, 1), 'tilt', std(B(:,2:3)*cf(2:3), 1), ...
               'defocus', std(B(:,4)*cf(4), 1), 'rest', std(res, 1), 'coef', cf);
end
function [c, r] = centroid_(m)
    [rr, cc] = find(m);  c = [mean(cc) mean(rr)];  r = sqrt(nnz(m)/pi);
end
function s = iff_(c, a, b), if c, s = a; else, s = b; end, end
function say_(rep, varargin)
    fprintf(varargin{:});  fprintf(rep, varargin{:});
end
