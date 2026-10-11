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
C.p_null = C.phasef(zeros(P.grid.N_G));
C.measf  = @(M) angle(exp(1i*(C.phasef(M) - C.p_null))) * P.LAM/(4*pi);
C.reg = R;
C.test_launch = test_launch;  C.th_pass = th_pass;  C.lo_launch = lo_launch;
end

% ---------------------------------------------------------------------
function phi = cophi_phase_(rx, b, th, P, launch, iTO, M, Elo, opt)
%COPHI_PHASE_  the heterodyne phase of the test arm with grid M on the DM.
    E = arm_field_(rx, b, th, P, launch, iTO, M);
    [Fr, alpha] = dmg_cophi_frames(E, Elo, opt);
    phi = dmg_cophi_recon(Fr, alpha);
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
    if isnan(an_deg)
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
