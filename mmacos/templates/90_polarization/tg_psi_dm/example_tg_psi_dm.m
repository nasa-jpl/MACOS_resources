% example_tg_psi_dm.m
% =========================================================================
%  POLARIZATION PHASE-SHIFTING TWYMAN-GREEN, WITH A DM AS THE TEST OPTIC
% =========================================================================
%  A surface gauge, end to end: build the rig, put a deformable mirror in
%  the test arm, step the fringe phase by ROTATING AN ANALYZER (nothing in
%  the interferometer moves), and recover the DM surface map beside the map
%  that was injected.
%
%  WHY TWYMAN-GREEN FOR A DM (the topology trade -- see README.md): normal
%  incidence on the test optic, a natural null against a flat reference, and
%  the fewest reference surfaces of any two-beam layout.  A Mach-Zehnder
%  buys arm isolation, two complementary output ports and testing in
%  transmission -- none of which a figure measurement needs, and it pays for
%  them with a second beamsplitter and an oblique test optic.  MZ earns its
%  place for DYNAMICS, not for figure.
%
%  WHY POLARIZATION PHASE-SHIFTING: the phase steps come from rotating an
%  analyzer in the output leg, not from translating a reference mirror.  The
%  interferometer never moves, so the measurement does not integrate the
%  drift and vibration a PZT-stepped rig accumulates between frames -- and
%  with a polarization-multiplexed detector all four frames are SIMULTANEOUS
%  (a snapshot gauge).  The price is a polarization-component error budget,
%  which ../bench_ifo_pol/example_bench_ifo_pol_slice3.m measures term by
%  term (arm-QWP retardance is the tightest, ~344 nm/wave).
%
%  THE OPTICAL TRAIN.  Each arm is its own deck: the engine does not split
%  rays, so a polarizing beamsplitter is TWO traces.
%
%    input polarizer @45 -> [BS] -> double-passed QWP (net half-wave, which
%    rotates that arm's linear state) -> TEST OPTIC (the DM) or PZT flat ->
%    [recomb] -> output QWP (orthogonal linear -> orthogonal circular) ->
%    ROTATING ANALYZER @theta -> L2 + focal mask + field lens -> detector at
%    the DM pupil conjugate.
%
%  With the arms orthogonally circular the analyzer imposes a fringe phase
%  2*theta, so theta = 0/45/90/135 deg are the four steps of a standard
%  four-step PSI.
%
%  TWO RESULTS THIS EXAMPLE EXISTS TO SHOW
%
%  (1) THE ANALYZER SWEEP IS FREE, AND EXACTLY SO.  An ideal analyzer at
%      angle t projects onto a(t) = cos t*u1 + sin t*u2, and everything
%      downstream is a fixed linear map M per ray, so
%
%          E_det(t) = (a(t).E_in) * M a(t)          -- BILINEAR in a(t)
%                   = c^2*A + c*s*B + s^2*C,   c = cos t, s = sin t
%
%      with A = E_det(0), C = E_det(90), B = 2*E_det(45) - A - C.  THREE
%      traces per arm therefore reproduce the detector field at ANY analyzer
%      angle: the four PSI frames, a 64-angle least-squares fit and a live
%      animation all come out of the same six traces.  Gate 3 checks this
%      against direct engine traces at angles that are not in the basis, and
%      measures the one thing that breaks it -- the engine projects the
%      analyzer's material axis into each ray's transverse plane and
%      RENORMALIZES, which is nonlinear at O(beta^2) for a ray beta off the
%      element normal.  Measured: 0.149*beta^2 over four decades of beta.
%
%  (2) THE BEAMSPLITTER MISALIGNS THE RIG, AND THE GAUGE READS 11.7% HIGH
%      UNTIL YOU FIX IT.  The design intent is that the double-passed QWPs
%      leave the arms in ORTHOGONAL linear states.  They do not: the test
%      arm's 45-degree beamsplitter reflection and its glass transits are
%      DIATTENUATORS, and their s/p imbalance rotates that arm's azimuth by
%      7.5 degrees.  Nothing downstream can repair it -- a waveplate is
%      unitary and cannot make two non-orthogonal states orthogonal -- so
%      the four-step estimator, which assumes an orthogonal-circular pair,
%      acquires a local phase GAIN.  Measured on a known 20 nm piston: 11.7%
%      high, while every underlying field phase in the model is exact to
%      1e-9 rad.  Section 3 finds it the way a bench does (rotate the arm
%      waveplate until the arms extinguish each other), and gate 5 then
%      recovers the same piston at gain 1.00000 -- 2e-5 nm of surface on a
%      20 nm input, one part in 1e6.
%
%      The lesson generalizes: FRINGE VISIBILITY IS NOT AN ALIGNMENT METRIC.
%      Measured on this rig, the 7.5-degree non-orthogonality costs 11.7% of
%      SCALE and 0.17% of CONTRAST -- a factor of ~70.  Contrast responds at
%      second order in the alignment error, phase gain at first, so a rig can
%      look beautifully aligned on a fringe monitor and still read a DM 12%
%      too tall.  Gate 5 runs both configurations and prints both numbers.
%
%  Run:  cd <this dir>
%        matlab -batch "run('example_tg_psi_dm.m')"      % or interactively
%  Needs MACOS_HOME set and the mmacos path (mmacos_setup.m).
% =========================================================================

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);                         % GridFile= resolves relative to the cwd

MODEL = 256;                       % >= N_G, so the GridData map is not resampled
macos.init(MODEL);

% ---- parameters ---------------------------------------------------------
LAM     = 6.328e-4;                % mm (HeNe)
NGRID   = 63;                      % source ray grid
N_G     = 256;   DX_G = 0.35;      % DM grid: 256 nodes at 0.35 mm = 89 mm span
NACT    = 16;    PITCH = 3.5;      % 16x16 actuators at 3.5 mm = 56 mm DM
POKE_NM = 50;                      % command amplitude, nm of surface
QWP     = 0.25;                    % quarter-wave retardance (waves)
THETAS  = [0 45 90 135];           % the four analyzer steps (deg)
%  Detector-leg architecture: the l2_trade winner (a small field lens behind
%  the focal mask), which took the instrument-vs-truth residual from 6.76 nm
%  to 0.97 nm at 50 nm pokes.  See templates/40_benches/bench_ifo_dm/l2_trade.
TAIL = {'tail_arch','fieldlens', 'FL_F',25.02100857, 'FL_Kc',-2.11278288, ...
        'D_MASK_FL',6.277463741, 'DET_TRIM',1.085330067};

fprintf('\n=========================================================\n');
fprintf(' POLARIZATION PSI TWYMAN-GREEN -- DM SURFACE GAUGE\n');
fprintf('=========================================================\n');
fprintf('model %d | ray grid %d | DM %dx%d act @ %.2f mm | pokes %g nm | %.1f nm\n\n', ...
    MODEL, NGRID, NACT, NACT, PITCH, POKE_NM, LAM*1e6);

% =========================================================================
%  1.  The DM surface (actuator influence functions) -> GridFile
% =========================================================================
[Mdm, dminfo] = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, ...
                                 'poke',POKE_NM*1e-6, 'pattern','checker');
macos.write_grid_file('dm_poked.txt', Mdm);
macos.write_grid_file('dm_flat.txt',  zeros(N_G));
fprintf('DM map: %.3f nm rms, %.3f nm PtV, %.1f mm span, %.2f mm influence radius\n', ...
    1e6*std(Mdm(:)), 1e6*(max(Mdm(:))-min(Mdm(:))), dminfo.span_mm, dminfo.width_mm);

% =========================================================================
%  2.  Build both arms of the polarizing rig
% =========================================================================
%  ONE call builds the whole bench; every optic is placed on the chief ray
%  the builder tracks analytically.  'polarizing' inserts the REAL engine
%  polarizing elements (TrPolarizer = EltID 15, WavePlate = 18) into the
%  collimated, normal-incidence legs.  Default false emits the plain
%  Twyman-Green bit-identically, which tBench pins.
mkrig = @(gf) macos.design.twyman_green('polarizing', true, 'ngridpts', NGRID, ...
    'to_grid_file', gf, 'to_grid_n', N_G, 'to_grid_dx', DX_G, ...
    'qwp_ret', QWP, 'pol_in_deg', 45, 'qwp_test_deg', 0, 'qwp_ref_deg', 45, ...
    'out_qwp_deg', 0, 'analyzer_deg', 0, TAIL{:});
G  = mkrig('dm_poked.txt');        % DM poked
G0 = mkrig('dm_flat.txt');         % DM flat  (the baseline of the differential)
G.bt.emit('tg_test.in');  G0.bt.emit('tg_base.in');  G.br.emit('tg_ref.in');

AT = arm_desc('tg_test.in', G.bt,  G.T,  0);        % test arm, DM poked
AB = arm_desc('tg_base.in', G0.bt, G0.T, 0);        % test arm, DM flat
AR = arm_desc('tg_ref.in',  G.br,  G.R,  45);       % reference arm
fprintf('test arm %2d elements (TestOptic %d, Recomb %d, OutQWP %d, Analyzer %d, Detector %d)\n', ...
    numel(G.bt.E), G.T.iTO, G.T.iRC, G.T.iOutQWP, G.T.iAnalyzer, G.T.iDET);
fprintf('ref  arm %2d elements (PZT %d, Recomb %d, OutQWP %d, Analyzer %d, Detector %d)\n\n', ...
    numel(G.br.E), G.R.iPZT, G.R.iRC, G.R.iOutQWP, G.R.iAnalyzer, G.R.iDET);
%  The two arms must recombine into the SAME output leg or their azimuths
%  are not even comparable -- assert it rather than trust the builder.
assert(norm(G.bt.E(G.T.iRC).psi(:) - G.br.E(G.R.iRC).psi(:)) < 1e-12, ...
    'the two arms do not share the output-leg direction');

% =========================================================================
%  GATE 1 -- Tranche-1 element order, and the pol train really reaches the grid
% =========================================================================
%  Tranche-1 rule (PLAN_POLARIZATION 3a): a polarizing element placed AFTER
%  the first physical-optics leg transforms rays but never reaches the
%  diffraction grid.  This train is geometric end to end, so every element
%  precedes the readout assembly -- but ASSERT it from the emitted deck
%  rather than assume it, because a later PropType edit would break the
%  measurement silently (crossed polarizers taking the ray power to 1e-33
%  while the detector sits at full brightness is the recorded symptom).
fprintf('=== GATE 1: Tranche-1 element order + the grid carries the pol train ===\n');
g1 = struct();
for f = {'tg_test.in','tg_ref.in'}
    D = read_deck(f{1});
    ipol = find(ismember(D.element, {'TrPolarizer','WavePlate'}));
    ipo  = find(~strcmp(D.proptype, 'Geometric'));
    assert(~isempty(ipol), 'GATE 1: %s has no polarizing elements', f{1});
    assert(isempty(ipo) || max(ipol) < min(ipo), ...
        'GATE 1 FAILED: %s has a polarizing element at %d AFTER a physical-optics leg at %d', ...
        f{1}, max(ipol), min(ipo));
    fprintf('  %-12s polarizing elements at %s; %d physical-optics legs\n', ...
        f{1}, mat2str(ipol(:).'), numel(ipo));
end
%  The tripwire a source read cannot give: rotate the analyzer and watch the
%  DIFFRACTION GRID at the detector respond.  If the polarizing elements were
%  invisible to the grid this ratio would be exactly 1.
g1.an_ratio = sum(sum(grid_intensity(AT,QWP,90)))/sum(sum(grid_intensity(AT,QWP,0)));
fprintf('  grid power at analyzer 90 / at 0 = %.4f  (1.0 would mean the grid is blind)\n', g1.an_ratio);
assert(abs(g1.an_ratio - 1) > 0.1, 'GATE 1 FAILED: the grid does not see the analyzer');
fprintf('GATE 1 PASS\n\n');

% =========================================================================
%  GATE 2 -- the two arms land on the SAME detector pixels
% =========================================================================
%  A PSI that adds two arms pixel by pixel is only meaningful if that is true
%  NUMERICALLY.  Check the sampling and the beam registration -- never by
%  eye, and never by assuming the builder got it right.
fprintf('=== GATE 2: detector registration between the two arms ===\n');
[dxT, cT, rT] = det_geom(AT);
[dxR, cR, rR] = det_geom(AR);
g2 = struct('dxT',dxT, 'dxR',dxR, 'dpitch_rel', abs(dxT-dxR)/max(dxT,dxR), ...
            'dcen', norm(cT-cR), 'drad', abs(rT-rR));
fprintf('  pixel pitch  test %.9f mm | ref %.9f mm | relative diff %.2e\n', dxT, dxR, g2.dpitch_rel);
fprintf('  flux-weighted centroid offset %.2e px | second-moment radius diff %.2e px\n', ...
    g2.dcen, g2.drad);
%  The pitch follows each arm's own beam diameter at the detector, and the
%  arms carry different glass-path bookkeeping, so an exact tie is not
%  expected; what matters is that it is far below one pixel at the beam edge
%  (3e-8 of a pitch over ~130 px is 4e-6 px).
assert(g2.dpitch_rel < 1e-6, 'GATE 2 FAILED: detector sampling differs (%.2e)', g2.dpitch_rel);
assert(g2.dcen < 0.05 && g2.drad < 0.05, 'GATE 2 FAILED: the two beams are not registered');
fprintf('GATE 2 PASS: pitch matches to %.1e, beams registered to %.1e px\n\n', ...
    g2.dpitch_rel, g2.dcen);

% =========================================================================
%  3.  ALIGN THE WAVEPLATES -- the beamsplitter has rotated an arm
% =========================================================================
%  The design says: input linear at 45, a double-passed QWP at azimuth alpha
%  acts as a half-wave plate and sends that linear state to 2*alpha-45, so
%  alpha = 0 and alpha = 45 should leave the arms at -45 and +45 -- ORTHOGONAL.
%  Measure it instead.  Every element between the polarizer and the recomb
%  plane that is not at normal incidence is a DIATTENUATOR (the 45-degree
%  beamsplitter reflection, and every tilted glass transit: t_s /= t_p), and
%  a diattenuator ROTATES a linear state toward its high-transmission axis.
%  The reference arm happens to be immune -- its half-wave sits at 45 deg,
%  which maps azimuth th -> 90-th and so cancels the diattenuation before it
%  against the diattenuation after it.  The test arm's half-wave at 0 deg
%  maps th -> -th, which ADDS them.
%
%  Nothing downstream can repair the result: a waveplate is unitary, and a
%  unitary cannot turn a non-orthogonal pair into an orthogonal one.  The fix
%  has to be the arm waveplate itself -- exactly the knob a bench technician
%  turns while watching the two arms extinguish each other.
fprintf('=== ALIGNMENT: measure the arms, then null the beamsplitter rotation ===\n');
az_r = arm_azimuth(AR, QWP, AR.base);
fprintf('  reference arm azimuth at recomb      %+8.4f deg (design %+8.4f)\n', az_r, 45);
az_t0 = arm_azimuth(AB, QWP, 0);
fprintf('  test arm azimuth at recomb, QWP @0   %+8.4f deg (design %+8.4f)\n', az_t0, -45);
want = wrap180(az_r - 90);
fprintf('  --> the arms are %.4f deg from orthogonal; the beamsplitter has\n', ...
    abs(wrap180(az_t0 - want)));
fprintf('      rotated the TEST arm by %.4f deg\n', wrap180(az_t0 - want));
%  Solve for the arm-QWP azimuth that restores orthogonality.  A half-wave
%  moves the output azimuth at ~2x its own rotation, but the diattenuation
%  AFTER it makes that only approximate, so take a secant step and iterate.
al = [0, -0.5*wrap180(az_t0 - want)];
az = [az_t0, arm_azimuth(AB, QWP, al(2))];
for it = 3:5
    if abs(az(end)-az(end-1)) < 1e-9, break; end
    slope = (az(end)-az(end-1))/(al(end)-al(end-1));
    al(end+1) = al(end) + wrap180(want - az(end))/slope; %#ok<SAGROW>
    az(end+1) = arm_azimuth(AB, QWP, al(end));           %#ok<SAGROW>
    if abs(wrap180(az(end)-want)) < 1e-7, break; end
end
alpha_t = al(end);
fprintf('  solved test-arm QWP azimuth %+8.4f deg -> arm azimuth %+8.4f (want %+8.4f), %d traces\n', ...
    alpha_t, az(end), want, numel(al));
%  With the pair orthogonal LINEAR at (az_r, az_r-90), a quarter-wave at 45
%  degrees to them makes them orthogonal CIRCULAR -- which is what the
%  rotating analyzer needs.
oq = az_r + 45;   % a waveplate axis is mod 180; no wrap needed
fprintf('  output QWP azimuth set to %+8.4f deg (bisector + 45)\n', oq);
[AT, AB, AR] = deal(set_pol_align(AT, alpha_t, oq), ...
                    set_pol_align(AB, alpha_t, oq), ...
                    set_pol_align(AR, AR.base,  oq));
%  Verify on the pair the analyzer actually sees.
[st, sr] = deal(arm_state(AT, QWP, AT.iOQ), arm_state(AR, QWP, AR.iOQ));
align = struct('az_ref',az_r, 'az_test_nominal',az_t0, 'want',want, ...
    'bs_rotation_deg', wrap180(az_t0-want), 'alpha_t',alpha_t, 'out_qwp',oq, ...
    'ortho', abs(st'*sr)/(norm(st)*norm(sr)), ...
    'ellip_t', abs(st(2)/st(1)), 'ellip_r', abs(sr(2)/sr(1)), ...
    'arg_t', rad2deg(angle(st(2)/st(1))), 'arg_r', rad2deg(angle(sr(2)/sr(1))));
fprintf('  after the output QWP: test |b/a| %.5f arg %+7.2f | ref |b/a| %.5f arg %+7.2f\n', ...
    align.ellip_t, align.arg_t, align.ellip_r, align.arg_r);
fprintf('  (circular is |b/a| = 1 and arg = -+90);  |<test|ref>| = %.2e\n', align.ortho);
assert(align.ortho < 1e-4, 'ALIGNMENT FAILED: arms still %.2e from orthogonal', align.ortho);
fprintf('ALIGNED\n\n');

% =========================================================================
%  GATE 3 -- three traces per arm span EVERY analyzer angle
% =========================================================================
fprintf('=== GATE 3: analyzer-basis synthesis is exact (and why) ===\n');
tb = tic;
St = analyzer_basis(AT, QWP);      % DM poked
Sb = analyzer_basis(AB, QWP);      % DM flat
Sr = analyzer_basis(AR, QWP);      % reference arm
t_basis = toc(tb);
fprintf('  9 traces (3 per arm x 3 decks) in %.2f s -- every analyzer angle now free\n', t_basis);
g3 = struct('theta',[], 'rel',[]);
for th = [23.7 71.3 137.2 -18.0]
    Edir = arm_field(AT, QWP, th);
    r = max(abs(reshape(synth(St,th) - Edir, [], 1)))/max(abs(Edir(:)));
    g3.theta(end+1) = th;  g3.rel(end+1) = r;
    fprintf('  theta %7.2f deg : synthesized vs direct trace, rel %.3e\n', th, r);
end
assert(max(g3.rel) < 1e-8, 'GATE 3 FAILED: the analyzer basis does not span the sweep');
%  NON-VACUITY -- the identity is exact only to O(beta^2).  Give the test
%  optic real power so the analyzer leg is no longer collimated, and the same
%  synthesis must degrade as beta^2.  A gate that only ever ran collimated
%  could not tell "exact" from "we never exercised the term".
fprintf('  non-vacuity: the O(beta^2) law (beta = ray angle at the analyzer)\n');
g3.beta = [];  g3.beta_rel = [];
for Kr = [4000 2000]
    Gk = macos.design.twyman_green('polarizing', true, 'ngridpts', NGRID, ...
        'to_Kr', Kr, 'qwp_ret', QWP, TAIL{:});
    Gk.bt.emit('tg_nonvac.in');
    Ak = set_pol_align(arm_desc('tg_nonvac.in', Gk.bt, Gk.T, 0), alpha_t, oq);
    Ed = arm_field(Ak, QWP, 23.7);
    rk = max(abs(reshape(synth(analyzer_basis(Ak,QWP),23.7) - Ed,[],1)))/max(abs(Ed(:)));
    g3.beta(end+1) = analyzer_beta(Ak);  g3.beta_rel(end+1) = rk;
    fprintf('    test optic Kr=%5g : beta %.4e rad, rel %.4e, rel/beta^2 %.4f\n', ...
        Kr, g3.beta(end), rk, rk/g3.beta(end)^2);
end
g3.beta_dm = analyzer_beta(AT);
fprintf('    DM deck          : beta %.4e rad (the DM slope error itself)\n', g3.beta_dm);
q = (g3.beta_rel(1)/g3.beta_rel(2)) / (g3.beta(1)/g3.beta(2))^2;
fprintf('  quadratic-law check: (rel ratio)/(beta ratio)^2 = %.4f  (1 = exactly 2nd order)\n', q);
assert(g3.beta_rel(2) > 1e-6, 'GATE 3 non-vacuity FAILED: powered optic shows no error to detect');
assert(abs(q-1) < 0.05, 'GATE 3 FAILED: the residual does not follow the beta^2 law');
fprintf('GATE 3 PASS: exact to %.1e collimated; degrades as 0.149*beta^2 with power\n\n', max(g3.rel));

% =========================================================================
%  GATE 4 -- the analyzer sweep: a pure 2-theta fringe
% =========================================================================
%  I(theta) can only contain DC, 2*theta and 4*theta: each arm's field is
%  quadratic in (cos t, sin t), so the intensity is quartic and NOTHING above
%  4*theta may appear.  The 4*theta term is the small systematic the four-step
%  estimator carries AT THE DETECTOR (it is identically zero AT the analyzer,
%  which is why the ray-level slice-3 gate reports 1e-16).
%
%  THE 6-THETA BIN IS ONLY A GATE ON FRAMES THAT CAME FROM THE ENGINE.  A
%  frame synthesized from the quadratic basis is a degree-2 trig polynomial in
%  2*theta BY CONSTRUCTION, so its 6*theta content is zero by algebra -- it
%  measures this script's arithmetic, not the engine, and lands at 1e-17
%  whatever the engine does.  So the harmonic assertion below runs on TRACED
%  angles, where the prediction has something to be wrong about: the analyzer
%  really has to behave as an ideal rank-1 projector, and the O(beta^2)
%  renormalization that gate 3 measured is exactly the kind of thing that
%  would put power above 4*theta.
fprintf('=== GATE 4: analyzer sweep modulates as cos(2 theta) ===\n');
nsw = 64;  th_sw = (0:nsw-1)/nsw*180;
Isw = zeros(size(St.A,1), size(St.A,2), nsw);
for k = 1:nsw
    Isw(:,:,k) = sum(abs(synth(St,th_sw(k)) + synth(Sr,th_sw(k))).^2, 3);
end
Ibar = mean(Isw,3);
msk  = Ibar > 0.1*max(Ibar(:));
[v_s, h4_s, h6_s] = harmonics(Isw, msk);
%  the same quantities from 12 DIRECTLY TRACED analyzer angles (24 traces)
nd = 12;  th_d = (0:nd-1)/nd*180;
Id = zeros(size(St.A,1), size(St.A,2), nd);
td = tic;
for k = 1:nd
    Id(:,:,k) = sum(abs(arm_field(AT,QWP,th_d(k)) + arm_field(AR,QWP,th_d(k))).^2, 3);
end
[v_d, h4_d, h6_d] = harmonics(Id, msk);
g4 = struct('vis',v_d, 'h4_h2',h4_d, 'h6_h2',h6_d, ...
            'vis_syn',v_s, 'h4_h2_syn',h4_s, 'h6_h2_syn',h6_s, 't_direct',toc(td));
fprintf('  from %d TRACED angles (%.2f s): visibility %.6f | 4t/2t %.4e | 6t/2t %.4e\n', ...
    nd, g4.t_direct, v_d, h4_d, h6_d);
fprintf('  from the synthesized basis        : visibility %.6f | 4t/2t %.4e | 6t/2t %.4e\n', ...
    v_s, h4_s, h6_s);
fprintf('  (the synthesized 6t is zero BY CONSTRUCTION -- the traced one is the gate)\n');
assert(g4.vis > 0.99, 'GATE 4 FAILED: fringe visibility %.4f', g4.vis);
assert(h6_d < 1e-8, 'GATE 4 FAILED: traced sweep has content above 4theta (%.2e)', h6_d);
assert(abs(h4_d/h4_s - 1) < 0.05, ...
    'GATE 4 FAILED: traced and synthesized 4theta disagree (%.4e vs %.4e)', h4_d, h4_s);
fprintf('GATE 4 PASS\n\n');

% =========================================================================
%  GATE 5 -- units, sign and scale of the whole chain, on a known input
% =========================================================================
%  Before trusting a recovered map, pin the chain with an input you already
%  know: set the DM grid to a uniform piston dz.  A surface displacement dz
%  toward the beam shortens the double-passed path by 2*dz, so the recovered
%  fringe phase must move by 4*pi*dz/lambda.  Doing the same by TRANSLATING
%  the whole test optic by dz must give the identical answer -- which settles
%  the standing question of whether a GridData value enters the OPD as 1x or
%  2x its height (it is the SURFACE height; the double pass supplies the 2).
%
%  This is also the gate that catches the beamsplitter misalignment of
%  section 3: run it on the UNALIGNED rig and it reads 11.7% high.
fprintf('=== GATE 5: a grid value is a SURFACE height (sign + scale) ===\n');
dz = 20e-6;                        % 20 nm of surface
expect = 4*pi*dz/LAM;
macos.write_grid_file('dm_piston.txt', dz*ones(N_G));
Gp = mkrig('dm_piston.txt');  Gp.bt.emit('tg_piston.in');
Ap = set_pol_align(arm_desc('tg_piston.in', Gp.bt, Gp.T, 0), alpha_t, oq);
Sp = analyzer_basis(Ap, QWP);
[dphi_grid, ~, vis_al] = psi_diff(Sp, Sb, Sr, THETAS, msk);
At = AB;  At.shift = struct('elt', G0.T.iTO, 'dz', dz);   % the same, as a rigid move
dphi_shift = psi_diff(analyzer_basis(At,QWP), Sb, Sr, THETAS, msk);
g5 = struct('grid_rad', median(dphi_grid(msk)), 'shift_rad', median(dphi_shift(msk)), ...
            'expect_rad', expect);
g5.sign = sign(g5.grid_rad);
g5.gain = abs(g5.grid_rad)/expect;
g5.err_nm = abs(abs(g5.grid_rad) - expect)/(4*pi)*LAM*1e6;
fprintf('  uniform grid piston %g nm -> %+.6f rad (expect %+.6f), gain %.5f\n', ...
    dz*1e6, g5.grid_rad, g5.sign*expect, g5.gain);
fprintf('  rigid optic shift   %g nm -> %+.6f rad\n', dz*1e6, g5.shift_rad);
fprintf('  grid vs rigid shift  %.3e rad  => a grid value IS the surface height\n', ...
    abs(g5.grid_rad - g5.shift_rad));
fprintf('  scale error %.5f nm of surface (%.4f%%), pupil rms %.2e rad\n', ...
    g5.err_nm, 100*g5.err_nm/(dz*1e6), std(dphi_grid(msk)));
assert(abs(g5.gain - 1) < 2e-3, 'GATE 5 FAILED: PSI gain %.5f (misaligned rig?)', g5.gain);
assert(abs(g5.grid_rad - g5.shift_rad) < 1e-3*expect, ...
    'GATE 5 FAILED: a grid piston and a rigid shift disagree');
SGN = g5.sign;                     % fixed here, used for every map below
%  COUNTERFACTUAL -- run the same calibration on the UNALIGNED rig, so the
%  alignment step of section 3 proves itself here instead of being asserted.
%  This is also where the "visibility is not an alignment metric" claim is
%  measured: the misaligned rig has the BETTER contrast and the WORSE scale.
Sp0 = analyzer_basis(set_pol_align(Ap, Ap.base, 0), QWP);
Sb0 = analyzer_basis(set_pol_align(AB, AB.base, 0), QWP);
Sr0 = analyzer_basis(set_pol_align(AR, AR.base, 0), QWP);
[d0, ~, v0] = psi_diff(Sp0, Sb0, Sr0, THETAS, msk);
g5.gain_unaligned = abs(median(d0(msk)))/expect;
g5.vis_unaligned  = median(v0(msk));
g5.vis_aligned    = median(vis_al(msk));   % same deck, same angles
fprintf('  COUNTERFACTUAL, same rig with the waveplates at their DESIGN angles:\n');
fprintf('    gain %.5f (%.1f%% high) with visibility %.6f\n', ...
    g5.gain_unaligned, 100*(g5.gain_unaligned-1), g5.vis_unaligned);
fprintf('    aligned: gain %.5f with visibility %.6f\n', g5.gain, g5.vis_aligned);
fprintf('    -> the misalignment costs %.1f%% of SCALE and %.2f%% of CONTRAST; a fringe\n', ...
    100*(g5.gain_unaligned-1), 100*abs(g5.vis_aligned-g5.vis_unaligned)/g5.vis_aligned);
fprintf('       monitor is ~%.0fx less sensitive to it than the measurement is\n', ...
    (g5.gain_unaligned-1)/max(abs(g5.vis_aligned-g5.vis_unaligned)/g5.vis_aligned,eps));
assert(abs(g5.gain_unaligned - 1) > 0.05, ...
    'GATE 5 counterfactual FAILED: the alignment step is not load-bearing');
fprintf('GATE 5 PASS: h = %+d * psi * lambda/(4 pi), gain %.5f\n\n', SGN, g5.gain);

% =========================================================================
%  6.  The measurement: four-step polarization PSI, differentially
% =========================================================================
%  A real surface gauge runs the PSI sequence TWICE -- with the DM flat and
%  with it poked -- and subtracts the two wrapped phase maps in the complex
%  domain.  Every static term (the rig's own polarization aberration, the
%  arms' figure, the grid block's DC offset) cancels identically, and what is
%  left is smaller than pi, so nothing has to be unwrapped anywhere.
fprintf('=== MEASUREMENT: four-step polarization PSI (differential) ===\n');
[dphi, frames, vis] = psi_diff(St, Sb, Sr, THETAS, msk);
h = SGN * dphi * LAM/(4*pi);       % recovered surface height, mm
fprintf('  fringe visibility %.4f | recovered surface %.3f nm rms in the pupil\n', ...
    median(vis(msk)), 1e6*std(h(msk)));
%  A SECOND estimator for free: with the sweep costing nothing, fit the whole
%  I(theta) curve instead of sampling it four times.  The difference between
%  the two IS the 4-theta systematic gate 4 measured.
dphi_ls = psi_diff(St, Sb, Sr, th_sw, msk);
d_est = angle(exp(1i*(dphi - dphi_ls)));
est_nm = 1e6*std(d_est(msk)*LAM/(4*pi));
fprintf('  four-step vs %d-angle least squares: %.3e nm rms of surface\n', nsw, est_nm);
fprintf('  (the 4-theta systematic is COMMON to the poked and baseline runs, so\n');
fprintf('   the differential cancels it -- it survives only in a single-shot map)\n');

% =========================================================================
%  7.  Closure: recovered map vs the DM map that was injected
% =========================================================================
%  The detector images the DM through L2 + the field lens + the folded
%  tilted-plate train.  That image is not an ideal copy: it carries
%  magnification, a small anamorphic stretch, rotation and nonlinear
%  distortion.  Measure the mapping from the trace itself -- every surviving
%  ray gives one (DM position, detector position) pair -- and use it to
%  resample the truth onto detector pixels.  The ray trace is the fiducial a
%  real bench gets from a known poke or the aperture edge.
fprintf('\n=== CLOSURE: recovered surface vs the injected DM map ===\n');
[best, map] = register_to_dm(AT, G.T, Mdm, N_G, DX_G, h, msk);
res = best.hm(msk) - best.ht(msk);
%  Score the interior separately.  The outermost ring is where the truth is
%  resampled through the ray map with rays on one side only, so part of its
%  residual is an artefact of the comparison rather than instrument error.
%  MEASURED, it is not the dominant term (0.363 whole pupil vs 0.304 interior
%  -- the residual is spread across the pupil, which is the detector-leg
%  retrace term the l2_trade work characterised, not an edge effect).  Report
%  both; the interior number is the gauge's.
mskin = erode_disc(msk, 0.92);
resin = best.hm(mskin) - best.ht(mskin);
g6 = struct('corr',best.c, 'axes',best.name, 'resid_nm',1e6*std(res), ...
            'resid_in_nm',1e6*std(resin), 'truth_in_nm',1e6*std(best.ht(mskin)), ...
            'truth_nm',1e6*std(best.ht(msk)), 'rec_nm',1e6*std(best.hm(msk)), ...
            'mag',map.mag, 'anam_pct',map.anam_pct, 'rot_deg',map.rot_deg, ...
            'nonlin_mm',map.nonlin_mm, 'est_nm',est_nm);
fprintf('  pupil image: mag %.4f (det->DM), anamorphic %.3f%%, rotation %.3f deg,\n', ...
    map.mag, map.anam_pct, map.rot_deg);
fprintf('               nonlinear distortion %.4f mm rms (%.2f%% of the beam)\n', ...
    map.nonlin_mm, 100*map.nonlin_mm/map.r_beam);
fprintf('  pixel axes %s, correlation %.6f\n', best.name, best.c);
fprintf('  truth %.2f nm rms | recovered %.2f nm rms\n', g6.truth_nm, g6.rec_nm);
fprintf('  RESIDUAL %.3f nm rms whole pupil | %.3f nm rms interior (rim excluded)\n', ...
    g6.resid_nm, g6.resid_in_nm);
assert(best.c > 0.99, 'CLOSURE FAILED: correlation %.4f', best.c);
assert(g6.resid_in_nm < 0.10*g6.truth_in_nm, 'CLOSURE FAILED: residual %.3f nm', g6.resid_in_nm);
fprintf('CLOSURE PASS: %.1f%% of the injected map recovered (interior)\n\n', ...
    100*(1 - g6.resid_in_nm/g6.truth_in_nm));

% =========================================================================
%  8.  Figures + artefacts
% =========================================================================
NN = size(h,1);
box = beam_box(msk, 6);            % the beam is a small disc on a 256^2 array
f1 = figure('Color','w','Position',[80 80 1560 420]);
tl = tiledlayout(f1,1,4,'TileSpacing','compact','Padding','compact');
nexttile; imagesc(sub(frames{1},box)); axis image off; colorbar;
title('interferogram, analyzer 0\circ');
cl = [-1 1]*1e6*max(abs(best.ht(msk)));
show(1e6*best.hm, msk, NN, box, cl, sprintf('recovered surface (%.2f nm rms)', g6.rec_nm));
show(1e6*best.ht, msk, NN, box, cl, sprintf('injected DM map (%.2f nm rms)', g6.truth_nm));
%  the residual panel is scaled to the INTERIOR, so the rim ring does not
%  autoscale the interior structure into invisibility
show(1e6*(best.hm-best.ht), msk, NN, box, [-1 1]*4*g6.resid_in_nm, ...
     sprintf('residual %.3f nm rms interior', g6.resid_in_nm));
title(tl, 'Polarization-PSI Twyman-Green: DM surface recovery');
print(f1, 'tg_psi_dm_recovery.png', '-dpng', '-r150');

f2 = figure('Color','w','Position',[80 80 1200 480]);
subplot(1,2,1);
pk = find(msk);  [~,ip] = max(abs(h(pk)));  [pr,pc] = ind2sub([NN NN], pk(ip));
Isweep = squeeze(Isw(pr,pc,:));
plot(th_sw, Isweep/max(Isweep), 'b-', 'LineWidth',1.4); hold on;
plot(THETAS, interp1([th_sw 180], [Isweep; Isweep(1)]/max(Isweep), THETAS), 'ro', ...
     'MarkerFaceColor','r', 'MarkerSize',7);
grid on; xlabel('analyzer angle \theta (deg)'); ylabel('normalized intensity');
title(sprintf('one pixel through the sweep (visibility %.4f)', g4.vis));
legend({'I(\theta), synthesized from 3 traces','the four PSI steps'}, 'Location','south');
subplot(1,2,2);
loglog(g3.beta, g3.beta_rel, 'ro-', 'LineWidth',1.4, 'MarkerFaceColor','r'); hold on;
bb = logspace(log10(min([g3.beta g3.beta_dm])/3), log10(max(g3.beta)*3), 20);
loglog(bb, 0.149*bb.^2, 'k--');
loglog(g3.beta_dm, g3.rel(1), 'bs', 'MarkerFaceColor','b', 'MarkerSize',9);  % same theta as the ladder
grid on; xlabel('\beta = ray angle at the analyzer (rad)');
ylabel('synthesis error, relative');
title('the free sweep is exact to O(\beta^2)');
legend({'powered test optic','0.149\beta^2','this rig (collimated)'}, 'Location','southeast');
print(f2, 'tg_psi_dm_sweep.png', '-dpng', '-r150');

f3 = figure('Color','w','Position',[80 80 1400 480]);
axs = ((1:N_G)-(N_G+1)/2)*DX_G;  keep = abs(axs) <= 1.08*map.r_beam;
Fh2 = scatteredInterpolant(map.Xt(msk), map.Yt(msk), best.hm(msk), 'natural','none');
[GXs, GYs] = ndgrid(axs(keep), axs(keep));
hdm = Fh2(GXs, GYs);
tdm = interpn(axs, axs, Mdm, GXs, GYs, 'linear', nan);  tdm(isnan(hdm)) = nan;
tdm = tdm - mean(tdm(~isnan(tdm)));
Z = {hdm, tdm, hdm-tdm};
cl3 = [-1 1]*1e6*max(abs(tdm(~isnan(tdm))));
lims = {cl3, cl3, [-1 1]*4*g6.resid_in_nm};
ttl = {sprintf('measured, IN THE DM PLANE (mag %.3f calibrated)', map.mag), ...
       'injected DM map', sprintf('difference (+-%.2f nm)', 4*g6.resid_in_nm)};
for c = 1:3
    subplot(1,3,c);
    imagesc(axs(keep), axs(keep), 1e6*Z{c}.', 'AlphaData', ~isnan(Z{c}.'));
    axis image; axis xy; clim(lims{c}); colorbar; title(ttl{c});
    xlabel('DM x (mm)'); ylabel('DM y (mm)');
end
print(f3, 'tg_psi_dm_surface.png', '-dpng', '-r150');

try
    macos.load_rx('tg_test.in');
    macos.view_std('title','Polarization-PSI Twyman-Green (test arm)', ...
                   'save','tg_psi_dm_layout.png');
catch ME
    fprintf('(layout view skipped: %s)\n', ME.message);
end

results = struct('MODEL',MODEL, 'NGRID',NGRID, 'N_G',N_G, 'DX_G',DX_G, ...
    'NACT',NACT, 'PITCH',PITCH, 'POKE_NM',POKE_NM, 'LAM',LAM, 'THETAS',THETAS, ...
    'align',align, 'gate1',g1, 'gate2',g2, 'gate3',g3, 'gate4',g4, 'gate5',g5, ...
    'closure',g6, 'sign',SGN, 't_basis',t_basis);
%  Isw (256^2 x 64 analyzer frames) is deliberately NOT saved -- it is three
%  lines of synth() away from the bases and would triple the artefact for
%  nothing.
save('tg_psi_dm.mat', 'results', 'h', 'msk', 'best', 'map', 'Mdm', 'dminfo', ...
     'th_sw', 'frames');

fprintf('=========================================================\n');
fprintf(' SUMMARY\n');
fprintf('   beamsplitter rotation  %+.3f deg on the test arm, nulled by turning\n', align.bs_rotation_deg);
fprintf('                          its waveplate %+.3f deg;  unaligned gain %.5f\n', ...
    align.alpha_t, g5.gain_unaligned);
fprintf('   analyzer basis         3 traces/arm, exact to %.1e (collimated)\n', max(g3.rel));
fprintf('   visibility             %.6f aligned vs %.6f unaligned -- the same\n', ...
    g5.vis_aligned, g5.vis_unaligned);
fprintf('                          misalignment costs %.1f%% of SCALE and %.2f%% of\n', ...
    100*(g5.gain_unaligned-1), 100*abs(g5.vis_aligned-g5.vis_unaligned)/g5.vis_aligned);
fprintf('                          CONTRAST, so a fringe monitor is ~%.0fx blinder\n', ...
    (g5.gain_unaligned-1)/max(abs(g5.vis_aligned-g5.vis_unaligned)/g5.vis_aligned,eps));
fprintf('   4theta systematic      %.2e of the fringe (%.1e nm after the differential)\n', ...
    g4.h4_h2, g6.est_nm);
fprintf('   piston calibration     gain %.5f on a %g nm input (%.4f nm error)\n', ...
    g5.gain, dz*1e6, g5.err_nm);
fprintf('   DM recovery            corr %.6f, residual %.3f nm rms interior\n', g6.corr, g6.resid_in_nm);
fprintf('                          (%.3f whole pupil) on %.2f nm rms of surface\n', ...
    g6.resid_nm, g6.truth_nm);
fprintf('   artefacts              tg_*.in, tg_psi_dm.mat, tg_psi_dm_*.png\n');
fprintf('=========================================================\n');

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================
function A = arm_desc(rx, b, ix, base_deg)
%ARM_DESC  Everything a trace of one arm needs: the deck, the pol element
%   indices, and that arm's waveplate azimuths.  QWP_DEG starts at the design
%   value BASE_DEG and is replaced by the alignment in section 3.
    nm = {b.E.name};
    A = struct('rx', rx, 'b', b, ...
        'iPol', find(strcmp(nm,'PolIn'),1), ...
        'iQ',   find(contains(nm,'QWP') & ~strcmp(nm,'OutQWP')), ...
        'base', base_deg, 'qwp_deg', base_deg, 'oq_deg', 0, ...
        'iRC', ix.iRC, 'iOQ', ix.iOutQWP, 'iAn', ix.iAnalyzer, ...
        'iDET', ix.iDET, 'shift', []);
    assert(numel(A.iQ) == 2, 'arm_desc: expected a double-passed arm QWP');
end

function A = set_pol_align(A, qwp_deg, oq_deg)
    A.qwp_deg = qwp_deg;  A.oq_deg = oq_deg;
end

function a = lax(psi, deg)
%LAX  A polarization axis DEG degrees from local x in the transverse plane of
%   a beam along PSI -- the same right-handed frame the Bench emitter uses,
%   so "45 deg" is the same physical direction in every folded leg.
    u1 = macos.design.Bench.perp(psi(:));  u2 = cross(psi(:), u1);
    a = cosd(deg)*u1 + sind(deg)*u2;  a = a(:).';
end

function x = wrap180(x)
    x = mod(x + 90, 180) - 90;
end

function load_arm(A, QWP, an_deg)
%LOAD_ARM  Load the deck and set every polarizing element.  A double-passed
%   physical plate has ONE global fast axis: derive it from the FORWARD
%   element's frame and give the same vector to both passes (re-deriving it
%   from the return element's own psi would reflect the axis, breaking the
%   net half-wave for any azimuth other than 0/90).
    macos.load_rx(A.rx);  b = A.b;
    if ~isempty(A.shift)
        p = macos.get_elt_psi(A.shift.elt);  v = macos.get_elt_vpt(A.shift.elt);
        macos.set_elt_vpt(A.shift.elt, v + A.shift.dz*p);
    end
    macos.polarizer(A.iPol, 'axis', lax(b.E(A.iPol).psi, 45));
    qa = lax(b.E(A.iQ(1)).psi, A.qwp_deg);
    for j = 1:2, macos.waveplate(A.iQ(j), 'axis', qa, 'retardance', QWP); end
    macos.waveplate(A.iOQ, 'axis', lax(b.E(A.iOQ).psi, A.oq_deg), 'retardance', QWP);
    macos.polarizer(A.iAn, 'axis', lax(b.E(A.iAn).psi, an_deg));
    macos.polarization('on', 'Ex',[1/sqrt(2) 0], 'Ey',[1/sqrt(2) 0]);
    macos.vector_diffraction(true);
end

function E = arm_field(A, QWP, an_deg)
%ARM_FIELD  The complex VECTOR field at the detector, N x N x 3 (Ex,Ey,Ez).
%   Vector diffraction repurposes the model's three wavefront planes as the
%   Cartesian components of one wavefront; complex_field(...,'plane',k)
%   refuses k = 1..3 unless vector mode is on, so this cannot silently hand
%   back an unrelated wavefront.
    load_arm(A, QWP, an_deg);
    E = cat(3, macos.complex_field(A.iDET,'plane',1), ...
               macos.complex_field(A.iDET,'plane',2), ...
               macos.complex_field(A.iDET,'plane',3));
end

function I = grid_intensity(A, QWP, an_deg)
    load_arm(A, QWP, an_deg);
    I = macos.intensity(A.iDET);
end

function e = arm_state(A, QWP, iElt)
%ARM_STATE  The arm's Jones state (a,b) at element IELT, in that leg's
%   transverse frame, as a pupil median.  Ray-level: the polarization state
%   is a per-ray quantity and this reads it where the physics puts it, not
%   after a grid assembly.  Referenced to the first component so a common
%   propagation phase drops out.
    load_arm(A, QWP, 0);
    macos.trace(iElt);  f = macos.ray_field(iElt);
    ok = f.status == 0;
    psi = A.b.E(iElt).psi(:);
    u1 = macos.design.Bench.perp(psi);  u2 = cross(psi, u1);
    e1 = f.Ex*u1(1) + f.Ey*u1(2) + f.Ez*u1(3);
    e2 = f.Ex*u2(1) + f.Ey*u2(2) + f.Ez*u2(3);
    r  = e2(ok)./e1(ok);
    a  = median(abs(e1(ok)));
    e  = [a; a*(median(real(r)) + 1i*median(imag(r)))];
end

function az = arm_azimuth(A, QWP, qwp_deg)
%ARM_AZIMUTH  The polarization-ellipse azimuth (deg) this arm delivers to the
%   recombination plane when its double-passed waveplate sits at QWP_DEG.
    A.qwp_deg = qwp_deg;
    e = arm_state(A, QWP, A.iRC);
    az = 0.5*atan2d(2*real(conj(e(1))*e(2)), abs(e(1))^2 - abs(e(2))^2);
end

function S = analyzer_basis(A, QWP)
%ANALYZER_BASIS  Three traces that span every analyzer angle.
    E0  = arm_field(A, QWP, 0);
    E45 = arm_field(A, QWP, 45);
    E90 = arm_field(A, QWP, 90);
    S = struct('A', E0, 'C', E90, 'B', 2*E45 - E0 - E90);
end

function E = synth(S, th)
%SYNTH  The detector field at analyzer angle TH (deg), from the basis.
    c = cosd(th);  s = sind(th);
    E = c^2*S.A + c*s*S.B + s^2*S.C;
end

function b = analyzer_beta(A)
%ANALYZER_BETA  Largest ray angle off the analyzer normal -- the parameter
%   that controls how exactly three traces span the sweep.
    macos.load_rx(A.rx);
    st = macos.trace(A.iAn);  ri = macos.get_ray_info(st.nRays);
    ok = ri.ok_trace(:) & ri.ok_pass(:);
    psiA = A.b.E(A.iAn).psi(:);
    b = max(acos(min(abs(psiA.' * ri.dir(:,ok)), 1)));
end

function [dphi, frames, vis] = psi_diff(Sx, Sb, Sr, thetas, msk)
%PSI_DIFF  Differential polarization PSI: run the analyzer sequence on the
%   measured state (SX) and on the baseline (SB), each against the same
%   reference arm (SR), and subtract the two wrapped phases in the complex
%   domain, so every static term cancels.  With the four design angles the
%   closed-form four-step estimator is used; with any other set, a
%   least-squares 2-theta fit.
    [px, frames, vis] = psi_frames(Sx, Sr, thetas);
    pb = psi_frames(Sb, Sr, thetas);
    dphi = angle(exp(1i*(px - pb)));
    if nargin >= 5 && ~isempty(msk), dphi(~msk) = 0; end
end

function [psi, frames, vis] = psi_frames(Sx, Sr, thetas)
    nt = numel(thetas);
    frames = cell(1, nt);
    for q = 1:nt
        frames{q} = sum(abs(synth(Sx,thetas(q)) + synth(Sr,thetas(q))).^2, 3);
    end
    if nt == 4 && max(abs(thetas(:).' - [0 45 90 135])) < 1e-12
        % I(t) = A + B cos2t + C sin2t sampled at 2t = 0/90/180/270
        psi = atan2(frames{2}-frames{4}, frames{1}-frames{3});
    else
        t2 = 2*deg2rad(thetas(:));
        M = [ones(nt,1) cos(t2) sin(t2)];
        Sm = zeros(nt, numel(frames{1}));
        for q = 1:nt, Sm(q,:) = frames{q}(:).'; end
        c = M \ Sm;
        psi = reshape(atan2(c(3,:), c(2,:)), size(frames{1}));
    end
    % Visibility from the FITTED harmonic, never from min/max over the
    % samples: four samples of a cos(2t) fringe only touch its extremes when
    % the fringe phase happens to line up with them, so a min/max estimate
    % reads low (0.77 instead of 0.998 here) and looks like a contrast loss
    % that is not there.
    t2 = 2*deg2rad(thetas(:));
    M2 = [ones(nt,1) cos(t2) sin(t2)];
    Sm2 = zeros(nt, numel(frames{1}));
    for q = 1:nt, Sm2(q,:) = frames{q}(:).'; end
    cc = M2 \ Sm2;
    vis = reshape(sqrt(cc(2,:).^2 + cc(3,:).^2)./max(abs(cc(1,:)),eps), size(frames{1}));
end

function [vis, h4, h6] = harmonics(I, msk)
%HARMONICS  Fringe visibility and the 4-theta / 6-theta content of a stack of
%   analyzer frames, as pupil means over MSK.  The stack must span [0,180)
%   uniformly; FFT bin k holds the 2*(k-1)-theta harmonic.
    n = size(I,3);
    F = fft(I, [], 3)/n;
    h0 = abs(F(:,:,1));  h1 = 2*abs(F(:,:,2));
    h2 = 2*abs(F(:,:,3));  h3 = 2*abs(F(:,:,4));
    vis = median(h1(msk)./h0(msk));
    h4  = mean(h2(msk))/mean(h1(msk));
    h6  = mean(h3(msk))/mean(h1(msk));
end

function D = read_deck(fn)
%READ_DECK  Element / Surface / PropType per element, straight from the
%   emitted prescription -- so the order assertions are made against what the
%   engine will actually read, not against the builder's intent.
    L = regexp(fileread(fn), '\n', 'split');
    D = struct('element',{{}}, 'surface',{{}}, 'proptype',{{}});
    for k = 1:numel(L)
        tok = regexp(L{k}, '^\s*(\w+)=\s*(\S+)', 'tokens', 'once');
        if isempty(tok), continue; end
        switch tok{1}
        case 'Element',  D.element{end+1}  = tok{2};
        case 'Surface',  D.surface{end+1}  = tok{2};
        case 'PropType', D.proptype{end+1} = tok{2};
        end
    end
end

function [dxp, cen, rad] = det_geom(A)
%DET_GEOM  Detector sampling and the beam's FLUX-WEIGHTED centroid and
%   second-moment radius, in PIXELS.  Weighted, not thresholded: a mask
%   centroid is quantized to whole pixels and would report a reassuring exact
%   zero whatever the real misregistration was.
    macos.load_rx(A.rx);
    I = macos.intensity(A.iDET);
    dxp = macos.dx_at(A.iDET, 'mm');
    N = size(I,1);  [cg, rg] = meshgrid(1:N, 1:N);
    w = I/sum(I(:));
    cen = [sum(cg(:).*w(:)); sum(rg(:).*w(:))];
    rad = sqrt(sum(((cg(:)-cen(1)).^2 + (rg(:)-cen(2)).^2).*w(:)));
end

function [best, map] = register_to_dm(A, ix, Mdm, N_G, DX_G, h, msk)
%REGISTER_TO_DM  Build the instrument's pupil mapping from the trace itself
%   and use it to bring the injected DM map onto detector pixels.
    macos.load_rx(A.rx);
    s1 = macos.trace(ix.iTO);   ito  = macos.get_ray_info(s1.nRays);
    s2 = macos.trace(ix.iDET);  idet = macos.get_ray_info(s2.nRays);
    okr = ito.ok_trace(:) & ito.ok_pass(:) & idet.ok_trace(:) & idet.ok_pass(:);
    % DM-plane ray coordinates in the GRID frame the deck declares
    psi1 = macos.get_elt_psi(ix.iTO);  vpt1 = macos.get_elt_vpt(ix.iTO);
    u1 = macos.design.Bench.perp(psi1);  v1 = cross(psi1, u1);
    xy_to = [u1.'; v1.'] * (ito.pos - vpt1);
    psi2 = macos.get_elt_psi(ix.iDET);
    u2 = macos.design.Bench.perp(psi2);  v2 = cross(psi2, u2);
    xy_d = [u2.'; v2.'] * (idet.pos - idet.pos(:,1));
    xy_to = xy_to(:,okr);  xy_d = xy_d(:,okr);
    d = ito.pos(:,okr) - ito.pos(:,1);
    dch = ito.dir(:,1)/norm(ito.dir(:,1));  d = d - dch*(dch.'*d);
    r_beam = max(sqrt(sum(d.^2,1)));
    % affine part = the classical distortion report
    Aaf = [xy_d.' ones(nnz(okr),1)] \ xy_to.';
    Lm  = Aaf(1:2,:).';
    [Us,Ss,Vs] = svd(Lm);  sm = diag(Ss);  Rr = Us*Vs.';
    nl = xy_to - (Lm*xy_d + Aaf(3,:).');
    map = struct('mag', sqrt(abs(det(Lm))), 'anam_pct', 100*(sm(1)/sm(2)-1), ...
        'rot_deg', atan2d(Rr(2,1),Rr(1,1)), ...
        'nonlin_mm', sqrt(mean(sum(nl.^2,1))), 'r_beam', r_beam, 'Lm', Lm, 'Aaf', Aaf);
    Fx = scatteredInterpolant(xy_d(1,:).', xy_d(2,:).', xy_to(1,:).', 'linear','linear');
    Fy = scatteredInterpolant(xy_d(1,:).', xy_d(2,:).', xy_to(2,:).', 'linear','linear');
    % WF pixel centres -> detector mm.  The discrete row/col axis convention
    % is the classic grid-orientation trap, so resolve it empirically over all
    % eight candidates and PRINT the winner.
    N = size(h,1);  [cg, rg] = meshgrid(1:N, 1:N);
    cx = sum(cg(msk))/nnz(msk);  cy = sum(rg(msk))/nnz(msk);
    dxp = macos.dx_at(ix.iDET, 'mm');
    a1 = (cg-cx)*dxp;  a2 = (rg-cy)*dxp;
    c_d = mean(xy_d, 2);
    axs = ((1:N_G)-(N_G+1)/2)*DX_G;
    hm = h - mean(h(msk));
    cands = {a1,a2,'x=+col,y=+row'; a1,-a2,'x=+col,y=-row'; ...
             -a1,a2,'x=-col,y=+row'; -a1,-a2,'x=-col,y=-row'; ...
             a2,a1,'x=+row,y=+col'; a2,-a1,'x=+row,y=-col'; ...
             -a2,a1,'x=-row,y=+col'; -a2,-a1,'x=-row,y=-col'};
    best = struct('c',-inf, 'i',1);
    for c = 1:size(cands,1)
        [cc, ht] = reg_corr([0 0 0 0], cands{c,1}, cands{c,2}, c_d, Fx, Fy, axs, Mdm, hm, msk);
        if cc > best.c, best = struct('c',cc, 'ht',ht, 'name',cands{c,3}, 'i',c); end
    end
    A1 = cands{best.i,1};  A2 = cands{best.i,2};
    obj = @(p) -reg_corr(p, A1, A2, c_d, Fx, Fy, axs, Mdm, hm, msk);
    p = fminsearch(obj, [0 0 0 0], optimset('TolX',1e-7,'TolFun',1e-10,'Display','off'));
    [c2, ht2, Xt, Yt] = reg_corr(p, A1, A2, c_d, Fx, Fy, axs, Mdm, hm, msk);
    if c2 > best.c, best.c = c2;  best.ht = ht2; end
    best.hm = hm;  best.p = p;
    map.Xt = Xt;  map.Yt = Yt;
end

function [c, ht, Xt, Yt] = reg_corr(p, A1, A2, c_d, Fx, Fy, axs, Mdm, hm, msk)
%REG_CORR  Truth-vs-recovery correlation under a similarity adjustment
%   P = [dx dy rot log_scale] of the pixel->detector coordinates.  Those four
%   numbers ARE the instrument calibration a bench gets from a fiducial.
%   Spline resampling: bilinear costs hundreds of pm at actuator-scale
%   structure.
    s = exp(p(4));  ct = cos(p(3));  st = sin(p(3));
    X = s*(ct*A1 - st*A2) + c_d(1) + p(1);
    Y = s*(st*A1 + ct*A2) + c_d(2) + p(2);
    Xt = Fx(X,Y);  Yt = Fy(X,Y);
    ht = interpn(axs, axs, Mdm, Xt, Yt, 'spline', 0);
    ht = ht - mean(ht(msk));
    cc = corrcoef(hm(msk), ht(msk));  c = cc(1,2);
end

function show(Z, msk, N, box, cl, ttl)
    q = nan(N);  q(msk) = Z(msk);  q = sub(q, box);
    nexttile; imagesc(q, 'AlphaData', ~isnan(q)); axis image off;
    if ~isempty(cl), clim(cl); end
    colorbar; title(ttl);
end

function b = beam_box(msk, pad)
%BEAM_BOX  Bounding box of the illuminated pixels, padded -- the beam is a
%   small disc on the padded diffraction array, and an uncropped panel spends
%   90% of its area on black.
    [rr, cc] = find(msk);  N = size(msk,1);
    b = [max(1,min(rr)-pad) min(N,max(rr)+pad) max(1,min(cc)-pad) min(N,max(cc)+pad)];
end

function Z = sub(Z, b), Z = Z(b(1):b(2), b(3):b(4)); end

function m = erode_disc(msk, frac)
%ERODE_DISC  Keep the inner FRAC of the illuminated disc (no toolbox needed).
    N = size(msk,1);  [cg, rg] = meshgrid(1:N, 1:N);
    cx = mean(cg(msk));  cy = mean(rg(msk));
    r  = sqrt((cg-cx).^2 + (rg-cy).^2);
    m  = msk & (r <= frac*max(r(msk)));
end
