# zwfs_dm96 — a Zernike wavefront sensor against the PSI gauge, same DM truth

The ZWFS campaign (plan: `macos/BRIEF_zwfs_campaign.md`).  Question:
does a Zernike sensor beat the TG96 polarization-PSI gauge on the
differential benchmark (10 nm actuator deviation read to 0.021 nm,
base-independent — tg_psi_dm96 run 10)?

## Rulings (Dave 2026-09-04)

- **Ultimate target: measurement error ~ 1 pm.**  Report errors in pm
  everywhere; the pm road = differential protocol + actuator-space
  fitting + illuminated-pupil-compliant sampling + reconstructors
  beyond linear as needed.

- Scale: 96×96 rig; reduced-resolution dev runs allowed; **real work
  at 48×48 and 96×96** (battery + differential at both).
- Mask: single dimple first; **phase 2 = polarizing metasurface**
  producing two separate phase images (vector ZWFS), after the scalar
  system is built and tested.
- Optics: lens train first (identical to TG96 — only the sensor
  changes, so the comparison is clean).  OAP variant later, applied
  to both instruments.
- Location: this directory.

## The instrument

The TG96 TEST ARM ALONE (`twyman_green('polarizing',false)`, same
geometry, same tuned tail) — no reference arm, no polarizers, no
four-step.  The builder already places a `FocalMask` Reference element
at the internal focus of the detector leg; the ZWFS dimple (VSG2
hardware numbers: 346.2 nm etch in fused silica ≈ π/2 at 632.8 nm,
spot 9 = 1.06 λ/D — `40_benches/vsg_wip/vsg2_params.m` §9) is applied
there as a complex mask (`macos.apodize_complex`, the CTB idiom), and
the detector sees the reimaged pupil.  One frame per measurement.

Reconstruction (R1, linear): from three model-measured complex fields
on the flat DM — E0 (no mask), Eb (dimple-support disk only), and the
masked flat frame — the per-pixel linear coefficient is
2·Im[c·Eb·conj(E0)], c = exp(i·φm)−1; then h = ±φ·λ/(4π) (single
reflection doubles height; sign pinned by gate, not convention).
Three readings live in `../dm_gauge_lib/dmg_zwfs_gauge`: L (this
frozen-reference linear, 1 frame), S (phase-stepped, S2b, 4 frames),
and since S7 the ITERATED-REFERENCE EXACT reading I (per-pixel exact
solve, Ruane 2020 / N'Diaye 2013, with the reference wave re-propagated
from the estimate through the FFT surrogate of the mask model; 1
frame; 'I+' adds a one-time stepped retrieval of the working state as
a branch prior).

## Stages / gates

- **S1** (`zwfs_s1.m`): mask + response.  G0 focal-plane sampling
  printed and asserted (dimple ≥ 6 px across at the mask plane — THE
  new sampling interface); G1 exact superposition (masked field ==
  E0 + c·Eb to round-off — proves the mask bites and the algebra);
  G2 reference-wave sanity; G3 known low-order figure (parity-proof
  radial pattern) reconstructs with gain ≈ 1; G4 conjugate
  reconstruction FAILS (gain ≈ −1, non-vacuity).
- **S2**: bench at full res + two-poke registration (doctrine: 4 DOF
  classes; flip/transpose + sign are deck-dependent, never inherited).
- **S3**: battery — null, piston, single actuator, 12-mode transfer
  at 48×48 and 96×96; side-by-side vs tg96_report.txt.
- **S4**: differential head-to-head (the four rows) + dynamic-range
  break scale; R2 (exact inversion) if R1 limits.
- **Scoring ruling (Dave 2026-09-04):** S3/S4 score DM-state recovery
  in ACTUATOR space — fit the DM actuation model (the same influence-
  function forward model that builds the truth grids, through the
  registration affine) to the measurement, score actuator CHANGES.
  The TG96 map-space benchmark gets an actuator-space rescore
  (Stage E′ in tg_psi_dm96) so the head-to-head is one currency.
- **S5**: trades (spot table, etch error, leakage, chromaticity) on
  steer.

## Findings

> **MODEL CORRECTION (2026-09-09, found building S7's reconstructor;
> details in the S7 bullet at the end): every ZWFS number in S1-S6 was
> measured on a sensor whose reimaged pupil was Fresnel-DEFOCUSED by an
> effective 4.86 m.**  The `twyman_green` 'nf' sandwich emitted the exit
> reference sphere with zElt/Kr = 0.6*D_MASK_FL (23.86 mm) against the
> entrance sphere's 352.7 mm; the engine's SPH2PL applies a focal
> quadratic factor S ~ (Z2-Z1)*Z1/Z2 and PL2SPH is a plain FFT, so the
> round trip was not the identity the ctb_dcr.in precedent gets with EQUAL
> radii (its comment even says "matching sphere").  The ringed poke
> kernel (raw peak 0.27, corr 0.64), the OSCILLATORY fine-scale transfer
> with its null near 30 cyc/ap (a Talbot phase->amplitude null, predicted
> at 34), the 29% rms detector amplitude modulation under a 30 nm state,
> the 744 pm base-crosstalk floors and the "undetected grid-on-base"
> scenario were all this defocus.  'nf' now emits the SYMMETRIC sandwich
> (round trip 1.8e-15); 'nf_legacy' reproduces the old emission
> byte-for-byte (tBench gate).  **S1-S6 stand as the LEGACY-model record;
> S7 re-measures the S3/S4/S6 rows on the corrected model** (linear
> reading alone: single-on-base floor 744 -> 67 pm, grid-on-base SNR
> 1.46 -> 14).  The IFO twin is all-geometric and untouched.

- **S1 round 1 (G0 by design): the TG96 tail is geometric at the
  mask.**  Bench emits `PropType= Geometric` everywhere; the wavefront
  lands at FocalMask on a pupil-scaled 5.19 µm grid — dimple 0.54 px.
  Fix = the NF1/NF2 reference-sphere sandwich (`ctb_dcr.in` FPM idiom)
  as a `twyman_green` option `'mask_prop','nf'` (default 'geometric'
  emits bit-identically); Bench gained per-element `proptype` +
  conic/kr/zelt options on `add_reference` (defaults = legacy
  emission).  Landing dx scales as λF/D·NGRID/MODEL — dev runs use a
  SMALL ray grid (NGRID 65 at model 512 → 6.2 px dimple).
- **S1 rounds 2–5 (G2/G3): the mask plane was DEFOCUSED, ~5.6 mm.**
  L2's Kr is the l2_trade-OPTIMIZED value, not the thin-lens seed, so
  the true focus sits at MASK_TRIM = −5.58 mm; the axial peak is tens
  of µm deep (peak/sum 1e-4 wings → 1.22e-2 at focus), so mm-class
  scan steps straddle it — fminbnd in a ±3 mm bracket finds it, and
  S1 re-finds it every run (asserts peak ≥ 1e-2).  MISDIAGNOSIS
  CORRECTED: the round-2 "spot off the DC pixel" (~85 µm) was a
  speckle of the defocused blob, NOT the BS plate's lateral walk —
  at focus the spot lands exactly on the DC pixel (the SPH2PL
  transform centers on the reference-sphere axis).  The zwfs_mask
  settable center stays (harmless; measures 257.00 and centers there).
  Focused: core fraction 0.280 for the 1.06 λ/D disk — Airy encircled
  energy, as physics wants.
- **S1 round 5 (G3): defocus read at gain 0.54 — SUPERSEDED, see
  round 11.**  Interpreted at the time as ZWFS low-order
  self-referencing attenuation; the sign pin (S_CONV = −1) and the
  G3 move to a radial cosine stand.
- **S1 rounds 8–9: the truth FRAME was 25% off — ray affine fixes it.**
  The support-area pupil-radius estimate implied k = 0.754 of truth (a
  q=5 cosine decorrelates at half a rim cycle, so the fit read gain
  0.02 — "no response" that was pure frame).  The ray-measured
  DM→detector affine (registration doctrine DOF class 1, lifted from
  tg96) gives mag 10.158, anam 0.00%, nonlin 0.086 mm; a diagnostic
  scale sweep about the ray frame peaks at k* = 1.005, |corr| 0.997.
  Frame-before-angle, once more.
- **Round 11 (wf-figs, full res): the "low-order attenuation" was a
  THIRD frame artifact — pattern radius.**  The source cone fills
  only ~74% of the aperture (illuminated radius ~38 mm vs R_BEAM
  51.4), so every pattern defined on R_BEAM overhung the light and
  biased its fitted gain low (0.54 round 5; 0.324 round 10).  On the
  measured illuminated radius, **defocus reads at 0.986** (spot 2.0,
  model 1024/NGRID 193): with a ~1 λ/D-radius dimple the reference
  passband is ~0.5 cyc/pupil and defocus sits above it — the true
  self-reference attenuation lives at piston/tip/tilt class.  The S3
  transfer curve measures the real low-frequency edge.
- **Round 11, the poke:** one-frame single actuator (20 nm) reads at
  gain 0.445 (0.286 nm rms) at NGRID 193/spot 2.0.  Initially blamed
  in part on the illuminated-px sampling margin (1.49×) — **REFUTED
  by round 12's sweep**.  The illuminated-px accounting rule stands
  as doctrine (count lit px, not grid px), but it is not what limits
  the poke.  IFO same cases, same frames: poke 0.984 / 0.049 nm,
  defocus 1.024 / 0.086 nm (tg_psi_dm96/tg96_wf_figs.m).
- **Round 12 (zwfs_sweep.m, Dave's ask): sampling is NOT the binder —
  the SPOT is a band-select lever.**  Across NGRID 129→385 (detector
  margin 1.7×→5.0×; DM-side 1.3→3.7 rays/actuator) the poke gain is
  FLAT at fixed spot: 0.445 (spot 2.0, three configs), 0.585–0.589
  (spot 3.0, four configs).  It moves with the SPOT, in opposite
  directions per band: poke −0.381 → 0.445 → 0.585 as the spot grows
  1.06 → 2.0 → 3.0, while defocus FALLS 1.030 → 0.986 → 0.79.  The
  1.06 hardware spot sign-inverts the high-frequency response.  No
  single spot serves both bands with the fixed flat-state linear
  reconstructor — the transfer is stable and calibratable, which is
  exactly what the actuator-space fit (Dave's scoring ruling) and a
  measured interaction matrix absorb; that is S3's reconstructor
  path.  Errors in pm: poke 240–286, defocus 137–757, vs IFO 49/86
  and the 1 pm ultimate target — the pm road is calibration +
  differential, not raw single-frame transfer.
- **S1 GREEN (round 10, 20 s/run):** dimple 6.29 px (G0); masked field
  == E0 + c·Eb at 3.3e-16 (G1); core fraction 0.2801 ≈ Airy encircled
  energy for 1.06 λ/D (G2); q=2 radial cosine at 8 nm recovers gain
  +0.932, resid 0.45 nm (G3, ray frame); no-dimple frame reads −0.07
  (G4 — the signal is the dimple's).  S_CONV = −1 pinned.  The
  round-10 defocus property (0.324) is SUPERSEDED by round 11
  (pattern-radius bias; corrected 0.986).  q=5 read 0.675 at dev res
  = dev sampling rolloff — transfer curves belong to S3 at full res.
- **S2 GREEN (zwfs_s2.m, 0.3 min): registration + measured-kernel
  reconstructor.**  Two-poke registration at 20 nm strokes: parity 1,
  sign +1 in this script's candidate basis (deck-dependent, as
  doctrine says; selection 0.527 vs 0.025 runner-up — the gate is
  SELECTION CONFIDENCE, not map fidelity: the ringed ZWFS kernel caps
  raw correlation near 0.5 by physics, so the IFO's 0.8 bar does not
  transfer).  Calibration by the multiplexed-poke doctrine (measured
  response kernel + lattice deconvolution, Tikhonov pcg): held-out
  poke recovers at **gain 0.903, 132 pm** (raw transfer was 0.445);
  kernel spatial variation measured at 13% (poke B via A's kernel:
  0.868) — the single-kernel bound; field-dependent kernels are the
  refinement if S3 needs it.  **Dense-random single-shot remains
  transfer-limited**: the λ scan is monotonic to λ=3.2 with gain
  collapsing (0.75→0.21) — no optimum, and fit-removed resid rewards
  over-smoothing (metric caveat recorded).  The treatment is S3's
  MODAL calibration (measure the transfer on the lattice modes), not
  more λ.
- **Sensitivity stage (zwfs_sens.m / tg96_sens.m twins, Dave's ask):
  NO additive floor down to 0.1 pm in the noiseless model.**  Across
  10 nm → 0.1 pm (single poke + 47-site grid poke × flat/30 nm base,
  differential, actuator space): gain constant, accuracy = (1−g)·amp
  exactly, unpoked floor PROPORTIONAL to amplitude (crosstalk, not
  noise), so detection SNR is amplitude-independent.  SNRs: IFO
  single 207 / grid 19.6 (base-independent); ZWFS single 59 flat /
  11.5 on base; grid 11 flat / **1.64 on base = the one UNDETECTED
  scenario** (36% proportional leakage — the linear reading's
  base-induced crosstalk).  First numerics whisper: ~0.1–0.3% gain
  jitter at the 0.1 pm rows only.  The 1 pm objective in the no-noise
  limit: met by linearity everywhere except ZWFS-grid-on-base;
  ACCURACY is multiplicative — calibrate the constant gain and the
  residual is gain stability (~0.1%).
- **S2b (zwfs_s2b.m): multi-DEPTH mask phase stepping (Dave's ask).**
  STRUCTURAL FINDING: |c|² = −2Re(c) identically, so a depth ladder
  yields only TWO observables per pixel — |Eb|² is NOT
  self-calibrating from depth steps (one-time calibration, or a
  diameter change; hardware note for the multi-depth substrate).
  Rank-2 solve is exact (frame consistency 6e-16).  Results: poke
  gain 0.519 vs 0.445 linear (the linearization error removed; the
  remaining deficit is optical and calibrates); **RANGE: a 150 nm
  poke (3.0 rad — the linear reading folds) recovers within 9% of
  the 20 nm gain**; defocus-on-base reads 0.743 — the self-reference
  attenuation reappears under EXACT retrieval because the reference
  core MOVES with a low-order deviation, while the frozen-reference
  linear reading (0.986) avoids it.  **Doctrine: the reconstructors
  are complementary — stepped for range/exactness/high-f, frozen-
  reference linear for small low-order differentials.**  4 frames
  per measurement vs the IFO's 6 traces.
- **S3 (zwfs_s3.m / tg96_s3.m): the modal-calibrated battery, both
  DM sizes (96×96 @ 1 mm; 48×48 @ 2 mm, DST-class, same bench).**
  IFO: smooth monotonic transfer (1.02→0.58 by 57 cyc/ap; near-
  isotropic), corrected poke 0.92 / 92 pm (96²) and 0.95 / 67 pm
  (48²), random 10 nm at 4.2 / 1.7 nm, piston 0.98.  ZWFS: transfer
  SEPARABLE (validated ~2%) but OSCILLATORY at fine scales — the
  ringed kernel's FFT crosses zero ((64,0) −0.52, (80,0) +1.04), so
  interpolated modal corrections destabilize there; the low end dips
  where modes enter the dimple passband ((2,0) at 1 cyc/ap: 0.50);
  piston structurally invisible (expected).  Corrected poke 1.01 /
  157 pm; dense random stays the weak axis (12.5 / 5.5 nm).
  **THE RESCUE: grid-on-30nm-base — the sensitivity stage's one
  undetected scenario (SNR 1.64) — is DETECTED at 48×48 through the
  phase-stepped retrieval: SNR 9.15 (linear reading: −1.23), floor
  97 pm on 1 nm pokes.**  At 96×96 stepped reaches 2.32 — fine-pitch
  content is where the sensor genuinely dims.  OPEN (named path):
  dense per-frequency 1-D calibration (~95 frames along one axis, or
  a stepped-retrieval kernel) for dense-command fine-scale work; or
  concede that axis to the IFO — the instruments are complementary
  by measurement, not by assumption.
- **Machinery factored (2026-09-05): `../dm_gauge_lib` is the ONE copy**
  of the scoring machinery (registration / sampling / actuator fit /
  modal correction / both instruments' measurement factories).
  `zwfs_s3.m` retrofitted onto it and re-run: report IDENTICAL to the
  10cf593 record (the equivalence gate).  S1/S2/S2b/sens keep their
  private copies as history; S4 consumes the lib only.
- **S4 (zwfs_s4.m): the head-to-head rows + break scale, in pm.**
  Four differential rows through the CALIBRATED estimator (measured
  kernel + separable Wiener from zwfs_s3.mat), BOTH readings per row:
  flat/single 144 pm (linear) / 125 pm (stepped); on the 30 nm working
  state the linear gain collapses to 0.66 while STEPPED HOLDS 1.13
  (773 pm) -- the complementary-reconstructor doctrine in actuator
  currency.  Dense random stays the weak axis (10-42 nm).  BREAK
  SCALE (single-act 10 nm differential on a growing base): linear
  folds beyond ~30 nm rms working state; stepped keeps a usable gain
  to ~60 nm (0.68 at 96x96) and beyond ~120 nm both readings stop
  being MEASUREMENTS (sign-flipped / aliased single-site recoveries
  that still clear a formal detection SNR -- detection without
  measurement; quote the gain column, not SNR, past the fold).
  VERDICT vs the IFO (tg96_s4): the IFO does not break -- 46 pm
  single-row, base-independent, and a 480 nm rms working state costs
  it ~5% gain / ~25% floor.  The scalar ZWFS serves flat-ish states
  and small working states via stepped; it cannot follow the IFO into
  large working states.  Its remaining card is photon economy (4
  frames vs 6 traces) -- the noise stage decides its niche.
- **S5 (zwfs_s5noise.m / tg96_s5noise.m): photon noise prices the
  1 pm target -- and settles the economy question.**  Method: the
  optical fields are noise-independent, so noiseless frames are
  captured once per DM state and shot noise is Monte-Carloed
  numerically (trace-free); calibration is treated as noiseless (the
  long-exposure assumption); axis = photons per DM STATE, split across
  each reading's frames (linear 1, stepped 4, four-step 4) -- equal
  light, equal time.  On the head-to-head scenario (single 10 nm act
  on the 30 nm base, 96x96): noise sigma at the poked site follows
  1/sqrt(N) cleanly, and each instrument's high-N floor converges to
  its S4 systematic floor (internal consistency).  PRICES:
  IFO four-step sigma ~ 2.8e7/sqrt(N) pm -> N(1 pm) ~ 8e14
  photons/state; ZWFS linear ~ 3.7e7/sqrt(N) -> 1.4e15; ZWFS stepped
  ~ 7.0e7/sqrt(N) -> 5e15.  VERDICT: per unit light the modalities
  are within ~2x of each other -- photon economy does NOT
  discriminate; the discriminator is systematics (the IFO's
  working-state immunity).  And ~1e15 photons/state at 633 nm is
  ~0.3 mJ -- trivial for a bench source -- so photon noise is NOT the
  1 pm blocker; gain stability (~0.1%, the sensitivity-stage finding)
  and the systematic floors are.
- **S6 COLOR (zwfs_s6color.m, Dave 2026-09-08: "try running both
  systems at multiple colors, maybe the combination will help with some
  poor SNR regions"): YES for the ZWFS -- the transfer nulls are
  CHROMATIC and a multi-color combination fills them.**  Five colors
  (480/532/632.8/700/780 nm) through ONE physical mask (the 346.2 nm
  etch, 2.0 lam/D at 633; Malitson fused-silica index, so the dimple
  reads 2.10/1.88/1.57/1.41/1.27 rad and 2.64/2.38/2.00/1.81/1.62
  lam/D across the set); only the deck header `Wavelen=` changes per
  color, and EVERY calibration (flat references, den, b2cal, anchor,
  measured kernel, 13-row modal transfer) is redone per color; 96x96,
  model 1024 / NGRID 193 (dimple 10.4 -> 6.4 px, all above the 6-px
  rule).  Combination = the multi-channel Wiener on the actuator
  lattice (`../dm_gauge_lib/dmg_color_comb`, a_hat(f) = sum_k G_k A_k /
  (sum_k G_k^2 + beta^2)), scored against every single color, the plain
  mean, and the best pair.  THE MECHANISM, measured: the oscillatory
  null of the (p,0) transfer sits near 30 cyc/ap at 632.8 (0.20 at 28,
  -0.52 at 32), and moves as 1/lambda with the dimple's angular size --
  40 cyc/ap at 480, 36 at 532, 24 at 700, 20 at 780 -- so no two colors
  are null together and the five-color transfer after Wiener never
  drops below **0.991** where the best single color bottoms at 0.91
  (532) and 632.8 at 0.80.  ROWS (linear reading, pm; 632.8 single ->
  five-color comb): flat hold-out 202 -> **143** (gain 1.16 -> 1.08,
  the over-correction eased); dense random 10 nm 16440 -> **7203**
  (2.3x; best single 532 at 10899); single 10 nm on the 30 nm base:
  floor 720 -> **224**, SNR 9.1 -> **35**; random 10 nm on base 41180
  -> 17305; and the sensitivity stage's one UNDETECTED scenario (grid
  47 x 1 nm on base) SNR 1.46 -> **3.43** (mean of singles 2.97, best
  pair 2.63 -- more colors keep helping, slowly) -- still under the
  SNR-5 line at 96x96.  Stepped reading: the base rows improve the same
  way (single-on-base floor 755 -> 198, SNR 14.8 -> 36; best pair
  480+532 reaches 49; grid 2.28 -> 3.32), dense random 36581 -> 13822,
  but the FLAT hold-out gets slightly WORSE (287 -> 313): the stepped
  retrieval's own systematic is larger at the other colors (607-1158
  pm) and an equal-weight combiner inherits it -- weight by a measured
  per-color systematic (the flat rows) before combining stepped
  readings.  Why the base rows gain most: the linear reading's base
  crosstalk is a second-order term whose pattern differs per color
  (phi_base ~ 1/lambda, c(lambda)), so the floors partly average AND
  the Wiener no longer amplifies a near-null mode in any one color.
  Checks: the 632.8 column reproduces the record (dimple 7.92 px, tie-in
  g 1.158 / 203 pm vs S3's 1.181 / 212 on 12 probes); the combiner's DC
  normalization (g1(0)=1 vs the record's lowest-probe extension) moves
  the grid SNR 3.32 vs 3.40 -- immaterial.  Cost: K x the frames (5
  frames linear, 20 stepped per DM state); noiseless model -- at equal
  TOTAL light the shot-noise part is neutral (1/sqrt(K) per color,
  recovered by the combination) while the systematic part improves,
  and S5 says systematics are what block 1 pm.  Per-color decks
  `zwfs_test_<nm>nm.in` are regenerated by the script (gitignored).
  IFO twin: `../tg_psi_dm96/tg96_s6color.m`.
- **S7 (zwfs_s7iter.m, 2026-09-09): the model correction + the
  iterated-reference exact reading (literature import #1).**
  *Block A, legacy on record:* unmasked entrance->exit round trip
  0.159 (flat DM!); SPH2PL's focal factor S = 8.60e-5 rad/px^2 (the
  identity E_mask == exp(iS r^2) IFFT(E_in) holds to 1.1e-6 -- the deck's
  6-digit zElt print; DFOURN's +i sign = MATLAB's ifft2); z_eff =
  Z1(Z1-Z2)/Z2 = 4860 mm -> Talbot null predicted at 34.1 cyc/ap vs the
  S3 record's 28-32; 30 nm base: |E|/|E_flat| std 0.290 (range
  0.017..2.38) where a phase-only state must give 1; linear center-poke
  kernel raw peak 0.268 / ring -0.158 / corr 0.635; S2 tie-in
  reproduced (hold-out 0.9029 / 132 pm).  *Block B, corrected gates:*
  round trip 1.8e-15; the b surrogate T(D Ti(E)) vs the engine's Eb
  2.0e-15 (the engine's NF2 leg IS a shifted FFT and the geometric tail
  is the identity on the grid); amplitude modulation 4e-16; the exact
  solve with the ORACLE b and the true branch is exact to 2.8e-14 on
  every pupil pixel (the algebra), so the reading's errors are (i) the
  BRANCH -- on the rng(7) 30 nm base 7.8% of pupil pixels sit beyond the
  quarter-wave sensor's fold (phi - Theta > 0, the -pi/4 bound), a
  per-pixel ambiguity of one dimple frame -- and (ii) PISTON: the
  intensity is invariant under a common phase on E and b, so an
  ITERATED b converges to the truth up to a constant (the sensor's
  piston null, S3; the oracle b carries the true piston; measured: the
  refined-prior residual IS a piston, mean -0.0902 rad = |b_iter -
  Eb1|/|Eb1| 0.0902).  Map space on that base, piston removed (rad rms
  vs the true detector phase, 0.537 rad rms signal): L 0.259 | F 0.247
  | I 0.235 | I+ with the plain stepped prior 0.034 (it agrees with the
  true branch on 97.0% of pixels) | **I+ with the REFINED prior 2.8e-4**
  (`dmg_zwfs_gauge` priorS: the stepped retrieval re-solved with the
  iterated |b|^2, two passes -> 99.99% of pixels on the true branch) --
  the whole 30 nm working state read from ONE frame to 5e-4 of itself.
  Flat hold-out 20 nm, piston removed: L 8.5e-4, F 7.3e-5, **I 7.5e-7
  rad** (the iteration converges 6e-3 -> 1e-6 in five steps; raw with
  piston 8.9e-4 / 2.4e-4 / 2.3e-4).  *Block C, 96x96:* kernels
  ring-free (L 0.751 / I 0.928 / S 0.928 peak, corr 0.98-0.995); modal
  transfer SMOOTH -- L 1.04-1.18 from 2 to 40 cyc/ap, I 1.03 -> 0.89, S
  0.96 -> 0.89, the only dip the dimple passband at 0.5-1 cyc/ap (L
  0.72/0.50) -- the (64,0) -0.52 null is GONE.  Rows (raw actuator-space;
  legacy in brackets): flat/hold20 L 0.920 / 68 pm, all exact readings
  0.900 / 61 pm [1.158 / 202 corrected]; flat/rand10 I 0.88 / 2.2 nm, S
  2.3 nm [16.8 / 10.2]; rand30/single10 floor+SNR: L 67 pm / 80 [744 /
  9.2], F 30 / 192, I 36 / 204, **I+ 13 pm / 584**, S 23 / 312 [755 /
  15]; rand30/grid@1nm SNR: **L 14.1 [1.46], F 11.9, I 13.4, I+ 33.5, S
  36.7 [2.28] -- the sensitivity stage's one undetected scenario is
  DETECTED by every reading**; rand30/rand10: L 9.6 nm [42], I+ 5.8, S
  3.3 [37.6].  48x48: hold-out exact readings 0.997 / 39 pm (L 0.949 /
  65); rand30/single10 **I+ 1.009 / 24 pm / SNR 466** where the one-frame
  readings WITHOUT the refined prior read sign-FLIPPED (L -0.82, F -1.43,
  I -2.07: the base put 29.5% of that actuator's footprint beyond the
  fold and the plain stepped prior saw 2.3% of it -- `fold_site` probe);
  grid-on-base SNR L 16.4, I 14.8, **I+ 115**, S 99; dense random S 2.4
  nm, I+ 14 nm (dense commands stay the stepped reading's).  *Break
  scale (single-act 10 nm differential on a growing base; true
  beyond-fold fraction at 96x96 7.7 / 12.6 / 16.9 / 20.6% at 30 / 40 /
  50 / 60 nm rms, which the refined prior finds to 4 digits; the plain
  one 4.7 / 7.2 / 8.6 / 8.7%):* g(SNR) at 96x96 -- L 0.51(85) 0.41(53)
  0.32(35) 0.24(24); F 0.60(161) 0.73(108) 0.24(25) 0.03(5); I 0.80(185)
  0.09(11) 0.09(12) 0.00(0); **I+ 0.85(266) 0.86(201) 0.91(179)
  0.86(147)**; S 0.85(202) 0.73(194) 0.61(131) 0.56(103); at 48x48 I+
  1.00 / 1.07 / 1.01 / 1.05 (SNR 97-132) vs S 0.97 / 0.89 / 0.76 / 0.52.
  **So the cliff the un-primed iterated readings have between 30 and 40
  nm rms (wrong-branch pixels re-propagated INTO b corrupt the good
  ones) is the PRIOR's, not the sensor's: with the refined prior the
  one-frame reading holds to 60 nm rms and beats the four-frame stepped
  one there.**  Past ~120 nm rms the stepped retrieval itself wraps
  (prior fold fraction 0) and every reading aliases (S4: quote gain, not
  SNR).  *Spec verdicts:* grid-on-base SNR >= 5 from ONE frame: MET (I
  13.4 at 96, I+ 33.5; the model correction alone gets L there);
  hold-out RAW gain within 3% of 1: MET at 48x48 (0.997), 0.900 at
  96x96 -- NOT the reading and NOT the Tikhonov (lambda 0.05 -> 0.002
  moves it 0.900 -> 0.906; the map's own peak ratio is 0.911): the
  NGRID-193 dev ray grid samples a 1 mm actuator at ~2 px; the
  compliant NGRID 385 run is the check (sampling doctrine, S1 round
  11).  **Doctrine, updated:** the model correction is the bigger lever;
  on the corrected sensor the one-frame iterated reading with the
  refined base prior (I+; the base's 4 stepped frames taken ONCE) is the
  best small-actuator-count reading to at least 60 nm rms working state
  (13-24 pm floors, SNR 466-584 on the head-to-head row); the frozen
  exact (F) is the prior-free one-frame reading to 40 nm; the stepped
  (4 frames) owns dense commands.  Follow-ons (named): NGRID 385; re-run
  S6 color on the corrected model (the chromatic null was largely
  Talbot); the S5 noise pricing of I+ (its frames carry the prior's
  noise too); deck fold.  Figure `zwfs_s7iter.png`; report
  `zwfs_s7iter_report.txt`; `zwfs_test_legacy.in` re-emitted by the
  script (gitignored).
