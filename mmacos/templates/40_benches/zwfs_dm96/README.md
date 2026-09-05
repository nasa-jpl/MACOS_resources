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
- (further stages appended; reports are the record)
