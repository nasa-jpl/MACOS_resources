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

## Run it yourself (`zwfs_run`)

One entry point drives the whole system; the per-stage scripts
(`zwfs_s1.m` .. `zwfs_s7iter.m`) are the historical record, not the way
to run it.  The defaults in `zwfs_params.m` are the values of record, so
a bare call reproduces the S7 battery -- the runner's own equivalence
gate: `runs/rec193/rec193_report.txt` against `zwfs_s7iter_report.txt`,
64 row/ladder lines compared, 8 differ in the last printed digit only
(the actuator fit's pcg tolerance), every gate value identical.  Since
2026-09-10 ONE default differs from the S7 record on purpose --
`reg.stencil_site` 'lattice' (S9 bullet); pass `'reg.stencil_site','grid'`
to reproduce S7 exactly.

    cd templates/40_benches/zwfs_dm96
    out = zwfs_run;                                    % bench + battery + figs, defaults
    out = zwfs_run('tag','ng385', 'NGRID',385);        % the 1 Mpix-class detector
    out = zwfs_run('tag','s3', 'mask.DIA_LAMD',3, 'stages',{'battery','color','noise','figs'});
    P = zwfs_params;  P.dm = P.dm(1);  P.readings = {'L','I+'};  out = zwfs_run(P, 'tag','quick');

Headless, memory-capped, logged (a MODEL 1024 run takes ~11 GB; run
ONE at a time -- two model-1024 MATLABs have taken this box down):

    ./zwfs_batch.sh ng385 "'NGRID',385, 'stages',{'battery','figs'}"     # log: runs/ng385.log
    ZWFS_MEMMAX=20G ./zwfs_batch.sh m2048 "'MODEL',2048, 'NGRID',385, 'param_file','macos_param_2048.txt'"
    ./zwfs_batch.sh loop193 "'stages',{'bench','loop','figs'}"                # the closed-loop hold metric (S11), ~80 min

Everything lands in `runs/<tag>/`: `<tag>_report.txt` (every number,
every gate with its threshold), `<tag>.mat` (`out` = P + bench + battery
+ color + noise), the emitted deck(s), and the PNGs
(`zwfs_run_figs(out)` or `zwfs_run_figs('runs/<tag>/<tag>.mat')`
re-draws them from a saved run).

| knob (`zwfs_params`) | default | what it does |
|---|---|---|
| `stages` | bench battery figs | add `color` (the multi-wavelength combination), `noise` (photon pricing) and `loop` (the closed-loop HOLD metric, S11) |
| `readings` | L F I I+ S | any subset: linear / exact frozen-b / exact iterated-b / I with the base's refined stepped prior / phase-stepped |
| `MODEL`, `NGRID` | 1024, 193 | engine grid, ray grid across the aperture (385 = the 1 Mpix-class detector) |
| `param_file` | `''` | a custom engine size table (`macos_param.txt` namelists) copied into the run dir, where the engine looks FIRST; `'macos_param_2048.txt'` (this dir) trims MODEL 2048 to fit a 30 GB box -- `mGridSrf` 200 -> 4, `mpts` -> 512, `mElt` -> 64, and `mGridMat` UP to 512 for the 384-across DM grid (the stock 2048 entry's 128 corrupts the heap) |
| `LAM` | 632.8 nm | the record color; `color.lams_nm` lists the others, record color FIRST |
| `bench.*` | the tg96 test arm | every `twyman_green` option: lenses, legs, tuned tail, `mask_prop` (`'nf'` = the corrected symmetric sandwich; `'nf_legacy'` = the Fresnel-defocused S1-S6 sensor) |
| `mask.*` | 346.2 nm etch, 2.0 lam/D | etch depth, substrate index (`'malitson'` or a number), dimple diameter, the phase-stepped depth ladder, `NITER` |
| `samp.*` | 6 px dimple, 2 px/actuator | the sampling-budget lines the bench stage asserts; `enforce` = `'warn'` or `'error'` |
| `reg.*` | `'search'` | parity + sign from an off-center poke (two-poke doctrine; the selection metric is the gate), or `'record'` to take `PARb`/`sgn` as given |
| `dm(i)` | 96x96 @ 1 mm; 48x48 @ 2 mm | actuator count, pitch, hold-out site, modal probes -- each config gets the full battery |
| `battery.*` | 30 nm base; 10 nm devs; 1 nm grid | amplitudes, seeds, Wiener beta, Tikhonov weight, the break-scale ladder, which rows |
| `color.*`, `noise.*` | 5 colors; 1e6..1e14 photons/state | the optional stages' own knobs (readings, rows, combiner form; realizations, prior treatment) |
| `loop.*` | L I+ S; g 0.5; 60 cycles; 1e12..1e15 photons/cycle | the closed-loop hold stage: readings, set point (`'base'` = the working surface with the matrix ON it), gain, cycles, photon levels, drift models (`walk_sigma` 2 pm/actuator/cycle, `thermal_rate` 5 pm/cycle), noiseless `steps`, the noise-only `floor`, reference frames `'noiseless'` or `'noisy'`, the drift `seed` (shared with the IFO), `hold_spec` 3 pm |

Rules: one engine model size per MATLAB process (a second `macos.init`
at another size corrupts the heap); `exit(0)` lives only in the batch
wrapper `zwfs_run_batch`, never in `zwfs_run`; the sampling budget
WARNS by default so a deliberate dev grid still runs -- read the
budget lines before quoting a number.

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
- **S11 (zwfs_run stage 'loop', 2026-09-11, Dave: "on-orbit the DM
  surface needs to remain constant to << 10 pm, with frequent
  remeasurement and closed-loop DM actuator servo control -- how can
  performance in this mode be made into a metric?"): the CLOSED-LOOP
  HOLD metric.**  The DM is held at the 30 nm working surface by a
  proportional loop (gain 0.5, 60 cycles) closed through ONE reading:
  each cycle the state is traced, photon noise injected (N photons per
  state, a reading's frames share it), the differential to the set
  point's frames fitted through the response matrix measured ON that
  surface (S10), and g times the estimate removed.  The loop code is
  shared with the interferometer (`../dm_gauge_lib/dmg_loop.m`, gated by
  `tests/tDmgLoop.m` on a synthetic instrument: contraction 1 - gG,
  noise-only steady state sigma_n sqrt(g/(2-g)), the random-walk and
  ramp laws, a biased reading converging to a non-zero surface, seeded
  drift, a single-shot reference as a fixed bias).  The metric: the
  steady-state hold error (rms over lit, pm) against a drift, as a curve
  in photons per cycle; the ONE number = photons per cycle to hold 3 pm.
  Drifts: random walk 2 pm per actuator per cycle; thermal ramp 5 pm rms
  per cycle of defocus + astigmatism; noiseless steps of 1 and 10 nm
  (time constant, dynamic range); the noise-only floor.  `P.loop.*`;
  `runs/loop193` (98 min, 2562 traced states; 385-ray confirmation
  `runs/loop385`).
  *What closed loop changes, measured:* the loop propagates noise
  exactly as theory says on the real instrument (noise-only: L 4.39 /
  1.39 / 0.44 / 0.14 pm at 1e12..1e15 photons per cycle vs 4.25 / 1.35 /
  0.43 / 0.13 from the in-run single-shot noise; S 4.81 / 1.52 / 0.48 /
  0.15 vs 4.72 / 1.49 / 0.47 / 0.15), and the random walk likewise (L
  5.02 / 2.77 / 2.42 / 2.38 vs 4.84 / 2.67 / 2.35 / 2.31; S 5.33 / 2.77 /
  2.36 / 2.32 vs 5.26 / 2.75 / 2.36 / 2.31: the 2.31 pm floor is the walk
  itself at g = 0.5, sigma_d / sqrt(g(2-g))).  So the per-photon
  comparison in this mode is the single-shot noise: **L and S are the
  same per photon per state** (sig_n 7.4 vs 8.2 pm at 1e12), and both
  hold 3 pm from ~2e12 photons per cycle (noise only) / ~7.4e12 (walk).
  *What discriminates is the SYSTEMATIC term, the thing the loop was
  built to expose:* (1) **the stepped reading S has NO noiseless floor**
  -- a 1 nm and a 10 nm step both converge to 0.000 pm (rho 0.61-0.63,
  tau ~2 cycles), and its thermal hold is 10.05 pm = exactly the
  proportional-loop lag rate/g of a unit-gain reading (theory 10.00;
  low-order gain 0.995).  (2) **The linear one-frame reading L holds the
  walk and the noise like S but not a persistent low-order residual**:
  its thermal hold is 27.6 pm and still creeping at cycle 60 -- 9 pm of
  the same lag plus 26 pm of HIGH-frequency error (> 12 cyc/ap) that the
  loop imprints on the DM, distributed (15 pm with the worst 100
  actuators removed; two 4-actuator clusters at 0.5-0.8 nm), and its
  noiseless steps decay with a slow mode (rho 0.82-0.83: 1.2 pm left at
  cycle 60 from 1 nm, 9.8 from 10 nm).  Mechanism: on the working
  surface the linear reading's local sensitivity is near zero at some
  sites, the loop is nearly open there, and any crosstalk from a
  persistent residual integrates to bias/(g G_site) -- a zero-mean walk
  does not excite it, a ramp does.  (3) **The exact one-frame reading
  with the set point's branch prior (I+) DIVERGES**: a 1 nm step grows
  to 99 nm in 60 cycles, a 10 nm step to 178 nm, and even the noise-only
  loop at 1e15 photons wanders to 0.9 nm -- actuators whose footprint
  sits beyond the quarter-wave fold read with the wrong sign (S10's fold
  sensitivity), the loop pushes them the wrong way, and the fixed branch
  prior gets wronger as they move; no gain fixes a negative gain.  I+ is
  not a loop reading on a 30 nm surface (`P.loop.rmax` now stops such
  runs at 1 um).  *The one-number table (photons per cycle to hold 3 pm
  rms):* noise-only L 2.1e12 / S 2.6e12 / I+ diverged; walk L 7.3e12 / S
  7.5e12 / I+ diverged; thermal: no photon count reaches 3 pm at g = 0.5
  (a proportional loop lags a ramp by rate/g = 10 pm), the floors are L
  27.6 / S 10.0 / I+ diverged -- an integral term or a higher gain is the
  thermal fix, and the same loop code takes it.  DOCTRINE for the
  head-to-head: run the IFO through the identical `dmg_loop` (same
  seeds, drifts, gains, photon levels; CCMac, brief oap2 addendum) and
  compare the three rows of that table plus the noiseless floor and the
  held residual's spectrum.

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

- **S8 (zwfs_run, 2026-09-10, Dave: "work down the open list; a
  parameterized runner users can modify and rerun without AI"): the
  RUNNER, and the three open items measured through it.**  Runs are in
  `runs/<tag>/` (report, .mat, deck, PNGs); every number below is from
  those reports.  *Equivalence gate:* `runs/rec193` (defaults) against
  `zwfs_s7iter_report.txt` -- 64 row/ladder lines, 8 differ in the last
  printed digit (pcg tolerance), every gate value identical.
  *NGRID 385 (open item 2):* three configurations -- `ng385` (model
  1024, spot 2.0: dimple 3.96 px at the mask, FAILS the 6-px line;
  5.03 px/actuator), `ng385s3` (1024, spot 3.0: 5.94 px), and `m2048`
  (MODEL 2048 via the trimmed size table `macos_param_2048.txt`, spot
  2.0: 7.92 px AND 5.03 px/actuator -- the first fully COMPLIANT run;
  32.5 min, < 4 GB resident).  Findings: (i) the 96x96 hold-out raw gain
  moves 0.900 -> 0.935 from NGRID 193 to 385 and is then IDENTICAL at
  model 1024 and 2048 (0.9348 / 0.9348) and spot-independent (0.9347 at
  spot 3.0); at 48x48 it moves the other way (0.997 -> 0.964).  So the
  S7 "sampling" attribution was half right: NGRID moves it, the dimple
  sampling does not, and the remaining 6.5% is neither -- the named
  suspect is kernel spatial variation between the centre (where the
  kernel is measured) and the hold-out site (S2 measured 13% on the
  legacy model).  (ii) Every actuator-space row at 385 is the same at
  model 1024 and 2048 to the third digit (I+ single-on-base 0.913/25 pm/
  SNR 359 vs 0.914/26/355; grid-on-base 37.4 vs 37.3) even though the
  reference-wave profile |Eb|/|E0| (the new bench-stage diagnostic; a
  function of lam/D only, so it must not depend on the ray grid) differs
  by 2.5% at pupil centre between the 4-px and 8-px dimples -- the
  gray-edged 4-px dimple is adequate for actuator-space results at the
  1% level; the 6-px rule stays as the budget line.  (iii) THE SAMPLING
  TRADE (Dave): mask-plane px per lam/D = fill*MODEL/NGRID (0.74 here),
  detector px per actuator ~ NGRID -- the two lines pull opposite ways
  in NGRID; only MODEL buys both.  (iv) Spot 3.0 only deepens the
  dimple-passband dip below 2 cyc/ap (min gain 0.22 vs 0.50), transfer
  identical above -- spot 2.0 stays the sensor of record.  (v) The
  break-scale ladder on 47 grid sites (`battery.ladder_sites` 'grid';
  the record's single hold-out site made "I+ holds to 60 nm" a
  one-site statement): at 96x96 I+ holds to 40 nm rms (gain 0.81-0.85,
  SNR 26-30) and collapses at 50-60 (floor 0.7-3 nm: a subset of sites
  beyond the fold), the stepped reading holds to 50-60 (0.83/0.66);
  at 48x48 I+ holds to 50 (0.84-0.89) and S to 50 (0.87-0.99).  The
  multi-site floor (300-600 pm) is the crosstalk of 47 simultaneous
  10 nm pokes, a different quantity from the single-site floor.
  *S6 colour re-run on the corrected model (item 3, `rec193full`):*
  colour is NOT a lever any more.  The chromatic transfer null left with
  the Talbot artefact, so the 5-colour combination's minimum transfer
  (0.991) beats the best single colour (0.962 at 632.8) by 3% instead of
  3x; on the rows the combination is neutral for I+/S (single-on-base
  SNR 269 -> 183, grid 31.8 -> 32.4) and WORSE for L (the 480 nm channel
  reads NEGATIVE on the 30 nm base, |c| = 1.73, 2.1 rad dimple, and the
  equal-weight combiner inherits it).  What stays chromatic is RANGE:
  780 nm is the best single colour on nearly every row (smaller phase
  per nm of height, 1.62 lam/D dimple), best pair 700+780.
  *S5 noise pricing of I+ (item 4, `rec193full`):* photons per DM
  state for 1 pm -- L 5.4e13, F 3.7e13, I 6.4e13, I+ 8.8e13 (the prior's
  own shot noise costs 5%: 8.4e13 with a noiseless prior), S 1.0e14;
  all within a factor 3, all ~25x cheaper than the legacy defocused-model
  pricing (1.4e15 / 5e15); high-N floors converge to the battery's
  systematic floors (L 61 / F 37 / I 48 / I+ 31 / S 42 pm).  Photon
  noise is not the blocker; the 6.5% hold-out gain and the 25-60 pm
  systematic floors are.  *Open:* the hold-out-site kernel check
  (measure the kernel AT the hold-out site); deck fold (deck_zwfs still
  tells the legacy-model S1-S6 story).

- **S9 (zwfs_run, 2026-09-10, Dave: "go ahead with ZWFS next steps"):
  the two calibration questions, measured -- and a real fix to the
  actuator fit.**  Runner knobs added for them: `reg.kernel_site`
  ('center' | 'hold' | [r c]: where the response kernel is measured;
  the registration anchor always comes from the centre poke),
  `battery.calib_surface` ('flat' | 'base': kernel + modal transfer
  measured differentially on the working surface, the exact class read
  with the base's refined sign map), `reg.stencil_site` ('grid' |
  'lattice'), `dm_use`, `hold`, and a per-row fold-crossing diagnostic
  (pixels whose side of the quarter-wave fold differs between the base
  and base+change).  All at NGRID 193 / model 1024, 96x96, unless stated;
  runs/ks_hold, ks_hold_tc, ks_hold_hw12, ks_hold_lat, cal_base,
  fold_diag, rec193_lat, ng385_lat.
  *(1) Where the kernel is measured.*  Centre kernel tested at (60,40):
  0.900 (the record).  Kernel at (60,40) tested at the centre: 0.987.
  Kernel at (60,40) tested at (60,40): 0.958 -- NOT 1: even at its own
  site the fit recovered 96%.  Widening the stencil (half-width 12 vs 6)
  changed nothing (0.9576 both).  THE CAUSE: `dmg_anchor` returns the
  poke's peak on the 0.28 mm MAP grid (`tax = xg(tc)`), so the stencil
  was sampled up to half a grid pitch (0.14 mm) off the actuator centre
  the fit samples at.  Snapping the stencil site to the exact lattice
  point (`reg.stencil_site` 'lattice', now the DEFAULT; 'grid'
  reproduces S7) gives own-site 0.9915 / floor 5 pm / SNR 4200 (the
  near-identity check it should be), record configuration 0.900 ->
  0.946 (floor 49 -> 37 pm), and at NGRID 385 / model 1024 the
  test-actuator gain 0.935 -> **0.9963** -- the 3%-of-1 spec is MET at
  the 1 Mpix-class sampling; I+ on the 30 nm surface there 0.913 / 25 pm
  / SNR 359 -> **0.975 / 18 pm / 529**, grid-on-base SNR 37 -> 45.  So
  the "remaining 6.5%" was ~5% stencil-site quantization + ~1-5% true
  site variation (kernel at (60,40) vs centre: raw peak 0.911 vs 0.928,
  a broader shape), the latter absorbed by a kernel measured where it
  is used.  The interferometer's tg96_s3/s4 sample the TRUE kernel the
  same way -- flagged to CCMac for the tg96 runner (BRIEF_ccmac_tg96_oap
  addendum).
  *(2) Calibrating on the working surface* does NOT recover the
  working-surface gain deficit: I+ on the 30 nm surface 0.736 with the
  on-surface calibration vs 0.788 flat-calibrated (S 0.767 vs 0.705; the
  linear class cannot be calibrated on a 30 nm surface at all -- kernel
  corr -0.28).  Multi-site ladder on-surface: I+ 0.73 / 0.75 / 0.39 /
  0.59 vs flat-calibrated 0.79 / 0.82 / 0.76 / 0.31 at 30/40/50/60 nm.
  So the deficit is in the READINGS on a working surface, not in the
  calibration.  Fold crossings are NOT it for a single change: 3 of
  2258 beyond-fold pixels move under a single 10 nm change (0.01% of
  msk), 10 under the 47-site 1 nm grid, but 946 (3.3%) under a dense
  10 nm random change -- the mechanism that makes the stepped reading
  own dense commands.  With the lattice stencil the single-site
  working-surface deficit at NGRID 385 is 2.5% (I+ 0.975 vs 0.996 flat);
  the 47-site ladder keeps its shape (I+ 0.80 / 0.83 / 0.74 / 0.34).
  Open, named: the single-site 2.5% and the 47-site 20% on a working
  surface (the stepped reading's flat-DM |Eb|^2 calibration, 14% off on
  a 30 nm surface, is the S-class suspect; for I+ the differential of
  two exact readings whose map error is 2.8e-4 rad rms should be
  smaller than measured -- the actuator-fit stage on a working surface
  is the next place to look).

- **S10 (zwfs_run, 2026-09-10, Dave: "modes can also be measured -- and
  should be when using a real DM: poke the actuators in grids wide enough
  that there is no overlap, then measure, to obtain dw/da"): the
  MEASURED response matrix is now the default calibration.**
  `battery.calib_mode` 'matrix' (`calib_matrix_`): every 8th actuator
  poked in a sparse grid, the 64 grid offsets stepped so every lit
  actuator is poked once (64 states for 3252 actuators), each response
  cut from its own +/-half-step detector window (registration only
  PLACES the windows), columns per reading class, estimator = a
  regularized least-squares solve on J (dense 3252x3252 normal matrix,
  Cholesky).  No single-site kernel, no shift-invariance, no frequency
  correction.  `matrix_step` 8, `matrix_lam` 1e-3 of the median column
  energy, `matrix_sign` 'same' | 'alternate'.
  *The one property of THIS sensor the method must carry:* the ZWFS
  cannot see piston, so every reading is mean-referenced over the pupil
  and a multiplexed frame carries the pokes' shared negative pedestal.
  Cut naively, each column gets its halo concentrated in an 8-actuator
  cell instead of spread over the pupil, and the estimator over-responds
  2-4x to patterns below ~12 cycles/aperture (runs/mat193: 2.1 / 2.2 /
  3.7 / 3.2 / 1.6 at 0.5-8 cyc/ap for L).  Fix (runs/mat193b): subtract
  the frame's pedestal (its median over the mask) before cutting, and
  give each column its own volume spread over the mask -- J'J = Jl'Jl -
  v v'/A_mask, J'm = Jl'm - v (1'm)/A_mask (a rank-one term; the uniform
  command is nulled as the sensor nulls it).  With it the exact
  reading's response through the matrix is **0.99-1.08 at EVERY probed
  frequency (0.5-40 cyc/ap) with no correction**; S 0.66 at 0.5 cyc/ap
  (its flat-DM |Eb|^2) and 0.93-1.02 above; L 0.74 at 0.5 then a slow
  rise to 1.28 at 40 (its own nonlinearity).
  *Rows (193 rays, 96x96, matrix vs kernel-lattice):* flat single
  actuator **0.994 / floor 4 pm / SNR 5500** (kernel: 0.946 / 37 pm /
  515); 30 nm surface, single 10 nm change: I+ 0.73 / 42 pm (kernel:
  0.82 / 13 pm), S 0.75 / 17 pm (0.74 / 17); 47-site 1 nm grid on the
  surface: I+ SNR 60 (kernel 36), S 82 (37); dense random I+ 0.90 / 4.6
  nm (0.82 / 6.1), S 0.82 / 2.1 nm (0.74 / 3.5); 47-site 10 nm ladder at
  30 / 40 nm rms: I+ 0.68 / 34 pm / SNR 200 and 0.67 / 47 / 143 (kernel
  0.80 / 290 / 27.5 and 0.83 / 250 / 33) -- the FLOOR falls 6-10x, the
  GAIN on a working surface falls.  So the matrix removes the crosstalk
  entirely and isolates the working-surface gain loss as a READING
  effect.  Its mechanisms, now with evidence: for I+ the few pixels that
  cross the quarter-wave fold under a change (3 per single change) sit
  INSIDE the changed actuator's footprint, where they are a large
  fraction of its pixels at 2.5 px/actuator (site-dependent: the (60,40)
  site at 385 rays loses only 2.5%); for S the flat-DM |Eb|^2
  calibration is 14% off on a 30 nm surface.  Both are fixes in the
  readings (a fold-aware branch choice near the change; b2cal on the
  working surface), the next work.
  *Alternating +/- pokes (Dave's question, runs/mat193c):* the exact
  readings are neutral to slightly worse in the model (flat single
  actuator 0.987 / 11 pm vs 0.994 / 4 pm; surface rows identical); the
  linear reading's +/- asymmetry costs it (0.67); the piston-null term
  handles the pedestal in either case.  On a real bench alternating
  cancels common-mode drift between sets (the Steeves 2020 protocol), so
  it stays as `matrix_sign` 'alternate'; 'same' is the model default.
  `runs/mat385`: the 385-ray matrix run (the deck's numbers).
  *The matrix ON THE WORKING SURFACE (runs/matbase: `calib_mode` 'matrix'
  + `calib_surface` 'base'; the operating-point interaction matrix, 64
  states x 4 frames once per working state) -- THE result of the day:*
  the map-space diagnostic (runs/mapdiag; the same change read on the
  flat vs its differential on the surface, over the changed actuator's
  window) showed EVERY reading's differential map 29-60% off in rms and
  0.49-0.82 in amplitude on the surface (L 0.60 / 0.49, F 0.57 / 0.65, I
  0.35 / 0.82, I+ 0.36 / 0.78, S 0.29 / 0.73): the sensor's local
  sensitivity depends on the local phase of the working surface (the
  per-pixel sensitivity factor of the Ruane budget form).  A matrix
  measured on that surface carries each actuator's local sensitivity,
  which a single-site kernel could not (why S9's on-surface kernel
  calibration failed).  Rows at 193 rays, 30 nm surface, on-surface
  matrix (flat matrix in brackets): **S single 10 nm change 0.989 / 5 pm
  / SNR 2160** [0.75 / 17 / 444]; S 47-site 1 nm grid 0.999 / 4 pm / SNR
  284 [0.83 / 10 / 82]; S dense random 10 nm 0.98 / 0.68 nm [0.82 /
  2.1]; **L (linear, one frame) single 1.04 / 21 pm / SNR 492** [0.32 /
  140 / 23], grid 1.05 / 22 pm / 47, dense 1.04 / 4.7 nm; the exact
  one-frame readings do NOT benefit (I+ 0.69, I 0.81, F 0.76): their
  surface error is the fold sensitivity, which moves with the change
  itself (the multiplexed pokes cross the fold on different pixels than
  the test change).  Ladder on 47 sites with the 30 nm-surface matrix:
  S 1.08 / 0.98 / 0.92 / 0.70 at 30 / 40 / 50 / 60 nm rms (floors 243 /
  228 / 399 / 664 pm), L 1.03 / 0.96 / 0.83 / 0.68 -- a calibration
  made at 30 nm degrades slowly (0.92 at 50).  DOCTRINE, updated: on a
  working surface, measure the response matrix ON that surface (256
  frames once), then the four-frame stepped reading reads changes at
  the 5 pm floor with gain 0.99; the linear one-frame reading is then
  usable at 21 pm; the exact one-frame readings stay the flat-surface
  tools.  The 1 pm target: for 1 nm changes on the 47-site grid the S
  reading is at 1 pm of gain error + 4 pm of floor.  runs/matbase385:
  the same at 385 rays.
