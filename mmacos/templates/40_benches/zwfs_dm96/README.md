# zwfs_dm96 — a Zernike wavefront sensor against the PSI gauge, same DM truth

The ZWFS campaign (plan: `macos/BRIEF_zwfs_campaign.md`).  Question:
does a Zernike sensor beat the TG96 polarization-PSI gauge on the
differential benchmark (10 nm actuator deviation read to 0.021 nm,
base-independent — tg_psi_dm96 run 10)?

## Rulings (Dave 2026-09-04)

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
- **S1 round 5 (G3): defocus reads at gain 0.54 — real ZWFS
  low-order attenuation, not a bug.**  The reference wave is built
  from the aberrated field's own core, so aberration content inside
  the dimple's ~0.5 cyc/pupil passband leaks into the reference and
  self-subtracts.  Sign pinned: S_CONV = −1.  The G3 gate pattern
  moved to a radial cosine above the cutoff; the defocus response is
  printed as an instrument property.  Expect the S3 transfer curve to
  roll off at LOW frequency — the mirror image of PSI's high-frequency
  rolloff; the actuator-space scoring (Dave's ruling) must carry it.
- **S1 rounds 8–9: the truth FRAME was 25% off — ray affine fixes it.**
  The support-area pupil-radius estimate implied k = 0.754 of truth (a
  q=5 cosine decorrelates at half a rim cycle, so the fit read gain
  0.02 — "no response" that was pure frame).  The ray-measured
  DM→detector affine (registration doctrine DOF class 1, lifted from
  tg96) gives mag 10.158, anam 0.00%, nonlin 0.086 mm; a diagnostic
  scale sweep about the ray frame peaks at k* = 1.005, |corr| 0.997.
  Frame-before-angle, once more.
- **S1 GREEN (round 10, 20 s/run):** dimple 6.29 px (G0); masked field
  == E0 + c·Eb at 3.3e-16 (G1); core fraction 0.2801 ≈ Airy encircled
  energy for 1.06 λ/D (G2); q=2 radial cosine at 8 nm recovers gain
  +0.932, resid 0.45 nm (G3, ray frame); no-dimple frame reads −0.07
  (G4 — the signal is the dimple's).  S_CONV = −1 pinned.  **Defocus
  property in the correct frame: gain 0.324** — the ZWFS low-order
  self-referencing attenuation (round-5's 0.54 was frame-biased).
  q=5 reads 0.675 at dev res = dev sampling rolloff (65-px pupil),
  not instrument — transfer curves belong to S3 at full res.
- (further stages appended; reports are the record)
