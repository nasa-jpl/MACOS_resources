# Polarization phase-shifting Twyman-Green — a DM surface gauge

A surface-figure gauge built end to end in MACOS: a compensated
Twyman-Green interferometer with a **deformable mirror as the test optic**,
whose fringe phase is stepped by **rotating an analyzer** rather than by
translating a reference mirror.  Four analyzer angles are a standard
four-step PSI; the recovered map is compared against the DM map that was
injected.

| file | what it is |
|---|---|
| `example_tg_psi_dm.m` | the full, gated measurement — five gates, the closure, all the numbers |
| `demo_tg_psi.m` | the same measurement as seven narrated beats, one PNG each |
| `dm_influence_map.m` | actuator influence-function DM surfaces (`checker` / `random` / `single`) |
| `tg_*.in`, `demo_*.in` | the emitted prescriptions (both arms) |
| `dm_*.txt`, `demo_flat.txt` | the DM grid maps — **gitignored**, ~1 MB each; run either script once to recreate the files the decks reference |
| `tg_psi_dm.mat` | every result struct, the recovered map, the sweep cube |
| `tg_psi_dm_*.png`, `demo_beat*.png` | figures, and the pre-rendered demo backups |

Gates live in `mmacos/tests/tTgPol.m` (9 tests, model 128, `SUITE_FAST`).

```
cd <this dir>
matlab -batch "run('example_tg_psi_dm.m')"     # gated, ~2 min
matlab -batch "run('demo_tg_psi.m')"           # the beats + PNGs
demo_tg_psi                                    # interactive: pauses per beat,
                                               # and animates the analyzer sweep
```

### Demo beat timings (measured)

| beat | what it costs |
|---|---|
| 1 build + emit both arms | < 1 s |
| 2 `view_std` layout | ~2 s |
| 3 align (measure both arms, solve the waveplate) | 7 traces, ~1.5 s |
| 4 null (the analyzer basis for both arms) | **6 traces, 1.58 s** |
| 5 live single-actuator poke (`set_elt_grid`) | 3 traces, ~0.7 s |
| 6 analyzer sweep, 36 frames | **0.036 s — zero traces** |
| 7 full DM + four-step PSI + closure | 3 traces + the pupil-map fit, ~5 s |

Nothing in the demo takes long enough to lose a room.  Every beat also
writes its PNG, so a live hang costs ten seconds — show the picture and
move on.

---

## Why Twyman-Green, and not Mach-Zehnder

Dave's ruling (2026-08-28), for **figure** measurement of a DM:

| | Twyman-Green | Mach-Zehnder |
|---|---|---|
| test optic | **normal incidence**, double-passed (2× sensitivity) | oblique, single pass |
| null | **natural** against a flat reference | needs a matched arm |
| reference surfaces | **fewest of any two-beam layout** | a second beamsplitter |
| what it buys | — | arm isolation, two complementary output ports, testing in **transmission** |

None of what MZ buys is needed to measure a surface, and each costs a
component and an alignment.  **MZ earns its place for dynamics, not for
figure.**  The double pass is not a detail: it makes a surface height `h`
appear as `2h` of OPD, which gate 5 below pins end to end.

## Why polarization phase-shifting

The phase steps come from rotating a polarizer in the output leg.  Nothing
in the interferometer moves, so the measurement does not integrate the
drift and vibration that a PZT-stepped rig accumulates between frames —
and with a polarization-multiplexed detector all four frames are
**simultaneous**, i.e. a snapshot gauge.

The price is a polarization-component error budget, measured term by term
in the sibling template
[`../bench_ifo_pol/example_bench_ifo_pol_slice3.m`](../bench_ifo_pol):
arm-QWP retardance is the tightest at ~344 nm/wave, then chromaticity
(~8 nm per 10% Δλ), then axis errors (~5 nm/deg).  That study's
conclusion still stands — **mechanical PZT stepping is less sensitive to
polarization imperfection and is preferred wherever a moving mirror is
acceptable**; this configuration earns its place when one is not.

## The train

Each arm is its own deck: the engine does not split rays, so a polarizing
beamsplitter is two traces.

```
input polarizer @45 → [BS] → double-passed QWP (net half-wave, rotating
that arm's linear state) → TEST OPTIC (the DM) or PZT flat → [recomb] →
output QWP (orthogonal linear → orthogonal circular) → ROTATING ANALYZER
@θ → L2 → focal mask → field lens → detector at the DM pupil conjugate
```

The polarizing elements are the **real engine elements** — `TrPolarizer`
(EltID 15) and `WavePlate` (18), gated by `tPolElement` — inserted by
`macos.design.twyman_green('polarizing', true)`, not a Jones model bolted
on afterwards.  Every one sits in a collimated, normal-incidence leg,
where the off-normal material-axis question is identically absent.

The detector leg is the `l2_trade` winner (a small field lens behind the
focal mask), which took the instrument-vs-truth residual from 6.76 nm to
0.97 nm at 50 nm pokes — see
[`../../40_benches/bench_ifo_dm/l2_trade`](../../40_benches/bench_ifo_dm/l2_trade).

---

## Results

Model 256, ray grid 63, HeNe, 16×16 actuators at 3.5 mm, checkerboard
commands at 50 nm of surface.

| quantity | value |
|---|---|
| beamsplitter rotation of the test arm | **+7.479°** (nulled by turning its waveplate +3.768°) |
| PSI scale error if that is not corrected | **+11.7%** |
| fringe visibility, aligned / unaligned | 0.998304 / 0.996612 |
| analyzer basis: 3 traces/arm reproduce any angle | **3.5e-10** relative (collimated) |
| … and degrades as | **0.149·β²**, β = ray angle at the analyzer |
| 4θ systematic in the four-step estimator | 8.9e-4 of the fringe; **1.7e-14 nm** after the differential |
| 6θ content, from 12 **traced** angles | **3.0e-14** of the 2θ fringe (5.1e-17 synthesized — structural, not a gate) |
| known 20 nm piston recovered | gain **1.00000**, error 2e-5 nm |
| grid piston vs rigid optic shift | agree to **2.3e-10 rad** |
| DM map recovered | corr **0.998434**, residual **0.304 nm rms** interior (0.363 whole pupil) on 6.35 nm rms |
| runtime | 9 traces (~1.8 s) for the whole analyzer basis |
| flat-DM null (the gauge with nothing to measure) | 1.46e-3 rad rms = **0.073 nm** of surface |

### 1. The analyzer sweep is free, and exactly so

An ideal analyzer at angle *t* projects onto `a(t) = cos t·u1 + sin t·u2`,
and everything downstream of it is a fixed linear map `M` per ray.  So the
detector field is **bilinear** in `a(t)`:

```
E_det(t) = (a(t)·E_in) · M a(t)
         = c²·A + c·s·B + s²·C ,   c = cos t, s = sin t
         A = E(0°),  C = E(90°),  B = 2·E(45°) − A − C
```

Three traces per arm therefore reproduce the detector field at **any**
analyzer angle.  The four PSI frames, a 64-angle least-squares fit and a
36-frame animation all come out of the same six traces — the live sweep
costs nothing.

Two consequences worth keeping:

- `I(θ)` can contain only DC, 2θ and 4θ, and **nothing above** — the
  analyzer has to behave as an ideal rank-1 projector for that to hold.
  Measured from **12 directly traced** angles: 6θ/2θ = 3.0e-14.  It has to
  be traced: a frame synthesized from the quadratic basis is a degree-2
  trig polynomial in 2θ *by construction*, so its 6θ bin is zero by
  algebra (5.1e-17) and would pass against any engine at all.  The traced
  and synthesized stacks agree on the 4θ term to all printed digits
  (8.8951e-04), which is what ties the cheap path to the honest one.
- The 4θ term (8.9e-4 of the fringe) is a real systematic of the four-step
  estimator **at the detector**, where the tail's transport makes `|Mâ(θ)|`
  vary; it is identically zero *at* the analyzer, which is why the
  ray-level slice-3 gate reports 1e-16.  It is **common to the poked and
  baseline runs, so the differential protocol cancels it** — it survives
  only in a single-shot map.

The identity is exact only to **O(β²)**, β = the largest ray angle off the
analyzer normal, because the engine projects the analyzer's *material*
axis into each ray's transverse plane and renormalizes.  Measured on a Kr
ladder (model 256, aligned): `error / β² = 0.1458, 0.1455` across a factor
of 2 in β.  The COEFFICIENT is mildly configuration-dependent — the same
ladder run at the design azimuths gives 0.1487/0.1482 — so what is gated
is the EXPONENT: `(error ratio)/(β ratio)² = 1.0020`, i.e. exactly second
order.  In a collimated analyzer leg — the design condition — β is just
the DM's own slope error, 2.3e-5 rad, and the residual is 3.5e-10.

### 2. The beamsplitter misaligns the rig, and the gauge reads 11.7% high

The design intent is that the double-passed QWPs leave the arms in
**orthogonal linear** states: a half-wave at azimuth α sends a 45° input
to `2α − 45`, so α = 0 and α = 45 should give −45° and +45°.  Measured,
the reference arm leaves at exactly +45.0000° and the test arm at
**−37.5214°** — 7.479° from orthogonal.

The cause is ordinary Fresnel: every element between the polarizer and the
recombination plane that is not at normal incidence is a **diattenuator**
(`t_s ≠ t_p`), and a diattenuator rotates a linear state toward its
high-transmission axis.  The reference arm is immune by accident — its
half-wave sits at 45°, which maps `θ → 90 − θ` and so cancels the
diattenuation before it against the diattenuation after it.  The test
arm's half-wave at 0° maps `θ → −θ`, which **adds** them.

Nothing downstream can repair this: a waveplate is unitary, and a unitary
cannot turn a non-orthogonal pair into an orthogonal one.  The four-step
estimator assumes an orthogonal-circular pair, so it acquires a local
phase **gain** — measured on a known 20 nm piston, **1.11661**.  The fix
is the arm waveplate itself, which is exactly the knob a technician turns
while watching the two arms extinguish each other; the example solves for
it in five traces (+3.768°) and the gain returns to 1.00000.

Every underlying field phase in the model is exact throughout — scalar
field phase, geometric OPD, polarized field phase at the detector, and
ray-level field phase at the analyzer all recover the 20 nm piston to
1.00000 with pupil scatter ≤ 2.5e-9 rad.  **The error is in the
measurement, not the physics**, which is precisely why a model is the
right place to find it.

> **Fringe visibility is not an alignment metric.**  The misalignment
> costs **11.7% of scale** and **0.17% of contrast** — a factor of ~69.
> Contrast responds at second order in the alignment error, phase gain at
> first, so a rig can look beautifully aligned on a fringe monitor and
> still read a DM 12% too tall.  Gate 5 runs both configurations and
> prints both numbers, so the alignment step cannot be quietly deleted.

### 3. A GridData value is the SURFACE height

Long-standing open question, closed here.  A uniform grid piston of
`dz = 20 nm` recovers as `+0.397167 rad`, which is exactly
`4π·dz/λ` — i.e. the double pass supplies the factor 2 and the grid value
is the **surface** displacement, not the wavefront one.  Translating the
whole test optic by the same `dz` gives the identical answer to
**2.3e-10 rad**, which is what makes the identification airtight rather
than a coincidence of magnitude.

### 4. Closure

The detector images the DM through L2, the field lens and the folded
tilted-plate train, at magnification 10.53 (det→DM) with a 180° inversion,
0.000% anamorphic stretch and 0.0068 mm rms of nonlinear distortion —
all **measured from the trace itself** (one DM-position/detector-position
pair per surviving ray), not assumed.  Resampling the injected map through
that mapping and comparing pixel by pixel:

```
truth 6.35 nm rms | recovered 6.27 nm rms | residual 0.304 nm rms interior
correlation 0.998434
```

The residual is spread across the pupil rather than concentrated at the
rim (0.363 whole pupil vs 0.304 interior), which identifies it as the
detector-leg retrace term the `l2_trade` study characterised — not an edge
artefact.

---

## Notes for anyone extending this

- **A gate must not be evaluated on data that satisfies it by
  construction.**  The 6θ check above is the live example: it is a real
  statement about the analyzer only when the frames come from traces.
- **The array parity of a diffraction/complex-field map is deck-dependent**
  (`../../../doc/opd_conventions.md` §2): probe it, never assume it.  The
  example resolves it over the eight candidates and prints the winner; the
  gate calibrates on one actuator and verifies on a second.
- **Tranche-1**: a polarizing element placed after the first
  physical-optics leg transforms rays but never reaches the diffraction
  grid, and the failure is silent.  This train is geometric end to end, but
  gate 1 asserts the order from the *emitted deck* anyway, and checks that
  the grid really responds to the analyzer.  That second check must be run
  on a **linear** arm state — after alignment each arm leaves circular, and
  a circular state carries the same power through an analyzer at any angle,
  so the tripwire would pass vacuously on the aligned rig.
- **A grid setter needs no separate `macos.modify()`** — `set_elt_grid`
  invalidates the cached trace itself.  The demo's live poke relies on it.
- `dm_influence_map` defaults to a **checkerboard** because a random
  command set spreads its power over frequencies the pupil imaging cannot
  carry, so the residual then reports the optics rather than the gauge.
  Use `'random'` to score a realistic command set once the checker closes.
- The rig is built by `macos.design.twyman_green`; `'polarizing'` defaults
  to false and then emits **bit-identically** to the plain Twyman-Green,
  which `tBench/test_twyman_green_polarizing` pins.

## Deliberately not done

- **No engine work.**  Everything here is builder/example level.
- **The polarizing beamsplitter is bookkeeping across two decks.**  The
  physical rig routes all returned light to the output port because each
  arm double-passes a QWP and the PBS then sees the rotated state; in the
  model that is two independent traces, so the routing is expressed by the
  arm azimuths rather than by a splitting element.
- **The waveplates are ideal retarders**: no o/e walk-off, no face Fresnel
  loss, and retardance independent of incidence angle.  Bounding the
  field-of-view effect of a real crystal plate needs a birefringent-plate
  model.
- **The four-step estimator is kept as the headline** even though the free
  sweep makes a 64-angle least-squares fit available, because four steps is
  what a polarization-multiplexed snapshot detector actually delivers.  The
  example reports the difference between the two.
