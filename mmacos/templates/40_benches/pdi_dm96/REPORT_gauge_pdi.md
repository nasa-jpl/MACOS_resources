# The point-diffraction gauges for the DM Surface Gauge Comparison deck

TO's part of `macos/BRIEF_gauge_deck.md`, answering
`macos/BRIEF_to_gauge_deck.md` (Dave, 2026-09-13).  Numbers first; every
one has a run tag in this directory's `runs/` (or, for the record taken
before the 2026-09-13 split, in `../zwfs_dm96/runs/`).  Departures from
the brief are flagged **[departure]**.

Sheet and runner: `pdi_params` / `pdi_run` (the code is shared —
`zwfs_run` + `../dm_gauge_lib`, nothing copied).  Headless:
`./pdi_batch.sh TAG "pdi_params, <args>"`.  The full chain that produced
this report is `runs/gmaster.sh` (`gsmoke` → `gseq1` … `gseq4`).

---

## 0. The best configuration, and why

*(filled at the end, from sections 1-5)*

---

## 1. PF through the two decks — the reference arm TRACED

**What changed.**  Through 2026-09-12 the P/SRI reading `PF` carried a
**synthesized** reference: the recollimated LP01 mode of the paper's
waveguide, at unit rms over the pupil, scaled by the state's overlap
coupling κ — *a complex scalar, so the reference's shape was fixed by
construction*.  The reading now has a second bench, `pdi.bench 'psri'`
(`macos.design.psri_bench`), on which the reference arm is **traced**:
`psri_ref.in` carries the beam through Lr1, the physical pinhole in its
near-field sphere bracket, Lr2, the fold and BS3 to its own camera,
while `psri_test.in` carries the test beam to the same camera plane.
Two traces and two deck loads per state; the solver still uses the flat
state's R₀ and κ = 1 (or |κ| from a shutter frame), so whatever the real
arm does to the reference — amplitude, **shape**, piston — is an error
the reading has to live with, not a model input.

The bench is the one `psri_layout_fig` solves and emits, unchanged:
chief optical paths equal to 0 mm (4453.1061 mm each), compensator
21.857 mm, exit chiefs 0 mm apart, camera planes 2.3e-13 mm apart,
3210 of 3210 rays through each arm.

**The control that separates the two things a real arm does.**
`pdi.ref_frozen` traces the reference arm ONCE, on the flat, and reuses
it: that run carries the real arm's **shape** but not its **motion**.
Three points, not two: synthesized ↔ frozen-traced ↔ traced.

### 1a. The bench, and what the traced arm does to the reference (`pfdeck`)

Model 1024, 193 rays, 96×96 DM, 2.0 λ/D pinhole at the reference lens's
F/2.92 focus = **3.69 µm**, 7.84 px across at that focus (the dimple's
≥ 6 px rule: PASS).  Coupling through the real arm **η = 0.677**;
throughput (detected / incident, flat) **0.806**; visibility on the flat
**0.873**; the flat reads 3.3e-16 rad rms.  Its synthesized counterpart
(`../zwfs_dm96/runs/pdi193f`) has η_c 0.587, throughput 0.752,
visibility 0.863 — the traced arm is slightly *brighter*, because the
physical pinhole passes the real spot's core rather than a waveguide
mode's overlap with it.

**The reference's motion under the 30 nm working surface** — the whole
question the P/SRI form exists to answer:

| reference | total change | best complex scale | **SHAPE change** |
|---|---|---|---|
| traced arm, `pfdeck` G6 | 0.1439 | 0.858, +0.0239 rad | **0.0103** |
| the FFT pinhole surrogate at 2 λ/D, `pdi193f` G6 | 0.1539 | 0.846 | 0.0059 |
| the synthesized LP01 mode, `pdi193f` G6 | κ = 0.849, +0.005 rad | — | **0 by construction** |

The scale and the piston are what a shutter frame and a differential
absorb; the SHAPE change is what nothing absorbs.  The synthesized
reference reports zero there *because the model puts the state's whole
effect into one complex scalar*.  The real arm's is **1.03%**, about
1.7× the pinhole surrogate's on the test arm — the two arms are not
identical (F/2.92 against F/4.17, and the reference arm has two lenses
of its own).

**What that costs the reading, absolutely** (G5: a 100 nm poke on every
8th actuator, 12 983 pm rms on the mask, read against the engine's own
phase):

| reference | G5 absolute error |
|---|---|
| traced, moving with the state (`pfdeck`) | **5.889 pm = 0.045% of the figure** |
| traced ONCE on the flat and held (`pfdeck_frz`) | **0.000 pm** |
| synthesized LP01 mode (`pdi193f`) | 0.000 pm |

**The control settles it exactly.**  Freezing the traced arm — same
bench, same lenses, same pinhole, same aberrations, only held still —
takes the error to zero.  So the 5.889 pm is ENTIRELY the reference
MOVING with the state; none of it is the real arm's shape, its F/2.92
optics, or the bench.  And the synthesized model sits with the frozen
one, because a reference whose state dependence is one complex scalar
is, to a solver that takes κ = 1, a frozen reference.  That is the
idealization the two decks were built to price: **on a 13 nm figure it
is worth 5.9 pm**, ~40× below the 0.1% line the campaign's gates use.  The rows, the photons and the loop
below say what it costs differentially, which is the number a servo
cares about.

---

### 1b. The rows and the range (`pfdeck` against `pdi193fbase`)

Both with the response matrix measured ON the 30 nm working surface, 47
grid sites, 96×96, raw estimates.

| row | PF, reference **traced** (`pfdeck`) | PF, reference **synthesized** (`pdi193fbase`) | P (same run) | S (same run) |
|---|---|---|---|---|
| 10 nm on one actuator: gain / err / SNR | **0.9935 / 2 pm / 4895** | 0.9935 / 4 pm / 2842 | 0.9935 / 4 / 2790 | 0.9885 / 5 / 2120 |
| 1 nm on 47 grid sites | **0.9924 / 1 pm / 878** | 0.9992 / 3 pm / 346 | 0.9992 / 3 / 345 | 0.9993 / 4 / 284 |
| dense random 10 nm | **0.9926 / 144 pm** | 1.0002 / 330 pm | 0.9985 / 338 | 0.9838 / 681 |

**Capture range to 10%, matrix aged from 30 nm** (the ladder: a 10 nm
change on the 47 sites, read on a growing base):

| base rms | PF traced, gain / floor | PF synthesized | P | S |
|---|---|---|---|---|
| 30 nm | 0.9930 / 12 pm | 0.9968 / 20 | 0.9971 / 20 | 0.9993 / 24 |
| 60 nm | 0.9936 / 32 | 1.0055 / 109 | 0.9380 / 146 | 0.6569 / 669 |
| 120 nm | 0.9950 / 94 | 1.0230 / 319 | **−0.013 (folded)** | **−0.013 (folded)** |
| 240 nm | 0.9971 / 221 | 1.0579 / 742 | −0.019 | −0.022 |
| 480 nm | 0.9984 / 480 | 1.1277 / 1587 | −0.007 | −0.007 |
| **capture range to 10%** | **480 nm+ (holds at the last rung)** | 480 nm+ (but 12.8% off there) | ~55 nm | ~44 nm |

**The traced arm is not worse than the model of it — it is flatter.**
The synthesized reference drifts to +12.8% gain by 480 nm because the
model routes the state's whole effect through one complex scalar κ and
the solver then takes κ = 1; the real arm's reference simply does not
depend on the surface that much, and the gain stays inside 0.7% over a
16× range of working surface.  **This is the P/SRI's whole argument,
and it survives being built.**

*Read across benches with care.*  `pfdeck` runs on the P/SRI's own two
decks and `pdi193fbase` on the ZWFS test arm; the magnification (10.690
vs 10.158 DM-mm per detector-mm) and the seat's propagation differ, so
the floor and SNR columns carry a bench term as well as a reference
term.  The run that separates them is `pfdeck_frz` — the same bench,
the reference traced ONCE on the flat and held — below.

### 1d. What the reference's MOTION costs differentially (`pfdeck_frz`)

Same bench, same everything, the traced arm held still.  This is the
only comparison in which the reference is the ONLY variable.

| row (matrix on the 30 nm surface) | reference traced | reference frozen | the motion's cost |
|---|---|---|---|
| 10 nm on one actuator: gain / err / SNR | 0.9935 / 2 pm / 4895 | 0.9932 / 2 pm / 5573 | gain identical to 3 digits; **SNR −12%** |
| 1 nm on 47 grid sites | 0.9924 / 1 pm / 878 | 0.9925 / 1 pm / 884 | **−0.7%** |
| dense random 10 nm | 0.9926 / 144 pm | 0.9925 / 129 pm | floor **+12%** |
| capture range to 10% | 480 nm+ | 480 nm+ | none |

**Differentially the reference's motion is a 10%-class effect on the
noise floor and nothing at all on the gain or the range.**  Absolutely
it is the 5.9 pm of section 1a.  A servo cares about the differential;
an absolute figure measurement cares about both.

*The residual difference to `pdi193fbase` is NOT attributed.*  The
frozen run — the synthesized model's equivalent, by 1a — still reads
2 pm where the ZWFS-arm run reads 4 pm, and 129 pm dense where it reads
330.  Two candidates, and this report does not separate them: (i) the
BENCH — magnification 10.690 vs 10.158 DM-mm per detector-mm, and the
P/SRI test arm passes a plain plane where the ZWFS arm's mask sits
inside a near-field sphere bracket; (ii) the reference's COUPLING — the
physical pinhole takes 0.677 of the flat's focal light where the LP01
mode's overlap takes 0.587, which sets the reference amplitude (0.00325
vs 0.00271, both pickoff-budget-limited) and the visibility (0.873 vs
0.863).  **No hardware claim is made from that gap**; what section 1
claims is only what the frozen/moving pair isolates, which is the one
comparison with a single variable.

### 1c. Photons (`pfdeck` noise stage)

**N(1 pm) = 2.00e14 photons per measurement at the detector**, against
**1.9e14** for the synthesized reference on the ZWFS arm
(`../zwfs_dm96/runs/pdi193f`) — the same number to the digit the
comparison supports.  **Modeling the reference arm or tracing it does
not change what a picometre costs; the 60/40 pickoff does.**  Divide by
the run's throughput (0.806 here, 0.752 there) for incident photons.
For scale, on the same bench family: P 3.3e13, V 4.7e13, S 5.4e13 —
the P/SRI form is ~6× the light of the common-path pinhole, because
60% of the beam goes to an arm that returns a fraction of it as
reference while the test beam keeps only 40%.

---

---

## 2. Capture range and photons, P and PF

Two questions, with the same answer shape: how far from null can the
device still be used, and what does a picometre cost in light.

*Definition (Dave 2026-09-12, "they will not be operating at null"):* the
capture range to 10% is the largest working-surface rms at which a 10 nm
change on the 47 grid sites reads within 10% of its size — gain in
0.9…1.1 — the crossing log-interpolated between ladder rungs.  The
runner prints it after every ladder.  Two calibrations are scored
separately: one measured once at 30 nm and left to AGE as the surface
grows, and one RE-MEASURED on each surface.

*(a) Aging from 30 nm — `cap385p`, 385 rays.*

*(b) Re-measured on the surface — `cap385p_b60/90/120/160`, rows
`{'base/grid'}` with the matrix on that surface.*

*(c) Photons for 1 pm at 30 / 60 / 120 / 160 nm — `noise193p_b*`,
193 rays, `noise.nstates 10.^(8:2:14)`, `noise.nreal 6`.*

---

## 3. The pinhole diameter of record

Dave's ruling 2: whichever of **2.0 λ/D at model 1024 / 193 rays** and
**1.0 λ/D at 2048 / 385** performs better on the 30 nm-surface rows and
the loop; record both, state the choice with its numbers.

The two are matched in SAMPLING, not in size: the pinhole's width at the
mask plane is `fill × MODEL / NGRID` px per λ/D, which is 3.96 px per
λ/D at both 1024/193 and 2048/385 — so 2.0 λ/D at the first is 7.9 px
across and 1.0 λ/D at the second is 3.96 px, the budget line's floor
being 6 px.  Only MODEL buys both resolutions at once, which is why the
1.0 λ/D leg needs 2048.

Runs: `pin20_1024`, `pin20_loop`, `pin10_2048`, `pin10_loop`.

---

## 4. The shared loop knobs — the contract CCMac mirrors

Three knobs were added to `dm_gauge_lib/dmg_loop.m`, the loop BOTH
gauges run.  They are gated on a synthetic instrument in
`mmacos/tests/tDmgLoop.m` (G9-G12; 14 of 14 pass), so the loop code is
pinned before either gauge uses it.

| knob | meaning | instrument contract |
|---|---|---|
| `opt.start_rms` (+ `opt.start_shape`) | the loop OPENS at a surface of this rms instead of at the set point — the DM's initial figure.  The starting surface is `start_rms × unit(shape)`, the shape defaulting to the set point's own field rescaled, so the residual at cycle 1 is `|start_rms − rms(A0)|`.  `L.surf_rms` reports the starting surface; `opt.reach` levels are dated in `L.k_reach` | none |
| `opt.recal_every` | cycles between re-measurements of the response matrix ON the surface the loop holds now.  0 = never (the matrix measured once, at the start).  Its cost is counted in `L.nstates`; `L.n_recal` / `L.k_recal` record when | **`ins.recal(cmd)` → `struct('est', <new estimator handle>, 'nstates', <states it cost>)`**.  In `zwfs_run` this is `calib_matrix_` re-run with the loop's current DM command as the calibration surface |
| `opt.intra` | the fraction of the NEXT cycle's drift increment that develops WITHIN one measurement's scan: frame *j* of *nf* sees `(j−1)/(nf−1)` of it.  The DM/thermal analogue of the camera's `cam.intra`, and the IFO's PZT-form drift term | **`ins.measure(cmd, aux)`** with `aux.dstep` a DM map.  A temporally stepped reading traces its frames separately; a single-frame or simultaneous reading ignores it |
| `opt.ref_walk` (+ `opt.ref_seed`) | a random walk, rad per cycle rms, of a NON-COMMON-PATH reference arm's phase relative to the test arm.  `L.ref_phase` is the walk | **`ins.measure(cmd, aux)`** with `aux.ref_phase` a scalar.  A common-path reading has no such arm and ignores it |

`aux` is passed only when a knob is on, so every instrument written
before this slice (the synthetic one in `tDmgLoop`, `tg96_run`'s) keeps
working unchanged, and **every run taken before it reproduces to the
last bit**: the drift increments are now drawn once, ahead of the loop,
in the same order from the same stream (gate: `tDmgLoop/G4`, plus the
`intra 0` identity in G11).

**What G11 does NOT assert, deliberately.**  A reading whose frames
straddle its scan reads the MIDDLE of that scan to first order, and
under a pure random walk that half-step of prediction can help as
easily as hurt.  Which way it goes is the instrument's business and is
measured on the engine (section 4b), not asserted on a synthetic.

---

## 5. The reference arm's own drift (P/SRI)

The one systematic a NON-COMMON-PATH interferometer has and a
common-path one does not: the reference arm's phase relative to the test
arm, walking between measurements.  Modeled as `pdi.ref_walk` rad per
cycle rms of a random walk, applied to PF only, in the loop under the
2 pm-per-actuator walk drift.  P rides along in the same runs as the
common-path control — it has no such arm and must be untouched.

Runs: `rw193_1e3`, `rw193_1e2`, `rw193_1e1`.

---

---

## Method notes carried by every section

- **Currency.**  Actuator-space rows on a 96×96 DM, 1 mm pitch, at the
  30 nm rms working surface (`battery.base_rms`, seed 7), with the
  response matrix measured ON that surface (`battery.calib_mode
  'matrix'`, `calib_surface 'base'` — Dave's S10 default).  Errors in
  pm.  Photons are counted **at the detector** and are per MEASUREMENT
  (one DM shape measured once; a reading's frames share the count);
  divide by the run's printed `throughput` for incident photons.
- **Capture range to 10%**: the largest working-surface rms at which a
  10 nm change on the 47 grid sites reads within 10% of its size (gain
  in 0.9…1.1), log-interpolated between ladder rungs.  The runner prints
  it after every ladder.
- **385 rays for the ladders, 193 for the photons** — the ZWFS record's
  choices, kept so the two campaigns' tables can sit side by side.
- Each run's own report carries every line quoted here; the tables below
  are transcriptions, not re-computations.
