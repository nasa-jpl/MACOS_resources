# The point-diffraction gauges for the DM Surface Gauge Comparison deck

TO's part of `macos/BRIEF_gauge_deck.md`, answering
`macos/BRIEF_to_gauge_deck.md` (Dave, 2026-09-13).  Numbers first; every
one has a run tag in this directory's `runs/` (or, for the record taken
before the 2026-09-13 split, in `../zwfs_dm96/runs/`).  Departures from
the brief are flagged **[departure]**.

Sheet and runner: `pdi_params` / `pdi_run` (the code is shared —
`zwfs_run` + `../dm_gauge_lib`, nothing copied).  Headless:
`./pdi_batch.sh TAG "pdi_params, <args>"`.  The full chain that produced
this report is, IN THE ORDER IT ACTUALLY RAN: `runs/gsmoke.sh` →
`gseq1.sh` (deliverables 1 and 2) → `gcap.sh` (9) → `gseq3.sh` (4b and
the rest of 9) → `gseq4.sh` (5) → `gseq2.sh` (3).  The drivers that
sequenced it were `gmaster.sh` and then `gmaster3.sh` / `gmaster4.sh`,
re-launched twice as the brief's priorities changed and once after an
OOM; `gmaster2.sh` is superseded.  Order was
cheapest-complete-deliverable-first, so that what landed was whole.

**Status of the brief's nine items** (seven from
`BRIEF_to_gauge_deck.md`, two added by `BRIEF_to_capture.md`) — stated
here so the reader knows what is measured and what is still running:

| item | state |
|---|---|
| 1. PF through the two decks | **done** — rows, photons and the loop rows, plus the frozen-reference control (`pfdeck`, `pfdeck_frz`, `pfdeck_loop`) |
| 2. capture range and photons, P and PF | **done** — sections 2a, 2b, 2c (`cap385p`, `cap385p_b*`, `noise193p_b*`) |
| 3. the pinhole diameter of record | **done** — sections 3a–3c; **2.0 λ/D stays the diameter of record**, both recorded (`pin20_1024`, `pin20_loop`, `pin10_2048`, `pin10_loop`) |
| 4. the four shared loop knobs + gates | **done** — built, gated (tDmgLoop G9–G13, 15/15) and run: the descent in section 9, the within-scan drift in **4b** (`intra193_0`, `intra193`) |
| 5. the reference arm's own drift | **done** — section 5 (`rw193_1e3/1e2/1e1`) |
| 6. layouts and parts | **done** (`pdi_layout.png`, `psri_layout.png`, `psri_render.png`, `pdi_vfig_util`; parts tables in the README) |
| 7. conclusions; README | **done** — the conclusions section below (which Dave ruled CCL lifts into the deck), README beside it |
| 8. unwrap the differential (`BRIEF_to_capture.md`) | **done and gated** — `dm_gauge_lib/dmg_unwrap.m`, tDmgLoop G13, 15/15; section 8 |
| 9. the start-rms ladder, both ways | **done** — sections 9a–9d (`cap_nouw`, `cap_uw`, `cap_*_recal`, `cap_state_*`, `descent193`, `descent193s/f`) |

---

## 0. The best configuration, and why

*Dave 2026-09-13: the deck's main body shows each approach ONCE, in its
best-performing configuration; everything else is backup.  So this
section names it and gives the numbers that make it the best; the route
there is sections 1-9 and the backup slides.*

**The point-diffraction approach's best configuration is the STEPPED
PINHOLE `P`, with a pinhole-only (shutter) frame per state and the
five-frame Schwider–Hariharan scan.**  Six frames a measurement, all in
the common path, one plate in the mask seat and nothing else added to
the shared front end.

| what the configuration buys | number | record |
|---|---|---|
| a 10 nm change on one actuator, matrix on the 30 nm surface | **0.9935 gain, 4 pm error, SNR 2790** — indistinguishable from the vector Zernike reading and from the P/SRI | `pdi193fbase` |
| photons for 1 pm | **3.3e13** at the detector through a flat-calibrated matrix (V 4.7e13, S 5.4e13, PF 1.9e14); through the matrix measured ON the working surface — the campaign's operating point — **9.8e13, level with S at 9.3e13** and 3.8× cheaper than the P/SRI's 3.7e14 | `pdi193f`, `noise193p_b30` |
| closed-loop hold of 3 pm | **2.3e12** photons per cycle noise-only, **7.0e12** under a 2 pm-per-actuator walk; steps to 0.000 pm, no fixed error | `ploop193` |
| capture range to 10% with the calibration left to age | **1.02 / 1.06 / 1.13 at 120 / 240 / 480 nm** — the P/SRI's range, in the common path, for the one extra frame | `pdi193state` |
| a 2% phase-step error | **4.9 pm** on a 12 nm figure, and the differential rows are the error-free ones to the digit (four-step least squares gives 421 pm) | `pdi193se_sh5` |
| camera bias drifting within the servo | exactly immune while it is constant across a scan; pays only for what develops BETWEEN its frames (5.3 pm at 1e15 with the whole step inside the scan) | `pcam193r`, `pcam193ri` |

| capture — the largest initial surface it can bring to 3 pm | **60 nm** on the reading alone; **100 nm (200 nm WFE) with unwrapping AND on-surface re-calibration every 10 cycles** — 0.207 pm at 1e15 and 2.066 pm at 1e13, 3 pm at cycle 26 | `cap_state_uw`, `cap_state_uw_recal` |
| within-scan DM / thermal drift | **helps**, 26–32% less hold error than a still DM, because a scan reads the surface at its midpoint | `intra193` |

**Why not the P/SRI (`PF`), which the paper builds.**  Its reference does
not depend on the working surface at all, and that is real: traced
end-to-end it holds gain inside 0.7% over a 16× range of surface where
every other reading here folds (section 1), and its picometre costs the
same at 160 nm as at 30 while P's costs 18× more (section 2c).  But the stepped pinhole
with a shutter frame *buys the same range in the common path*, and the
P/SRI's price is ~2× the light (5.1e12 / 1.5e13 photons per cycle
traced), a second arm to build and balance, and the one systematic no
common-path form has — its own reference drifting (section 5).  **Its
place in the deck is as the CAPTURE instrument, not the hold
instrument**, which is what sections 8 and 9 test.

**Why not the 4-step scan or the flat `|b|²`.**  Both are strictly worse
for one frame each: the flat `|b|²` is what makes P fold at 120 nm
(conclusion 3), and four-step least squares is what turns a 2% step
error into 421 pm (conclusion 6).

**The pinhole diameter is 2.0 λ/D** (section 3).  Against 1.0 λ/D the
30 nm rows are indistinguishable and the loop is not close — 3 pm from
< 1e13 photons per cycle against 3.6e13, the small pinhole passing 0.294
of the light against 0.821.  Its one advantage, range, is what the
shutter frame already buys here for free.

*Every item of both briefs is now measured; nothing in section 0 is
pending.*

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
the two chief optical paths from source to camera are **4453.1061 mm
each, difference 0.00e+00 mm**, bought with a 21.857 mm lens-glass
compensator in the test arm; the exit chiefs coincide after BS3 to
0.00e+00 mm and the two camera planes to 2.3e-13 mm.  (At the layout
figure's own resolution, 65 rays, both decks put 3210 of 3210 rays on
that camera; the record runs at 193.)  Its reference lens Lr1 is f 300
mm, F/2.92 on the 102.9 mm beam, conic **−0.5784** solved on the trace,
with the pinhole seat **+1.096 mm** past the thin-lens focus — the
values of record now live in `pdi_params` (`P.pdi.psri`) and
`psri_layout_fig('solve', false)` uses them.

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

### 1e. The loop rows (`pfdeck_loop`)

The same closed-loop hold metric as the rest of the campaign — gain 0.5,
60 cycles, the set point the 30 nm working surface with the matrix
measured on it, one measurement per cycle — with both arms traced.
610 traced states, 138 min.

| | PF, reference **traced** | PF, **synthesized** (`ploop193`) | P (`ploop193`) |
|---|---|---|---|
| noiseless step 1 nm / 10 nm, residual at cycle 60 | **0.000 / 0.000 pm** | 0.000 / 0.000 | 0.000 / 0.000 |
| per-cycle contraction ρ | 0.511 / 0.510 | 0.509 | 0.509 |
| photons per cycle for a 3 pm hold, noise only | **5.1e12** | 7.0e12 | 2.3e12 |
| … under a 2 pm-per-actuator walk | **1.5e13** | 2.5e13 | 7.0e12 |
| σ_n at 1e12 photons | 11.7 pm | 13.5 pm | 7.8 pm |
| held residual's spectrum under the walk, [<4, 4–12, >12] cycles/aperture | 0.25 / 0.72 / 2.20 pm | 0.25 / 0.72 / 2.2 | as V |

**No fixed error** — the traced reference takes both noiseless steps to
0.000 pm, so nothing about the real arm biases the held surface.

**And the built arm is CHEAPER in light than the model of it**, by 1.37×
noise-only and 1.67× under the walk.  That is not a modelling surprise
once the bench numbers are in hand: the physical pinhole couples 0.677
of the flat's focal light where the LP01 mode's overlap takes 0.587, so
the reference amplitude is larger, σ_n is 1.16× smaller, and photons go
as σ_n² (1.16² = 1.35, against the 1.37 measured).  **The synthesized
model was pessimistic about the P/SRI, not optimistic.**  It remains ~2×
the light of the common-path pinhole P, which is the 60/40 pickoff and
nothing else.

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

### 2a. A calibration that AGES — the matrix measured once at 30 nm (`cap385p`, 385 rays)

A 10 nm change on the 47 grid sites, read on a growing working surface,
with the response matrix left at the one measured on the 30 nm surface.

| base rms | P: gain / floor / SNR | PF: gain / floor / SNR |
|---|---|---|
| 30 nm | 0.9970 / 20 pm / 492 | 0.9967 / 20 pm / 493 |
| 40 | 0.9858 / 41 / 242 | 0.9989 / 40 / 247 |
| 50 | 0.9678 / 78 / 123 | 1.0012 / 72 / 140 |
| 60 | 0.9425 / 142 / 67 | 1.0034 / 105 / 96 |
| 80 | **0.4674** / 873 / 5.4 | 1.0078 / 171 / 59 |
| 100 | 0.0618 / 488 / 1.3 | 1.0122 / 239 / 42 |
| 120 | **−0.0147** (folded) | 1.0166 / 306 / 33 |
| 160 | −0.0417 | 1.0255 / 441 / 23 |
| 240 | −0.0182 | 1.0431 / 711 / 15 |
| 480 | −0.0094 | **1.0962** / 1522 / 7.2 |
| **capture range to 10%** | **62 nm** | **480 nm+ (holds at the last rung)** |

**Beside the Zernike readings on the same ladder** (`../zwfs_dm96/runs/cap385`):

| reading | L | I+ | S | V | **P** | **PF** |
|---|---|---|---|---|---|---|
| capture range to 10%, aging from 30 nm | 44 nm | 36 nm | 42 nm | 70 nm | **62 nm** | **480 nm+** |

That row is the deck's capture slide in one line.  Everything with a
reference that depends on the surface dies between 36 and 70 nm; the one
whose reference does not is still within 1% at 100 nm and within 10% at
480.  P sits with the Zernike readings here because its `|b|²` is taken
from the FLAT state — with a shutter frame per state it moves to PF's
column (conclusion 3, `pdi193state`).

*(b) Re-measured on the surface — `cap385p_b60/90/120/160`, rows
`{'base/grid'}` with the matrix on that surface.*

### 2b. The matrix RE-MEASURED on the surface (`cap385p_b60/90/120/160`, 385 rays)

The same 1 nm change on the 47 grid sites, but with the response matrix
measured ON the surface the DM is actually holding — the calibration a
bench would make in place.

| working surface | P: gain / floor / SNR | PF: gain / floor / SNR |
|---|---|---|
| 60 nm | 1.0003 / 3 pm / 297 | 0.9999 / 3 pm / 378 |
| 90 nm | 0.9581 / 41 pm / 23 | 1.0007 / 2 pm / 415 |
| 120 nm | 0.9687 / 19 pm / 51 | 1.0014 / 2 pm / 458 |
| 160 nm | 0.9670 / 17 pm / 56 | 1.0021 / 2 pm / 516 |

**Re-measuring buys P a factor of ~2.6 in working surface** — from a
62 nm aging range to gain within 5% at 160 nm — and costs PF nothing it
did not already have (0.2% at 160 nm).  That is the same conclusion the
Zernike readings reached (`../zwfs_dm96/runs/cap385_b*`: gain within 5%
to 160 nm for all of them), and it is *not* what makes the descent
work: section 9 shows re-calibration helping and the wrap still
stopping it.

**What it costs is light**, which is 2c.

### 2c. Photons for 1 pm at 30 / 60 / 120 / 160 nm (`noise193p_b*`, 193 rays)

What the re-measured calibration of 2b COSTS.  N(1 pm) at the detector,
each priced through the matrix measured on that surface,
`noise.nstates 10.^(8:2:14)`, `noise.nreal 6`.

| working surface | P | × its 30 nm value | PF | × its 30 nm value |
|---|---|---|---|---|
| 30 nm | 9.77e13 | 1.0 | 3.72e14 | 1.0 |
| 60 | 1.73e14 | 1.8 | 4.13e14 | 1.1 |
| 120 | 2.95e15 | **30** | 6.58e14 | 1.8 |
| 160 | 1.75e15 | **18** | 3.67e14 | **0.99** |

**The reading whose reference does not depend on the surface does not
pay for the surface in light either.**  PF's picometre costs the same at
160 nm as at 30 — within 1% — while P's costs 18–30× more.  P's ×30 at
120 nm against ×18 at 160 is NOT smooth, and is not smoothed here: the
same non-monotonicity shows in 2b's floor column (41 pm at 90 nm, 19 at
120, 17 at 160), so the 120 nm point should be read as "tens of times
worse", not as a number.

That factor sits ON TOP of the capture range: re-measuring gets P from
62 nm to 160 nm of working surface (2b), and the picometre there costs
~18× the light.  Both numbers belong on the slide.  For comparison the
Zernike readings pay 5× (L) to 40× (V) over the same span
(`../zwfs_dm96/runs/noise193_b*`), so P is inside that family and PF is
not in it at all.

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

### 3a. The two legs

| | **2.0 λ/D**, model 1024 / 193 rays | **1.0 λ/D**, model 2048 / 385 rays |
|---|---|---|
| surround transmission `t` | 0.719 | **0.283** |
| pinhole coupling η | 0.682 | **0.240** |
| throughput (detected / incident) | **0.821** | 0.294 |
| pinhole at the mask plane | 7.92 px (PASS) | 3.96 px (**NOT MET**) |
| detector px per actuator | 2.52 | 5.03 |
| **rows** 10 nm on one actuator | 0.9935 / 4 pm / SNR 2790 | 0.9937 / 4 pm / SNR 2967 |
| 1 nm on 47 grid sites | 0.9992 / 3 pm | 0.9990 / 3 pm |
| dense random 10 nm | 0.9985 / 338 pm | 0.9997 / 330 pm |
| **ladder** 30 nm | 0.9971 / 20 pm | 0.9967 / 20 pm |
| 60 nm | 0.9380 / 146 | **1.0024** / 104 |
| 120 nm | **−0.013 (folded)** | **1.0175** / 306 |
| **capture range to 10%** | **62 nm** | **120 nm+ (holds)** |
| N(1 pm), on-surface matrix | **9.8e13** | 2.1e14 |
| photons per cycle to hold 3 pm under the 2 pm walk | **< 1e13** | 3.6e13 |

*Runs: `pin20_1024`, `pin20_loop`, `pin10_2048`, `pin10_loop`.*

### 3b. The range is the DIAMETER's, not the sampling's

The two legs differ in two variables at once — diameter AND model/rays —
so the range could in principle be the better pupil sampling.  It is
not: `../zwfs_dm96/runs/pdi193d1` runs 1.0 λ/D at **1024 / 193**, the
same sampling as the 2.0 λ/D leg, and it also does not fold (1.02 / 1.06
/ 1.13 at 120 / 240 / 480 nm).  With sampling held fixed, the small
pinhole still buys the range.  **A cleaner reference is what a smaller
pinhole is for**, and the measurement says so.

### 3c. The choice: 2.0 λ/D stays the diameter of record

**On the brief's own two criteria the answer is not close.**  On the
30 nm-surface rows the two are indistinguishable — 0.9935 against
0.9937, 4 pm both.  On the loop 2.0 λ/D wins by more than 3.6×: 3 pm
held from **< 1e13** photons per cycle against **3.6e13**.  The small
pinhole passes 0.294 of the light where the large one passes 0.821, and
that is the whole story of its photon cost (2.2× on N(1 pm), > 3.6× in
the loop).

**Its one advantage — range — is available at 2.0 λ/D for one extra
frame.**  The 1.0 λ/D pinhole's 120 nm+ capture range against 62 nm is
real, and it is bought by having a reference the surface cannot corrupt.
But a pinhole-only (shutter) frame per state buys exactly that at 2.0
λ/D (conclusion 3, `pdi193state`: P becomes PF's twin on every row AND
the ladder), for a fifth frame and no change in throughput.  **Paying
2.8× in light for what a fifth frame gives free is the wrong trade**, so
**2.0 λ/D is the diameter of record** and the deck quotes it.

*Both are recorded, as the ruling asked.*  If a future bench cannot
afford the extra frame — a shutter has to be actuated, and at some scan
rate that matters — the 1.0 λ/D pinhole is the fallback that reaches the
same range with no moving part, at 2.8× the light.

*Run-record note:* `pin10_2048` was killed by a SIGTERM from another lane
sixteen minutes into its first attempt (exit 143), so `gseq2` moved on to
`pin10_loop` and the battery leg was re-queued behind it (`runs/gpin.sh`)
— which is why its timestamp is later.  Both model-2048 runs then ended
with **exit 137, which is NOT a failure here**: it is the documented
model-2048 crash AT EXIT, after everything is written (the campaign has
met it before, `../zwfs_dm96/runs/mask385`).  Both reports end "run
complete" (123.6 and 69.5 min) and both are intact.

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

### 4b. The within-measurement DM drift (V4) — it HELPS, and the camera case does not

`intra193_0` against `intra193`: the same loop with the DM still during
a scan, and with the whole cycle's drift developing across it.  Hold
error (pm rms over lit) at 1e15 photons per cycle:

| drift | | L | S | V | P | PF |
|---|---|---|---|---|---|---|
| 2 pm walk | DM still | 2.38 | 2.32 | 2.32 | 2.32 | 2.33 |
| | drift across the scan | 2.38 | **1.57** | 2.32 | **1.58** | **1.71** |
| 5 pm thermal ramp | DM still | 27.64 | 10.05 | 9.87 | 9.86 | 9.88 |
| | drift across the scan | 27.64 | **7.16** | 9.87 | **6.70** | **7.30** |

**The single-frame reading L and the simultaneous pair V are unchanged
to the last digit**, which is the plumbing gate: they see one instant
and the knob cannot reach them.  **The stepped readings improve by
26–32%.**

*Why, and why this is the opposite of the camera result.*  A reading
whose frames straddle its scan measures the surface at the scan's
MIDPOINT, which is half a measurement closer to "now" than its start —
a free half-step of prediction against any smooth drift.  Under the
ramp, where the lag is rate/(gG) ≈ 10 pm, that is worth a third of it.
The camera's within-scan drift (`pcam193ri`) costs 5.3–10.7 pm because
it is ADDITIVE BIAS on the detector, the thing a zero-sum step scheme
exists to cancel and can only cancel while it is constant across the
scan.  **DM drift within a scan is SIGNAL read at the right moment;
camera drift within a scan is bias read at the wrong one.**  Same knob,
opposite signs, for legible and different reasons.

This is why `tDmgLoop` G11 gates the CONTRACT and refuses to assert a
sign: on a synthetic instrument under a pure random walk the half-step
can help as easily as hurt, and which way it goes is the instrument's
business.  Measured on the engine, it helps.

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

Runs: `rw193_1e3`, `rw193_1e2`, `rw193_1e1` — 1e-3, 1e-2 and 1e-1 rad
per cycle, a hundredfold range.  P rides along as the common-path
control.

| | P (common path) | PF, ref walk 1e-3 | 1e-2 | 1e-1 |
|---|---|---|---|---|
| hold, noise only, 1e13 | 1.45 pm | 2.50 | 2.51 | 2.53 |
| hold, noise only, 1e15 | 0.14 | 0.25 | 0.25 | 0.25 |
| hold under the 2 pm DM walk, 1e13 | 2.72 | 3.42 | — | 3.44 |
| photons per cycle to hold 3 pm under the walk | < 1e13 | 4.8e13 | 4.9e13 | **5.0e13** |

**The floor it sets is 4% in photons for a HUNDREDFOLD range of walk.**
P is bit-identical across all three, as it must be — it has no such arm.

*Why it is nearly free, and the scope of that.*  A path-length change
between two arms is a PISTON on the retrieved phase: the solve returns
`X e^{-iψ}`, so ψ subtracts uniformly.  And piston is exactly the mode
the actuator estimator nulls — the sensor cannot see it, and since S10
the response matrix carries that null explicitly as a rank-one term.  So
the P/SRI's most-feared systematic lands entirely in the one mode a DM
servo neither observes nor controls.

**This is benign FOR A DM SURFACE SERVO and not in general.**  For the
paper's own application — absolute complex E-field reconstruction — the
same walk is a direct piston error on the answer, with nothing to null
it.  And the model is a path-length change: anything that TILTS or
distorts the reference arm rather than translating it is not a piston
and is outside this scope.

---

---

## 8. Unwrapping the differential

*Added by `macos/BRIEF_to_capture.md` (Dave / CCL, 2026-09-13) after the
descent finding was accepted: capture is a wrap problem, and it is the
same problem for every approach — the interferometer's four-step
included, since its phase wraps at the same ±π.*

**What was wrong before.**  Every phase reading in the campaign returns
a WRAPPED differential: `stepdiff` for S, `diffV` for V, the PDI `diff`
for P and PF are all `angle(X₁ conj X₀)`.  So a change larger than ±π of
phase — **±158 nm of surface** at 632.8 nm double-pass — comes back
folded *whatever the reading's absolute range is*.  That is why nothing
descended from a 100 nm surface toward a 30 nm set point: the opening
differential is ~70 nm rms of surface = **~1.4 rad rms of phase**, whose
peaks run well past ±π, so a large fraction of the map was wrapped
before the estimator ever saw it.

**`dm_gauge_lib/dmg_unwrap.m`.**  Two-dimensional least-squares phase
unwrapping on a mask (Ghiglia & Romero, *JOSA A* **11**, 107 (1994)), in
two stages:

1. the **unweighted** solve — the Poisson equation whose source is the
   divergence of the wrapped gradients, with Neumann boundaries, solved
   directly by mirroring the source into an even-symmetric 2M×2N array
   and dividing its FFT by the discrete Laplacian's eigenvalues.  That
   is the FFT form of their DCT solution; `dct2` is a toolbox function
   and this tree must run with **no external dependency** (the release
   gate), so it is written out.
2. the **masked** refinement — the weighted normal equations (weights =
   the mask, so no phase crosses the boundary) by preconditioned
   conjugate gradients with stage 1 as the preconditioner, their §5.
   Without it the region outside the mask, where there is no data, pulls
   on the answer inside it.

The work is done on the mask's bounding box: the pupil is ~NGRID px
across a MODEL-px frame, so this is a ~200×200 solve, not 1024×1024, and
it costs ~30 ms per differential — nothing beside a trace.

**Residues.**  A wrapped field is consistent only where every 2×2 loop
of wrapped differences sums to zero.  Least squares *spreads* an
inconsistency rather than failing on it, so `info.nres` counts those
loops inside the mask: a map beyond the pixel-gradient limit is
**reported, not silently wrong**.  `info.maxgrad` is the largest wrapped
gradient, in rad per pixel, which is the limit itself.

**Gates** (`tests/tDmgLoop.m`, G13; **15 of 15 pass**, and the whole
mmacos fast suite is **469 pass / 0 fail** with these changes in — the
push gate the brief named, re-run on 2026-09-14 against the tree that is
now on origin, i.e. including the OOM fix to the descent's calibration
handling):

| gate | result |
|---|---|
| a wrapped ramp, 1.5 waves across a full box | exact to **8.5e-14 rad**, 0 residues |
| a band-limited random surface, 1.5 waves PV, on a DISC mask | **6.7e-13 rad**, 0 residues, max gradient 0.90 rad/px, 16 PCG iterations |
| the same at 3.0 waves PV | **1.3e-12 rad**, 0 residues, max gradient 1.81 rad/px |
| a map that was never wrapped | passes through to **5.4e-14**, and reports `wrapped = false` |
| 40 waves PV — beyond the pixel-gradient limit | **644 residues reported**, max gradient 3.14 = π |

**Where the limit moves to.**  From the wrap (±158 nm of surface) to the
**pixel gradient**: adjacent detector pixels must differ by less than π,
i.e. by less than 158 nm of surface *between neighbouring pixels*.  The
DM's surface is smooth at that scale — 4 detector px per actuator at 385
rays, 2.5 at 193 — so the limit becomes several hundred nm rms.  The
ladder of section 9 measures where it actually lands.

**The knob, and what it does not disturb.**  `battery.unwrap` (default
**false**, so every record taken before 2026-09-13 reproduces) applies
it to the four wrapped readings — S, V, P, PF — before the estimator,
in the measurement differential AND in the calibration's class maps, so
the matrix and the measurement always agree.  `loop.unwrap` is `'auto'`
by default: the loop turns it on exactly when `loop.start_rms` is set,
which is the case it exists for.  **L is untouched** (a linear map, no
wrap); so are F, I and I+ — they solve an absolute phase and are
differenced afterwards, which unwrapping the difference cannot mend.

*Non-disturbance, measured:* with the knob off, the dev-resolution bench
record is **bit-identical** to the pre-unwrapper run — G3, G4, G5, G6 and
G7 reproduce exactly, **v3dev G4 = 0.296 pm** (`runs/uwoff_ref` against
`runs/pfsmoke_ref`), and the battery rows likewise (`runs/uwoff_bat`
against `runs/pfdeck_smoke2`).

---

## 9. The start-rms ladder, both ways

Starts 30 / 60 / 100 / 150 / 200 / 300 nm rms, the matrix measured at
each start, gain 0.5, `recal_every` 10 and never, 1e13 and 1e15 photons
per cycle, readings L, S, V, P, PF — run with the unwrapper off and on.
Runs: `cap_nouw`, `cap_uw`, `cap_nouw_recal`, `cap_uw_recal`, and —
because `gcap` runs P in its RECORD configuration (a flat `|b|²`, four
frames, which is what folds it at 120 nm) while section 0 recommends the
SHUTTER form — `cap_state_uw` and `cap_state_nouw`, the same ladder for
P with `pdi.b2 'state'`.  The slide has to quote the capture of the
configuration it shows.

**[departure] 193 rays, not 385.**  The box is the limit: the 385-ray
ladder is ~4× these states and the queue behind it (the within-scan
drift, the reference-arm walk, the pinhole trade) would not run at all.
The wrap is a property of the PHASE, so the threshold the unwrapped-off
arm measures does not depend on the ray count; the pixel-gradient limit
the unwrapper trades it for DOES (4 px per actuator at 385, 2.5 at 193),
so **193 is the pessimistic choice for the unwrapped arm** — a 385-ray
ladder can only do better.

**[departure] K 40, not 60.**  At gain 0.5 the contraction is 0.5 per
cycle, so a descent that has not reached 3 pm by cycle 40 (0.5⁴⁰ = 9e-13
of the start) will not; the steady-state tail is still 20 cycles.

*The runner also prints, per start and reading, the OPENING
differential: its wrapped rms, the residue count inside the mask, the
largest wrapped gradient in rad per pixel, and the rms the unwrapper
returns — against the truth, which is printed beside it.  That line is
the direct evidence, independent of whether the loop then converges.*

**It also separates the TWO ways a descent can fail, which the ladder
alone would confuse.**  A reading can fail because its differential is
FOLDED — it is returning nearly the right map, wrapped — or because its
REFERENCE has collapsed: at a large working surface the focal core is
gone, and then the reading returns a small, wrong map that is not
wrapped at all.  Unwrapping answers the first and cannot touch the
second.  The diagnostic tells them apart: a folded reading shows a
wrapped rms near the wrapped-random value (π/√3 × 50.4 nm ≈ 91 nm of
surface) with residues, and an unwrapped rms that recovers the truth; a
collapsed reference shows a wrapped rms FAR BELOW the truth with no
residues, and unwrapping changes nothing.  The dev-resolution smoke
(`runs/sm_cap`, 48×48 DM, 1.7 px per actuator) shows the second: from a
150 nm start, truth 119 nm, S and P both read ~25–29 nm with zero
residues, and the unwrapper returns the same — their references are
gone, not folded.  Which failure each reading meets, and at what start,
is what the record ladder measures.

---

### 9a. The result, and it reverses the premise for four of the five readings

**`BRIEF_to_capture.md` accepted that "capture is a wrap problem, not a
gain problem ... S and P diverge, PF barely moves".  At record
resolution that is true of PF and FALSE of S, V and P.**

The opening-differential diagnostic (`cap_nouw`, 193 rays, 96×96, the
matrix measured at each start) says so directly.  Truth is the
differential the reading is being asked for:

| start | truth | S | V | P | PF |
|---|---|---|---|---|---|
| 60 nm | 30 nm | 16.4 nm, **8 res**, grad 3.14 | 25.6, 0 res, 1.97 | 23.3, 0 res, 1.77 | 27.0, 0 res, 2.08 |
| 100 | 70 | 22.3, **0 res**, 1.83 | 26.5, 0 res, 2.10 | 26.3, 0 res, 2.09 | 59.8, **70 res**, 3.14 |
| 150 | 120 | 24.1, **0 res**, 1.58 | 28.5, 0 res, 1.79 | 28.4, 0 res, 1.78 | 75.4, **928 res**, 3.14 |
| 200 | 170 | 24.2, **0 res**, 1.62 | 28.7, 0 res, 1.88 | 28.5, 0 res, 1.88 | 77.2, **2580 res**, 3.14 |
| 300 | 270 | 24.0, **0 res**, 1.48 | 28.4, 0 res, 1.66 | 28.3, 0 res, 1.65 | 77.8, **4928 res**, 3.14 |

*(rms of the wrapped map, 2π residues inside the mask, largest wrapped
gradient in rad per pixel.)*

**S, V and P are not folded — they are blind.**  Past ~60 nm they return
~24–28 nm whatever the truth is (70, 120, 170, 270), with **zero**
residues and a peak gradient well under π.  There is nothing to unwrap.
Their focal-plane references have collapsed: the core that makes the
reference is gone at that surface.  **Only PF wraps**, because it is the
only reading whose reference survives a large surface, so it is the only
one still returning a large map — and that map genuinely folds, 70 →
4928 residues with the gradient pinned at π.

**And the ladder confirms it: unwrapping alone moves nothing.**
`cap_uw` against `cap_nouw`, largest start that reaches 3 pm in 40
cycles, at 1e13 and 1e15 photons per cycle alike:

| reading | L | S | V | P | PF |
|---|---|---|---|---|---|
| unwrap **off** | 30 nm | 30 nm | 60 nm | 60 nm | 60 nm |
| unwrap **on** | 30 nm | 30 nm | 60 nm | 60 nm | 60 nm |

The S, V and P rows are bit-identical between the two arms, as they must
be for a map with no residues.  For PF the unwrapper does real work
short of convergence — from a 100 nm start the final residual goes
64 095 → 5 383 pm and the loop starts CONTRACTING (ρ 0.615, reaching
10 nm at cycle 6) where it had diverged — but 3 pm stays out of reach.

### 9b. What DOES raise it: unwrapping AND re-calibration, together

| PF from a 100 nm start (200 nm WFE), 1e15 photons per cycle | final residual | k(10 nm) | k(3 pm) |
|---|---|---|---|
| neither (`cap_nouw`) | 64 095 pm | — | — |
| unwrap only (`cap_uw`) | 5 383 pm | 6 | — |
| re-calibrate only (`cap_nouw_recal`) | 63 520 pm | — | — |
| **both** (`cap_uw_recal`, `descent193`) | **0.245 pm** | **6** | **26** |

**Neither alone does anything; together they take PF's capture from 60
nm of surface to 100 nm — 200 nm WFE, the top of the DM's stated initial
figure.**  The mechanism is legible: the unwrapper makes the FIRST
estimate good enough to move the surface at all, and the re-calibration
then keeps the matrix valid as the surface changes underneath it.  It
works at **1e13 as well as 1e15** photons per cycle (3 pm at cycle 28
against 26), so capture at 100 nm is reference-limited, not light-limited.

**The brief's own descent question, answered** (`descent193`, all five
readings from 100 nm, K 60, recal 0 and 10, both photon levels): **only
PF captures, and only with re-calibration.**  L, S, V and P diverge
without it and stall at 170 000–450 000 pm with it.

### 9c. The recommended configuration captures too — P with a shutter frame IS PF here

`cap_state_uw` / `cap_state_nouw` run P with `pdi.b2 'state'`, the
configuration section 0 recommends.  It tracks PF to four digits at
every rung: 5 383.2 pm against PF's 5 383.4 at a 100 nm start with
unwrapping, 64 092 against 64 095 without, ρ 0.615 both.  So the
common-path form inherits the capture behaviour along with the range.
**Measured, not inferred** (`cap_state_uw_recal`, `descent193s`): P with
a shutter frame, unwrapping and re-calibration every 10 cycles captures
a **100 nm** surface at BOTH photon levels — 0.207 pm at 1e15 and
2.066 pm at 1e13, 3 pm at cycle 26–27, ρ 0.716, 3 re-calibrations.  At
150 nm it reaches 10 nm at cycle 22 but not 3 pm (9 324 pm), so the
ceiling sits between 100 and 150 nm of surface.  **The configuration the
deck recommends captures the DM's stated initial figure and then holds
it.**

### 9d. The re-calibration CADENCE is not the binding constraint

`descent193f` probes it (P shutter, unwrapping, 1e15, K 20): every 2
cycles gives 6.5 pm at cycle 20 for **9** re-calibrations, every 5 gives
11.2 pm for **3**.  Both contract; neither reaches 3 pm, because K 20 is
short of the ~26 cycles the descent needs at ρ ≈ 0.7.  **Every 10 cycles
is enough** — a faster cadence buys a little early speed at 3× the
calibration cost, and what actually sets the time to 3 pm is the cycle
count, not the cadence.

---

## Conclusions for `deck_pdi` (the brief's item 7)

What the point-diffraction lane has settled, in the form the deck can
carry.  Each line names its record.  (Placed last because it draws on
every section above.)

**This section IS the deck text for CCL to lift** (Dave, 2026-09-14:
"Leave deck_pdi alone, CCL will lift them").  `deck_pdi.md` and
`deck_pdi.pptx` are therefore untouched — they are the lane's DRAFT
record, the .pptx is built from the .md, and editing one without
re-syncing the other splits them.  The gauge deck (`deck_gauges`) is
CCL's to assemble; these numbered conclusions plus section 0's
configuration table are what it needs.

1. **At the operating point the three exact readings are one reading.**
   On the 30 nm working surface with the matrix measured there, a 10 nm
   change on one actuator reads 0.9935 / 4 pm for V, P and PF alike,
   where the stepped Zernike reading S reads 0.9885 / 5 pm
   (`pdi193fbase`).  Nothing separates them at null; everything that
   separates them is range, light and systematics.

2. **P is the cheapest of the point-diffraction forms, and at worst
   level with the stepped Zernike reading.**  N(1 pm) at the detector
   depends strongly on which calibration the reading is priced through,
   so the two conditions are separated:

   | calibration | P | S | V | PF |
   |---|---|---|---|---|
   | matrix on the FLAT (`pdi193f`) | **3.3e13** | 5.4e13 | 4.7e13 | 1.9e14 |
   | matrix ON the 30 nm working surface (`noise193p_b30`, `pdi193state`) | 9.8e13 | 9.3e13 | — | 3.7e14 |

   **The campaign's operating point is the second row** (the response
   matrix measured on the working surface is the default since S10), and
   there **P and S cost the same light** — P's 1.6× advantage is a
   flat-matrix artefact, and quoting 3.3e13 against 9.25e13 would
   compare two different calibrations.  What survives both rows is the
   ORDER of magnitude between the common-path forms and the P/SRI: the
   waveguide form is 3.8–5.8× either of them, because 60% of the beam
   goes to an arm that returns a fraction of it as reference while the
   test beam keeps 40%.  In closed loop (on-surface matrix throughout)
   3 pm is held from **V 1.5e12 < P 2.3e12 < S 2.6e12 ≪ PF 7.0e12**
   noise-only, and **V 5.3e12 < P 7.0e12 < S 7.5e12 ≪ PF 2.5e13** under
   a 2 pm walk (`ploop193`).  With the arm actually TRACED the P/SRI is
   cheaper than its own model — 5.1e12 and 1.5e13 (`pfdeck_loop`),
   because the physical pinhole couples more of the focal light than the
   waveguide mode's overlap does — but still ~2–3× the common path.
   The P/SRI form costs ~6× the common-path pinhole because 60% of the
   beam goes to an arm that returns a fraction of it as reference while
   the test beam keeps 40%.  In closed loop the same ordering holds:
   3 pm held from **P 2.3e12** / PF 7.0e12 photons per cycle noise-only,
   **P 7.0e12** / PF 2.5e13 under a 2 pm walk (`ploop193`).  With the
   arm actually TRACED the P/SRI is cheaper than its own model — 5.1e12
   and 1.5e13 (`pfdeck_loop`), because the physical pinhole couples more
   of the focal light than the waveguide mode's overlap does — but still
   ~2× the common-path pinhole.

3. **P's fold was the flat |b|² assumption, not the pinhole.**  With a
   pinhole-only (shutter) frame per state — 5 frames instead of 4 — P
   becomes PF's twin on every row AND the ladder, 1.02 / 1.06 / 1.13 at
   120 / 240 / 480 nm (`pdi193state`).  The same range in the common
   path, for one extra frame.

4. **The P/SRI's range survives being built, and it is the only reading
   on this bench that has one.**  Capture range to 10% with the matrix
   left at the 30 nm surface (`cap385p`, and `../zwfs_dm96/runs/cap385`
   for the Zernike readings): **I+ 36, S 42, L 44, P 62, V 70 nm — and
   PF 480 nm+**, still within 1% of unit gain at 100 nm.  With both arms
   traced the gain stays inside 0.7% over that whole 16× range, flatter
   than the synthesized model of it (`pfdeck`).  Every reading whose
   reference depends on the surface dies between 36 and 70 nm; the one
   whose reference does not, does not.  **And it does not pay for the
   surface in LIGHT either**: N(1 pm) is 3.7e14 at 30 nm and 3.7e14 at
   160 nm — within 1% — where P's goes from 9.8e13 to 1.8e15, 18×
   (`noise193p_b*`).  Re-measuring the matrix on the surface restores
   every reading's GAIN (P within 5% to 160 nm, 2b) but not its photon
   cost; that factor is the real price of operating far from null, and
   only the P/SRI escapes it.

5. **The non-common-path reference is priced, and it is small at the
   operating point.**  Absolutely, the reference moving with the state
   costs 5.9 pm on a 13 nm figure; differentially it is a 10%-class
   effect on the noise floor and nothing on the gain or the range
   (`pfdeck` vs `pfdeck_frz`); in closed loop it costs no fixed error at
   all — both noiseless steps go to 0.000 pm (`pfdeck_loop`).  **And its
   own reference-arm drift turns out to be nearly free for a DM servo**:
   a hundredfold range of walk (1e-3 to 1e-1 rad per cycle) costs 4% in
   photons, because a path-length change between arms is a PISTON and
   piston is the one mode the actuator estimator nulls (`rw193_*`).
   That is benign for a surface servo and NOT in general — for the
   paper's absolute E-field reconstruction the same walk is a direct
   error, and a reference arm that tilts rather than translates is
   outside the model.

6. **The five-frame scan buys step-size immunity for one frame.**  Under
   a 2% phase-step error, four-step least squares turns a 12 nm figure
   into 421 / 251 pm of error (P / PF) and the flat stops reading zero;
   the Schwider–Hariharan five-frame scan of the paper reads **4.9 /
   2.2 pm** and its differential rows are the error-free ones to the
   digit (`pdi193se_ls`, `pdi193se_sh5`).

7. **Camera drift: a temporal PSI is exactly immune to a bias constant
   within its scan, and pays only for what develops between its own
   frames.**  At the campaign's photon levels the paper's 1 electron per
   pixel is invisible to every reading (`pcam193`); in the relative form
   the single-frame reading L imprints 10.8 nm and the simultaneous pair
   V 89 pm, while S / P / PF read their noise-only values to the digit
   (`pcam193r`).  With the whole step developing WITHIN each scan
   (`pcam193ri`) S / P / PF pay 5.4 / 5.3 / 10.7 pm at 1e15 — 2000×
   less than the single-frame reading and 17× less than the pair.
   **The DM's own drift within a scan is the opposite case and HELPS**
   (`intra193` vs `intra193_0`): the stepped readings gain 26–32%
   because a scan measures the surface at its midpoint, half a
   measurement closer to now, while L and V are unchanged to the digit.
   Camera drift within a scan is bias read at the wrong moment; DM drift
   within a scan is signal read at the right one.

8. **The pinhole diameter of record is 2.0 λ/D.**  Against 1.0 λ/D the
   30 nm rows are indistinguishable (0.9935 vs 0.9937, 4 pm both) and
   the loop is not close — 3 pm held from **< 1e13** photons per cycle
   against **3.6e13**, because the small pinhole passes 0.294 of the
   light where the large one passes 0.821.  The small pinhole's one
   advantage is range (capture 120 nm+ against 62 nm, and it is the
   DIAMETER's doing, not the sampling's — `pdi193d1` shows it at matched
   sampling), and **a shutter frame buys exactly that at 2.0 λ/D for one
   extra frame and no light**.  Paying 2.8× in light for what a fifth
   frame gives free is the wrong trade.  The 1.0 λ/D pinhole stays
   recorded as the fallback for a bench that cannot actuate a shutter.

9. **Capturing the initial figure: only the surface-independent
   reference captures, and only with BOTH unwrapping and
   re-calibration.**  The largest initial surface a loop closed through
   each reading can bring to 3 pm: L 30, S 30, V 60, P 60, PF 60 nm —
   **and unwrapping alone changes none of them** (`cap_nouw` vs
   `cap_uw`, bit-identical for S, V and P).  The premise it was built on
   is right only for PF: past ~60 nm S, V and P return ~24–28 nm
   whatever the truth is, with ZERO 2π residues — they are BLIND, not
   folded, their focal references gone.  PF is the only reading still
   returning a large map, and that map does fold (4 928 residues at a
   300 nm start).  **Unwrapping AND re-calibrating together take PF from
   60 nm to 100 nm of surface — 200 nm WFE, the top of the DM's stated
   initial figure — where neither alone moves it** (0.245 pm against
   5 383 and 63 520; `cap_uw_recal`, `descent193`).  It works at 1e13 as
   well as 1e15 photons per cycle, so capture there is reference-limited,
   not light-limited.  P with a shutter frame tracks PF to four digits
   (`cap_state_uw`), so the recommended configuration inherits this.

**The recommendation the lane supports**, for the deck's main body:
**the point-diffraction approach's best configuration is the stepped
pinhole P with a shutter frame per state and the five-frame scan** — the
common path's light and the P/SRI's range and step immunity in one
instrument, at ~1/6 the photons of the waveguide form and with no
reference arm to drift.  The P/SRI's own value is that its reference
does not depend on the surface AT ALL, which is what makes it the
capture instrument rather than the hold instrument; sections 3 and 4
say whether that holds at the diameters and the initial figures that
matter.

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
