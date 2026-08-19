# afocal4 S4c — one-page status

**Three questions: is basin 2 a design or a stall, what does the RIM pupil convention
measure, and does it soften the 343 mm fork.** Detail: `RESULTS.md` §S4c. Reproduce:
`afocal4_basin2` (one process per standoff) + `afocal4_basin2_merge`, and `afocal4_fork`.
Gated by `tPupilMap` (12/12) and `tAfocal4` (8/8).

---

## 1. Basin 2, solved long — the caveat closes, the curve barely moves

The S4b solves did not converge; they stalled. At 3e-3 scaled steps the study's forward
difference reads the merit's gradient **17% low** on an objective that is smooth to 1e-5,
which is enough to trip `FunctionTolerance` and hand back `exitflag 3`. Re-solved from
three independent seeds at ~1900 evaluations per point (against ~110), with the merit,
sampling, DOF set and packaging wall untouched:

| iface | WFE nm, S4b | WFE nm, S4c | blur µm | breathing % | wander µm | delivered by | where it stands |
|---|---|---|---|---|---|---|---|
| 50 | 14 481 | **12 915.5** | 238.1 | 1.342 | 241.1 | restart | floor, \|g\| 0.085 |
| 90 | 17 066 | **17 063.3** | 190.6 | 0.561 | 194.8 | restart | **on the packaging wall** |
| 140 | 16 367 | **15 927.6** | 152.9 | 0.082 ✓ | 157.4 | restart | floor, \|g\| 1.44 |
| 220 | 12 524 | **12 285.8** | 154.8 | 0.077 ✓ | 159.0 | **third seed** | floor, \|g\| 1.07 |
| 343 | 10 521 | **10 407.0** | 157.0 | 0.124 ✓ | 161.2 | restart | floor, \|g\| 0.80 |
| target | 71 | 71 | 47 | 0.4 | 56 | | |

**Nothing that was concluded from basin 2 changes.** It still runs 10.4–17.1 µm of
wavefront against 71 nm, still holds 153–238 µm of pupil blur against 47, and is still the
branch that buys pupil control. The flat wavefront column was not an artefact: 90 mm, its
worst point, moves 0.02% under a 17× budget.

**Two rows earn their re-solve.** 50 mm falls 10.8%, and at 220 mm a seed the S4b sweep
never had wins outright — breathing 0.118 → 0.077% at 1.9% less wavefront error.

**90 mm is constrained, not unconverged.** Its closure puts M3 at z = +0.500 m against a
500 mm minimum, and the finite-difference step in the M1–M2 spacing walks it into the
incoming beam. The merit still has a gradient there; descending it is not allowed.

**Worth Dave's eye:** at 343 mm the third seed reaches 10 331 nm — the same wavefront — while
holding magnification breathing at **0.0117%**, ten times better than the delivered design.
The merit cannot see it: both are far inside target and `merit_floor` stops paying.

## 2. The rim convention — and the anchor is not where the action is

`pupil_map` gains `'anchor','rim'` (cones anchored on the flat plane through the object
element's rim, normal to its axis) and `.rim_zone` (blur and wander over the outer 10% of
the pupil radius). Both opt-in; the surface anchor is bit-identical to the pre-change
function on every field of both fixture decks, and pinned there by test.

**His decks already declare the rim plane.** Stop 50 mm ahead of the pole; sag of an
R = −2500 mm parabola at r = 500 mm is 50.000 mm; measured gap between the two planes
**1.7e-13 mm**. Getting there required evaluating the rim's r² fit at the *declared* beam
edge — the outermost traced ray sits 1.15 mm inside a 500 mm aperture and reports the sag
0.23 mm short, sampling dressed as geometry. The decks `afocal4_build` emits declare their
stop at the vertex, so on those the rim sits 50.000 mm ahead of it; the rim is therefore
never read from `ApStop`.

**And the choice is resolvable:** pupil-image depth of focus λ/(2·NA²) = 14–16 µm (λ/NA²,
the form Dave quoted, is ~30 µm) against 54–61 µm of primary sag imaged at m².

**Then the measurement.** Pole → rim moves the blur by **0.01–0.06%** across Rodgers'
whole ladder and both fork branches. The outer-10% zone moves it by **10–34%**. Everything
the convention does to these numbers, it does through *where on the pupil you look*, not
*which plane you look from*. Where the anchor does bite is the column it exists for — the
convergence surface, which carries 15.3 µm rms of imaged primary sag under the surface
anchor and 0.11 µm under the rim, a factor of 140.

## 3. The fork at 343 mm, both conventions — it does not soften

| at 343 mm | WFE nm | blur, surface | blur, rim | blur, RIM ZONE | breathing % |
|---|---|---|---|---|---|
| basin 1, flat M4 (his three-mirror) | **266.0** | 761.0 | 760.6 | **845.7** | 3.875 |
| basin 2, powered M4 (long-solved) | 10 407.0 | **157.0** | 157.1 | **210.7** | **0.124 ✓** |
| target | 71 | 47 | 47 | 47 | 0.4 |

* **Does the flat branch's rim-edge image look better? No — 11% worse than its own
  average** (845.7 against 761.0 µm), and still 18× its target. The edge-conjugacy metric
  does not rescue the design that declines pupil control.
* **Does the powered requirement relax? Partly, and against the fourth mirror.** The
  powered design degrades *more* toward the rim (34% against 11%), so what it buys falls
  from **4.8× to 4.0×** — the rim metric takes 17% off the fourth mirror's value while the
  wavefront price stays at **39.1×**.

**One spec should be rewritten in the rim convention.** The convergence surface against its
ideal image reads 0.0174 mm surface-anchored and **0.1853 mm** rim-anchored — the
difference between 12× inside the 0.2 mm target and 1.08× inside it. A flat object's ideal
image is a plane and the powered design does not deliver one; the curved-object reference
was absorbing that.

**And the metric does not move the design either.** Re-solving the powered branch against a
rim-weighted merit — same log-domain form, rim-zone blur and wander replacing the
full-aperture terms, 504 evaluations from the long-solved design — buys **0.4%** on the
quantity it was asked to minimise (210.7 → 209.8 µm), costs 0.17% in wavefront error, and
leaves the exchange rate at **39.2× against 39.1×**. The exchange rate is a property of the
optics, not of which part of the pupil the metric is read on.

## Open for Dave and Mike

Unchanged: the operating point and the instrument envelope, jointly. Added: whether
magnification breathing is worth more than the merit's floor says — the 343 mm third seed
holds 0.0117% for nothing measurable — and whether the interface surface-figure spec should
be written in the rim convention, where it nearly binds.

## Caveats stated rather than smoothed over

* Only the 50 mm and 220 mm rows moved by more than 3%; the rest of the long solve bought
  less than 1%, and no point reached first-order optimality. Every delivered design is
  reported as a **plateau demonstrated by measurement** (gradient + hand-walked descent),
  not as a converged minimum.
* Steepest descent found nothing available at every design, before and after the long
  solve, while Gauss-Newton restarts found 0.02–2.1%. The descent probe is a lower bound.
* The 343 mm cold seed never reached a sound design (anchoring residual 151 mm); it is
  excluded from the basin spread rather than averaged into it.
* The φ₄ → 0 conditioning limit of the pupil metric (S4b §4) is unchanged and still applies
  to the flat-M4 row's pupil columns.
