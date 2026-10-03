# e2e2 — the improved TMA design flow

Input parameters → Korsch axial starting point → off-axis → fold → relay
and focal plane → scoring, each stage a gated step that emits a committed
`.in`, a parameter-provenance table, a thorough report, and standard
views.  **The product is the FLOW** — reusable stage drivers a user can
re-parameterize — not one telescope.

Sibling of `../e2e/`, which this supersedes as the design-flow example.
It folds in everything the Rodgers offset-field reproduction taught us
(`../../rodgers1/PACKET.md`, Addenda 8–10): joint solve, stated
references, solve set ≠ scoring set, pupil gate, parameter provenance per
stage.  `e2e` remains the reference for the stages e2e2 does *not*
duplicate — segmentation onward.

All user knobs live in **`e2e2_params.m`** — one file, commented.  Nothing
is computed there: a number in that file was **chosen**, a number in a
report was **derived**.

## The design point

The Rodgers offset-field coaxial TMA, **scaled to D = 3 m** (every length
× 3/5 from his 5 m `.seq` geometry), **f/20** (EFL 60 m) off an f/1.2358
primary, **λ = 500 nm**, a **0.4° × 0.4° used box**, and an M1
perforation **measured** with the secondary's shadow as its floor
(0.2331 m, 2.4 % of the area).  Target:
**diffraction-limited at 500 nm across the used box** — RMS ≤ λ/14 ≈ 36 nm
and Strehl ≥ 0.8, both reported.

The field is **twice his**, and the width was set by measurement — see
*How wide a field does this hold?* below.  It was briefly 0.6° (three
times his); that box drove the relay past the size of the primary and
spent 46 % of the wavefront budget at stage 1, so it was narrowed.

Scaling a validated design rather than inventing one buys a **first-order
gate no de-novo layout has**.  f/# is scale-invariant, so the constraint
that pins M3's radius — his `CUY UMY -0.025` paraxial marginal-angle
solve — carries over unchanged, and the builder derives R3 from the f/20
constraint alone.  The two must agree.  They do, to **2.4 ppm**.

Note the wavelength is *not* scaled.  His study closed at ≤ 40 nm at
1 µm; geometric wavefront error scales with the design, so that is ~24 nm
here — the target is reachable in principle, but at 500 nm it is a much
tighter ask in waves, and only at the reference convention his numbers
were reported under.

## Stages

**Three stages, and the telescope is finished at stage 2.**

| runner | consumes | produces |
|---|---|---|
| `s1_axial.m` | `e2e2_params.m` | the axial anchor: `s1_axial.in/.mat`, views, field map, `s1_report.txt` |
| `s1_fov_sweep.m` | `s1_axial.in` | *diagnostic*: how much field the architecture holds — `s1_fov_sweep.{txt,mat,png}` + a solved deck per candidate |
| `s2_fold.m` | s1 artifacts | **the delivered telescope**: fold + the (tilt, bias) clearance frontier, `s2_fold.in/.mat`, views, field map, `s2_report.txt` |
| `s3_score.m` | s2 artifacts | final ladder, coma and distortion maps, cross-stage provenance: `s3_report.txt`, `s3_score.mat`, `s3_maps.png`, `s3_views.png` |

`relay_followon/` holds the relay driver and the record of two failed
attempts.  It is **not** part of the flow — see *The relay* below.

### The delivered telescope

Scored on a uniform 13 × 13 grid over the used box, all four references:

| rung | max RMS | avg | min Strehl | vs the 35.71 nm bar |
|---|---|---|---|---|
| strict-chief | 49.220 nm | 28.351 | 0.679 | **FAIL** |
| strict-centroid *(primary)* | **30.001** | 20.047 | 0.867 | PASS |
| + best focus | 29.380 | 18.691 | 0.872 | PASS |
| + LS tip/tilt | **20.380** | 12.176 | 0.936 | PASS |

**Diffraction-limited at the primary reference and everything more
permissive; missed at the strictest.**  The four spread by **2.42×**,
which is why every number in this flow names its rung.  The 13 × 13 grid
reproduces the 9 × 9 the stage verdicts used, so these are not
sampling-limited.

Alongside, never inside, the wavefront table: coma (centroid-minus-chief)
runs 0.11–5.33 µm, and the chief-ray mapping is **f·θ to −0.0075 %** —
the traced EFL right to 75 ppm — with a 630 µm rms departure of which
522 µm is genuinely nonlinear.

### S1 — the Korsch axial starting point

A coaxial on-axis three-mirror anastigmat, solved jointly with its
detector.  **4.140 nm** (+LS tip/tilt) / 7.182 centroid, Strehl 0.9973 —
11 % of the error budget, so what stage 2 then measures is attributable
to the fold and the bias rather than to a sloppy start.

Three gates run **before any wavefront number is believed**, ordered by
what a failure would invalidate:

1. **The conic solver**, against the shared `tma_fixture.json` — engine-free
   arithmetic.  `max|ΔK| = 4.6e-08` against a 1e-06 bar.  A miss here is
   stop-and-fix; never widen the bar, never hand-edit the fixture.
2. **The first-order layout**, against the scaled reference geometry.
   Derived R3 = 1.612788 m vs 1.612784 m — **2.38e-06 relative**.
3. **The pupil**, against the declared `Aperture=`: greatest chord and
   outermost radius, both 0.997696 ×, zero rays outside.

Free cross-check, reported but deliberately **not** gated: conic
constants are dimensionless, so they do not scale.  A correct solve must
land on the reference stage-1 conics without ever having been given them.
It does — `|ΔK| = [2.9e-06, 4.4e-05, 5.7e-05]` — which validates the whole
chain (layout → builder → CALIB) at once.

## How wide a field does this hold?

Stage 1 at the original 0.2° box landed at **0.638 nm** — 56× inside the
diffraction bar.  That headroom is an invitation to ask for more field,
and `s1_fov_sweep.m` answers it by measurement.  Two curves, and the
difference between them is the point:

| half-field | as-solved (0.2° design) | re-solved at that box |
|---|---|---|
| 0.10° | 0.638 nm | 0.638 nm |
| 0.15° | 4.241 | 1.728 |
| 0.20° | 12.849 | 4.087 |
| 0.25° | 29.448 | 8.616 |
| 0.30° | 57.913 | **16.457** |

(max RMS over the box, +LS tip/tilt rung, 500 nm.)  **0.20° is the
adopted point**; 0.30° was adopted first and then narrowed.

**Re-solving is worth 3.5× at 0.30°** — the conics rebalance, and a design
merely *re-scored* over a wider box understates what the architecture can
do by that factor.  The re-solved residual grows as **θ^2.96**; a pure
Petzval/astigmatism wall would be θ², so the extra power is the
higher-order field aberration three conics cannot reach.

**The full 0.6° box is achievable**, and it is worth saying at which
convention:

| rung | max RMS at ±0.3° | min Strehl | vs the 35.7 nm bar |
|---|---|---|---|
| chief | 37.25 nm | 0.803 | **4 % over** |
| centroid *(primary)* | 27.74 | 0.885 | 22 % margin |
| + best focus | 23.49 | 0.916 | 34 % margin |
| + LS tip/tilt | 16.46 | 0.958 | 54 % margin |

So: **passes at the primary reference and everything more permissive,
marginal at the strictest.**  That is a statement about the convention,
not a design deficiency — but it is the reason 0.25° half-field is kept
in `e2e2_params.m` as the conservative alternative, where every rung
including chief passes with room (20.5 nm at chief, Strehl 0.936).

**What a wide field costs, stated plainly.**  At 0.2° the axial anchor was
1.8 % of the error budget; at 0.6° it is 46 %.  Added in quadrature, that
leaves 31.7 nm for stages 2–4 rather than 35.7 — a 9 % reduction in the
remaining budget for a 3× wider field, which is a good trade, but the
off-axis stage now starts with much less room than it did.

### S2 — the fold, and the clearance frontier

The coaxial design's detector sits in its own beam.  Two knobs move it
out — a fold with an M3 extraction tilt, and a field bias — and **both
cost wavefront**: the tilt as ~tilt², the bias as ~bias^1.80 (measured).
So clearance is swept on **geometry alone** (cheap), the
**Pareto-minimal** clearing combinations are taken, and solves are spent
only on those.  Two survive:

| | max RMS |
|---|---|
| **tilt 0° + bias 13′** | **20.4 nm** |
| tilt 2.50° + bias 0′ | 2600 nm |

**Bias beats extraction tilt by two orders of magnitude here** — the
reverse of the e2e VIS telescope, where the tilt won.  2.5° on a
*powered* f/20 M3 is ruinous astigmatism; 13′ of bias is cheap.  The e2e
lesson does not transfer between architectures, and only pricing the two
together shows it.

Clearance passes on M2's accepted central obscuration alone; AOI spread
13.90° against the 15° bar.

## The relay — parked, and why

The telescope reaches the target **without a relay**, so the relay's job
is to feed instruments rather than rescue the wavefront, and it should be
designed to its own requirements.  Two attempts failed the same way — a
relay scaled from another design instead of laid out for this one — and
the second did it inside the branch adopted to avoid the first.  Full
record, including the still-**untested** field-corrector hypothesis, in
`relay_followon/README.md`.

The one transferable number: **a relay is sized by the IMAGE it accepts,
not by the aperture.**  image half-height = EFL·tan(half-field) — e2e
0.042 m, e2e2 0.209 m at the adopted box (0.314 m at 0.6°).

## The design procedure — why the solves are staged this way

Each rule below was established by a **failed run**, and the failure is
recorded with it.  The runners implement all of them, so a parameter
change should just work; if you restructure the chain, keep the rules.

1. **Never fit a detector on an unsolved design.**
   `align_focal_plane` fits each field's best focus *from rays*.  Called
   on the K = 0 spherical seed — where this design carries **15 mm RMS**
   of spherical aberration — it locks onto the spherical caustic
   **1.796 m** from the paraxial focus, reports an **11 mm** best-focus
   blur, and moves the detector there.
   *Failed alternative:* the first version of `s1_axial.m` did exactly
   that.  The design still converged — the FPA piston in the joint DOF
   set spent itself undoing the excursion — so nothing looked wrong in
   the WFE.  The damage was silent and downstream: `add_pupil` had
   derived `FP_return` and the ExitPupil sphere from the abandoned
   station, so the **saved deck declared an exit pupil belonging to a
   design that no longer existed**.

2. **And do not fit one on a rotationally symmetric design at all.**
   Solving the conics first does not rescue the fit here.  On a coaxial
   on-axis telescope the per-field foci form a rotationally symmetric
   **disc**: the plane fit's two in-plane singular values are *equal*, its
   basis is arbitrary, and it returned a normal **90° from the arriving
   chief** — a detector edge-on to its own beam.  There is nothing for it
   to find.  Symmetry fixes the normal along the axis; the builder's
   paraxial `derive` places the station (correct to **22 µm** here,
   measured by how far the joint solve then moved it); the FPA piston
   refines what is left.  The fit earns its keep in stage 2, where the
   field bias breaks the symmetry and the focal surface genuinely tilts.

3. **Solve AT the target field, never merely re-score.**  The conic basin
   is path-dependent, and the sweep above measures the cost of getting it
   wrong: 57.9 nm re-scored vs 16.5 nm re-solved at ±0.3°, a factor of
   3.5 attributed to the optimizer's starting field rather than to the
   optics.

4. **Joint solve, never alternate.**  Conics and the detector's tip and
   focus go into ONE CALIB DOF set.  The merit's reference sphere is
   centred on each field's chief-ray intercept *on the detector*, so
   solve-then-refit-FPA chases two objectives and need not contract —
   measured drifting 0.6–13 mm per round on the rodgers1 TMA.  What S1
   runs is a **seeding sequence**, not an alternation: one on-axis conic
   solve to leave the spherical regime, then one joint solve.

5. **Quote the traced first order, never the paraxial seeder.**  The
   seeder only *places* the focal plane before the solve refines it, and
   it is 96 % wrong on the 100 m Rodgers deck.  Here paraxial and traced
   EFL agree (60.000 vs 59.999 m) — which is a *result*, not a licence to
   quote the seeder elsewhere.

6. **Name the rung every number came from.**  All four reference rungs
   (chief / centroid / +best focus / +LS tip/tilt) are tabled every time;
   the verdict rung only picks which one the pass/fail reads.  On comatic
   fields the rungs spread by 1.3–1.7×, and per-field focus is worth < 3 %
   of that — the tilt treatment is the whole game.

7. **Score on a uniform grid, never on the solve set.**  Statistics come
   from a uniform 9 × 9 over the used box; the solve set is reported
   alongside only to show what the optimizer saw.  An edge-weighted solve
   sampling biases the average ~8 % at an identical max.

8. **Gate the pupil after every source or aperture edit**, on the
   *greatest chord* — never a span.  The ColSource pupil was 5 % oversize
   for decades and read *exactly right* along both axes; only the diagonal
   exposed it (macos PR #70, PACKET Addendum 10).  The engine is correct
   now; the gate stays because the failure mode is silent and the cost is
   one trace.

9. **Frame before angle.**  Every reported tilt names its reference.  The
   focal plane gets both angles it has: the **beam AOI** (arriving chief
   vs the detector normal — the FPA acceptance angle, the assembly driver)
   and the **mechanical tilt** (detector normal vs the optical axis — the
   mount requirement).  A "14.3° tilted detector" that turned out to be
   the same surface as CODE V's −0.07° is what earned this one.

## Found while building this example

- **The best-focus rung was losing to the rung below it.**
  `strict_rungs` slid the reference sphere with `fminbnd` at its default
  `TolX` — 0.1 mm of focus slide on a metre deck, ≈ 31 nm of wavefront at
  f/20.  On a design corrected to the nm level the search cannot resolve
  its own minimum: rung 3 returned **2.386 nm where rung 2 read 0.890 nm**.
  On the rodgers1 deck, ~100× more aberrated, the same defect is 1.7e-4
  relative — invisible exactly where the metric was developed, ruinous
  exactly where it is now used.  Fixed (relative tolerance + an explicit
  `ff(0)` evaluation, so the ordering is true by construction); the
  rodgers1 rung-3/rung-4 artifacts are stale by 2–3e-4 and flagged for
  regeneration as a separate reviewed step.

## Handoff

The `.in` that stage 5 emits feeds the **existing** pipeline unchanged —
`run_segmentation` → `run_sensitivities` → `run_met` → `run_compare` →
`run_simulator` (`design/runners/`, worked end-to-end in `../e2e/`).
Segmentation and everything after it are deliberately **not** duplicated
here.

## House rules

Figures and reports land in this directory.  No `exit(0)` inside these
scripts — batch wrappers supply it.  Run the stages in order with
`run('.../sN_*.m')` after `mmacos_setup`.
