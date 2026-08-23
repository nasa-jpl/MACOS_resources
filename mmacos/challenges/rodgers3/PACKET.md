# PACKET — rodgers3: the offset-field imager challenge

**Status: numbers complete (2026-08-20 run).  Outward-facing use
requires Dave's sign-off (standing rule).**

Mike Rodgers, `260802-WFOVimager_Offsetfield-jmr.pptx` (referenced, not
committed) + five CODE V `.seq` decks (committed here): 3-mirror
imager, EPD 75 mm, F/4, λ 1 µm, 20°×20° field box offset 22°; exit beam
horizontal (r2+); clearances > 50 mm and > 35 mm (r4+); 9-point
optimization field, XAN ±10°.

## The metric, stated before any number

Every WFE value in this packet is the **strict RMS WFE** (design/src
kernel): reference sphere centred on the **spot centroid** on the
stage's frozen focal plane, anchored at the exit pupil, piston-only
removal, fields tangent-composed, chief aimed through the stop centre
by real-ray iteration.  **His "RMS WFE ≤ x" is the MAXIMUM of CODE V's
dense RMS-vs-field map over the full box** (decoded from his slides'
own EMF metadata — `r3_s0_report.txt`), so the comparable statistic is
our dense-map maximum (11×11 unless stated).  Field sampling is stated
with every number.

## 1. His ladder | our verbatim reproduction (Stage 0)

| rung | design state | his nm | our map max (11×11) | ratio | verdict |
|---|---|---|---|---|---|
| r1 | coaxial, on-axis box, aspheres | 159 | 157.74 | 0.992 | PASS |
| r2 | same mirrors, offset box, FPA refit | 8810 | 9664.11 | 1.097 | PASS |
| r3 | re-asphered at the offset | 168 | 166.32 | 0.990 | PASS |
| r4 | + tilt/dec + radii | 117 | 124.29 | 1.062 | PASS |
| r5 | + 8th-order ZRN + radii | 53 | 54.66 | 1.031 | PASS |

Band rule [0.8, 1.25]×; nothing tuned toward his numbers.  Known,
bounded approximation: our uniform perpendicular-to-chief pupil bundle
vs CODE V's stop-gridded rays inflates the r2/r4 map slightly and the
map MINIMA visibly (full caveat list in `r3_s0_report.txt`).

## 2. OUR template's ladder at his parameters (T3)

`run_t3.m` → `t3/r3t_REPORT.md`.  Same EPD/F#/box/offset/λ, his
packaging envelope (spacings verbatim — designer inputs), his
constraint set pinned: exit chief along [0 0 −1] (measured from his own
decks: r3/r4/r5 hold it to ≤ 2e-5 rad), clearances 50/35 mm.

| stage | freedom | our map max (11×11) | clearance (min pair) | his rung | his nm | ours/his |
|---|---|---|---|---|---|---|
| S1 | conics+aspheres at the on-axis box | 37.8 | 3.4 mm (n/a) | r1 | 159 | 0.24 |
| S2 | offset box, FPA refit only | 303586 | 27.0 mm (n/a) | r2 | 8810 | 34.5 |
| S3 | re-solved at the offset | 252.0 | 13.2 mm (n/a) | r3 | 168 | 1.50 |
| S4 | + tilt/dec (+ radii), clearances enforced | **113.6** | **34.1 mm PASS** | r4 | 117 | **0.97** |
| S5 | + Zernike (his term set), clearances enforced | 118.2 | 34.6 mm PASS | r5 | 53 | 2.23 |

Exit chief within 0.03° of horizontal at every stage (the pin).
Clearances: measured with the OI_CLEAR model (nine leg/obstacle pairs,
per-field footprint disks ×1.15 over the box centre + YAN extremes,
the FP counted as an obstacle) — the SAME model the S4/S5 solves pay
via hinge residual rows.  His constraint binds from r4 on, and so does
ours: S1–S3 report their (unconstrained) floors for context.

**The S4 story is the headline of this table.**  Unconstrained, our S4
reached 58.3 nm — with the M3→FP beam parked in M2's patch (min pair
0.0 mm; Dave's read of the field-envelope figure, confirmed by the
model).  Enforcing his ≥35 mm moves it to **113.6 nm at 34.1 mm — his
r4 is 117 nm at his stated ≥35 mm**.  Paying the same constraint lands
the same number to 3%: strong evidence the two toolchains are
measuring the same design space, and a clean exhibit of what the
clearance constraint costs (≈2× in WFE at this rung).

Honest notes.  (1) Our S2 "disaster" is 34× his: our S1 solves 4×
deeper on-axis than his r1, and harder-tuned on-axis aspheres cost
proportionally more at the offset — the S2 rung measures the S1 design
as much as the offset.  (2) Our S5 (118.2) sits at our S4 (113.6), far
from his 53: 82 Zernike variables against a 3×3 solve grid at a
30-iteration cap, now also carrying the clearance rows — the stage is
convergence-limited, not physics-limited (the min-rule branch value is
113.6).  Closing that gap is solver budget, not convention.

### Attribution of every difference

1. **Focal ratio: slide CONFIRMED by the deck.**  The r1 .seq measures
   EFL 300.003 mm = F/4.00004 at EPD 75 (paraxial chain on the deck
   radii/thicknesses, computed by run_t3 from `rodgers3_seq` at
   runtime).  Recorded here because a station-vs-spacing transcription
   slip briefly suggested F/4.95 — the .seq THICKNESSES are
   th4 = −722.9 / th6 = +740.8 mm while the m1/stop/m3 STATIONS are
   +665/−58/+683 mm; feeding stations into the paraxial chain gives the
   wrong EFL and makes the exit-horizontal constraint look infeasible.
   run_t3 reads packaging and focal ratio from the truth file so the
   comparison cannot drift this way again.
2. **Solve machinery.** CODE V damped least squares with his error
   function vs our damped Gauss–Newton on the strict per-field RMS over
   a 3×3 solve grid; both score the same statistic afterwards.
3. **Term sets.** S5 uses exactly his varied C-set mapped to BornWolf
   (piston carried; power pinned to radii; tilts to pointing).
4. **Stop pose.** Ours is the entrance-pupil construction re-derived
   per stage; his stop YDE values are close but committed per deck.
5. **Constraint interpretation.** "Exit beam horizontal" = box-centre
   exit chief ∥ [0 0 −1] (measured, above).  The two clearance values
   are applied as min-over-all-pairs ≥ 35 mm with WARN < 50 mm — his
   slide does not name the pairs.

## 3. Counter-design looks (bounded, flagged — not iterated)

Both counters run under the SAME constraint set as the main ladder
(exit pin + the nine clearance hinge rows) and the same 30-iteration
budget.

**(a) Sphere+Zernike from the start** (sz doctrine): S3's radii, K = 0,
no aspheres, straight to the S5 Zernike solve.  Result: **73.1 nm map
max** vs the heritage path's 118.2 — a 1.6× win for the sphere+Zernike
start under identical constraints, landing 1.38× of his 53.  (The
earlier UNCONSTRAINED run of this counter reached 27.9 nm — with its
beam through the glass; the clearance constraint is worth ~2.6× on
this branch too.)  Verdict: the asphere heritage is a burden, not a
help — same doctrine as sz_tma, now demonstrated on his own problem at
his own constraints — but the outright beat of his 53 does not survive
buildability at this solver budget.

**(b) Is his 53 nm term-set-limited?**  He froze thicknesses (piston as
the surrogate) and held power to the radii.  Releasing power (mode 5)
and y-tilt (mode 3) into the S5 basis from the S4 design: 1372.9 nm —
WORSE than the pinned-set 118.2 under the same budget (88 variables
against 9 solve fields plus the constraint rows; the freed modes fight
the pins' jobs and the solve chokes).  Verdict: no evidence his 53 nm
is term-set-limited, and affirmative evidence for the pinning doctrine
(power to radii, tilt to pointing) at finite solver budget.

## 4. Reproduction instructions

```matlab
run mmacos_setup.m
rodgers3();          % the Stage-0 five-rung gate ladder (11x11 maps)
run_t3();            % the template at his parameters + counter-designs
```
Suite: `./run_mmacos_tests.sh freeform` carries tRodgers3 (coarsened
5×5 maps + the r5 C-offset negative control) and tOffsetImager (the T4
second-parameter smoke).

---

## Addendum, 2026-08-21 — clearance-model corrections, the S5 budget
## answer, and the t4 retraction

**Verdict up front.**  (1) The clearance MEASURE had three defects, all
fixed and gated; under the fixed measure the S4 headline STANDS and the
S5-of-record's clearance PASS retracts.  (2) The S5 gap to the reported
53 nm was the SOLVE-FIELD COUNT, not solver effort; the honest re-solve
(25 fields, fixed clearance rows) reaches **45.4 nm at a true 33.4 mm
floor** — better than the reported 53.  (3) The t4-wide second instance
is RETIRED: its committed gates were artifacts of the blind measure.
Every number below: strict RMS WFE, centroid reference, dense 11×11 map
maximum (the packet metric, unchanged).

### A. The clearance measure, corrected (design/src/oi_clear)

Dave caught it ON THE LAYOUT FIGURE: the t4 graphic showed beams
threading the mirrors while the gate said PASS.  Three defects, each
found by that thread and fixed (macos_res commits 44dee1e, 6f70d98,
plus the leg-pairing fix):

1. **Sampled-past-piercing.**  The leg/glass distance was a
   25-fixed-sample minimum (~60 mm spacing on a 1.4 m leg); a leg that
   PIERCES an obstacle between samples reported a small positive
   clearance.  Fixed: every plane crossing is tested AT the crossing
   point (exact at any sampling) + proximity sampling at ≤ a quarter of
   the tightest requirement.
2. **Zero has no gradient.**  A pierced pair returning 0 is flat under
   small pokes, so the solve hinge rows had no slope and the optimizer
   never moved (measured: an identical re-converge on a fully blocked
   train).  Fixed: SIGNED distance — minus the deepest penetration —
   which the hinge deficit consumes unchanged.
3. **Unpaired rays.**  Leg endpoints were masked per element; the
   moment rays are lost mid-train the ends mismatch and the measure
   throws.  Fixed: each leg pairs the same ray at both ends.

Validated against an ns=2001 reference on every pair of both
instances.  Under the fixed measure, on the designs of record:

| design | recorded floor | TRUE floor | gate (≥35 − 1.5 knee) |
|---|---|---|---|
| S4 of record | 34.1 mm | **34.11 mm — CONFIRMED** | PASS |
| S5 of record | 34.6 mm | **29.97 mm** | **FAIL — retracted** |

The S4 buildability headline (113.6 nm at 34.1 mm vs the reported 117
at ≥35) is sampling-safe and stands.  The hull footprint option
(`P.clear_footprint='hull'`, same 1.15 margin) reads the S4 design at
**42.0 mm** — under the truer glass model the S4 design clears the
stated ≥35 comfortably; disk remains the model of record.

### B. The S5 budget probe (s5_budget/, run_s5_budget + run_s5_signed)

All legs seed from the S4-of-record state exactly (start reproduces
the record's 2150.1 nm).  Legs A/B solved with the PRE-FIX rows
(their true floors re-gated post-hoc); leg C with the fixed rows.

| leg | solve fields | iters | map max | true floor | note |
|---|---|---|---|---|---|
| S5 of record | 9 | 30 (cap) | 118.2 | 29.97 mm | the committed rung |
| A | 9 | 150 | 110.7 | 28.9 mm | iterations do NOT close the gap |
| B | 25 | 60 | 54.1 | 28.2 mm | fields DO — 83 vars vs 9 fields was the gap |
| C | 25 | 49 (plateau) | **45.4** | **33.4 mm** | fixed rows; exit err 0.008° |

The record's "convergence-limited" diagnosis sharpens: the stage was
SOLVE-FIELD-limited (9 fields under-determine 82 variables — the solve
set converges while the dense map stalls).  With 25 fields and honest
clearance rows the template lands at 45.4 nm — 0.86× the reported 53 —
at a floor 1.6 mm shy of the stated 35 (the rows bought back the ~5 mm
the blind measure had conceded).  The §2 table's S5 row should be read
with this addendum; the committed t3 artifacts are unchanged (instance
of record).

The template-side continuation of this solve-field lesson is
`oi_walk` (a NARROW box first, then walk `box_deg` outward carrying each
solution as the next warm start): on the t5 instance — where the cold
`run_t5r` stalls at 595565 nm / −205 mm — the walk reaches **69.8 nm at
the 15×15° target box (8531× better, exit PASS), clearance the sole
binding gate at 17.8 mm**.  See
`templates/10_telescopes/offset_imager/t5_walk/t5_walk_REPORT.md`
(driver `run_t5_walk`).

The walk's by-product is the instrument's **field-vs-packaging
frontier in one run**: 5/8/11/13/15° boxes land 10.9/21.5/27.3/40.0/
69.8 nm at signed floors 98.0/67.4/25.1/24.6/17.8 mm (every number:
the §1 metric contract; disk footprint model of record).  Read as a
frontier, the walk *closes* the t5 instance both ways: the largest
spec-compliant (≥ 25 mm) box this ×1.65 envelope carries is
**11×11° at 27.3 nm map max and a 25.1 mm floor** — a finished,
gate-PASS instrument, committed as `t5_walk_k03.in` — and the full
15×15° box stands **aberration-solved at 69.8 nm** with its 7.2 mm
clearance deficit priced separately.  Closing that deficit is the
open endgame, with three costed routes: re-score under the truer
convex-hull glass model (free if it reads ≥ 25), buy the millimetres
with WFE in the fixed envelope, or stretch the envelope at fixed
WFE class.  Quantifying the second and third is queued
(BRIEF_ccmac_endgame); the frontier statement above stands on the
committed walk record either way.

### C. t4-wide RETIRED (templates/10_telescopes/offset_imager/run_t4.m)

Under the fixed measure the committed t4 gates are false: true floors
are −107..−124 mm — the incoming corridor pierces M2, and the M3 fan
pierces M1 and M2 (exactly what the layout figure showed).  Re-solving
with honest rows does not rescue it: the envelope needs a ~65° fold of
a 100 mm leg (measured: the solve trades ~120 nm of WFE for ~2 mm of a
126 mm deficit).  A form-true rescale (rodgers3 W-fold × EFL ratio) is
still −156 mm pierced at S1, and the geometry says why: the field-walk
separation tan(offset)×leg at 12° is ~0.26 m against a ~0.32 m
beam-plus-patch need — **no envelope of this family packages a 200 mm
F/2.5 beam at a 12° offset.  Buildability constrains the field choice,
not just the surfaces.**  A 20° attempt exposed (and fixed) two more
robustness gaps — a degenerate offset stop-pose construction, now
bounded to the envelope span, and the leg-pairing defect — but
converges too slowly to gate at the smoke budget; re-instancing t4 is
an open item (run_t4.m's header carries the retraction and the
lesson).  The committed t4_wide/ artifacts stay as the historical
exhibit; their REPORT gates are superseded by this addendum.

### D. Reproduction

```matlab
run_s5_budget();   % legs A+B (pre-fix rows; ~8 h)
run_s5_signed();   % leg C (fixed rows; ~4 h)
run_t5_walk();     % the continuation walk (t5 instance; ~2 h)
```
Artifacts: `s5_budget/s5b_run.mat` + per-leg maps;
`templates/10_telescopes/offset_imager/t5_walk/` for the walk.  The
oi_clear fixes are gated by the r3t reference floors (34.11 / 29.97 mm)
reproducing exactly, and the freeform suite green.
