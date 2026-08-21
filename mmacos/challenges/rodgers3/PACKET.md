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
