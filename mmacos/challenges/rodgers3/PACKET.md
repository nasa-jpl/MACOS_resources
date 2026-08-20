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

| stage | freedom | our map max (11×11) | his rung | his nm | ours/his |
|---|---|---|---|---|---|
| S1 | conics+aspheres at the on-axis box | 37.8 | r1 | 159 | 0.24 |
| S2 | offset box, FPA refit only | 303586 | r2 | 8810 | 34.5 |
| S3 | re-solved at the offset | 252.0 | r3 | 168 | 1.50 |
| S4 | + tilt/dec (+ radii) | **58.3** | r4 | 117 | **0.50** |
| S5 | + Zernike (his term set) | 78.4 | r5 | 53 | 1.48 |

Exit chief within 0.03° of horizontal at every stage (the pin).  Three
honest notes.  (1) Our S2 "disaster" is 34× his: our S1 solves 4×
deeper on-axis than his r1, and harder-tuned on-axis aspheres cost
proportionally more at the offset — the S2 rung measures the S1 design
as much as the offset.  (2) Our S5 lands ABOVE our S4 (78.4 vs 58.3):
82 variables against a 3×3 solve grid, still descending at the
30-iteration cap — quoted as-solved, not min-taken; the
more-DOFs-never-worse rule would report min(S4,S5) = 58.3 per branch.
(3) CLEARANCES ARE NOT COMPARABLE YET, and this cuts against our S4/S5
numbers.  The gate reads 0.0 mm at every stage.  Part is model
crudeness (a circular union-footprint disk over-covers the real
patches; the S4 layout figure shows M1's patch visibly separated from
the M2→M3 beam by the ~50 mm class his slide claims), but part is
REAL: in the S4 layout the M3→FP return beam passes through/near M2's
patch — our solve never paid the clearance constraint his optimizer
enforced, so our 58.3 nm at S4 is bought with packaging his 117 nm
respects.  Template follow-up before any head-to-head clearance claim:
convex-hull footprints + clearance as a solve wall.

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

**(a) Sphere+Zernike from the start** (sz doctrine): S3's radii, K = 0,
no aspheres, straight to the S5 Zernike solve.  Result: **27.9 nm map
max** vs the heritage path's 78.4 — and 1.9× BELOW his 53 nm rung,
under his own term set and exit constraint.  Verdict: the asphere
heritage is the burden, not the help — the sphere+Zernike start
converges deeper in the same 30-iteration budget (20.8 µm → 3.1 nm
qmean vs the heritage path's stall at 18.8).  This is the sz_tma
doctrine reproducing on his own problem.  Caveat: both paths are
iteration-capped, so the 78.4-vs-27.9 split partly measures
convergence, not just the reachable floor; the 27.9 stands on its own
against his 53 either way.

**(b) Is his 53 nm term-set-limited?**  He froze thicknesses (piston as
the surrogate) and held power to the radii.  Releasing power (mode 5)
and y-tilt (mode 3) into the S5 basis from the S4 design: 78.1 nm vs
the same-start 78.4 — a wash.  Verdict: NO evidence his 53 nm is
term-set-limited; the released modes buy nothing the pinned quantities
(radii, pointing) had not already provided.  Same iteration-cap caveat
as (a).

## 4. Reproduction instructions

```matlab
run mmacos_setup.m
rodgers3();          % the Stage-0 five-rung gate ladder (11x11 maps)
run_t3();            % the template at his parameters + counter-designs
```
Suite: `./run_mmacos_tests.sh freeform` carries tRodgers3 (coarsened
5×5 maps + the r5 C-offset negative control) and tOffsetImager (the T4
second-parameter smoke).
