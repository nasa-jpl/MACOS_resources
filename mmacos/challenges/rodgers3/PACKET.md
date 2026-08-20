# PACKET — rodgers3: the offset-field imager challenge

**Status: DRAFT — T3 numbers land when the template run completes; every
XX below is a placeholder.  Outward-facing use requires Dave's sign-off
(standing rule).**

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

| stage | freedom | our map max (11×11) | his rung | his nm |
|---|---|---|---|---|
| S1 | conics+aspheres at the on-axis box | XX | r1 | 159 |
| S2 | offset box, FPA refit only | XX | r2 | 8810 |
| S3 | re-solved at the offset | XX | r3 | 168 |
| S4 | + tilt/dec (+ radii) | XX | r4 | 117 |
| S5 | + Zernike (his term set) | XX | r5 | 53 |

### Attribution of every difference

1. **EFL.** His decks measure EFL 371 mm (paraxial chain on his radii/
   spacings, validated by real rays) = **F/4.95 at EPD 75 mm**, while
   the slide says F/4.  The template holds EFL = EPD·F# = 300 mm as an
   identity, so our system is 19% shorter — plate scale 87 vs 108
   µm/arcmin.  Aberration at fixed EPD and field generally FAVOURS the
   longer system; the comparison carries this scale difference in his
   favour... (finalized with the T3 numbers).
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
no aspheres, straight to the S5 Zernike solve.  Result: XX nm map max
vs the heritage path's XX.  Verdict: XX.

**(b) Is his 53 nm term-set-limited?**  He froze thicknesses (piston as
the surrogate) and held power to the radii.  Releasing power (mode 5)
and y-tilt (mode 3) into the S5 basis: XX nm vs XX.  Verdict: XX.

## 4. Reproduction instructions

```matlab
run mmacos_setup.m
rodgers3();          % the Stage-0 five-rung gate ladder (11x11 maps)
run_t3();            % the template at his parameters + counter-designs
```
Suite: `./run_mmacos_tests.sh freeform` carries tRodgers3 (coarsened
5×5 maps + the r5 C-offset negative control) and tOffsetImager (the T4
second-parameter smoke).
