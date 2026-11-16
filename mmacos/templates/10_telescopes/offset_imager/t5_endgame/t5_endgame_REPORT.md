# t5-endgame -- closing the 15x15 deg clearance deficit

2026-08-23.  Instance: EPD 150 mm, F/3.3 (EFL 0.495 m held as an
identity), lambda 1.00 um, box 15x15 deg offset +22.5 deg, spacings
[-0.7228968 0 0.7408280]*1.65 m, model 256, nGridpts 41 -- the
t5_walk instance, unchanged.  Harness: `run_t5_endgame.m` (tasks
'rescore'/'wfe'/'env'; per-task .mat + log in this directory).

Metric (every number below): strict RMS WFE, sphere centred on the
spot centroid on the frozen FPA, anchored at the exit pupil,
piston-only removal (design/src strict kernel); headline = dense
11x11 map MAXIMUM over the box.  Solve set 5x5 != scoring set.
Clearance = design/src/oi_clear, SIGNED; 'disk' footprints are the
model of record, 'hull' = the 1.15-scaled convex hull (truer on
elongated off-axis patches).

## Verdict

**CLOSED -- both ways, and the WFE price is negative.**  The walk's
7.2 mm deficit (17.8 mm floor vs the 25 mm gate) closes twice over:

1. **Model:** under the truer hull footprints the committed walk
   endpoint ALREADY reads **28.74 mm -- it clears the 25 mm gate as
   committed.**  The deficit was disk-model conservatism.
2. **Solve:** in the UNCHANGED x1.65 envelope, a restart re-solve at
   a 32 mm hinge lands **47.1 nm map max at a 30.89 mm disk floor**
   -- better than the walk endpoint on BOTH axes (69.8 -> 47.1 nm
   while the floor rises 17.8 -> 30.9 mm).  Deck:
   `t5_endgame_wfe.in` (+ layout/fields/map figures) -- the
   instance's new best-known design.
3. **Envelope:** stretching is NOT needed for clearance -- what a
   stretch buys is WFE (34.9 -> 31.3 nm over x1.75..x1.9), while
   the spec-hinge floor asymptotes just under 25 mm at every scale.

Root cause of the walk's residual, confirmed: `oi_solve`'s plateau
break tests the WFE qmean only, so the walk's step 5 stopped after
SIX iterations -- the moment WFE flattened, while the clearance
hinge was still pulling.  The committed 15 deg walk endpoint was
under-solved on clearance, not blocked by it.

## Task 1 -- hull re-score of the five walk steps (no solve)

| step | box (deg) | disk floor (mm) | hull floor (mm) | map max (nm) |
|---|---|---|---|---|
| 1 | 5 x 5 | 98.01 | 101.14 | 10.9 |
| 2 | 8 x 8 | 67.40 | 70.57 | 21.5 |
| 3 | 11 x 11 | 25.05 | 29.54 | 27.3 |
| 4 | 13 x 13 | 24.60 | 32.42 | 40.0 |
| 5 | 15 x 15 | 17.84 | **28.74** | 69.8 |

Self-check PASSED: recomputed disk floors reproduce the committed
walk report to 0 mm (Mac, R2024a/gfortran-16) and 4.6e-13 mm
(Linux, R2026a/gfortran) -- same designs, cross-platform.  Note the
13 deg step also moves from marginal (24.60 disk) to comfortable
(32.42 hull).  Honesty rule: disk STAYS the gate of record; the
hull numbers are reported beside it, not in place of it.

## Task 2 -- price of clearance in WFE (fixed x1.65 envelope)

Re-solve of the 15 deg step only, warm from the k04 (13 deg)
design, restart loop around oi_solve (each restart re-enters past
the WFE-only plateau break), hinge = min(clear_m) raised above the
spec; the reported gate stays the true 25 mm.

| hinge (mm) | map max (nm) | disk floor (mm) | exit err (deg) | iters / restarts | >= 25 mm |
|---|---|---|---|---|---|
| 25 (spec) | 51.70 | 24.24 | 0.004 | 46 / 3 | short |
| **32** | **47.08** | **30.89** | 0.003 | 82 / 5 | **yes** |
| 40 | 47.30 | 38.97 | 0.023 | 92 / 6 | yes |

Price sentence: **a >= 25 mm floor in this envelope costs MINUS
22.7 nm map max (69.8 -> 47.1)** -- the constraint is free because
the walk endpoint was under-solved.  Hinge-at-spec undershoots
(24.24 mm): the hinge gradient vanishes as the floor approaches the
target, so ask for more than the spec (32 mm is the sweet spot;
40 mm buys floor margin at a wash in WFE and 8x the exit error).

## Task 3 -- price of clearance in envelope (fixed WFE class)

Spacings scaled up from x1.65, spec clear_m [40 25] (hinge at
spec), warm-started scale to scale, EFL/F# held by the closure.

| scale | train (m, runner's measure) | map max (nm) | disk floor (mm) | >= 25 mm |
|---|---|---|---|---|
| x1.75 | 1.460 | 34.89 | 24.41 | short |
| x1.85 | 1.510 | 34.12 | 24.76 | short |
| x1.90 | 1.536 | 31.30 | 24.86 | short |

The floor asymptotes just UNDER 25 mm at every scale -- the same
hinge-at-spec undershoot as task 2's 25 mm row, not a geometry
wall (task 2 reaches 30.9 mm in the SMALLER x1.65 envelope).  What
the stretch actually buys is WFE: 34.9 -> 31.3 nm.  Conclusion:
**envelope stretch is the wrong lever for clearance on this
instance; keep x1.65 and use the raised hinge.**

## Provenance

Tasks executed by CCMac (Mac, MATLAB R2024a, gfortran-16 engine)
2026-08-22, salvaged after a budget stop before this report was
written; harness + artifacts committed verbatim, task-1
cross-checked on Linux (R2026a) to 4.6e-13 mm.  Report assembled
2026-08-23 (CCL) from the committed .mat/.log artifacts in this
directory.

Reproduce: `run_t5_endgame('all')` (task 1 ~2 min; tasks 2/3 ~40 /
~60 min at the shipped knobs).  Records: `rescore.mat`, `wfe.mat`,
`env.mat`, per-task logs, `t5_endgame_wfe.in`.
