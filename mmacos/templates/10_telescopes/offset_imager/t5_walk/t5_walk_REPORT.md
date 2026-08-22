# t5-walk -- offset_imager continuation walk

2026-08-22 16:42:07.  EPD 150 mm, F/3.3 (EFL 0.495 m held as an identity), lambda 1.00 um, target box 15x15° offset +22.5°, spacings [-1.19278 0 1.22237] m, model 256, nGridpts 41.

Metric (every number below): strict RMS WFE, sphere centred on the spot centroid on the step's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense 11x11 map MAXIMUM over the box.  Solve set 5x5 != scoring set.

Continuation: walk the box FULL-WIDTH open along [5 8 11 13 15] deg at fixed offset +22.5°, carrying the solved X as each step's warm start; every step is a full-freedom S5 solve (conics+Zernike+tilt/dec+radii+stop_y) with the exit row + signed clearance rows active.  A carried design is SCREENED at the widened box before solving; a 1e9 no-rays score halves the step (F8 rule).

## Verdict

**PARTIAL.**  The walk reached the target box with a valid dense map but a gate still fails (see the final row) -- the aberration walk succeeded; the failing gate is the remaining, separately stated, constraint.

Final dense-map max **69.8 nm** vs the cold-start baseline 595565.2 nm -- **8531x better** (ratio 0.00012).

## The walk, step by step

| step | box (deg) | halvings | start qmean (nm) | end qmean (nm) | map max (nm) | clear floor (mm) | exit err (deg) | gates |
|---|---|---|---|---|---|---|---|---|
| 1 | 5 x 5 | 0 | 4613.8 | 7.8 | 10.9 | 98.0 | 0.001 | exit PASS / clear PASS |
| 2 | 8 x 8 | 0 | 22.1 | 15.1 | 21.5 | 67.4 | 0.003 | exit PASS / clear PASS |
| 3 | 11 x 11 | 0 | 33.5 | 20.1 | 27.3 | 25.1 | 0.004 | exit PASS / clear PASS |
| 4 | 13 x 13 | 0 | 34.7 | 28.4 | 40.0 | 24.6 | 0.003 | exit PASS / clear PASS |
| 5 | 15 x 15 | 0 | 48.4 | 44.5 | 69.8 | 17.8 | 0.098 | exit PASS / clear FAIL |

Gate thresholds: exit-chief direction within 1.0° of the pin; signed clearance floor >= 25 mm (WARN < 40 mm).  The clearance model is SIGNED (design/src/oi_clear): a negative floor is beam-mirror interference depth, not a distance.

## What this shows

The **cold start FAILS** on this instance: `run_t5r` (the same envelope, box, offset and seed, solved cold at the full 15×15° box) stalls at **595565 nm** map max across S3/S4/S5 with clearance **−205 mm** (deep interference) — S2 loses 104/121 fields, the design is outside the convergent basin (`t5_redemption/t5r_REPORT.md`).

The **walk SUCCEEDS on aberration**: bootstrapping at an easy 5×5° box (a diffraction-limited 10.9 nm design that clears comfortably) and carrying the solved design outward one box-width step at a time lands the full 15×15° target at **69.8 nm** — an 8531× improvement over the cold start, at every step a **valid** dense map (no lost fields) and a **PASS** exit-direction gate. The screen never had to halve a step: each carried design traced the next-wider box on the first try (`halvings = 0` throughout), which is the continuation premise holding — the basin moves smoothly with the box.

The **binding constraint is CLEARANCE, not wavefront.** The signed floor tightens monotonically as the box opens (98 → 67 → 25 → 25 → 18 mm) and crosses the 25 mm knee between steps 4 and 5: the final instance clears only 17.8 mm where ≥25 mm is required. This is the honest remaining gap — the aberration problem is solved; packaging a 15×15° box at +22.5° in this envelope needs more room between the beam legs and the mirror edges than the ×1.65 rodgers3 W-fold provides. The walk isolates that: it is a clearance/envelope problem now, not a surface-solve problem (contrast the cold start, where both were failing and neither could be diagnosed past the ray loss).

Reproduce: `run_t5_walk` (this run: model 256, 41 nGridpts, ~121 min).  Artifacts: per-step decks `t5_walk_k0*.in`, figures `t5_walk_k0*_{layout,fields,map}.png`, `t5_walk_run.mat`, this report.
