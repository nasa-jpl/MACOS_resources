# dyson5 round 4 — the three-mirror anastigmat by the design layer

Written for: Dave (round 4 of `macos/BRIEF_ccmac_dyson_size.md`; the telescope side
of Jim's comparison). CCMac, 2026-10-03, `dev-candidate`. Different tooling from TO's:
the design layer (`macos.design.tma_layout` + `Telescope`, the `tma_onaxis` idiom) and
the native optimizer (CALIB), solving the biased configuration from the start. Driver
`dyson5_tma_step1.m`; records `dyson5_tma_step1.{txt,mat}` + `_layout.png`. Engine
rebuilt to `e43b126` (gates re-checked: tSpectrometerRx 7, tGratingImmersed 4, tAsphCalib 2).

## Step 1 — the coaxial TMA parent (f=330 mm, F/1.8, D=183 mm)

**First, a first-order correction (beat5's seed caution, confirmed).** The convex-
secondary Korsch's `seidel_seed` paraxial EFL is unreliable: `tma_layout` asked for
EFL 330 mm but the K=0 seed built from it traces to **385 mm (F/2.1)**. I calibrate the
layout — iterate the requested system f/# until the **exact-traced** EFL = 330 mm — giving
the on-record parent at **EFL 334 mm, F/1.82 at D=183 mm** (R=[0.367 0.100 0.074] m,
t=[0.148 0.258] m, primary f/1.0, secondary mag 3.5). Always check the seed EFL by trace.

**The optimizer works, and the result is a wall.** CALIB (conic-only, so the EFL stays
330 mm — freeing ROC would drift f/#) parabolizes the mirrors and images beautifully on
axis, then degrades fast off axis:

| field off-axis | K=[M1 M2 M3] | field-centre spot | WFE |
|---|---|---|---|
| on-axis | [−0.80 −1.15 −0.45] | **0.01 µm (DL)** | 0.008 waves |
| 1.0° | [−0.62 1.18 −0.54] | 7.4 µm (0.41 px) | 4.2 waves |
| 2.0° | collapses | 11 mm | — |
| ≥3° | spheres (CALIB loses rays) | 12 mm | — |

**Spot per field across the 9.4° strip (±4.7° cross-track), best focus, per bias (µm):**

| bias | −4.7° | −3.1° | −1.6° | 0° | +1.6° | +3.1° | +4.7° | imaged |
|---|---|---|---|---|---|---|---|---|
| **1°** | 2905 | 3223 | **84** | **7.5** | **84** | 3223 | 2905 | 7/7 |
| 2° | 6665 | 41921 | 21321 | 9591 | 21321 | 41921 | 6665 | 7/7 (all huge) |
| 3° | lost | 10643 | 51520 | 23722 | 51520 | 10643 | lost | 5/7 |
| 4° | lost | 8522 | 11452 | 12244 | 11452 | 8522 | lost | 5/7 |
| 5° | lost | lost | 2224 | 3596 | 2224 | lost | lost | 3/7 |
| 6° | CALIB `calib_run` failed (LM failure path — caught) | | | | | | | 0/7 |

**The finding.** The coaxial Korsch conic parent does **not** image the 9.4° strip at F/1.8
at any bias. The best bias is the smallest (**1°**, 7/7 fields reached): the strip *centre*
is sharp — **7.5 µm = 0.41 px** — but the imageable cross-track half-field is only
**~±1.5°** (84 µm at ±1.6°, then 3 mm = 180 px by ±3°). So the parent covers roughly
**one third of the swath** (±1.5° of the needed ±4.7°). And the bias cannot be raised to
unobscure the coaxial form: by 2° the conics are already failing, and at 3–6° CALIB loses
rays in its trial steps and returns spheres (12 mm) or fails outright. Three conics cannot
correct a 9.4°-wide field at F/1.8 — exactly the premise of round 4.

This is the obscured-baseline wall. The way out is **step 2** (the unobscured eccentric-
pupil section, `set_offaxis` / `tma_offaxis`, which removes the obscuration so the bias is
no longer needed to clear the FP) and **step 4** (freeform / aspheres, more degrees of
freedom than three conics). Both are next.

**Geometry (bias 1°, the record parent):** length 258 mm (≈0.78 EFL — M&G Fig 6's "roughly
equal to the focal length" class); M1 183 mm, M2 59 mm, M3 221 mm; mirror mass **1.69 kg** at
a stated **25 kg/m² areal density** (lightweighted). Layout rendered in `dyson5_tma_step1_layout.png`
(a correct Korsch: concave M1, convex M2 at the front, intermediate focus, M3 behind M1 to
the biased FP). The M3 at 221 mm is oversized by the strip's cross-track footprint — the
unobscured section will cut it down.

**Two engine/veneer notes for CC (Linux's lane).**
1. `macos.stop(2)` (stop on M2) is rejected by the engine's `stop_info_set` on a
   `Telescope`-emitted deck (fresh or optimized) — "stop_info_set failed"; `telescope_score`
   succeeds only on a `spectrometer_rx`-style deck. Step 1 scores with the emitted entrance
   stop (same D beam → same parent spot); the M2 / exit-pupil stop is step 3's and may need a
   veneer fix. Not an engine correctness bug, a Telescope-emit vs stop_info_set interaction.
2. `optimize('fields_arcmin', [])` is a silent no-op (leaves spheres); a non-empty field set
   is required for CALIB to engage. (The explorer's "[] = bias field only" is not what the
   veneer does.) Worth a one-line guard/doc in `Telescope.optimize`.

Next: step 2 (unobscured section + clearance against the Dyson bodies), step 3
(telecentricity/flatness/exit-pupil), step 4 (freeform if conics stall), step 5 (end-to-end
with each module's Dyson). Each a committed, pushed record while Dave is away.
