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

## Step 2 (Linux, CC, 2026-10-03 evening) — the unobscured eccentric section: the 1.5k strip comes within an order of the pixel, the 3k strip does not

Run on Linux while CCMac was silent (announced in `BRIEF_ccmac_dyson_size.md`,
round 4 note 17:45).  Driver `dyson5_tma_step2_linux.m`, records
`dyson5_tma_step2_linux_d170.{txt,mat}` (+ `_d190`), layouts
`dyson5_tma_step2_linux_d170_layout_{3k,1k5}.png`, decks of record
`dyson5_tma_step2_linux_d170_1k5.in` (+ `_d190_1k5.in`).  Step-1 helpers
verbatim; every number the engine's.

**Two things learned on the way, both in the driver's comments.**
1. **The seed fix is visible here.**  With the 2026-10-03 `seidel_seed`
   the layout calibration converges in ONE iteration (requested F/1.8 →
   exact-traced F/1.799, R = [0.3667 0.0998 0.0903] m); step 1 on the Mac
   needed six, down to a requested F/1.34 (R3 0.0737), because it ran the
   pre-fix seed.  Same first order either way; the "seidel_seed caution" of
   step 1 is now historical.
2. **`set_offaxis('all')` and `('M2')` cannot package a Korsch**: both ran
   the bisection to its 1.5 D bound (warning at `clearance_solve_`).  Probed
   (`probe_sec.m`): an explicit decenter clears M1 at 0.10 m and M2 at
   ~0.17–0.19 m, but **the FP is pierced by the M2→M3 beam at every
   decenter** — in a Korsch the FP and the M1 hole are concentric, so only
   the along-track BIAS separates them, and step 1 showed three conics
   break above ~2° bias.  Geometry, not the solver.  The section's as-is
   spot (seed conics, no solve) also grows 11 µm → 0.16 / 0.27 / 4.4 mm at
   d = 0 / 0.10 / 0.15 / 0.20 m: the seed is not a nulled anastigmat over a
   sub-pupil, so the conics must be re-solved ON the section.

**The ladder at d = 170 mm, biases 0/1/2°, rungs as-is / inner half-strip /
full strip (conics only, FP not enrolled).**

| module | best rung | max rms spot across the strip | EE_min | clearance | M2 / M3 | mass |
|---|---|---|---|---|---|---|
| 3k (±4.7°) | any — the strip ENDS sit at 10–12 mm whatever the solve; the centre reaches 29 µm (inner solve, bias 0) | 10.8 mm = 600 px | 0 | FAIL (M2 −1 to −12 mm) | 62 / 253 mm | 2.0 kg |
| **1.5k (±2.35°)** | **bias 2°, full-strip solve, K = [−0.846 −1.273 −0.492]** | **104 µm = 5.8 px, uniform 86–104 µm across the strip** | 0.004 | FAIL by 11 mm at M2 (6 of 7 rows: M2 is the only body short) | 51 / 135 mm | 1.07 kg |

So: three conics on the eccentric section **cannot hold the 9.4° strip**
(the step-4 premise — freeform or aspheres, and CALIB's `OptAsph=` is the
engine's own route there, exposed to the design layer only on axis so
far), but they hold the **4.7° strip to 5.8 px**, unobscured, one rung from
packaging (a ~20 mm larger decenter clears M2).  The `_d190` run (1.5k
only, d = 190 mm, biases 2/3°, strip solve, plus a rung with M2/M3 spacing
free and the EFL reported) is appended below when it lands.

**`_d190` (1.5k only, d = 190 mm, biases 2/3°).**  Record
`dyson5_tma_step2_linux_d190.txt`, deck `_d190_1k5.in`, layout
`_d190_layout_1k5.png` — a genuine unobscured section (M1 above, M2 below
the incoming beam, intermediate focus, M3, the FP beside it and clear).

| bias | rung | max rms spot | per field (µm) | M2 clearance | local plate scale (engine, ±0.02° probe about the bias) |
|---|---|---|---|---|---|
| 2° | conics | 124 µm = 6.9 px | 113 124 116 106 116 124 113 | −10.9 mm | 375 mm |
| 2° | conics + M2/M3 spacing | 136 µm | 136 124 90 64 91 125 136 | −10.5 mm | 478 mm |
| **3°** | **conics** | **118 µm = 6.6 px** | 118 118 91 69 91 118 118 | **−1.2 mm** (8 conflicts, M2 only) | **450 mm** |
| 3° | conics + spacing | 142 µm | 142 137 104 80 105 138 142 | −1.2 mm | 454 mm |

Two readings.  (a) Three conics hold the 1.5k strip at **6–7 px**, uniform,
on a section that is one small decenter (~0.20 m) from clearing M2 — the
best TMA number in the record and a real packaged form; the spacing
freedom does not help.  (b) **The local plate scale at the bias is NOT the
330 mm nominal**: the engine's ±0.02° probe about the bias field measures
375 mm at 2° and 450 mm at 3° (the eccentric section's magnification off
its parent axis), so the 54.5 µrad IFOV would be 24.5/20 m, not 30 m, and
the ±2.35° strip would overfill a 27 mm slit (±15–18 mm).  That has to be
re-centred (a layout at the right local f at the bias) before this form
is a candidate for the end-to-end join; it is a first-order correction,
not a new form.  **Not handed to TO yet** (clearance FAIL by 1.2 mm and
the plate scale).  Next (step 2b): re-calibrate the layout so the TRACED
plate scale at the working bias is 330 mm, decenter ~0.20 m, re-solve;
then step 3's telecentricity rows (CALIB `OptBeamDir=` now rides on the
WFE target) and the clearance PASS.  The 3k strip stays with step 4
(aspheres via CALIB's `OptAsph=`, or freeform).

**Step 2b, first attempt (19:25) — recorded, not landed.**  Re-calibrating
the layout on the traced plate scale of the AS-IS section (seed conics,
bias 3°, d = 0.20 m) diverges: that scale reads 530 mm and does not move
monotonically with the requested f/# (the un-solved section is too
aberrated at the probe for the plate scale to mean anything).  The 450 mm
of `_d190` is a property of the SOLVED conics (M2 at K = −1.65 changes the
local power the sub-pupil sees), so the re-centre has to put the strip
solve INSIDE the calibration loop: layout(f_req) → section → CALIB on the
strip → traced plate scale at the bias → adjust f_req, ~4–6 rounds of one
CALIB each.  Driver option `recentre_bias_deg` is in `dyson5_tma_step2_linux.m`
for the outer loop; the inner solve is not yet wired.  Left for a fresh
session (or CCMac) with this note; the `_d190` deck stands as the step-2
result.

**Step 2b, second attempt (19:55) — the coupled loop is non-monotone; the
plate scale must be constrained IN the solve.**  With the strip CALIB inside
the calibration loop (d = 0.205 m, bias 3°), lowering the requested f/#
RAISES the solved section's traced plate scale: f_req 1.80 → 1.33 → 0.72 →
0.42 gave 448 → 610 → 563 → 893 mm.  The scale at the bias is a property of
the solved conics (M2 near K = −1.6 re-powers the sub-pupil), not of the
first-order layout, so no outer loop on f_req can pin it.  **The right
tool is the engine's new per-field position rows** (macos 64c0a90,
`OptBeamPosFov=` / `macos.calib_set_beam_pos_fov`, which ride on the WFE
target): give the solve the image heights 330 mm · tan θ_k of the strip
fields as targets at the FP, weighted (`OptBeamWt=`), and the conics are
solved for blur AND plate scale together.  `Telescope.optimize` has no
hook for the beam rows yet (it configures CALIB through the api and runs
it); adding one (`'beam_pos_fov'`, `'beam_wt'`) is a veneer change of a
few lines and is the next engineering step, after which step 2b is one
run.  Left here with the `_d190` deck as the step-2 result.

**Step 2b, third attempt (20:30) — the hook is in, and it shows the trade.**
`Telescope.optimize` now takes `'beam_pos_fov'` (3 × nfov image-position
targets at the FP, CALIB field order) and `'beam_wt'`, wired to the engine's
per-field position rows (api `calib_set_beam*`, cleared after the solve so
nothing leaks); the driver's rung `strip+ps` builds the targets from the
nominal trace (bias chief's FP hit + 330 mm·tan θ along the FP's in-plane
field directions).  Records `dyson5_tma_step2b_linux_d205{,_w1}.txt`, deck
`_d205_1k5.in`, layout `_d205_layout_1k5.png`.  tDesignTelescope 72/72,
tBeamRows 3/3 after the veneer change.

| d = 205 mm, bias 3°, 1.5k strip | max rms spot | per field (µm) | clearance | plate scale | K |
|---|---|---|---|---|---|
| **conics only** (spots here are BEST-FOCUS radial rms per field, step 1's helper; on the deck's FP as placed, tilted 37° to the exit chief, TO's t5e reads 106 / 223 µm centre / edge) | **161 µm = 8.9 px** | 161 151 108 77 108 151 161 | **PASS, +0.1 mm at M2, 0 conflicts** | 447 mm | [−0.887 −1.691 −0.532] |
| conics + position rows, wt 1e-2 | 166 µm | 166 154 109 76 109 154 166 | PASS +0.1 mm | 447 mm | [−0.887 −1.697 −0.532] |
| conics + position rows, wt 1 | 739 µm | 739 503 278 136 278 503 739 | FAIL −0.5 mm | 415 mm | [−0.954 −2.277 −0.560] |

So the first **packaged, unobscured, imaging TMA** in the record: three
conics, 8.9 px worst field (4.3 px at the strip centre), every body clear,
1.05 kg of mirror — at a plate scale of 447 mm, not 330.  Asked to hold the
plate scale as well (wt 1) the same three conics give up the blur (739 µm)
and M2's clearance: blur and plate scale compete, and three conics cannot
buy both.  The plate scale is therefore step 4's problem too (aspheres
through CALIB's `OptAsph=`, or freeform) — or a re-posed first order with
the section's own magnification in the layout, which the design layer
does not have yet.  For Jim's table the honest TMA line is: unobscured
3-conic section, 1.5k strip 8.9 px (4.3 px centre), packaged, at 447 mm
plate scale (22 m GSD, the ±2.35° strip 36.6 mm at the slit); 3k strip not
imaged by conics.  `_d205_1k5.in` can be joined to the 1.5k Dyson as an
INFORMATIVE end-to-end row (the slit admits ±13.5 mm of its ±18.3 mm
strip); it is not the number.

**End to end (TO, t5e, resources 29a6df0 — informative).**  `_d205_1k5.in`
joined to the 130 mm silica module: plate scale 450 mm local / 521 mm at
the strip edge (spec 330), the 27 mm slit admits ±1.61° of ±2.34° (68.7 %
of the swath), the sky line images bowed by 7 mm at the strip end, and
over the admitted field smile 44.5 px (the same at every λ: the
telescope's chief off the slit line through the pupil mismatch, not the
Dyson), keystone 0.14 px, CRF 15.7 px, clearance +0.37 mm.  Chain = engine
to 4e-12 m.  To score the next deck: `dyson5_run(struct('stages',{{'t5e'}},
'tel_npix_xt',1500,'tel_gsd_m',30,'tel_alt_m',550e3,'tel_dyson',
'size:D:130','tel5e_deck','<deck>.in','tel5e_suffix','_x'))`.
