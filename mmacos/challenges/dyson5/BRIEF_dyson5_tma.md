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

## For CCMac — step 4 hand-off (CC, 2026-10-04 14:10): the asphere hook is in

`Telescope.optimize` now takes `'asph_elts'` (the mirrors whose even-radial
terms CALIB varies) and `'asph_terms'` (1 = h⁴, 2 = h⁶, 3 = h⁸; default
[1 2]), alongside `'beam_pos_fov'` / `'beam_wt'` (per-field image-position
rows that pin the plate scale).  Engine macos 9fe033e (`elt_asph_get`),
resources 669e217 (hook, `macos.get_elt_asph`, gate `tAsphHook`).  Rebuild
the engine (both trees) and the mex after pulling; `./run_mmacos_tests.sh
tAsphHook` is the smoke (2 tests, ~4 min at model 256).

What it does, so the trap is known: the mirror is emitted `Surface=
Aspheric` from a zero seed; `OptAsph=` goes after `VarElt=`; and because
CALIB's zero-coefficient step is sag-based only with a CIRCULAR aperture
(a Telescope deck declares `ApType= None`, which would hand it a round-off
step on a metre deck), a vertex-centred circle enclosing the mirror's
footprint is declared for the solve and removed after.  On the coaxial
parent, h⁴+h⁶ on M1/M3 over two fields take the WFE [529 484 753] →
[55 117 308] nm.

Suggested start for step 4: `dyson5_tma_step2b_linux_d205_1k5.in` as the
seed (or rebuild it from the step-2 driver at d 0.205 / bias 3), then in
the driver's ladder a rung

```
P = plate_targets_(tel, nE, fields_full, 0.330);     % the driver's helper: 330 mm·tanθ targets
tel.optimize('fields', fields_full, 'dofs', [0 0 0 0 0 0 0 1], ...
             'asph_elts', [1 2 3], 'asph_terms', [1 2], ...
             'beam_pos_fov', P, 'beam_wt', 1e-2, 'max_iters', 150);
```

then walk `beam_wt` up (1e-2 → 1e-1 → 1) and watch blur vs the traced
plate scale (`efl_of_built_`) and M2's clearance (`check_clipping`).  If the
aspheres buy the scale without the blur, that is the 330 mm deck TO scores
with t5e.  The 3k strip is the same recipe on the ±4.7° fields.  Record as
`dyson5_tma_step4_*`; the brief here is the shared log.

## Step 3 (TO, 2026-10-04): telescopicity measured -- the pupil is the limiting loss; TO is editing `Telescope.optimize` NOW (the `'beam_dir'` block only)

**CCMac: please hold edits to `Telescope.optimize` until TO's next push to
this brief says the block is in.**  TO's change is confined to one new
option, `'beam_dir'` (per-element chief-DIRECTION target, CALIB
`OptBeamDir=` rows on the WFE target), added beside `'beam_pos_fov'` with
the same three api calls (`calib_set_beam('dir', ...)`,
`calib_set_beam_wt`, and the reset after the solve), plus a gate in
`tBeamRows`' idiom.  Your `'asph_elts'` / `'asph_terms'` options are not
touched.

Measured (t5e's new PUPIL MATCH table, `dyson5_t5e_1k5.txt`; the
telescope's own beam, chief through the deck's ApStop, on the joined deck,
no stop set, engine admitted fractions):

| field deg | x at slit mm | chief to slit normal deg | tel. pupil from slit | Dyson chief deg | Dyson pupil | miss at grating mm | admitted |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.00 | -- | 0.00 | -- | 0 | 1.000 |
| ±0.586 | ∓4.64 | ±11.7 | **−22 mm** | 0.02 | +13.8 m | 58 | 1.000 |
| ±1.172 | ∓9.52 | ±23.2 | −22 mm | 0.03 | +17.4 m | 114 | **0.420** (BlockSphereOut 0.57) |
| ±1.758 (off slit) | ∓14.9 | ±34.6 | −22 mm | 0.03 | +34.6 m | 166 | 0 |
| ±2.344 (off slit) | ∓21.3 | ±45.5 | −21 mm | 0.03 | −44 m | 212 | 0 |

The d205 TMA's exit pupil sits **22 mm in front of its image** (the
chiefs cross the slit axis there at every field); the Dyson of record
wants a TELECENTRIC input (its chiefs at ≤0.03° to the slit normal,
crossing 14–44 m away).  The chief arrives 11.7° off the Dyson's at
±0.59°, 23° at ±1.17°; the beam misses the grating (r 80.8 mm) by 114 mm
there and only 42% reaches the FPA, clipped at the Dyson's BlockSphereOut.
That is the limiting loss on the slit, ahead of blur (the strip ends
beyond ±1.61° fall off the slit anyway).  A telecentric image is a
first-order (pupil-position) property: a conics/asphere solve will not
move a pupil 22 mm -> infinity by itself; the `'beam_dir'` rows let the
solve SEE it, the layout (spacings/radii, the step-2 driver's d and f_req)
has to be free for it to move.  Suggest step 4 carries the beam_dir rows
(target = the slit normal, i.e. the FP's psi) at each strip field.

**Step 3 hook in (TO, 2026-10-04): `Telescope.optimize` is free again, CCMac.**
`'beam_dir'` (3x1, the chief's TRAVEL direction at the FocalPlane) adds the
engine's `OptBeamDir=` rows on the WFE target, weighted by the existing
`'beam_wt'`, switched off after the solve.  For a telecentric image pass the
detector's normal on the travel side (the d205 deck: its FP `psiElt`,
`[0 0 -1]` in the deck frame).  **One target is scored at EVERY CALIB field**,
so on a non-telecentric design the per-field chiefs straddle it (gate
measured: two fields 3.2e-3 / 3.0e-3 either side, mean within 8.5e-5); the
spread IS the pupil position, and only layout DOFs (pistons / radii) move
it -- conics + aspheres alone will not take a 22 mm pupil to infinity.
Gate `tBeamDirHook` (3 tests, SUITE_FAST; control without the rows stays
>1e-3 off; no leak into the next solve); `tAsphHook` re-run green on the
same tree.  Suggested step-4 rung: add `'beam_dir', [0;0;-1]` (deck frame)
with `'beam_wt'` walked like the position rows, PIST free on M2/M3.

## For CCMac — step 4 is now a TWO-stage task (CC, 2026-10-04, after TO's step 3)

TO measured the d205 section's pupil (`d603c87`, t5e pupil-match table):
the exit pupil sits **22 mm in front of the image**, so the chief meets the
slit at 11.7° / 23.2° / 34.6° / 45.5° at 0.59° / 1.17° / 1.76° / 2.34° of
strip, against a Dyson that is telecentric to 0.03°; the grating vertex is
missed by 58–114 mm; only the inner field is fully admitted.  That is the
STOP'S PLACE — first order — and aspheres cannot move it (TO's gate shows
the new `'beam_dir'` rows respond only to layout DOFs).  So:

**Stage A — telecentric first order.**  Re-pose the layout so the exit
pupil is at infinity: the aperture stop at the front focal point of the
M2+M3 group (equivalently, `tma_layout`'s spacings chosen so the section's
chief directions at the FP are parallel), then the eccentric section and
clearance as in step 2.  Measure with TO's t5e pupil table (chief angle to
the slit normal, pupil distance) before any figure solve; the acceptance
number is the Dyson's: chief within ~1° of the slit normal across the
strip, pupil distance ≫ the slit-to-grating 16.8 m-class figure.  If
`tma_layout` has no telecentric knob, the `'beam_dir'` rows with PIST free
on M2/M3 (TO's suggested rung in this brief) are the solver route; the
target direction is the FP normal in the deck frame.

**Stage B — figure.**  Only on a telecentric layout: `'asph_elts' [1 2 3]`
+ `'beam_pos_fov'` at 330 mm·tanθ (+ `'beam_dir'`), walking the weights as
written above.  Blur, plate scale, chief angle, pupil, grating miss and
admitted fraction all come out of ONE t5e run now.

The d205 deck stays the step-2 record; do not start stage B from it.

## Step 4 (CCMac, 2026-10-04) — aspheres: they buy the centre, not the field, and not the scale

Restarted from CC's d205 unobscured section (decenter 205 mm, bias 3° on the step-1
calibrated f=330/F1.8/D=183 Korsch — reproduced to the digit on the current engine:
conics 161.0 µm / 8.94 px / 447.5 mm plate, which is CC's d205 exactly). Added even-
radial aspheres h⁴+h⁶ on M1/M2/M3 via CC's hook (`asph_elts`/`asph_terms` → CALIB
`OptAsph=`), alone (blur) and with the plate-scale position rows at `beam_wt` 1e-2→1→
walked. Driver `dyson5_tma_step4.m`; records `dyson5_tma_step4.{txt,mat}` + layouts.
tAsphHook 2/0 on the rebuilt engine (macos 9fe033e) + mex.

**1.5k strip (±2.35°):**

| variant | worst spot | centre | plate | clearance |
|---|---|---|---|---|
| conics (= CC d205) | 161 µm / 8.9 px | 76.8 µm / 4.3 px | 447 mm | PASS +0.1 mm |
| **aspheres, blur only** | 150 µm / 8.3 px | **41.7 µm / 2.3 px** | 449 mm | FAIL (−, M2) |
| asph + scale wt 1e-2 | 426 µm | 99 µm | 422 mm | PASS |
| asph + scale wt 1e-1 | 4321 µm | 2824 µm | 426 mm | FAIL |
| asph + scale wt 1 | 55356 µm | 52230 µm | 17855 mm | FAIL |

**3k strip (±4.7°):**

| variant | worst spot | centre | plate | clearance |
|---|---|---|---|---|
| conics | 1487 µm / 82.6 px | 493 µm / 27 px | 496 mm | FAIL −0.4 mm |
| aspheres, blur only | 34660 µm (edges) | **20.2 µm / 1.1 px** | 450 mm | FAIL |
| asph + scale (any wt) | 24000–32000 µm | 259–5470 µm | 300–820 mm | FAIL |

**Three conclusions, all negative and all precise:**

1. **Aspheres buy the strip CENTRE, not the field.** h⁴+h⁶ pull the centre from 4.3→2.3 px
   (1.5k) and 27→1.1 px (3k) — a real improvement — but the ±4.7° / ±2.35° EDGES barely
   move (1.5k 161→150 µm) or explode (3k edges → 35 mm). Rotationally-symmetric aspheres
   cannot correct the field-growing coma/astigmatism of a wide off-axis strip; that needs
   **freeform** (non-symmetric), not aspheres. The hook works (CC's coaxial WFE
   529→55 nm reproduces); the limit is the symmetry, not the solver.

2. **The 447→330 mm plate scale is unreachable by aspheres.** Any weight on the position
   rows trades the blur away and then diverges (wt 1 → 17.8 m / 820 mm plate, nonsense):
   a 35 % magnification change is not something h⁴/h⁶ can carry. This confirms CC's
   suspicion — the plate scale needs a **re-posed first order with the eccentric section's
   own magnification**, which `tma_layout` / the Telescope class do not have (they calibrate
   the parent-axis EFL, not the section's).

3. **The 3k section does not even clear at d205** (−0.4 mm, 4 conflicts): the wider 3k beam
   needs its own decenter/geometry, not the 1.5k section's.

**For Jim's table, the TMA line stays CC's d205:** unobscured 3-conic 1.5k section, **8.9 px
worst / 4.3 px centre, 447 mm plate scale (22 m GSD, not 30), 1.05 kg, packaged, clearance
PASS**; the 3k strip and the 330 mm/F1.8 target are out of reach with the coaxial-Korsch
section + conics/aspheres. **Two capabilities the design layer is missing to close it:**
(a) an eccentric-section first order (for the plate scale), and (b) freeform field correction
(for the strip edges). Absent those, the TMA at Jim's numbers wants a freeform-from-scratch
or TO's two-mirror form — not more conics/aspheres on this parent. Layouts
`dyson5_tma_step4_layout_{1k5,3k}.png` (the unobscured eccentric section, beam on the upper
M1). Step 5 (the real end-to-end) is moot until the plate scale closes; the informative t5e
row (d205 + 130 mm Dyson, 68.7 % swath, smile 44.5 px) stands as the current TMA⊕Dyson number.

## For CCMac — stage A's tool is in: `tma_layout(..., 'telecentric', true)` (CC, 2026-10-04 18:00; resources d8035b8)

Your step 4 (0d0022e) and TO's step 3 agree: figure cannot move the pupil
or the plate scale; the first order must be re-posed.  `tma_layout` now
takes `'telecentric', true` (and `'stop'`, `'M1'` default or `'M2'`; `info`
reports `chief_exit_slope` and `exit_pupil_from_m3` always).  The measured
facts behind it: with a REAL intermediate focus between M2 and M3 no stop
placement is telecentric (M3's front focus lies between that focus and M3
-- the chief exit slope is −11..−39 per unit field at every M3 position,
which is the 22 mm pupil TO found); the telecentric Korsch is the
VIRTUAL-intermediate-image regime, M3 in FRONT of the intermediate focus,
between M2 and M1 (Cook / EMIT form).  On the dyson5 parent, f/1.8, m2 3.5:

| stop | R (m) | t (m) | M3 z | engine |
|---|---|---|---|---|
| M1 (what the Telescope emits) | [0.3667 0.0998 **0.1667**] | [0.1477 **0.0461**] | −0.102 (46 mm behind M2) | chiefs at the FP parallel to **9.9e-9 rad**, EFL 0.3300 m, all rays alive |
| M2 | [0.3667 0.0998 0.1283] | [0.1477 0.0642] | −0.084 | t2 = R3/2 exactly (needs the stop at M2 on the deck — your open `stop(2)` item) |

Gate `tTmaTelecentric` (SUITE_FAST).  **Stage A, concretely:** build the
M1-stop telecentric parent above, scan the bias and the eccentric decenter
with the step-2 ladder (`set_offaxis('none','dist',d)`, clearance via
`check_clipping` — the geometry is new: M3 sits between M2 and M1, so the
section's clear paths differ from d205), confirm telecentricity on the
SECTION with TO's t5e pupil table (chief angle at the slit, pupil
distance) and read the traced plate scale at the bias BEFORE any figure.
If the plate scale on this parent stays off 330 at the working bias, that
is the next first-order knob (the section's own magnification), and it is
measured, not guessed.  Then stage B (aspheres / freeform + position
rows) on that section.  The 3k strip gets its own geometry on the same
parent (your finding 3).

## Lane note (2026-10-04, Dave): TO runs stages A and B; CCMac on hold

Stage A (the section on the telecentric parent, measured by the t5e pupil
table and plate scale before figure) and stage B (aspheres + position
rows, the 3k geometry) are TO's, running now (addendum 37 of
`BRIEF_to_dyson5.md`).  CCMac stands down until stage B shows the strip
edges are the limit on a telecentric, correctly-scaled section; the
freeform step is then theirs.

## Stage A (TO, 2026-10-04): the telecentric section -- the pupil is FIXED; the first order closes at the bias; the figure moves the scale

Runner stage **`tA`** (`dyson5_run`, knobs `tA_*` in `dyson5_params`): the
telecentric Korsch parent (`tma_layout(..., 'telecentric', true)`, M1 stop;
R = [0.3667 0.0998 0.1667], t = [0.1477 0.0460]), bias x decenter scanned AS
IS with `check_clipping`, the three numbers read from the engine before any
figure, the parent F/# calibrated so the SECTION's local plate scale is
330 mm, CC's step-2 conic ladder on the pick, and the deck through `t5e`.
Records `dyson5_tA_{1k5,1k5c,3k}.{txt,mat}`, decks `dyson5_tA_*_{as-is,strip}.in`,
end to end `dyson5_t5e_tA_*_strip.*`.

**The scan (as is, 1.5k strip).** Clearance needs NEGATIVE bias (the field
pushed away from the decenter): every bias >= -2 deg FAILs at M2 (-6 to
-21 mm, the M3->FP beam through M2's body); -3 deg clears from d 140 mm,
-4 from 120, -6/-8 widely.  The three numbers on the clear cells:

| bias | dec | clear | chief spread | pupil (edge chief x centre) | plate local / edge |
|---|---|---|---|---|---|
| -3 | 160 | +4.7 mm M2 | 0.48 deg | +1.7 m | 340.4 / 349.8 mm |
| -4 | 120 | +3.9 mm M3 | 0.47 deg | -1.0 m | 353.3 / 362.0 mm |
| -6 | 120 | +19.6 mm M3 | 1.60 deg | -0.5 m | 389.5 / 396.4 mm |
| -8 | 120 | +29.3 mm FP | 2.92 deg | -0.3 m | 432.0 / 436.3 mm |

(the d205 deck: 45 deg / 22 mm.)  **The pupil is fixed**: chiefs within
0.5 deg of each other on the strip, the pupil ~1 m out.  The section's
chiefs share a common tilt atan(d/f) to the parent FP normal (26.9 deg at
160 mm), about the slit axis -- harmless along the slit.  **The plate scale
grows with |bias|** (330 -> 340 -> 353 -> 390 -> 432 mm at 0/-3/-4/-6/-8
deg): the section's local magnification off the parent axis, a first-order
property of where the section sits, so the clearance and the scale trade
through the bias.

**Calibration (first order).**  At the working point (-3 deg, 160 mm) the
parent's F/1.745 (EFL 0.3199 m) puts the AS-IS section's local plate scale
at **329.5 mm** (one pass), spread 0.51 deg, pupil +1.5 m, clearance PASS
+3.7 mm.  **Stage A's three numbers close at first order.**

**Two first-order residuals with mechanisms (not figure jobs):**
1. **The conic solve moves the scale +-3 %, bistably.**  With the strip
   conic solve INSIDE the calibration loop the plate scale flips between two
   CALIB basins every pass -- 341 mm (K3 +0.2, spread 0.88 deg) and 319 mm
   (K3 -15, spread 0.53 deg) -- so the solved section's scale is set by
   which conic minimum CALIB lands in, not by the first order.  Stage B must
   therefore carry the position rows (`beam_pos_fov`) in EVERY figure rung
   (CC's recipe), not calibrate and then solve.  (`tA` now keeps the best
   pass when the loop alternates.)
2. **Distortion across the 3k strip.**  On the 3k section (-3 deg, 180 mm,
   clear +4.1 mm) the local scale calibrates to 329.8 mm but the EDGE reads
   355.5 mm (+7.8 %): the +-4.69 deg strip images 58.3 mm long against the
   54 mm slit, 93.3 % of the swath on it.  1.5k: +2.6 % (338/330 mm).  A
   field-dependent magnification is in the position rows' reach (per-field
   targets), or the slit/Dyson length absorbs it -- a call for stage B.

**Blur (conics only, informational):** 1.5k strip-solved 18.6 px worst /
6.2 centre; 3k 36 px (the 3k conic solve did not move the conics).  Worse
than d205's 8.9 px -- the telecentric section starts further from a nulled
anastigmat.  That is stage B's job.

**End to end (t5e, informative):** the pupil match is solved -- admitted
**0.997-1.000 at every 1.5k field** (d205: 0.42 at 1.17 deg), 0.98-1.00 on
3k; the telescope chief within 0.5 deg (1.5k) / 0.94 deg (3k) of the
Dyson's, grating miss <= 2.4 / 9 mm; 100 % / 93 % of the swath on the
slit.  Smile/CRF/SRF are still the telescope's blur.  **Packaging finding:**
the joined decks FAIL `spectrometer_clearance` -- the Dyson's FPA package
sits 4-5 mm inside the telescope's M2 (1.5k first run, 3k), or the
M1->M2 leg grazes M3 by 4.8 mm under its 1.15x footprint discs (1.5k
calibrated) -- the Dyson is placed in the telescope's frame by the slit
alone; its roll about the chief is free and is the knob.
