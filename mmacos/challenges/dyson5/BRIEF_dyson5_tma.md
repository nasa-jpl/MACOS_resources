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

## Stage B, 1.5k (TO, 2026-10-04): aspheres + position rows hold 330 mm and kill the distortion; the edges still blur

`tA` rung 3 (`tA_rungs` [0 3]): on the stage-A working point (parent
F/1.745, bias -3 deg, decenter 160 mm) the strip solve carries conics +
h4+h6 on M1-M3 (`asph_elts` 1:3, CCMac's hook) + per-field POSITION rows at
f*tan(theta) (`beam_pos_fov`, CC's plate_targets_), one row per `beam_wt`.
Records `dyson5_tA_B1k5.{txt,mat}`, decks `dyson5_tA_B1k5_B{0.01,0.1,1}.in`.

| rung (beam_wt) | clear | spread | pupil | plate local / edge | rms spot per field, um (best focus) | worst |
|---|---|---|---|---|---|---|
| as-is | +3.7 M2 | 0.51 deg | +1.5 m | 329.5 / 338.2 | 252 307 341 356 341 307 252 | 19.8 px |
| B 1e-2 | +2.7 M3 | 0.88 | +0.9 | 330.3 / 331.7 | 293 135 48 24 48 135 293 | 16.3 px |
| B 1e-1 | +2.8 M3 | 0.87 | +0.9 | 330.5 / 332.0 | 292 134 47 24 47 134 292 | 16.3 px |
| **B 1** | **+2.9 M3** | **0.85** | **+0.9** | **330.8 / 332.6** | **284 131 48 23 48 131 284** | **15.8 px** |

With the position rows the scale no longer jumps basins: 330.3-330.8 mm
local at every weight and the strip-edge distortion goes from +2.6 % to
+0.5 %.  The centre and the inner half-strip come to 1.3-2.7 px; the outer
field (+-2.35 deg) stays at 284 um -- the edge blur is the residual,
rotationally symmetric aspheres do not reach it (CCMac's step-4 finding
reproduced on the telecentric section).

**End to end (B 1 + `size:D:130`, t5e; bridge 21 330 rays chain == engine
to 7e-15 m):** plate 330.8 / 331.9 mm, **99.4 % of the swath on the slit,
admitted 1.000 at every field, the chief within 0.82 deg of the Dyson's,
grating miss <= 4.1 mm**; smile 1.96 px, keystone 0.68 px, SRF 3.0 px,
CRF 15.5 px (the edge blur), EE 0.004; clearance FAIL -4.95 mm (the
Dyson's FPA package inside the telescope's M2 -- the join's roll about the
chief is the knob, admitted fraction to be re-read after it).

**Parse bug found by the bridge gate (fixed, `tel_deck_geom`):** the key
lookup matched `AsphCoef=` inside `nAsphCoef=`, so the chain read
nAsphCoef's "2" as M1's h^4 coefficient -- 1.4 mm at M1, 2.4 cm at the FP,
a garbage first t5e row (378 mm, admitted 0.48), overwritten.  All keys are
now line-anchored, and a wrapped `AsphCoef=` line is refused (count vs
`nAsphCoef`).  Stage-A rows (conic decks) were unaffected (gate 1e-15).

## Stage B, 3k + the join's roll (TO, 2026-10-04)

**3k at -3 deg / 180 mm: CALIB cannot evaluate the section.**  Its
per-field WFE reads 8.5 / 3.9 / 6.8 mm (centre, +-1.56 deg) against
0.05-0.24 mm at the outer fields, with 27-234 of 254 rays passing, while a
plain engine trace of the same deck loses 18 rays and gives a 410 um spot;
every rung leaves the conics where they were.  Reproduced in isolation
(`Telescope.optimize`, 3 iterations): mm-scale at -3 deg for 180 and
200 mm and on the 1.5k strip too; sane at -4 deg / 180 mm.  **OPEN item
(CALIB per-field evaluation vs the engine trace on an eccentric section;
not chased further).**

**3k at -4 deg / 180 mm (clear +18.9 mm as is):** first order calibrates
in two passes (parent F/1.7133, 330.3 mm local).  Stage B
(`dyson5_tA_B3k4`): the position rows close the scale AND the +9.8 %
distortion -- **328.1 / 325.4 mm, 100 % of the swath on the 54 mm slit** --
and the centre to 58 um; **but CALIB flags the +-4.69 deg fields as
failed (9.9999e36) and solves without them**, so the edges sit at 1.2 mm
(67 px).  The rows also push the chief spread to 2.5 deg (grating miss
24 mm of a 154 mm radius); admitted still 0.99-1.00 at every field.
E2E (B 1e-1, `size:F:240`): smile 28 px / keystone 0.37 px / CRF 15.7 /
SRF 15.0 -- the edge blur.  The 3k blocker is the same CALIB evaluation
anomaly at the strip edge, then figure.

**The join's roll** (`tel5e_roll_deg`, `tel_deck_geom` 'roll_deg': the
telescope rolled about the exit chief; 180 keeps the image line on the
slit): on the 1.5k B1 deck, roll 180 **clears the Dyson's FPA package from
M2**; admitted re-read: 1.000 at every field, smile/keystone unchanged
(1.96 / 0.69 px).  What is left is TELESCOPE-internal: the M1->M2 leg vs M3
at -4.88 mm under `spectrometer_clearance`'s bodies (1.15 x footprint
discs) where `check_clipping` (exact footprints) passes it at +2.9 mm --
the two gates' body models differ by exactly that margin; a larger
decenter (the scan has 160 -> 140 mm worse, 180 mm +5 mm better) is the
knob, re-calibrated.

**Where stage B stands.**  1.5k: telecentric, 330.8 mm, distortion 0.5 %,
99.4 % on the slit, admitted 1.000, centre 1.3 px, inner half-strip
<= 2.7 px, **edges 15.8 px** -- the rotationally symmetric aspheres stop
at the outer field (CCMac's step-4 finding again).  3k: first order and
distortion close; the solve cannot see the edge fields (CALIB anomaly).
Next DOFs, in order: (1) the CALIB edge-field evaluation (3k blocker);
(2) freeform / non-symmetric terms for the outer field (both modules);
(3) decenter + re-calibration for the clearance margin.

## Step 5 (CCMac, 2026-10-04) — freeform on the outer field: taking item (2)

Dave released CCMac (`BRIEF_ccmac_dyson_size.md` round-4 orders) to take item (2):
non-symmetric (freeform / Zernike) terms for the outer field on both modules, on
the stage-B sections (−3°/160 mm 1.5k = `dyson5_tA_B1k5_B1.in`; −4°/180 mm 3k =
`dyson5_tA_B3k4_B0.1.in`), the per-field position rows in EVERY rung so the
330 mm plate scale holds, each deck scored by t5e; budget one bounded rung per
module.  Record `dyson5_tma_step5_*`.

**Two coordination notes.**

1. **`Telescope.optimize_freeform` extended (CCMac, my tooling — NOT
   `Telescope.optimize`, which stays TO's).**  Added `'beam_pos_fov'` /
   `'beam_dir'` / `'beam_wt'` (defaults empty = prior behaviour unchanged),
   plumbed with the SAME `calib_set_beam*` calls `optimize()` uses, so the
   Zernike (OptZern) solve pins the plate scale and telecentricity in one pass.
   TO is a caller only of the freeform path; nothing in `optimize` is touched.

2. **The emitter carries conic+Zernike OR conic+asph, not both**
   (`Telescope.build`, `if hasAsph && ~hasFree`).  Dave: extend the emitter to
   carry both — but matching the engine's `Surface= Zernike` `ZernCoef`
   normalization for an ABSOLUTE asphere→Zernike conversion hit a ~2× (plus a
   per-mode residual) wall; the exact fix is to emit via the `FreeForm`
   `MonZern` channel (gated-exact to `zernike_grid_basis`).  **Attempts +
   fix documented in `NOTE_asph_zernike_fold.md`; to-do added to
   `macos/PLAN_DESIGN_LAYER.md` "Sprint 6+ (deferred)" — CC-Linux's to land
   (no engine change, A/B round-trip gate `tTmaAsphZernFold`).**  The combined
   asph+freeform emit path now ERRORS (informative) instead of emitting a wrong
   deck.  `asph_to_zern_` (the projection, correct in the MonZern convention)
   and the `ansi_zernike_eval` analytic-norm extension (bit-identical 1–15)
   stay in the tree, reusable by that fix.

3. **Step 5 proceeds via route B (Dave):** a self-consistent `optimize_freeform`
   solve from the stage-B CONIC geometry (same parent / bias / decenter /
   conics) over the CORRECT symmetric modes {5, 13, 25} (defocus + primary +
   secondary spherical — the aspheres' job) PLUS non-symmetric {4, 6, 7, 9, 10}
   (astig / coma / trefoil), with the position rows in the solve.  No absolute
   asph→Zernike conversion, no `ZernCoef` convention dependence — the optimizer
   finds the coefficients self-consistently.  (My first step-5 run's "ANSI 4–11"
   OMITTED the spherical modes 13/25, which is why it could not reproduce the
   aspheres' centre.)  3k edges are scored from the trace (the CALIB
   edge-evaluation anomaly, item 1, still drops ±4.69°).

**Result (route B, one bounded rung per module; `dyson5_tma_step5.{txt,mat}`,
decks `dyson5_tma_step5_{1k5,3k}.in`, modes {4,5,6,7,8,9,10,13,25} ANSI, lMon =
footprint radius, beam_wt 0.1, 200 iters).**  Best-focus rms spot per field
(trace); e2e by t5e.

| module | conic base worst/ctr | **freeform worst/ctr** | plate (my efl) | e2e t5e (smile / keystone / CRF / SRF, px) | e2e plate local/edge | slit admit | clearance |
|---|---|---|---|---|---|---|---|
| 1.5k (−3°/160) | 28.5 / 28.5 px | **11.2 / 5.0 px** | 324 mm | 6.47 / 0.54 / 15.70 / 15.66 | 327.8 / 335.6 mm | 98.3 % | −4.89 mm (TelM1→M2 vs M3) |
| 3k (−4°/180) | 57.5 / 57.5 px | **46.7 / 21.1 px** | 320 mm | 39.1 / 1.05 / 15.96 / 15.01 | 320.6 / 344.0 mm | 96.1 % | −1.75 mm (FPApkg vs TelM2) |

**Reading.**  Adding the spherical modes 13/25 is the fix to my step-4/first-
step-5 negative: it drops the 1.5k CENTRE 14→5 px and the WORST 15.8→**11.2 px**
(beating stage-B's 15.8 px edge), and the 3k centre 57→21 px.  But one bounded
low-order freeform rung does NOT reach the pixel at the F/1.8 strip edge
(±2.35° / ±4.69°); it also trades ~2 % against the 330 mm plate scale (324–328 mm
local, position rows mostly holding it).  The e2e CRF/SRF ~15.7 px is the
telescope's residual strip blur seen through an unmasked slit (not the Dyson's
distortion — t5e's own reading).  **3k stays blocked by the CALIB edge-eval
anomaly (item 1):** the solve is blind to ±4.69°, so 3k edges hold ~47 px —
item 2 cannot help 3k until item 1 lands.  Clearance FAILs are TELESCOPE-internal
(the join-roll knob `tel5e_roll_deg`, TO's), not the figure.  Net: freeform
flattens and improves the worst-case strip but the fast wide eccentric section
does not reach the pixel with one bounded rung — consistent with step 4.  Next
levers if pursued: more rungs / higher orders, a stronger plate weight, or the
exact asph+freeform co-emit (deferred) to start from TO's corrected centre.
## Addendum 39 items (TO, 2026-10-04): the decenter walk, the two clearance checks reconciled, the 3k edge from the trace

**1. More decenter on the 1.5k section** (`tA` per point: first order
re-calibrated to 330, stage-B rung at `beam_wt` 1 -- position rows in the
rung, roll 180 in the join; records `dyson5_tA_B2_1k5_b<bias>d<dec>.*`,
`dyson5_t5e_tA_B2_1k5_*`).  CALIB's own per-field evaluation is used as a
FLAG only ("ANOMALOUS" = failed or > 1 mm at the BEFORE state; those fields
were not solved), never as a number.

| bias / dec | CALIB | check_clipping | spectr. clearance, tel-internal: +5 mm mount / zero | plate local / edge | spot (trace, best focus) centre / +-1.17 / edge | e2e smile / keyst / SRF / CRF px | admitted |
|---|---|---|---|---|---|---|---|
| -3 / 160 (stage B) | ok | +2.9 | -4.95 / -- | 330.8 / 332.6 | 23 / 48 / 284 um | 1.96 / 0.68 / 3.0 / 15.5 | 1.000 |
| -3 / 170 | **fields 1,4,5** | +3.8 | -4.65 / +5.27 | (unsolved) | 361 um | -- | -- |
| -4 / 160 | ok | +7.1 | -4.82 / +4.46 | 329.8 / 329.5 | 61 / 96 / 473 um | 4.45 / 1.10 / 3.1 / 15.9 | 1.000 |
| -4 / 180 | ok | +11.9 | -2.12 / +7.38 | 331.3 / 331.5 | 47 / 86 / 421 um | 1.87 / 1.01 / 4.3 / 15.8 | 1.000 |
| **-4 / 190** | **ok** | **+14.4** | **-1.17 / +8.53** | **332.9 / 333.2** | **43 / 79 / 371 um (20.6 px)** | **1.49 / 0.87 / 5.5 / 15.5** | **1.000** |
| -4 / 200 | **fields 1,4,5** | +16.1 | +4.72 / +14.17 | (unsolved) | 722 um | -- | -- |
| **-5 / 200** | **field 1** | +20.5 | **+2.54 / +12.02 PASS** | 331.3 / 329.2 | 72 / 116 / 453 um (25.2 px) | 3.42 / 0.92 / 6.6 / 15.2 | 1.000 |

Reading: the margin grows with decenter and |bias| (record's number
-4.95 -> -1.17 -> +2.54 mm); the cleanest SOLVED point is -4 / 190 (1.2 mm
short with the mount); the first point that PASSES with the mount is
-5 / 200, where CALIB flags the centre field (the solve still lowered the
other fields; its centre number stands from the trace).  The CALIB
anomaly region (CC's item) sits exactly where the margin is.

**2. The two clearance checks, reconciled.**  They are not 1.15x discs vs
footprints.  `Telescope.check_clipping`: bodies = the telescope's own
beam-footprint discs (its draw fans), NO mount.  `spectrometer_clearance`:
bodies = the footprint of the INSTRUMENT's bundle (grating-stopped,
slit centre + ends x 3 lambda x marginal rings) grown by `mount_margin_m`
(5 mm) AND the leg's distance to that grown body then reduced by the
mount again (`body_pts_` R = footprint + mount, `d - Bd.mount`): **the
mount is counted twice -- a ~10 mm standoff**, which is the 9.3-9.7 mm
measured between the with/without columns above (the rest is the 2 mm
surface sampling).  At zero mount the gate still reads 2.6-8.5 mm LESS
than check_clipping on the same pair: the two bundles differ (the
instrument's bundle is aimed through the grating, the telescope's fans
through its own stop; and check_clipping's min is over its own pairs).
**The record uses `spectrometer_clearance` with the 5 mm mount** (the
stage's FAIL criterion since addendum 6, and the only gate that sees the
Dyson's bodies).  Whether the mount should be counted once (grown disc OR
subtracted distance) is a gate-definition call for Dave -- counted once,
-4 / 190 passes by ~+3.8 mm and -4 / 180 by ~+2.6 mm.  Not changed.

**3. The 3k edge from the trace** (-4 deg / 180 mm, `dyson5_tA_B3k4`):
+-4.69 deg = **1196 um (B 0.1) / 1206 um (B 1) best-focus rms = 67 px**;
on the deck's own FocalPlane along the slit line (t5e gate, as placed)
2.19 mm.  Centre 58-60 um, +-1.56 deg 237-240 um, +-3.13 deg 653-660 um.
These are engine-trace numbers; CALIB's edge WFE is not reported.

**Mount counted once (Dave, 2026-10-04).**  `spectrometer_clearance` now
counts the mount once, as the body's extent (aperture disc = footprint +
`mount_margin_m`); the extra subtraction from the distance is gone.
Re-scored (t5e, roll 180, bridge 7e-15 m, everything else unchanged):

| point | telescope-internal worst (M1->M2 vs M3), 5 mm mount / zero | worst overall |
|---|---|---|
| **-4 / 190** | **+3.83 / +8.53 mm** | +0.45 mm, BlockSphereIn->BlockFaceOut vs FPApackage (Dyson-internal, no mount) -- **PASS** |
| **-4 / 180** | **+2.88 / +7.38 mm** | +0.42 mm, same pair -- **PASS** |

Both decks now clear with the mount.  -4 / 190 is the 1.5k telescope of
record for stage B: 332.9 mm (+0.9 %), 99 % of the swath on the slit,
admitted 1.000, trace spots 43 / 79 / 371 um, smile 1.49 / keystone 0.87 /
SRF 5.5 px, CRF 15.5 px (the edge blur).  **Records scored before this
change carry the double count** on every mirror-body entry (~5 mm too
pessimistic; box bodies -- slit mask, FPA package -- had no mount and are
unchanged).

## Stage B re-run on FwdRoot + the all-field asphere circle (TO, 2026-10-04 evening)

Engine macos 1ba6874 (FwdRoot: the forward conic root on the VERTEX
sheet) + resources 125ea9f (the asphere hook's circle encloses every solve
field), shared mex relinked (tFwdRoot 5/5).  Every point re-run through
`tA` (stage-B rung, position rows in the rung) and `t5e` (roll 180, mount
counted once); bridge chain == engine <= 1.1e-14 m on all eight decks.
Records `dyson5_tA_B3_*`, `dyson5_t5e_tA_B3_*` (the pre-fix `_B2_` /
`_B1k5` / `_B3k4` records stay, as cited above -- their solves were partly
blind: wrong-sheet rays and clipped edge fields).  All 1.5k rows beam_wt 1.

| point | check_clipping | spectr. clearance tel-internal (5 mm mount, once) | plate local / edge | trace spot centre / +-1.17 / +-2.35 deg | e2e smile / keyst / SRF / CRF px | swath on slit |
|---|---|---|---|---|---|---|
| -3 / 160 | +3.1 | +0.31 | 331.2 / 332.3 | 21 / 43 / 273 um | 2.75 / 0.55 / 3.7 / 15.3 | 99.2 % |
| -3 / 170 | +5.4 | +0.05 | 332.4 / 333.1 | 19 / 43 / 270 um | 2.32 / 0.54 / 4.6 / 15.2 | 99.0 % |
| -4 / 160 | +6.7 | +0.60 | 329.0 / 329.7 | 52 / 81 / 371 um | 6.41 / 0.70 / 4.5 / 14.5 | 100 % |
| -4 / 180 | +11.9 | +2.80 | 331.9 / 332.1 | 41 / 71 / 352 um | 4.12 / 0.67 / 4.9 / 14.7 | 99.3 % |
| **-4 / 190** | **+14.6** | **+3.97** | **333.8 / 333.9** | **35 / 67 / 343 um (19 px)** | **3.26 / 0.63 / 5.9 / 15.5** | **98.8 %** |
| -4 / 200 * | +17.3 | +5.11 | 336.1 / 336.1 | 32 / 64 / 337 um | 2.64 / 0.60 / 7.5 / 15.1 | 98.1 % |
| -5 / 200 * | +19.3 | +6.44 | 330.9 / 329.1 | 69 / 109 / 423 um | 4.92 / 0.75 / 5.6 / 15.6 | 100 % |

Admitted 1.000 at every field on every point.  Worst OVERALL entry on all
seven is the Dyson's own FPA package vs its block face, +0.37..+0.40 mm
(-3/160 and -3/170: the telescope pair is the worst, +0.31 / +0.05 mm).
(*) the tA flag fires on CALIB field 1 at the BEFORE state only: those
seeds start above 1 mm at the centre; the solve moved and the trace spot
is sane -- the flag's 1 mm threshold on the seed is too strict there.

**Readings.**  (1) Every point now SOLVES, and every point PASSES
clearance with the mount counted once.  (2) The edge blur is the same wall
everywhere: 270-420 um (15-24 px) at +-2.35 deg with conics + h4/h6 --
the symmetric-asphere limit, now measured on honest solves (CCMac's
freeform).  (3) Margin vs blur vs scale: -3/170 has the best edge
(270 um) at +0.05 mm; -4/180 holds 0.9 % scale at +2.8 mm; -4/190 gives
+4 mm at 343 um and +1.45 % scale (the rows at wt 1 let the scale drift as
the decenter grows).  The pre-fix -4/190 row (smile 1.49 px) was a
partly-blind solve; its honest smile is 3.26 px.

**3k -4 / 180, re-run:** CALIB now sees every field (no flag); the edges
stay at 1.23 mm (68 px) best-focus from the trace -- with every field in
the solve, that is the asphere limit at +-4.69 deg, not a blind spot.
Scale 326.8 / 319.2 mm (-3.3 % at the edge), smile 10.6 px, keystone
1.37 px, admitted 0.98-1.00, clearance PASS (+1.68 mm telescope-internal).

## Addendum 42, checkpoint 1a (TO, 2026-10-05): the add_pupil/optimize stop TRAP is real -- TO is editing `Telescope.m` (the two stop calls only)

On the -4 / 190 mm 1.5k section, `add_pupil` ends in the right state (its
final build restores the header ApStop; chief at M1 y 193.5 mm), but it
calls `macos.stop(stop_elt=1)` with NO offset BEFORE its FEX call: FEX ran
with "Computed StopPos = (0,0,0)" -- the PARENT vertex, 190 mm off the beam --
and put the exit-pupil sphere at (0, -6.7 mm, 0.425 m), radius 0.571 m, the
two FEX probe axes disagreeing 0.333 vs 0.809 m (a beam that is not this
telescope's).  `optimize`'s OptFEX branch has the same bare `macos.stop(1)`,
and smacos_compute re-issues the STOP OBJECT of the last StopPos at every
evaluation, so it would poison every iterate.

**Fix (TO, now, by path):** a private `stop_at_apstop_` in `Telescope`
sets the stop with `macos.stop_obj` at the spec's ApStop point -- the same
(0, aperture_decenter, 0) `build()` writes to the deck header -- and
replaces the bare `macos.stop(1)` in `optimize` (use_ep) and the bare stop
in `add_pupil` when `stop_elt` is 1 (other stop elements unchanged).
Coaxial decks: ApStop = M1's vertex, the same point -- asserted
bit-identical in the gate.  Gate: the EP lands on the section's beam
(radius class ~1 m, probe axes agreeing), StopPos == ApStop after
`optimize`, and the coaxial twin unchanged.

## Addendum 42 (TO, 2026-10-05): the strict merit vs the FP merit on the -4 / 190 section -- the edge wall is the OPTICS

Runner stage **`tEP`** (`tEP_*` knobs).  Records `dyson5_tA_EP_cp1.*`
(checkpoint 1), `_cp2.*` (strict solve from the seed), `_cp2w.*` (strict
solve warm-started from the FP-merit B1), end to end
`dyson5_t5e_tA_EP_cp2w_S1.*`.

**Construction.**  The engine's exit-pupil merit (`add_pupil` + OptFEX)
cannot serve this section even with the stop fixed (d33abac): FEX finds a
far, astigmatic pupil (probe axes disagree by 0.47-0.77 m) and the EP
sphere loses every ray -- CALIB aborts.  So the strict metric is computed
from the engine trace: per field, rms OPL on a 1 m sphere about the
field's chief DETECTOR intercept (the solve's merit; the best-focus-centred
variant is reported too), each ray carried back along its final leg.  The
strict solve is `lsqnonlin` over the stage-B DOFs (conics + h4/h6 on
M1-M3) with the per-field position rows (f tan theta), weighted as CALIB
weighs them at beam_wt 1; every evaluation writes the deck and reloads it
(no engine asphere setter).  The evaluator reproduces checkpoint 1's seed
numbers exactly (FP 807.945 / STRICT 40.394 um at the centre).

**Checkpoint 1 (seed + the FP-merit solve):** the FP merit floors at ~16 um
at every field (strict 2.0 um at the centre) and sees the edge 2.1x the
centre where the strict metric sees 13.7x -- it hides most of the edge
from the solve (Spearman with the spot: strict 1.00, FP 0.89).

**Checkpoint 2:**

| solve | spot centre / 0.78 / 1.56 / 2.34 deg (um) | strict@focus centre / edge (um) | plate | e2e smile / keyst / SRF / CRF px | clearance (tel-int) |
|---|---|---|---|---|---|
| FP merit (B1, beam_wt 1) | 37 / 75 / 176 / **343** | 2.0 / 26.8 | 333.8 mm | 3.26 / 0.63 / 5.9 / 15.5 | +3.97 mm |
| strict, from the seed | 251 / 264 / 291 / 324 | 17.1 / 20.2 | -- | -- | -- |
| **strict, warm from B1** | **31 / 67 / 166 / 332** | **1.5 / 25.7** | **334.0 mm** | **3.23 / 0.66 / 10.5 / 15.7** | **+4.45 mm** |

From the seed the strict solve converges (first-order optimality, 5
iterations) into a WORSE basin than B1 (its own cost 2.77e-6 vs B1's
1.29e-6); warm-started from B1 it converges after 5 iterations having
improved its own cost 10 % and moved the edge spot **343 -> 332 um (3 %)** --
inside the basin scatter.  **The edge wall is the optics** (these
rotationally symmetric DOFs on this section), not the FP metric: the
metric mis-ranks the fields, but optimising the right metric does not buy
the edges.  The strict merit is the right REPORTING metric from here on;
it is not, for this question, a design lever -- the engine-side fixed-
station EP sphere is not needed to answer it (it remains the right tool for
OptFEX on near-telecentric decks, whenever that is wanted).  Next lever:
non-symmetric DOFs (CCMac's freeform) or the layout.

**Why the strict solve's SRF went 5.9 -> 10.5 px (TO, 2026-10-05).**  The worst
SRF MOVED from the strip edge to the centre: per field (max over lambda)
B1 5.85 / 3.39 / 3.53 / 4.36 px (edge -> centre), S1 3.00 / 7.16 / 9.71 /
10.51 px; CRF at the centre 1.8 -> 5.3 px.  The strict merit was centred on
each field's chief DETECTOR intercept, so defocus relative to the detector
is in it, and the solve balanced focus across the strip (field curvature):
the centre sits ~6 um rms of defocus off the detector (strict@chief 6.2 um vs
strict@focus 1.5 um there; ~4 px of blur at F/1.8) while the edge comes into
focus.  The best-focus spot metric removes focus, which is why the
telescope-alone spots read better.  Not the position rows, not t5e's slit
sampling.  (The strict merit is the REPORTING metric from here on; report it
about the detector intercept AND about best focus -- they answer different
questions.)

**The strict-merit solver stays as a tool:** `tEP` rung `S1`
(`tEP_strict_solve_` in `dyson5_run.m`: lsqnonlin, deck write + reload per
evaluation, `tEP_from` seed|B1, `tEP_center` chief|focus) -- the only
strict-merit solver until a fixed-station Return is in CALIB.

## Addendum 43 (TO, 2026-10-05): the Petzval scan -- the "curvature" is astigmatism; m2 3.0 helps a little, the flat-medial m2 6.0 is worse

**Step 1** (`tPZ`, `dyson5_tA_pz_1k5.txt`; per m2 the telecentric parent
re-solved and F/# calibrated to the 330 mm section plate, -4 / 190 mm, as-is
conics).  The first-order prediction is **falsified**: P = 1/R1 - 1/R2 +
1/R3 crosses zero near m2 2.9 (3.0 clears, +3.9 mm; 2.5 FAILs) while the
engine's MEDIAL best-focus shift edge-minus-centre is +1.37 mm there and
crosses zero near m2 5.9 (-0.17 mm at 6.0); slope vs P 0.83 mm per 1/m
against h^2/2 = 0.09.  The section's best-focus surface is the medial one,
Petzval plus the astigmatic term, and the astigmatism dominates.

**Step 2** (`tEP` rung B1 per m2 -- conics + h4/h6 + position rows at
beam_wt 1 -- records `dyson5_tA_EP_pz_m{35,30,60}.*`, t5e roll 180
`dyson5_t5e_tA_EP_pz_m*_B1.*`; bridges <= 1.3e-14 m).  Edge = +-2.34 deg.

| m2 | parent F/# | spot centre / 0.78 / 1.56 / edge (um, best focus) | strict@focus / @chief at the edge (um) | defocus term sqrt(c^2-f^2) | x-fan / y-fan / medial focus, edge-centre (mm) | T-S at edge | plate | e2e smile / keyst / SRF / CRF px | M2 R (F/# on its beam) / M1-M2 |
|---|---|---|---|---|---|---|---|---|---|
| 3.5 | 1.7157 | 37 / 75 / 176 / 343 | 26.8 / 33.9 | 20.8 um | -2.42 / +0.10 / -1.12 | -2.77 mm | 333.8 mm | 3.26 / 0.63 / 5.9 / 15.5 | 99.8 mm (F/0.70) / 147.7 mm |
| **3.0 (P ~ 0)** | 1.7140 | **29 / 56 / 141 / 291** | **23.0 / 27.9** | **15.8 um** | **-2.03 / +0.27 / -0.79** | **-2.22 mm** | 334.1 mm | **2.63 / 0.49 / 9.7 / 15.5** | 120.3 mm (F/0.73) / 143.2 mm |
| 6.0 (flat medial, as is) | 1.6474 | 187 / 241 / 402 / 677 | 52.2 / 78.1 | 58 um | -4.15 / -0.68 / -2.61 | -5.30 mm | 321.1 mm | 12.8 / 1.22 / 7.4 / 15.7 | 55.0 mm (F/0.60) / 160.4 mm |

**Readings.**
1. **The curvature is astigmatism.**  The along-track (y-fan) focus is
   nearly flat at every m2; the cross-track (x-fan) focus curves 2-4 mm, and
   T-S at the edge is 2.2-5.3 mm.  The medial curvature is mostly that
   astigmatism, which the Petzval sum does not control -- so no m2 flattens
   both foci.
2. **The flat-medial m2 6.0 is a coincidence of the as-is medial and does
   not survive the solve**: its solved section is 2x worse everywhere
   (edge 677 um, smile 12.8 px, plate drifted to 321 mm).  Do not pursue.
3. **m2 3.0 (the Petzval zero) is the best of the three, modestly**: edge
   spot 343 -> 291 um (-15 %), the defocus term 20.8 -> 15.8 um AND the
   best-focus term 26.8 -> 23.0 um (both moved -- the power split changed
   the astigmatism too, as the addendum asked to be said loudly), smile
   2.63 px, keystone 0.49 px; but SRF 5.9 -> 9.7 px.  M2 grows to R 120 mm.
4. **For the spectrometer**: the dispersion runs along-track, where the
   focus is flat -- the SRF is not where the curvature lands; the CRF
   (along the slit, x) is, and it sits at ~15.5 px on every row.  The edge
   wall is the cross-track astigmatism of the section; that is
   non-symmetric-DOF territory (CCMac), with m2 3.0 as the better parent.

**Why SRF rose 5.9 -> 9.7 px at m2 3.0 (TO).**  Not the centre field (the
worst SRF is the strip EDGE in both rows: 5.85 px at m2 3.5, 9.68 px at
3.0; inner fields 3.4-5.6 px), not the plate (333.8 vs 334.1 mm), not the
slit sampling of a changed smile (the edge SRF is flat over lambda,
9.1-9.7 px).  The edge's RMS width along the dispersion is IDENTICAL in both
rows (SV 5.07 px): the SRF is a FWHM of the convolved line profile, so the
rise is the edge profile's SHAPE (same second moment, wider core -- the
astigmatic edge spot re-oriented by the power split), not more blur.  Not
chased further.  CRF at m2 3.0 is better at every inner field (1.4-2.0 vs
1.8-4.1 px) and the same 15.5 px at the edge.

**Addendum 43 closed (TO side):** the edge wall is the cross-track
astigmatism of the eccentric section -- non-symmetric DOFs (CCMac) -- and
m2 = 3.0 is the modestly better parent for that work (-15 % edge, both the
focus and the best-focus terms moved).

## Addendum 44 (TO, 2026-10-05): the freeform ladder on the strict merit -- 1.5k

Records `dyson5_tA_FF_1k5*` (ladder, R4 re-converged, R4 under LM), end to end
`dyson5_t5f_FF_R4c*` / `_R4lm*` (engine join, `dyson5_t5f.m`), gate
`dyson5_t5f_gate_m30*`.

**Construction (the way around the emitter's refused asphere+Zernike co-emit):**
Surface=FreeForm on M1-M3 with two channels on one surface -- the Mon channel
about the parent VERTEX carries R0's even asphere exactly (unnormalized ANSI
1/5/13/25; R0' == R0 to 7.8e-15 m, the sign-flipped leg 7 cm off) and is
held; the FF channel about the section POLE carries the freeform modes.
(The Telescope emitter wrote pMon = the parent vertex on sections until CC's
f9f93e6 -- CCMac's step-5 and the 7cd8b24 re-run are vertex-centred,
superseded.)  Strict merit about the chief's detector intercept + the
position rows; residuals in um (in metres lsqnonlin stopped at the first
Jacobian -- 0258faa's R1 and 427b60b's cp2 S1 are superseded).

| rung | edge spot bf / as placed | centre spot bf | edge T-S | solver |
|---|---|---|---|---|
| R0 (B1 aspheric) | 291 um | 29 um | -2.22 mm | -- |
| R1 {4,6} | 230 | 124 | -1.60 | TRR, cap 1500 |
| R2 +{7-10} | 220 | 110 | -1.57 | TRR, cap |
| R3 +{5,13,25} | 192 | 93 | -1.11 | TRR, cap |
| R4 +{11,12,14,15} | 190 | 88 | -1.07 | TRR, cap |
| R4 re-converged | 185 / 225 | 84 | -0.96 | TRR 8000: optimality 1.7e6 (a crawl) |
| **R4, LM + Jacobian scaling** | **170 / 218 um (9.4 px)** | **73** | **-0.79** | LM 4009 evaluations: optimality 1.4e5, cost -16 %, still moving |

**End to end (engine join, roll 180, `size:D:130`; the join's identity gate
reproduces t5e's row on the aspheric deck to every printed digit):**

| deck | smile | keystone | CRF | SRF (centre) | plate local / edge | admitted |
|---|---|---|---|---|---|---|
| B1 aspheric (t5e) | 2.63 | 0.49 | 15.5 | 9.7 (5.6) | 334.1 / 334.6 | 1.000 |
| R4 re-converged | 0.68 | 0.35 | 7.7 | 13.6 (11.3) | 324.5 / 330.1 | 1.000 |
| **R4 LM** | **0.82** | **0.33** | **7.7** | **9.2 (8.6)** | **324.1 / 330.3** | **1.000** |

Freeform halves the spatial blur (CRF 15.5 -> 7.7 px) and kills most of the
smile (2.63 -> 0.82 px) by flattening the field; the price is spectral
resolution at the CENTRE (5.6 -> 8.6 px) -- the field-flattening trade in the
dispersion direction.  Whether a smaller position-row weight or a
centre-weighted field set recovers the centre SRF is NOT yet tested (Jim's
spec is at the centre too); the better-conditioned LM solve already recovered
part of it (11.3 -> 8.6 px).  **Clearance: carried over from B1 (+0.39 mm,
PASS), NOT re-scored** -- the engine join has no chain body model; the
vertices and poles are unchanged and the FF sag over the lit patches is at
most 19 / 167 / 78 um (p-v 31 / 299 / 125 um) on M1 / M2 / M3, from an
ANSI evaluator checked against the engine (M1 hit shift, corr 0.998).

## Addendum 44 (TO, 2026-10-05): the freeform ladder -- 3k

Section -4 deg / 180 mm on the m2 = 3.0 telecentric parent (F/1.7135
calibrated to 330 mm, clear +4.1 mm; `dyson5_tA_pz_3k_m30`), R0 = its B1
aspheric rung (`dyson5_tA_EP_3k_m30`), the same two-channel FreeForm, strict
merit + position rows, **Levenberg-Marquardt + ScaleProblem 'jacobian' from
the start**, 3000 evaluations per rung (every rung at the cap).  Records
`dyson5_tA_FF_3k*`; end to end `dyson5_t5f_FF_3k_R4*`; the engine-join gate
on the 3k aspheric deck `dyson5_t5f_gate_3k` == `dyson5_t5e_tA_EP_3k_m30_B1`
to every printed digit.

| rung | best-focus spot per field, centre / +-1.56 / +-3.13 / +-4.69 deg (um) | strict@chief field-rms / max (um) | edge T-S | coef. norm |
|---|---|---|---|---|
| R0 (B1) | 68 / 132 / 486 / 1086 | 65.4 / 111.0 | -8.9 mm | -- |
| R1 {4,6} | 470 / 388 / 289 / 806 | 47.7 / 73.2 | -6.0 | 524 um |
| R2 +{7-10} | 460 / 382 / 290 / 794 | 46.0 / 68.8 | -6.3 | 574 |
| R3 +{5,13,25} | 350 / 342 / 286 / 729 | 36.9 / 52.8 | -5.6 | 622 |
| R4 +{11,12,14,15} | **190 / 192 / 174 / 485** | **19.1 / 29.2** | **-2.8** | 753 |

(1.5k for the same columns: R0 29/56/141/291 um, field-rms/max 16.7/27.9 ->
R4 under LM 73/77/94/170 um, 7.7/11.5 um, norm 161 um.)

**Readings.**  (1) R1 on BOTH modules is field FLATTENING by trade, not
astigmatism removal -- on 3k the centre pays x7 (68 -> 470 um) for 26 % of edge;
CC's field-quadratic prediction ("R1 moves the 3k edge even less") does not
hold.  (2) On 3k the ladder has NOT stalled: R4's {11,12,14,15} buy -33 % of
edge (729 -> 485 um, 26.9 px) and the field is flat to 172-193 um inside
+-3.13 deg; the coefficient norm grows WITH the edge improving (not the
Tikhonov signature yet).  The next order (R5) is in play on 3k.  (3) Every rung
at the cap; first-order optimality ~1.6e6 -- not converged.

**End to end (engine join, roll 180, `size:F:240`):**

| deck | smile | keystone | CRF | SRF | plate local / edge | grating admits | swath on slit |
|---|---|---|---|---|---|---|---|
| B1 aspheric (t5e == t5f) | 7.71 | 1.20 | 15.7 | 11.8 | 328.1 / 321.7 | 0.982 | 100 % |
| **R4 freeform** | **4.57** | **0.76** | **15.1** | **15.1** | **317.1 / 330.9** | **0.984** | **99.5 %** |

Smile and keystone improve; CRF stays at the edge-blur level (485 um is still
27 px); SRF worsens (the same field-flattening trade); the local plate scale
drifts to 317 mm (-3.9 %).  **Clearance: carried over from B1, NOT re-scored**
(+0.43 mm overall PASS, the Dyson's own FPA package vs its block face;
TELESCOPE-internal +7.35 mm with the 5 mm mount, +12.06 mm without, M1->M2 leg
vs M3).  The FF sag over the lit patches reaches 0.23 / 0.51 / 0.75 mm (p-v
0.35 / 0.89 / 1.23 mm) on M1 / M2 / M3 (ANSI evaluator vs the engine, M1 hit
shift, corr 0.998) -- under the 7.35 mm telescope-internal margin by ~10x, so
the carry-over is defended, though less comfortably than on 1.5k (max 0.17 mm).
(Corrected 2026-10-05: an earlier line of this section quoted a 1.33 mm margin
that was never measured -- written before the record was read.)

## For Dave: Freeform on the eccentric section -- what it buys and what it costs (TO, 2026-10-05)

Both ladders STOPPED (1.5k at R4 under LM; 3k at R5, whose edge gained 4 %,
inside scatter).  Construction: the telecentric Korsch section (-4 deg; 1.5k
190 mm, 3k 180 mm; m2 = 3.0), two-channel FreeForm on M1-M3 (the aspheres held
exactly in the vertex-centred Mon channel, the freeform modes about each pole),
STRICT merit about the chief's detector intercept with EQUAL field weights +
the f tan(theta) position rows, warm continuation rung by rung.  Spots in um:
best focus / AS PLACED on the detector; fields centre / mid1 / mid2 / edge
(1.5k 0 / 0.78 / 1.56 / 2.34 deg; 3k 0 / 1.56 / 3.13 / 4.69 deg).

**1.5k (size:D:130)**

| | spots best focus | spots as placed | strict@chief rms / max | e2e smile / keyst / CRF / SRF px | plate local / edge | FF norm |
|---|---|---|---|---|---|---|
| R0 (B1 aspheric) | 29 / 56 / 141 / 291 | 50 / 78 / 175 / 351 | 16.7 / 27.9 | 2.63 / 0.49 / 15.5 / 9.7 | 334.1 / 334.6 mm | -- |
| **R4 freeform (LM)** | **73 / 77 / 94 / 170** | **99 / 106 / 131 / 219** | **7.7 / 11.5** | **0.82 / 0.33 / 7.7 / 9.2** | **324.1 / 330.3 mm** | 161 um |

Coefficient norm per rung (TRR ladder): 370 / 157 / 158 / 158 um; LM re-run 161 um.

**3k (size:F:240)**

| | spots best focus | spots as placed | strict@chief rms / max | e2e smile / keyst / CRF / SRF px | plate local / edge | FF norm |
|---|---|---|---|---|---|---|
| R0 (B1 aspheric) | 68 / 132 / 486 / 1086 | 77 / 153 / 625 / 1442 | 65.4 / 111.0 | 7.71 / 1.20 / 15.7 / 11.8 | 328.1 / 321.7 mm | -- |
| **R4 freeform** (best e2e) | **190 / 193 / 175 / 487** | **216 / 232 / 234 / 690** | **19.1 / 29.2** | **4.57 / 0.76 / 15.1 / 15.1** | **317.1 / 330.9 mm** | 753 um |
| R5 freeform (best telescope edge) | 178 / 177 / 158 / 469 | 224 / 230 / 198 / 680 | 17.6 / 26.8 | 6.25 / 1.04 / 15.8 / 15.5 | 317.7 / 331.0 mm | 788 um |

Coefficient norm per rung: 524 / 574 / 622 / 753 / 788 um.  Note R5 betters
R4 on the telescope merit and spots but WORSENS the end-to-end smile and
keystone -- the strict telescope merit is not the instrument's merit, so the
3k pick is R4.

**What it buys.**  The edges: 1.5k 291 -> 170 um best focus (16.2 -> 9.4 px),
3k 1086 -> 487 um (60 -> 27 px); edge astigmatism T-S 2.2 -> 0.8 mm (1.5k),
8.9 -> 2.8 mm (3k); smile 2.63 -> 0.82 px (1.5k), 7.71 -> 4.57 px (3k);
keystone 0.49 -> 0.33 / 1.20 -> 0.76 px; CRF halved on 1.5k (15.5 -> 7.7 px).
The pupil match is untouched (admitted 1.000 / 0.98).

**What it costs.**  The CENTRE: 1.5k 29 -> 73 um, 3k 68 -> 190 um -- the equal-
weight strict merit FLATTENS the field (on both modules R1 already traded the
centre for the edge; it never removed the field-quadratic astigmatism, it
balanced it); the spectral resolution at the centre goes with it (1.5k SRF
centre 5.6 -> 8.6 px; 3k total SRF 11.8 -> 15.1 px); the 3k CRF does not move
(the edge is still 27 px); the local plate scale drifts low (1.5k 324 mm, 3k
317 mm, -2 / -4 %; the edge holds 330); FF sag up to 0.17 mm (1.5k) / 0.75 mm
(3k).  **Clearance carried over from B1, not re-scored** (the engine join has no
body model): telescope-internal +9.46 mm (1.5k) / +7.35 mm (3k) with the mount,
~10x the FF sag, so the carry-over is defended.

**The telescope's merit is not the instrument's.**  On 3k, R5 lowered the
strict telescope wavefront (rms 19.1 -> 17.6 um) and the best-focus spots, yet
the end-to-end smile, keystone and CRF all got WORSE (4.57 -> 6.25, 0.76 ->
1.04, 15.1 -> 15.8 px).  Any field weighting of the TELESCOPE's wavefront is
a proxy; the merit for the next rung should be the spectrometer's own --
smile / keystone / CRF / SRF through the join -- so the ruling below is about
what that merit weights, not only how the telescope fields are weighted.

**Two unconverged facts.**  Every rung on both modules stopped at its
evaluation cap with first-order optimality 1e5-1.6e6 against 1e-6.  Levenberg-
Marquardt with Jacobian scaling roughly halved the crawl (8 % of 1.5k edge for
4000 evaluations vs 3 % for 8000 under trust-region), and the coefficient norm
grew only with improving edges (no stalled-edge growth), so a Tikhonov term is
not yet indicated.  The numbers above are lower bounds on what this
construction can reach, not its optimum.

**THE RULING (Dave):** the field weighting.  EQUAL weights (all rows above)
flatten the strip -- edges 9.4 px (1.5k) / 27 px (3k), centres 73 / 190 um,
spectral resolution at the centre lost; a CENTRE-WEIGHTED field set, or a
smaller position-row weight, would hold the centre (and its SRF) at the edges'
expense.  Which the strip must deliver where -- Jim's pixel spec applies at
the centre as much as at the edge -- is your call, and it sets the next rung's
merit on both modules.

## Addendum 45 (TO, 2026-10-05): the transverse metric and the geometry lever -- (a), the control a0, (b)

Stage `tGM` (dyson5_run.m), records `dyson5_tA_GM_<module>_<run>.*` (+ `_rs` = report-only re-scores on the final body
model), e2e `dyson5_t5f_GM_*`, conic fits `dyson5_tA_GM_conicfit.txt`.  Every run warm from R4 (1.5k: the LM rung; 3k: the
R4 rung of `dyson5_tA_FF_3k`), the same two-channel FreeForm DOFs (3 conics + 13 FF modes x 3), LM + ScaleProblem
'jacobian', 3000 evaluations, EQUAL field weights.  Commits (resources, dev-candidate, LOCAL): c26e7c5 (tooling), 72a907d
(records + body fix), 3752678 / be5cef6 (conic fit).

**Gates, every run (in each record's head).**  (1) the warm state re-traced == its record, to 0.00 um.  (2) the rigid-body
move is applied to the deck TEXT (M2/M3 in their TElt = pole frame, about the pole; FP along its normal) and checked against
`macos.perturb` on the same element: 5.1e-7 m at the test move, 5.1e-8 m at a tenth of it -- second order, i.e. only the two
rotation compositions differ; pivot and frame agree.  (3) the LIVE clearance (below) on the R0 aspheric deck == t5e's chain
record to <= 0.01 mm on every Dyson entry and on the binding telescope pair (9.48 vs 9.46 mm).

**The residuals.**  (a) SPOT: per field every passing ray's in-plane offset from the field's centroid ON THE DETECTOR as
placed, sqrt(254/N) per field (CALIB's SPOT-row balance at beam_wt 1), + the chief's 2 in-plane position rows, um.  a0 (the
CONTROL, added because R4's strict solve had ONE rms row per field -- 28 rows for 42 DOFs): the strict reference-sphere OPD
about the chief's detector intercept as PER-RAY rows, same balance, same position rows, same seed and budget.  (b) = (a) +
M2/M3 rigid body (rx ry rz mrad, dx dy dz mm) + FP focus, warm from (a).

Spots best focus / AS PLACED, centre / edge, um; e2e (engine join re-placed per deck, roll 180) smile / keystone / CRF /
SRF px; plate local / edge mm.

| | 1.5k spots | 1.5k e2e | plate | 3k spots | 3k e2e | plate |
|---|---|---|---|---|---|---|
| R4 | 73/170 / 99/219 | 0.82 / 0.33 / 7.7 / 9.2 | 324.1/330.3 | 190/487 / 216/690 | 4.57 / 0.76 / 15.1 / 15.1 | 317.1/330.9 |
| a0 per-ray STRICT | 33/60 / 60/74 | 1.38 / 0.04 / 3.74 / 3.73 | 328.3/330.4 | 246/376 / 378/478 | **43.7** / 1.01 / 15.5 / 15.4 | 326.6/330.8 |
| (a) SPOT | 17/25 / 18/27 | 1.54 / 0.03 / 2.34 / 4.46 | 325.2/329.8 | 67/102 / 73/108 | 5.48 / 0.12 / 10.9 / 5.06 | 313.4/328.1 |
| **(b) SPOT + geometry** | **6/13 / 6/13** | **0.60 / 0.02 / 1.49 / 2.30** | 327.0/330.6 | **25/40 / 27/41** | **2.11 / 0.13 / 8.26 / 3.88** | 322.1/330.7 |

**The control is the key sentence.**  On 1.5k the per-ray ROW FORM moved most of it (R4 170 -> a0 60 um at the edge) and the
SPOT metric the rest (-> 25 um; CRF 3.74 -> 2.34, though a0's SRF 3.73 beats (a)'s 4.46).  On 3k the per-ray strict control
does NOT reproduce (a) (edge 376 um, centre WORSE than R4, smile 43.7 px): the METRIC moved 3k.  So R4 was solver-limited on
1.5k and metric-limited on 3k; neither number in the "For Dave" section was the construction's optimum.  None of these runs
converged (first-order optimality 1e5-7e6 at the cap) -- lower bounds again.

**(b) is a new layout, not a trim.**  Solved (pole frame, about the pole): 1.5k M2 rx -88 mrad, M3 rx -190 mrad (11 deg), M3
decenter -13.8 / -10.0 mm (y / z), FP +1.68 mm; 3k M2 rx +70 mrad, M3 rx -285 mrad (16 deg), M3 -31.2 / -18.1 mm, FP -13.24
mm.  Bauer's diagnostic, the FF norm (a) -> (b): 1636 -> 1499 um on 1.5k (SHRINKS 8 %: the geometry took burden), 3644 -> 4840
um on 3k (GROWS 33 %: the two fight -> (c'), the stop at M2, is indicated for 3k).

**What the surfaces are (CC's ask).**  FF channel over the LIT patch (engine hits, 7 fields, in the solved pole frame), max
|sag| / p-v um / max slope mrad, M1 M2 M3:
- 1.5k R4 11/21/0.6, 94/133/4.0, 26/53/10.0; (a) 143/283/6.5, 1314/1502/61, 286/544/38; (b) 93/176/2.9, 1453/2219/67, 166/299/23; a0 299/466/6.2, 636/957/50, 666/1190/50
- 3k R4 179/273/6.6, 367/717/32, 578/921/46; (a) 367/526/14, 1275/2150/56, 1371/2682/122; (b) 651/1231/26, 1746/3344/126, 3784/5502/173; a0 446/888/23, 2118/3960/185, 2720/2899/167
(Correction: the "For Dave" section's R4 1.5k FF sag 19/167/78 um came from a full-disc grid; over the lit hits it is 11/94/26.)
Then `dyson5_conicfit`: a conic of FREE vertex, axis, R, K (+ h4/h6 about that vertex) fitted to the mirror's real lit
surface (gate: the B1 aspheric deck reproduced to <= 0.11 um rms).  Residual after the re-fit, p-v um / slope mrad:
**1.5k (b): 11 / 0.4, 17 / 1.7, 11 / 1.1 -- in SAG an off-axis aspheric TMA**; the 1.45 mm "freeform" on M2 is mostly a
re-fitted conic (R 0.120 -> 0.090 m, axis 56 mrad).  **But optically the residual is load-bearing:** the same deck emitted as
three Surface= Aspheric mirrors from the fit (hits on the fitted surfaces to 2e-9 um) scores edge 119 um best focus (b: 13),
centre 30 (b: 6), plate error 1.7 mm, e2e smile 2.57 / keystone 0.08 / CRF 4.18 / SRF 9.08 px (b: 0.60 / 0.02 / 1.49 / 2.30).
1.7-2.8 mrad of residual SLOPE x 2 x a 0.3 m throw is ~1 mm of transverse blur -- 17 um p-v is a sag statement, not an image
statement.  The freeform deck stays the record; the asphere-only re-solve warm from the emission is the open test.  1.5k (a) 27-55 um p-v, R4 8-11 um: asphere-like too.  1.5k a0 77-359 um and every 3k
solve (a 18/219/561, a0 123/811/328, R4 75/425/272; (b) M1 41, M3 839 um, M2 no physical conic within reach) are GENUINE
freeforms.  So on 1.5k the transverse metric + geometry found an aspheric off-axis solution; the strict wavefront found a
freeform one.

**Clearance (live, every deck).**  `dyson5_t5f` now runs `spectrometer_clearance` VERBATIM on the joined engine deck: the
bundle and footprints from ENGINE rays (3 fields x 3 wavelengths, grating-stopped launches), the telescope bodies at the
engine's (moved) vertices.  Two defects found building it: (1) the engine's 41-point circular grid is a lattice clipped by the
circle -- its edge rays sit up to a spacing inside the aperture, which read the binding pair 3.4 mm optimistic; the clearance
bundle is traced on a 201 grid with the aperture edge-matched.  (2) the chain lifts a conic body onto its BASE sphere, which on
these eccentric sections drops the conic (M1's lit patch departs ~20 mm from it); the live gate puts each telescope body on the
best-fit sphere of its own lit hits and prints the residual (M1 1.8-2.7, M2 0.4-2.7, M3 0.02-1.13 mm).  On B1 the binding pair
does not move (9.48 vs 9.46), so earlier records stand on that pair.  Telescope-internal worst with the 5 mm mount:
- R4 1.5k **+6.78** mm, 3k **+6.35** (the "For Dave" carry-over from B1, +9.46 / +7.35, was 2.7 / 1.0 mm optimistic -- PASS)
- 1.5k: a0 +11.3, (a) +14.7, **(b) +30.3** (overall min +0.38 = the Dyson's own package): DEFENDED
- 3k: a0 +8.4, **(a) +0.28 mm against an M3 body residual of 0.37 mm: UNDEFENDED**; (b) +22.1 telescope-internal, overall
  **+0.06 mm** (the TelM2 -> TelM3 leg vs the Dyson's BlockFaceIn, a ray leg vs a precise body): a marginal PASS.

## Addendum 45 (TO, 2026-10-05): (c) -- the instrument's merit, one bounded rung

Warm from (b) on both modules (the better e2e of (a)/(b)), the same DOFs (conics + FF + the 13 geometry), residuals = the e2e
scorer through the engine join, RE-PLACED every evaluation: per (field, lambda) the centroid's dispersion-direction deviation
from its lambda's field mean (smile), its slit-direction deviation from its field's lambda mean (keystone), the rms ray widths
SU / SV (CRF / SRF -- the scored FWHMs are 0.01 px histograms in a +-8 px window, no FD derivative), all px, + the telescope's
7 x 2 position rows in px; equal weights; LM + jacobian scaling, 1500 evaluations (~4.5 s each; 26 iterations; NOT converged,
first-order optimality 2.7e2 / 1.5e3).  Records `dyson5_tA_GM_<module>_c.*`, `dyson5_t5f_GM_<module>_c.*`.

| | spots bf / as placed, centre / edge um | e2e smile / keystone / CRF / SRF px | plate local / edge | cost | clearance (telescope-internal / overall) |
|---|---|---|---|---|---|
| 1.5k (b) | 6/13 / 6/13 | 0.60 / 0.02 / **1.49** / 2.30 | 327.0 / 330.6 | 143 | +30.3 / +0.38 |
| **1.5k (c)** | 6/11 / 6/13 (mid fields 15) | **0.41** / 0.02 / 1.95 / 2.35 | 327.1 / 330.7 | 129 | +30.3 / +0.38 |
| 3k (b) | 25/40 / 27/41 | 2.11 / 0.13 / 8.26 / 3.88 | 322.1 / 330.7 | 2939 | +22.1 / +0.06 |
| **3k (c)** | 31/76 / 35/82 | 2.05 / **0.05** / **4.22** / 4.90 | 323.6 / 331.6 | 1490 | +21.4 / +0.06 |

The geometry barely moved in (c) (tilts and decenters equal to (b) to 0.01 mrad / mm); the figure did the work, FF norm
unchanged (1500 / 4842 um), the M2 / M3 sag and slope as (b).  **3k:** the instrument merit halves the CRF (8.26 -> 4.22 px)
and the keystone (0.13 -> 0.05) by TRADING the telescope's own spot (edge 40 -> 76 um) -- R5's lesson in the other direction:
the telescope image is a proxy, the Dyson's FPA is the merit.  **1.5k:** smile 0.60 -> 0.41, but the FWHM CRF got WORSE (1.49
-> 1.95) while its rms-width surrogate improved -- (c) as built optimises the second moment, not the FWHM the spec reads; a
FWHM-faithful residual (a smoothed-LSF width, or the ray histogram with a finite-difference step above its 0.01 px bin) is
the fix if (c) is to be the number.  **The number to Jim, as scored today:** 1.5k (b) for CRF (1.49 px, on spec) or (c) for
smile (0.41); smile / keystone spec 0.1 px is met by neither on smile.  3k (c): smile 2.05 / CRF 4.22 / SRF 4.90 px, 3k
clearance a marginal +0.06 mm (a ray leg vs the Dyson block), a genuine freeform with M2 at 1.7 mm / 126 mrad.

## For Dave: addendum 45 closed -- the decks of record, what changed the answer, what is open (TO, 2026-10-05)

**The decks of record (equal field weights throughout).**

| module | deck | merit that solved it | spots bf / as placed, centre / edge (um) | e2e smile / keystone / CRF / SRF px | plate local / edge | surfaces | clearance |
|---|---|---|---|---|---|---|---|
| **1.5k** | `dyson5_tA_GM_1k5_bAs.in` | SPOT rows (per-ray, as placed) + position rows; DOFs per mirror R, K, h4/h6, vertex, axis + M2/M3 rigid body + FP focus; warm from the conic-fit emission of (b) | **7/13 / 8/14** | **0.51 / 0.01 / 1.22 / 2.20** | 327.2 / 330.7 | **three off-axis ASPHERES** (Surface= Aspheric, no Zernike); departure from the best-fit sphere over the lit patch M1 2.1 mm (its conic), M2 0.30, M3 0.01 mm | telescope-internal **+39.1 mm**; overall +0.38 (the Dyson's own package) |
| **3k** | `dyson5_tA_GM_3k_c.in` | the e2e scorer through the join (rms-width surrogates), warm from (b); geometry IDENTICAL to (b) | 31/76 / 35/82 | 2.05 / 0.05 / 4.22 / 4.90 | 323.6 / 331.6 | genuine freeform: M2 FF 1.74 mm sag / 3.3 mm p-v / **126 mrad**, M3 3.8 mm / 5.5 mm / **172 mrad**; M2 has no physical conic within reach | telescope-internal +21.4 (M3 body residual 0.63); **overall +0.06 mm FLAG** (the TelM2 -> TelM3 ray leg vs the Dyson's BlockFaceIn) |

Neither run converged (first-order optimality 1e5-1e6 at the cap): lower bounds.  Spec: smile and keystone < 0.1 px, CRF
< 1.5 px.  1.5k meets CRF and keystone, misses smile (0.51); 3k misses all but keystone.  The 1.5k freeform (b)
(`dyson5_tA_GM_1k5_b.in`: 6/13 um, 0.60 / 0.02 / 1.49 / 2.30 px, M2 1.45 mm / 67 mrad of freeform) is superseded by the
asphere deck, which beats it on every e2e number.

**What changed the answer, in order.**
1. **The residual's FORM (the a0 control).**  R4's strict solve had ONE rms row per field (28 rows, 42 DOFs).  The same
   strict wavefront as PER-RAY rows took the 1.5k edge 170 -> 60 um; on 3k it did not help (376 um, smile 43.7 px).
   R4 was solver-limited on 1.5k, metric-limited on 3k.
2. **The METRIC.**  SPOT (as-placed transverse) per-ray rows: 1.5k edge 60 -> 25 um, 3k 487 -> 102 um.
3. **The GEOMETRY.**  M2/M3 tilts and decenters + FP focus as DOFs, clearance live: 1.5k 25 -> 13 um, CRF 2.34 -> 1.49;
   3k 102 -> 40 um.  Not a trim: M3 tilts 11-16 deg, decenters up to 31 mm.  Bauer's diagnostic: the FF norm shrank 8 % on
   1.5k (the geometry took burden) and GREW 33 % on 3k (the two fight).
4. **The instrument's merit (c).**  3k CRF 8.26 -> 4.22 px by trading the telescope's own spot (40 -> 76 um).  On 1.5k the
   FWHM CRF rose (1.49 -> 1.95) while its rms surrogate fell: (c) optimises second moments, the spec reads FWHMs.  A
   FWHM-faithful residual -- a smoothed-LSF width with an analytic derivative -- is the next merit item.
5. **Asphere-only, after the conic fit.**  See the lesson below.

**The conic-fit lesson (sag vs slope; score the emission).**  A conic with free vertex/axis/R/K + h4/h6 fits the 1.5k (b)
mirrors to <= 17 um p-v -- "an off-axis asphere" in SAG.  Emitted as such and SCORED, it gave edge 119 um and CRF 4.18 px:
1.7-2.8 mrad of residual slope x 2 x a 0.3 m throw is ~1 mm of blur.  I called it buildable before scoring it and retracted
it.  Re-OPTIMISED as aspheres from that emission (above) it beats the freeform.  The fit is a starting point, the engine's
image is the verdict.  Two tool traps recorded in `dyson5_conicfit`'s header: fit in the engine's own (explicit) asphere form
-- an implicit form missed by 7 mm at large |K|; and state slope beside sag.

**Clearance method change.**  `dyson5_t5f` now runs `spectrometer_clearance` verbatim on the joined ENGINE deck (bundle and
footprints from engine rays on a 201-point grid with the aperture edge matched -- the 41-point lattice's ragged edge read the
binding pair 3.4 mm optimistic), telescope bodies at the engine's moved vertices, lifted onto the BEST-FIT sphere of each lit
patch (the chain lifts onto the base sphere, which drops the conic: M1's lit patch departs ~20 mm from it) with the residual
printed.  Gated: == the chain record to <= 0.01 mm on the B1 deck (binding pair 9.48 vs 9.46).  Corrections: the R4
telescope-internal clearance is +6.78 (1.5k) / +6.35 (3k) mm, not the +9.46 / +7.35 carried over from B1; the R4 1.5k
lit-patch FF sag is 11 / 94 / 26 um, not the full-disc 19 / 167 / 78.

**Open, for your ruling.**
- **3k layout: (c') the stop at M2** (or a different section), first order only.  The geometry and the freeform fight on 3k,
  and M2 carries 1.7 mm at 126 mrad with no physical conic within reach -- more rungs on this form are not indicated.
- The asphere-only solve on **3k** (the 1.5k result says try it; the 3k conic fits say it is less likely to hold).
- A FWHM-faithful e2e residual before (c) is the number to Jim.
- The field-weighting ruling (still equal weights).
- Pushes: everything above is LOCAL on resources dev-candidate (c26e7c5 .. this commit).
