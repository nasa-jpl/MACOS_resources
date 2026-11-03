# singlet — best-form singlet: the bending that minimizes spherical aberration

A singlet of given focal length, f-number and glass can still be *bent*: the two radii trade at constant power, and the spherical aberration depends on the split.  Coddington's shape factor X = (c1 + c2)/(c1 − c2) names the bending (+1 convex-plano with the curved face toward the object, 0 equi-convex, −1 plano-convex the other way round), and thin-lens third-order theory puts the minimum at X = −2(n² − 1)p/(n + 2), with p = (m + 1)/(m − 1) the position factor of the conjugates (p = −1 for a distant object; Kingslake & Johnson, *Lens Design Fundamentals*, 2nd ed., Sec. 4.3; Smith, *Modern Optical Engineering*, 4th ed., Sec. 6.3).  The runner builds the plano singlet the benches use (both ways round), the textbook best form, then lets the engine find the bending — MACOS CALIB on both radii from an equi-convex start — and compares the two shape factors over a bending scan.

The lens is a `macos.design.Lens`; its deck comes from `macos.design.lens_emit`, its first-order layout from `macos.design.lens_pupils`, and its score from `macos.design.lens_eval`.  These are the IODC 2010 evaluator's tools generalized to any lens.  The test `tLensSinglet` holds them to that evaluator, which is gated against an independent ray tracer.

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    singlet                                   % the defaults
    out = singlet('FNUM', 4, 'GLASS', 'N-SF6', 'OBJECT_MM', 300)

Choices (name-value): `EFL_MM` 100, `FNUM` 5, `GLASS` 'N-BK7' (any name in `macos_f90/macos_glass_list.txt`), `LAMBDA_NM` 587.5618, `THICK_MM` 4, `OBJECT_MM` Inf, `NGRID` 63, `MODEL` 128, `NSCAN` 41 (0 skips the scan), `OUTDIR` `./out`.  The run writes `out/singlet_report.txt`, `out/singlet_bending.png`, `out/singlet.mat` and the decks it scored.  It takes about 20 s.

**Metric** (stated before the numbers): RMS wavefront error in waves over the full aperture, on the exit-pupil sphere centred on the chief ray's image point (`macos.opd_ref('chief')`), piston removed, tilt and focus kept, image plane at the lens's own paraxial focus.  "Best focus" moves the image plane to the RMS minimum by CALIB on the image plane alone.

**CALIB settings.** The bending solve uses `OptMaxItrs= 3000`, OptRayGrid 31, both radii free (VarElt DOF 7), and the image plane held at the start's paraxial focus.  CALIB moves radii, not curvatures, and the constant-power valley is curved in radius space, so the LM takes many short steps: 721 iterations at the defaults, under 2 s, stopping on its own floor rule, not the cap.

## What a run gives

N-BK7 at f = 100 mm, f/5, 4 mm thick, 587.56 nm, object at infinity:

| lens | X | R1 (mm) | R2 (mm) | RMS, paraxial focus | RMS, best focus |
|---|---|---|---|---|---|
| convex-plano (curved face to the object) | +1 | 51.680 | ∞ | 1.4006 | 0.3536 |
| plano-convex (flat face to the object) | −1 | ∞ | −51.680 | 5.4548 | 1.4074 |
| textbook best form | 0.7397 | 59.235 | −395.886 | 1.2941 | 0.3269 |
| CALIB from equi-convex | 0.7393 | 59.249 | −395.225 | 1.2941 | 0.3269 |

The engine's bending agrees with Coddington's to 0.0004.  The best form beats the bench's convex-plano lens by only 8 %, so the add_lens orientation is already close; turning that lens around costs a factor of 3.9.  Balancing focus cuts every lens's RMS by the third-order factor of 4 (3.96 measured).  With the object at 300 mm the textbook form moves toward equi-convex (X = 0.2466), and CALIB follows it (0.2489).

## Source

Written 2026-11-01 with the template (lens templates beat, PLAN_CONSOLIDATION 1.4); the numbers are the run above.  The README is the template's own: edit it with the code.
