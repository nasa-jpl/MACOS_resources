# triplet — Cooke triplet: the first lens that carries a field

A singlet or a cemented doublet corrects the axis.  Off axis, their field is set by astigmatism and the Petzval curvature, which no bending removes.  H. Dennis Taylor's triplet (Cooke, 1893) spreads the power over a positive–negative–positive train with air spaces.  The separation lets the negative element flatten the Petzval sum while the outer pair keep the power, and the near-symmetry about the central stop cancels coma, distortion and lateral color to first order.

The runner:

1. solves the textbook **thin-lens seed** (Kingslake & Johnson, *Lens Design Fundamentals*, 2nd ed., Ch. 12; Smith, *Modern Optical Engineering*, 4th ed., Sec. 12.6): outer crowns of equal power, a central flint at the stop, two equal air spaces, chosen for the total power, zero axial color (Σ y²φ/V = 0) and a chosen Petzval sum (Σ φ/n = PETZVAL/f), with lateral color and distortion vanishing by the symmetry;
2. builds the **thick seed**: crowns equi-convex, flint equi-concave, at their centre thicknesses, scaled to the EFL;
3. runs **MACOS CALIB** in alternating rounds, one on the six radii, then one on the axial positions of surfaces 3–6, at three fields (0, 0.7 and 1 times the semi-field, d light), with the image plane held and the EFL restored after every round;
4. scores the field: RMS wavefront error and distortion from 0 to the semi-field.

**Why alternating rounds.**  The engine cannot take a radius DOF and a position DOF on the *same* element in one solve: `calib_run` fails at its first step (the evidence is in `~/dev/MACOS_sandbox/calib_pist_roc_2026-11-01/`, queued as an engine item).  Its WFE merit also does not charge lost rays, so a free spacing can collapse an air gap onto its neighbour and "win" on the few rays left (measured: 1412 of 1437 rays lost, reported WFE ≈ 0).  So each round is **accepted** only when all four checks hold:

- every axial gap clears `FLOOR_MM` (a spacing round's gap below the floor is clamped to it first, the IODC 2010 rule);
- every edge thickness at the clear radius is positive;
- the full field loses no ray;
- the merit has gone down, measured independently by `macos.design.lens_eval` (RMS over the CALIB fields).

These are the lessons of `challenges/iodc2010`.  Its other lesson stands too: CALIB has no distortion row, so distortion is **reported, not fitted**.

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    triplet                                          % the defaults
    out = triplet('HALF_FIELD', 15, 'FNUM', 4)

Choices (name-value): `EFL_MM` 100, `FNUM` 5, `HALF_FIELD` 20 (deg), `CROWN` 'SK16', `FLINT` 'F4' (Taylor's pair), `PETZVAL` 0.2, `T_MM` [5 2 5], `LAMBDA_NM` 587.5618, `CAL_FIELDS` [0 0.7 1], `N_CYCLES` 3, `FLOOR_MM` 1.0, `NGRID` 45, `MODEL` 128, `OUTDIR` `./out`.  The run writes `out/triplet_report.txt`, `out/triplet_field.png`, `out/triplet.mat` and its decks.  It takes about 12 s.

**Metric** (stated before the numbers): per field, RMS wavefront error in waves at `LAMBDA_NM` over the full paraxial entrance pupil, on the exit-pupil sphere centred on the chief ray's image point (`macos.opd_ref('chief')`), piston removed, tilt and focus kept, image plane fixed; distortion is 100 (yR − yP)/yP against the paraxial chief height.  A lost ray is reported.

**CALIB settings.** `OptMaxItrs= 3000` per round, OptRayGrid 21, `OptFEX= Yes`, fields 0/14/20°, three cycles.  Rounds stop on the LM floor rule after 14–99 iterations.

## What a run gives

SK16 / F4 / SK16, f = 100 mm, f/5, ±20°, d light:

- **Thin-lens seed:** f_a = f_c = 45.97 mm, f_b = −24.90 mm, air spaces 11.40 mm.
- **Thick seed:** 12.5 waves on axis, 244 at 20°.  The equi-convex shapes are only a starting point.
- **CALIB:** two of six rounds were accepted.  Both radii rounds improved the lens.  The first spacing round wanted the flint thinner than the floor and crossed faces at the clear radius when clamped; later spacing rounds did not lower the merit.  So the spacings are the scaled seed's.  Merit 143.4 → 1.009 waves.

| field (deg) | 0 | 4 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|
| RMS (waves) | 0.532 | 0.578 | 0.740 | 1.035 | 1.278 | 1.164 |
| distortion (%) | 0 | −0.023 | −0.091 | −0.194 | −0.314 | −0.413 |

R = [62.83, −105.69, −29.94, 57.75, ∞ (−3.4e6), −28.67] mm, axial spacings [5.55, 7.49, 2.22, 7.49, 5.55] mm, stop 14.15 mm behind the first vertex.  The solve finds Taylor's form unaided: a biconvex front crown, a biconcave flint and a plano-convex rear crown.  Distortion stays under half a percent across the field without being fitted.

## Source

Written 2026-11-01 with the template (lens templates beat, PLAN_CONSOLIDATION 1.4); the numbers are the run above.  The README is the template's own: edit it with the code.
