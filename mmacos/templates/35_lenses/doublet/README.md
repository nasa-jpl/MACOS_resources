# doublet — cemented achromat from its glass pair, and vendor achromats from a catalog

Two glasses of different dispersion bring two wavelengths to one focus.  With Abbe numbers V = (n_d − 1)/(n_F − n_C), the thin-lens element powers that join the F and C foci at total power φ are φ1 = φV1/(V1 − V2) and φ2 = −φV2/(V1 − V2).  What is left, the d focus off the common F/C focus, is the **secondary spectrum** f(P1 − P2)/(V1 − V2), where P = (n_d − n_C)/(n_F − n_C) is the partial dispersion (Kingslake & Johnson, *Lens Design Fundamentals*, 2nd ed., Sec. 5.3; Smith, *Modern Optical Engineering*, 4th ed., Sec. 6.5).  A cemented pair has three radii: power and color fix two, and the third, the **bending**, sets the spherical aberration.

The runner:

1. takes the powers from the Abbe numbers (catalog glasses from the engine's own table);
2. solves the **thick** lens paraxially (y-nu): EFL exact at d, the F and C foci coincident, with an equi-convex crown;
3. lets the **engine** find the bending: MACOS CALIB on the three radii with F, d and C all in the solve (`OptWaveLen=`), holding the image plane;
4. measures the focal shift at F, d and C from three engine traces, one wavelength per trace, and compares the secondary spectrum with the textbook;
5. loads a vendor achromat from the lens catalog and checks its EFL against the vendor's.

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    doublet                                         % the defaults
    out = doublet('FLINT', 'SF2', 'FNUM', 4)

Choices (name-value): `EFL_MM` 100, `FNUM` 5, `CROWN` 'N-BK7', `FLINT` 'F2', `T_MM` [6 2.5], `LAM_NM` [486.1327 587.5618 656.2725] (F, d, C), `VENDOR_PART` 'PAC095AR.14' ('' skips stage 5), `NGRID` 63, `MODEL` 128, `OUTDIR` `./out`.  The run writes `out/doublet_report.txt`, `out/doublet.mat` and its decks.  It takes about 6 s.

**Metric** (stated before the numbers): RMS wavefront error in waves at each wavelength, over the full aperture, on the exit-pupil sphere centred on the chief ray's image point (`macos.opd_ref('chief')`), piston removed, tilt and focus kept.  Every lens is scored on one image plane, the paraxial achromat's d focus, which is the plane CALIB holds.  One wavelength per trace.  Focal positions are the engine's near-axis back focal distances.

**CALIB settings.** `OptMaxItrs= 3000`, OptRayGrid 21, three radii free (VarElt DOF 7), wavelengths F, d and C.  It converges in 158 iterations.  The same solve at d alone runs to the 3000 cap.

## What a run gives

N-BK7 + F2, f = 100 mm, f/5, crown 6 mm, flint 2.5 mm:

| lens | R1 (mm) | R2 (mm) | R3 (mm) | RMS F | RMS d | RMS C |
|---|---|---|---|---|---|---|
| paraxial achromat (equi-convex crown) | 44.781 | −41.547 | −701.595 | 0.2672 | 0.0239 | 0.0497 |
| CALIB at F, d, C | 46.439 | −42.354 | −476.901 | 0.0403 | 0.0759 | 0.0509 |
| CALIB at d alone (contrast) | 47.287 | −42.144 | −403.979 | 0.0922 | 0.0056 | 0.1270 |

The paraxial achromat's F and C foci coincide.  The engine puts its d focus 0.04895 mm short of them, on three traces and exactly at the y-nu value, against 0.04993 mm from the thin-lens formula (2 %, the thickness).  The three-wavelength solve gives back a little of that color (F −0.0035, C +0.0687 mm from d) to balance the spherochromatism, and it holds the worst RMS across the band at 0.076 waves against 0.267.  Solving at d alone nulls d (0.006 waves) but separates the F and C foci by 0.088 mm and leaves C at 0.127.  A single wavelength cannot see the color, which is why the template solves at three.

## The lens catalog

`macos.design.lens_catalog(part)` returns a vendor lens as a `macos.design.Lens`, with the vendor's quoted EFL, BFL and wavelength beside it.  The table is `src/+macos/+design/lens_catalog.txt`; copy it and add your parts, or pass `'file'`.  Its header documents the format: radii in the CODE V sign convention, a cemented face entered once, and `glass` lines carrying Sellmeier rows for any glass the engine table lacks.  Every emitted deck writes `GlassCoef=` beside `GlassElt=`, so a build without a row traces the same glass.

Shipped parts (the VSG2 collimator and imager, `templates/40_benches/vsg_wip`), prescriptions from Newport's product pages (fetched 2026-11-01):

| part | radii (mm) | thicknesses (mm) | glasses | EFL paraxial / vendor | BFL paraxial / vendor |
|---|---|---|---|---|---|
| PAC097AR.14 | 455.878, −336.756, −1010.204 | 11.0, 6.5 | N-BK7 / N-SF5 | 750.10 / 750 | 741.97 / 741.8 |
| PAC095AR.14 | 140.890, −101.720, −454.867 | 14.5, 4.0 | N-BK7 / N-F2 | 250.32 / 250 | 239.88 / 240.1 |

(At 589 nm.  Newport does not say which element is which glass; the crown is the biconvex front element, as a positive achromat requires.)  On a bench:

    L = macos.design.lens_catalog('PAC095AR.14');
    b.add_optic(L, dist);                % dist to surface 1; 'reverse', true turns it round

## Source

Written 2026-11-01 with the template (lens templates beat, PLAN_CONSOLIDATION 1.4); the numbers are the run above.  The README is the template's own: edit it with the code.
