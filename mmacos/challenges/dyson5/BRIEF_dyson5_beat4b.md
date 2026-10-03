# dyson5 beat 4b -- the slit-loss factor (addendum 9), the global meniscus search, decks re-emitted (2026-10-01, TO lane on Fable)

## The slit-loss factor: four one-knob tests, table first

`dyson5_slitloss_tests.m` -> `dyson5_s2l_tests.txt` (36 um slit, far-field
sandwich to the grating plane at 0.7 m, F/1.8 acceptance, model 1024).  The
closed form is the exact integral of sinc^2(w sin(theta)/lambda) over
|sin(theta)| <= 1/(2F) relative to |sin(theta)| <= 1 (not the asymptote).

| knob | 380 nm engine / sinc^2 | 2500 nm engine / sinc^2 | what moved |
|---|---|---|---|
| (1) grid 255 / 511 / 1023 pts (window doubles with the grid) | 0.30 / 1.10 / 1.26 | 1.36 / 1.37 / 1.37 | the 380 nm ratio follows the WINDOW, not the grid |
| (2) window 1.11 / 2 / 4 x the acceptance (255 pts) | 0.30 / 1.025 / 1.26 | 1.36 / 1.37 / 1.38 | at 2 x the acceptance the 380 nm engine = sinc^2 to 2.5 % |
| (3) slit length 0.15 mm, strip vs circle acceptance | 1.26 vs 1.66 | 1.37 vs 1.46 | the circle counts the SHORT modelled slit's own x-spread; the strip is the long-slit definition.  2 and 5 mm rows: window < acceptance, INVALID |
| (4) normalise to the PROPAGATING region |sin theta| <= 1 | 1.03 / 1.05 / 1.07 | 0.97 (255 pts, 7 x) / 0.92 (1023 pts, 29 x) | the long-wavelength excess is energy the planar FFT assigns to |sin theta| > 1 |

**Resolution.**  The 0.30 at 380 nm was ALIASING: the record's window was
1.11 x the acceptance and the diffracted tail folded back inside.  The 1.36
at 2500 nm was NORMALISATION: the engine's planar far field carries 0.7-1.3 %
of the energy at spatial frequencies beyond 1/lambda (|sin theta| > 1,
non-propagating), and the loss fraction counted it.  With a window >= 2 x the
acceptance and the propagating normalisation the engine reads 1.04 / 1.14 /
0.90 / 0.86 x sinc^2 at 380 / 700 / 1440 / 2500 nm (`dyson5_s2l.txt`); the
remaining few percent at the long end is the pitch across an acceptance only
four sidelobes wide (0.97 at the finest pitch tried).  **Slit diffraction past
the grating is 0.3 % at 380 nm and 1.8 % at 2500 nm; the sinc^2 numbers are
the record's.**  Engine note for CC (minor): the FarField kernel could zero
|f| > 1/lambda; today a wide window silently carries evanescent energy into
any energy-fraction metric.

## Decks on disk (addendum 9 item 3)

The twin stage had re-emitted the rung deck it ran on WITHOUT apertures,
overwriting `dyson5_s3_r3.in` and the R4 deck.  Fixed: the twin writes its
own `<tag>_s2w_<rung>_src.in` (with apertures) and its far-field deck
carries the optics' apertures too; the ladder stage re-emits every rung with
apertures (this beat's s3 run), so tables, renders and clearances agree
across steps.

## The global meniscus search (addendum 8 item 2)

`dyson_r4_global.m` -> `dyson5_s3_r4global.txt`: 12 starts (the record's R4
plus 11 stratified random seeds over vertex 0.24-0.60 m, thickness 2-40 mm,
both curvatures +-8 /m, either sign), lsqnonlin 40 iterations each with
every R4 variable free and the ladder's operands.

| start | merit | meniscus | engine keystone / smile / CRF / EE |
|---|---|---|---|
| R4 of record (bounded solve) | 1.915 | 4 mm plate at 240 mm, c 0.5 /m | 0.0026 / 0.0051 / **1.327** / **0.759** |
| 1 (record's R4 continued, bounds open) | **1.401** | 2.3 mm plate at 245 mm, c 0.32 /m | 0.0019 / 0.0041 / 1.482 / 0.663 |
| 2 | 1.433 | 2 mm plate at 313 mm, c 0.78 | -- |
| 12 | 1.437 | 29 mm meniscus at 247 mm, c -3.8 / -3.4 | -- |
| 7 | 1.611 | 31 mm meniscus at 437 mm, c -1.7 / -1.6 | -- |
| 3, 4, 6, 8-11 | 5e5-1e8 | infeasible (the chain loses rays) | -- |

**Finding: a merit-weighting statement, not a better design.**  The four
feasible basins reach similar merits with different glass; the lowest merit
buys 0.0007 px of keystone -- already 40x under Joe's 0.1 px -- with
+0.15 px of CRF and -0.10 of ensquared energy, because the 10 x quadratic
distortion term keeps paying for distortion that no longer matters.  **The
R4 of record stands (CRF 1.33 px, EE 0.76).**  For the native optimize the
operands change form: smile and keystone become HINGE WALLS at half the spec
(0.05 px) and the blur is what is minimised -- Dave's rule from the Keysight
room (walls on iterates, weights do not make walls) and `feedback_wall_scale`.

## Next

Native optimize (CALIB, the pixel-unit operands as walls + blur); R5's fold
prism under the clearance gate with the cold-shield height as the parameter;
beat 5's telescope at the EMIT parameters.  Then addendum 10's three
future-work items once the design of record is stable.

