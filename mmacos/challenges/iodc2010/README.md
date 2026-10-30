# iodc2010 — the IODC 2010 lens design problem, "the green lens"

The 2010 International Optical Design Conference problem (R. C. Juergens,
*The 2010 IODC lens design problem: the green lens*, Proc. SPIE 7652, 76522I,
doi 10.1117/12.871174): a 100 mm focal length lens in which **every spherical
surface has the same radius magnitude** (either sign, or plano), one glass
(Schott N-BK7), one wavelength (532 nm), scored by the product of the entrance
pupil diameter and the semi-field over which it stays diffraction-limited.
37 entries from eight countries; the published winner is Rob Bates at
81.30 mm × 43.50° = **3537**.

This directory is our entry: the **evaluator** (the problem's own metric,
argued from its definition and gated against an independent ray trace), the
prescription format, the runner, and the designs we score with it.  The first
deliverable is the evaluator (`BRIEF_to_iodc_challenge.md`, 2026-10-26 ruling);
the design search follows on it.

## The problem, as stated (Sec. 2.1–2.2 of the paper)

| | |
|---|---|
| Focal length | magnitude 100.00 ± 0.01 mm, paraxial (+100 or −100: intermediate images allowed) |
| Wavelength | monochromatic, 532 nm |
| Glass | Schott N-BK7 only |
| Surfaces | spherical or plano, centred, rotationally symmetric; **every spherical surface has the same radius magnitude**, either sign |
| Glass constraint | positive axial and edge thickness for every element, positive axial and edge air space |
| Vignetting | none: the stop is fully filled over the whole field |
| Image quality | RMS wavefront error ≤ 0.070 wave over the field, "chief ray reference; tilt and focus not removed" |
| Distortion | ≤ 5.0 %, either sign, everywhere in the field |
| Object / image | infinity / flat, real, in air |
| Stop | anywhere (inside or outside an element or on a surface) |
| Merit | MF = D [mm] × θ [deg]; θ is where the RMS first exceeds 0.070 wave **or** the distortion first exceeds 5 %, whichever comes first |

Loopholes the paper names and allows (Sec. 5): element and edge thicknesses
and air spaces may go to almost zero; no mounting material is needed beyond
the clear aperture; no limit on length, diameter or element count (entries ran
from 105 mm to 3.6 km long, 3 to 64 elements).

## The metric, stated before any number

The 2010 paper computes its RMS "in the same manner as the 2006 IODC LDP";
the 2006 paper (Juergens & Manhart, Proc. SPIE 6342, 63421U, Sec. 4.4–4.5)
spells it out: every entry converted to CODE V, RMS wavefront error with the
rays **uniform across the paraxial entrance pupil** (no pupil-aberration
correction; internal stops fully filled), **referenced to the chief ray**
(not the centroid: tilt is kept, because removing it shifts the image and
alters the distortion), computed at the image plane the prescription gives
(focus kept), and distortion = 100 (yR − yP)/yP with yP the paraxial chief-ray
height at that plane.

Ours, term by term:

* **Wavefront reference.**  MACOS's chief-ray OPD at a FEX-placed exit-pupil
  Return: the sphere is centred on the chief ray's image-plane point with its
  vertex at the exit pupil (`notes/opd_reference.md`, `notes/pupil_fex_stop.md`).
  Piston is removed (the population RMS about the mean); tilt and focus are not.
  The RMS is taken over every ray that reaches the sphere, the chief included,
  from the per-ray optical path.  The engine's `opd()` map is the same quantity
  on the same rays except that it never writes the chief (`OPD` loops
  `iRay = 2..nRay`); the gate pins that identity to 2e-16, and the omission
  is 0.1 % of the RMS at 741 rays, 1e-4 at 7081.
* **Pupil sampling.**  A collimated source whose beam diameter is the
  **paraxial** entrance-pupil diameter D (the stop imaged into object space by
  the y-nu trace in `iodc2010_paraxial.m`), the chief ray aimed through the
  stop centre (`macos.stop`), 97 rays across the diameter = 7081 rays in the
  disc.  The RMS is a lattice sum and depends on the grid; measured on the f/14
  singlet on axis: 0.09095 / 0.09330 / 0.09439 / 0.09436 waves at 33 / 65 / 97 /
  127 across, so 97 is converged to 4e-4 relative (0.02 s per field).
* **Vignetting.**  Every launched ray must reach the image plane unclipped and
  untrapped (no surface miss, no TIR, no declared clear aperture in the beam).
  Clear apertures are the prescription's when it gives them; otherwise they are
  sized to the footprint at the semi-field limit (the loophole, used openly) and
  the sized deck is re-verified there.  A ray lost = the field fails with reason
  `vignetting`.  **The stop itself does not clip** (`p.stop_clips = false`): the
  2006 paper's Sec. 4.4 says the uniform spacing across the paraxial pupil "may
  introduce pupil aberration at any internal aperture stops, but was deliberately
  done", and "any internal aperture stops were fully filled" — i.e. no vignetting
  factors; the stop is the paraxial construct that defines D and the chief ray.
  Our first ladder (2026-10-30 a.m.) had the stop as a hard circular aperture at
  its paraxial radius, and every lens whose stop sits behind power lost a ring of
  edge rays a fraction of a percent outside it (the probe: 32 of 7081 rays at a
  pupil radius of 2.8587 mm against a beam radius of 2.8610); that convention is
  kept as a switch, and `tIodc2010` pins the difference.
* **Distortion.**  yR = the traced chief ray's height on the image plane; yP =
  the paraxial chief-ray height there from the y-nu trace (= EFL·tan θ when the
  image plane is the paraxial focus, which is the default).
* **EFL.**  Paraxial, from the y-nu trace; the engine's near-axis ray (a ray at
  1e-4 of the pupil) is read beside it as a cross-check.
* **The semi-field.**  Fields are scanned outward until the first failure, then
  bisected to 0.01°; the limit is the largest field verified to pass, and the
  report names the first criterion that fails beyond it.
* **Glass.**  N-BK7 at 532 nm by Schott's Sellmeier (n = 1.51947), the same six
  coefficients in the paraxial model, the independent tracer and the deck
  (`GlassElt= N-BK7` with its `GlassType`/`GlassCoef` row, so the deck is the
  named glass on any engine build) and in the engine table row added
  2026-10-30 (`macos_f90/macos_glass_list.txt`: N-BK7 and N-SF6, Schott data
  sheets as tabulated by refractiveindex.info; the catalog's older `BK7` row
  carries the same six numbers under the un-prefixed name).

## The evaluator's gate (how we know it is the metric)

`iodc2010_gate.m` compares the engine's wavefront with `iodc2010_reftrace.m`,
an exact skew-ray trace written from the textbook (Welford Sec. 4.2 / Spencer
& Murty: the vertex-side sphere crossing, vector Snell, the optical path to the
reference sphere), on the **same rays the engine launched**, with the sphere
FEX placed.  Gate lens: two plano-convex N-BK7 singlets facing a central stop
(R = 80 mm, EFL 100.000000, f/11), 2026-10-30, 741 rays at 33 across:

| field | RMS engine (waves) | RMS reference | max per-ray difference | first-surface crossings | sphere points | chief at the stop |
|---|---|---|---|---|---|---|
| 0° | 0.737882 | 0.737882 | 1.6e-10 wave | 3.6e-14 mm | 2.9e-14 mm | 0 |
| 2° | 0.972163 | 0.972163 | 2.3e-10 wave | 6.4e-14 mm | 5.7e-14 mm | 4e-16 mm |
| 10° | 6.310890 | 6.310890 | 2.0e-10 wave | 7.9e-14 mm | 7.3e-14 mm | 9.9e-14 mm |

The image height agrees to the digit both ways (17.766593 mm at 10°, paraxial
17.632698 → +0.76 % distortion); the engine's near-axis EFL reads 100.000000
against the paraxial 100.000000.  The gate passes at 1e-6 wave by six orders.
Re-measured 2026-10-30 p.m. on the fixed engine with the stop as the paraxial
construct: the same three rows to the printed digit (this lens's central stop
behind a plano face clips none of the paraxial pupil, so the convention change
moves nothing here); at 97 across the 10° row reads 6.441634 both ways.
Because the reference tracer takes its index from the Sellmeier row
independently, the agreement also proves the engine applied N-BK7's index
(an index error of 1e-3 would show as ~1 wave).

Sensitivity of the convention to the exit-pupil distance (the one term CODE V
and MACOS could place differently): changing the sphere radius by ±10 % about
the same centre moves the 10° RMS by ∓0.4 % (6.86 / 6.83 / 6.80 waves) and the
on-axis RMS by ∓0.1 %; FEX's tangential/sagittal split on this lens is 3 % of
the radius, so the effect on a 0.070-wave reading is below 1e-4 wave.

**The CODE V column.**  Every scored lens is also written as a CODE V `.seq`
(`iodc2010_seq.m`: the prescription in CODE V's convention, the stop as a
dummy surface, `WAV` with `NRD 32`, `DIS`, `FIR`).  We have no CODE V on this
machine; the numbers below are to be confirmed by a CODE V user (Luis) running
`symmetric_pair.seq` and reporting the RMS per field, the distortion and the
EFL — the brief asks for agreement to 1e-3 wave.

| field | MACOS RMS (waves) | CODE V RMS | MACOS distortion | CODE V distortion |
|---|---|---|---|---|
| 0° | 0.04759 | | — | |
| 0.609° | | | | |
| 1.219° | 0.06979 | | +0.011 % | |
| 1.341° | | | | |
| 1.828° | | | | |

(`symmetric_pair_report.txt` has the full field table; the `.seq` carries the
sized clear apertures as `CIR`.)

## Files

| file | what |
|---|---|
| `iodc2010_params.m` | the knobs, documented (grid, scan, tolerances, aperture sizing) |
| `iodc2010_lens.m` | a prescription: one R, a sign per surface, thicknesses, media, the stop `[z a]`, optional clear semi-diameters |
| `iodc2010_glass.m` | N-BK7 / N-SF6 Sellmeier (the engine's six numbers) |
| `iodc2010_paraxial.m` | y-nu trace: EFL, BFL, pupils, D, yP |
| `iodc2010_emit.m` | the MACOS deck (mm / nm, the double-pass tail: Return at the image, Return at the exit pupil, FocalPlane) |
| `iodc2010_eval.m` | one field: set_src_fov → stop → fex → trace → opd/opd_mask; ray count; yR |
| `iodc2010_score.m` | a lens: EFL check, field scan + bisection, aperture sizing + verification, rules, report |
| `iodc2010_reftrace.m` | the independent exact tracer (the gate's analytic) |
| `iodc2010_gate.m` | engine vs tracer on the same rays; EFL; convergence; sphere-radius sensitivity |
| `iodc2010_seq.m` | the CODE V `.seq` for the cross-check |
| `iodc2010_search.m` | the first ladder: the CALIB spacing solve inside an R × stop scan on one seed, each candidate scaled to EFL 100 and scored |
| `iodc2010_solve.m` | the inner solve: CALIB on the spacings in rounds (re-emitted from its own result), a thickness floor (active set), the stop filled on axis, scaled to EFL 100 |
| `iodc2010_forms.m` | the forms: eight shapes per element, Petzval-balanced sign patterns, seed spacings in units of R, stop places; paraxially screened and scaled |
| `iodc2010_outer.m` | the outer loop: forms → pre-screen by the seed's own merit → solve → score → the (D, θ) ladder → the map against the top ten |
| `iodc2010_run.m` | the runner: `evaluate` (the bundled lenses or yours), `gate`, `search` |
| `../../tests/tIodc2010.m` | the gates as a unittest class (SUITE_FAST) |

## How to run it yourself

```matlab
addpath('~/dev/MACOS_resources/mmacos/src');
run('~/dev/MACOS_resources/mmacos/challenges/iodc2010/iodc2010_run.m')   % scores the bundled lenses
R = iodc2010_run('stage', 'gate');                                        % the evaluator gate
R = iodc2010_run('stage', 'search');                                      % the first CALIB ladder (search/)
R = iodc2010_run('stage', 'outer');                                       % the outer design loop (search/outer/)
L = iodc2010_lens('R', 80, 'sgn', [1 0 0 -1], 't', [6 62.94 6 NaN], ...  % your own lens
                  'glass', {'N-BK7','air','N-BK7','air'}, 'stop', [37.47 3.5], 'name', 'mine');
R = iodc2010_run('lens', L, 'nGridpts', 97);
```

A run writes `<name>.in` (the deck), `<name>_report.txt` (metric first, then
the field table and the merit) and `<name>.seq` into this directory.  Nothing
ends in `exit(0)`; wrap it for batch use.

## Scores so far (2026-10-30, the evaluator's first lenses, not designs)

| lens | EFL (mm) | D (mm) | semi-field (°) | limited by | merit |
|---|---|---|---|---|---|
| `singlet_f14` — one plano-convex singlet, stop on the lens, R = 51.947 | 100.0000 | 7.000 | 1.80 | RMS | 12.6 |
| `symmetric_pair` — two plano-convex singlets about a central stop, R = 80 | 100.0000 | 9.091 | 1.22 | RMS | 11.1 |

For scale, the published 2010 top ten (Table 2 of the paper; pupil and field
after the evaluator's adjustments):

| rank | entrant | D (mm) | semi-field (°) | merit |
|---|---|---|---|---|
| 1 | Rob Bates | 81.30 | 43.50 | 3537 |
| 2 | Jeff Hoffman | 94.00 | 37.25 | 3502 |
| 3 | John Isenberg | 110.65 | 19.60 | 2169 |
| 4 | Akira Yabe | 100.36 | 19.60 | 1967 |
| 5 | Jim McGuire | 55.30 | 30.00 | 1659 |
| 6 | Alexander Epple | 56.00 | 29.00 | 1624 |
| 7 | Jon Ehrmann | 81.97 | 15.85 | 1299 |
| 8 | Eugene Olczak | 87.50 | 14.00 | 1225 |
| 9 | Chuck Hardesty | 63.20 | 17.64 | 1115 |
| 10 | George Duckett | 44.30 | 24.70 | 1094 |

The winner is f/1.23 with an intermediate image (Caldwell's form: the front
group controls the field, the rear the spherical aberration); only four
entries had one and they placed 1st, 2nd, 7th and 12th.  Twenty-one entrants
used a global optimizer; the winner did not.  The radius of the entries
follows R ≈ 4·OAL^0.6 (Fig. 15).

## The design search: first ladder (2026-10-30, re-run p.m. with the paraxial stop)

`iodc2010_search.m` is the inner solve of brief steps 3–4 on one seed: for each
radius in a scan and each stop semi-diameter, the symmetric pair is scaled to
that radius, a CALIB block is emitted (`OptTarget= WFE` at the exit-pupil
Return, `OptFEX= Yes`, the design fields 0 / 1.5 / 3°, every surface's axial
position and the image plane free), CALIB solves, the surfaces are read back,
the whole lens is scaled to |EFL| = 100 and the result is scored.
`iodc2010_run('stage','search')` runs it over R = 60 / 80 / 100 / 130 ×
a = 3.5 / 5 / 7 mm (artifacts in `search/`, table `search/search_pair.csv`;
the numbers below are the afternoon re-run on the fixed engine with the stop
as the paraxial construct — the morning table had the stop clipping):

| R (mm) | stop a (mm) | CALIB | EFL before scaling | D (mm) | semi-field (°) | merit | limited by |
|---|---|---|---|---|---|---|---|
| 80 | 3.5 | 30 it, cap | 108.8 | 9.337 | 2.80 | **26.1** | RMS |
| 130 | 5 | 30 it, cap | 176.9 | 8.292 | 3.11 | 25.8 | RMS |
| 130 | 3.5 | 30 it, cap | 178.1 | 5.995 | 4.28 | 25.7 | RMS |
| 100 | 3.5 | 30 it, cap | 136.1 | 7.598 | 3.37 | 25.6 | RMS |
| 100 | 5 | 30 it, cap | 130.0 | 10.326 | 1.27 | 13.2 | vignetting (surface misses at the field edge) |
| 60 | 3.5 | 30 it, cap | 76.5 | 12.04 | 0.16 | 1.9 | vignetting |
| 80/5, 100/7, 130/7 | | 30 it, cap | | 12.9 / 14.5 / 11.1 | 0 | 0 | RMS on axis (f/7–f/9 with two elements) |
| 60/5, 60/7, 80/7 | | 30 it, cap | | 17.2 / 24.1 / 18.1 | 0 | 0 | rays miss the lens on axis |

One CALIB solve of the spacings takes the seed from 11.1 to 26.1 (the morning's
20.2 was the hard stop's ring of lost edge rays).  The ladder taught three
things, each built in or recorded:

1. **Unconstrained PIST collapses thicknesses.**  CALIB knows no bounds, and
   the paper's loophole invites zero thickness; the rules check (positive edge
   thickness and air space at the clear aperture) rejects a crossed-surface
   solution.  The outer loop's solve now carries a floor (below).
2. **The stop must be element-bound for CALIB.**  A header `ApStop=` is a fixed
   object-space point: as the solve moves the lens the stop's image moves away
   from it and the chief is no longer aimed through the stop.  The CALIB deck
   writes `ApStop= 0 0` inside the stop element, which the engine re-aims at
   every trace.
3. **Run down (2026-10-30, same day):** at R = 130 CALIB's own after-solve
   evaluation read 1e37 (every ray lost) at the two off-axis fields while the
   evaluator passed them to 3.5°.  Reproduced in isolation: the CALIB deck's
   off-axis chief rows (`OptChfRayPos=`) started 100 mm ahead of the **seed's**
   entrance pupil, and the solve moved the first surface 31 mm upstream, past
   that source plane, so those fields started inside the lens.  The evaluator
   backs its source off further and so passed the same lens.  Two things
   followed: (a) `iodc2010_emit` now backs a CALIB deck's source off by the
   lens's whole length and starts every field's chief on that plane, and
   `iodc2010_solve` re-emits the deck from the solved lens each round (so D,
   the pupil position and the source track the lens); (b) an **engine
   finding**: inside the LM a field whose rays are all lost contributed ZERO
   rows (`OPD` zeroes `OPDMat` before its ray loop and returns its 99999d32
   sentinel in the RMS only; `funcs_app` copies the map), so the trial's
   chisq FELL and the optimizer was rewarded for losing the field (R 130: two
   of three fields lost, chisq 1.8e-9, stop 'cap').  The fix (design_optim.F
   `funcs_app`: a lost field in a value or probe evaluation sets `lOpsFail`,
   the existing failed-trial path of PLAN 3.4 rejects the trial) is in the
   engine (`notes/calib.md`, the CALIB test classes 39/39, the corpus A/B in
   the commit's record).  Measured on the same decks: through the pre-fix mex
   every solve of the outer loop died at its first LM step (`gaussj: singular
   matrix (2)` after every derivative probe lost every ray); on the fixed mex
   the same decks solve.

## The outer loop (2026-10-30 p.m.)

`iodc2010_run('stage','outer')` → `iodc2010_outer.m` (artifacts in
`search/outer/`: `outer_prescreen.csv`, `outer_solved.csv`, `outer_ladder.csv`,
every solved lens's deck / report / `.seq`, `dtheta_map.png`):

1. **Forms.**  `iodc2010_forms`: eight shapes per element (the paper's nine
   less plano-plano), the sign patterns with **equal converging and diverging
   surface counts** (the paper's Sec. 5 observation: 36 of 37 entries had a
   zero Petzval sum, and with focus not removed field curvature is what ends
   the field), seed air spaces in units of R, the stop ahead / in each air
   space / behind; `iodc2010_groups`: a converging group, a gap, a diverging
   group, a gap, a converging group of equal elements with b = a + c — power
   from the separations (why the entries are long, Fig. 15), the
   intermediate-image form among them.  Each seed is traced paraxially and
   scaled to |EFL| = 100.
2. **Pre-screen.**  Each seed's stop is shrunk until it passes on axis, then
   the seed is scored by the metric itself on a 21-across grid: its own
   D × θ.  Two and three elements with air spaces 0.2 / 0.7 / 2.5 R plus the
   groups (a, c ≤ 2, gaps 0.3–10 R): 2512 seeds in 823 s (0.33 s each), 2010
   with a nonzero merit; the top 8 per element count go on.
3. **Solve.**  `iodc2010_solve`: CALIB on every surface's position and the
   image plane, in rounds re-emitted from the solved lens (D held at the
   seed's f-number, the pupil position and the source plane tracking the
   lens), design fields 0 / θ/2 / θ with θ the seed's own semi-field; a
   thickness floor of 0.03 R glass / 0.01 R air — a round that clamps a gap is
   not accepted, both surfaces of the gap are frozen and the next round
   re-solves the rest; the stop then filled on axis; the lens scaled to
   EFL 100 and scored with apertures sized at its limit.
4. **Ladder.**  The three best re-solved at D = 4 / 7 / 12 / 20 mm × design
   field 3 / 6 / 12 / 20°.
5. **Map.**  Every scored point against the top ten (`dtheta_map.png`).

What came out (39 solved, 32 pass every rule):

| lens | elements | R (mm) | length (mm) | seed D × θ | solved D (mm) | semi-field (°) | merit | limited by |
|---|---|---|---|---|---|---|---|---|
| `f3_men+_xcp_pcv_g5_front` — meniscus, plano-convex, plano-concave, stop ahead | 3 | 14.7 | 21.9 | 3.14 × 15.8 = 49.5 | 3.139 | 23.83 | **74.8** | RMS |
| `f3_men-_xcp_pcv_g5_front` | 3 | 14.5 | 21.5 | 3.11 × 16.2 = 50.3 | 3.110 | 23.70 | 73.7 | RMS |
| `f3_bicv_men+_bicx_g6_air2` — concave, meniscus, convex; stop in the second space | 3 | 276 | 930 | 4.67 × 9.5 = 44.5 | 4.670 | 11.48 | 53.6 | vignetting |
| `f3_bicv_pcx_pcx_g3_back` | 3 | 251 | 921 | 6.03 × 13.4 = 81.0 | 6.029 | 8.74 | 52.7 | RMS |
| ladder: `f3_bicv_men+_bicx_g6_air2` at D 4, design field 12° | 3 | 146 | | | 4.000 | 12.38 | 49.5 | distortion |
| ladder: the same at D 7, design field 6° | 3 | 207 | | | 7.000 | 5.72 | 40.1 | RMS |
| `f2_pcx_vcp_g1_back` | 2 | 8.0 | 2.6 | 2.25 × 13.0 = 29.2 | 2.252 | 11.71 | 26.4 | vignetting |
| groups, best (`grp_a1b2c1_g0p3_0p3_front`) | 4 | 27 | 37 | 4.47 × 1.25 = 5.6 | 4.473 | 4.26 | 19.0 | RMS |

The best lens of the day is 75 against the published 3537 (47× short; 10th
place 1094), reached by a 22-mm, three-element, f/32 lens covering 48° of full
field.  Findings, each measured:

* **The pre-screen's own merit already sorts the field** (the seeds' D × θ
  spans 0–81 across the same patterns), and **CALIB's objective is not the
  merit**: minimizing the WFE sum at three fields lowered one seed from 81 to
  53 (`f3_bicv_pcx_pcx_g3_back`: the solve bought the design fields with the
  edge of the field).  The merit-driven loop is the (D, θ) ladder outside the
  solve; a merit-shaped inner objective (the largest θ at RMS 0.070 for a
  given D) is the next step, not a WFE sum.
* **The wide-field lenses are small, the large-pupil lenses are long**: the
  tiny f/32 forms reach 24° but fail the ladder at D ≥ 4 mm (rays miss a
  15-mm lens); the 0.9-m telephoto forms hold 4–7 mm pupils to 5–12°.  The
  published frontier joins both with many weak elements and an intermediate
  image; our group family's seeds reached merit 15 before the solve and the
  solve did not add to them (the design fields were small, the air spaces at
  the floor).  Element counts ≥ 4 with the right spacings are the open part
  of the space.
* **Absolute thickness floors wreck a solve that runs at the seed's scale** (a
  3-mm lens at EFL 100 is 1.4 mm long at the solve's EFL 45); floors are in
  units of R, and a clamped gap re-reading a round-off below its floor is
  tolerated.
* **The runner's path guard** used `exist()` on a package function, which is 0,
  so every run through the runner prepended the shared `mmacos/src` and ran
  on whichever mex that was (the pre-fix one, this afternoon); `which()` is
  the test.  The resumable stages (`stage = 'prescreen' | 'solve' | 'map'`)
  came out of re-running the solve on the kept pre-screen.

## Next

A merit-shaped inner objective (the largest semi-field at RMS 0.070 for a
given D, instead of CALIB's WFE sum at fixed fields), four and more elements
in the forms family with air spaces that reach the intermediate-image regime,
the stop position as a variable (`var_stop` exists, off by default), and
Luis's CODE V column.
