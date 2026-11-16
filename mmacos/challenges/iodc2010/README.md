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
* **A virtual exit pupil** (2026-10-31).  When the paraxial exit pupil lies
  BEYOND the image plane -- the stop inside a rear group's front focal
  distance, or ahead of a diverging front: the retrofocus form of every long
  entry -- the engine's FEX keeps the vertex at the pupil station but signs
  the radius by the station-to-image distance, so the sphere's centre lands
  on the far side of the station and every ray misses it (and the image-plane
  trace behind it dies: the first eight-meniscus trains read "vignetting" at
  a 1 mm pupil).  The evaluator checks FEX's centre against the chief's image
  point (read from the image Return, which precedes the pupil Return) and,
  when it is off, re-places the sphere as the MIRROR sphere: the same centre,
  the same radius, the vertex r before the image along the chief -- the
  geometry FEX writes for a real pupil -- and the engine evaluates its near
  crossing (`F.xp_virtual`).  The near and far crossings of one sphere differ
  by rho^2 / r, second order in the transverse aberration: 2e-7 wave at D 1 mm
  and 3.5e-4 wave (0.2 % of the RMS) at D 5 mm on the trains that found it;
  the engine on the mirror sphere matches the independent tracer to 2e-9
  wave.  `tIodc2010` pins both and the must-fail (FEX's own sphere: 741 of
  741 rays lost).  A CALIB deck with `OptFEX= Yes` has the same defect inside
  the engine (the solve dies at round 1 on such a lens); the deck's own
  sphere is now written at the mirror vertex and `iodc2010_solve(..., 'fex',
  false)` solves on it -- ON-AXIS only, since a fixed sphere is wrong for an
  off-axis chief.  The engine-side fix is queued for Dave's ruling.
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
| `iodc2010_solve.m` | the inner solve: CALIB on the spacings in rounds (re-emitted from its own result), a thickness floor (active set), the stop filled on axis, scaled to EFL 100; `'fex', false` writes `OptFEX= No` (the deck's own sphere) -- recorded, not usable: the LM comes back singular (the on-axis section) |
| `iodc2010_forms.m` | the forms: eight shapes per element, Petzval-balanced sign patterns, seed spacings in units of R, stop places; paraxially screened and scaled |
| `iodc2010_groups.m` | the groups: converging / diverging / converging groups of equal elements, b = a + c, power from the separations |
| `iodc2010_chain.m` | the chains: four and more elements as a chain of zero-Petzval pair units (an equi-convex and an equi-concave element, power from their spacing), links long enough for an intermediate image; names carry the gap values |
| `iodc2010_capacity.m` | the on-axis pupil capacity: a log-bisection of D on the evaluator (the baseline meter of the on-axis record) |
| `iodc2010_retro.m` | the retrofocus family: a weak equi-concave front, a long gap, a thick meniscus rear train with the balancing equi-convex (the paper's Figures 6-11) |
| `iodc2010_relax.m` | relax and snap: curvatures free with the spacings in a least squares around the engine's trace, then snapped surface by surface to ±R / plano at the common magnitude the free solution implies |
| `iodc2010_curv.m` | the signed curvature per surface (one R and signs, or the relaxed lens's `Rk`) -- every reader of the shape goes through it |
| `iodc2010_climb.m` | the merit-shaped inner objective: the design field bracketed on the lens's own semi-field with CALIB inside, the fields re-weighted toward the worst one (iteratively re-weighted least squares); never returns worse than its start |
| `iodc2010_outer.m` | the outer loop: forms → pre-screen by the seed's own merit → solve → score → the (D, θ) ladder → the climb → the map against the top ten |
| `iodc2010_run.m` | the runner: `evaluate` (the bundled lenses or yours), `gate`, `search` |
| `../../tests/tIodc2010.m` | the gates as a unittest class (SUITE_FAST) |

## How to run it yourself

```matlab
addpath('~/dev/MACOS_resources/mmacos/src');
run('~/dev/MACOS_resources/mmacos/challenges/iodc2010/iodc2010_run.m')   % scores the bundled lenses
R = iodc2010_run('stage', 'gate');                                        % the evaluator gate
R = iodc2010_run('stage', 'search');                                      % the first CALIB ladder (search/)
R = iodc2010_run('stage', 'outer');                                       % the outer design loop (search/outer/)
R = iodc2010_run('stage', 'outer', 'outer', struct('stage', 'climb'));    % resume: the merit-shaped climb on the solved table
R = iodc2010_run('stage', 'chain');                                       % four and six elements, the chain family (search/chain4/, search/chain6/)
R = iodc2010_run('stage', 'axis');                                        % the on-axis construction: capacities, relax and snap (search/axis/)
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
4b. **Climb** (added 2026-10-31, below).  The three best through the
   merit-shaped inner objective, `iodc2010_climb`, at their own D and at
   D = 4 / 7 / 12 mm (`outer_climb.csv`, triangles on the map).
5. **Map.**  Every scored point against the top ten (`dtheta_map.png`).

The options above are `p.outer` in `iodc2010_params.m` (set to the
production run's values on 2026-10-31; the committed `outer_prescreen.csv`
had been overwritten by a default-grid run the same afternoon and is
regenerated -- a seed's name carries its gap *index*, so the table and the
grid must match).

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

## The merit-shaped inner objective (2026-10-31)

`iodc2010_climb.m`, stage 4b of the outer loop.  The merit is a threshold
(the largest field at which every field passes), CALIB's objective a sum of
squares at fixed fields; the climb makes the solve merit-shaped from the
outside.  For a given D it brackets the **design field** on the lens's own
semi-field: each step solves the spacings (`iodc2010_solve`, D held) for
fields 0 … θ at four equal spacings, scores the solved lens by the metric
(21 rays across), and raises θ (× 1.3, or to the bracket's geometric mean)
when the lens **held** it, lowers it when not; a step starts from the last
lens that held its field (a solve asked for a field beyond reach wrecks the
lens — the first draft started every step from the previous one and one
seed of merit 81 never recovered from its first step); the lens returned is
the best over every step and the start, so the climb never returns worse
than it was given.  Per-field CALIB weights (`OptFOVWt=`, through
`iodc2010_solve`'s `fov_weights`) are wired and tested, and an iteratively
re-weighted mode (`reweight`, each field's weight × (RMS / 0.070)² per step)
exists; measured, it buys nothing on average and is off by default.

**The A/B on the three production seeds** (the same seed, D and mex; the
plain solve reproduces the 2026-10-30 numbers to the digit; final scores on
the 97-ray grid):

| seed | seed as scored | plain WFE sum (the 10-30 solve) | field bracket | bracket + re-weighting | limited by |
|---|---|---|---|---|---|
| `f3_bicv_pcx_pcx_g3_back` | 81.0 | 52.7 | 81.3 | 81.3 | distortion at 13.5° |
| `f3_men+_xcp_pcv_g5_front` | 49.5 | 74.8 | **82.5** (D 3.139 × 26.27°) | 76.5 | RMS |
| `f3_bicv_men+_bicx_g6_air2` | 44.5 | 53.6 | 52.1 | 56.7 | RMS |

Findings, each measured:

* **The bracket is the merit-shaped part.**  It returns the seed where the
  sum had lost a third of it, and beats the sum by 10 % on the lens that is
  now the best (82.5 then; 93.8 after the re-run below).  Re-weighting toward the worst field (IRLS) sent one
  solve wandering (every field over a wave) and gained on one seed only:
  the sum over fields spanning 0 … θ is already dominated by its largest
  residual.
* **A seed whose field is ended by distortion gains nothing from a
  wavefront objective** (`f3_bicv_pcx_pcx_g3_back`, 13.5° by distortion, the
  wavefront fine to 11°): CALIB has no distortion row.  A merit that counts
  distortion needs either a distortion target in the engine or a MATLAB-side
  objective around a different inner solver.
* **The pick is made on a 21-ray grid and re-scored at 97**: the third seed's
  best step (53.6 at 21) reads 52.1 at 97, below the sum's 53.6 — a 3 % grid
  noise in the pick, the price of a 12-second climb (`score_grid` raises
  it).
* **The record's pre-screen table was wrong** (found by this A/B: the climb's
  first seed was a different lens of the same name): the runner's accidental
  default-grid run of 10-30 had overwritten `outer_prescreen.csv`, and a
  seed's name carries its gap *index* under the grid in force.  The
  production options are now `p.outer`'s defaults and the table is
  regenerated; the solved, ladder and climb tables now carry each lens's
  full prescription (`t`, `stop`) so a resume needs no report on disk (a
  missing one is re-solved from its seed, deterministically).

**The production climb stage** (`outer_climb.csv`, re-run 2026-10-31 p.m.
with the two starts the chain run called for, below: the three best lenses by
the better of their seed and solved merit, each climbed from the SEED and from
the solved lens, at its own D and at D = 4 / 7 / 12 mm; 319 s; the triangles
on `dtheta_map.png`):

| lens | start | D (mm) | start D × θ | climb D × θ | merit | limited by |
|---|---|---|---|---|---|---|
| `f3_men-_xcp_pcv_g5_front` | seed | 4 | fails on axis | 4.00 × 23.45 | **93.8** | RMS |
| the same | seed | 3.11 | 3.11 × 16.21 = 50.4 | 3.11 × 26.56 | 82.6 | RMS |
| the same | solved | 3.11 | 3.11 × 23.87 = 74.2 | 3.11 × 25.56 | 79.5 | vignetting |
| the same | solved | 4 | fails on axis | nothing | 0 | RMS |
| `f3_men+_xcp_pcv_g5_front` | seed | 3.14 | 3.14 × 15.78 = 49.5 | 3.14 × 26.27 | 82.5 | RMS |
| the same | solved | 3.14 | 3.14 × 23.98 = 75.3 | 3.14 × 25.83 | 81.1 | RMS |
| the same | seed / solved | 4 | fails on axis | 4.00 × 4.41 / 4.32 | 17.7 / 17.3 | vignetting / RMS |
| `f3_bicv_pcx_pcx_g3_back` | seed | 6.03 | 6.03 × 13.48 = 81.2 | returned its start | 81.3 | distortion |
| the same | solved | 6.03 | 6.03 × 8.91 = 53.7 | 6.03 × 11.48 | 69.2 | RMS |
| the same | seed / solved | 7 | fails on axis | 7.00 × 5.08 / 5.99 | 35.5 / 41.9 | RMS |
| the same | seed / solved | 4 | 4.00 × 13.48 / 11.72 | 4.00 × 13.49 / 13.45 | 53.9 / 53.8 | distortion |
| any | any | 12 | fails on axis | nothing (one 1.2) | ≤ 1.2 | — |

The best lens on record is now **93.8**: `f3_men-_xcp_pcv_g5_front` climbed
from its *seed* at D = 4 mm (R 19.00, 29 mm long, f/25, 4.00 × 23.45°, RMS
limited; `f3_men-_xcp_pcv_g5_front_climb_seed_D4.in` / `_report.txt` / `.seq`),
where the same lens climbed from its *solved* prescription takes no 4-mm
pupil at all (the ladder of 10-30 said the same of it).  The seed start
also gives 82.6 / 82.5 at the two small lenses' own pupils and keeps the
distortion-limited 81.3 the plain solve had lowered to 53.7.  The published
winner is 3537, 38× away; the frontier the entries occupy (D 45–110 mm) is
still untouched (every lens of ours is at D ≤ 7 mm): see the chain family
below for what it needs.

## Four and six elements: the chain family (2026-10-31 p.m.)

`iodc2010_chain.m`, `iodc2010_run('stage','chain')` (artifacts in
`search/chain4/` and `search/chain6/`, the same tables and map as the outer
loop).  The full forms enumeration at four elements is ~150 000 seeds (the
balanced sign patterns × 27 spacings × 5 stop places), so four and more
elements go in through a structured family built on the one-radius problem's
own weak element: a **unit** of an equi-convex and an equi-concave element
separated by an air space u, whose Petzval sum is zero and whose net power
comes from the spacing alone, φ_unit = u R φ² with φ = 2(n − 1)/R (positive in
either order; focal length 0.93 R at u = 1, 3.1 R at u = 0.3).  Units are
chained with link gaps; a link longer than the unit's focal length puts a real
intermediate image inside it, the next unit re-images it — the relay form.
Every chain is Petzval-balanced by construction, the stop goes ahead, in any
unit gap or link, or behind, and the names carry the gap *values*
(`ch4_np3_L0p5_pn0p3_s_l1`), not indices.  Two units with u ∈ {0.1, 0.3, 1, 3} R
and links {0.5, 1.5, 4, 10} R gave 719 paraxially viable four-element seeds
(360 with an intermediate image); three units on a coarser grid (u {0.3, 1},
links {1.5, 4, 10}) gave 2824 six-element seeds (2817 with an intermediate
image).  A test leg pins the counts, the balance of each unit, |EFL| = 100,
D = 10 at the seed, and the intermediate-image flag against the unit's
first-order focal length.

What came out (the pre-screen 250 s + 1181 s; the top eight per element
count solved, the three best laddered and climbed):

| family | seeds (nonzero merit) | best seed D × θ | best solved | best on the ladder / climb | limited by |
|---|---|---|---|---|---|
| chain4 | 719 (411) | `ch4_np3_L4_np3_s_back`: 2.52 × 7.73 = 19.5 (R 3.6 m, 37 m long) | 19.8 (`ch4_np3_L0p5_np0p3_s_l1`, D 4.50 × 4.39) | **30.4** (`ch4_np3_L0p5_pn0p3_s_l1` at D 4: 4.00 × 7.61) | RMS |
| chain6 | 2824 (1437) | `ch6_np1_L1p5_pn1_L10_np1_s_back`: 1.81 × 12.97 = 23.5 (R 3.1 m, 46 m long) | 14.3 (`ch6_np1_L1p5_pn1_L1p5_np1_s_back`, D 2.32 × 6.17) | **24.4** (the same at D 2.32: 2.32 × 10.51) | RMS |

Against the three-element 82.5 (93.8 after the climb re-run below), the
chains did not move the frontier.  What the tables say, each measured:

* **No chain takes a pupil.**  The pre-screen shrinks every seed's stop
  until it passes on axis; no four- or six-element chain passed above
  D = 6.4 mm (the median 3.3 / 2.0 mm), and the ladder and the climb read
  zero at D = 12 for every chain and at most 10.7 (7.00 × 1.53°) at D = 7.  A unit's two
  elements have powers ±φ with a small net — its on-axis aberration is a
  strong telephoto's, not a weak lens's.  The three-element forms do reach
  the 10-mm target on axis (`f3_bicv_pcx_pcx_g3_back` at 6.03 mm passes
  13.4°).
* **The relay form buys field at no pupil.**  The chains the pre-screen
  ranks first are the longest (R 3–3.6 m, 23–46 m long, an intermediate image
  in a 10–40 m link): 8–13° of semi-field at D 1.8–2.6 mm.  The compact
  chains (R 300 mm, 1.2 m long) take 4.5–5.8 mm at 2.3–3.4°.
* **The plain solve wrecked the long seeds** (19.5 → 0.0, 17.2 → 0.0; two
  of eight four-element solves and seven of eight six-element solves stopped in
  the first round, clamped: the thickness floors are in units of R, 30 mm of
  air and 90 mm of glass at R = 3 m), and **the climb from the solved lens
  returned its start** at nearly every pupil (the bracket closed *below* the
  start's own field: the WFE sum at the start's own design field does not
  reproduce the start).  The climb stage therefore now starts from the
  **seed as well as the solved lens**, and ranks its candidates by the better
  of the two merits (the A/B above had already read 82.5 from the seed
  against 81.1 from the solved lens of the same name); the runner also now
  merges a struct override into `p.outer` instead of replacing it (the
  documented resume line would have dropped the production grids and
  re-named every seed).

**The climb re-run with both starts** (`outer_climb.csv` in each directory
now carries a `start` column; the candidates are the three best by the
better of the solved and the seed merit):

| directory | lens | start | D (mm) | start D × θ | climb D × θ | merit | limited by |
|---|---|---|---|---|---|---|---|
| chain4 | `ch4_np3_L4_np3_s_back` (the 37-m relay) | seed | 2.52 | 2.52 × 7.73 = 19.5 | 2.52 × 10.95 | **27.6** | vignetting |
| chain4 | the same | solved (0.0 after the plain solve) | 2.52 | 0 | 2.52 × 1.42 | 3.6 | vignetting |
| chain4 | `ch4_np3_L0p5_pn0p3_s_l1` | solved | 4 | 4.00 × 7.81 = 31.2 | returned its start | **30.4** | RMS |
| chain4 | the same | seed | 4 | 4.00 × 7.38 = 29.5 | returned its start | 28.8 | RMS |
| chain6 | `ch6_np1_L1p5_pn1_L4_np1_s_back` (plain solve: failed in round 1) | seed | 2.12 | 2.12 × 9.96 = 21.1 | 2.12 × 12.83 | **27.2** | distortion |
| chain6 | `ch6_np1_L1p5_pn1_L10_np1_s_back` (the 46-m relay; plain solve failed) | seed | 1.81 | 1.81 × 12.97 = 23.5 | 1.81 × 13.00 | 23.5 | distortion |
| chain6 | `ch6_np1_L1p5_pn1_L1p5_np1_s_back` | solved | 2.32 | 2.32 × 9.84 = 22.8 | 2.32 × 10.51 | 24.4 | RMS |
| chain6 | the same | seed | 2.32 | 2.32 × 6.17 = 14.3 | 2.32 × 10.20 | 23.7 | RMS |
| outer (production) | `f3_men-_xcp_pcv_g5_front` | seed | 4 | fails on axis | 4.00 × 23.45 | **93.8** (the new best) | RMS |
| outer | the same | solved | 4 | fails on axis | nothing | 0 | RMS |
| outer | `f3_bicv_pcx_pcx_g3_back` | seed | 6.03 | 6.03 × 13.48 = 81.2 | returned its start | 81.3 | distortion |
| outer | the same | solved | 6.03 | 6.03 × 8.91 = 53.7 | 6.03 × 11.48 | 69.2 | RMS |

The seed start recovers what the plain solve destroyed (19.5 → 0 → 27.6 from
the seed; the six-element best, 27.2, is a seed the plain solve could not
solve at all) and is within 5 % of the solved start where the solve had
helped, and on the production lenses it found the new best: the seed of
`f3_men-_xcp_pcv_g5_front` re-sized to a 4-mm pupil climbs to 93.8 where
its solved prescription takes no 4-mm pupil.  The chains' bests stand at
30.4 (four elements) and 27.2 (six),
every one of them at D ≤ 4 mm; the two six-element relays that distortion
ends (12.8–13.0° at D < 2.2 mm) join `f3_bicv_pcx_pcx_g3_back` as the lenses
a wavefront objective cannot move further.

## The on-axis construction (2026-10-31 p.m.): the pupil wall measured

Dave's order was the pupil first.  Before another family went into the loop,
the on-axis capacity -- the largest entrance pupil a lens passes on axis at
EFL 100, by a log-bisection of D on the evaluator (33 rays across, the stop
in object space so D = 2a) -- was measured on every construction the
problem's physics suggests, seeds first, then under three inner solvers.
Every number below is from the scratch runs `axis_char1..8` of that
afternoon (the scripts are not products; the constructions are described so
they can be rebuilt).

**Seeds, unoptimized** (capacity in mm; every one ends by RMS, i.e.
spherical aberration):

| construction | best | note |
|---|---|---|
| one thick equi-convex, t = 1.5 R (R 77, the near-ball) | **10.4** | 9.7 / 9.0 / 8.5 at t = 1.0 / 0.5 / 0.1 R; Petzval unbalanced |
| the forms family (pre-screen, D_target 10) | ≥ 10 (capped) | `f3_pcx_pcx_bicv` at R 41 and 82 reached the cap with RMS 0.048 |
| shells: a converger then m thick menisci placed γ R before the running cone target | 5.2 | most placements infeasible: a thick equi-convex's cone target lies closer than one radius, so no shell fits behind it; where one fits it halves the capacity |
| trains of K adjacent zero-Petzval pair units | 6.6 (K 2) | a pair-unit train's focal length is bounded below by ~0.96 R √(pitch/u) ≥ 0.96 R: at EFL 100 its R cannot exceed ~100 mm without an internal focus (K ≥ 4 adjacent units focus inside) |
| the chain relays of 10-31 (R 3.6 m, 36.5 m) | 2.5 | the beam inside is 47× the entrance pupil (unit marginal heights 1 → 4.2 → 8.9 → 46.9; stop semi-diameter 53.7 mm at D 2.5): the aberration scale is the internal beam, not D -- RMS 0.045 / 0.72 / 11.5 waves at D 2.5 / 5 / 10, the D⁴ of spherical aberration |
| the retrofocus form of the long entries: an equi-concave front, a 3-30 R gap over which the beam expands ×M, a rear train of 3-17 thick menisci + the balancing equi-convex (288 built, 108 feasible, Fig. 6-10 of the paper) | 6.4 | 5.3-6.0 for the 8-15-element trains at R 250-1040; the capacity does not grow with R because M/R stays 0.008-0.010: the rear always works at the system's own f-number on a beam ∝ R, so its aberration in waves grows ∝ R |

**Under the solvers, D held** (the stop moved behind the last surface for the
solve so the pupil is real -- on axis the wavefront does not depend on where
the stop is -- then put back):

| solver | seed | result |
|---|---|---|
| CALIB on the spacings (OptFEX, 5 rounds × 15 it) | 4-element retrofocus, capacity 6.4 | D 20: 1.18 waves on axis, the lens returned passes 2.7; D 40: 11.6 waves |
| the same, as a pupil climb in 1.2× steps | the same | 6.4 → 8.5 at the first step, then every step loses; the 8-meniscus trains clamp a gap in round 1 and never accept a round |
| bounded least squares outside the engine (`lsqnonlin` on log spacings, floors as bounds, the per-ray OPD at 21 across as the residual, the lens re-scaled to EFL 100 at every evaluation; 41 iterations, 500-840 evaluations per step) | 6- and 10-element retrofocus | at D 7: 0.57 → 0.35 waves, capacity 6.1-6.2; the solver stretches the lens (1.6 → 4.4 m) and stalls |
| CALIB with `OptFEX= No` (the deck's own sphere) | any | the LM's derivative columns all vanish (`gaussj: singular matrix (2)` after the probes): an engine item, with the FEX one |

**What this says.**  No seed family passes much above 6 mm, and no local
solve over the spacings climbs from there, because with the radii pinned to
one value the spacings are the ONLY continuous variables and no enumerated
sign pattern starts near a balanced solution: the spherical aberration of a
thick train at f/10 is tens of waves and the balance the entries reached
(f/0.9 at 110 mm for Isenberg, no intermediate image, 10-15 m, ~40 elements)
is a global-optimizer result with the curvatures RELAXED -- Sec. 4 of the
paper: a constraint function c_j (c − c_j)(c + c_j) with zeros at 0 and ±c,
weighted low so the curvatures roam, then high so they snap.  CALIB has the
radius as DOF 7 of `VarElt` but no penalty target.  The on-axis construction
therefore needs a relax-and-snap inner stage (free radii with the constraint
penalty, then the snap and a spacing re-solve), either as an engine target or
in MATLAB around the engine's trace; a spacing-only solve of any family is
measured not to reach the pupil.  Decision for Dave.

The afternoon also found the virtual-pupil defect (the metric section above):
every retrofocus seed with its stop ahead of the rear group had read
"vignetting at 1 mm" through FEX's mis-centred sphere, and CALIB had died on
them at round 1 for the same reason.

## Relax and snap (2026-10-31 evening, Dave's ruling: "Relax and snap -- yes!")

`iodc2010_relax.m`, `iodc2010_run('stage','axis')` (artifacts in `search/axis/`:
`axis_seeds.csv` = every seed's on-axis capacity, `axis_relax.csv` = each
relax at each pupil target, the snapped lenses' decks / reports / `.seq`).
The stage is a MATLAB least-squares (`lsqnonlin`) around the engine's trace,
CCL's recommendation over an engine penalty target (no engine change, no
corpus A/B; the distortion-aware objective rides the same solver):

* **Variables:** every surface's curvature κ_j, continuous, and the log of
  every gap; the lens re-scaled to |EFL| = 100 at every evaluation (the
  problem is scale-free but for the EFL), the stop re-sized to hold D.
* **Residual:** the engine's per-ray OPD on axis at D (21 rays across), plus
  the Petzval sum, soft walls on the edge thickness of every gap and on the
  beam staying inside every sphere (a ray outside the sphere is a surface
  miss, which no residual can grade), and a graded row for lost rays.
* **The free stage, then the snap.**  The free stage lets the curvatures
  roam.  The grid is then ±k_c and plano with **k_c the common magnitude the
  free solution implies** (the median |κ| of its curved surfaces): κ's units
  are arbitrary under the re-scale, so a free solution at ±0.5 IS a green
  lens of twice the radius -- the first product run snapped to 1 and moved
  every surface by half its curvature (320 waves, every ray lost).  The snap
  is SEQUENTIAL: the free curvature nearest its grid value is fixed, the rest
  re-solved, surface by surface; the entrants' cubic c_j(c − c_j)(c + c_j)
  pulled up in stages is kept as `'snap', 'pull'` for the record -- with a
  local solver it stalls 0.18 off the grid at every weight (the cubic has no
  gradient near 0.58 c).
* **Seeds:** `iodc2010_retro` (the retrofocus family above, now a product)
  and the thick equi-convex; each seed's capacity by `iodc2010_capacity`
  (the log-bisection of the previous section, now a product); the
  `p.axis.top_k` best relaxed at every `p.axis.D_targets`.

**First result** (the 6-element retrofocus seed `retro_nf1_G3_m4_t0p6_g0p3_last_rear`,
R 260, capacity 5.33, at a 20-mm target; 611 s):

| stage | RMS on axis at D 20 (waves) | R (mm) | note |
|---|---|---|---|
| seed | (fails above 5.33 mm) | 259.6 | |
| free (61 iterations) | **0.041** | 16.1 | the relaxation reaches the limit at 20 mm: a conventional small fast lens |
| k_c = 0.492 → fixes 1-5 (two planos, three +R) | 0.0176 → **0.0002** | 9.6 | the snap costs nothing for five surfaces |
| fixes 6-9 | 0.02 / 0.003 / 0.013 / 0.007 | 6-10 | |
| fixes 10-12 (the last three, the most ambiguous) | 0.77 / 0.69 / **2.24** | 53 | the last fixes break it: the free solution's remaining curvatures sit 0.4 k_c off any grid value |
| snapped `[0 0 1 1 0 1 1 1 1 0 1 0]`, capacity | | 107.7 | **10.89 mm** against the seed's 5.33 and the baseline 10.4; Petzval −0.17 (unbalanced: the weight 1 did not hold it) |

So the relaxation does what the entrants' did (0.04 wave at f/5 with six
elements of one glass), the snap recovers most of it for nine of twelve
surfaces, and the last three fixes lose it: the on-axis pupil moved from 5.3
to 10.9 mm, past the 10.4-mm baseline by a hair, not to the 20 the relaxed
lens had.

**The stage's first run** (`p.axis`: the four best seeds by capacity × D 20 /
40 / 80; 41 seeds measured in 40 s, 12 relaxes in 44 min; `search/axis/axis_relax.csv`,
the best lens's deck / report / `.seq` kept):

| seed | capacity | target D | free RMS at D (waves) | snapped RMS | snapped pattern | capacity after | R / length (mm) |
|---|---|---|---|---|---|---|---|
| thick equi-convex | 10.37 | 20 / 40 / 80 | 2.06 / 34 / 673 | 2.2 / 37 / 759 | `[1 -1]` | 10.89 (all three) | 65 / 142 |
| `retro_nf1_G3_m2_t0p3_g0p05_first_rear` (4 elements) | 6.36 | 20 | 0.069 | 1.06 | `[0 0 0 0 1 0 1 1]` | 13.12 | 84 / 153 |
| the same | | 40 | 0.86 | 10.6 | the same | 14.90 | 95 / 1229 |
| the same | | 80 | 65 | lost rays | | 0 (rejected) | |
| `retro_nf1_G3_m2_t0p3_g0p3_first_rear` | 6.23 | 20 | 16.7 | 0.56 | `[0 0 -1 -1 1 0 1 1]` | 15.34 | 102 / 359 |
| the same | | 40 | 76 | 1.25 | `[0 1 0 -1 1 1 1 1]` | **16.92** | 79 / 243 |
| the same | | 80 | 14.6 | lost rays | | 0 (rejected) | |
| `retro_nf2_G3_m2_t0p6_g0p05_last_rear` (6 elements) | 6.17 | 20 | 0.021 | 2.23 | `[0 0 0 0 1 1 1 0 0 1 1 -1]` | 10.89 | 65 / 252 |
| the same | | 40 | 10.5 | 10.0 | `[0 0 0 0 0 0 0 0 1 0 1 -1]` | 15.04 | 97 / 289 |
| the same | | 80 | lost rays | lost rays | | 0 (rejected) | |

The on-axis best is now **16.9 mm** (a 4-element lens, R 79, 243 mm long,
`retro_nf1_G3_m2_t0p3_g0p3_first_rear_relax_D40`), 63 % above the 10.4-mm
baseline and 2.7× its seed; every relax beat its seed except the singlet's
three (10.9 each, the thick equi-convex re-found).  What the table says:

* **The free stage is iteration-starved at the larger targets**: 60
  iterations reach 0.02-0.07 wave at D 20 on the 4- and 6-element seeds but
  0.9-76 waves at D 40 and 15-673 at D 80 (the sequential fixes, 25
  iterations each, then went on improving -- 16.7 → 0.56 on the third seed --
  so the solver was still descending, not stuck).  `iters_free` is the first
  knob to raise, with the pupil targets as a ladder (relax at 20, then from
  that lens at 40) rather than three cold starts.
* **The snap goes plano-heavy**: the free solution spreads its curvature
  over every surface at a fraction of k_c, and the nearest-value rule makes
  the small ones plano -- the 6-element seed at D 40 came back with two
  curved surfaces of twelve.  A choice at each fix (all three values, the
  best kept) and a k_c chosen to maximize the number of curved surfaces that
  snap cleanly would keep the elements working.
* **D 80 is out of reach of the snap at this element count**: the relaxed
  lenses lose rays there and so do the snapped ones.  The entries at that
  pupil have 10-40 elements; the stage's seeds have 1-6.
* The Petzval weight of 1 did not hold the balance (the best pattern has
  five converging surfaces and one diverging); a hard count balance at the
  snap belongs with the choice above.

**The second run** (2026-10-31 evening to 11-01, 5 h in the capped scope;
Dave's "Push. Then TO start next"): the snap as a CHOICE (a curvature more
than 0.15 k_c off its nearest grid value tries both neighbouring values and
keeps the better re-solve) under a HARD Petzval count balance (a value the
surfaces still free could not balance is refused), the seeds picked per
element count (1 / 4 / 6 / 8 / 10 / 12), each pupil target relaxed from the
previous target's snapped lens, 120 free iterations.  Every capacity
re-measured with the edge rule in the meter (unchanged; `axis_relax.csv`
carries `capacity_edge`, `rules_at_capacity`):

| seed (elements) | seed capacity | D 20 → 40 → 80 rungs, capacity after (mm) | best | run 1 (unbalanced, no ladder) |
|---|---|---|---|---|
| thick equi-convex (1) | 10.37 | 8.87 → 8.95 → 6.36 | 8.95 | 10.89 |
| `retro_nf1_G3_m2_t0p3_g0p05_first_rear` (4) | 6.36 | 9.22 → 9.97 → 9.97 | 9.97 | 14.90 |
| `retro_nf2_G3_m2_t0p6_g0p05_last_rear` (6) | 6.17 | 3.30 → 5.28 → 8.87 | 8.87 | 15.04 |
| `retro_nf2_G3_m4_t0p3_g0p05_last_rear` (8) | 5.94 | 13.91 → **26.80** → 26.80 | **26.80** | not relaxed |
| `retro_nf1_G3_m8_t0p3_g0p3_last_rear` (10) | 5.49 | 11.00 → 15.49 → 13.24 | 15.49 | not relaxed |
| `retro_nf2_G3_m8_t0p3_g0p05_last_rear` (12) | 5.99 | 9.68 → 4.60 → 6.36 | 9.68 | not relaxed |

The on-axis best is now **26.8 mm**, a Petzval-BALANCED 8-element lens (R
59.7, 251 mm long, signs `[0 0 0 0 +1 0 +1 +1 +1 +1 +1 +1 +1 +1 -1 0]`: five
converging and five diverging surfaces, six planos;
`retro_nf2_G3_m4_t0p3_g0p05_last_rear_relax_D40`), 2.6× the 10.4-mm baseline,
1.6× run 1's unbalanced 16.9, 4.5× its seed.  What this run says:

* **The balance costs the small lenses and pays on the larger ones.**  The
  4- and 6-element lenses lost a third to two thirds of their unbalanced
  pupils (14.9 → 10.0, 15.0 → 8.9); the 8-element lens, which run 1 never
  relaxed (its seed measured 5.9, below the overall top four), took 26.8
  under the balance.  Element count buys the pupil, as the entries say.
* **The choice matters:** of the snap's ambiguous fixes, the two candidates'
  re-solves differed by more than 3× in 29 of 48 (the log lines carry
  both, e.g. `<-1:3.18 +0:198.2>`); the nearest value alone would have taken
  the wrong one each time.
* **The ladder stepped from worse lenses:** it took any passing lens as the
  next start, so the singlet went 8.9 → 9.0 → 6.4 and the 12-element lens
  9.7 → 4.6 → 6.4; the rule now climbs from the BEST lens so far (fixed for
  the next run).  D 80 is still unreached: every 80-mm rung returned the
  40-mm lens or worse.
* **The edge rule is in the meter.**  The scorer rejected the 26.8-mm lens
  at the 40-mm pupil by two edges 0.1 and 0.005 mm negative at that
  footprint (the relax's soft wall at weight 10 let them through; it is now
  a wall with a 0.2 % R margin), and `iodc2010_capacity` now applies the
  rule at the marginal height, so a quoted capacity is a pupil the rules
  accept; at 26.8 mm this lens passes them.
* **Cost:** 30-45 min per relax at 10-12 elements (39-47 variables, 120 +
  N × 25 iterations, two re-solves per ambiguous fix); 5 h for the run.

**The field on the 26.8-mm lens** (2026-11-01, `iodc2010_climb` at its own
pupil and below, the merit on the same meter as the 93.8 record; 44 s in all):

| D (mm) | the lens as it stands, semi-field | after the climb | merit |
|---|---|---|---|
| 26.8 | 0.03° | 0.04° (one step solved, every other `image-inside` or `calib-failed`) | 1.0 |
| 20 | 0.41° | 0.40° | 8.0 |
| 15 | 0.84° | 0.84° | 12.5 |
| 10 | 1.56° | 1.54° | **15.4** |

So the first merit from a relaxed lens is 15.4, against the 93.8 of the
3-element seed climb: the on-axis construction buys the pupil (26.8 mm) and
nothing of the field, as it was built to, and the spacing-only climb cannot
add any -- the relaxed lens's air gaps sit at 0.7-1.0 mm, 0.012-0.017 R,
on the solve's 0.01 R floor, so CALIB's first move clamps a gap and the
round is refused.  Measured, not a surprise: the field has to be IN the
relaxation.

## Next

**Parked 2026-11-01 (Dave's word, through CCL).**  The state of record is
resources a341a8c (the choice snap under the hard Petzval balance, the
ladder), 19c8d09 (run 2: the balanced 8-element lens passes 26.80 mm on axis,
the edge rule in the capacity meter) and 97f0f53 (the field on that lens:
merit 15.4 at D 10 against the 93.8 best so far); the macos tree is unchanged
since 035b18c7.  The runner's products under `search/` (442 untracked files,
the five hours of relaxes among them) are not committed; the whole folder is
copied to `~/dev/MACOS_sandbox/iodc2010_search_2026-11-01/` so the repository
re-clone does not erase them.  Resume point: the merit-shaped relax as the
next paragraph states it.  The engine's FEX for a virtual exit pupil (the
OPEN entry in `macos_f90/notes/pupil_fex_stop.md`) stays open for Dave.

The relax solver with the field in its residual: the per-ray OPD at the
design fields 0 / θ/2 / θ (and the distortion row the climb lacks), D and θ
both held, the bracket on θ outside it as `iodc2010_climb` does -- the
merit-shaped relax, which is also the distortion-aware inner objective of
the standing order.  Then the pupil again with more elements at R 60-100
(the 8-element lens's regime) and the free stage warm-started along the
ladder (now from the best lens).  Ruling needed: the
engine's FEX/SXP for a pupil beyond the image (with it, CALIB's field solves on
retrofocus lenses; the `OptFEX= No` singular LM beside it).  Then the
distortion-aware inner objective (a MATLAB-side objective around the engine's
trace serves both: the distortion-limited seeds are the ones the wavefront
solve cannot move), the stop position as a variable (`var_stop` exists, off by
default), and Luis's CODE V column after the weekend.
