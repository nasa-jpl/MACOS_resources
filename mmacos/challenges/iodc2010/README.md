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
  untrapped (no surface miss, no TIR, no clear aperture including the stop's).
  Clear apertures are the prescription's when it gives them; otherwise they are
  sized to the footprint at the semi-field limit (the loophole, used openly) and
  the sized deck is re-verified there.  A ray lost = the field fails with reason
  `vignetting`.
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
| `iodc2010_search.m` | the CALIB spacing solve inside an R × stop scan, each candidate scaled to EFL 100 and scored |
| `iodc2010_run.m` | the runner: `evaluate` (the bundled lenses or yours), `gate`, `search` |
| `../../tests/tIodc2010.m` | the gates as a unittest class (SUITE_FAST) |

## How to run it yourself

```matlab
addpath('~/dev/MACOS_resources/mmacos/src');
run('~/dev/MACOS_resources/mmacos/challenges/iodc2010/iodc2010_run.m')   % scores the bundled lenses
R = iodc2010_run('stage', 'gate');                                        % the evaluator gate
R = iodc2010_run('stage', 'search');                                      % the CALIB ladder (search/)
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

## The design search: first ladder (2026-10-30)

`iodc2010_search.m` is the inner solve of brief steps 3–4: for each radius in a
scan and each stop semi-diameter, the seed lens is scaled to that radius, a
CALIB block is emitted (`OptTarget= WFE` at the exit-pupil Return, `OptFEX= Yes`,
the design fields 0 / 1.5 / 3°, every surface's axial position and the image
plane free), CALIB solves, the surfaces are read back, the whole lens is scaled
to |EFL| = 100 (a scale keeps the common radius and the merit's D·θ form) and
the result is scored by the evaluator.  `iodc2010_run('stage','search')` runs
it on the symmetric pair over R = 60 / 80 / 100 / 130 × a = 3.5 / 5 / 7 mm
(artifacts in `search/`, table `search/search_pair.csv`):

| R (mm) | stop a (mm) | CALIB | EFL before scaling | D (mm) | semi-field (°) | merit | limited by |
|---|---|---|---|---|---|---|---|
| 80 | 3.5 | 30 it, cap | 108.1 | 9.289 | 2.18 | **20.2** | vignetting (rays outside the stop at the field edge) |
| 100 | 3.5 | 30 it, cap | 135.2 | 7.553 | 2.23 | 16.9 | vignetting |
| 130 | 3.5 | 30 it, cap | 174.4 | 5.754 | 3.48 | 20.1 | RMS |
| 100 | 5 | 30 it, cap | 130.0 | 10.327 | 1.27 | 13.2 | vignetting |
| 130 | 5 | 30 it, cap | 187.9 | 8.150 | 2.13 | 17.3 | RMS |
| 80 | 7 | 30 it, cap | 101.9 | 18.07 | — | 0 | REJECTED: the first element's thickness solved to 0.39 mm against a 0.52 mm edge sag (surfaces cross; the glass rule) |
| 60, any; 80/5; 100/7; 130/7 | | CALIB's initial evaluation lost every ray | | | | 0 | not solved |

So one CALIB solve roughly doubles the seed (11.1 → 20.2); the published
winner is 175× further.  Three things the ladder taught, each built in or
recorded:

1. **Unconstrained PIST collapses thicknesses.**  CALIB knows no bounds, and
   the paper's loophole invites zero thickness; the rules check (positive edge
   thickness and air space at the clear aperture) rejects the crossed-surface
   solution.  A thickness floor belongs in the outer loop (re-seed, or hold the
   element thicknesses and vary air spaces only).
2. **The stop must be element-bound for CALIB.**  A header `ApStop=` is a fixed
   object-space point: as the solve moves the lens the stop's image moves away
   from it and the chief is no longer aimed through the stop.  The CALIB deck
   now writes `ApStop= 0 0` inside the stop element, which the engine re-aims
   at every trace.
3. **Open:** at R = 130 CALIB's own after-solve evaluation reads 1e37 (every
   ray lost) at the two off-axis fields while the evaluator traces those fields
   and passes them to 3.5°; the solve then only saw the axis.  To be run down
   before the search is trusted off axis (OptFEX's probes, or the fixed-D grid
   about a re-aimed chief, are the suspects).

## Next

The outer loop: sign patterns and element counts (the paper's nine lens forms
per element; the winner's intermediate-image form), a thickness floor, the
stop position as a variable, and item 3 above; then the merit boundary
mapped on the (D, θ) plane against the published entries (the paper's
Fig. 20).
