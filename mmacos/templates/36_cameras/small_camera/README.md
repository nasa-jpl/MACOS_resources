# small_camera — the small dedicated imager: a Cassegrain with a refractive field corrector

A single-purpose remote sensing camera whose telescope and instrument are one design: two mirrors, a corrector behind the primary's hole, a filter and one detector.  The relay's power is part of the camera's focal length.  That is what sets this family apart from an observatory front end feeding a `Bench` (the mmacos manual's chapter 2 thread).

The worked case is the deck beside the runner, `small_camera.in`: a 350 mm, f/10 Ritchey–Chrétien imager with a 2048-pixel line array of 13 µm pixels (3.7 µrad per pixel, a 0.44° field) and a 500–900 nm band.  These design values are the runner's defaults, and the deck is checked against them.

The design layer under it is `macos.design.cass_camera` and `macos.design.camera_emit`.  The mirrors come from `Telescope`'s closed form, or are given outright.  Relay elements are `macos.design.Lens` objects.  The y-nu trace runs through the folded train, and the deck is scored with `macos.design.lens_eval`.

The runner has five stages:

1. **The deck.**  The builder rebuilds `small_camera.in` from the deck's own numbers, and the two decks trace alike.  The stage then reports the field at the design wavelength, the clear apertures the beam needs, and whether the primary's hole clears.
2. **The glasses.**  The deck's fixed indices are matched to the engine's catalog (`macos_glass_list.txt`), and the band is traced with the matched glasses' dispersion.
3. **The bare telescope.**  This is the closed-form Ritchey–Chrétien with the same m and β and no relay, which shows what the corrector is for.
4. **The camera from its spec** (aperture, focal length, pixel, band):
   - the RC closed form;
   - the deck's corrector, scaled and in catalog glass;
   - the first-order solve that sets the focal length and puts the focus behind the filter;
   - CALIB rounds over the band and the field.
5. **The sampling**: λN/p at each wavelength.  The camera is undersampled by design, and the stage says by how much.

**Why rounds, and why "best kept".**  CALIB has no focal-length row.  Its corrector radii move the relay's magnification, so each round ends with the first-order re-solve: the two-mirror f-number and back focal distance, by two nested 1-D solves, put the focal length and the focus back on the spec.

The first re-solves shift m and undo part of the balance.  The rounds settle as the re-solve's move shrinks.  The design kept is the best one seen by an independent merit: the worst `lens_eval` RMS over the band at the CALIB fields and the full field, among designs that lose no ray beyond the shadow.  The LM merit does not charge lost rays, as the lens templates found.  A failed CALIB, a failed re-solve, or a round that loses rays ends the rounds.

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    small_camera                                          % the defaults: small_camera.in's camera
    out = small_camera('D_MM', 200, 'EFL_MM', 2400, 'PIXEL_UM', 9)

Choices (name-value):

| choice | default | meaning |
|---|---|---|
| `DECK` | `''` | the deck; empty means `small_camera.in` beside the runner |
| `D_MM` | 350 | aperture (mm) |
| `EFL_MM` | 3500 | camera focal length (mm) |
| `PRIMARY_FNUM` | NaN | primary f-number; NaN means the deck's, f/1.334 |
| `PIXEL_UM` | 13 | detector pitch (µm) |
| `NPIX` | 2048 | pixels across the field (a line array) |
| `BAND_NM` | [500 700 900] | the band; the middle entry is the design wavelength |
| `GLASSES` | `{}` | corrector and filter glasses; empty means the catalog matches |
| `CAL_FIELDS` | [0 0.5 1] | CALIB fields as fractions of the half field |
| `N_ROUNDS` | 4 | stage-4 rounds |
| `NGRID` | 33 | ray grid |
| `MODEL` | 128 | model size |
| `OUTDIR` | `./out` | output directory |

The run writes `out/small_camera_report.txt`, `out/small_camera_field.png`, `out/small_camera.mat` and its decks, including `deck_ab.in`.  The figure draws the deck from the side (`macos.view_rx`, with the axis and field-edge bundles from the engine's ray history), then a closer view behind the primary, then the field curves.  The run takes about 20 s.

**Metric** (stated before the numbers): per field and wavelength, RMS wavefront error in waves over the annular pupil.  It is measured on the exit-pupil sphere centred on the chief ray's image point (`macos.opd_ref('chief')`), with piston removed, tilt and focus kept, and the image plane fixed.  Distortion is 100 (yR − yP)/yP against the paraxial chief height on that plane.  The diffraction limit quoted is Maréchal's 0.0714 wave.

**What the A/B drops.**  The deck's three spider vanes and its 380 mm beam (overfilling the 350 mm aperture ring at the secondary's plane) are removed from the A/B copy, because `camera_emit` writes neither.  The spiders matter for the deck's diffraction image, not for these geometric numbers.

## What a run gives

The defaults: small_camera.in's camera.

- **The deck.**
  - Mirrors: R1 933.79, K1 −1.00972; R2 156.59, K2 −1.93228; separation 400.254 mm.
  - Paraxial at 700 nm: f = 3507.7 mm, with the chief height 3506.8 mm per tan θ.  The two-mirror focal length is 3136.10 mm, and the corrector multiplies it by 1.1185.
  - The paraxial focus is 0.017 mm behind the detector plane, inside the ±0.141 mm depth of focus: the focus is the RMS balance over the field.
  - Builder and deck agree to 3.4e-10 wave at 0, 0.109 and 0.217°.

  | field (deg) | 0 | 0.054 | 0.109 | 0.163 | 0.217 |
  |---|---|---|---|---|---|
  | RMS at 700 nm (waves) | 0.0133 | 0.0100 | 0.0024 | 0.0121 | 0.0295 |
  | distortion (%) | 0 | 0.094 | 0.378 | 0.866 | 1.579 |

  Clear semi-diameters at full field: secondary 26.5 mm, corrector 12.2–12.6 mm, filter 12.9–13.3 mm.  The returning beam needs a 13.45 mm hole at the primary, well inside the 46.7 mm shadow.
- **It is an RC.** The closed-form RC at the deck's own m = 6.717 and β = 0.1014 has K1 −1.00738 and K2 −1.90593, close to the deck's values.  The classical Cassegrain's K2 would be −1.822.  The deck's conics carry a small re-balance for the corrector.
- **The glasses.**  At 700 nm the fixed indices are:
  - 1.61783: BaF8 (+2.7e-6) or SK10 (+3.8e-6), a tie;
  - 1.657863: SF19 (+1.8e-4);
  - 1.535018 (the filter): LLF2, the nearest at +5.8e-4.

  In BaF8 / SF19 / LLF2 the deck's design stays diffraction-limited over 500–900 nm and the whole field; the worst case is 0.0638 wave on axis at 500 nm.  Its paraxial focus moves 0.098 mm over the band, inside the ±0.141 mm depth of focus.  Fixed indices make the deck itself monochromatic.
- **The bare RC.** At the same image heights it reaches 0.78 waves at the field edge, against 0.03 with the corrector.  The corrector flattens the field.  It also brings the distortion: 1.6 % at the edge, against 0.017 % for the bare telescope.
- **From the spec.**  The seed (RC conics and the deck's corrector at f = 3500 mm) scores 0.1395 waves.  Round 2 is kept at **0.0447 waves**: f = 3500.000 mm, f/10.000, the detector 0.127 mm behind the filter.  The design is two-mirror f/8.887 with K1 −1.01013 and K2 −1.94683, and corrector R [−43.43, 22.94, 174.34] mm.  It beats the deck's design in glass (0.0638) across the band.

  | field (deg) | 0 | 0.055 | 0.109 | 0.163 | 0.218 |
  |---|---|---|---|---|---|
  | RMS at 500 nm | 0.0380 | 0.0356 | 0.0310 | 0.0293 | 0.0447 |
  | RMS at 700 nm | 0.0040 | 0.0071 | 0.0139 | 0.0180 | 0.0099 |
  | RMS at 900 nm | 0.0139 | 0.0161 | 0.0216 | 0.0259 | 0.0214 |
  | distortion (%) | 0 | 0.076 | 0.309 | 0.710 | 1.310 |

- **Sampling.**  The pixel is 3.714 µrad, against λ/D = 2.0 µrad at 700 nm.  Q = λN/p is 0.385, 0.538 and 0.692 at 500, 700 and 900 nm, so the camera is undersampled ×5.2, ×3.7 and ×2.9 against Nyquist (Q = 2).  The optics are diffraction-limited across the band, and the camera's resolution is set by the pixel.  This is by design: a line-scan mapper trades sampling for swath.

**Off the worked case.**  `small_camera('D_MM', 200, 'EFL_MM', 2400, 'PIXEL_UM', 9)` is f/12 with the deck's corrector scaled ×0.571.  It keeps its seed (0.0987 waves): round 1 does not beat it, and round 2 loses rays beyond the shadow, which ends the rounds.  The spec path re-balances a corrector of the deck's form near its own f-number.  A new corrector shape is design work, not this runner's.

## Source

Written 2026-11-01 (the small-camera family, PLAN_CONSOLIDATION section 4 item 6; PLAN_MMACOS_MANUAL's note of 2026-10-23).  The numbers are from the run above.  The README is the template's own: edit it with the code.
