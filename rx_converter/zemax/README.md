# zmx2macos — Zemax OpticStudio (.zmx) to MACOS prescription (.in)

A sibling of `../codev/cv2macos.seq`.  It reads a sequential `.zmx` file as OpticStudio
writes it and writes a MACOS deck in the current Rx dialect, ready for the engine, pymacos
and mmacos.  Python 3 with numpy only; it needs no Zemax.

    python3 zmx2macos.py design.zmx [-o design.in] [--base-units m|mm] [--keep-dummies]
                         [--local-frames] [--grid N] [--aperture A] [--glass NAME=INDEX]
                         [--glass-list FILE] [--exit-pupil]

| option | meaning |
|---|---|
| `-o` | output deck (default: the input name with `.in`) |
| `--base-units` | `BaseUnits=` of the deck; default `mm` for a `UNIT MM` file, else `m` |
| `--keep-dummies` | keep surfaces with no glass, aperture or stop, as `Reference` elements |
| `--local-frames` | write each element's 6×6 `TElt` from its Zemax frame (Matt Wette's `-e l`) |
| `--grid` | `nGridpts=` (default 65) |
| `--aperture` | `Aperture=` in base units: a diameter, or for a point source a full angle in rad |
| `--glass NAME=INDEX` | a fixed index for a glass the engine catalog lacks; `INDEX 1` makes it air |
| `--glass-list` | the engine catalog (default `$MACOS_GLASS_LIST`, else `macos_glass_list.txt` beside the script, else `../macos/macos_f90/macos_glass_list.txt` beside this repository) |
| `--exit-pupil` | insert the exit-pupil pair (`FP_return`, `ExitPupil`) before the FocalPlane, located by the engine's FEX recipe; see [The exit pupil](#the-exit-pupil---exit-pupil) |

The run prints its notes and warnings, and writes them as comments at the head of the deck.
A refusal prints `REFUSED:` with the reason, exits with status 2, and writes no file.

## What is converted

- **The file.**  UTF-16 LE with a BOM and CRLF (OpticStudio's own format), UTF-16 BE,
  UTF-8 or ASCII, detected from the BOM or the NUL bytes.  A blank line inside a surface
  block (OpticStudio writes one after `FIMP`) does not end the block.  The surface list
  ends at `BLNK` (current files) or `CONF` (2003 files).
- **The header.**
  - `UNIT` (MM, CM, IN, METER) sets the length scale.
  - `WAVM` with `PWAV` sets `Wavelen=`, the primary wavelength.
  - `NAME` and `VERS` go into the deck's comments.
  - The aperture comes from `FLOA`, `OBNA`, `FNUM` or `ENPD` (below).
- **Surfaces.**  The keys read are `TYPE CURV CONI DISZ DIAM GLAS STOP COMM PARM CLAP
  OBDC OBSC SQAP FLAP`.  Every other key (`SSID FIMP HIDE MIRR MEMA POPS SLAB`, and any
  other) is ignored and counted in one `ignored keys:` note.
- **Types.**
  - `STANDARD` is written as Conic or Flat.
  - `EVENASPH` is written as `Surface= Aspheric`: PARM 2–8 (r⁴…r¹⁶) become `AsphCoefs`
    h⁴…h¹⁶, with their sign taken along psi.
  - `IRREGULA` is converted for its decenter (PARM 1–2) only.
  - `COORDBRK` changes the frame and emits nothing.
- **Elements.**  Each optical surface gives one element:
  - `GLAS MIRROR` gives a `Reflector`.
  - Another glass gives a `Refractor` with `GlassElt=`, the engine's catalog name, plus
    `IndRef=` at the primary wavelength.  Zemax's `F_SILICA` is the catalog's `Silica`.
  - The surface after a glass, with no `GLAS`, is the exit face: a `Refractor` with
    `IndRef= 1`.
  - A surface with only an aperture, or the stop, is a `Reference`.
  - Any other surface is a dummy, and is dropped unless `--keep-dummies`.
  - The last surface is the `FocalPlane`.
  - The stop element carries `ApStop= 0 0`, so the engine aims the chief ray through the
    stop at load.
- **Apertures.**
  - `CLAP` is written as a `Circular` aperture about the vertex, and a non-zero `CLAP` minimum
    as a `Circle` obscuration.
  - `SQAP` is written as `Rectangular`.
  - `FLAP` is written as a circle of the semi-diameter.
  - `OBSC` is written as a `Circle` or `Annulus` obscuration, with its `OBDC` offset
    carried into the element's obscuration frame.
  - An aperture with an `OBDC` decenter is the off-axis part of a conic (an OAP placed by
    coordinate breaks).  The engine's aperture is centred on the parent's vertex, far from
    the beam, so it is written `ApType= None`, with a comment giving its radius and centre.
    The dst2 loader made the same choice.  An aperture that the chief ray misses is also
    written `None`, with a warning.
  - `DIAM` (the drawn semi-diameter) does not clip rays in Zemax and is not written.
- **The source.**
  - An object at a finite distance is a point source at the object point: `zSource` is
    −1 nm, with `ChfRayPos` 1 nm along the chief ray (the manual's convention for a
    diverging beam).
  - `DISZ INFINITY` on surface 0 gives a collimated beam (`zSource= 1e22`).
  - The chief ray is the on-axis field (a non-zero field 1 is warned).  It is aimed at the
    stop's vertex by a real-ray trace in the Zemax frames (below).
  - `Aperture=` depends on the header:
    - `FLOA`: the stop's semi-diameter, by a real-ray marginal solve at four azimuths
      about the chief ray; the mean is written and the spread reported.
    - `OBNA`: 2 asin(NA).
    - `ENPD`: the diameter, for an object at infinity.
    - `FNUM`: EFL / F#, with the EFL from a real near-axis ray, for an object at infinity.
- **Units.**  Lengths are scaled from the lens unit to `BaseUnits`, curvatures inversely,
  and aspheric coefficients as unit^(1−2i).  `WaveUnits` is set equal to `BaseUnits`.

## What is refused (by name: surface number and type, or the line)

- Surface types other than the four above, among them `TOROIDAL`, `ZERNSAG`, `ZERNSTAN`,
  `GRID_SAG`, `DGRATING`, `TILTSURF` and `PARAXIAL`.  The refusal lists every such surface.
- `EVENASPH` with an r² term (PARM 1; MACOS starts at h⁴), or with terms above r¹⁶.
- `EVENASPH` on a flat base.
- `IRREGULA` with tilt or aberration terms (PARM 3–8).
- `SCBD` with any non-zero value.  This undocumented per-surface key holds five values;
  all-zero lines are noted and ignored.
- A catalog glass the engine lacks, named with every surface that uses it.  The `GLAS`
  line's nd is filler for a catalog glass (OpticStudio writes `1.5 40` on `MIRROR` lines
  too), so it is never written.  Give the index with `--glass NAME=INDEX`; `INDEX 1`
  declares a drawing-only material air (Dave's ruling, 2026-11-03).  Without a glass
  catalog every glass is unchecked and refused the same way.
- `MODE` other than `SEQ`.
- An object in glass.
- A file with no aperture definition and no `--aperture`.
- `ENPD` or `FNUM` with a finite object, and `OBNA` with an infinite one, unless
  `--aperture` is given.
- Bad files:
  - an empty file, or one with no `SURF` block;
  - a surface list with no terminator, i.e. truncated mid-list;
  - an undecodable UTF-16 file (an odd byte count);
  - surfaces out of order;
  - fewer than three surfaces.

A file cut after its surface list converts, and the missing closing `MNUM` line is warned.

Warned and written:
- a model glass, written at a fixed nd;
- a curved image surface, written flat;
- a multi-configuration file: the stored configuration is converted.

## The Zemax conventions that matter

- **The frame.**  The geometry is Matt Wette's (A, b) propagation from `zmx2macos.c`
  (rev 2.2, 2003-11-04, JPL; memo of 2001-01-17), ported unchanged:
  - A rotates the current Zemax local frame into the global frame, and b is its origin.
  - Every `DISZ` moves b along A's z column.
  - A coordinate break moves b by its decenters and rotates A by its tilts.  Order flag 0
    decenters first and then tilts X, Y, Z; any other flag tilts Z, Y, X and then
    decenters.  The rotations are right-handed about the current axes.
  - The global origin is surface 1, and surface 0's thickness places the object.
- **Thickness after a mirror.**  Zemax does not turn the frame at a mirror.  The beam
  travels −z and the thicknesses that follow are negative.  The frame math needs nothing
  for this.
  - Matt kept a z-sign that flips at each mirror, used only to flip `OBDC`, and marked it
    "FIX: need to verify".
  - It is verified, and no flip is needed: on the DST deck the converted chief ray meets
    every OAP at exactly its `OBDC` centre, before and after the folds.
- **psi.**  KrElt < 0 always, and psi points toward the centre of curvature: along the
  local +z for CURV > 0, along −z for CURV < 0, along +z for a flat.  An aspheric term's
  sign follows psi.
- **Negative steps.**  The engine traces backwards only onto a Reference or the surface
  after one.  A kept dummy at the focal plane's vertex with a different tilt therefore
  costs the focal plane the rays that must step back.  On the DST deck with
  `--keep-dummies`, surface 64 costs 1459 of 3210 rays.  The default drops that surface.

## Validation (tests/, pytest)

Run with the pymacos environment (numpy, pytest, the engine):

    ../../pymacos/.venv/bin/python -m pytest tests -q

The program decks (DST, AFTA) are program files and are not committed: they become test fixtures only if Dave says they may.  Until then they are read from `$ZMX2MACOS_DECKS` (default
`~/dev/MACOS_sandbox/zemax2macos`), and their tests skip with a message when absent.

**G1, the DST deck.**  `dst2v2_aox_v5_di_2025_05_07_fiximgtilt_stop_at_dm1_20260507_Design-Locked.zmx`
is checked against the CODE V import `mmacos/templates/30_instruments/bench_ctb/dst2/dst2_F01_norm.in`.
It converts to 14 elements: 11 Reflectors, the two FPM faces and the FocalPlane.
Coordinate breaks and 24 dummies are dropped, and there are no warnings.  The results,
measured 2026-11-03:

| pair | Zemax-converted (mm) | CODE V (mm) | difference | fold angle, both (deg) |
|---|---|---|---|---|
| oap1 – dm1 | 2258.871400 | 2258.871400 | 0 | 5.00000 |
| dm1 – dm2 | 1000.000000 | 1000.000000 | −3.4e-13 | 0 |
| dm2 – oap2 | 1414.213562 | 1414.213562 | −2.3e-13 | 5.00000 |
| oap2 – oap3 | 2667.000000 | 2667.000000 | −9.1e-13 | 0 |
| oap3 – apod | 667.326803 | 667.326803 | −4.5e-13 | 10.00000 |
| apod – oap4 | 743.303437 | 743.303437 | −2.3e-13 | 10.00000 |
| oap4 – fpm_front | 1350.000000 | 1350.000000 | −4.5e-13 | 7.40633 |
| fpm_front – fpm_rear | 5.070000 | 5.070000 | −7.4e-14 | 0 |
| fpm_rear – oap5 | 671.592847 | 671.592847 | −4.5e-13 | 7.40633 |
| oap5 – oap6 | 1507.088294 | 1507.088294 | −9.1e-13 | 0 |
| oap6 – oap7 | 1270.000000 | 1270.000000 | −6.8e-13 | 0 |
| oap7 – oap8 | 895.144042 | 895.144042 | −3.4e-13 | 0 |
| oap8 – fpa | 762.000000 | 762.000000 | −5.7e-13 | 14.17547 |

- **Fold angles** agree to 1.1e-16 rad.
- **Rigid-body fit.**  The best rotation and translation of all 14 vertices and normals
  leaves under 1e-9 m and 1e-9 rad.
- **Kr and Kc** agree to 1e-9 relative: K = −1, and Kr = −5000, −3048, −2286, −2700, −1350,
  −1270, −1270 and −1524 mm.
- **The FPA tilt agrees.**  The 2× difference reported in the assessment came from Zemax
  surface 64 ("fpa"), which sits between two −7.0877° breaks and so carries half the tilt.
  The image surface 66 carries the full 14.1755°, the same as the CODE V fpa.
- **Trace.**  The deck traces 3210/3210 rays at RMS 5.54e-7 mm, λ/1040 at 575 nm.
  The CODE V import traces 0.517 on the same grid.
- **Aim.**  The engine's `ApStop` re-aim leaves the converter's chief ray unchanged.
- **Cross-trace.**  The converter's tracer and the engine agree on the focal plane to
  1.8e-12 mm.
- **Aperture.**  `Aperture=` is 0.0179369 rad, the mean of four azimuths.  The y-azimuth
  marginal, 0.0089856 rad, matches the CODE V deck's half-angle 0.0089855; the OAPs stretch
  the beam across the fold, so the x azimuths differ by 0.4 %.  The mean is kept (Dave's
  ruling, 2026-11-03): compare a CODE V deck's half-angle with the y-azimuth value, not
  with half the written `Aperture=`.
- **Options.**  `--local-frames` and `--base-units m` trace identically.  `--keep-dummies`
  gives 38 elements (the patched C's count), and every ray it keeps lands where the default
  deck puts it.

**G2, the AFTA testbed deck.**  `AFTA COR V5B-2-3-5-Testbed-O ...zmx` has more than 30 X
tilts, more than 25 Y decenters and two order-1 breaks.  It is converted with
`--glass STRUCTURE=1`.  The fingerprint, measured 2026-11-03:
- 48 elements;
- 3210 rays: all traced, 3209 pass the apertures;
- RMS 2.54e-6 mm, about λ/220 at 570 nm;
- the chief ray within 0.18 mm of all 13 OAP aperture centres, which are the designer's
  round numbers;
- converter and engine agree on the focal plane to 1.8e-11 mm.

`STRUCTURE` is in none of the catalogs the converter knows.  It is the deck's
mechanical-drawing material: plates of −100 mm and back.  Written with its placeholder
index 1.5, the deck traces no ray; written as air, it traces as above.  Without `--glass`
the deck is refused, naming STRUCTURE and its six surfaces (82, 83, 91, 92, 108, 134):
the CLI exits 2 and writes no file.

**G3, must-fail.**
- The program decks are UTF-16 LE as saved.  A hand-written deck converts to the same text
  from ASCII, UTF-16 LE (with and without a BOM), UTF-16 BE and UTF-8.
- `TOROIDAL`, `ZERNSAG`, `ZERNSTAN`, `GRID_SAG`, `DGRATING`, `TILTSURF`, `PARAXIAL` and
  `BINARY_2` are each refused by surface number and type.  The CLI exits 2 and writes no
  file.
- These are refused:
  - a DST file cut inside its surface list, and one cut at an odd byte;
  - an empty file, and a header with no surfaces;
  - `MODE NSC`, an r² EVENASPH term, a non-zero `SCBD`, and a missing aperture;
  - a glass the catalog lacks: the AFTA deck without `--glass STRUCTURE=1`, and a
    hand-written one (accepted with `--glass NAME=1.6`, written as that fixed index).

**Also tested:**
- **Frames.**  The coordinate break against the textbook rotation matrices, for both
  order flags.  An IRREGULA decenter equals a decenter break, the surface, and the break
  that returns.
- **EVENASPH, lens.**  A sphere plus a paraboloid's even terms (the binomial series,
  r⁴…r¹⁶) traces like the STANDARD paraboloid, in mm and in m, on a lens face (CURV > 0)
  to 2e-7 mm.  The 6.5e-8 mm measured comes from the engine's asphere root solve; its conic
  path agrees to 1e-14.  With the coefficients negated the rays move 0.1 mm, so the sign
  is visible.
- **EVENASPH, mirror.**  The same holds on a concave mirror (CURV < 0, psi and the signs
  flipped), and the engine focuses it like the paraboloid.
- **Glass.**  An N-BK7 singlet writes `GlassElt= N-BK7` on its front face, with nd
  1.5168, and an `IndRef 1` exit face.  It cross-traces to 1e-9 mm.
- **Apertures.**  `FLOA`, `ENPD` and `--aperture`.

**The cross-trace.**  For a deck the engine has loaded, each engine ray's hit point on
element 1, together with the source geometry, fixes the ray.  The converter's own tracer
then follows it through every Zemax surface in the Zemax frames, and the two image points
are compared.  This checks every psi, Kr sign, vertex, unit and glass the converter
writes, against the engine's reading of the deck.

## The exit pupil (`--exit-pupil`)

A Zemax file carries no MACOS exit pupil, and every wavefront read in the mmacos chain
(sensitivities, `dw_dx`, PSFs) is taken at one.  With `--exit-pupil` the converter writes
the pair that `macos.design.Telescope.add_pupil` writes, immediately before the
FocalPlane.  `nElt` grows by 2 and the FocalPlane stays last:

| element | Element | Surface | KrElt | psi | VptElt | zElt |
|---|---|---|---|---|---|---|
| `FP_return` | Return | Flat | −1e22 | −(the chief ray arriving at the FP) | the FP vertex | R |
| `ExitPupil` | Return | Conic | −R | from the EP toward the image (its centre of curvature) | the EP | R |

Both have Kc 0, the FP's IndRef, Extinc 0, ApType None, no obscurations, PropType
Geometric and `nECoord -6`.  The deck's head comments give the EP vertex, psi, R, the
tangential and sagittal crossings, their ± spread, and the skew of the probe lines.

**Where the pupil is.**  The converter uses the engine's FEX recipe (`FEXProbeCross`,
`tracesub.F`), in its own Zemax-frame tracer, so the numbers need no engine:
- Four probe chief rays: the source chief ray turned by ±5e-6 rad about two axes normal
  to it, about the object-space entrance pupil (the engine's `StopPos`).  Each probe is
  re-aimed through the stop vertex.
- Each probe's image-space line is crossed with the chief ray's.  The EP is the point on
  the chief ray at the mean crossing distance, which is the medial pupil.
- R is the chief-ray distance from the EP to the focal plane, the engine's FEX radius for
  a flat image surface.

The probes are ±5e-6 rad, not 1 arcmin, because a real ray's crossing moves as the square
of the probe angle: on the Cassegrain by 59.17 mm/rad² × θ², which is 5.0e-6 mm at
1 arcmin and 1.5e-9 mm at 5e-6 rad.

**What was measured** (`tests/test_exit_pupil.py`; each gate has a case that must fail):

| deck | EP vertex (mm) | R (mm) | against |
|---|---|---|---|
| Cassegrain (`tests/fixtures/cassegrain_example.zmx`, synthetic) | (0, 0, −492.3076923062) | 692.3076923062 | analytic −492.3076923077 and 692.3076923077 (1/s′ = 2/R₂ − 1/s): 1.5e-9 |
| DST | (1117.097534, −0.136878, 1643.447118) | 962.212798 | the engine's FEX: 1.15e-7 in the vertex and R, 1e-15 in psi |

- **The engine's FEX on the emitted deck** finds the converter's EP to 5.7e-14 mm on
  the Cassegrain and 1.15e-7 mm on the DST.  On the DST its tangential/sagittal split,
  5.2e-5 mm, is the converter's.
- **Chain-ready.**  Element nElt−1 is a Return with a finite Kr (the test in
  `+macos/private/wf_elt_auto.m`).  The deck traces the same rays at the FP as without
  the option: 3097 on the Cassegrain; 3210 traced and 3209 passing on the DST.
- **OPD at the EP.**  6.1e-13 mm RMS on the on-axis Cassegrain.  On the DST it is
  5.0994e-7 mm, against 5.5438e-7 at the FP.  With the sphere's psi reversed, the gate
  fails.
- **Without the option** the output is byte-identical to res b025646, the converter before
  it (`tests/fixtures/cassegrain_example_b025646.in`).  On every deck at hand the option
  adds the pair, the head lines and 2 to nElt, and changes nothing else.

**A far pupil.**  The AFTA testbed's image space is nearly telecentric.  Its EP is
2.61e6 mm beyond the image, and the tangential and sagittal crossings differ by 5e5 mm.
The converter writes the sphere there, toward the image, and the deck traces: 3208 rays
pass, with 2.44e-6 mm RMS at the EP.  The engine's FEX on that deck puts the vertex 108 mm
(4e-5) away, reverses psi and the sign of zElt, and then loses every ray.  That is an
engine finding, recorded here and not changed.

The next section takes the deck from here into the MACOS chain.

## Next: from the emitted .in into the MACOS chain

The deck loads as it is in macos, pymacos and mmacos for tracing, OPD maps, PSFs and
CALIB.  The design pipeline needs one step first: segmentation, the sensitivities (dw/dx,
dw/dz, dw/dgrid), laser metrology, and the compare, simulate and budget stages.  Zemax has
no exit-pupil element.  Every wavefront read in that chain takes the OPD at the element
before the focal plane, and refuses a deck where that element is an optic.  There are two
ways to add the pupil, with the same result (they agree to 1.5e-9 mm on the Cassegrain):

- **In MATLAB, for any prescription:**
  `xp = macos.design.add_exit_pupil('design.in', 'design_xp.in')`.  The engine's FEX
  locates the pupil, and the output is checked by reloading and tracing it.
- **From this converter, with no MATLAB:** `python3 zmx2macos.py design.zmx --exit-pupil`.
  The converter's own tracer locates the pupil, gated against the engine (above).

Then the stage runners carry the `.in` from stage to stage (mmacos
`design/runners/README.md`):

    seg  = run_segmentation(xp.rx_out, 'elt', 1, 'rings', 1, 'grid', 'Hex');   % the primary -> segments + Hx
    sens = run_sensitivities(seg.in, 'fov_rad', 2.9e-4);                        % dwdx / dwdz (/ dwdgrid)
    met  = run_met(seg.in, 'hx', seg.hx, 'hub', seg.seg.nseg + 1, 'jac', sens.mat);  % edge + laser truss

From the converter's route, pass the emitted deck in place of `xp.rx_out`.

The worked example is mmacos `templates/20_segmentation/zemax_cassegrain/`.  It takes the
Cassegrain of `tests/fixtures/`, with a hole in the primary, through all four steps and
writes its report.

## Not yet (phase 2)

- An aperture centred on the beam footprint, for the off-axis conics now written
  `ApType= None`.
- Off-axis fields: one deck per field, as `cv2macos` writes.
- The focal-plane origin option (Matt's `-L`).
- `TOROIDAL`, `ZERNSAG`/`ZERNSTAN` (ZernType), `GRID_SAG` (GridSrf), `DGRATING`,
  `TILTSURF` and `PARAXIAL`.
- A page in the mmacos manual's tools chapter.

## Source

Written 2026-11-03 from CCL's assessment
(`~/dev/MACOS_sandbox/zemax2macos/ASSESSMENT_zmx2macos.md`) and brief
(`macos/BRIEF_to_zmx2macos.md`); `--exit-pupil` 2026-11-04 from
`macos/BRIEF_to_zmx_exit_pupil.md`.  The frame algebra is Matt Wette's, and its validation
against Zemax's global-coordinate reports is his 2001 memo (`zmx2in/zmxmemo.ps` in the
sandbox).
