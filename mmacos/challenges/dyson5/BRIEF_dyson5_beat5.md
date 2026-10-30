# dyson5 beat 5 -- the telescope that feeds the slit, and the instrument end to end

Written for: Dave (review).  Record: `dyson5_t1.txt`, `dyson5_t1.mat`,
`dyson5_t1_t<k>.in` (one deck per rung; the last is the telescope of
record), `dyson5_t1_maps_t<k>.png`, `dyson5_t1_t<k>_view3d.png` /
`_viewyz.png`; `dyson5_t2.txt`, `dyson5_t2.mat`, `dyson5_t2_r4.in`,
`dyson5_t2_r5.in`, `dyson5_t2_maps_r4.png`, `dyson5_t2_maps_r5.png`,
`dyson5_t2_r4_view3d.png` / `_viewyz.png`, `dyson5_t2_r5_...`.  Runner:
`dyson5_run(struct('stages', {{'t1','t2'}}))`.  Gate:
`tests/tTelescopeRx` (SUITE_FAST).

## 1. The question (addenda 7 and 17)

The spectrometer of record is scored from a point on its slit.  What
feeds the slit is the telescope: at EMIT's parameters (420 km, 60 m
ground sample) the pixel's 0.143 mrad fixes f = 126 mm, the Dyson's
F/1.8 fixes the aperture at 70 mm, and 3000 pixels across track fix a
24.6 deg field onto the 54 mm slit.  Dave's asks: the telescope laid out
and engine-rendered, its spot and telecentricity over the field, the
pupil match to the spectrometer's stop (the grating), then telescope and
spectrometer traced END TO END as one prescription and scored by the
spectrometer's scorer, the clearance gate across both.

## 2. The form, and three facts that shaped it

**The spectrometer is telecentric at the slit.**  The Dyson's own aim
lines (the chief from each slit point through the grating vertex, run
back out through the plate, the block and the meniscus) cross 16.84 m
behind the slit; the chief at the slit ends is 0.09 deg off the slit
normal.  So "telecentric" in the spec is the right word, and the
telescope's exit pupil target is that apparent pupil.  The pupil-match
NUMBER used throughout is the chief's miss of the grating vertex when the
telescope's own chief, as it arrives at the slit, is sent on through the
Dyson's exact chain -- millimetres on the grating, not an angle.

**The first order collapses to a one-parameter family.**  For a coaxial
positive-negative-positive three-mirror anastigmat with the stop at M2,
a flat field (Petzval sum zero) and an exit pupil at the spectrometer's
(effectively at infinity) force phi3 = 1/t2 and then the EFL condition
reads t2 = f y2 whatever the front pair does -- the EFL and the pupil
condition are the same equation in that limit, which is why a general
three-equation Newton solve has no basin.  `telescope_seed` works on that
family: choose t1 and the beam compression y2 at M2, and R1, R2, R3, t2,
t3 follow (the finite 16.84 m carried as a correction).  The Seidel
n-flip conic seed (`macos.design.seidel_seed`, validated on the Korsch
fixture) does not describe this geometry -- its paraxial EFL check reads
381 mm against the chain's exact 126.0 -- so the rungs start from
spheres and solve the conics themselves.

**The 0.7 m block cannot sit inside the telescope.**  A flat fold after
M3 turns the converging beam into the spectrometer; the telescope then
lies entirely in front of the slit plane, looking along -y (the Dyson's
axis is +z).  The unobscuring itself is Bauer's: the coaxial section
grazes a body at every field bias, fold distance and small tilt tried
(70 mm beams, spacings of the beam's size), so the chief is folded at
each mirror (32, -32, 21 deg; a fold deviates the beam by twice the
angle where a degree of bias buys 2 mm).  The flat fold is
aberration-neutral and the slit is on the folded chief by path.

The field itself is the sky line that images ONTO the straight slit: for
each angle along the slit the across-slit angle is solved so the chief
lands on the slit line.  Forcing a straight sky line at a fixed bias was
the dominant residual of the first solve (the image of that line bows by
+-12 px) and is not a requirement -- the image of a straight ground line
on the slit is curved in every push-broom, and the orthorectification
carries it (Mouroulis & Green, Sec. 5.2).

## 3. Method (the Dyson's doctrine, reused)

The exact tracer is the Dyson's (`design/src/chain_trace.m`, lifted
verbatim from `spectrometer_geom`; the R4 and R5 record decks re-emit
byte-identical through it; `tSpectrometerRx` 7/7), so one tracer serves
the telescope chain, the spectrometer chain and the end-to-end chain.
`telescope_geom` builds the telescope in the spectrometer's frame (the
centre-field chief exits through the slit along the Dyson's own aim);
`e2e_geom` prepends it to the Dyson chain with the slit as a recorded
station and the grating as the stop; `chain_bundle` / `chain_footprints`
launch collimated fields and declare apertures; `spectrometer_rx` writes
both decks (collimated source, a pass-through Reference at the slit,
per-surface margins -- the grating's F/1.8 footprint + 0.2 mm is the
stop); `spectrometer_clearance` scores the combined chain (the
telescope's mirrors and fold join the body list; the slit station is
exempt from the mask and the face it sits on).

`telescope_ladder` solves rungs by lsqnonlin on the exact chain, residuals
in pixels over the field: T0 the layout (bias, spacings, M2/M3 decentre
and tilt -- the Bauer freedoms -- and the fold's distance, with the
clearance wall dominant: a hinge per leg-body pair, the telescope's own
discs plus the spectrometer's bodies as a point cloud); T1 conics +
radii + spacings + bias; T2 + h^4, h^6 aspheres on all three mirrors;
T3 everything.  Each rung is emitted and scored in the ENGINE at the slit
(`telescope_score`): spot, slit admittance, telecentricity, pupil match,
field flatness, mapping; the identity engine == chain is the gate
`tTelescopeRx` (every ray at the slit and at the FPA to 1e-9 m, with a
convex mirror, aspheric reflectors, the fold and the collimated source on
the line).  The end-to-end decks are scored by `spectrometer_score` with a
field launch (`'fields'`): the grating declared the stop, the chain's own
launch written per (field, wavelength), and the fraction of the launched
bundle the grating admits reported per field -- the pupil match in
energy.

## 4. Result (dyson5_t1.txt, dyson5_t2.txt; engine scores)

**The layout closes; the image does not.**  Seed: t1 = 140 mm, y2 = 0.6
(R [700 125 152] mm, t [140 76 141] mm, Petzval sum 0, f 126.00), fold
angles (32, -32, 24) deg from a scan over fold angles (the coaxial
section of this family grazes a body at every bias, fold distance and
small tilt tried: with 70 mm beams and spacings of the beam's size there
is no room, which is the review's "mirrors pushed to large off-axis
angles" at F/1.8).

| rung | spot max (px rms) | pupil: chief miss of the grating vertex (mm) | field p-v (um) | ends off the slit ends (px) | clearance, full gate (mm) |
|---|---|---|---|---|---|
| T0 layout | 108.8 | 7.9 | 102 | 0.6 | +0.75 PASS |
| T1 conics + radii + spacings + bias | 69.9 | 8.5 | 1955 | 17.8 | +0.76 PASS |
| T2 + h^4, h^6 aspheres | 66.9 | 9.4 | 1594 | 5.9 | +0.75 PASS |
| T3 everything | 66.7 | 9.4 | 1587 | 5.8 | +0.75 PASS |

Telescope of record (T3): R [894.8 126.0 123.9] mm, spacings [253.9 60.9
98.0] mm, fold angles [31.9 -32.2 21.3] deg, conics [-1.05 -10.6 0.54],
the flat fold 28.9 mm before the slit; EFL by the map 126.4 mm;
telecentricity 1.1 deg at the field ends; chief identity engine vs chain
2e-14 m at every rung; the worst clearance pair is the spectrometer's own
detector package against its returning beam (+0.75 mm, R4's record).
The spot runs 63-67 px rms across the field with the best focus 34 um at
the ends and 1.6 mm at the centre: a field-curvature swing the merit
charges (the flat-field term, 1 per 100 um) but cannot buy back with
these freedoms.  Nothing reaches the slit: the admitted fraction is 0.

**End to end** (the spectrometer's scorer, 7 fields x 7 wavelengths, the
grating the stop, the chain's own launch per field):

| deck | elements | smile (px) | keystone (px) | SRF (px) | CRF (px) | grating admits | clearance (mm) |
|---|---|---|---|---|---|---|---|
| telescope + R4 | 16 | 3.14 | 0.59 | 15.7 | 15.5 | 1.000 at every field | +0.75 PASS |
| telescope + R5 | 19 | 3.13 | 0.59 | 15.7 | 15.6 | 1.000 | -0.11 (package vs the prism's fold mirror) FAIL |

These are the telescope's blur passed through the spectrometer (R4 alone:
smile 0.005, keystone 0.003, CRF 1.33 px).  The pupil match costs no
light: the grating admits the whole 70 mm bundle at every field.  R5's
package-vs-fold-mirror pair, +0.90 mm on the spectrometer's own record,
reads -0.11 here because the fold mirror's footprint is cut from the
end-to-end bundle, which the telescope's blur widens.

## 5. What is open

1. **The telescope's image.**  Conics and symmetric even aspheres on
   three folded mirrors do not image at the pixel over +-12.3 deg at
   F/1.8 (67 px rms, a 1.6 mm field swing).  Two routes, both the
   review's: the two-mirror modified Schwarzschild ("the widest field and
   lowest F-number", 50 deg at F/1.6 with one conic and one asphere; an
   inverted telephoto, large against f -- acceptable beside a 0.7 m
   spectrometer), and FREEFORM mirrors on the folded three-mirror, which
   the engine has (Surface= Zernike, CALIB's OptZern) and the chain does
   not yet carry.  A third, cheaper check first: seed the conics from the
   coaxial parent's anastigmat solution and fold afterwards, since the
   image rungs here started from spheres at 32 deg folds.
2. **The quick wall vs the gate.**  The quick wall (rectangles, rims)
   reads +1.0 mm where the gate (2 mm sampled bodies) reads +0.75; the
   same sign, close enough for the solve; the gate is the record.
3. **R5's fold-mirror footprint** should be cut from the F/1.8 cone the
   slit admits, not from the blurred end-to-end bundle, once the
   telescope images.
4. **The Seidel seed.**  `macos.design.seidel_seed`'s n-flip model gives
   the wrong paraxial focus for this PNP geometry (EFL check 381 mm vs
   126); worth a look by whoever owns it, the chain's exact first order
   is what the telescope uses.
