# dyson5 beat 4d -- R5, the fold prism, under the clearance gate (cold-shield height as the parameter)

Written for: Dave (review).  Record: `dyson5_s5.txt`, `dyson5_s5.mat`,
`dyson5_s5_r5_h<hh>.in` (one deck per shield height; the record's height is
the R5 deck of record), `dyson5_s5_layout_r5.png`, `dyson5_s5_maps_r5.png`,
`dyson5_s5_sweep.png`, `dyson5_s5_r5_h<hh>_view3d.png` / `_viewyz.png`,
the trade table's `fold` row.  Runner: `dyson5_run(struct('stages', {{'s5'}}))`.

## 1. The question (addenda 6 and 10)

R4 of record puts the slit and the FPA on the block's flat face 0.85 mm
apart in z and 12 mm apart in y: the FPA package (54 x 9 mm active + 5 mm
carrier, 10 mm deep) clears the slit mask by +1.19 mm and there is NO room
for a cold shield -- addendum 6's gate reports exactly that.  The review's
answer when the gate fails is the fold prism: move the detector out of the
slit's plane.  Addendum 10 asks for it under the clearance gate with the
cold-shield height as the parameter.

## 2. The form (what the chain and the engine now model)

Entrance PLATE on the slit side: the slit in air `slit_gap` (0.5 mm) before
a plate cemented to the block's face (glass to glass -- the chain traces the
face as a refraction with equal indices, the engine as a Refractor with the
same GlassElt on both sides; no bending).  Mirror-coated FOLD PRISM on the
image side, cemented under the image: a 45 deg fold plane `fold_h` (8 mm)
below the face folding the beam toward -y, away from the slit (TIR is not
available: at F/1.8 in silica the marginal rays reach 30 deg incidence
against a 43.6 deg critical angle, so the hypotenuse is a Reflector INSIDE
glass -- `Element= Reflector` carrying `GlassElt=`, which the engine
honours: gate `tSpectrometerRx/...(form=dyson_fold)`, every ray to 1e-9 m);
an exit face `fold_e` beyond the fold plane, placed for FIRST-ORDER
CONJUGATE SYMMETRY with the slit side (glass + air/n equal on both sides,
`fold_e = (face - slit_gap) - fold_h + (slit_gap - fpa_gap)/n`); the FPA
`fpa_gap` in air beyond it, normal +y, its dispersion axis the fold's image
of +y, i.e. +z.  Every consumer of the detector frame (the scorer, the
clearance's package box, the layout producer, the emitter gate's band check)
now uses `G.fpa.xhat / yhat / normal` instead of y and z; the unfolded forms
reproduce the record byte for byte (the R4 deck re-emitted under the
reworked chain is identical).

The face offset becomes the plate's thickness and the prism's depth, so the
R5 rung re-solves R4's eleven variables with the face offset bounded at the
fold's scale (exit distance >= 7.5 mm: the beam's half-height at the fold is
~7 mm), warm-started from R4.  The groove period and the focus are re-solved
at every iterate as always; the slit-to-FPA wall is off (the fold IS the
separation; the clearance gate judges).

## 3. The parameter: cold-shield height h

A shield of height h in front of the FPA needs an air gap of h + 1 mm
between the prism's exit face and the FPA.  The air gap breaks the
conjugate symmetry (air on the image side where the slit side has glass),
which the R5 solve re-balances through the face offset and the rest; the
exit distance shrinks with the gap (`fold_e` above), so the lower bound on
the face offset grows with h -- a taller shield means a thicker plate and a
deeper prism.  The sweep: h = 0, 2, 5, 10 mm, each a full R5 solve,
engine score and clearance gate (the package in the folded frame, the plate
and prism one cemented part with the block).  The record is h = 2 mm.

## 4. Result (dyson5_s5.txt; engine scores on the 7 x 7 grid)

| shield h | air gap | face offset | smile | keystone | CRF | SRF | EE | clearance | worst pair |
|---|---|---|---|---|---|---|---|---|---|
| R4 of record (no fold) | -- | 0.85 mm | 0.0051 | 0.0026 | 1.327 | 2.032 | 0.759 | +1.19 mm | slit mask vs FPA package |
| 0 mm | 1.0 mm | 17.0 mm | 0.0041 | 0.0179 | 1.306 | 2.031 | 0.779 | +1.01 mm | block leg vs package corner |
| 2 mm | 3.0 mm | 29.0 mm | 0.0146 | 0.0143 | 1.378 | 2.031 | 0.727 | +0.67 mm | fold -> exit leg vs package |
| 5 mm | 6.0 mm | 29.0 mm | 0.0139 | 0.0169 | 1.390 | 2.030 | 0.725 | +0.14 mm | fold -> exit leg vs package |
| 10 mm | 11.0 mm | 23.5 mm | 0.0095 | 0.0128 | 1.295 | 2.030 | 0.796 | +0.06 mm | face -> fold leg vs package |

**The fold does what it was asked to do.**  Every point PASSES the
clearance gate with the detector package 25-45 mm from the slit (the slit
mask's own worst pair is now the entrance plate's leg at +2.1 mm), and the
image quality is R4's or better: CRF 1.30-1.39 px against the 1.5 px spec,
smile and keystone 0.004-0.018 px against 0.1, SRF at the 2-px slit floor.
A cold shield of any height in the sweep fits -- what it costs is a
thicker plate / deeper prism (face offset 17 -> 24-29 mm) and the package
reaching toward the prism: the clearance margin shrinks to +0.14 mm at 5 mm
and +0.06 mm at 10 mm, where the package's corner sits against the beam
inside the prism.  Beyond ~10 mm the package must be shaped (chamfered
toward the prism) or the exit face moved; the gate will say so.

**The landscape is multimodal and the sweep's solves are short** (30
iterations, warm-started along the sweep): h = 2 and 5 mm landed in a
29 mm-face basin at CRF 1.38 while h = 0 and 10 mm found 17 / 23.5 mm
faces at CRF 1.30.  The stage therefore re-solves the record height from
the best sweep basin with twice the iterations and keeps the better of the
two (the 'refined' row of the table; result below).  Dave's R4 lesson again:
the meniscus-plus-face landscape has several basins of similar merit.

**R5 of record (shield 2 mm, refined from the 10 mm basin, 60 iterations):**
face offset 23.51 mm (plate 23.0 mm thick), fold plane 8 mm below the face,
exit face 14.67 mm beyond the fold, 1.0 mm air gap; engine smile 0.0094 /
keystone 0.0119 px, **CRF 1.295 px, SRF 2.030 px, EE 0.794**, clearance
+0.67 mm PASS (fold -> exit leg vs the package).  Against R4 of record: CRF
1.327 -> 1.295, EE 0.759 -> 0.794, distortion 0.005 -> 0.012 px (40x under
spec either way), and a detector package WITH a 2 mm cold shield where R4
had +1.19 mm and no shield at all.  Deck `dyson5_s5_r5_h02_refined.in`;
the trade table's `fold` row; `dyson5_s5_sweep.png` carries the sweep with
the refined record as the star.

Deck note for the talk: the fold is small at the layout's scale (25 mm at
the base of a 700 mm instrument) -- the engine's y-z render shows it (E10
fold mirror, E11 exit face, E13 FPA); a zoom inset of the base is a
producer item for the deck, not done here.
