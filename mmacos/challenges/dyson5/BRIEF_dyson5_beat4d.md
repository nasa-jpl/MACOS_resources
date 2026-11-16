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

## 4. Result -- SUPERSEDED, see section 5

The first sweep (commit 18c225a) is withdrawn: its warm start carried the
previous point's air gap along (every point ran at 1.0 mm), its fold was
8 mm deep so the package stood 1.5 mm inside the block above the face
plane, and the clearance gate of that day scored legs against bodies only
and could not see a package inside the glass.  The gate now flags a leg
inside a box body and scores the mask and the package against every other
body (section 5); the old record fails it at -1.49 mm.

## 5. Result, second sweep (2026-10-01, run 11): the air gap is the price

**What the air gap costs, measured on the chain before any re-solve** (the
unfolded R4 of record with its 0.85 mm air at slit and FPA widened to 2 and
3 mm on both sides): CRF 1.33 -> 1.78 -> 1.96 px, keystone 0.02 -> 0.23 ->
0.42 px.  Physics, not the solver: at F/1.8 a plane air/glass boundary
ahead of a converging cone carries spherical and field aberration growing
with the gap (~0.5 px per 3 mm), which is why Dyson slits and detectors
are proximate.  So the cold-shield height, which needs an air gap of
h + 1 mm between the prism's exit face and the detector (and the same gap
on the slit side, to keep the concentric form's object and image media
equal -- air on the image side alone cost CRF 1.19 -> 1.54 px), is bought
with image quality.

The second sweep's form: fold plane 16 mm below the face (the package,
9 mm + 2 x 5 mm carrier, centred on the fold, stays below the face plane
by the mount margin), the exit face placed after the focus solve so the
gap is exact, the slit plane's axial position an R5 variable (twelve
variables, 40 iterations per point), heights 0 / 1 / 2 / 3 / 5 mm, the
record the TALLEST height that closes (spec + clearance gate, which now
also scores the package and the mask against every body).

| shield h | air gaps (both sides) | face offset | smile | keystone | CRF | SRF | EE | clearance | worst pair |
|---|---|---|---|---|---|---|---|---|---|
| R4 of record (no fold) | 0.85 mm | 0.85 mm | 0.0051 | 0.0026 | 1.327 | 2.032 | 0.759 | +0.79 mm | slit mask against the face (adjacent by design) |
| **0 mm (record)** | 1.0 mm | 25.0 mm | 0.0051 | 0.0085 | **1.267** | 2.035 | 0.695 | **+0.90 mm PASS** | package against the fold mirror |
| 1 mm | 2.0 mm | 32.8 mm | 0.0052 | 0.0524 | 1.745 | 2.139 | 0.258 | +1.15 mm | fold -> exit leg vs package |
| 2 mm | 3.0 mm | 32.8 mm | 0.0055 | 0.0581 | 2.306 | 2.411 | 0.135 | +1.04 mm | fold -> exit leg vs package |
| 3 mm | 4.0 mm | 32.8 mm | 0.0047 | 0.0686 | 2.776 | 2.805 | 0.072 | +1.00 mm | fold -> exit leg vs package |
| 5 mm | 6.0 mm | 32.8 mm | 0.0151 | 0.1201 | 3.746 | 3.839 | 0.040 | +1.00 mm | package against the exit face |

**R5 of record: the fold with NO cold shield** (1 mm air at slit and
detector, plate 24 mm thick, fold plane 16 mm below the face, exit face
8 mm beyond the fold): engine CRF 1.267 px (R4: 1.327), EE 0.695 (R4:
0.759), smile 0.0051 / keystone 0.0085 px, every clearance pair positive
with the package 27 mm from the slit (+0.90 mm against the fold mirror's
mount).  The refinement from the same basin at 80 iterations gave CRF 1.321
with keystone 0.0033 and EE 0.725 -- not a lower CRF, so the sweep solve
stands (the rule as stated).  Deck `dyson5_s5_r5_h00.in`; `dyson5_s5_sweep.png`.

**The answer to addendum 10's question.**  The fold prism does its job --
it takes the detector package out of the slit's plane, where R4 had it
+1.19 mm from the slit mask with no room for anything -- but the cold-shield
height cannot be bought: every millimetre of air between the prism's exit
face and the detector costs ~0.5 px of CRF (1.27 -> 1.75 -> 2.31 -> 2.78 ->
3.75 px for 1 -> 6 mm), and the twelve-variable re-solve at each gap does
not recover it (the air/glass boundary ahead of an F/1.8 cone carries
spherical and field aberration that the block, meniscus and slit-plane
knobs do not cancel).  So a cold shield in this form has to live INSIDE the
1 mm the design tolerates, or be a cold WINDOW cemented as the prism's exit
face with the shield as the detector housing behind it -- the next R5
variant if Dave wants it on the record.  A slower cone (the envelope's
F/2.2 point has CRF 1.15 px at R4) would buy air; that trade is the
envelope's, not this beat's.

Deck note: the layout producer shows the fold small at the instrument's
scale; the engine's y-z render (`dyson5_s5_r5_h00_viewyz.png`) shows it; a
zoom inset of the base is a producer item for the deck.
