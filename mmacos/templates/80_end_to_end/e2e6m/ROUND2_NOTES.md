# NOTES — e2e6m round 2 (dry-run feedback, Dave 2026-08-25)

Round 1 is the DRY RUN (demo-complete, commits through 5c51371).
These are the recorded items for the redo — NOT started; CTB-focused
coronagraph work comes first (separate arc, "return to ctb").

## The slide-11 disagreement — DIAGNOSED (confirm in redo)

Dave's objection is correct: engine and linear model CANNOT disagree
at 1 nm/1 nrad if they share a basis — the pokes ARE the model.  The
check table's fingerprint (s5_report.txt [1], Seg19): the model is
x/y-SYMMETRIC (|Rx|=|Ry|, |Tx|=|Ty| — a hex segment in its own
frame) while the engine is strongly axis-ASYMMETRIC (Ry/Tx active;
Rx/Ty near-null), and Tz agrees to 0.6%.  That is an AXIS
CLOCKING between two "local" frames, not physics: both the Jacobian
channels and the drift/check perturbs say 'frame','local', but a
SEGMENT's local frame carries the clocked-Mon/SegXgrid heritage
ambiguity (the documented e5-corpus 180°/clocking flip family) — the
channel triads and the perturb-path TElt resolve x/y differently.
Piston is along the (near-shared) normal → basis-invariant → the
0.6% (≈1−cosθ of the parent tilt).  The 1.0-class errors are
response attributed to the missing axis; the ~55× errors are model
response where the engine's axis is null.

REDO FIX: apply the drift and the check through the SAME channel
objects that built J (or transform explicitly via get_elt_csys);
then all six DOFs should close at FD-linearity level and the control
basis un-shrinks from piston-only to the full six.  Confirm the
clocking numerically (channel triad vs TElt triad on one segment)
as the first act of the redo.

## Dave's items for round 2 (verbatim intent)

1. DMs (etc.) for the coronagraph.
2. A sketch showing the SEQUENCE of optics — PM–SM–TM–… through
   both legs.
3. A Bench layout graphic isolating the optics AFTER M2 at
   interpretable size (the full-observatory views bury them at 6 m
   scale).
4. Add the MET.
5. Show the MASK (occulter) — figure of the thing itself.
6. Show the APODIZER — figure of the thing itself.

Plus from round 1's own findings: the slide-11 frame fix above; the
S3b engine-faithful design operator stays queued post-demo.

## Sequencing

CTB coronagraph deepening FIRST (Dave: "more specific coronagraph
work to improve our approach" — DMs/masks/apodizers on the CTB
substrate, where the PROPER arbiter and tCtbProp gates live), THEN
the e2e6m redo folds the improved approach in.
