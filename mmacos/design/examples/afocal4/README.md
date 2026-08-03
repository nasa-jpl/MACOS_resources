# afocal4 — a 30× afocal telescope with a controlled interface pupil

The user-facing example arc of `../PLAN_AFOCAL4.md`: build an afocal telescope in the
MACOS design layer and score **both** its image quality and its interface-pupil quality.
The demonstration is J.M. Rodgers' 30× coaxial afocal TMA benchmark
(`../rodgers2/`) and his assertion that a fourth mirror is needed for pupil control.

**Where the arc is: S3 (the form study) and S4 (the joint solve, the answer ladder,
the interface-standoff trade and the Mersenne hedge) are delivered.**

## Files

### S0–S3 — parameters, first-order closure, the form study

| file | what |
|---|---|
| `afocal4_params.m` | the one parameter struct — aperture, magnification, field box, stop, the parent's transcribed layout, the targets, and the S4 solve settings (operating point, merit weights, DOF scales and bounds).  Nothing below this file invents a number. |
| `afocal4_close.m` | first-order **closure** of one 4-mirror candidate: the afocal condition and the interface-pupil condition, in algebra.  Three forms — `field`, `relay`, `mersenne`. |
| `afocal4_forms.m` | the S3 study driver: closures, builds, traces, measures, figures. |
| `FORM_STUDY.md` | **the S3 answer** — a section per candidate, the comparison table, the recommendation, and what would change it. |
| `afocal4_{parent3,field_*,relay,mersenne}.in` | the committed first-order layouts, one per candidate. |
| `afocal4_*_3view.png`, `afocal4_forms_compare.png`, `afocal4_forms.mat` | the S3 artifacts. |

### S4 — the joint solve

| file | what |
|---|---|
| `afocal4_seed.m` | the design struct the ladder starts from: the S3 recommendation, unoptimised. |
| `afocal4_build.m` | **the inner loop.**  One design struct → a committed prescription, with the afocal condition, the collimator's station and the pupil station re-closed EXACTLY; rigid bodies applied to the emitted deck; the interface plane posed on the traced exit chief. |
| `afocal4_score.m` | **the merit.**  Six terms — WFE (afocal rung 2) plus the four-part pupil ladder plus magnification — normalised to their targets and turned into log-domain residuals. |
| `afocal4_score_print.m` | one score as a target-versus-achieved block. |
| `afocal4_solve.m` | **the outer loop.**  `lsqnonlin` over conics, the field-mirror standoff, the front end and (rung 4) the rigid bodies, with a finite-difference Jacobian in scaled DOFs. |
| `afocal4_ladder.m` | **the S4 driver**: the four rungs, the interface-standoff trade curve, the merit A/B, the figures. |
| `afocal4_mersenne.m` | the bounded hedge: does a conic-relaxed double Mersenne reach DL? |
| `RESULTS.md` | **the S4 answer** — the rung table, the trade curve, the Mersenne verdict, the parameter provenance, and the rules the runs earned. |
| `afocal4_r{1..4}_*.in` | one committed prescription per rung. |
| `afocal4_r*_field.png` / `_pupil.png` | the WFE field map and the pupil ladder per rung. |
| `afocal4_trade.png`, `afocal4_ladder_summary.png`, `afocal4_ladder.mat` | the trade curve, the ladder against its targets, and every number in `RESULTS.md`. |

## Run

```matlab
run('~/dev/MACOS_resources/mmacos/mmacos_setup.m')

% --- S3, the form study ---
afocal4_forms                          % everything, ~15 min, model size 256
afocal4_forms('sections',0:1)          % the first-order closures only, seconds

% --- S4, the joint solve ---
afocal4_ladder                         % rungs + trade + A/B + figures (hours)
afocal4_ladder('sections',0:1)         % the four rungs only
afocal4_mersenne                       % the hedge, time-boxed

% --- pieces ---
P = afocal4_params();                  % the parameters
C = afocal4_close(P,'field','standoff',0.20,'phi4',2);   % one closed layout
D = afocal4_seed(P);                   % a design struct
b = afocal4_build(P, D, 'x.in', 'quiet',false);          % close, emit, pose
S = afocal4_score(P, 'x.in');          % score it
```

Model size 256, one MATLAB process per model size.  Batch:
`matlab -batch "run('.../afocal4_forms.m'); exit(0)"` with `MACOS_HOME` set — and note
the `exit(0)` belongs to the batch wrapper, never inside these scripts.

## The headline

**The three-mirror already closes both first-order conditions.**  It recollimates,
magnifies by 30.000, and images the stop onto a plane 0.81 mm from the coldstop Rodgers
placed by hand.  So the fourth mirror is not repairing a first-order deficiency — there
isn't one.  What it has to buy is pupil *aberration* control, and the question each
candidate answers is whether it can be **given power without spending the first-order
solution to pay for it**.

Only one form can: a **field mirror at the intermediate image**, where the marginal ray
is small enough that magnification and the afocal condition are preserved identically
for any power.  One unoptimised K = 0 convex mirror cuts pupil blur 2.8× and drives the
chief-normal magnification breathing to 0.156%, inside the ±0.4% target, with every
conic still unspent.  The double Mersenne has the better pupil ladder and stays alive as
the runner-up on one bounded experiment; the downstream relay is eliminated.

Full reasoning, the closures, the traps and the recommendation: **`FORM_STUDY.md`**.

## What this example adds to the design layer

* `macos.design.Telescope.add_exit_reference` — terminate a train in a collimated beam
  at an interface plane instead of a focus (flat `Element= Reference`, **never**
  `Return`, which reverses the ray directions).
* `align_exit_reference` / `exit_pupil` — the afocal 2-pass pattern: put the interface
  plane on the traced exit chief, then find the exit pupil by chief-ray crossing and
  report the **traced** angular magnification and how far the pupil missed the interface.
* `afocal_first_order` (`design/src`) — the paraxial marginal + chief kernel that carries
  both first-order conditions, pinned by
  `optical_design/fixtures/afocal_tma_fixture.{json,md}` (stop-and-fix).
* `pupil_map` now reports magnification in **two frames** — see the rodgers2 PACKET §4
  refinement; a footprint read on a tilted interface plane carries that plane's own
  1/cos obliquity and is not a pupil-imaging defect.

## What the S4 layer adds

* `afocal4_build` — the **nested** pattern: an outer optimiser over aberration DOFs
  wrapped around an exact first-order closure, so magnification, collimation and the
  pupil station are *identities* of every design the solver sees rather than merit terms
  it has to buy.  Reusable for any afocal design driver.
* `afocal4_score` — a merit that mixes quantities six orders of magnitude apart
  (nanometres of wavefront, micrometres of pupil blur, percent of magnification
  breathing) by scoring each in the **log** of its ratio to target, with a floor.
* rigid bodies applied to the **emitted deck** and the interface plane re-posed on the
  traced chief, on one code path for perturbed and unperturbed builds alike —
  `tAfocal4` pins that path against `Telescope/align_exit_reference` at 1e-12.

Gated by `tDesignAfocal` and `tAfocal4` (size-256 group:
`./run_mmacos_tests.sh freeform`).
