# afocal4 — a 30× afocal telescope with a controlled interface pupil

The user-facing example arc of `../PLAN_AFOCAL4.md`: build an afocal telescope in the
MACOS design layer and score **both** its image quality and its interface-pupil quality.
The demonstration is J.M. Rodgers' 30× coaxial afocal TMA benchmark
(`../rodgers2/`) and his assertion that a fourth mirror is needed for pupil control.

**Where the arc is: S3 (the form study), S4 (the joint solve, the answer ladder,
the interface-standoff trade and the Mersenne hedge), S4b (the same trade redone
under the packaging constraint), S4c (the rim convention and the long solve),
PACKAGING (the 343 mm back end, measured and folded), CLEARING (the collimator
was standing in its own feed beam) and WALL (the clearance made a wall, and the
cleared curve converged) are delivered.**

> **The committed 343 mm design had a BODY STANDING IN A BEAM, and the whole
> trade curve does.**  Over the field box the M2 → field-mirror feed beam runs
> through the collimator's own glass — **−79.9 mm** with the declared body
> model, −55.4 mm against bare lit glass — while *per field* there is 10.8 mm
> of daylight all round.  A monolithic collimator has to cover its **union**
> footprint (17.0 mm per field, 87.0 mm over nine) and that glass is where the
> other fields' feed beams pass.  It is structural, it obeys a **field-walk
> ratio law**, and only a **field-independent** remedy can defeat it: a −10°
> extraction tilt on the field mirror takes the gate to **+37.8 mm** with the
> wavefront 13.6 % *better*, and packages the back end with zero flats.  The
> price is the fourth mirror's pupil control.  See `RESULTS.md` § CLEARING,
> `clearing/README.md`, `wall/README.md`.

> **The S4 designs are NOT BUILDABLE** and are retained, unaltered, as the
> unconstrained reference.  They put the collimator, the field mirror, the interface
> pupil and the whole instrument behind it *in front of* the primary, inside the
> incoming beam (Dave, 2026-08-03).  One extra mirror flips the parity of the back end:
> his three-mirror parent has M3 640 mm **behind** M1, the four-mirror child built from
> the same front end has it 200–440 mm **in front**.  S4b re-derives the trade with the
> constraint enforced as a solver wall and demonstrates the fold that takes the
> instrument out of the beam.  See `RESULTS.md` §S4b.

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

### S4b — the buildable redo

| file | what |
|---|---|
| `afocal4_pack.m` | **the packaging gate**, from traced rays: vertex stations, fold daylight on the collimator's exit leg against *every* other bundle crossing it, and where the interface pupil and a stated instrument envelope end up once that fold is inserted. |
| `afocal4_phi4.m` | the interface-condition root, factored out of the builder so the seeder selects roots by the rule the builder uses.  A sign change is not a root: `d(phi4)` is rational and fzero converges onto its poles. |
| `afocal4_pack_seed.m` | a **compliant** seed at a given operating point — his front end first, the weakest field mirror that clears the bound with margin.  A wall needs a compliant seed or it is a cage. |
| `afocal4_s4b.m` | **the S4b driver**: the constraint in numbers, the anchor, the buildable trade (via `afocal4_ladder('prefix','b_')`), the folded demonstration, the figures. |
| `afocal4_b_*.in`, `afocal4_b_*.png`, `afocal4_b_ladder.mat`, `afocal4_s4b.mat` | the S4b artifacts, beside the S4 ones rather than on top of them. |
| `STATUS_S4B.md` | **the S4b one page** — the buildable trade table, what the constraint cost against the free curve, the folded package, and the feasible window. |
| `afocal4_trade.png`, `afocal4_ladder_summary.png`, `afocal4_ladder.mat` | the trade curve, the ladder against its targets, and every number in `RESULTS.md`. |

### S4c — the rim convention and the long solve

| file | what |
|---|---|
| `afocal4_basin2.m` / `_merge.m` | basin 2 re-solved long, from three seeds, with a central-difference polish and an explicit gradient probe — "converged" and "plateau" told apart by a number rather than by an exit code. |
| `afocal4_fork.m` | the 343 mm fork (flat M4 vs powered M4) re-scored in both pupil conventions. |
| `STATUS_S4C.md` | **the S4c one page.** |

### PACKAGING, CLEARING, WALL — the buildability arc (2026-08-30)

Three stages, each in its own directory, **nothing above them overwritten**.
Read them in this order; the canonical write-up is `RESULTS.md` § CLEARING.

| directory | what it answers |
|---|---|
| `packaging/` | how deep is the 343 mm back end, and can folds take the depth out?  Delivered a four-fold route (Path A) — **since superseded**, see its §3 — and found something bigger: **the collimator stands in its own feed beam.** |
| `clearing/` | why that is structural (**the field-walk ratio law**), which remedies the law forbids (a flat fold, the station, the standoff — all retired *with measurements*), and the one it does not (**an extraction tilt**).  Delivers `afocal4_clear_343mm.in` and the standing gate `../afocal4_union.m`. |
| `wall/` | the clearance made a **wall** in `afocal4_build` (with a compliant seeder, or it is a cage), and the cleared tilt-vs-price curve re-solved to convergence with central differences rather than to a budget. |

| file | what |
|---|---|
| `afocal4_union.m` | **the standing gate**: does a BODY stand in a BEAM?  Union footprint over the field box, convex hull never a disk, declared allowance printed beside bare lit glass, sampling-free pierce and clearance.  Run as part 4 of `afocal4_pack`; `'union',false` reproduces the old verdict. |
| `afocal4_union_wall.m` | the same floor **as a wall** in `afocal4_build` (`P.pack.union_enforce`, default OFF — with it on, the builder cannot re-emit the deck the trade shipped). |

## Run

```matlab
run('~/dev/MACOS_resources/mmacos/mmacos_setup.m')

% --- S3, the form study ---
afocal4_forms                          % everything, ~15 min, model size 256
afocal4_forms('sections',0:1)          % the first-order closures only, seconds

% --- S4, the joint solve (UNCONSTRAINED -- not buildable) ---
afocal4_ladder                         % rungs + trade + A/B + figures (hours)
afocal4_ladder('sections',0:1)         % the four rungs only
afocal4_mersenne                       % the hedge, time-boxed

% --- S4b, the buildable redo ---
afocal4_s4b                            % constraint + anchor + trade + fold (hours)
afocal4_s4b('sections',[0 1])          % the constraint in numbers, and the anchor
afocal4_s4b('sections',[3 4],'trade','load')   % fold + figures off a finished trade
afocal4_ladder('prefix','b_')          % just the constrained rungs + trade
K = afocal4_pack(P, 'x.in');           % is one deck buildable?
[D, i] = afocal4_pack_seed(P, 0.14);   % a compliant seed at 140 mm

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

## What the S4b layer adds

* the **packaging constraint as a wall, never a merit term** — `afocal4_close` returns the
  vertex *stations* the spacings imply, `afocal4_build` refuses any closure whose last
  powered mirror is less than `P.pack.m3_behind_min` behind the primary, and it reads the
  emitted stations back off the committed deck to pin them to the ones the wall judged.
  A mis-scaled penalty owns the solve; an unbuildable layout is not a worse design.
* `afocal4_pack` — a packaging check that includes **instrument-volume placement**, which
  is the gap that let the S4 layouts through: train length, AOI and self-obscuration all
  pass on a design whose instrument sits in its own incoming beam.
* `afocal4_pack_seed` — the companion rule: a wall is only a wall if the solver starts
  inside the feasible region.

## What the buildability arc adds

* **a gate that asks whether a BODY stands in a BEAM**, not just where there is
  daylight — `afocal4_union`.  *A margin is a number, not a body*: the S4b gate
  measured 17.5 mm of daylight for a fold whose own flat, sized over the field box,
  clips the feed beam by −73.6 mm.  Bodies are convex **hulls** of the **union**
  footprint over the field box, with a **declared** allowance printed beside bare
  lit glass, and both distance measures sampling-free so a fold cannot appear to
  move a clearance an isometry cannot move.
* **the field-walk ratio law** — a reusable statement, not a fact about this deck.
  On a coaxial train a part's footprint and a beam's are two scaled copies of the
  same off-axis field box, so they separate only if their scales differ by more
  than `(bias+half)/(bias−half)`.  Splitting each centroid into a field-proportional
  walk and a field-INDEPENDENT offset says immediately which remedies can ever
  work — and rules out the station, the standoff and any flat fold before one is
  built.  `clear_law`.
* **an extraction tilt as an exact rigid motion of the traced chief** —
  `clear_tilt` swings about the point the chief actually strikes, takes the new
  chief from the rotated *local* surface normal (engine truth, `N = unit(d_in −
  d_out)`), and re-poses the train by the rotation carrying old chief onto new.
  Chief path preserved to 4.45e-16 m.
* **the clearance as a second wall** — `afocal4_union_wall` +
  `P.pack.union_enforce`, deferred past the tilt by `clear_build` so it bounds
  the design rather than caging it, and `wall_seed` to start inside it.

Gated by `tDesignAfocal`, `tAfocal4`, `tAfocal4Clear` and `tAfocal4Wall`
(size-256 group: `./run_mmacos_tests.sh freeform`).
