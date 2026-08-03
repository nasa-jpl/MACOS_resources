# afocal4 — a 30× afocal telescope with a controlled interface pupil

The user-facing example arc of `../PLAN_AFOCAL4.md`: build an afocal telescope in the
MACOS design layer and score **both** its image quality and its interface-pupil quality.
The demonstration is J.M. Rodgers' 30× coaxial afocal TMA benchmark
(`../rodgers2/`) and his assertion that a fourth mirror is needed for pupil control.

**Where the arc is: S3 (the form study) is delivered; S4 (the joint solve) is next.**

## Files

| file | what |
|---|---|
| `afocal4_params.m` | the one parameter struct — aperture, magnification, field box, stop, the parent's transcribed layout, and the S3 targets.  Nothing below this file invents a number. |
| `afocal4_close.m` | first-order **closure** of one 4-mirror candidate: the afocal condition and the interface-pupil condition, in algebra.  Three forms — `field`, `relay`, `mersenne`. |
| `afocal4_forms.m` | the S3 study driver: closures, builds, traces, measures, figures. |
| `FORM_STUDY.md` | **the answer** — a section per candidate, the comparison table, the recommendation, and what would change it. |
| `afocal4_*.in` | the committed first-order layouts, one per candidate. |
| `afocal4_*_3view.png` | YZ / XZ / XY layout per candidate. |
| `afocal4_forms_compare.png` | the two rivals' first-order walls, and the field mirror's leverage. |
| `afocal4_forms.mat` | every number in `FORM_STUDY.md`. |

## Run

```matlab
run('~/dev/MACOS_resources/mmacos/mmacos_setup.m')
afocal4_forms                          % everything, ~15 min, model size 256
afocal4_forms('sections',0:1)          % the first-order closures only, seconds
afocal4_forms('save',false)            % numbers, no artifacts
P = afocal4_params();                  % the parameters
C = afocal4_close(P,'field','standoff',0.20,'phi4',2);   % one closed layout
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

Gated by `tDesignAfocal` (size-256 group: `./run_mmacos_tests.sh freeform`).
