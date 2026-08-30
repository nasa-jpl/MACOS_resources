# offset_imager — wide-field imager with an offset field

A parameterized, illustrated five-stage design flow for the recurring
problem: a wide field box that must sit **far off the optical axis**
(packaging, stray light, or scan geometry demand it), and what each
class of design freedom buys back.  The flow is the product; the
`challenges/rodgers3` instance (Mike Rodgers' 22°-offset imager ladder)
is the validation.

## Run it

```matlab
run('<path-to>/mmacos/mmacos_setup.m');
addpath('<path-to>/mmacos/templates/10_telescopes/offset_imager');  % F1: setup does NOT add templates
OUT = oi_story();                       % the WHOLE story: ladder + counter-designs + <tag>_STORY.md manifest
OUT = offset_imager();                  % the five-stage ladder only (rodgers3 instance)
OUT = oi_story(struct( ...              % your instrument (see "Choosing an envelope" below)
        'EPD_m',0.150,'Fno',3.3,'box_deg',[15 15],'offset_deg',22.5, ...
        'z_m1_m',0.665*1.65,'spacings_m',[-0.7229 0 0.7408]*1.65, ...
        'seed_R1_m',8.8*1.65,'clear_m',[0.040 0.025],'exit_dir',[0 0 -1]));
OUT = oi_walk(struct(...), 'steps',[5 8 11 13 15]);   % HARD instances: walk the box open
OUT = oi_demo_step(12);                 % ADJACENT instance: ONE warm-started step off the walk
```

`oi_story` is the one-call form (ladder + both counter-designs + the
deck-asset manifest); `offset_imager` is the ladder alone.  All
parameters live in `offset_imager_params.m` (single source of truth,
the e2e2 pattern).  Artifacts: per-stage decks `<tag>_s*.in`, figures
`<tag>_s*_{layout,fields,map}.png`, `<tag>_REPORT.md`, `<tag>_run.mat`
(+ `<tag>_STORY.md` from oi_story).

**For a HARD instance -- a wide box at a large offset -- use `oi_walk`
(the recommended path).**  A cold `oi_story`/`offset_imager` solve at
the full box can sit outside the convergent basin and stall (the t5
instance: cold start 595565 nm, clearance -205 mm, S2 loses 104/121
fields).  `oi_walk` solves an easy NARROW box first, then WALKS
`box_deg` outward, carrying the solved design as each step's warm start
-- the same envelope/offset, a smaller aberration span at each rung.
It bootstraps step 1 with the full `offset_imager` ladder, screens the
carried design at each widened box before solving (halving the step if
it would not trace -- the F8 rule), and scores the final step exactly
like any `oi_story` run.  On the t5 instance the walk reaches 69.8 nm
at the target 15x15deg box (8531x better than the cold start, exit gate
PASS; clearance becomes the binding constraint -- see
`t5_walk/t5_walk_REPORT.md`).  Driver: `run_t5_walk.m`.

## The adjacent problem: one warm-started step off the walk

A finished walk is not just a design — it is a **frontier**, and
`oi_demo_step` extends it.  Give it a box full width and it warm-starts
from the largest committed step *below* the ask, states the outcome the
frontier PREDICTS (the bracketing committed rows and the line between
them) **before** solving, runs ONE full-freedom S5 continuation step,
and prints a verdict block: dense-map max, clearance floor, exit error,
gates, and predicted-vs-measured.

```matlab
OUT = oi_demo_step(12);                          % the t5 instance, 12x12 deg
OUT = oi_demo_step(14, 'gn_iters',20);           % more polish, more minutes
type(OUT.files.verdict)                          % re-print the block any time
oi_demo_show(OUT);                               % the reveal windows (auto on desktop; no-arg = newest run)
```

It refuses in two places rather than reporting a number it cannot stand
behind: an ask outside the walk's own span (5–15° here), and the F8
traceability screen — the carried design is scored at the widened box
*before* any solve, and a no-rays sentinel stops the run.  Both refusals
are one accurate sentence and neither raises.

**In practice the RANGE screen is the operative guard.**  Widening the
box alone never trips the F8 sentinel on this instance: the carried
design keeps tracing all the way out, degrading smoothly rather than
losing rays (measured 2026-08-28 — 30° → 1765 nm, 60° → 1.25e5,
90° → 2.59e5, 120° → 4.55e6, 150° → 5.71e6, never the 1e9 sentinel).
F8 is inherited from `oi_walk`, where it guards the *cold-start* failure
mode (the t5 unguided run losing 104/121 fields) rather than a
continuation one — the committed walk itself took zero halvings.  Worth
knowing before trusting it to catch an out-of-envelope ask: it will not.

The instrument is otherwise FIXED (offset +22.5°, EPD 150 mm, F/3.3, the
×1.65 W-fold envelope): **box width is the only validated continuation
axis** — walking the offset re-enters the t4 field-walk infeasibility.

Batch runner for a background solve: `run_oi_demo.m` (`OI_DEMO_WIDTH=12
matlab -batch "run('.../run_oi_demo.m')"`).  Gates: `tests/tOiDemoStep.m`
(not in a suite — two full solves).  It is also the live-demo beat for
the 2026-09 CodeV talk; the spoken script, the scripted refusals and the
fallback ladder are in `demo_adjacent/REHEARSAL.md`.

### Measured at the pinned knobs (2026-08-28)

Three widths run end to end; these are the committed `demo_adjacent/`
bundle.  Metric as always: strict RMS WFE, exit-pupil anchor,
piston-only removal, headline = 11×11 dense-map MAXIMUM.

| ask | warm start | predicted | measured | floor pred → meas | exit err | verdict | wall |
|---|---|---|---|---|---|---|---|
| 7×7° | step 1 (5°) | 18.0 nm | **20.0 nm** (1.11×) | 77.6 → 93.8 mm | 0.002° | PASS | 14.4 min |
| 12×12° | step 3 (11°) | 33.7 nm | **33.6 nm** (1.00×) | 24.8 → 24.9 mm | 0.012° | PASS | 14.5 min |
| 14×14° | step 4 (13°) | 54.9 nm | **51.2 nm** (0.93×) | 21.2 → 24.9 mm | 0.001° | PASS | 14.5 min |

Deterministic: four independent 12° solves reproduced 33.6240 nm /
24.9496 mm / 0.012103° to every printed digit.  Concurrency is nearly
free — three runs abreast cost the same wall clock as one — which is how
the bundle is regenerated.

**The 14° row is worth reading twice.**  The frontier's straight line
between the committed 13° and 15° rows predicts a **21.2 mm** floor, i.e.
below the 25 mm spec; the re-solve delivers **24.9 mm and passes both
gates**, at a better wavefront than predicted too.  The reason is the one
the endgame found at 15°: the walk's widest rows stopped early on
`oi_solve`'s WFE-only plateau break while the clearance hinge was still
pulling, so those rows — and any line drawn through them — *understate*
what the envelope holds.  A continuation step re-solves; an interpolation
cannot.

### The knob study (measured 2026-08-28)

A 12×12° ask, warm-started from the committed 11° step.  Every row is
scored the SAME way — 11×11 dense map at nGridpts 41 — so the rows differ
only in solve knobs.

| gn_iters | nsolve | solve_sampling | dense map max | avg | floor | exit err | wall |
|---|---|---|---|---|---|---|---|
| 2 | 3 | 21 | 40.96 nm | 27.35 | 24.65 mm | 0.0141° | 13.1 min |
| 8 | 3 | 21 | 38.53 nm | 26.19 | 24.88 mm | 0.0013° | 56.6 min |
| 2 | 5 | 21 | 35.23 nm | 24.09 | 23.89 mm | 0.0427° | 29.5 min |
| **2** | **5** | **11** | **33.55 nm** | 22.24 | 24.94 mm | 0.0005° | 29.5 min |
| 3 | 4 | 11 | 103.85 nm | 84.47 | 18.87 mm | **0.6981°** | 31.5 min |

Three rules come out of it, and they are worth obeying in any warm-started
step off a walk, not just in the demo:

- **Spend on FIELDS, not iterations.**  Iterations 2→8 at a 9-field grid
  bought 6% of dense-map max for +43 min; 9→25 fields bought 14% for
  +16 min.  This is the recorded S5 solve-field lesson reproducing: at
  `nsolve = 3` the solve-set merit converged to 25.9 nm while the dense
  map sat at 38.5.  The tell is visible in the map itself — the box
  CORNERS (which a 3×3 grid samples) came out best on the map, 15–22 nm,
  while the unsampled ridge at YAN 25–27° ran 38–41 nm.
- **`nsolve` must be ODD.**  `oi_solve` imposes the exit-direction
  equality on the solve field nearest the box centre; an even grid has no
  centre field, so the exit chief is pinned off centre.  The `nsolve = 4`
  row above is that failure: the exit beam walks 0.70° off pin (1400× the
  odd-grid rows) and the wavefront degrades 3×.
- **`solve_sampling` is not a cost lever.**  The solve is deck
  write/parse bound, not ray bound: 11 vs 21 nGridpts measured 29.51 vs
  29.49 min.  It is pinned at 11 on QUALITY (better on all three reported
  axes at 12°), not to save time.

Pinned: `gn_iters = 1`, `nsolve = 5`, `solve_sampling = 11`, model 256,
nGridpts 41, 11×11 map.

## Choosing an envelope, offset, and seed (read before a new instance)

Compiled from the t4 retraction and the t5 unguided experiment
(challenges/rodgers3/PACKET.md addendum + t5_unguided_REPORT.md):

- **Feasibility screen first**: the field walk `tan(offset) x
  |spacings(1)|` is what separates an unobscured W-fold.  Keep it above
  ~1.5x EPD or expect clearance blocking no surface solve can fix
  (offset_imager prints an ADVISORY when you are below it).  No
  envelope of this family packages a fast 200 mm beam at a 12 deg
  offset — buildability constrains the FIELD choice too.
- **Envelope**: scale the rodgers3 W-fold by YOUR EFL / 0.3 m
  (form-true — preserves the focal proportions the seed needs).
  Aperture-ratio scaling breaks the form when your F# differs.
- **Seed and S1 depth**: `seed_R1_m` ~ 8.8 x (EFL/0.3).  More
  important: CAP the S1 depth with `s1_target_nm` near your class's
  reference (rodgers3 class: ~159 nm).  An uncapped S1 solves far past
  the reference and the offset box then loses every ray (the S2
  mechanism past traceability) — the t5 failure mode.
- **S5 solve grid**: with the full Zernike set (80+ variables) use
  `nsolve_s5 = 5`; the default 3x3 under-determines the solve (the
  118.2 -> 45.4 nm lesson).
- **Constraints**: `clear_m` is order-free (min = hard knee, max =
  WARN); a three-mirror train exits REVERSED, so "exit horizontal" with
  entry along +z is `exit_dir = [0 0 -1]`.
- **Hard instance? Walk it.**  If the box is wide AND the offset large,
  a cold solve at the full box may be outside the convergent basin (the
  t5 stall).  `oi_walk(over, 'steps',[...])` solves a narrow box first
  and walks `box_deg` outward, carrying each solution as the next warm
  start; it isolates whether the residual difficulty is surfaces
  (aberration, which the walk usually clears) or packaging (clearance /
  envelope, which no surface solve fixes).  Walk the BOX WIDTH, never
  the offset (offset-down re-enters the t4 field-walk infeasibility).

## The stages (each = a solve + a layout figure + a dense WFE map + a report section)

| stage | freedom opened | the lesson |
|---|---|---|
| S1 | symmetric conics + aspheres, solved at the ON-AXIS box | the classical coaxial wide-field imager |
| S2 | FPA tilt/focus refit ONLY, field box moved to the offset | the **disaster map** — what the offset costs when nothing else follows |
| S3 | re-solve the symmetric surfaces AT the offset field | the bias doctrine: solve at the used field (expect oblate-class conic flips) |
| S4 | + mirror tilts/decenters (+ radii stay open) | constraint set becomes live: exit-beam direction, clearances |
| S5 | + Zernike surface departures (aspheres replaced) | what true freeform buys at fixed packaging |

## Architecture

- `offset_imager_params.m` — every knob (EPD, F#, box, offset, λ,
  packaging spacings, constraint set, Zernike term set, densities).
- `oi_paraxial.m` — signed-convention paraxial chain: EFL/BFD/Petzval +
  the first-order seed solve (EFL exact, Petzval = 0).
- `oi_seed.m` — parameter set → starting design struct (spheres).
- `oi_close.m` — the first-order closure run at EVERY solve iterate
  (afocal4 doctrine: identities re-derived, never penalized):
  EFL = EPD·F# exactly (R3 eliminated), stop posed by the
  entrance-pupil construction, FP posed on the traced exit chief.
- `oi_deck.m` — design struct → MACOS prescription.  The stop is a
  Reference element carrying the **native element-bound stop**
  (`macos.stop` → engine `ChiefRayAiming` real-ray aiming, A/B'd against
  the Stage-0 Newton aiming in `challenges/rodgers3/probe_native_stop.m`
  to ≤0.04 nm).  The header `ApStop=` (StopPos) form is deliberately NOT
  used — it aims with no optics traversal, wrong for a stop behind M1.
- `oi_score` — the metric: strict RMS WFE (design/src kernel),
  centroid reference on the stage's frozen FPA, exit-pupil anchor,
  piston-only removal.  Stated next to every quoted number.  PROMOTED
  to `design/src/oi_score.m` (2026-08-20), with `oi_clear` (the
  beam-leg/obstacle clearance model; per-field disk of record or
  `P.clear_footprint = 'hull'` convex hulls) beside it.
- `oi_solve.m` — damped Gauss–Newton over per-field WFE residuals with
  natural per-variable scales; walls (not penalties) for constraints;
  solve set ≠ scoring set.
- `oi_gates.m` — exit-beam direction + beam/mirror clearance gates.
- `oi_map_fig.m` / `oi_layout_fig.m` — the per-stage illustrations.
  Each stage emits THREE figures: `*_layout.png` (the `macos.view_std`
  four-panel solid-body hardware render), `*_fields.png` (a Y-Z
  elevation with per-field beam ENVELOPES — filled patches, not ray
  spaghetti — plus stations and the exit-chief annotation), and
  `*_map.png` (the dense strict-WFE-vs-field map).
- `oi_walk.m` — parameter continuation over `box_deg` (the solution
  finder for hard instances); `oi_demo_step.m` — ONE warm-started
  continuation step off a finished walk, with the frontier prediction
  stated before the solve, plus `run_oi_demo.m` (background batch
  runner) and `demo_adjacent/` (fallback bundle + rehearsal script).

## Conventions inherited from the rodgers3 challenge

Global frame, metres, beam enters +z; `KrElt` = signed CODE V radius;
fields tangent-composed; Zernikes `ZernType= BornWolf` with lMon frozen
at the traced footprint (power pinned to radii, tilt to pointing —
the Zernike solve doctrine).  The paraxial sign convention in
`oi_paraxial.m` was validated against the rodgers3 r1 deck by real rays
(engine plate scale vs paraxial EFL).

## Suite coverage

`tests/tOffsetImager.m` (freeform group, size 256) runs a reduced-knob
smoke of S1–S3 at a second parameter set — proving the template is
parameterized, not a rodgers3 replay.  The full five-stage runs live in
`challenges/rodgers3/PACKET.md` (T3) and this directory's committed
report (T4).
