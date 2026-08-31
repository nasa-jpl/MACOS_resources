# afocal4 descent — start at seven mirrors, walk back toward buildability

Origin: `macos/BRIEF_afocal4_descent.md` (Dave, 2026-08-31). The S4 arc asked
*what does the fourth mirror buy* and found a requirement pair with only half
met. The wall slice then found the committed design missed even its own pupil
optimum because **the extraction tilt was never in the DOF set** — the
signature pathology of building up from too little freedom. This stage inverts
the question: **start where everything is met with margin, remove one powered
mirror at a time, and measure what each removal costs.**

Time box: 48 h from 2026-08-31. Every rung is a finished, checked design; if
the box runs out mid-ladder the completed rungs stand.

**Status: Task 0 (the N-mirror closure and builder) is done and verified.**
The ladder itself has not started.

---

## 1. The closure generalizes in three lines — and they are the 4-mirror ones

`descent_close.m`. For an N-mirror coaxial afocal with an interface pupil the
three first-order conditions stay **exact closures**, never merit terms:

| condition | closed by | how |
|---|---|---|
| recollimate, `u_out = 0` | `phi_N` | analytic |
| magnify by 30×, `\|y_out\| = (D/2)/M` | `t_{N-1}` | analytic |
| exit pupil at `iface` | `phi_{N-1}` | the only root-find |

That is `afocal4_close`'s own `FIELD_D_`, verbatim — *the marginal ray fixes
everything but the pupil, so imposing the two first-order conditions is not a
solve at all, it is substitution; the chief ray is the residue.* Nothing in it
is about four mirrors: propagate the paraxial marginal and chief through
mirrors 1..N−2 with their free radii and spacings and the same three lines
close. **The descent's closure is the 4-mirror closure with a longer front
end.**

**Verified at N = 4, two ways.** Against `afocal4_close` directly:
`max|ΔR| = 4.4e-16`, `max|Δt| = 0`, `Δφ = 2.2e-16`, convex flags identical,
closure residuals `(u_out, M/30−1, pupil−iface) = (0, 2.2e-16, 0)`. And
through the full builder against the committed deck on disk: **byte for
byte**, given the same element names and the same scan recipe.

> That last qualifier is real and is not a fudge. `fzero` converges to
> whichever root its BRACKET contains, so a different scan grid lands 2e-16
> away and the emitted `KrElt` differs in its last digit. The identity is
> asserted under `afocal4_phi4`'s own window (`[-0.9 5.0]`, 119 points)
> rather than by widening a tolerance until the check passes. Element names
> are the other difference: the committed decks call their mirrors
> `M1/M2/FM/M3` because that is the *form*'s vocabulary, where the generic
> builder emits `M1..MN` for a *layout*.

## 2. Two things that had to be derived rather than inherited

**The sign of the exit marginal height is a property of the layout, not a
constant.** `afocal4_close` hard-codes `y_out = -(D/2)/M` because a 4-mirror
'field' train forms exactly one intermediate image, hence one axis crossing,
hence a negative exit height. At other N the number of crossings is not fixed.
Both signs are closed and the one that puts the last mirror at a **positive**
spacing wins — the rule `afocal4_close`'s own 'relay' branch already states
for its second image ("that sign is taken from the requirement that M4 sit at
a positive distance, not assumed"). If both close, it warns rather than
picking silently.

**A sign change is not a root** (RESULTS rule 11), and it matters more here:
`d(phi_{N-1})` is rational, so it changes sign across its *poles* as well as
its zeros. Every candidate is closed and CHECKED — finite positive spacings,
the pupil where it was asked for, both first-order identities intact — and the
lowest-|power| survivor wins, because the weakest penultimate mirror is the one
closest to the train the rung was grown from.

## 3. First finding: the packaging station obeys a PARITY LAW

`descent_seed.m` closes front ends by the thousand (the closure is algebra, so
a candidate costs ~0.7 ms and only survivors are ever built) and counts how
many put the last powered mirror at least `P.pack.m3_behind_min` + 30 mm
BEHIND the primary. Over a common grid:

| N | parity | closures | compliant | rate |
|---|---|---|---|---|
| 5 | odd | 232 | 205 | **88.4 %** |
| 6 | **even** | 7406 | **2** | **0.03 %** |
| 7 | odd | 95849 | 86024 | **89.7 %** |

**That is a factor of ~3000 between adjacent N, and it is not about N. It is
about parity.** The beam folds along z, one flip per reflection, so the vertex
stations are an alternating sum

```
    z_N = -t_1 + t_2 - t_3 + ... = sum_k (-1)^k t_k
```

and the closure's OWN last spacing `b = t_{N-1}` — the one the magnification
condition fixes, and typically the largest in the train — therefore enters
with sign `(-1)^(N-1)`. **At odd N it pushes the last powered mirror behind
the primary; at even N it pushes it in front.** Everything else is a
second-order fight against that one term.

This is S4b's finding — *one extra mirror flips the parity of the back end;
his parent has M3 at +640 mm, the four-mirror child built from the same front
end has it at −442* — stated as a law rather than as an observation about one
design, and now with a rate attached.

**Two consequences for the ladder, both worth knowing before it starts.**

1. **The top rung is easy to seed.** N = 7 offers 86 024 compliant front ends
   on a coarse grid; Task 1's difficulty will be meeting the *requirement set*
   with slack, not finding a buildable layout.
2. **N = 6 is the hard rung, and it fails on PACKAGING, not wavefront.** The
   brief expects the bottom of the ladder near N = 4–5 because the 4-mirror
   family demonstrably cannot meet the wavefront half. Parity says a
   *different* failure mode arrives first, at a different rung, and for a
   reason that has nothing to do with image quality. The descent should not be
   surprised by it, and should not misread a packaging failure at N = 6 as the
   wavefront bottom.

> **A grid is a grid.** The N = 4 row of that scan first read "no compliant
> closure" — for the very design sitting in the repository complying at
> +1323 mm. The coarse grid simply did not contain its spacings. The parent's
> own spacings are now always injected into the search, because a statement
> about the grid must never be reported as a statement about the topology.

> **And a wall with only one side is not a constraint.** The first N = 7 seed
> this produced put the last powered mirror **10.96 m** behind the primary —
> and it was, by every check in the study, compliant: the S4b packaging wall
> bounds that station from BELOW only, the closure does not care how long a
> train is, and the power-economy tie-breaker rewards weak mirrors, which are
> exactly the ones that need distance. Nothing said no. The seeder now carries
> an upper bound too, stated in the study's own unit — the packaging record
> measures depth as a MULTIPLE OF THE M1–M2 SPACING (committed 1.81×, cleared
> 1.24×), so the bound is 3× that spacing rather than a round number of
> metres. Caught before a five-hour solve was spent on it, not after.

### The seed already nominates its own removal candidates

The compliant N = 7 seed comes back with powers

```
    phi = [+0.800  -4.266  -0.625  -0.625  -0.625  +1.342  +2.188] /m
```

— M1 and M2 are Rodgers' front end, the last two are what the closure
consumes, and **the three added mirrors all sit at the weakest power the grid
offers**. That is the power-economy preference doing what Task 2 wants it to
do before the ladder has even started: the near-flat mirrors identify
themselves, and mirrors 3–5 are the removal candidates the descent will
probe. It is a *ranking*, not a decision — the brief's rule is that
predictions rank and measurements decide, and the field-walk law's 5–6×
optimism as a standoff predictor is the standing warning against trusting
closure arithmetic on its own.

## 3b. A prediction the ladder has to be designed around: DELETE flips parity, RETAIN does not

The descent removes a mirror by driving its power to zero and then either
**deleting** the flat or **retaining** it as a fold — and Dave's ruling 4 says
a retained flat is not a mirror, so either way the rung's N goes down by one.
Under the parity law those two are not cosmetic variants of each other:

* a flat mirror still **reflects**, so retaining it keeps the fold count and
  therefore the SIGN of every downstream station;
* deleting it removes a reflection, which **flips the parity of the whole back
  end** — the same flip S4b measured when a mirror was ADDED.

So the two options land on opposite sides of the 88 % / 0.03 % divide measured
in § 3. Concretely, stepping from the N = 7 top (odd, 89.7 % compliant) down
to N = 6:

| how the mirror is removed | reflections | parity | expected compliance |
|---|---|---|---|
| **deleted** | 6 | even | **~0.03 %** — the hard side |
| **retained as a fold** | 7 | odd | **~90 %** — the easy side |

**The descent therefore has a lever the brief did not anticipate**, and it is
legitimate rather than a dodge: ruling 4 already says a retained flat does not
count as a mirror, and the packaging round — not this slice — owns whether the
fold is worth its own cost. If this holds, "N = 6 cannot be built" and "N = 6
cannot be built *without keeping a flat*" are different statements, and only
the second is true.

**It is a prediction, not a result.** It follows from arithmetic that has been
measured (§ 3) applied to a step that has not been taken yet, and this arc's
standing warning is exactly against that move — the field-walk law was exact
for the tilt and 5–6× optimistic for the standoff. The ladder will measure
both options at the first even rung and report which happened.

**The mechanism is in the model, checked.** `stations_` alternates the fold
direction once per ELEMENT regardless of that element's power, so a retained
flat does set the parity and a deleted one does flip it — the closure is not
being asked to represent something it cannot. And the machinery carries a
zero-power element end to end: a free radius of 1e12 m closes at
φ = 2e-12 /m, the closure reports *elements 5, powered 4, flats 1 at index 3*
with residual 2.2e-16, and the deck emits and traces. `descent_remove`
implements both modes and refuses to touch M1, M2 or the two elements whose
powers the closure consumes.

> **A first check that came out wrong is recorded rather than dropped.** The
> first attempt to demonstrate the parity flip used an arbitrary N = 7 base
> and got NO CLOSURE from *both* modes — because the base itself was
> non-compliant (its last mirror sat 40 mm in FRONT of the primary). The test
> needs a compliant, solved base, which is what the top rung is for. A
> demonstration run on a broken fixture proves nothing in either direction.

> **And a paraxially exact closure can still be a bad telescope.** That same
> retained-flat probe closed its three conditions at 2.2e-16 and traced
> **M = 40.45 against a paraxial 30.0000**, with 0.22 rad of collimation
> error. Nothing is wrong: the closure is a FIRST-ORDER statement, the conics
> were unsolved, and `paraxial_ok` flagged it. It is worth stating plainly
> because the closure's residuals are so small that they invite being read as
> a quality claim, and they are not one — only the solve makes a closure a
> design.

## 4. Machinery

| file | what |
|---|---|
| `descent_close.m` | the N-mirror first-order closure (three conditions, one root) |
| `descent_build.m` | close → emit → pose the interface on the traced chief → apply the extraction tilts upstream-first → the three walls |
| `descent_seed.m` | a BUILDABLE N-mirror front end: cheap algebra filters, the packaging station (both sides) decides, and the weakest total power breaks ties |
| `descent_require.m` | the requirement set on one footing — TARGETS (with margin), WALLS (with room left), GATES (facts); the interface surface scored RIM-anchored per the S4c spec rule |
| `descent_solve.m` | the outer loop, N-generic: scaled deviations, log-domain merit, central differences, walls on iterates never on reports, a wall residual that scales with the merit's weights |
| `run_descent_top.m` | one top-of-ladder attempt, one process, checkpointed |

**Extraction tilts are in the DOF set from the start**, per mirror — the wall
slice's lesson, and this stage does not get to rediscover it. They are applied
to the emitted deck by `clear_tilt` (chief-hit pivot, downstream re-posed),
**upstream first**, so each swing composes with the ones before it.

Walls, in the order they are cheap: degenerate spacing (algebra) → the S4b
packaging station (algebra) → the union body-in-beam floor (a nine-field
trace, and DEFERRED past the tilts, or it judges the train the tilts exist to
get away from).

## 5. The first top-rung attempt (N7a) — and why it is NOT a verdict on seven mirrors

`descent_N7a.mat`, DOFs `{conic, spacing, tilt}` (radii frozen at the seeder's
grid values), 951 evaluations over 3 rounds:

| row | value | target | verdict |
|---|---|---|---|
| WFE rung-2 max | 12422.1 nm | 71.0 | MISSED, 175× |
| pupil blur | 506.2 µm | 47.0 | MISSED |
| wander | 530.6 µm | 56.0 | MISSED |
| breathing | 2.3744 % | 0.4 | MISSED |
| iface surface (rim) | 0.3433 mm | 0.2 | MISSED |
| M error | **3.8646 %** | 0.1 | MISSED |
| union floor | −111.58 mm | ≥ 0 | MISSED |
| last powered behind M1 | **499.88 mm** | ≥ 500 | MISSED by 0.12 mm |

**It missed every row, and it is still not evidence that seven mirrors cannot
meet the set.** Four things say the attempt failed, not the topology:

1. **M error 3.86 %.** The closure makes magnification an IDENTITY — its
   paraxial residual is 1e-16 on this very design. A 3.86 % *traced* error
   means the layout is so aberrated that real rays have left the paraxial
   regime; the committed 4-mirror deck sits at 0.0221 %. That is a statement
   about how bad the design is, not about how many mirrors it has.
2. **It converged to a STALL.** Round 2 bought 3.4e-4 and round 3 bought
   4.3e-11, at merit 70.78 — *worse than the four-mirror designs* (30.2
   committed, 32.7 for the wall slice's best). Meeting the set needs a merit
   near zero (≈2.4 sitting exactly on every target).
3. **It walked onto the packaging wall**, ending at 499.88 mm against a
   500 mm minimum. S4c settled what that means: *90 mm is CONSTRAINED, not
   unconverged* — a design pinned against that wall has a compromised
   gradient, and a NaN or a stall there is a constraint, not a failure.
4. **The radii were frozen.** All three added mirrors sat at the grid's
   weakest power (R = 3.2 m) with every added spacing at the grid minimum
   (0.4 m) — a cramped, arbitrary layout the solve was never allowed to
   loosen.

> **Reporting this as "seven mirrors cannot do it" would repeat both errors
> this arc has already paid for**: S4b's *a wall needs a compliant seed or it
> is a cage*, and the wall slice's finding that the committed design missed
> its own pupil optimum because **the DOF set, not the merit, was the
> reason**. A stage set up to inherit those lessons does not get to
> rediscover them.

So the N = 7 question is being answered the way the brief asks — with
independent seeds and the full DOF set — and N7a is retained as the first
datum of that spread rather than as its answer.

| run | seed Σφ² | DOFs | status |
|---|---|---|---|
| N7a | 26.6 | conic, spacing, tilt | done — every row missed, stalled at 70.78 |
| N7b | 26.6 | + **radius** | running |
| N7c | 38.6 | + radius, independent seed | running |
| N7d | 41.6 | + radius, independent seed | running |
| N8a | — | + radius, N = **8** | running (the pre-approved "add a mirror" branch) |

## 6. Still to do

Tasks 1–4: the 7-mirror top rung with slack on every requirement row and the
71 nm wavefront target in the set; the descent itself (rank by |φ| and power
economy, probe, commit, control); the ladder table and figures; the record,
`tAfocal4Descent`, and the delivery log.
