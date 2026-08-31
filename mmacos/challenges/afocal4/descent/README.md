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

## 5. Still to do

Tasks 1–4: the 7-mirror top rung with slack on every requirement row and the
71 nm wavefront target in the set; the descent itself (rank by |φ| and power
economy, probe, commit, control); the ladder table and figures; the record,
`tAfocal4Descent`, and the delivery log.
