# afocal4 wall — make the clearance a wall, then converge the cleared curve

Origin: `macos/BRIEF_afocal4_wall.md` (Dave, 2026-08-30), the follow-on to the
clearing stage. That stage proved the collimator interference is structural
(the field-walk ratio law), retired the fold and the station with
measurements, and delivered a −10° extraction tilt that clears the union gate
at **+37.82 mm** with the wavefront 13.6 % better. Two things kept it from
being the *clean* answer:

1. **The clearance was not a wall, and the re-solve spent it.** At −8/−9° the
   solver walked +23.3/+42.3 mm of margin down to +2.3/+0.7 mm, because
   `afocal4_score` cannot see clearance. So the delivered −10° was a point
   that *happens* to hold margin, not a point chosen on a frontier.
2. **The delivered numbers were budget-limited, not converged** — exitflag 0
   at 427 evaluations, on the 3e-3 **forward** difference S4c measured as
   reading this merit's gradient 17 % low.

**Nothing under `challenges/afocal4/` is overwritten.** Everything new lives
here, plus three additive clauses beside it: `../afocal4_union_wall.m` (new),
one deferred-wall clause in `../afocal4_build.m`, one post-tilt wall call in
`../clearing/clear_build.m`, and four `P.pack.union_*` fields in
`../afocal4_params.m` — all **default OFF**, all asserted to leave the
committed decks rebuilding byte for byte.

Every number is engine truth over the whole 0.5° × 0.5° field box. No `.in`
file is text-parsed for a geometric claim — the one place a prescription is
read as text is design **recovery** (`wall_recover`: conics, `R_M2`, `t_M1M2`
and the interface standoff, spacings from `zElt` and never from the vertices),
and that recovery is *verified* by rebuilding the file byte for byte before
anything is measured.

---

## 1. The wall

`afocal4_union`'s floor on the **declared body model** (1.15 × union footprint
+ 15 mm) is now a wall inside `afocal4_build`, beside the S4b `m3_behind_min`
one — and on the same terms: **a wall, never a merit term.** The log-domain
merit doctrine is untouched. A body standing in a beam is not a worse
telescope; it is not a telescope.

```matlab
P.pack.union_enforce  = false;   % DEFAULT -- see below
P.pack.union_min      = 0.000;   % the floor it holds, m
P.pack.union_body_k   = 1.15;    % declared body = this x union footprint
P.pack.union_body_pad = 0.015;   % ... grown by this, m
```

**Default OFF is load-bearing, not timidity.** With the wall on,
`afocal4_build` cannot re-emit the committed 343 mm deck — it reads
−79.89 mm — so every S4 / S4b / S4c / clearing artifact in the record would
stop reproducing. `P.pack.enforce = false` keeps the unbuildable S4 reference
reproducible for exactly the same reason.

**`clear_build` DEFERS it past the tilt, and that is the whole design.**
`afocal4_build` emits the *untilted* train and `clear_build` swings it
afterwards, so a wall applied inside the build would judge the design the tilt
exists to get away from and reject every iterate — a cage, not a wall. Both
halves are gated on the same `P` in `tAfocal4Wall`: the build alone refuses,
the build-plus-tilt does not.

### What it costs, measured

The wall is evaluated **inside the build**, so every iterate the solver sees is
compliant. Per evaluation, on this design at solve sampling (ngrid 21, nodes
11), cold:

| | s |
|---|---|
| `clear_build` | 1.61 |
| `afocal4_score` | 6.56 |
| **`afocal4_union`, the wall** | **4.18** |
| evaluation, before | 8.17 |
| evaluation, with the wall | 12.35 (**+51 %**) |

In a warm solver loop the whole evaluation runs about 3.2 s, so the fleet's
17 points at up to 1200 evaluations each fit an overnight run six-wide.

Nearly all of the wall's cost is the **nine-field re-trace** inside
`afocal4_union` — a trace `afocal4_score` has already paid for once. Sharing it
would mean restructuring the committed scorer, so it is not done; that is
stated rather than tuned around. The probe count is almost free (314 probes
4.18 s, 65 probes 3.52 s), so it is **not** turned down: a wall must hold the
same quantity the gate reports.

### The one sampling caveat, and why the margin covers it

The wall is judged at SOLVE sampling and every table quotes the gate at
REPORTING sampling. More rays make a bigger union hull, so the wall's number is
the **optimistic** one — measured at **+1.9 to +2.5 mm** on this design, and
carried per point as `R.sampling_bias_mm`. The seeder's 10 mm margin is what
absorbs it, and `tAfocal4Wall` pins the bias *below that margin* rather than
pinning a particular millimetre. It is not a caveat that can be waved at: it
is exactly what broke the report path (§4).

---

## 2. The compliant seeder — and the law is not its predictor

*A wall needs a compliant seed or it is a cage* (S4b rule 10; warm-starting the
S4 trade lost four of its five points that way). `wall_seed` is
`afocal4_pack_seed`'s sibling here, and building it produced a finding worth
keeping.

### The field-walk law is 5–6× optimistic as a predictor of a STANDOFF change

The obvious cheap ranking is the law itself:

```
    f_hat = f(tilt alone) + 2*|alpha| * (d - d_0)
```

with `d` the field-mirror → collimator spacing. It is pure closure arithmetic
and it is **wrong here**. Measured at −6°, moving the standoff from the
parent's −38.6 mm to +250 mm takes `d` from 0.563 to 0.680 m:

| | floor |
|---|---|
| law predicts | **+11.45 mm** |
| measures | **−8.25 mm** |

and over the parent's whole admitted range the realised sensitivity is
**33 mm per metre of `d`** against the law's **209**. The law is not wrong —
the tilt really does supply a field-independent `2αd` — but a **standoff**
change moves the field-PROPORTIONAL part at the same time, and the two very
nearly cancel. That is **leverage 2 showing up inside leverage 3**: the
clearing stage retired the station as "nearly powerless", and it is nearly
powerless here too.

Ranking by the other obvious rule — `afocal4_pack_seed`'s own "weakest field
mirror first" — is worse still: at −6° it spent ten gate evaluations walking a
−13.0 mm floor down to **−83.2 mm**.

### So it probes and bisects

1. **The tilt alone.** If the parent swung by the requested angle already
   clears, nothing else moves. That is the answer worth having — the clearance
   came from the tilt, and the frontier point stays comparable with the
   delivered row, which was seeded exactly that way.
2. **Probe the extreme** of the admitted closure range (the most lever the
   closure allows) and, if it clears, **bisect back toward the parent** for the
   smallest standoff change that still clears. Four to six gate evaluations,
   no prediction relied on. `INFO.slope_mm_per_m` carries the realised
   sensitivity beside the law's.
3. **A different M2 radius**, because the S4b anchor is that this study
   changes the front end last.
4. **The delivered −10° design's own DOFs**, last resort — a known-feasible
   point, but a different basin, and `INFO.source` says so rather than letting
   a reader assume the whole curve was seeded the same way.

Closure validity and the S4b packaging station are pure algebra on
`afocal4_phi4`'s output, so the candidate list is filtered by them before a ray
is traced; only survivors pay the ~6 s a build-and-gate costs.

### The trap that cost a cycle

`P.parent` carries **Mike's raw secondary** (R_M2 468.8 mm, t_M1M2 1.0492 m)
while the committed 343 mm deck has a **re-solved front end** (448.4 mm,
1.0420 m). Filtering seed candidates through `P.parent` admitted 21 standoffs
of 57 and **not one of them was the parent design's own**; carrying `D.R2` and
`D.t1` into the closure — which is exactly what `afocal4_build` does — admits
54, spanning `d` = 0.255…0.821 m against the parent's 0.563.

A failure to seed is reported as a failure to **seed**: `INFO.ok` false with
the best floor it reached, never "this tilt has no design".

---

## 3. The frontier — and the two things it retired

Full tables and the operating point: `../RESULTS.md` § C.4c and § C.4d.
In brief, both halves of what this slice was set up to show turned out
differently under measurement, and both reversals are the result:

**1. The margin is not spent.** The clearing stage's −8/−9° wall-off
re-solves ended at +2.32 / +0.69 mm on 427 budget-capped forward-difference
evaluations. The *same* solves at 1209 central-difference evaluations end at
**+28.05 / +38.67 mm**. At −8° and −10° the wall-on and wall-off runs are
identical to the last digit of round-1 merit — **the wall never rejected an
iterate**. The margin-spending was a stalled solve, not a merit blind to
clearance. The wall is still right to have (nothing else holds the clearance,
and it refuses the committed deck at −79.89 mm) but here it is insurance;
**convergence** is what changed the answer. It does bind at a +15 mm floor and
at −9°, and both times it landed a *better* design.

**2. The free-standoff sweep is not a tilt-vs-price curve.** Its points order
by the standoff each solve reached (−229 → +536 mm from a −38.6 mm parent),
with wavefront and blur falling monotonically along the way and every solve
still descending 24–43 % per round. Pinning the standoff at +276 mm and
solving `{conic, front}` gives the real curve:

| tilt | floor (mm) | WFE (nm) | blur (µm) | breathing (%) | wander (µm) |
|---|---|---|---|---|---|
| **−8°** | +15.18 | **6513.9** | **279.9** | **0.7210** | **284.2** |
| **−9°** | **+45.44** | 7794.7 | 352.0 | 0.9190 | 356.4 |
| −10° | +48.54 | 7682.3 | 456.8 | 1.1912 | 460.9 |
| −11° | +47.17 | 7464.9 | 482.9 | 1.2044 | 487.0 |

**The clearance saturates by −9°** and the pupil price keeps climbing, so −10°
and −11° are dominated — **the delivered design sits past the knee**. The
operating point is −9° (it beats the delivered −10° on wavefront, blur, wander
*and* clearance); −8° is the answer if the pupil budget binds, at 49 % less
blur while still holding the declared +15 mm pad.

**3. And the delivered numbers were budget artifacts.** The −10° deck polished
with central differences moves every quoted number: WFE −25.0 %, blur −36.3 %,
wander −36.2 %, breathing +20.3 %, floor +19.2 %.

### What was run, and what was dropped

17 points were launched; 13 carry a checkpoint and are in the record. Dropped,
with the reason, because a run that is not reported has to be accounted for:

* **−11° and −12°, both walls** — past the clearance saturation (the floor is
  flat from −9° on), so they pay pupil for nothing the −9/−10 points do not
  already show. The `ctl` series covers −11° at a fixed standoff instead.
* **−9° and −10° at the +15 mm wall** — the raw floor at those tilts already
  exceeds the seeder's need, so the wall never binds and the point duplicates
  its 0 mm sibling. The +15 mm arm's information is carried by −6/−7° (where
  it binds and forces an *inadmissible* design) and −8° (where it binds and
  lands a better one).
* Killed mid-solve for memory headroom (16 MATLAB processes, 4 GB free, load
  37 — the OOM pattern this box has hit before). **Their decks were deleted:
  a prescription on disk with no checkpoint behind it is a mid-solve snapshot,
  and mid-solve snapshots get quoted.**

Two points were re-run rather than dropped: `t-80_u15` and `t-90_u15`, both
lost to the report-path wall bug in §4. A running MATLAB caches the function
it has already called, so the fix reached only processes the fleet started
afterwards; the affected runs were identified from the fleet log's start
times.

---

## 4. Things that cost a cycle, recorded

* **A wall belongs on ITERATES, never on the REPORT that follows them.**
  `clear_solve` built its final, quotable deck through the same walled builder
  the objective used. Inside the objective a violation becomes a large finite
  residual and the solver backs out of it; in the report path there is nobody
  to back out — and because the wall is judged at SOLVE sampling while the
  report builds at REPORTING sampling (~2.5 mm lower floor), **a converged
  design sitting on its wall throws out of its own report and takes the solve
  with it.** Cost: one hour of a −8 deg run. Fixed in `clear_solve` (the
  report build measures, it does not judge) and guarded in `wall_point` (a
  round that throws costs a round, not a night).
* **`P.parent` is not the parent design.** See §2 — 21 admitted standoffs of
  57, none of them the deck's own.
* **The field-walk law is exact for the tilt and useless for the standoff.**
  See §2 — 5–6× optimistic, because the station moves the field-proportional
  part at the same time.

---

## 5. Files

| file | what |
|---|---|
| `../afocal4_union_wall.m` | **the wall** — `afocal4_union`'s floor, applied and thrown, with the spec read defensively off `P.pack` so an older saved `P` gets the wall OFF |
| `wall_seed.m` | the compliant seeder: probe, then bisect |
| `wall_recover.m` | the design struct behind a committed deck, **verified** by rebuilding it byte for byte (third copy of this recovery in the study; the `zElt`-not-vertices trap is why it verifies rather than trusts) |
| `wall_point.m` | one frontier point: seed → converged solve (central differences, restart until plateau) → gate and score at reporting sampling → checkpoint |
| `run_wall_point.m` | one point, one process, driven by the environment |
| `run_wall_fleet.sh` | the whole frontier, six MATLAB processes wide |
| `afocal4_wall.m` | the assembler: the frontier table, the operating point, the non-vacuity A/B, the polish, the addendum, the figure |
| `../../../tests/tAfocal4Wall.m` | the gates (8, `SUITE_FREEFORM`) |
| `wall_*.mat` | one checkpoint per point |
| `afocal4_wall_*.in` | one deck per point |

Run:

```bash
MACOS_HOME=~/dev/macos/macos_f90 ./run_wall_fleet.sh 6      # hours
```
```matlab
R = afocal4_wall();                     % assemble and report
R = afocal4_wall('sections',[0 3]);     % the wall's non-vacuity + the addendum
[D,i] = wall_seed(Q, D0, -8);           % one compliant seed
W = afocal4_union_wall(P, 'x.in', 'throw',false, 'bare',true);
```

Model size 256, one MATLAB process per model size,
`MACOS_HOME=~/dev/macos/macos_f90`.
