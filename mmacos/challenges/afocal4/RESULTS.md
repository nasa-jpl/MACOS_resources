# S4 RESULTS — the joint solve, the answer ladder, and what the fourth mirror actually buys

> **RETRACTED ON PACKAGING, 2026-08-03 (Dave). Every design in §2, §3 and §4 below is
> NOT BUILDABLE.** They put the collimator 200–440 mm *in front of* M1 — and with it the
> field mirror, the interface pupil and the entire instrument that follows the pupil —
> inside the incoming beam. The optical results stand exactly as measured and are left
> unaltered, because a retraction whose numbers have been deleted cannot be checked; they
> are the **unconstrained reference** against which the buildable trade is read. The redo
> is **§S4b**, at the end of this file. Nothing between here and there has been rewritten.
>
> One extra mirror flips the parity of the whole back end: his three-mirror parent puts M3
> 640 mm *behind* M1 and the four-mirror child, built from the same front end, puts it
> 200 mm in front. That is the finding in one number, and neither the S3 packaging check
> (train length, AOI, self-obscuration) nor the S4 gates looked at instrument-volume
> placement at all.

Answers `PLAN_AFOCAL4.md` S4 against the S3 ruling in `FORM_STUDY.md`: take the convex
field mirror near the intermediate image into a joint solve, replay J.M. Rodgers' four-slide
ladder with the fourth mirror in place, and score every rung on **both** axes — image
quality *and* interface-pupil quality.

Reproduce: `afocal4_ladder` (the four rungs, the trade curve, the merit A/B, the figures)
and `afocal4_mersenne` (the hedge). Every number below lives in `afocal4_ladder.mat` /
`afocal4_mersenne.mat`.

## One page

**What the fourth mirror buys, and what it costs.** A convex field mirror at the
intermediate image **solves the interface-pupil problem**: on axis, one joint solve puts
blur, wander, breathing, convergence-surface figure and magnification all inside their
targets — including the 56 µm wander that the S3 gate flagged as unreachable by any form
at first order (best first-order value: 1415 µm). It is paid for in **image quality**. At
the flagged 140 mm operating point the wavefront error floors near 8.5 µm at the design
field, 120× the diffraction limit, and no conic, standoff, front-end or rigid-body freedom
moves it more than a few percent.

**The two are the same knob.** The field mirror's power is consumed holding the exit pupil
at the interface standoff; more pupil control means more power means more field curvature
and astigmatism. The interface standoff therefore *is* the exchange rate, and carrying it
as a parameter (the S4 ruling) rather than a spec is what makes the result reportable:
§3 is the curve, and at the far end of it — 343 mm, φ₄ → 0 — the fourth mirror becomes a
flat, the wavefront returns to **15.8 nm** on axis, and the pupil reverts to the
three-mirror's.

**The runner-up is closed.** Relaxing all four conics of the double Mersenne, confocal
spacings held, takes its wavefront error from 59.4 µm to 35.1 µm against a 71 nm target,
and only to 3955 nm even on axis. Its 59 µm was never mostly about the parabolas (§4).

**The answer to Rodgers.** He is right that three mirrors do not control the pupil, and
this is the first quantitative statement of by how much: his best three-mirror runs blur
469 µm and wander 557 µm, and a fourth mirror takes those to 43 and 44 µm. What his deck
does not say — because it has no pupil column — is that on this prescription the two
qualities are in direct exchange, and the instrument's interface standoff is what sets the
rate.

**Status: delivered.** The machinery and its ruling §0, controls §1, the four rungs and
their parameter provenance §2, the wavefront-only bracket §2.1, the merit A/B §2.2, the
trade curve §3 with the basin repair §3.1, the Mersenne hedge §4, and the rules the runs
earned §5. Gated by `tAfocal4` (6/6) in the size-256 group.

**Open for Dave and Mike:** the operating point. The curve is the deliverable, not a
chosen value; 140 mm is the flagged default and it survives the trade, but the instrument's
interface standoff is what should set it.

---

## 0. The machinery, and the one ruling that shaped it

The solve is **nested**: an outer optimiser over aberration degrees of freedom wrapped
around an **exact first-order closure**.

At every iterate `afocal4_build` re-derives, in algebra, the field mirror's power, the
collimator's power and the collimator's station from the marginal and chief rays. So

* the afocal condition (`u_out = 0`),
* the magnification (`M = 30.000`),
* and the exit-pupil station (`= P.iface`)

are **identities of every design the solver ever sees**, not merit terms it has to buy.
They are gated at 1e-9 by `tAfocal4/test_closure_holds_the_first_order_identities`.

The consequence is the S4 ruling in operational form: **the field mirror's power is not an
outer DOF.** The closure consumes it to hold the pupil station at `P.iface`, and `P.iface`
is carried as a **parameter** — the operating point — with the trade reported as a curve
(§3) rather than a value optimised. Sweeping `P.iface` *is* sweeping φ₄.

**Outer DOFs**, one joint set per rung, never alternating:

| group | what | rungs |
|---|---|---|
| `conic` | K₍M2₎, K₍FM₎, K₍M3₎ — M1 **held** a parabola, as in his study | 1, 3, 4 |
| `standoff` | the **field-mirror** standoff before the intermediate image (two-sided: the mirror may sit past the image) | 1, 3, 4 |
| `front` | M2's radius and the M1–M2 spacing — his own rung 3 re-optimises "conics and radii" | 1, 3, 4 |
| `rb` | y-decenter and x-tilt on M2, FM, M3 | 4 |

M1's radius is held: it sets the aperture and the f-number the benchmark is posed at.
The field mirror's radius and the collimator's radius and station are **closed** — they
move rung to rung without ever being free.

---

## 1. Controls, before any result

| check | value | against |
|---|---|---|
| his 3-mirror, on axis, through this scorer | **13.97 nm** | his S1 slide, 15 nm |
| the seed rebuilt at the S3 operating point | blur 338.6 µm, breathing 0.157 %, wander 1645 µm, WFE 14 034 nm | the S3 `field_p2` row: 338.6 / 0.156 / 1644.5 / 14 046 |
| the interface pose, this path vs `Telescope/align_exit_reference` | identical to **1e-12** | `tAfocal4` |

The first is the one that licenses everything else: the same kernel that produces the
numbers below reproduces his own on-axis result to 0.93× through an independent pipeline.

---

## 2. The answer ladder  *(NOT BUILDABLE — see §S4b)*

> All four rungs sit at the 140 mm operating point, whose collimator is 442 mm in front of
> the primary. The optical numbers are the unconstrained reference; the buildable ladder is
> §S4b.2.

His four slides, replayed with the fourth mirror in place. Wavefront error is the afocal
**rung 2** (piston + per-field tip/tilt) — the rung that matches his CODE V field maps —
quoted on his 3×3 solve set and on a uniform 9×9 grid over the box; the two agree here.
Magnification and breathing are **chief-normal**; wander is at the **refit** interface
plane, whose pose is reported below the table.

| rung | WFE nm | blur µm | breathing % | wander µm | surface mm | M | worst miss |
|---|---|---|---|---|---|---|---|
| seed, on axis (unoptimised) | 2307 | 63.6 | 0.049 | 64.8 | 0.0003 | 29.9835 | 32.5× |
| **1  on axis, joint solve** | **1392** | **43.4 ✓** | **0.233 ✓** | **44.0 ✓** | **0.0003 ✓** | **29.9811 ✓** | 19.6× |
| **2  offset +0.6°, FROZEN** | 8835 | 426.9 | 0.582 | 437.6 | 0.016 ✓ | 30.3117 | 124.4× |
| **3  offset, joint re-solve** | 9600 | 167.1 | **0.113 ✓** | 171.0 | 0.019 ✓ | **30.0156 ✓** | 135.2× |
| **4  + M2/FM/M3 tilt+dec** | 9558 | 152.9 | **0.083 ✓** | 156.4 | 0.016 ✓ | **30.0158 ✓** | 134.6× |
| **TARGET** | 71 | 47 | 0.4 | 56 | 0.2 | 30.000 | 1.00× |

*(✓ = inside target.)* His three-mirror ladder over the same four steps, his metric:
15 → 430 → 160 → 119 nm. He reports no pupil column at all.

### What the ladder says

**Rung 1 — the fourth mirror does its job.** On axis, one joint solve puts **every pupil
target inside**: blur 0.92×, breathing 0.58×, wander 0.79×, surface 0.00×, magnification
0.63×. The wander target that the S3 gate flagged as the open risk — 56 µm against a best
first-order value of 1415 µm — is met with 21 % margin. That is the study's first result
and it is unambiguous: **the interface-pupil problem is solved by a convex field mirror at
the intermediate image.**

**Rung 1 also finds the wall.** The wavefront error stops at 1392 nm, 19.6× the
diffraction limit, with the conics, the field-mirror standoff and the front end all free.
That is not a convergence failure — see §3, where the same wall moves by two orders of
magnitude as a function of one parameter.

**Rung 2 — the collapse, and it is bigger than his.** Moving the field box to +0.6° with
the design frozen costs 6.3× in wavefront (1392 → 8835 nm) and 10× in pupil blur
(43 → 427 µm). His three-mirror loses 29× in wavefront over the same step (15 → 430 nm),
so the four-mirror is *less* fragile in relative terms and much worse in absolute ones.
The magnification error goes to 10.4× target — his own 30 → 28.7× slip, in our units.

**Rung 3 — the re-solve buys the pupil back, and pays 8.7 % of wavefront for it.** Blur
427 → 167 µm, wander 438 → 171 µm, breathing 0.58 → 0.11 % (inside target), magnification
error 10.4× → 0.52× (inside target); wavefront error 8835 → 9600 nm. The merit fell from
49.2 to 29.9, so this is the joint objective working exactly as specified — every term has
a vote, and at 124× off on wavefront and 9× off on pupil the cheaper improvement is the
pupil's. It is stated here rather than smoothed over: **a rung that trades wavefront for
pupil is what a two-axis merit does, and the bracket in §2.1 says how much was available
to trade.**

**Rung 4 — the rigid bodies buy almost nothing, and that is a result.** Six more degrees
of freedom (y-decenter and x-tilt on M2, FM and M3) move the wavefront by **0.4 %**
(9600 → 9558 nm) and the pupil blur by 8 % (167 → 153 µm), and the solve drives every
rigid body to **0.2 µm and 0.1 µrad** — it finds no use for them. His three-mirror gains
25 % of wavefront from the same freedom (160 → 119 nm). The wavefront decomposition says
why: the residual here is dominated by **field astigmatism and field curvature that grow
across the box** — astigmatism runs 1108 → 3312 → 6809 nm as YAN goes −0.25° → 0 → +0.25°
— and a rigid body adds a field-constant term. It can move the mean; it cannot flatten a
field dependence.

### 2.1 The bracket: what the wavefront alone could have reached

Quoting rung 3's 8.7 % wavefront increase without saying how much wavefront was available
to win would make a balance look like a failure. So the same DOF set was re-solved at the
same field against the **wavefront alone**, pupil terms switched off
(`afocal4_solve(..., 'pupil',false)`, 224 evaluations):

| at the +0.6° bias, 140 mm operating point | WFE nm | blur µm | wander µm | breathing % | M error |
|---|---|---|---|---|---|
| rung 2, frozen (nothing re-solved) | 8835 | 426.9 | 437.6 | 0.582 | 10.4× |
| **wavefront-only re-solve** (the floor) | **8467** | 438.5 | 449.2 | 0.500 | 9.5× |
| rung 3, joint re-solve (delivered) | 9600 | **167.1** | **171.0** | **0.113** | **0.52×** |
| target | 71 | 47 | 56 | 0.4 | 1.0× |

**The wavefront is at a wall and the DOFs barely touch it.** Six optical degrees of
freedom, solved against nothing but the wavefront, move it from 8835 to 8467 nm — **4 %**.
The frozen design was already within 4 % of the best these DOFs can do. So the wavefront
was never the thing rung 3 could win, and the joint solve spending 13 % of it (8467 →
9600) to take the pupil 2.6× better is not a bad trade; it is the only trade on offer.

That is also the answer to "why does rung 3 read worse than rung 2 in the wavefront
column": because the two are within 9 % of each other and both are within 13 % of a floor
that no amount of conic, standoff or front-end freedom can lower. `afocal4_r3_wfe_only.in`
is committed so the bracket can be re-scored.

### 2.2 The merit A/B — the log rule, measured

The same rung-3 solve, same seed, same DOFs, run once with the shipped **log** residuals
and once with the rejected **linear** ones:

| merit | WFE nm | blur µm | breathing % | wander µm |
|---|---|---|---|---|
| log (shipped) | 9600 | **167.1** | **0.113** | **171.0** |
| linear (rejected) | **8248** | 420.3 | 0.425 | 430.3 |
| *wavefront-only bracket (§2.1)* | *8467* | *438.5* | *0.500* | *449.2* |

The linear merit reproduces the **wavefront-only** solve to within 3 % on every column.
That is the claim, measured: with residuals linear in metric/target, a term that is 120×
off carries the whole sum of squares and the pupil terms cast no vote at all — the solve
is a wavefront-only solve wearing a pupil merit's clothes. Note also that "worst
normalised miss" *prefers* the linear result (116× against 135×), because the worst miss
is the wavefront in both cases; that is why the headline is a table and not a scalar.

### Interface-plane pose per rung

The plane is emitted normal to the box-centre exit chief, then refit (station + tilt) for
the operational wander number. The refit tilt is the direct analogue of the coldstop DAR
tilt Rodgers tunes by hand — 0° / 4.289° / 3.577° / −0.356° across his four variants.

| rung | refit shift mm | refit tilt ° | wander as emitted µm |
|---|---|---|---|
| 1  on axis | +0.88 | 0.000 | 141.5 |
| 2  offset, frozen | −1.48 | 10.075 | 552.9 |
| 3  offset, re-solved | −5.88 | 8.583 | 936.8 |
| 4  + tilt/dec | −6.52 | 8.705 | 1027.3 |

On axis the emitted plane *is* the wander-optimal plane, to three decimals — which is the
check that the chief-normal construction is right. At the bias the wander-optimal plane
wants 8.6–10° of tilt off the exit chief, which is the same phenomenon his coldstop DAR
tilt addresses and about twice as large.

---

## 3. The interface-standoff trade — the exchange rate  *(NOT BUILDABLE — see §S4b)*

> Every row of the table below places the collimator in front of the primary: 559 mm at
> the 50 mm standoff, 442 at 140, and even the two points that come closest — 243 mm at
> 220 and 21 mm at 343 — fall short of the 500 mm the constraint requires. The curve is
> retained as the unconstrained reference; the buildable one is §S4b.3.

The S4 ruling carries the interface standoff as a **parameter**, so what the instrument is
owed is the **curve**, with the design fully re-solved at every point (rung-3 DOF set,
seeded by a conics-only pass). This is the study's central result.

| iface mm | φ₄ 1/m | R_FM m | s_FM mm | WFE nm | blur µm | breathing % | wander µm | M |
|---|---|---|---|---|---|---|---|---|
| 50 | +4.031 | 0.496 | 313 | 27 497 | **126.4** | 0.482 | **131.3** | 29.9966 |
| 90 | +2.679 | 0.747 | 297 | 17 171 | 142.0 | 0.119 | 147.0 | 30.0148 |
| **140** *(flagged)* | +1.782 | 1.122 | 291 | 9 560 | 152.9 | 0.083 | 156.5 | 30.0156 |
| 220 | +1.213 | 1.650 | 308 | **7 532** | 226.2 | **0.029** | 230.9 | 29.9965 |
| 343 | +0.419 | 4.770 | 278 | 9 352 | 764.7 | 0.146 | 778.4 | 29.9810 |
| target | | | | 71 | 47 | 0.4 | 56 | 30.000 |

**The two qualities are the same knob, and it runs the way the physics says.** Squeezing
the interface standoff from 343 mm to 50 mm takes the pupil blur from 765 µm to 126 µm — a
factor of six, monotone — and the wander from 778 µm to 131 µm. The field mirror's power
is what buys that, and its Petzval and astigmatism contributions are what the wavefront
pays: from 220 mm down, the wavefront error climbs 7.5 → 9.6 → 17.2 → 27.5 µm.

**The wavefront has an optimum, and it is not at either end.** 7.5 µm at 220 mm, rising to
9.4 µm at 343 mm as well. That is the balance between the fourth mirror's own contribution
and the three-mirror front end's, and it means "make the fourth mirror weaker" stops
helping the image at 220 mm while the pupil keeps getting worse all the way out.

**No point on the curve meets the wavefront target**, and the curve is what makes that
structural rather than a solve failure: the best wavefront available anywhere on it is
106× the diffraction limit, at the operating point whose pupil is second-worst. On axis the
same design at 343 mm reads **15.8 nm** — so the whole wavefront problem is the **field**,
not the fourth mirror's presence.

**Choosing the point.** Since the wavefront target is unreachable everywhere, the choice is
a pupil choice with a wavefront price attached. Between 140 mm and 220 mm the pupil blur
degrades 48 % for a 21 % wavefront gain; between 140 mm and 90 mm it improves 7 % for a
80 % wavefront loss. **The 140 mm flagged default sits on the knee** and the flagging
survives the trade. Below 90 mm the pupil gains flatten and the wavefront cost keeps
climbing, so 50 mm buys nothing worth having. This is what the curve supports; it is not a
spec, and the operating point is Dave's and Mike's to set.

### 3.1 Basin path-dependence, and how the 220 mm point was caught

The trade's first pass warm-started every point from the 140 mm design. At 220 mm that
walked into a basin with K_FM = −14.2, a traced magnification of **21.8×** and four of the
nine fields scrambled — and the *wavefront* column looked like the best on the curve
(7553 nm) while the pupil column read 16.7 mm of blur. Two things caught it:

* **`pupil_map`'s anchoring residual**, which is a validity check and not a metric: it
  reads 0.1 µm on every sound design here and **84 mm** on that one. `afocal4_score_print`
  now says so out loud when the residual exceeds 10 % of the blur.
* the magnification, which the closure holds at 30.000 **paraxially** — a traced 21.8×
  means the real rays have left the paraxial model behind, which is the S3 on-axis
  verification rule showing up again at a different stage.

Re-solved from two independent seeds, 35 iterations each:

| seed | WFE nm | blur µm | breathing % | wander µm | M | worst |
|---|---|---|---|---|---|---|
| fresh at 220 mm | 9058 | 157.1 | 0.019 | 160.9 | 30.0016 | 127.6× |
| warm from the 343 mm point | **7532** | 226.2 | 0.029 | 230.9 | 29.9965 | **106.1×** |

Both are sound designs and they are genuinely different — 20 % apart in wavefront and 44 %
apart in blur. The merit picks the lower worst-miss, and the pair is reported rather than
the winner alone, because on a six-DOF problem with a finite-difference Jacobian the basin
is part of the answer.

One useful property fell out of the repair: **an afocal4 design is fully recoverable from
its committed prescription.** The four conics, M2's radius, the M1–M2 spacing and the
field-mirror standoff (the difference between the paraxial image distance and the emitted
M2→FM spacing) are all readable from the `.in`, and rebuilding from them reproduced the
committed deck to **0.000e+00** relative. That is what the deck-as-the-artifact discipline
buys: the record cannot drift from the design, because the design is *in* the record.

---

## 4. The Mersenne hedge — closed

The S3 gate kept the double Mersenne alive on one bounded experiment. Its pupil ladder
was the best anything in the form study measured, its wavefront error was 59 µm, and the
argument for keeping it was that **59 µm is not a property of the form** — it is the price
of insisting all four mirrors be parabolas, which is what the *name* requires and not what
the design does. Two confocal pairs are afocal because of their **spacings**.

The experiment: hold the confocal spacings, relax all four conics (M1's too — its parabola
is a naming convention here, not a benchmark constraint), same solve machinery, same
targets. Two solve paths, because basin dependence on a four-conic problem is expected and
the honest report is both.

| | four parabolas | four conics free | ratio |
|---|---|---|---|
| WFE rung 2, max (nm) | 59 370 | **35 050** | 0.59× |
| pupil blur rms (µm) | 175 | 170.2 | 0.97× |
| breathing, chief-normal (%) | 0.285 | 0.098 | 0.35× |
| wander at the refit plane (µm) | 191 | 182.6 | 0.96× |
| M at box centre | 29.94 | 30.01 | 1.00× |

Conics: `[-1 -1 -1 -1]` → `[-0.9871, -1.0731, -0.7090, -0.5730]`.

**Verdict: CLOSED.** Four conics take the wavefront error from 59.4 µm to 35.1 µm against
a 71 nm target — a factor of 1.7 where a factor of 500 was needed. Solved **on axis**,
where a coaxial design carries no field aberration at all, it still reaches only 3955 nm.
Nothing was traded away to get there either: the pupil ladder is essentially unmoved. The
form carries wavefront error the conics cannot reach, and the 59 µm in the S3 table was
never mostly about the parabolas.

The field mirror stands. 43 minutes of machine time against a two-hour box. The 53 mm
interface standoff — the objection that would have killed the form regardless — never had
to be argued.

Reproduce: `afocal4_mersenne`. Artifacts: `afocal4_mersenne_{parabolic,conics,
conics_onaxis,conics_carried}.in`, `afocal4_mersenne_hedge.png`, `afocal4_mersenne.mat`.

---

## 5. Rules earned, each with the alternative it replaced

Every one of these was a wrong answer first. They are stated in the imperative because
that is how they should be applied to the next afocal design, and each carries the
measurement that earned it.

1. **Close the first-order conditions; never put them in the merit.**
   A design decision, not a measurement — taken from the S4 brief and stated here
   because it is the one that shapes everything else. The afocal condition, M and the
   pupil station are what make the object an afocal telescope with an interface pupil;
   as merit terms they would be tradeable, and a solver that can buy wavefront error
   with half a percent of magnification eventually will. `afocal4_build` instead
   re-derives the field mirror's power, the collimator's power and the collimator's
   station from the paraxial rays at every iterate, so all three are exact — 1e-9 — for
   every design the solver sees. The cost is real and is stated in §0: the field
   mirror's power stops being a degree of freedom and becomes the operating point.

2. **Score each merit term in the log of its ratio to target.**
   *Alternative rejected:* residuals linear in metric/target, which is the obvious
   reading of "normalise every target to 1". An unoptimised four-mirror layout misses
   the wavefront target by ~200× and the pupil targets by 6–7×, so in a sum of squares
   the wavefront term carries 99.9 % of the merit and the pupil terms cannot vote.
   Measured in §3's A/B.

3. **Solve in scaled DEVIATIONS from the seed, not in scaled values.**
   *Alternative rejected:* `x = value/scale`. The M1–M2 spacing is 1.05 m and a conic
   is order 1, so the vector handed to `lsqnonlin` was dominated by one component, the
   initial trust region was sized by it, and the first step threw the design across the
   parameter space: rung 1 went 2323 → 2308 → **2516 nm** over twenty evaluations.
   With `x = (value - seed)/scale` every DOF starts at zero and one unit means the same
   size of design change on all of them.

4. **Seed a joint solve with a short conics-only pass.**
   This is not alternation — the rung is still one joint solve over its whole DOF set.
   *Alternative rejected:* going straight to the full DOF set from the carried conics.
   Cold, the six-DOF solve sat at **2246 nm after 40 evaluations**, worse than a
   three-conic solve from the same seed had already reached (**1391 nm**). The carried
   conics are far enough from any solution that the first step decides the basin, and a
   finite-difference Jacobian over DOFs of very different sensitivity gives that step
   badly. The seeding pass costs about a tenth of the rung.

5. **Normalise the pupil before fitting Zernikes to a wavefront.**
   `afocal_refs` returns pupil coordinates in **metres** (16.7 mm here). A fifth-order
   polynomial in metres has r⁶ ≈ 1e-11, so an un-normalised least-squares fit returns
   coefficients of **1e10 nm that cancel to nanometres** — the same pathology the
   rodgers1 Zernike-solve doctrine records. Normalise to the unit disk and use a
   Noll-normalised basis, and each coefficient is that term's RMS contribution.

6. **Reject a degenerate closure as a wall, not as a datum.**
   A closure can return arithmetically valid spacings that are not a telescope — a
   collimator behind the mirror that feeds it, two surfaces stacked. `afocal4_build`
   errors below a 20 mm spacing and the solver turns back on a large finite residual,
   rather than building and scoring nonsense or dying.

7. **Score the wander at the REFIT interface plane, and report the as-emitted number
   beside it.** A cold stop is a mechanical part whose pose is chosen after the optics;
   Rodgers tunes his coldstop's DAR tilt exactly this way. Quoting only the placed-plane
   number charges the design for a plane nobody would build.

8. *(inherited from S3, and it kept paying)* **Verify a traced layout against its own
   paraxial prediction ON AXIS before taking any metric from it.** At the design bias
   the comparison reads the design's own field aberration as a builder failure.

---

# S4b RESULTS — the same trade, buildable

Answers the S4b brief against the `PLAN_AFOCAL4.md` **BUILDABILITY CONSTRAINT** (Dave,
2026-08-03). Everything in §0–§5 above is the unconstrained reference and is retracted on
packaging; nothing there has been rewritten.

Reproduce: `afocal4_s4b` (the constraint in numbers, the anchor, the folded
demonstration, the figures) and `afocal4_ladder('prefix','b_')` (the constrained rungs and
the buildable trade). Every number below lives in `afocal4_s4b.mat` /
`afocal4_b_ladder.mat`.

## One page

**The constraint does not remove the S4 design's performance. It splits it, and forces a
choice.** Holding Rodgers' front end verbatim, compliance requires the fourth mirror
250–600 mm *past* the intermediate image rather than on it. There it gains a footprint —
so its conic finally does wavefront work — and loses the field conjugate, which is where
pupil control comes from. The buildable anchor therefore reads **2.8× better in wavefront
and 6.8× worse in pupil** than the unbuildable design it replaces. The exchange rate runs
the opposite way to S4's.

**There is a second basin, and it has to be seeded by hand.** A 4.5% slower secondary
pushes the intermediate image 900 mm behind M1; the field mirror can then sit *on* it and
the collimator still lands 567 mm behind. That recovers the S4 geometry almost exactly and
beats it on both pupil columns, for 2.5–3× the wavefront error and 240 mm of extra length.
The solver does not find it unaided, so §S4b.3 sweeps both basins and reports both.

**The sharpest statement is one row-pair.** At 343 mm — the only standoff where the whole
package closes around a real instrument — two buildable designs exist: one whose fourth
mirror has gone to a flat (**266 nm** wavefront, and his three-mirror's 3.9% breathing),
and one with a real fourth mirror (**160 µm** blur, **0.081%** breathing, 10.5 µm
wavefront). **Pupil control costs a factor of 40 in wavefront, and declining it returns you
to the telescope Rodgers already has.** Unconstrained, the same exchange cost a factor of 6.

**A second constraint appeared that S4 never had to face.** The fold needs lever arm; what
is left of the interface standoff becomes the pupil's lateral offset once folded; and that
sets the largest instrument that fits — 0 mm at a 90 mm standoff, 464 mm at 343 mm. **The
pupil metrics want a short standoff and the package wants a long one.**

**No target is reached anywhere on either buildable curve** (best: 266 nm against 71, and
152 µm against 47), which is the same structural answer S4 gave, now with the packaging
half of it honest.

**Status: delivered.** The constraint in numbers §S4b.0, the anchor and the front-end
variant §S4b.1, the buildable ladder §S4b.2, both trade basins and the feasible window
§S4b.3, the folded demonstration §S4b.4, the Mersenne §S4b.5, and six more earned rules
§S4b.6. Gated by `tAfocal4` (7/7).

**Open for Dave and Mike:** the operating point and the instrument envelope, which are now
one question rather than two.

## S4b.0 The constraint, and what it is written on

**Sky is at −z, so behind the primary is +z.** His three-mirror parent puts M3 at
z = **+0.640 m**; the four-mirror child built from the same front end puts the collimator
at **−0.200 m** and the delivered S4 rung at **−0.442 m**. One extra mirror flips the
parity of the whole back end — after M2 the beam runs +z, the field mirror reverses it,
and the collimator lands *back toward the sky* from wherever the field mirror sits.
Nothing in the S3 packaging check (train length, incidence angles, self-obscuration) is
sensitive to that, which is how it got through.

The constraint has three clauses and they are enforced in three different places, on
purpose:

| clause | where | why there |
|---|---|---|
| `z(last mirror) − z(M1) ≥ 500 mm` | `afocal4_build`, as a **wall** | cheap (pure algebra), so it can sit inside every solver iterate |
| a fold fits on the collimator's exit leg, with daylight against **every** other bundle crossing that station | `afocal4_pack`, at delivery | needs a trace; and the fold's size is a mechanical choice, not an optical DOF |
| the interface pupil and a stated instrument envelope end up **behind M1 and out of the beam** | `afocal4_pack` + `afocal4_s4b` §3, demonstrated | the clause the S3 check did not have at all |

**It is a wall, never a merit term.** The log-merit lesson of §5 rule 2 is that a
mis-scaled term owns the solve; and buildability is not a quantity to trade against
wavefront error in any case — an unbuildable layout is not a worse telescope, it is not a
telescope. `afocal4_build` errors, the solver sees a large finite residual and turns back,
exactly as it does for a degenerate closure.

**Which leg the fold picks off, and why that one.** The fold goes on the collimator's
exit leg — between the last mirror and the interface pupil — which is where Rodgers puts
his own recenter fold, and it is the only choice that moves the pupil. The field mirror
sits *upstream* of the collimator, so no fold downstream can move it; what the constraint
has to deliver for the field mirror is that it too is behind the primary, and
`z_M3 ≥ 500 mm` delivers that for free here, because in every compliant closure the
collimator sits back toward M1 *from* the field mirror (`z_FM > z_M3`).

## S4b.1 The anchor, and what the constraint actually costs

The brief's first thing to try: hold his M1→M2 spacing and his M2→image geometry — which
is to say M2's radius — verbatim, so the front end is exactly the benchmark's, and let the
conics, the field-mirror standoff and φ₄ (consumed by the closure) carry the whole job.

**It closes, everywhere.** His front end admits a compliant closure at *every* operating
point from 50 mm to 343 mm. The compliant band is the field-mirror standoff: compliance
needs the fourth mirror **250–600 mm PAST the intermediate image**, never before it. That
is forced, not chosen — with one more mirror the collimator sits back toward the sky *from*
the field mirror, so the only way to put it behind M1 is to put the field mirror further
behind still.

All three rows at 140 mm, +0.6° bias, same DOFs where comparable:

| | R_M2 mm | s_FM mm | φ₄ /m | WFE nm | blur µm | breathing % | wander µm | M | worst |
|---|---|---|---|---|---|---|---|---|---|
| *S4 rung 3 — NOT BUILDABLE* | *473.0* | *+291* | *+1.78* | *9600* | *167* | *0.113* | *171* | *30.0156* | *135×* |
| **S4b anchor** (his front end) | 468.78 | −381 | +3.97 | **3451** | 1141 | **0.046 ✓** | 1159 | 29.9805 ✓ | 48.6× |
| **S4b variant** (image behind M1) | 447.5 | −50 | +2.11 | 16760 | **149** | **0.149 ✓** | **154** | 30.0047 ✓ | 236× |

**The constraint does not cost image quality. It costs pupil quality, and the exchange
rate runs the opposite way to S4's.** The anchor is 2.8× *better* in wavefront than the
design it replaces and 6.8× worse in pupil. The mechanism is exactly the S3 form
argument, running backwards: a mirror at the intermediate image moves pupil imaging at
first order and leaves image quality nearly untouched *because the marginal ray is small
there* — and the packaging constraint is precisely the requirement that the fourth mirror
NOT be there. Moved 380 mm downstream it acquires a 46 mm footprint, so its conic finally
does wavefront work, and it stops sitting at the field conjugate, so it stops doing pupil
work.

**There is one way to have both, and it is a front-end change.** A slower secondary pushes
the intermediate image further back; once the image itself is ~900 mm behind M1 the field
mirror can sit ON it (s_FM = −50 mm) and the collimator still lands 567 mm behind. That is
the variant row, and it recovers the S4 geometry almost exactly (φ₄ = 2.11 against 1.78,
R_FM = 946 mm against 1122 mm) with the whole train shifted behind the primary. It beats
the unbuildable S4 design on *both* pupil columns. It is paid for in length — a 2.02 m
envelope against 1.78 m for the anchor and 1.69 m for his three-mirror — and in wavefront.

The solver will not find this basin on its own: M2's radius is 4.5% away and the ground
between is not downhill, so it is seeded explicitly (`image_behind_seed_`) and reported
beside the anchor rather than left to luck. Both solves ended on `exitflag 3` — converged
on the merit's own tolerance, at 115 and 77 evaluations — and that is stated rather than
smoothed over: these are local basins, reported as such.

Artifacts: `afocal4_b_anchor.in`, `afocal4_b_frontvar.in`.

## S4b.2 The buildable ladder

His four slides again, with the constraint enforced at every iterate. Same DOF set, same
merit, same rungs; the seed is the compliant one (field mirror 400 mm past the image).

| rung | WFE nm | blur µm | breathing % | wander µm | surface mm | M | worst miss |
|---|---|---|---|---|---|---|---|
| **1  on axis, joint solve** | **554** | 86.6 | **0.101 ✓** | 88.3 | **0.0004 ✓** | **30.0038 ✓** | 7.81× |
| **2  offset +0.6°, FROZEN** | 5384 | 958 | **0.340 ✓** | 971 | **0.016 ✓** | 29.8560 | 75.83× |
| **3  offset, joint re-solve** | 4591 | 1074 | **0.082 ✓** | 1091 | **0.036 ✓** | **29.9810 ✓** | 64.66× |
| **4  + M2/FM/M3 tilt+dec** | 4591 | 1074 | **0.082 ✓** | 1091 | **0.036 ✓** | **29.9810 ✓** | 64.66× |
| TARGET | 71 | 47 | 0.4 | 56 | 0.2 | 30.000 | 1.00× |
| *S4 rung 4 — not buildable* | *9558* | *153* | *0.083* | *156* | *0.016* | *30.0158* | *134.6×* |

**Rung 1 is 2.5× better than S4's** (554 nm against 1392) and its worst miss is 7.81×
against 19.6×. **Rung 4 is bit-identical to rung 3**: the six rigid-body degrees of freedom
bought *nothing at all*, against 0.4% in S4 and 25% in his three-mirror. The S4 explanation
holds and hardens — the residual is a field dependence and a rigid body adds a
field-constant term — but here even the 0.4% is gone, because the compliant layout's
residual is more purely field-quadratic.

The interface plane's refit tilt runs 0.0° / 4.04° / 4.52° across the rungs, against
0.0° / 10.1° / 8.6° in S4 and his own coldstop DAR tilts of 0° / 4.29° / 3.58°. **The
buildable four-mirror wants almost exactly the coldstop tilt Rodgers tunes by hand**; the
unbuildable one wanted twice it.

## S4b.3 The buildable trade — two basins, and the choice the constraint forces

The sweep of §2 stays in the basin its seeder hands it. That turns out to matter more than
usual here, so **both** basins are swept, same machinery, same merit, same gates.

**Basin 1 — his front end, field mirror far past the image:**

| iface mm | φ₄ /m | R_FM m | s_FM mm | WFE nm | blur µm | breathing % | wander µm | collimator behind M1 |
|---|---|---|---|---|---|---|---|---|
| 50 | +3.03 | 0.660 | −741 | 11 776 | 759 | **0.247 ✓** | 774 | 909 mm |
| 90 | +3.84 | 0.521 | −419 | 5040 | 872 | **0.113 ✓** | 887 | 500 mm |
| 140 | +4.58 | 0.436 | −384 | 4591 | 1074 | **0.082 ✓** | 1091 | 501 mm |
| 220 | +3.60 | 0.556 | −300 | 8928 | 800 | **0.098 ✓** | 814 | 533 mm |
| **343** | **+0.03** | **60.0** | −250 | **266** | 761 | 3.875 | 763 | 559 mm |

**Basin 2 — slower secondary, intermediate image behind M1, field mirror on it:**

| iface mm | R_M2 mm | φ₄ /m | s_FM mm | WFE nm | blur µm | breathing % | wander µm | max instrument Ø |
|---|---|---|---|---|---|---|---|---|
| 50 | 603.7 | +1.78 | +21 | 14 481 | 238 | 1.371 | 242 | 0 mm |
| 90 | 448.0 | +2.45 | −50 | 17 066 | 191 | **0.561** | 195 | 0 mm |
| 140 | 448.0 | +2.15 | −49 | 16 367 | 153 | **0.086 ✓** | 157 | 71 mm |
| 220 | 448.6 | +1.68 | −51 | 12 524 | 152 | **0.118 ✓** | 156 | 210 mm |
| **343** | **446.0** | **+1.06** | −41 | 10 521 | **160** | **0.081 ✓** | **164** | **464 mm** |

> **SUPERSEDED IN PART by §S4c.1 (2026-08-04).** These points were solved with a
> finite-difference step that reads the gradient 17% low, and they are not minima. Re-solved
> from three seeds at ~1900 evaluations each, the wavefront column reads
> **12 915 / 17 063 / 15 928 / 12 286 / 10 407 nm** — 0.02% to 10.8% better, with the pupil
> columns inside 2% except the breathing at 220 mm (0.118 → 0.077%) and at 343 mm
> (0.081 → 0.124%). The 220 mm row is delivered by a seed this sweep never had. **The shape
> of the curve and every conclusion drawn from it stand**; the numbers are retracted in
> place and superseded by the §S4c.1 table. The 90 mm design is not under-solved but
> CONSTRAINED — it sits on the packaging wall to the millimetre.

Three things, and they are the study's answer under the constraint.

**1. Basin 1 never buys blur.** Across the whole curve the pupil blur sits between 759 and
1074 µm against a 47 µm target — a factor of 16 at best — while the breathing goes to
0.08%. The fourth mirror, forced off the field conjugate, buys **magnification stability
and nothing else**. Basin 2 holds blur at 152–191 µm throughout: a factor of 5–7 better,
for 2.5–3× the wavefront error.

**2. At 343 mm, basin 1's fourth mirror becomes a flat** — φ₄ = +0.03 /m, R = 60 m — the
wavefront falls to **266 nm** and the breathing reverts to the three-mirror's **3.875%**.
That is the far end of the S4 curve reappearing intact: with the fourth mirror's power
spent to zero, the design *is* his three-mirror, and its pupil is his three-mirror's.

**3. So the choice the constraint forces is visible in one row-pair.** At 343 mm — the only
operating point where the whole package closes around a real instrument — there are two
buildable designs:

| at 343 mm, both BUILDABLE | WFE nm | blur µm | breathing % | wander µm | max instrument Ø |
|---|---|---|---|---|---|
| basin 1, φ₄ → 0 (a flat: his three-mirror) | **266** | 761 | 3.875 | 763 | unlimited |
| basin 2, φ₄ = +1.06 (a real fourth mirror) | 10 521 | **160** | **0.081 ✓** | **164** | 464 mm |

**The fourth mirror's pupil control costs a factor of 40 in wavefront, and doing without it
returns you to the telescope Rodgers already has.** That is the buildable form of the S4
headline, and it is harsher: unconstrained, the same exchange cost a factor of 6.

### The feasible window, and what closes it

The packaging gate has three clauses and they do not bind in the same place.

* **`z_M3 ≥ 500 mm`** binds nowhere on either curve once the seeder is compliant — every
  delivered point clears it, by 0 to 1900 mm.
* **Fold daylight** binds at the short end: at 50 mm standoff the gap between the
  collimator's exit leg and the feed leg is **−2.7 mm** at every station — there is no room
  for a fold at all, so the pupil cannot be taken out of the beam.
* **Instrument volume** binds everywhere else, and it is the clause that sets the window.
  The fold needs ~20 mm of lever arm for its own body; whatever is left of the interface
  standoff becomes the pupil's lateral offset once folded, and the largest instrument that
  clears the near-axis bundles is `2 × (offset − beam radius)`:

| iface mm | 50 | 90 | 140 | 220 | 343 |
|---|---|---|---|---|---|
| largest instrument Ø, basin 2 | 0 | 0 | 71 mm | 210 mm | **464 mm** |

**The interface standoff sets the instrument's girth, and that is a constraint the S4 trade
never had to face.** For a 300 mm instrument the window is **iface ≳ 250 mm**; his own
three-mirror, at 344 mm, admits 344 mm. Below ~90 mm nothing fits at all. The pupil metrics
want a short standoff and the package wants a long one, which is a second exchange rate
crossing the first.

## S4b.4 The folded demonstration

> **SUPERSEDED 2026-08-30 as a RECIPE; retained as the null demonstration it
> was.** The single fold below satisfies constraint clause 3 and its
> fold-is-null check is still the machinery that caught two real defects (see
> "The check earned its keep twice"). What it is no longer is a packaging
> answer. The 2026-08-30 packaging stage measured that this flat **does not
> touch the depth** (the deepest optic is identical to the last digit), costs
> the shroud (the instrument leaves *radially*, envelope Ø2.779 × 2.626 m
> against Ø1.120 × 3.825 m unfolded), and **does not itself fit**: over the
> field box its own footprint is 103.7 mm in radius and it clips the FM→M3
> feed beam by −73.6 mm, where `afocal4_pack` had measured 17.5 mm of daylight
> and passed it. *A margin is a number, not a body.* The four-fold Path A that
> replaced it is in turn superseded — see `packaging/README.md` §3 — because
> the clearing stage removed the interference by design instead, with an
> extraction tilt, and the swing packages the back end with **zero** flats.
> Full account: `clearing/README.md` and § CLEARING below.
>
> The interface-pupil coordinates and instrument z-slab quoted below for the
> FOLDED deck do not reproduce from the committed file. The dated correction is
> at the foot of this section (signed off 2026-08-31); the *conclusion* those
> numbers support — the instrument sits entirely behind the primary — is
> unaffected either way.

Constraint clause 3 says *demonstrated, not asserted*, so the flat is inserted, the
prescription is emitted, and the design is re-scored on the same kernel — because a
nominally null fold is not a free fold (e2e2 s3). The subject is the **basin-2 design at
343 mm**: the buildable point that keeps a real fourth mirror and passes the whole gate.

`afocal4_b_final.in` (unfolded) and `afocal4_b_final_folded.in` (folded) are committed side
by side.

**The fold is null, measured:**

| | WFE nm | blur µm | breathing % | wander µm | M |
|---|---|---|---|---|---|
| unfolded | 10 520.53 | 159.71 | 0.0807 | 163.90 | — |
| folded | 10 520.53 | 159.71 | 0.0807 | 163.90 | — |

Every column identical to the printed precision, and unchanged when the fold's aperture is
quadrupled — so it is not clipping either.

**And the package lands where it has to.** Interface pupil at
[+0.304, −0.004, +0.614] m, the stated 1000 mm instrument envelope running to
[+1.304, −0.017, +0.614] m, its z-slab **+0.464 … +0.764 m — entirely behind the primary**,
and the closest approach of any other traced bundle **+137 mm**. Renders: `afocal4_b_layout_
{unfolded,folded}.png` (yz and xz, with the envelope drawn) and `rodgers2_final_folded.png`
for the deck.

### The check earned its keep twice

**Once on a real defect of mine.** The first folded emission read 3.9% breathing → 23% and
9% shifts in blur and wander, on a fold that is an exact isometry. The cause was the
interface plane's placement: the unfolded rule projects the *last mirror's vertex* onto the
exit chief, and with a fold in between the code was projecting the *fold's* vertex, which
is a different offset. Fixed by applying the fold's own reflection to the unfolded plane
instead of re-deriving it — the unfolded pose then moves by **1.3e-18 m** (so every
committed deck re-emits unchanged) and the folded one agrees to **1.1e-12**.

**Once on a conditioning limit of the pupil metric.** After that fix, one design still
reads a fold-induced change: basin 1 at 343 mm, where φ₄ has gone to zero. Its wavefront
and magnification are exact under the fold (1.4e-10, 4.4e-16) but its blur, wander and
breathing move 9%, 9% and 6×, at any fold aperture. That design's exit pupil sits
essentially *at* the interface plane, so the chief-normal magnification is a ratio of two
ill-conditioned footprints, and a determinant −1 mapping is enough to move it. **The
number is not trustworthy there, and the fold-is-null check is what says so** — it doubles
as a conditioning check on the pupil metric. The affected row is the one already flagged as
"his three-mirror wearing a flat"; its pupil column should be read as the three-mirror's,
which is what §S4b.3 says it is.


> **DATED CORRECTION (2026-08-31).**  The interface-pupil coordinates and the
> instrument z-slab quoted above for the FOLDED deck do not reproduce from the
> committed file.  Measured by `packaging/check_record` on
> `afocal4_b_final_folded.in`: interface pupil **[+0.2483, −0.0051,
> +1.3782] m**, envelope ending **[+1.1990, −0.3151, +1.3782] m**, z-slab
> **+1.2282 … +1.5282 m**; the fold and the interface plane are both at
> z = +1.3782 m.  The statement the numbers support is unaffected — every z is
> positive, so the instrument is entirely behind the primary — and the
> fold-is-null result is a measurement of a different quantity.  The recipe
> itself is superseded (see the head of this section), so this corrects the
> historical record, not a live design.
>
> *(Signed off by Dave 2026-08-31 via CC; `macos/BRIEF_afocal4_wall.md` §7.)*

## S4b.5 The Mersenne, closed a second time

The double Mersenne was closed in §4 on wavefront error: four conics take it from 59.4 µm
to 35.1 µm against a 71 nm target, a factor of 1.7 where a factor of 500 was needed.

**It also fails the packaging constraint structurally.** Its second confocal pair lives
inside the M1–M2 space — the closed layout puts M3 at **z = −0.542 m** and M4 at
**z = −0.942 m**, i.e. 942 mm *in front of* the primary — and no gap, stage split or conic
moves them behind it, because the form's entire compression happens before the beam ever
gets back to M1. `afocal4_mersenne` therefore runs with `P.pack.enforce = false`, so the
experiment stays reproducible and its verdict stands as measured; the constraint closes it
again on a second, independent ground.

## S4b.6 Rules earned, on top of the S4 eight

9. **Package as a wall in the closure, never as a term in the merit.**
   *Alternative rejected:* a penalty on `z_M3`. The log-merit lesson (rule 2) is that a
   mis-scaled term owns the solve, and buildability is not tradeable against wavefront
   error in any case — an unbuildable layout is not a worse telescope, it is not a
   telescope. The wall is pure algebra on the closure's own stations, so it costs nothing
   to sit inside every iterate, and the solver turns back on it exactly as it does on a
   degenerate closure.

10. **A wall needs a compliant seed, or it is a cage.**
    Dropped in at a non-compliant point, every finite-difference direction is an error,
    the Jacobian is walls all the way round, and `lsqnonlin` hands back the seed it was
    given. That is a *seeding* failure and would have been reported as "this operating
    point has no design": measured, the warm-started trade lost **four of its five points**
    that way, every one of which `afocal4_pack_seed` then closed. Seed the search inside the
    feasible region and let the wall bound it.

11. **A sign change is not a root.**
    `d(φ₄)` is a rational function, so it changes sign across its *poles* as well as its
    zeros, and `fzero` converges onto a pole quite happily — returning a layout with a
    collimator 1e14 m away. S4 never noticed because its roots were the first sign change;
    the constraint moved them from φ₄ ≈ 2 to ≈ 4, past a pole, and the builder started
    failing outright. Close and CHECK every candidate; take the lowest power that is
    actually a telescope.

12. **Sweep the basin, then check it is the right basin.**
    The trade of §S4b.3 was first run only in the basin its seeder produced — his front
    end, field mirror past the image — where pupil blur never falls below 759 µm. A
    second basin, seeded by hand at a slower secondary, holds 152 µm across the same
    curve. A trade curve swept in one basin is a statement about that basin, and saying
    so requires having looked at another.

13. **Check a null operation is null, and believe the check when it fires.**
    A flat fold is an exact isometry and cannot change anything. Emitting it and
    re-scoring anyway caught, first, a real placement defect in this code (§S4b.4) and
    then a conditioning limit of the pupil metric on a degenerate design. Both were
    invisible to every other gate in the study.

14. **Packaging includes where the INSTRUMENT goes.**
    Train length, incidence angles and self-obscuration all pass on a design whose
    instrument sits in its own incoming beam — that is the S3 gap that let the whole S4
    trade through. The check that catches it needs the fold's lever arm and the envelope's
    girth, and its useful output is not a verdict but a number: the largest instrument
    that fits at this standoff.

---

# S4c RESULTS — the rim, the zone, and how far the second basin actually goes

Answers the S4c brief (Dave, 2026-08-04): close the S4b caveat that basin 2 was
hand-seeded and every one of its solves stopped on `exitflag 3`; implement the ruling that
the pupil object for a **coldstop** metric is the **rim** of the primary rather than its
pole; and re-score the 343 mm fork under it.

Reproduce: `afocal4_basin2` (one process per trade point, then `afocal4_basin2_merge`) and
`afocal4_fork`. Every number lives in `afocal4_basin2.mat` / `afocal4_fork.mat`.

## One page

**The `exitflag 3` caveat closes, and it closes as a stall rather than a wrong answer.**
The S4b basin-2 solves stopped because a 3e-3 forward difference reads the merit's gradient
17% low on an objective that is smooth to 1e-5 — not because the designs were minima.
Re-solved from three seeds at ~1900 evaluations per point, basin 2's wavefront column
improves by **0.02% to 10.8%** and every pupil column by less than 2%, except the breathing
at 220 mm (0.118 → 0.077%) and 343 mm (0.081 → 0.124%). **Nothing concluded from basin 2
changes.** Its flat wavefront column was not an artefact of under-solving: 90 mm, its worst
point, moves 0.02% under a 17× larger budget — and turns out to sit **on the packaging
wall**, to the millimetre, rather than at a minimum.

**The rim ruling is implemented, and the anchor is not where the action is.** Rodgers'
declared stop and the rim plane traced off his primary agree to **1.7e-13 mm**, so the
convention is his deck's own. But moving the pupil object from the pole to the rim changes
the measured blur by **0.01–0.06%** across his whole ladder and both fork branches, while
scoring the outer 10% of the pupil instead of all of it changes it by **10–34%**. Where the
anchor does bite is the convergence surface — 15.3 µm rms of imaged primary sag under the
surface anchor against 0.11 µm under the rim, a factor of 140 — which is what the anchor
exists for and where the thin-primary gate is written.

**The fork does not soften.** At 343 mm the flat branch's rim-edge image is **11% worse
than its own aperture average** (845.7 against 761.0 µm) and still 18× its target, so the
edge metric does not rescue doing without the fourth mirror. The powered branch degrades
*more* toward the rim (34%), so what the fourth mirror buys falls from **4.8× to 4.0×**
while its wavefront price stays at **39.1×**. Measured on the annulus a coldstop actually
masks, the case for it is weaker than the S4b table made it.

**One spec moves.** The interface convergence surface against its ideal image reads
0.0174 mm surface-anchored and **0.1853 mm** rim-anchored — 12× inside the 0.2 mm target
against 1.08× inside it.

**Status: delivered.** The premise §S4c.0, the long solve §S4c.1, the convention §S4c.2,
the fork §S4c.3, eight more earned rules §S4c.4. Gated by `tPupilMap` (12/12) and
`tAfocal4` (8/8).


## S4c.0 The premise, measured

The ruling rests on two claims that can be checked rather than repeated.

**His decks already declare the rim plane.** His stop is typed into the source block 50 mm
ahead of the pole. The sag of an R = −2500 mm parabola at r = 500 mm is 50.000 mm exactly.
The rim plane `pupil_map` traces off his primary lands **1.7e-13 mm** from that stop — two
independently supplied numbers, one plane. (The decks `afocal4_build` emits declare their
stop at the **vertex**, so on those the rim sits 50.000 mm ahead of the declared stop.
That is why the rim is measured off the traced bundle and never read from `ApStop`.)

**And the choice is resolvable.** The pupil image's own depth of focus, λ/(2·NA_field²),
against the primary's sag imaged at m²:

| deck | NA_field | λ/(2·NA²) | M1 sag P-V | imaged at m² | resolvable? |
|---|---|---|---|---|---|
| basin 1, flat M4 (343 mm) | 0.1783 | 15.7 µm | 48.375 mm | 60.5 µm | yes, 3.8× |
| basin 2, powered M4 (343 mm) | 0.1874 | 14.2 µm | 48.375 mm | 54.5 µm | yes, 3.8× |
| his 3-mirror parent | 0.1781 | 15.8 µm | 48.375 mm | 60.6 µm | yes, 3.8× |

The imaged object is about four times deeper than the depth over which the pupil imager
holds focus, so rim-conjugate and pole-conjugate really are different states of it. (Dave's
~30 µm is the λ/NA² form of the same quantity, twice the column above; the conclusion is
the same either way.)

## S4c.1 Basin 2, solved long — what the exitflag was hiding, and what it was not

**Why the S4b solves stopped, measured before anything was changed.** At the 343 mm
basin-2 design the merit is smooth in the DOFs down to a 1e-5 scaled step: the
central-difference slope in K_M2 reads −1.4490 at 3e-3 and −1.4490 at 1e-5, four figures
across two decades, with no noise floor between. The study's default **forward** difference
at 3e-3 reads **−1.198** — 17% low. That gradient error, not the objective's shape, is what
stalls `lsqnonlin` on its `FunctionTolerance`. And the slope itself is the finding: −1.45
is not a stationary point, so the delivered design was never a minimum of anything.

**What was run.** Every trade point re-solved from three independent seeds — the S4b
delivered design restarted, the compliant image-behind-M1 seeder, and the *second*
compliant closure (a different field-mirror branch, not a neighbouring radius) — with
forward differences at 3e-4, `FunctionTolerance` 1e-8, `StepTolerance` 1e-9, restart rounds
of 250 evaluations each, and a central-difference polish on the winner. ~1900 evaluations
per point against S4b's ~110. The merit, the sampling, the DOF set and the packaging wall
are untouched: a long solve that changed the objective would not be comparable to the curve
it is correcting.

**The result: the curve moves, and it does not move the answer.**

| iface | WFE nm, S4b | WFE nm, S4c | Δ | blur µm | breathing % | wander µm | merit gained | delivered by |
|---|---|---|---|---|---|---|---|---|
| 50 | 14 481 | **12 915.5** | −10.8% | 238.1 | 1.342 | 241.1 | 2.09% | restart |
| 90 | 17 066 | **17 063.3** | −0.02% | 190.6 | 0.561 | 194.8 | 0.02% | restart |
| 140 | 16 367 | **15 927.6** | −2.7% | 152.9 | 0.082 ✓ | 157.4 | 0.86% | restart |
| 220 | 12 524 | **12 285.8** | −1.9% | 154.8 | 0.077 ✓ | 159.0 | 0.78% | **third seed** |
| 343 | 10 521 | **10 407.0** | −1.1% | 157.0 | 0.124 ✓ | 161.2 | 0.33% | restart |

Basin 2 still runs 10.4–17.1 µm of wavefront error against a 71 nm target, still holds its
pupil blur at 153–238 µm against 47, and is still the branch you buy pupil control with.
**The flat wavefront column was not an artefact of under-solving** — 90 mm, the worst point
on it, moves by 0.02% under a 17× larger budget.

**Two rows do move enough to matter.** At 50 mm the wavefront error falls 10.8%, and at
220 mm the *third seed* — the one the S4b sweep never had — wins outright, taking the
breathing from 0.118% to 0.077% at 1.9% less wavefront error. Every other column is inside
2%.

**Where each design actually stands, by measurement rather than by exit code.** After the
polish, the gradient at each delivered design was taken by central differences at a 1e-4
scaled step, and the merit walked by hand along −g:

| iface | \|g\| (free DOFs) | best gain along −g | what it is |
|---|---|---|---|
| 50 | 0.085 | none (−2e-6) | a floor |
| 90 | 3.91 | undefined | **on the packaging wall** — `t_M1M2`'s step leaves the feasible set |
| 140 | 1.44 | 1.3e-5 | a floor in the descent direction |
| 220 | 1.07 | 2.4e-5 | a floor in the descent direction |
| 343 | 0.80 | none (−7e-6) | a floor in the descent direction |

**90 mm is the interesting one: it is not converged, it is CONSTRAINED.** Its closure puts
M3 at z = +0.500 m — the packaging minimum, to the millimetre — and the finite-difference
step in the M1–M2 spacing walks it into the beam. The merit still has a gradient there;
descending it is simply not allowed. That is a different sentence from "the solver gave up",
and it is the sentence the S4b caveat should have carried.

**And a warning about the probe itself.** Walked along −g, *every* S4b design reported
"nothing available": 2e-5 to 1.6e-4 of the merit, at all five standoffs. The Gauss-Newton
restarts then took 0.02–2.1% off the same designs. **Steepest descent is not a convergence
test on an ill-conditioned least-squares problem** — the probe is a lower bound, and it is
reported as one.

**Gates, all passed.** The anchoring residual is 0.09–0.21 µm at every delivered design
(0.1 µm sound, tens of mm on a scrambled solve — S4 rule 8); the on-axis paraxial check
traces M = 30.07–30.13× against the closure's 30.000 (+0.24 to +0.42%, real-ray against
paraxial at f/1.25); and every delivered design clears the packaging gate, with M3 500 mm
(90) to 5210 mm (50) behind the primary.

**One seed is worth Dave's attention.** At 343 mm the third seed reaches essentially the
same wavefront error as the delivered design — 10 331 nm against 10 402 — while holding
the magnification breathing at **0.0117%**, ten times better. The merit cannot see the
difference, because `P.merit_floor` stops a term earning credit once it is twice inside
target and both are far inside. If breathing is worth more than the merit says, that design
is on the table and it is committed as part of `afocal4_basin2_343mm`'s seed record.


## S4c.2 The rim convention, and what it actually moves

`pupil_map` grows two opt-in additions and no new defaults. `'anchor','rim'` anchors the
cones on the flat plane through the object element's **rim**, normal to that element's own
axis; `.rim_zone` reports blur and wander over the outermost fraction of the pupil radius
(10% by default) beside the full-aperture numbers, under **every** anchor. The surface
anchor's output is bit-identical to the pre-change function on every field of both fixture
decks, and `tPupilMap` now pins it there.

**The rim plane is measured, not declared, and that mattered.** The first implementation
took the rim from the outermost traced ray. A circular grid's last ring lands 1.15 mm
inside a 500 mm aperture, so the sag came back 0.23 mm short — a sampling artefact wearing
geometry's clothes, and enough to break the identity with his declared stop by four orders
of magnitude. The sag is now fitted in r² across every ray that reaches the element and
evaluated at the **declared beam edge**; `.edge` records which edge was used.

**A flat object has a flat ideal image.** Under any flat anchor the curved-object
correction `.ideal.beta` is NaN and says so, and the residual is taken against the best-fit
**plane** instead. Same sentence, different object — and it is what makes the convergence-
surface term quotable in both conventions.

**The anchor changes almost nothing; the zone changes a lot.** Measured on Rodgers' own
ladder, at the same 21-node lattice as the §4 baseline table:

| variant | blur, surface | blur, rim | blur, RIM ZONE | edge penalty | wander, rim zone |
|---|---|---|---|---|---|
| S1 on-axis | 152.9 | 152.9 | **195.0** | 1.275× | 197.0 |
| S2 offset | 801.6 | 801.1 | **877.8** | 1.096× | 880.8 |
| S3 newconics | 774.8 | 774.4 | **854.0** | 1.103× | 857.1 |
| S4 tilt/dec | 468.6 | 468.5 | **588.0** | 1.255× | 591.5 |

Moving the object plane from pole to rim moves the blur by **0.01–0.05%** — less than the
anchoring residual on the offset variants. Looking at the outer 10% of the pupil instead of
all of it moves it by **10–27%**. So on these designs "the pupil is at the rim" is a
statement about *where on the pupil you measure*, not about *which plane you measure from*.

Where the anchor does bite is the column it exists for: the **convergence surface**. Under
the surface anchor the exit sag carries the primary's imaged sag — 15.3 µm rms on S1 —
and under the rim anchor it does not: 0.11 µm, a factor of 140. That is also how the
thin-primary gate is written: shrink the used aperture tenfold and the anchor-to-anchor
difference in the convergence surface falls to 1% of itself, which is the statement that
the two conventions differ by the imaged object sag *and by nothing else*.

**His best variant has the worst rim.** S4's tilt/dec re-solve takes the aperture-average
blur from 774.8 to 468.6 µm — 40% — while its edge penalty rises to 1.26×, the highest on
the ladder after the on-axis case. Rigid bodies buy the middle of the pupil.


## S4c.3 The 343 mm fork, re-scored in both conventions

The fork is the S4b headline: at the only standoff where the package closes around a real
instrument, a fourth mirror that has gone to a flat gives 266 nm of wavefront error and his
three-mirror's pupil, and a real fourth mirror gives the pupil for a factor of 40 in
wavefront. Dave's question is whether the physically-correct metric softens it. The powered
branch below is the LONG-SOLVED design of §S4c.1; the flat branch and his parent are the
committed S4b decks, unchanged.

| at 343 mm | WFE nm | blur, surface | blur, rim | blur, RIM ZONE | wander, rim zone | breathing % |
|---|---|---|---|---|---|---|
| basin 1, flat M4 | 266.0 | 761.0 | 760.6 | **845.7** | 848.6 | 3.875 |
| basin 2, powered M4 | **10 407.0** | 157.0 | 157.1 | **210.7** | 216.2 | 0.124 ✓ |
| his 3-mirror parent | 429.1 | 794.0 | 793.6 | **878.1** | 881.1 | 3.966 |
| target | 71 | 47 | 47 | 47 | 56 | 0.4 |

**The anchor is a no-op and the zone is not.** Moving the pupil object from the pole to the
rim changes the blur by **−0.055%** on the flat branch and **+0.025%** on the powered one.
Looking at the outer 10% of the pupil instead of all of it changes it by **11%** and
**34%**. Every effect the rim convention has on this fork, it has through the zone.

**The answer to the question, in the form it was asked.**

* *Does the flat branch's rim-edge image look better?* **No — it looks worse.** Its edge
  blur is 845.7 µm against its own 761.0 µm average, 11% worse, and **18× its target**. The
  metric that asks only for edge conjugacy does not rescue the design that declines pupil
  control.
* *Does the powered requirement relax?* **Partly, and against the fourth mirror.** The
  powered branch degrades *more* toward the rim (34% against 11%), so the advantage it buys
  falls from **4.8× on the aperture average to 4.0× at the rim** — the rim metric takes 17%
  off what the fourth mirror is worth. The wavefront price is unchanged at **39.1×**.

So the fork does not soften. Measured on the annulus a coldstop actually masks, the case
for the fourth mirror is *weaker* than the S4b table made it, not stronger.

**One column the anchor does move, and it is the one the anchor exists for.** The
convergence surface against its ideal image reads 0.0174 mm under the surface anchor and
**0.1853 mm** under the rim — a factor of 10.7, and the difference between 12× inside the
0.2 mm target and 1.08× inside it. A flat object's ideal image is a plane, and the powered
design does not deliver one; the curved-object reference was quietly absorbing that. Anyone
writing a surface-figure spec for the interface pupil should write it in the rim
convention, where it nearly binds.

### The rim-weighted re-solve — the metric does not move the design

The zone is where the two conventions differ, so the powered branch was re-solved against
it: same log-domain merit, same DOFs, same sampling, with the blur and wander terms scoring
the rim-anchored outer 10% annulus instead of the whole aperture. Seeded from the
long-solved 343 mm design; 504 evaluations.

| at 343 mm | WFE nm | rim-zone blur | rim-zone wander | full blur | breathing % | surface vs ideal, rim |
|---|---|---|---|---|---|---|
| surface-weighted (§S4c.1) | **10 407.0** | 210.7 | 216.2 | 157.0 | 0.124 | 0.1853 |
| rim-weighted | 10 424.2 | **209.8** | **215.3** | 156.4 | 0.136 | 0.1855 |

**0.4% on the very quantity it was asked to minimise**, 0.17% the wrong way in wavefront
error, and the exchange rate against the flat branch reads **39.2× against 39.1×**. The
rim-weighted merit does not buy a different design. That is the strongest form of the
answer: the rim metric neither softens the fork nor moves the design that has to pay for
it — the exchange rate is a property of the optics, not of which part of the pupil the
metric is read on.



## S4c.4 Rules earned, on top of the S4 eight and the S4b six

15. **An exit code is not a convergence test; a gradient is.**
    Every S4b basin-2 solve stopped on `exitflag 3`, which reads as convergence and was
    reported as such with a caveat. Measured, the merit at those points had gradients of
    0.16 to 7.2 in scaled DOFs. What stopped the solver was a 17%-low forward difference at
    a 3e-3 scaled step, on an objective that is smooth to 1e-5. *Alternative rejected:*
    trusting the exit code and tightening `tol_fun` alone — that changes when the solver
    gives up, not what it can see.

16. **And steepest descent is not a convergence test either.**
    Walked along −g, all five S4b designs reported 2e-5 to 1.6e-4 of merit available, i.e.
    "converged". A Gauss-Newton restart of the same designs then took 0.02–2.1%. On an
    ill-conditioned least-squares problem −g is a poor direction; the probe is a lower
    bound and is now labelled as one.

17. **A round that ran out of budget is not a plateau.**
    The first version of the restart loop declared a plateau whenever a round bought less
    than 1e-6 — including rounds that hit `MaxFunctionEvaluations` mid-line-search, where
    "no progress" means the budget was short. Plateau now requires `exitflag ≠ 0`.

18. **A NaN in a gradient probe is a CONSTRAINT, not a failed measurement.**
    At 90 mm the delivered design sits on the packaging wall to the millimetre (M3 at
    z = +0.500 m against a 500 mm minimum), so the finite-difference step in the M1–M2
    spacing is unbuildable and `|g|` is undefined in that DOF. Reporting that as "not
    converged" is backwards: there is no interior minimum to converge to along it. The
    verdict now names the walled DOFs, and the descent probe refuses to fabricate a
    direction out of the ones that remain.

19. **Sampling can masquerade as geometry.**
    The first rim implementation took the rim from the outermost traced ray. A circular
    grid's last ring lands 1.15 mm inside a 500 mm aperture, so the sag came back 0.23 mm
    short — and broke the identity with Rodgers' declared stop by four orders of magnitude.
    The r² fit is now evaluated at the *declared* beam edge, and the identity closes to
    1.7e-13 mm.

20. **Ask whether a new metric changes the NUMBER before believing it changes the ANSWER.**
    The rim ruling is about the pupil OBJECT, so the natural expectation is that the anchor
    is what matters. Measured, the anchor moves the blur by 0.05% and the zone moves it by
    10–34%. Had the zone not been implemented alongside, "we re-scored under the rim
    convention" would have been a true sentence about a metric that had not moved.

21. **A seed that fails is not a second basin.**
    The 343 mm cold seed reported blur 44 mm and wander 68 mm — and an anchoring residual of
    **151 mm** against 0.09 µm on the sound seeds. Averaged into a seed-to-seed spread it
    manufactured a 70% "basin" disagreement out of one failed solve. The basin report now
    computes its spread over the sound seeds only and states how many it excluded.

22. **Two scoring modes must carry one field set.**
    `afocal4_score`'s WFE-only branch fills a blank list so its struct matches the full
    score's; adding the rim fields to one branch and not the other would have broken every
    `arr(k) = S` in the ladder — and only when reached, an hour into a sweep. Gated now.


---

# CLEARING RESULTS — the collimator was standing in its own feed beam

Answers `macos/BRIEF_afocal4_clear.md` (Dave, 2026-08-30) and its follow-on
`macos/BRIEF_afocal4_wall.md`. The full numbers-first accounts are
`clearing/README.md` and `wall/README.md`; this section is the *canonical*
record — what the defect was, why it is structural, which remedies are retired
and on what measurement, what the delivered design costs, and what the standing
gate now refuses.

Every number here is engine truth — traced rays and the engine's own element
getters — measured over the **whole 0.5° × 0.5° field box**, never the deck's
own single field. No `.in` file is text-parsed for a geometric claim.

> The one place a prescription IS read as text is design **recovery** —
> `wall_recover` and `afocal4_clearing`'s `recover_` take the conics, `R_M2`,
> `t_M1M2` and the interface standoff off a committed deck so the design struct
> behind it can be rebuilt (RESULTS rule 9). That is not a geometric
> measurement, and it is not trusted either: the recovered struct is rebuilt and
> compared with the file **byte for byte** before anything is measured. Read the
> spacings from `zElt`, never from the vertices — the builder poses the
> interface plane on the traced chief, so on this deck the last mirror's vertex
> is 359 mm from the interface vertex while the standoff is 343 mm, and the
> vertex reading silently rebuilds a different design. It cost the clearing
> stage a whole scan.

## C.0 The defect, and the number that hid it

On the committed 343 mm family-2 design — `afocal4_b2long_343mm.in`, the deck
the S4b/S4c trade delivered, **unfolded, before any packaging** — the
M2 → field-mirror feed beam runs **through the collimator's own glass**.

| what is measured | floor |
|---|---|
| declared body model, 1.15 × union footprint + 15 mm | **−79.89 mm** |
| **bare lit glass**, 1.00 ×, 0 mm | **−55.36 mm** |
| what **one** field sees | **−5.9 … +26.8 mm** |

**That last row is the whole reason it shipped.** Per field the collimator sits
inside the feed cone with 10.8 mm of daylight all round, passed in a
27.8–55.6 mm annulus. But a *monolithic* collimator must cover its **union**
footprint over the field box — 17.0 mm per field, **87.0 mm** over the nine —
and that glass is exactly where the other fields' feed beams pass. A
centre-field check passes the design. `afocal4_pack` ran precisely such a
check: it asks where there is *daylight* and never asks whether a *part* is
already in the way.

> **A margin is a number, not a body.** A gate that reports clearance without
> sizing the part it is making room for can pass a design nobody can build.

Real hardware would vignette a fraction of the field box; the trace does not
show it, because bodies are not obscuring elements.

## C.1 Why — the field-walk ratio law

On a coaxial train every chief-ray height is proportional to its field angle,
so a part's union footprint and a beam's union footprint **at the same
station** are two scaled copies of the *same* field box, anchored on the axis.
Write each centroid as

```
    centroid(theta) = c * theta + o
```

* `c` — the **field-proportional walk**. On a coaxial train this is all there is.
* `o` — the **field-independent offset**. Identically zero on a coaxial train,
  and the *only* quantity that can defeat the law.

Two scaled copies of the box `[B−A, B+A]` are disjoint only if
`c_beam / c_body > (B+A)/(B−A)`. Measured:

| quantity | value |
|---|---|
| field box in the bias direction | 0.3500 … 0.8500 deg |
| **ratio the box demands** | **2.4286** |
| walk, collimator body | 11.1179 m/rad |
| walk, feed beam at the collimator's plane | 14.4811 m/rad |
| **ratio achieved** | **1.3025** |
| field-independent offset | +4.46 mm (i.e. zero — the train is coaxial) |
| centre-set gap, minus the two beam radii | **−107.15 mm** |
| residual of the walk + offset fit | 1.2e-03 m |

**And one of the two scales is pinned by the customer interface.** The chief
converges from the last powered mirror to the exit pupil `iface` away at the
exit angular magnification, so

```
    c_body(collimator) = M * iface = 30 * 0.343 = 10.29 m/rad
    measured                                      11.12 m/rad   (+8.0 %)
```

The pin is paraxial and the measurement is a traced footprint centroid, so the
8 % *is* this design's own pupil aberration — the quantity the fourth mirror
exists to control. What matters is that the **scale is set by the interface
specification**, not by anything a layout change can reach. That is why it is
the *collimator* standing in the beam and not some other part, and why the
interference worsens with the interface standoff — the same knob the S4 ruling
already carries as its operating point, now pulling a third way.

**What the law rules out: every remedy whose separation is proportional to
field.** A different collimator station moves `c_beam` by the chief slope
alone; a different interface standoff moves `c_body`; a flat fold moves
*neither*, because an isometry carries both copies together. What it does not
rule out is a **tilt**, whose `o` does not shrink with the field angle.

## C.2 The three leverages, as measurements

### C.2a Leverage 1 — one flat extraction fold. RETIRED, exactly.

A fold inserted *between* the two conflict partners re-routes one of them, so
"an isometry carries every clearance across unchanged" genuinely does not
apply — which is why it was tried first. Let `LEG` be the M2 → field-mirror
spacing and `B` the field-mirror → collimator spacing; on the folded axis the
collimator sits **exactly on the flat** when the flat is `LEG − B` along the
leg (here 2.3654 m, **0.808 of the leg**). Either side of that station:

| where the flat goes | what binds | floor |
|---|---|---|
| before 0.808 LEG | the fold has not separated the partners | **−79.89 mm — the parent's own number** |
| after 0.808 LEG | the return beam comes back through the flat, which is union-sized (~250 mm across) | −76.9 … −103.4 mm |
| very early | the flat lands in the M1 → M2 beam | −77 … −111 mm |

The two conditions are complementary and meet at a single station, so there is
no window. **Best over every station and both turn directions: −79.89 mm.** A
flat buys exactly nothing. The obvious escape — put the flat far enough before
the focus that the returning beam misses it sideways — is closed by the same
law: the ratio of those two bundles only reaches 2.43 about a metre *past* the
field mirror, well beyond where the collimator is allowed to be.

### C.2b Leverage 2 — the collimator station. RETIRED, with the figure.

Sweeping the field-mirror standoff moves the collimator through **1.40 m** of
z. Over all of it the walk ratio reaches **1.48** against the 2.4286 the field
box demands; the demanded footprint runs 83.6…100.3 mm against 23.0…34.1 mm
available. *The curves never cross* (`clearing/afocal4_clear_scan.png`).

Sweeping the interface standoff instead crosses the ratio between 140 and
90 mm, exactly as the law predicts — but even at 50 mm the declared body still
does not clear (−13.17 mm on the collimator pair, −39.43 mm on the gate as a
whole); only bare glass does, by 7 mm.

**Every point of the committed trade curve fails the gate:**

| deck | z_collimator (m) | collimator pair, bare | collimator pair, body | **GATE (all pairs)** | ratio |
|---|---|---|---|---|---|
| 50 mm | **+5.2101** | +55.49 | **+36.77** | **−29.04** | 11.74 |
| 90 mm | +0.5001 | −13.83 | −33.40 | **−33.40** | 3.351 |
| 140 mm | +0.6829 | −33.43 | −54.18 | **−54.18** | 2.328 |
| 220 mm | +0.8737 | −45.61 | −67.51 | **−67.51** | 1.660 |
| 343 mm | +1.3234 | −55.36 | −79.89 | **−79.89** | 1.303 |

At 50 mm the *collimator pair* genuinely clears (+36.8 mm, ratio 11.7) — but
its collimator sits **5.21 m** behind the primary and its field mirror at
5.95 m, a back end nearly six times the M1–M2 spacing and not a package; and
the deck still fails, at −29.0 mm, on a *different* pair. **Retreating along
the operating point is not a way out**, and that row is why both columns are
reported: the one-column version of this table reads "50 mm clears".

### C.2c Leverage 3 — the extraction tilt. DELIVERED.

`clear_tilt` swings one mirror about the point **where the chief ray actually
strikes it** — read from the traced ray history, never from the vertex —
computes the new outgoing chief from the *rotated local surface normal* (itself
engine truth, `N = unit(d_in − d_out)`), and re-poses every downstream element
by the rotation carrying the old outgoing chief onto the new one. Nulls, all at
machine precision: the chief path up to the swung mirror moves **4.45e-16 m**,
the chief still lands on the pivot to **4.45e-16 m**, and the beam turns
**20.0000°** for a 10.0000° tilt.

**Why the law cannot stop it, isolated as a measurement.** At −10° the walk
ratio is *unchanged* (1.30 → 1.37, still hopeless) and the fitted **offset**
goes from +4.5 mm to **+195.3 mm**. The tilt does not improve the proportional
part at all — it adds a term the proportional part cannot see. The offset runs
**19.35 mm per degree**, dead linear, against the `2αd` the geometry predicts
(19.66 mm/deg at the 0.5632 m field-mirror → collimator spacing): **1.6 %**.

Two things fall out of the raw sweep that matter beyond this design:

* **The wavefront is not the price.** A mirror at the field conjugate carries
  1.8 mm of beam per field against a 113 mm union footprint, so swinging it
  barely touches the wavefront (10407 → 10353 nm, *better* by 0.5 %). What it
  moves is the **pupil** — the one thing the fourth mirror was added to
  control.
* **The bias plane is the cheaper axis, which is not obvious in advance.** A
  +10° tilt about **y** buys +27.30 mm of floor for 930.6 µm of blur; −10°
  about **x** buys +57.44 mm for 1195.6 µm — **2.1× the clearance for 1.3× the
  blur.**

## C.3 The delivered design — the price table

**`clearing/afocal4_clear_343mm.in`** — the committed 343 mm design with a −10°
extraction tilt on the field mirror and the conics, the field-mirror standoff
and the front end re-solved around it. Zero rays lost over the field box.

| | committed 343 | **cleared, −10°** | ratio |
|---|---|---|---|
| WFE rung 2, max (nm) | 10406.98 | **8992.68** | **0.864** |
| pupil blur rms (µm) | 157.02 | 553.34 | 3.524 |
| breathing, chief-normal (%) | 0.1240 | 0.8160 | 6.579 |
| wander at the refit plane (µm) | 161.23 | 559.87 | 3.472 |
| surface vs imaged sag (mm) | 0.0174 | 0.0102 | 0.588 |
| M at the box centre | 30.0066 | 30.0148 | 1.000 |
| anchoring residual (µm) | 0.0946 | 0.0791 | 0.835 |
| exit beam diameter (mm) | 33.633 | 33.571 | 0.998 |
| collimation (µrad) | 1477.1 | 1265.5 | 0.857 |
| chief AOI on the field mirror (deg) | 2.11 | **10.15** | 4.806 |
| **max chief AOI, any mirror (deg)** | **12.84** | **10.86** | **0.846** |
| **union body-in-beam floor (mm)** | **−79.89** | **+37.82** | — |
| rays lost over the field box | 0 | 0 | — |

**The customer interface is held** — 30.015× at the box centre, a 33.57 mm exit
beam against 33.63, collimation 14 % *better*. The interface plane is carried
through the swing by the same rigid motion as the rest of the train, so its
pose relative to the exit chief is unchanged.

**The AOI reads better, not worse**, which is the opposite of the obvious
expectation: the field mirror goes 2.1° → 10.1°, but the design's *worst*
worked mirror improves 12.84° → 10.86°, because the re-solve moved the front
end. Everything stays under the design drivers' standing 15° rule.

**What it costs, in one line: the fourth mirror's pupil control.** Blur and
wander 3.5×, breathing 6.6×. The wavefront and the interface do not pay.

> The delivered numbers above came out of a 427-evaluation re-solve that
> stopped on its budget (exitflag 0) at the study's default 3e-3 **forward**
> difference — the setting S4c measured as reading this merit's gradient 17 %
> low. They are re-run to convergence with central differences in § C.4c.

## C.4 The clearance as a WALL — `P.pack.union_enforce`

### C.4a Why the delivered point is a point and not a frontier

The gate could measure the interference but nothing was *holding* it. Measured
on the clearing stage's own re-solves: at −8° and −9° the raw tilt gives
+23.34 and +42.25 mm of floor, and the re-solve walks them down to **+2.32 and
+0.69 mm** — the solver, free to move the standoff and the front end, walks the
design back toward the beam it was swung away from, because `afocal4_score`
cannot see clearance and spending it is free. So the delivered −10° is a point
that *happens* to hold margin, not a point chosen on a frontier.

> **Read § C.4c before taking the paragraph above at face value.** Those
> −8°/−9° numbers come from the clearing stage's 427-evaluation
> forward-difference re-solves. Run the *same* solves to convergence and the
> margin is **not** spent: +28.05 and +38.67 mm with the wall off. The
> margin-spending was a stalled solve, not a merit blind to clearance — so the
> wall's justification is that nothing holds the clearance, not that something
> was observed taking it.

The fix is the S4b pattern exactly, and it is a **wall, never a merit term**
(the log-domain merit doctrine is untouched): `afocal4_union`'s floor on the
**declared body model** enters `afocal4_build` beside `m3_behind_min`.

* `P.pack.union_enforce` (default **false**), `P.pack.union_min` (default 0 m),
  `P.pack.union_body_k` / `union_body_pad` (1.15, 15 mm).
* **Default OFF is load-bearing.** With the wall on, `afocal4_build` cannot
  re-emit the committed 343 mm deck — it reads −79.89 mm — so every S4 / S4b /
  S4c / clearing artifact would stop reproducing. `P.pack.enforce = false`
  keeps the unbuildable S4 reference reproducible for the same reason.
* **`CLEAR_BUILD` defers it past the tilt.** `afocal4_build` emits the
  *untilted* train and `clear_build` swings it afterwards, so a wall applied
  inside the build would judge the design the tilt exists to get away from and
  reject every iterate: a cage, not a wall. Gated in `tAfocal4Wall`, on the
  same `P`, both halves.
* **Cost, stated rather than hidden:** the wall is evaluated INSIDE the build,
  so every iterate the solver sees is compliant. It adds **+51 %** to an
  evaluation (8.2 s → 12.4 s cold; ~3.4 s → ~5 s in a warm solver loop).
  Nearly all of it is the nine-field re-trace inside `afocal4_union` — a trace
  `afocal4_score` has already paid for once. The probe count is almost free
  (314 probes 4.18 s, 65 probes 3.52 s), so it is not tuned down.
* **The wall is judged at SOLVE sampling and quoted at REPORTING sampling.**
  More rays make a bigger union hull, so the wall's number is the *optimistic*
  one — measured at about **+1.9 mm** on this design. The seeder's 10 mm margin
  is what covers it, and `tAfocal4Wall` pins the bias below that margin rather
  than pinning the millimetre.

### C.4b The compliant seeder — and the law is not its predictor

*A wall needs a compliant seed or it is a cage* (S4b rule 10). `wall_seed` is
`afocal4_pack_seed`'s clearing-stage sibling, and building it produced a
finding worth keeping:

**The field-walk law is 5–6× optimistic as a predictor of what a STANDOFF
change buys.** The obvious cheap ranking is `f̂ = f(tilt alone) + 2|α|(d − d₀)`,
pure closure arithmetic. Measured at −6°, moving the standoff from the parent's
−38.6 mm to +250 mm takes `d` from 0.563 to 0.680 m; the law predicts the floor
going −13.00 → +11.45 mm, and it measures **−8.25**. Over the parent's whole
admitted range the realised sensitivity is **33 mm per metre of `d`** against
the law's **209**. The law is not wrong — the tilt really does supply `2αd` —
but a standoff change moves the **field-proportional** part at the same time,
and the two very nearly cancel. *That is leverage 2 showing up inside leverage
3*: the station was retired as nearly powerless, and it is nearly powerless
here too.

So the seeder **measures** instead of predicting: gate the parent's own
standoff, gate the extreme of the admitted closure range, and if the extreme
clears, **bisect back toward the parent** for the smallest departure that still
clears — four to six gate evaluations. Preferences, in order: the tilt alone
(nothing moved, which is what keeps a frontier point comparable with the
delivered row); the smallest standoff change on the parent's own front end; a
different M2 radius; and last, the delivered −10° design's own DOFs, flagged in
the record as a different basin.

**One trap paid for here, and it is the recurring one.** `P.parent` carries
*Mike's* raw secondary (R_M2 468.8 mm, t_M1M2 1.0492 m) while the committed
343 mm deck has a re-solved front end (448.4 mm, 1.0420 m). Filtering seed
candidates through `P.parent` admitted 21 standoffs of 57 and **not one of them
was the parent design's own**; carrying `D.R2` and `D.t1` into the closure — as
`afocal4_build` itself does — admits 54, spanning `d` = 0.255…0.821 m against
the parent's 0.563.

### C.4c What the converged solves actually said — and the premise they retired

The brief asked for two things here: reproduce the clearing stage's
margin-spending with the wall off, abolish it with the wall on, and read a
tilt-vs-price frontier off the result. **Neither half survived measurement,
and both failures are more useful than the assertions would have been.**

Method: one MATLAB process per point, `wall_point`, seeded compliantly by
`wall_seed`; conics + field-mirror standoff + front end; **central**
differences at 1e-4 with `FunctionTolerance` 1e-8 and `StepTolerance` 1e-9;
restart rounds until `exitflag 1` or a round buys less than 1e-6 of the merit;
403 evaluations per round, 3 rounds — **1209 evaluations per point** against
the clearing stage's 427. Every point reports its rounds, its exitflags and
its gains. Gate and score quoted at REPORTING sampling.

#### The margin is not spent — the clearing stage's solves had stalled

| tilt, 0 mm floor | raw tilt | clearing stage (427 ev, forward) | **wall OFF, converged** | wall ON, converged |
|---|---|---|---|---|
| −8° | +23.34 | **+2.32** | **+28.05** | +28.05 *(identical)* |
| −9° | +42.25 | **+0.69** | **+38.67** | +43.18 |
| −10° | +57.44 | +37.82 | **+45.07** | +45.07 *(identical)* |

Same tilt, same DOFs, same seed, wall off; the only difference is 427
budget-capped forward-difference evaluations against 1209 central-difference
ones. **The converged solve keeps tens of millimetres of margin.** The
"re-solve spends the clearance" behaviour was a stalled solve on the gradient
S4c had already measured as 17 % low — not a merit blind to clearance.

At −8° and −10° the wall-on and wall-off runs are identical to the last digit
of round-1 merit (46.181908 and 50.097881): **the wall never rejected an
iterate.** It binds only at a +15 mm floor, and at −9° at 0 mm — and in both
cases it changed the path and landed a *better* design (merit 34.34 against
36.89 at −8°/+15 mm; 31.47 against 32.89 at −9°). That is not what a cage
does.

**The wall is still right to have and still non-vacuous** — nothing else holds
the clearance, it refuses the committed deck at −79.89 mm, and `union_min` is
a real threshold (§ C.6). But on this design it is **insurance**, and
**convergence** is what changed the answer.

*(Determinism, checked: the −10° wall-off point ran twice in independent
processes and reproduced round merits 50.097881 / 47.094653 / 35.998867 both
times. The divergences above are genuine path differences, not noise.)*

#### The free-standoff sweep is not a tilt-vs-price frontier

Sorting the walled points by the field-mirror standoff their solve reached
rather than by tilt:

| point | converged s_FM (mm) | K_FM | WFE (nm) | blur (µm) |
|---|---|---|---|---|
| −6° | −229.2 | −3.00 | 12076.2 | 535.8 |
| −8° | +229.7 | −8.25 | 7813.1 | 347.5 |
| −10° (wall off) | +275.9 | −15.78 | 6744.1 | 352.5 |
| −9° | +438.8 | −18.53 | 5288.8 | 288.1 |
| −7° | +535.6 | −25.34 | 3212.5 | 227.8 |

**Wavefront and blur both fall monotonically with the standoff**, across five
points at five different tilts, walking from the parent's −38.6 mm toward the
+600 mm bound. Every one of those solves was still descending hard at round 3
(gains 24–43 %, all exitflag 0) except −8° (1.24e-4, a demonstrated plateau).
So the differences between tilts are mostly **how far each solve walked the
standoff**, not what the tilt costs — the −7° point is not a better tilt, it
is the solve that got furthest.

Two consequences, and the second is a design finding rather than a
methodological one:

1. **A tilt-vs-price curve has to hold the standoff fixed**, which is what
   `wall_point`'s `'seed_standoff'` + `'dofs'` control does (§ C.4d).
2. **The committed 343 mm design is nowhere near its own optimum in the
   standoff DOF either.** Every converged solve walks it hundreds of
   millimetres positive and the wavefront falls the whole way. That is a
   second unclaimed quantity beside the pupil one in § C.7, and a larger one:
   it is worth 3–4× of wavefront error, where the tilt is worth a few percent.

#### The delivered −10° deck, polished — every quoted number moves

The brief asks for this "regardless", and it is the one that touches the deck:

| | as delivered (427 ev, forward) | polished (1209 ev, central) | change |
|---|---|---|---|
| WFE rung 2 max (nm) | 8992.68 | **6744.10** | **−25.0 %** |
| pupil blur rms (µm) | 553.34 | **352.50** | **−36.3 %** |
| wander (µm) | 559.87 | **357.00** | **−36.2 %** |
| breathing (%) | 0.8160 | 0.9816 | **+20.3 %** |
| union floor, declared (mm) | +37.82 | **+45.07** | **+19.2 %** |

**Every one moves far past the 1 % flag threshold, and mostly in the cleared
design's favour** — a quarter less wavefront error and a third less pupil
price than the delivered row states, with 7 mm more clearance. The delivered
numbers were budget artifacts. Anything quoting 8993 / 553 / 0.82 % / +37.8
should be re-cut or should say it is quoting the budget-capped solve.


### C.4d The frontier, with the tilt actually isolated — and the delivered point is past the knee

The fix for § C.4c's confound: pin the field-mirror standoff at **+276 mm**
for every point (a real, reachable station — the one the −10° polish
converged to), drop it from the DOF set, and solve `{conic, front}` only. The
tilt is then the only thing that differs between points, which is what a
tilt-vs-price curve has to mean. Wall ON at a 0 mm floor, central
differences, **1628 evaluations over 4 restart rounds** each; round-4 gains
5.4e-2 / 8.3e-4 / 2.1e-2 / 7.7e-3, against the 24–43 % the free-standoff runs
were still moving at round 3.

| tilt | floor, declared (mm) | bare (mm) | WFE (nm) | blur (µm) | breathing (%) | wander (µm) | max AOI | M | merit |
|---|---|---|---|---|---|---|---|---|---|
| **−8°** | +15.18 | +39.38 | **6513.9** | **279.9** | **0.7210** | **284.2** | 10.68 | 30.0150 | **32.65** |
| **−9°** | **+45.44** | +63.33 | 7794.7 | 352.0 | 0.9190 | 356.4 | 11.01 | 30.0150 | 36.83 |
| −10° | +48.54 | +64.85 | 7682.3 | 456.8 | 1.1912 | 460.9 | 11.29 | 29.9846 | 40.52 |
| −11° | +47.17 | +63.52 | 7464.9 | 482.9 | 1.2044 | 487.0 | 11.34 | 29.9848 | 41.30 |

Zero rays lost at every point; M inside 0.05 % everywhere; every max chief AOI
inside the 15° standing rule.

**THE CLEARANCE SATURATES BY −9°.** The floor is +45.44 / +48.54 / +47.17 mm at
−9 / −10 / −11 — flat to ±3 mm — while the pupil price keeps climbing: blur
352.0 → 456.8 → 482.9 µm, breathing 0.919 → 1.191 → 1.204 %. So **−10° and −11°
are DOMINATED**: another point clears at least as well for materially less
blur. The clearing stage read the saturation at −10° from the *raw* sweep;
converged and with the standoff held, the knee is at **−9°**, and the
delivered design sits **past** it.

**The operating point is −9°**, and it dominates the delivered −10° row on
four of five columns:

| | delivered −10° | **−9°, converged, walled** | change |
|---|---|---|---|
| WFE rung 2 max (nm) | 8992.7 | **7794.7** | **−13.3 %** |
| pupil blur (µm) | 553.3 | **352.0** | **−36.4 %** |
| wander (µm) | 559.9 | **356.4** | **−36.3 %** |
| union floor, declared (mm) | +37.82 | **+45.44** | **+7.62 mm** |
| breathing (%) | 0.8160 | 0.9190 | +12.6 % |

Less wavefront error, a third less pupil blur and wander, **and 7.6 mm more
clearance**, for 13 % more magnification breathing.

**And if the pupil budget is the binding one, −8° is the answer**: blur
**279.9 µm — 49.4 % below the delivered row** — with breathing also *better*
(0.7210 against 0.8160) and the wavefront 27.6 % better, while still holding
**+15.18 mm**, i.e. exactly the declared allowance's own 15 mm pad. That is
the direct answer to the question this slice was set: *does a walled −8° hold
real margin at materially less pupil damage than −10°?* **Yes — the declared
pad, at half the blur.**

What the tilt costs, isolated, is therefore small and one-sided: over −8° to
−11° the wavefront moves 6514 → 7465 nm (a 15 % band with no monotone trend —
the clearing stage's "the wavefront is not the price" survives), while blur,
wander, breathing and AOI all grow monotonically with |tilt|. **Buy exactly as
much tilt as the clearance needs and not one degree more.**

*Caveat, stated: this curve holds the standoff at one station. § C.4c shows
the standoff is worth 3–4× of wavefront error on its own, far more than the
tilt — so this is the tilt's price AT a good station, not the design's
optimum. Finding the standoff's own operating point is a separate question
and a larger one.*

## C.5 The packaging consequences — and it packages itself

The swing was not a packaging move, but it re-poses the whole back end, so the
envelope was re-measured:

| | committed | committed + Path A (4 flats) | **cleared, no flats** |
|---|---|---|---|
| M1–M2 spacing, the yardstick (m) | 1.0420 | 1.0420 | 1.0416 |
| deepest optic behind M1 (m) | 1.8866 | 0.8932 | **1.2874** |
| … as a multiple of the yardstick | **1.81×** | **0.86×** | **1.24×** |
| overhang, deepest − yardstick (m) | +0.845 | −0.149 | **+0.246** |
| … as a multiple of the yardstick | 0.81× | −0.14× | **0.24×** |
| radius of the optics **behind** M1 (m) | 0.186 | 0.435 | **0.150** |
| back focal path (m) | 2.808 | 2.808 | **2.192** |
| extra flats | 0 | **4** | **0** |
| union body-in-beam floor (mm) | −79.9 | −79.9 | **+37.8** |

> **Two ratios, named apart on purpose.** The packaging stage's headline
> "1.81×" is the **deepest optic** over the M1–M2 spacing; the **overhang** over
> the same spacing is 0.81× on the same deck. Quoting one under the other's
> name makes an improvement look three times bigger than it is. (The
> whole-train envelope is the primary's own 0.500 m radius on every deck and
> cannot tell them apart, which is why the "radius behind M1" row exists.)

**The overhang four ~300 mm 45° flats were spent to remove, the swing takes
from 0.81× to 0.24× with none.** It does not reach Path A's *negative*
overhang. What it does that Path A does not: the back-end **girth shrinks**
(0.186 → 0.150 m) where the four folds nearly *tripled* it to 0.435 m; the back
focal path shortens by **0.62 m** (Path A leaves it — a fold is an isometry);
and it opens **no** polarization budget, where four 45° reflections in series
is one the packaging study explicitly did not open.

**Path A does not close on the cleared deck, and does not need to.** The route
was re-searched over all four of its stated quantities on the cleared deck's
own geometry: **96 routes tried, 15 satisfied both the route algebra and the
plane-intersection bound, and every one of those lost rays** (best: −72.74 mm,
592 rays). The reason is arithmetic — the four-fold trombone needs
`x_step + return + m3_gap + the next spacing` of leg to work in, and the swing
has already shortened the feed leg and moved the field mirror 600 mm nearer.
**The route runs out of leg because the depth it was invented to remove is
mostly gone.**

**And the rest of `afocal4_pack` passes on the cleared deck**, so this is not a
design that trades one buildability clause for another: last powered mirror
**+774 mm** behind M1 (minimum 500); fold daylight on the exit leg **27.1 mm**
(margin 15); instrument 233 mm off axis with the **largest that fits Ø421 mm**
against the stated Ø300; union **+37.8 mm**.

## C.6 The standing gate — `afocal4_union`, and its non-vacuity

`afocal4_union.m` sits beside `afocal4_pack` and is now part 4 of it.

* the **union** footprint over the field box, never the deck's own field;
* a **convex hull**, never a centred disk — a disk of the union's max radius
  fills exactly where this train's feed beam passes and invents a 107 mm
  interference belonging to the model, not the design;
* a **declared** allowance (1.15 × footprint + 15 mm) printed with every
  answer, and every table also carries **bare lit glass** (1.00 ×, 0 mm), so an
  interference that survives that one is the design's;
* both distance measures **sampling-free** — an exact plane crossing for a
  pierce, an exact ternary search for a clearance (distance to a convex set is
  convex along a straight segment). That is the only independent check that a
  fold really is an isometry: a station-sampled model re-samples the same
  geometry at a different phase after folding and reads 10.6 vs 5.8 mm on a
  pair an isometry cannot have moved;
* **through-holes are a requirement, not a collision** — a two-mirror front end
  sends its beam back through the primary on every deck in this family, so
  elements listed in `'hole'` report a hole *radius* instead of a pierce;
* **leg-versus-leg is deliberately not in it.** Light passes through light and
  on a wide-field system different fields' beams genuinely cross, so a leg-leg
  zero is not an interference. `pack_clear` reports it; the gate does not,
  because the gate's verdict has to mean one thing.

`'union',false` reproduces `afocal4_pack`'s previous verdict exactly and the
three sub-flags `tAfocal4` asserts are untouched — the promotion is additive.

**Non-vacuity, asserted in code and in two senses.** A gate nobody can fail is
not a gate, and a wall that refuses nothing is decoration:

| assertion | where |
|---|---|
| the GATE fails the committed 343 mm deck (−79.89 mm) and passes the cleared one (+37.82 mm), at the same declared allowance | `tAfocal4Clear`, `afocal4_clearing` §6b |
| bare lit glass is *also* pierced on the committed deck — the interference is the design's, not the body model's | `tAfocal4Clear` |
| one field would have passed it — the union is what makes the difference | `tAfocal4Clear` |
| the WALL refuses the committed design *through `afocal4_build`*, with the identifier the solver's catch clause turns into a residual | `tAfocal4Wall` |
| the WALL admits the swung design — `clear_build` applies it after the tilt, so it is not a cage | `tAfocal4Wall` |
| `union_min` is a **threshold**, not a boolean: raising it above the cleared deck's own floor refuses that deck too | `tAfocal4Wall` |
| the wall is OFF by default and `afocal4_build` still rebuilds the committed deck byte for byte | `tAfocal4Wall` |
| a failure to seed is reported as a **seeding** failure, with the best floor it reached — never as "this tilt has no design" | `tAfocal4Wall` |

## C.7 Addendum — the unclaimed pupil, measured

A *positive* extraction tilt makes the clearance worse and the pupil better.
The clearing stage found one point of that (blur 157.0 → 102.6 µm at +4°, no
re-solve at all); here is the whole signed curve on the **committed** deck,
1° steps, nothing re-solved, every number from the same `clear_price`
machinery the clearing stage used:

| tilt (deg) | floor, body (mm) | offset (mm) | WFE (nm) | blur (µm) | breathing (%) | wander (µm) |
|---|---|---|---|---|---|---|
| −8 | +23.34 | +156.9 | 10358.2 | 843.5 | 0.9890 | 852.5 |
| −6 | −14.95 | +119.0 | 10366.1 | 558.0 | 0.7137 | 565.1 |
| −4 | −53.31 | +81.2 | 10376.9 | 346.4 | 0.4409 | 352.0 |
| −2 | −86.63 | +43.2 | 10390.5 | 214.5 | 0.1701 | 219.1 |
| −1 | −86.59 | +23.9 | 10398.4 | 178.2 | **0.0381** | 182.6 |
| **0 (committed)** | **−79.89** | +4.5 | **10407.0** | **157.0** | **0.1240** | **161.2** |
| +2 | −86.77 | −35.3 | 10426.2 | 131.3 | 0.3929 | 135.3 |
| +3 | −84.27 | −55.7 | 10436.9 | 116.7 | 0.5271 | 120.7 |
| +4 | −77.31 | −76.4 | 10448.2 | 102.6 | 0.6613 | 106.5 |
| **+5** | −86.69 | −97.7 | 10460.3 | **101.3** | 0.7954 | **104.7** |
| +6 | −86.52 | −119.5 | 10473.1 | 129.4 | 0.9296 | 131.9 |
| +8 | −47.98 | −164.9 | 10500.9 | 271.4 | 1.1987 | 273.2 |

`wall/afocal4_wall_unclaimed.png`. Three things this says that the two-point
version did not:

1. **The blur optimum is at +5°, not +4°, and it is 101.3 µm** — 35.5 % below
   the committed design's, for a tilt and no re-solve. The clearing stage's 2°
   grid straddled it.
2. **It is NOT free, and calling it "unclaimed" needs the qualifier.** What
   moves the other way is **magnification breathing**, which is the one pupil
   target the committed design actually *meets*: 0.124 % against a 0.4 % target
   at 0°, and 0.66 / 0.80 % at +4 / +5° — i.e. the blur is bought by breaking
   the breathing spec. The two are one knob, and the committed point is on the
   breathing side of it. Wander tracks blur (they are the same measurement at
   different planes); the wavefront is flat to 0.5 % across the whole sweep.
3. **The genuinely free move is a SMALL NEGATIVE tilt.** At −1° the breathing
   reaches **0.0381 %** — three times better than the committed design's, and
   ten times inside target — while blur worsens only 178.2 vs 157.0 µm and the
   clearance is no worse. That is the same "the merit's floor cannot see it"
   observation S4c made at 343 mm about a third seed holding 0.0117 % breathing
   for nothing measurable.

### C.7a Why the S4 solve did not find it — and it is NOT that the merit is blind

The obvious explanation, and the one the clearing stage offered, is that a
wavefront term 130× off its target owns the log-domain sum of squares. **Read
off the residual vector, that is not what happened.** `afocal4_score` divides
the per-field wavefront residuals by `sqrt(K)`, so the whole wavefront block
contributes the MEAN squared log-miss — one term's worth, not nine:

| tilt | merit | WFE block | pupil block | blur | breathing | wander |
|---|---|---|---|---|---|---|
| −1° | 31.157 | 23.535 (75.5 %) | 7.622 | 4.105 | **0** (floored) | 3.517 |
| **0° (committed)** | **30.222** | 23.550 (77.9 %) | 6.672 | 3.608 | **0** (floored) | 3.065 |
| +5° | **29.403** | 23.621 (80.3 %) | 5.782 | 2.135 | 1.906 | 1.741 |

**The merit PREFERS +5° by 2.7 %.** The pupil terms hold ~22 % of it and cast a
real vote; the wavefront block barely moves across the whole sweep (23.53 →
23.62, 0.4 %). So the committed design is not sitting at its pupil optimum
because the merit could not see the move — **it is sitting there because an
extraction tilt was never in the DOF set that produced it.** The S4b/S4c long
solves that delivered this deck ran `{conic, standoff, front}`; rigid bodies
were not among them.

**And the rigid-body tilt rung 4 does carry is a DIFFERENT OPERATION — measured,
with the opposite sign.** `P.bounds.tilt` allows ±0.05 rad = ±2.86° of x-tilt
on the field mirror, applied by `rigid_body_` about the element's own **vertex**
with the train *not* re-posed; `clear_tilt` pivots on the point the **chief**
strikes and carries the train with it. On this deck, at the same angles:

| operation on the field mirror | WFE (nm) | blur (µm) | breathing (%) | floor (mm) |
|---|---|---|---|---|
| committed, nothing moved | 10407.0 | 157.0 | 0.1240 | −79.89 |
| rung-4 rigid body, +1.00° (vertex) | 10131.4 | 268.7 | 0.1296 | −84.34 |
| rung-4 rigid body, +2.00° (vertex) | 9931.3 | 455.1 | 0.1345 | −87.00 |
| rung-4 rigid body, +2.86° (vertex, the bound) | **9818.8** | **613.0** | 0.1381 | −86.19 |
| extraction tilt, +1.00° (chief, train re-posed) | 10416.3 | **143.6** | 0.2586 | −84.83 |
| extraction tilt, +3.00° (chief, train re-posed) | 10436.9 | **116.7** | 0.5271 | −84.27 |

They move the two quantities in **opposite directions**: the rigid-body tilt is
a *wavefront* knob that spends pupil (5.6 % of wavefront for 3.9× the blur at
the bound); the extraction tilt is a *pupil* knob that spends a little
wavefront. So the extraction tilt is a genuinely new degree of freedom and not
one the S4 parameter set already contained under another name — and there is a
second unclaimed quantity sitting beside it: **a rung-4 rigid-body pass on the
committed 343 mm deck is worth ~5.6 % of wavefront**, on a design whose rung-4
DOFs were never solved. Neither is chased here; both are recorded.

> **Correction to the clearing record.** `clearing/README.md` §12.3 and the
> `BRIEF_afocal4_clear` delivery log attribute the unclaimed pupil to a merit
> owned by the wavefront term. The residual vector says otherwise — the
> wavefront block is 78 % of the merit, not ~97 %, and the merit would have
> taken the move. The DOF set, not the merit, is the reason. The *finding* (the
> committed design is not at its own pupil optimum) is unaffected.

### C.7b The pupil-weighted polish — the incumbent beats both re-weighted solves, on their own merits

The addendum's second question: at a fixed tilt and wall, does a
pupil-weighted merit recover any of § C.7's slack without giving back
wavefront or margin?  Two solves at the operating tilt (−8°), standoff pinned
at +276 mm, DOFs `{conic, front}`, wall on at 0 mm — identical in every
respect to `ctl_t-80` except that `P.weights.{blur, breathe, wander}` are
multiplied by 4 and by 16.  Every design scored under all three merits:

| design | m @ ×1 | m @ ×4 | m @ ×16 | WFE (nm) | blur (µm) | breathing (%) | wander (µm) | M | floor (mm) |
|---|---|---|---|---|---|---|---|---|---|
| **`ctl_t-80`** — solved at the study's own weights | **32.7** | **229.9** | **3386.5** | 6513.9 | **279.9** | 0.7210 | **284.2** | **30.0150** | +15.18 |
| `pw4_t-80` — solved AT ×4 | 46.2 | 301.3 | 4383.1 | 6928.0 | 416.5 | **0.6511** | 421.2 | 30.1545 | +55.00 |
| `pw16_t-80` — solved AT ×16 | 54.5 | 397.5 | 5886.4 | **5318.1** | 570.7 | 1.3141 | 576.7 | 29.7049 | +8.83 |

**Answer: no — and the incumbent beats both re-weighted solves on the
re-weighted solves' OWN objectives.**  229.9 against 301.3 at ×4; 3386.5
against 5886.4 at ×16.  Neither run found anything the plain solve had not
already beaten, and the ×16 run **plateaued** there (round-2 gain 1.17e-5,
814 evaluations) rather than being cut short.

Both also **broke the customer interface** on the way — the one requirement
nothing in a re-weighting protects because it was previously satisfied
incidentally: M = 30.1545 (0.515 % off) and M = 29.7049 (0.98 % off) against a
**0.1 %** target, with collimation 3769 and 3597 µrad against 906.

> **The ×4 row is a PROBE, not a converged point, and is labelled so.**  Its
> round 2 gained **89 %** (merit 2704.8 → 301.3), i.e. it was still descending
> hard when its two rounds ran out.  It is reported because the conclusion
> does not rest on it: it would have to fall a further 24 % merely to reach
> what the incumbent already scores for free, and the ×16 run — which did
> converge — settles the question on its own.  Quoting a still-descending
> number as a result is the exact failure this slice exists to retire
> (§ C.4c), so it is not quoted as one.

**A re-weighted merit is a different optimisation problem, not a re-ranking of
the same one.**  The landscape moves, the basins move with it, and the design
you already have can beat what the re-weighted solve finds — measured on the
new merit.  The § C.7 slack is real, but this stage did not find a way to
claim it, and says so.

*The merit doctrine is not reopened by any of this: log-domain residuals and
walls-not-terms both stand, and `P.weights` has always carried these as knobs.
The measurement is what the slack is worth, not a proposal to re-weight the
study.*

**And this pair is where rule 32 came from** — the first attempt had to be
discarded because at ×16 a sound design scores ~4e4 while the wall residual
was a constant 5600, so the wall inverted from a barrier into an attractor and
the solver walked through it, returning a converged point with M3 1051 mm in
front of the primary.

## C.8 Leverage 4 — a fifth mirror, priced rather than built

Stated in the law's own terms rather than by building a design this arc did not
have time to solve:

```
    the collimator's walk is PINNED     M * iface = 10.290 m/rad
                                        measured    11.118  (+8.1 %)
    to clear with NO field-independent offset the feed's walk would have to
    reach                                           27.001 m/rad
    it is                                           14.481, and its ceiling is
    the intermediate image height itself
    => a fifth mirror must supply at least          111.6 mm
    of FIELD-INDEPENDENT separation
```

**Which is exactly what the −10° tilt already supplies (201.4 mm measured).**
So a fifth mirror's case does *not* rest on being able to clear the beam — the
fourth one can, by being swung. It rests on clearing it **without spending the
pupil control**, i.e. on beating the walled frontier's best point (§ C.4c).

Three architectures could do that, and the law says which lever each pulls:

1. **A powered extraction mirror near the intermediate image** — supplies the
   offset with an element designed for it, instead of by swinging the one
   element that sits at the field conjugate. Same lever, better place to pull.
2. **A relay to a second intermediate image** — the `relay` form of
   `afocal4_close`, eliminated in S3 on pupil grounds *with four mirrors*. With
   five, the collimator stops living inside the M2 → field-mirror cone at all,
   so the ratio law never applies to it.
3. **A fifth mirror that holds the pupil station**, freeing the field mirror's
   power — which the four-mirror closure spends *entirely* on that one
   condition — to put the collimator at the internal chief crossing, where its
   union footprint collapses to the beam radius. This is the only one of the
   three that attacks the **pinned** term rather than adding an offset, and the
   only one that could return the pupil metrics to the committed row's values.

## C.9 Rules earned, each with the alternative it replaced

23. **Ask whether a BODY stands in a BEAM, not whether there is DAYLIGHT.**
    *Alternative rejected:* the beam-to-beam clearance check `afocal4_pack`
    already had. It measured 17.5 mm of daylight for a fold whose own flat,
    sized over the field box, clips the feed beam by −73.6 mm — and it never
    asked about the collimator at all, which is 79.9 mm inside the feed cone on
    every point of the delivered trade curve. **A margin is a number, not a
    body.**

24. **Clearances over the FIELD BOX, and the body is the UNION, hulled.**
    *Alternative rejected:* the deck's own field, and a centred disk. Per field
    the collimator has 10.8 mm of daylight; over nine it is 79.9 mm inside the
    beam — the entire defect lives in that difference. And a centred disk of
    the union's max radius fills exactly where the feed passes, inventing a
    107 mm interference that belongs to the model.

25. **Split a separation into a field-PROPORTIONAL walk and a
    field-INDEPENDENT offset before proposing a remedy.**
    On a coaxial train the offset is identically zero, so *every* remedy whose
    separation is proportional to field is bounded by one ratio the field box
    fixes — `(bias+half)/(bias−half)`. That single statement retires the
    collimator station, the interface standoff and any flat fold before one is
    built, and it names the one class that can work. Fit the walk **with an
    intercept**: forcing it through the origin makes a tilted design report a
    meaningless "walk" that is silently absorbing the offset, and the offset
    *is* the remedy.

26. **A wall is a wall only where the artifact it judges is the one that gets
    scored.** `clear_build` tilts the deck *after* `afocal4_build` emits it, so
    a union wall inside the build judges the untilted train and rejects every
    iterate. Deferred past the tilt it bounds the design; applied in the
    build it is a cage. Both halves gated on one `P`.

27. **A wall that changes a committed answer must default OFF.**
    With the union wall on, `afocal4_build` cannot re-emit the deck the
    S4b/S4c trade shipped (−79.89 mm). Defaulting it on would have made every
    committed artifact in this study unreproducible — the same reason
    `P.pack.enforce = false` keeps the unbuildable S4 reference alive.

28. **A cheap predictor derived from a law still has to be MEASURED against
    the law's own domain.** `f̂ = f + 2|α|(d − d₀)` is the field-walk law
    written out, and it is 5–6× optimistic for a **standoff** change, because
    moving the station moves the field-proportional part too and the two nearly
    cancel. The seeder probes and bisects on measurements instead. *A law that
    is exact for one knob is not a predictor for a different knob.*

29. **Re-pose a closure on the DESIGN's front end, never on `P.parent`.**
    `P.parent` is Mike's raw secondary; a committed deck has a re-solved one.
    Filtering seed candidates through `P.parent` admitted 21 standoffs of 57
    and not one of them was the parent design's own.

30. **When a record explains a finding, check the explanation separately from
    the finding.** The unclaimed pupil is real (35.5 % of the blur, at +5°);
    the reason recorded for it — "a wavefront term 130× off target owns the sum
    of squares" — is not what the residual vector says (78 %, and the merit
    *prefers* the move). The real reason is that an extraction tilt was never
    in the DOF set. A right finding with a wrong mechanism will be generalised
    wrongly the next time.

31. **A wall belongs on ITERATES, never on the REPORT that follows them.**
    `clear_solve` built its final, quotable deck through the same walled
    builder the objective used. Inside the objective a violation becomes a
    large finite residual and the solver backs out of it; in the report path
    there is nobody to back out. Worse, the union wall is evaluated at SOLVE
    sampling inside the loop and would be re-evaluated at REPORTING sampling
    there, where a bigger ray grid makes a bigger union hull and the floor
    reads ~2.5 mm lower — so **a converged design sitting on its wall throws
    out of its own report and takes the whole solve with it.** Measured, once:
    an hour of a −8 deg walled run, lost. The report path now measures and
    does not judge, and each restart round is guarded so a round that throws
    costs a round rather than a night.

32. **A wall is only a wall while it dominates the merit's own SCALE.**
    `clear_solve` rejects a wall-violating iterate with a constant residual
    of 20 per component — merit 5600. At the study's own weights a sound
    design scores ~30, so that is an impassable barrier. Multiply the pupil
    weights by 16 to measure § C.7's slack and a *sound* design scores ~4e4:
    the same constant now looks **attractive**, and the solver walks through
    the wall deliberately. Measured: the ×16 run returned a converged point
    whose closure put M3 **1051 mm in front of the primary**. The
    walls-not-terms doctrine holds in principle, but implementing a wall as a
    fixed large residual quietly makes it *tradeable* once the merit outgrows
    it. The residual now scales with the largest merit weight in play; at the
    study's own weights it is exactly 20, bit-identical to every committed
    solve. **Any change to the merit's scale — a re-weighting, a
    regularizer, a new term — has to be checked against every wall's
    residual.**

33. **Isolate the variable before calling a sweep a curve.**
    The walled sweep was run with the field-mirror standoff in the DOF set,
    as the brief specified. Its points then ordered by the standoff each
    solve *reached* (−229 → +536 mm), not by tilt, with wavefront and blur
    falling monotonically along the way and every solve still descending
    24–43 % per round. Read as a tilt-vs-price curve it would have said −7°
    is a far better tilt; it is simply the solve that got furthest. Pinning
    the standoff and re-running made the tilt the only difference — and gave
    the opposite ordering, with the price rising monotonically in |tilt| and
    the delivered −10° sitting past the knee. *A sweep over one parameter
    with a second parameter free is a measurement of the solver, not of the
    parameter.*

34. **A re-weighted merit is a DIFFERENT PROBLEM, not a re-ranking of the
    same one — so score the design you already have on the new merit before
    believing the new solve.**
    Multiplying the pupil weights by 16 to chase § C.7's slack produced a
    converged, plateaued design scoring **5886.4** on the ×16 merit, where the
    design the *unweighted* solve had already found scores **3386.5** on that
    same merit — 74 % better, for free.  The ×4 run says the same: 301.3
    against the incumbent's 229.9, on the ×4 merit. The re-weighting moved the landscape
    and the solver landed in a worse basin and stayed. It also drifted the
    interface magnification to 0.98 % off 30, ten times its target, because
    nothing in a re-weighting protects a requirement that was previously
    satisfied incidentally. *Re-weighting is a last resort, and its result has
    to be compared against the incumbent ON THE NEW MERIT before it is
    believed.*

---

# DESCENT RESULTS — how many mirrors does the requirement set need?

Answers `macos/BRIEF_afocal4_descent.md` (Dave, 2026-08-31). The numbers-first
account is `descent/README.md`; this is the canonical record.

The stage was set up to start at seven powered mirrors where the whole
requirement set is met with margin, remove one at a time, and measure what
each removal costs. **The top rung was never reached.** What the stage
delivers instead is the reason, measured three independent ways.

## D.1 The answer, in one line

**No mirror count in this family reaches the requirement set**, and the
wavefront half is not close: with the pupil requirement **abandoned entirely**
and every degree of freedom free, seven mirrors floor at **3424 nm against a
71 nm target — 48×**, and three extra mirrors buy **11 %** over four.

## D.2 What was built, and what every route said

| route | N | merit | WFE (nm) | blur (µm) | M err (%) | verdict |
|---|---|---|---|---|---|---|
| committed (S4b/S4c) | 4 | 30.2 | 10407 | **157** | 0.0221 | the delivered design |
| cold seed | 7 | 70.78 | 12422 | 506 | 3.86 | missed, stalled |
| cold seed, radii freed | 7 | 70.40 | 11718 | 534 | 3.68 | missed, stalled |
| cold seed 2 | 7 | 707 | 3.7e9 | 1.7e6 | 95 | **scrambled — gates caught it** |
| cold seed 3 | 7 | 53.57 | — | — | — | missed, worst 177× |
| cold seed | 8 | 146.7 | — | — | — | missed, worst 1768× |
| ascent (warm) | 5 | 37.39 | 10775 | 332 | 0.1379 | missed |
| ascent (warm) | 6 | 44.20 | 9137 | 721 | 0.0651 | missed |
| ascent (warm) | 7 | 42.66 | **7894** | 705 | 0.0632 | missed |

The ascent rungs are the sound ones — M error 0.06–0.14 % against the cold
seeds' 3.9–95 % — and their wavefront does improve with N. It improves by
**24 % over four mirrors where the target needs 99 %**, charged at **4.5× the
pupil blur** (157 → 705 µm), with a body in the beam at every rung (union
floor ≈ −105 mm). Figure: `descent/afocal4_descent_ladder.png`.

## D.3 The decisive measurement — the wavefront floor, pupil requirement abandoned

A full solve is a COMPETITION, so a wavefront that will not move might merely
be losing an argument to the pupil terms. These solves score the wavefront
ALONE, every DOF free. It is the most optimistic wavefront the family can
produce, because it gives up the entire reason the fourth mirror exists.

| N | DOFs | start (nm) | **floor (nm)** | × target |
|---|---|---|---|---|
| 4 | conic, radius, spacing | 10407 | **3841.8** | 54× |
| 4 | + tilt | 10407 | 4497.7 | 63× |
| 5 | + tilt | 10775 | 8077.4 | 114× |
| 6 | + tilt | 9137 | 5689.0 | 80× |
| 7 | + tilt | 7894 | **3424.2** | **48×** |

**Three extra mirrors buy 11 %**, and the trend is not monotonic — N = 5 is the
worst of the set, which is basin scatter rather than a curve going anywhere.

> **Upper bounds, and labelled so.** Several rounds were still gaining 18–25 %
> when their budget ran out, so deeper solves would lower these. It does not
> touch the conclusion: closing 48× needs every rung to fall two orders of
> magnitude, where S4c's 17×-budget long solves moved the same designs by
> 0.02–10.8 %.

## D.4 A cleaner number than S4 had: what the pupil requirement costs

At the **343 mm** operating point, dropping the pupil requirement takes the
wavefront **10407 → 3842 nm — a factor of 2.7**.

S4 ran this same A/B at the **140 mm** operating point, got 8467 against a
frozen 8835 (4 %), and generalized it as *the DOFs do not touch it*. **That
generalization is operating-point specific and should not be carried.** At
343 mm — the standoff the packaging constraint forced this study to adopt —
the DOFs touch the wavefront a great deal, and it is the **pupil requirement,
not the optics**, that costs it the factor of 2.7.

## D.5 The packaging station obeys a parity law

The vertex stations are an alternating sum `z_N = Σ (−1)^k t_k`, so the
closure's own last spacing — the one the magnification condition fixes, and
typically the largest — enters with sign `(−1)^(N−1)`. Measured as a
compliance rate over a common grid of front ends:

| N | parity | closed | compliant | rate |
|---|---|---|---|---|
| 5 | odd | 232 | 205 | **88.4 %** |
| 6 | even | 7 406 | 2 | 0.03 % |
| 7 | odd | 95 849 | 86 024 | **89.7 %** |
| 8 | **even** | 717 679 | **1** | **0.00014 %** |

A factor of ~3000 between adjacent N, from one sign. This is S4b's *one extra
mirror flips the parity of the back end* stated as a law with a rate attached,
and it says the odd rungs are cheap to seed while the even ones are where the
packaging wall bites.

**It is a population statement, and it does NOT transfer to a single
removal** — measured, after being predicted too strongly. On the ascent's N = 7
rung, removing each free mirror both ways:

| mirror removed | retain (parity held) | delete (parity flipped) |
|---|---|---|
| M3 | +1.894 m clears | +0.122 m **fails** |
| M4 | +1.990 m clears | −0.149 m **fails** |
| M5 | +1.772 m clears | **+1.493 m clears** |

Retain clears 3 of 3, delete 1 of 3, and deleting typically costs 1.5–2.1 m of
station — the effect is real and large. But deleting element *k* does not drop
a term, it **merges** `t_{k−1} + t_k` and re-signs every spacing after it, so
which mirror is removed decides how the sum re-assembles. **"N = 6 cannot be
built" is false**: from that base there are at least four routes to a
compliant six-mirror layout.

## D.6 Rules earned

35. **A law measured over a POPULATION does not transfer to an INDIVIDUAL
    case without being re-measured there.**
    The parity law is exact as a rate (88 % / 0.03 % / 89.7 % / 0.00014 %) and
    merely *usual* for one removal — 1 of 3 deletes cleared anyway. It is the
    same shape of error as the field-walk law, which is exact for a tilt and
    5–6× optimistic for a standoff. Predict with the law; decide with a
    measurement.

36. **A lesson confirmed once does not become the explanation for the next
    stall.**
    The wall slice found the committed design missed its pupil optimum because
    the DOF set lacked a tilt. This stage reached for that twice and was wrong
    twice: freeing the radii bought **5.7 %** on a row needing 165×, and the
    wavefront-only control *without* tilts floors **better** (3841.8 vs
    4497.7 nm) than with them. Tilts are a pupil knob, not a wavefront knob.
    Both times the **seed** was the reason.

37. **A cold closure is a specification, not a design.**
    Four cold seven-mirror seeds landed at merit 54…707 with ~12 µm of
    wavefront — *worse than the four-mirror family with twice the freedom* —
    while the same machinery warm-started from the committed design produced
    sound rungs (M error 0.06–0.14 % against 3.9–95 %). A closure holding its
    three conditions at 1e-16 says nothing about quality: one such probe
    traced **M = 40.45 against a paraxial 30.0000**. Build ladders from a
    design that works.

38. **A wall with only one side is not a constraint.**
    The S4b packaging wall bounds the last mirror's station from BELOW only,
    and the power-economy tie-breaker rewards weak mirrors, which are exactly
    the ones that need distance. The first N = 7 seed put the last powered
    mirror **10.96 m** behind the primary and was compliant by every check in
    the study. Bounds are stated in the study's own unit — multiples of the
    M1–M2 spacing, the way the packaging record measures depth.

## D.7 Open, and for Dave

**The requirement set is not reachable by mirror count in this family, so the
useful next move is a judgement about what the work is for, not another
solve.** Two candidates, and the choice is not the study's to make:

* **the spec** — 71 nm was set at ≥ 10× Rodgers' best three-mirror variant
  (PLAN_AFOCAL4 S3 gate review). Nothing in this arc has come within 48× of
  it, with the pupil requirement abandoned and up to seven mirrors;
* **the family** — this is a coaxial all-reflective afocal, and the condition
  that keeps costing is the interface pupil consuming the last two powers.
  § D.4 now prices that: a factor of 2.7 of wavefront at the 343 mm operating
  point.


## D.8 The off-axis probe — what it did and did NOT establish

The descent's answer (§ D.1) is about the **coaxial** family: every design in
that ladder puts all mirrors on one axis, so the aberration field is centred
on it while the 0.5° box is used in an annular zone offset 0.6°. S4 identified
the residual as **field-varying astigmatism** (1108 → 3312 → 6809 nm across
the box) and found rigid bodies bought 0.4 % — but that was measured inside a
**perturbation** bound, `P.bounds.tilt = ±0.05 rad = ±2.86°` and
`P.bounds.dec = ±50 mm`, i.e. alignment tolerances rather than design
freedoms, and the rigid bodies were never in the DOF set that produced the
committed deck. Tilting and decentring *moves the aberration field centre*,
which is the standard tool for exactly that residual.

So the bounds were opened to design scale (±15°, ±300 mm), the DOF scales
moved with them, and the committed deck re-solved over
`{conic, standoff, front, rb}` in two arms:

| arm | WFE (nm) | × target | M error (%) | blur (µm) | rigid bodies USED |
|---|---|---|---|---|---|
| wavefront only | 5157.0 | 73× | **1.0291** | 532.0 | dec [+3.1, +0.4, −0.2] mm · tilt [−0.92, +0.16, −0.07]° |
| full set | 10191.2 | 144× | 0.0204 | 157.7 | dec [+0.3, +0.3, +0.1] mm · tilt [−0.05, +0.03, −0.01]° |

**The solver barely used the freedom it was given.** Against bounds of ±15°
and ±300 mm it moved at most **0.92° and 3.1 mm**, and in the full arm
essentially nothing.

**What that establishes, stated precisely: the coaxial point is a LOCAL
OPTIMUM UNDER RIGID-BODY PERTURBATION.** It does **not** establish that an
off-axis family cannot meet the requirement set. A gradient solver started at
rigid-body zero will not walk ten degrees away, because the path is uphill;
off-axis nodal-aberration designs are a **different basin**, not a
perturbation of the coaxial one. This stage has already been taught that twice
(§ 5a, § 6d) and does not get to forget it here.

Two details that make the probe trustworthy rather than merely negative:

* **the wavefront-only arm landed WORSE than the coaxial floor** (5157 against
  3841.8 nm) — basin scatter, a different and poorer local minimum, which is
  itself evidence that the rigid-body landscape is rough rather than flat;
* **it broke magnification to 1.03 %**, ten times its target, which is the
  cheat flagged *in advance*: with the pupil ladder unscored the `mag` term is
  unscored too, so a large tilt can buy wavefront by ceasing to be a 30×
  telescope. M and collimation are printed on every result for that reason,
  and **a floor reached by breaking M is not a design**.

**The family question therefore remains open**, and answering it needs a
design SEEDED off-axis rather than perturbed there — `macos/BRIEF_afocal4_offaxis.md`.

---

# OFF-AXIS RESULTS — is the FAMILY the wall?

*`BRIEF_afocal4_offaxis.md`, 24 h box from 2026-09-01. Additive: everything
new lives in `challenges/afocal4/offaxis/`. The one change outside it is an
optional `D.decenter` field on `descent_build`, which defaults to 0 and takes
the identical code path when it is 0 — no deck edit is attempted at all — so
every committed descent result is untouched.*

## O.1 What this stage had to fix about the previous answer

§ D.8 measured, correctly, that the coaxial point is a **local optimum under
rigid-body perturbation**: given ±15° and ±300 mm the solver used 0.92° and
3.1 mm and came back. That is a statement about a solver's basin. The family
question needs a design **seeded** off-axis, and the variable that a
rigid-body probe structurally cannot vary is the one that matters:

> **A decenter is not reachable by perturbing the mirrors.** Tilting and
> shifting bodies moves the optics; going off-axis moves the **pupil** — it
> changes *which part of each parent surface the light uses*. The probe was
> free to move every mirror and had no way to ask for that.

## O.2 The off-axis section construction — and why it is exact

An off-axis section is **not a different optical system from its parent**; it
is the same system used away from its axis. Paraxially the powers, the
spacings, and therefore the afocal condition, the magnification and the pupil
conjugate are all **identical** to the coaxial train's. What changes is the
aberration, the obscuration, and the parts you have to build.

Two consequences, and both are load-bearing:

1. **The whole descent machinery applies unchanged** — `descent_close`'s three
   exact closures, the merit, the walls, the scorer. Going off axis is a deck
   edit (`offaxis_decenter`), not a new closure. This is why the isolation
   experiment in § O.7 can hold *everything* fixed but `h`.
2. **A confocal parabola pair is exact off axis.** A parabola takes a
   collimated beam to its focus from any part of its surface, so a Mersenne
   stays afocal and stays 30× at any decenter — **by construction, not by
   convergence**. Measured on the f1/f2 = 30 pair, on axis, at decenters of
   0, 0.6, 0.8 and 1.0 m: **collimation 0.00 µrad at every decenter**, to the
   last bit.

That last point is worth contrasting with the descent's standing warning. A
*cold closure* is a specification and not a design — one descent probe traced
M = 40.45 against a paraxial 30. A *Mersenne* is the exception: its two hard
properties are identities of the geometry, so it is the one seed that needs no
convergence before it can be trusted.

## O.3 Two traps, closed at resolution time

**(a) The measuring pass must REMOVE the apertures, not widen them.** To fit
an off-axis clear aperture you must first trace unclipped. Widening every
`ApVec` to something that comfortably holds a decentered 1 m pupil looks
equivalent to removing them and is not: a clear aperture is read against the
**surface it sits on**, and a Mersenne secondary at M = 30 has |Kr| = 0.083 m
while carrying a 33 mm beam. A 6.2 m aperture asks the engine to intersect
that parabola ~75 radii from its own vertex, and it answers — correctly — with
a surface miss.

| measuring pass | rays traced | outcome |
|---|---|---|
| widen `ApVec` to 4(h+1) | **0 of 1185** | 522 surface miss, 663 obscured |
| `ApType= None` | **1185 of 1185** | every ray geometrically valid *and* unvignetted |

**(b) A clipped beam does not report as clipped — it reports as a different
telescope.** The emitted coaxial deck centres each `ApVec` on the element
*vertex*, so a decentered beam walks off the primary, and the survivors still
trace, still collimate, and still hand back a magnification:

| decenter | rays surviving | reported M |
|---|---|---|
| 0.00 m | 1184 / 1185 | 30.07 |
| 0.60 m | 408 / 1185 | **37.65** |
| 0.80 m | 174 / 1185 | **46.80** |
| 1.00 m | 12 / 1185 | **106.36** |

This is why **every quoted number in this section carries its ray count**, and
why the aperture fit runs before any metric is taken rather than after.

**(c) The ray count itself had to be fixed at the source.** The first version
of this instrumentation took its loss count from `ray_hist`'s `ok` flag — and
that flag is **geometric validity**. An obscured ray keeps a perfectly valid
intersection (the engine sets the flux flag and leaves `RayPos` alone, the same
principle behind the SPOT `LocalCoord` and OPD-reference rulings), so a count
taken from it reports a **fully vignetted beam as lossless**. The guard was
measuring the wrong thing. Loss is now taken from `ray_info`'s
`ok_trace .AND. ok_pass`, in `offaxis_decenter` and in the gate alike, and the
two counts are reported separately: `.nmiss` (geometric, during the
aperture-free measuring pass) and `.nlost` (throughput, on the fitted deck).

**(d) On these decks the fitted apertures are DESCRIPTIVE, not constraining —
and the isolation experiment needed that checked, not assumed.** The
afocal4/descent family emits `ApType= None` on every element, so the coaxial
control carries no clear apertures at all while a decentered deck would carry
fitted ones. That is a second difference between the two columns of § O.7, and
an unexamined one would have made "the decenter is the only variable" false.
Measured on the committed 4-mirror design, wavefront-only over the field box:

| decenter | fitted apertures | apertures removed | difference |
|---|---|---|---|
| h = 0.55 m | 22365.46 nm, 1185 pass | 22365.46 nm, 1185 pass | **0.000e+00** |
| h = 1.00 m | 53658.67 nm, 1185 pass | 53658.67 nm, 1185 pass | **0.000e+00** |

Exactly zero, with every ray passing either way: the fitted apertures are sized
to the measured footprint plus a margin and therefore clip nothing. The
asymmetry is null and the isolation holds. The fit earns its keep for
**realizability** — it is what says how big each off-axis parent has to be —
not for the wavefront.

## O.4 The mirror count is set by packaging PARITY, not by aberrations

`descent_close`'s last spacing **absorbs the free tail spacing exactly**.
Scanned over t2 from 0.3 m to 6.0 m with an N = 4 Mersenne front end, the last
powered mirror sits at `behind_m1 = −1388.7 mm` for **every** t2 (t3 tracks t2
with a constant offset of 0.1803 m). The packaging station is therefore not a
knob at all — it is a **constant of the front end**, and for a Mersenne it is
on the wrong side of the primary.

The parity law `z_N = Σ (−1)^k t_k` says why, and says what to do: one more
reflection flips the sign of the whole back end. Measured across form × f1 × N
(cass/greg × f1 ∈ {0.75, 1.25, 2.5} × N ∈ {4, 5, 6}), on a (t2, t3) grid:

| N | packaging-compliant closures |
|---|---|
| 4 | **none** on the grid |
| 5 | **essentially the whole grid** |
| 6 | **none** on the grid |

**Keep this apart from image quality.** N = 4 off-axis is not ruled out
because it images badly — it is ruled out because its back end lands in front
of its own primary. The coaxial study reached N = 4 comfortably because its
front end is *not* a Mersenne: there t1 is a fraction of f1, where a
Mersenne's is f1 − f2, essentially the whole focal length.

## O.5 `descent_close` is singular on a front end that has already closed the spec

This is the finding that redirected the slice, and it is a statement about
where the closure's **parameterization** is well-posed, not a defect in it.

The closure solves the last two powers from the marginal state after the free
mirrors: `u2 = um − ym·φ`, `b = (yout − ym)/u2`, `φ_N = u2/yout`. Marched by
hand through a Mersenne pair (f1 = 1.25, cass):

```
after free mirror 1:  y  0.016667 m,  u  -0.400000
after free mirror 2:  y  0.016667 m,  u   0.000000     <-- already collimated,
yout target        =  0.016667 m                            already at height
numerator (yout-ym)= -0.000000
```

**The specification is already met at mirror 2**, so the magnification lever's
numerator vanishes. Handed a five-mirror spec the closure does not fail — it
inserts a strong third mirror (φ₃ = −1.2308 /m) that **breaks** the Mersenne
(y → 0.0372, u → 0.0205) and then re-closes it with a 2.74 m lever. The result
is paraxially exact — residuals `[6.9e-18, −5.6e-16, −3.1e-16]`, paraxial
mag `30.000000` — and traces at **M = 26.73 with 25445 µrad of collimation
error**, because the intermediate beam grows to 1.6 m across on mirrors of
1.8 m radius.

**The check that ruled out the obvious wrong explanation.** The natural
suspicion is that the decenter broke it. It did not — the *coaxial* build of
the identical closure is **worse**:

| build | traced M | error | collimation |
|---|---|---|---|
| cass N=5, h = 0 (coaxial control) | 26.73 | **−10.89 %** | 25445 µrad |
| cass N=5, h = 0.55 (off axis) | 31.30 | −4.33 % | 16887 µrad |

So the Mersenne seed is measured **directly** (§ O.6) and the closure-based
comparison is run on front ends the closure is well-posed for (§ O.7). The two
answer different questions and neither substitutes for the other.

## O.6 The bare off-axis Mersenne — the family with the pupil requirement dropped

A two-mirror Mersenne has **no free parameter left**: both powers are consumed
by the magnification and the collimation, so its exit pupil lands where the
geometry puts it and cannot be moved. It therefore *cannot* meet the
interface-pupil requirement — which is exactly why it is worth measuring. It
separates two questions the full set fuses:

* **Q1** — is the off-axis family capable of 71 nm *at all*?
* **Q2** — is it capable of it *while also* placing the exit pupil?

Wavefront-only (rung 2), nine-field box, ray counts complete unless noted:

| form | f1 (m) | h = 0 | h = 0.55 | h = 0.75 | h = 1.00 | h = 1.50 | best × target |
|---|---|---|---|---|---|---|---|
| cass | 1.25 | 60200 | 62629 | 62671 | 62323 | 60765 | 848× |
| cass | 2.50 | 38267 | 37682 | 37415 | 37051 | **36246** | 511× |
| cass | 5.00 | 20994 | 20703 | 20592 | 20453 | **20168** | **284×** |
| greg | 1.25 | 2.06e7 | 1.58e7 | 1.28e7 † | 8.6e6 † | 3.8e6 † | 53972× |
| greg | 2.50 | 4.37e6 | 4.18e6 | 3.93e6 | 3.56e6 | 2.65e6 | 37291× |
| greg | 5.00 | 1.26e6 | 1.27e6 | 1.26e6 | 1.23e6 | — | 17333× |

*(nm, rung-2 max over the field box. † rays lost — 168, 1830 and 7982 of
10665 — so those cells are not comparable and are shown only to record that
the Gregorian branch degenerates rather than to rank it.)*

Three readings, and the third is the one that matters:

1. **The Gregorian Mersenne is not a viable seed** at this magnification.
   Its real internal focus sits between the mirrors and at 30× the beam
   through it is violent; the branch loses rays and never comes within four
   orders of the target.
2. **Speed dominates.** Slowing the primary from f1 = 1.25 to 5.0 m — a 4×
   change — improves the wavefront by 2.87×. That is the front end's own
   aberration, and it is the largest lever in the table by far.
3. **The decenter itself buys almost nothing: at most −3.9 % (f1 = 5.0) and
   −5.3 % (f1 = 2.5), and at f1 = 1.25 it is a small COST (+0.9 %).** On a
   two-mirror system with no correction freedom, moving off axis is very
   nearly wavefront-neutral.

**Q1 is answered no, for the bare pair**: 284× the target at its best (255× at
the coaxial family's 343 mm standoff — see § O.6f, which corrects the
comparison), against
a coaxial four-mirror family that reaches 7.5 µm and a coaxial seven-mirror
wavefront-only floor of 3424.2 nm (§ D.3; 3841.8 nm is the N = 4 floor with
tilt withheld, not the seven-mirror number — corrected here at the point of
use). A Mersenne has no correction freedom, so
this bounds the *seed*, not the family after solving — which is § O.7's job.

### O.6f The standoff is NOT neutral — a correction to the table above

The § O.6 sweep was built at `P.iface` = **140 mm**, while the coaxial
comparators (§ D.3, § D.4, and the h = 0 controls of § O.7, which recover
`iface` from the committed deck) sit at **343 mm**. § D.4 warns in terms that
this distinction matters — the same A/B gives 4 % at 140 mm and a factor of
2.7 at 343 mm — so the two columns were not directly comparable.

The reasoning that said it was safe: a bare Mersenne cannot *move* its pupil,
so its interface is only a **reporting** plane, and a collimated output's
wavefront should not care where that plane sits. **Measured, that reasoning is
wrong**, and how it is wrong is itself a confirmation:

| deck | standoff | rung 2 | rung 3 | union floor |
|---|---|---|---|---|
| off-axis Mersenne f1 = 2.5, h = 1.5 | 140 mm | 36246 | 9396 | −68.7 mm |
| | 343 mm | **29737** (−18 %) | 8725 (−7 %) | −82.6 mm |
| off-axis Mersenne f1 = 5.0, h = 1.5 | 140 mm | 20168 | 2778 | −81.1 mm |
| | 343 mm | **18127** (−10 %) | 2753 (**−0.9 %**) | −111.1 mm |

**Rung 2 moves by 10–18 % with the plane; rung 3 barely moves at all.** That is
exactly what § O.6b predicts and could not have been arranged: the output is
*not* collimated — its power varies with field — so sliding the evaluation
plane along the beam adds and removes **defocus and nothing else** to first
order. Remove the power term and the plane-dependence goes with it. The
premise "the output is collimated, so the plane does not matter" fails for
precisely the reason the whole section is about.

**What changes in the conclusions: nothing, and the corrected figure is
stated.** At the matched 343 mm standoff the best bare off-axis Mersenne is
**18127 nm = 255× target** rather than 284×. Still hundreds of times off, still
98 % power, and § O.6b's rung-3 comparison is the one that was already
standoff-robust (2778 vs 2753 nm, 0.9 %). Read the § O.6 table as a *relative*
sweep over form, speed and decenter — which is what it was built for — and
take absolute cross-family comparisons from the rung-3 column or from matched
standoffs.

**Realizability note, recorded rather than hidden:** almost every row is
*obscured* (union floor −57 to −81 mm), and not by the secondary. The
secondary clears easily — at h = 0.55 the entering beam spans y ∈ [76, 1022]
mm while M2's body sits at y ≤ 50 mm. What stands in the beam is the
**interface plane itself**, 140 mm past M2, whose footprint over the field box
reaches back into the M1→M2 leg. A two-mirror afocal puts its interface where
its own beam is; only two rows in the sweep clear (cass f1 = 1.25 at h = 1.50,
+8.3 mm; greg f1 = 2.50 at h = 1.50, +28.8 mm).

### O.6a Against the arc's own prior Mersenne work — a refinement, not a contradiction

The double Mersenne was examined twice before, **coaxially**, and both verdicts
sharpen what is measured here rather than conflicting with it.

**§ 4 measured the form's wavefront floor with the conics free.** Holding the
confocal spacings and relaxing all four conics took it from 59 370 nm to
35 050 nm — *a factor of 1.7 where a factor of 500 was needed*. The number
worth carrying forward is the other one in that section: solved **on axis**,
where a coaxial design carries **no field aberration at all**, the
conic-relaxed double Mersenne still reaches only **3955 nm**.

That is the honest bound on my § O.6 table. The bare pair measured here is
*parabolas only* and *over the field box*, so its 20 168 nm best is not
comparable with 3955 nm — but § 4 has already established that freeing the
conics of a Mersenne is worth a factor of ~1.7, not a factor of 500. **Nothing
in the off-axis construction changes that**, because relaxing a conic is a
pupil-independent freedom: it is the same surface figure whichever part of it
the beam uses.

**§ S4b.5 closed it a second time, structurally**, and this is where § O.4 is a
genuine refinement. That section recorded that the double Mersenne puts M4 at
z = −0.942 m, *942 mm in front of the primary*, and that **"no gap, stage split
or conic moves them behind it, because the form's entire compression happens
before the beam ever gets back to M1."**

That claim is now **measured and explained**, and one word of it is too strong:

* **Explained** — the mechanism is the closure's last spacing absorbing the
  free tail spacing exactly, so `behind_m1` is a *constant of the front end*
  (§ O.4: −1388.7 mm for every t2 from 0.3 m to 6.0 m). "No gap moves them"
  is not a heuristic about compression; it is an algebraic identity.
* **Refined** — *one* thing does move them, and it is neither a gap nor a
  conic: the **parity of the element count**. `z_N = Σ(−1)^k t_k`, so one more
  reflection flips the sign of the entire back end. N = 5 complies over
  essentially the whole (t2, t3) grid where N = 4 and N = 6 comply nowhere.

So the S4b.5 verdict stands exactly as recorded for the form it was about — a
*four*-mirror double Mersenne — and it should not be read as "a Mersenne front
end can never be packaged." It can, at odd element counts. What closes the
Mersenne route here is § 4's wavefront bound, not the packaging one.

### O.6b What the 20 µm actually IS — and it is not what it looks like

The § O.6 table reports the committed metric, rung 2 (piston + per-field
tip/tilt removed). **Rung 3 also removes POWER**, so the difference between
them is the residual *defocus* — the amount by which the real traced beam
fails to be collimated, field by field. Scored on the same decks, same box:

| deck | rung 2, box | rung 3, box | variance in POWER | traced collimation |
|---|---|---|---|---|
| committed 4-mirror | 10407 | 6555 | 60 % | 1477 µrad |
| descent rung N = 5 | 10775 | 5983 | 69 % | 1635 µrad |
| descent rung N = 6 | 9137 | 5670 | 61 % | 1281 µrad |
| descent rung N = 7 | 7894 | 4710 | 64 % | 1548 µrad |
| **off-axis Mersenne** f1 = 5, h = 1.5 | **20168** | **2778** | **98 %** | 4352 µrad |

*(nm; "variance in POWER" is 1 − (rung3/rung2)², on the max-field basis — the
two maxima need not fall at the same field, so read it as indicative.)*

**Read the last row again.** The off-axis Mersenne looks like the worst design
in this study at rung 2 and has **the best rung-3 number in the table** —
2778 nm, better than the coaxial *seven*-mirror floor of 4710 nm, from **two
mirrors with no free parameter at all**. Essentially all of its 20 µm is
power: it is a beam compressor whose output collimation varies across the
field, i.e. **field curvature**.

That reverses the conclusion § O.6 would support on its own, and the
distinction matters because the two errors have different cures:

* **field-varying POWER is what a third and fourth powered mirror are for.**
  A Mersenne pair has no freedom to correct it — both powers are spent on
  magnification and on-axis collimation — so its 20 µm is a statement about
  *how few mirrors it has*, not about the off-axis geometry.
* **the coaxial family's residual is only ~60–70 % power**, and the rest is
  higher-order and sits at 4.7 µm at N = 7 after three added mirrors. That
  part did not fall as mirrors were added (§ D.3's finding, now decomposed).

**So the two families are failing for different reasons**, which is exactly
the thing a single rung-2 number hides. Stated carefully, because it is the
most consequential claim in this section and it is a decomposition rather than
a solve: *on this evidence* the off-axis route is **not** closed by the
Mersenne's 20 µm, and the arm that matters is a Mersenne front end with enough
back end to correct field curvature — which is what the Mersenne-seeded solve
in § O.7 runs.

**The zero that anchors it.** At *true* on-axis — the field offset (0, −bias),
which cancels the deck's 0.6° bias — the off-axis Mersenne scores **0.00 nm**,
exactly. A confocal parabola pair is a perfect afocal compressor for a
collimated on-axis beam and the engine reproduces that to the last bit at
h = 1.5 m, which is one more independent confirmation that the off-axis
construction of § O.2 is exact. The coaxial decks read 2.3–3.6 µm at the same
point, but that is **not** a fair comparison and is recorded here so nobody
draws it: those designs are corrected at the 0.6° bias with a ±0.25° box, so
true on-axis is 2.4 box-half-widths *outside* their design field.

### O.6c Chasing the defocus to its source — one refuted hypothesis, one self-correction

§ O.6b says the residual is field curvature. The textbook source of field
curvature is the **Petzval sum**, which needs *signed* curvatures, and a
Cassegrain compressor (convex secondary) and a Gregorian one (concave)
contribute opposite signs. The arc's existing double Mersenne is **two
Cassegrain stages**, so its contributions add — a sufficient explanation for
why relaxing four conics bought it only 1.7× (§ 4). **Nobody had built the
mixed pair.** So it was built: two confocal parabola pairs in cascade, stage 1
Cassegrain and stage 2 Gregorian, exact at m1·m2 = 30 by construction (no
closure, so § O.5's singularity does not arise), scanning the split m1.

**The hypothesis is refuted as built, and the A/B is what refutes it** —
`cass_greg` against `cass_cass` at matched splits, h = 0:

| m1 | cass_cass rung 2 | cass_greg rung 2 | cass_cass % power | cass_greg % power |
|---|---|---|---|---|
| 2 | 38246 | 56104 | 99.9 | 99.9 |
| 5 | 38040 | 56499 | 99.9 | 99.9 |
| 10 | 36846 | 57056 | 99.2 | 99.5 |
| 15 | **34450** | 56942 | 97.8 | 98.4 |

The mixed pair is **worse at every split**, both curves are **monotone with no
interior minimum** (the signature cancellation would leave), and the power
fraction stays **94.7–99.9 % for both forms everywhere**. Mixing the signs did
not touch the defocus.

**The confound, stated rather than buried:** a Gregorian stage has a REAL
internal focus, and at these speeds it is an f/2.5 focus in a 0.5 m beam. That
is the same feature that made the single Gregorian Mersenne catastrophic in
§ O.6 (1.15e6 nm, four orders off). So this experiment does not cleanly
isolate Petzval — it measures Petzval *plus* a large internal-focus penalty,
and the penalty is bigger than any benefit. **The honest verdict is that the
mixed pair does not help here, not that Petzval balancing cannot help.**

*(These cascades trace M = 28.0–33.2 and 8500–11800 µrad of collimation error;
they are family probes, not designs, and their absolute numbers should be read
only against each other.)*

**The self-correction, made before anything was recorded.** The natural next
step was to compute each design's Petzval sum and correlate it with the
measured defocus. The routine that does this **was written as a Petzval sum
and is not one**: MACOS emits `KrElt = −|R|` for *every* mirror and carries
convexity in the **geometry**, so a sum built from the radii the engine returns
cannot tell a convex secondary from a concave one. The corpus proves it —
`cass_greg` and `cass_cass` come back with the **identical** value 11.600
while their defocus differs by 47 %. The quantity is now reported under its own
name, `C = Σ ±2/|R_k|`, a curvature-**magnitude** sum, and no Petzval claim is
made from it.

What `C` does show is still worth having, because at 30× the final mirror is
necessarily the smallest and most strongly curved in the train and dominates
it:

| deck | C (/m) | R_last (m) | defocus (nm) |
|---|---|---|---|
| **descent ascent rung N = 7** | **1.663** | −0.946 | **6335** |
| descent ascent rung N = 5 | 2.973 | −1.247 | 8961 |
| committed 4-mirror | 4.260 | −1.207 | 8083 |
| off-axis Mersenne f1 = 5 | 5.800 | −0.333 | ~20500 |
| four-parabola cascades | 11.600 | −0.167 | 34000–57000 |
| arc double Mersenne | 16.000 | −0.133 | 59297 |
| off-axis Mersenne f1 = 1.25 | 23.200 | −0.083 | ~60700 |

Spearman |C| vs defocus **+0.683** over 40 decks — moderate, and heavily
tie-degraded (27 of 40 share just two `C` values), so the rank statistic
understates a group-mean relationship that is monotone. **The best design in
the entire study has the smallest |C| of any deck measured.** Establishing the
signed statement needs convexity the decks do not carry in `Kr`; that is left
**open** rather than guessed.

### O.6d The spec is not unreasonable on étendue grounds

Worth settling, because "the spec is the wall" should not be reached for
before the arithmetic is done. At D = 1 m, ±0.25° half-field, λ = 1 µm:

* Lagrange invariant **H = (D/2)·u = 2.18e-3 m·rad**
* resolvable points per dimension **2H/λ = 4363**, i.e. ~19 million resolution
  elements
* the target, 71 nm rms at 1 µm, is **λ/14.1 rms** — a Maréchal-class
  criterion, not an exotic one

For scale, LSST's invariant is ~0.128 m·rad, **59× larger**, and it is met with
three mirrors plus refractive correctors. **So the difficulty is not the
étendue.** What is unusual here is the *combination*: all-reflective, afocal at
30×, with the exit pupil placed at a specified standoff — and the 30× is what
forces the small, strongly curved final mirror that the `C` table keeps
pointing at.

### O.6e Realizability — what an off-axis section actually costs to build

`run_offaxis_gates` scores `descent_require`'s full requirement set and adds
two columns the coaxial tables have no counterpart for. On the decks § O.6
quotes:

| deck | max chief AOI | union floor | behind M1 | primary PARENT radius |
|---|---|---|---|---|
| off-axis Mersenne f1 = 2.5, h = 1.5 | **16.07°** ✗ | −68.7 mm | −2417 mm | 2023 mm |
| off-axis Mersenne f1 = 5.0, h = 1.5 | 7.92° ✓ | −81.1 mm | −4833 mm | 2023 mm |
| cass-cass cascade m1 = 15, h = 0.55 | 8.81° ✓ | −82.1 mm | −2117 mm | **1074 mm** |

**The parent is the off-axis family's bill, and it is the size of the
decenter.** An off-axis section of radius *r* taken at height *h* must be cut
from a parent of radius |h| + r, and that parent is the part somebody has to
figure and test. At the minimum clearing decenter h = 0.55 m the primary's
parent is **2.15 m in diameter for a 1 m beam** — the textbook ≈ 2h + D — and
at h = 1.5 m it is **4.05 m**. Nothing in the wavefront tables shows this, and
it is the first thing a fabricator would ask about.

**The AOI trade is real and it bites the fast forms first.** Decentering moves
the chief off *every* parent axis at once, so incidence grows on every surface
together — unlike a fold tilt, which spends its angle at one station. The
f1 = 2.5 Mersenne at h = 1.5 m **breaks the 15° standing rule at 16.07°**;
slowing to f1 = 5.0 at the same decenter brings it to 7.92°. **Reported broken
and named rather than dropped from the table** — a design that fails a
realizability wall is a finding about the family.

**Packaging fails on all three**, at −2117 to −4833 mm, which is § O.4's parity
law doing exactly what it says: these are two- and four-mirror forms with a
Mersenne front end, and that count cannot put its back end behind its own
primary at any spacing. Layout renders written per deck.


## O.7 The isolation experiment — the wavefront floor vs the pupil decenter

Everything in this experiment is the descent's committed machinery — same
closure, same merit, same DOFs (`conic, radius, spacing, tilt`), same solver
settings, same starting designs at the same 343 mm standoff — with exactly
**one** variable changed: the pupil decenter `h`. `h = 0` is the coaxial
control and must reproduce the descent's own recorded starts; `h > 0` is the
same design used off axis. Any difference between the columns is attributable
to the decenter and to nothing else.

**The control reproduces, and not merely to within a tolerance.** Start values
10407.0 / 10774.9 / 7894.1 nm at N = 4 / 5 / 7, identical to § D.3's — and the
N = 4 control **floors at 4497.7 nm**, which is § D.3's recorded N = 4 "+ tilt"
floor **to the decimal**:

| | § D.3 recorded | § O.7 h = 0 control |
|---|---|---|
| N = 4 start | 10407 | 10407.0 |
| N = 4 floor, + tilt | **4497.7** | **4497.7** |
| N = 5 start | 10775 | 10774.9 |
| N = 5 floor, + tilt | **8077.4** | **8077.4** |
| N = 7 start | 7894 | 7894.1 |
| N = 7 floor, + tilt | **3424.2** | **3424.2** |

**Three of three, to the decimal**, on multi-round solves descending by 56.8 %,
25.0 % and 56.6 %. Two details make this a sharper check than a bare
reproduction: N = 5 reproduces the **non-monotonic** rung — the one § D.3 itself
called basin scatter, so the same *basin* is being found and not merely a
similar number — and N = 7 reproduces § D.3's **best coaxial result in the
entire study**, the 48× that the descent's conclusion rests on.

An exact reproduction of solves that long is a strong statement about the
whole chain — the closure, the emitter, the merit, the
solver settings, the solve/report sampling split and the starting design are
all confirmed to be the ones that produced the record, so a difference in the
`h > 0` column cannot be laid at any of them.

**Two asymmetries were checked rather than assumed** before any comparison was
drawn: the fitted off-axis apertures clip nothing (§ O.3d, difference exactly
`0.000e+00`), and the standoff is matched at 343 mm throughout (§ O.6f is the
correction that established this matters).

### O.7a The Mersenne-seeded arm — reported UNUSABLE, not as a result

A third arm started from the Mersenne front end instead of the committed
designs, on the brief's "one seed is one basin" point. **It diverged, and the
divergence is not diagnosed, so no conclusion is drawn from it:**

| | start | round 1 | round 2 | final |
|---|---|---|---|---|
| WFE (nm) | 542 865 | 26 618 366 | 26 786 007 | **26 786 007** |
| traced M | 26.73 | | | **0.8175** |

`M = 0.8175` against a required 30 says the end point is **not a telescope**,
and the M-guard — printed on every result for exactly this reason — is what
caught it.

**Why it is not being read as "the Mersenne basin diverges".** `lsqnonlin`
returns its best iterate and the objective's failure path scores *high*
(`wallr`), so there is no attractor in the failures; the solve genuinely
reduced its own residual while the reported wavefront rose 49×. The available
explanation is the **solve/report sampling gap** — the merit is evaluated at
`P.solve.ngrid = 21` (≈ 317 rays) and the report at `P.ngrid = 41` (1185) —
which is small for a well-behaved design and can be arbitrarily large for a
pathological one. **That gap has bitten this arc before** (the wall slice's
report-versus-iterate wall). It was not run down here because the arm was a
secondary probe and the finding it was meant to support — § O.5's closure
singularity — is established *directly*, from the closure's own algebra, and
does not depend on it.

The arm was **stopped** rather than run out over its remaining decenters, and
its CPU returned to the three arms that carry the experiment. One thing it did
show in passing, consistent with § O.5: at `h = 0.55` the same spoiled closure
starts at **264 054 nm with M = 31.30**, against `h = 0`'s 542 865 nm and
M = 26.73 — the off-axis version of that closure is the better-behaved one.
