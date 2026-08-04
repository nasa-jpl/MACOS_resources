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
    point has no design": measured, the warm-started trade lost **three of five points**
    that way, all of which `afocal4_pack_seed` then closed. Seed the search inside the
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
