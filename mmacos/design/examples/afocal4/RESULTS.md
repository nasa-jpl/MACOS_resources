# S4 RESULTS — the joint solve, the answer ladder, and what the fourth mirror actually buys

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

## 2. The answer ladder

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

## 3. The interface-standoff trade — the exchange rate

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
