# S3 FORM STUDY — which fourth mirror, and why

Answers `PLAN_AFOCAL4.md` S3 against the benchmark in `challenges/rodgers2/PACKET.md`:
J.M. Rodgers' coaxial 30× afocal TMA, EPD 1000 mm, λ = 1 µm, 0.5°×0.5° field box
offset +0.6° in Y, delivering a collimated beam to an interface pupil — and his verbal
finding that *"with 3 mirrors the pupil quality is not very good; a 4th mirror is needed
for pupil control."*

> **S4 is delivered.**  The joint solve, the answer ladder, the interface-standoff trade
> curve and the Mersenne hedge are in **`RESULTS.md`**, beside this file.  Read that for
> what the fourth mirror actually buys once the design is optimised; this file is the
> first-order argument that chose the form.

**Scope: first order only.** Every layout here is closed in algebra, built, traced and
measured with the conics carried from the parent and the new mirror seeded at K = 0.
Nothing is optimised. The output is a ranking and its reasons, not a design; the joint
solve is S4.

Reproduce: `afocal4_forms` (≈15 min, model size 256). Numbers below are from that run
and live in `afocal4_forms.mat`.

---

## 0. Verdict up front

1. **The three-mirror already closes both first-order conditions.** It recollimates
   (u_out = −3.1e−6), magnifies by **30.000**, and images the stop onto a plane
   **343.363 mm** past M3 — **0.81 mm** from the coldstop Rodgers placed by hand, on a
   33 mm beam. So there is no first-order deficiency for a fourth mirror to repair, and
   **a form study that closes only the first-order conditions returns a flat** (the
   closed field-mirror radius is 45–78 km). The fourth mirror has to be justified on
   pupil *aberration*, and the real question is which form can be *given* power without
   spending the first-order solution to pay for it.

2. **Recommendation: (i) the field mirror near the intermediate image, convex.** It is
   the only form that preserves the first-order solution *identically* — sitting where
   the marginal ray is small, it cannot change M or the afocal condition for any power —
   so its power is a free parameter that acts on the chief ray almost alone. One
   unoptimised K = 0 mirror cuts the pupil blur **2.8×** (794 → 286 µm), holds
   magnification at 29.92, and at φ₄ = +2 /m drives the chief-normal breathing to
   **0.156%**, *inside* the ±0.4% S3 target, with every conic still unspent.

3. **Runner-up, kept alive: (ii) the double Mersenne.** It has the best pupil ladder of
   anything measured — blur **175 µm**, breathing **0.285%**, wander **468 µm**, all
   better than the field mirror — and it is 26% shorter. It loses on two counts that are
   not obviously fatal and one that might be: its rung-2 WFE is **59 µm**, and its
   interface pupil lands **53 mm** past the last mirror. See §5 for what would change
   the ranking.

4. **(iii) the downstream relay is eliminated.** Worse than the *parent* on every pupil
   term (blur 1939 µm, wander 11.7 mm, breathing 7.6%), 39 µm of WFE, a 25% longer
   train, a real internal focus in air, and it cannot reach the parent's interface
   distance without R₃ → 0.

5. **A collimated exit space has no first-order freedom.** "A fourth mirror downstream
   of M3" cannot mean a powered element in the exit beam — that breaks the afocality
   that defines the system. It necessarily means re-splitting the collimation between M3
   and M4, which is what candidate (iii) is, and which converts the back end into a
   focal relay with everything that implies.

6. **A convex field mirror helps and a concave one hurts.** The leverage is *signed*:
   φ₄ = −1 /m makes every pupil term worse than the parent (blur 3077 µm, breathing
   7.2%). That is a design rule earned here, not a preference.

---

## 1. The parent, and the conditions it already meets

`afocal_first_order` (design/src) traces the two paraxial rays that between them carry
every first-order property an afocal telescope is specified by: the **marginal** ray
gives the afocal condition, the exit beam and hence M by the Lagrange invariant; the
**chief** ray gives the pupil.

| | value |
|---|---|
| afocal condition u_out | −3.103e−06 rad |
| magnification (chief) | 30.000096 |
| magnification (Lagrange) | 30.002014 |
| exit beam | 0.0333311 m (D/M = 0.0333333) |
| **exit pupil past M3** | **0.343363 m** |
| his coldstop past M3 | 0.344173 m → **0.81 mm apart** |

His coldstop station is an input to nothing above. That the paraxial exit pupil lands
within a millimetre of it is the witness that the transcription, the sign conventions
and the paraxial model all agree — and it is the fact that reframes the whole study.
Staged as `optical_design/fixtures/afocal_tma_fixture.{json,md}`, a stop-and-fix gate,
pinned by `tDesignAfocal/test_first_order_kernel_matches_the_fixture`.

**Which variant.** S1 (on-axis), not S3. His later re-solves moved R₃ by 3.8% to
recollimate the *real* marginal ray at f/1.25; paraxially they leave 41 µrad of
convergence and a 4.8% small exit beam. Both are right — his is a real-ray solution,
this is a paraxial study — but seeding a first-order form study from S3 builds a 31.4×
telescope. That cost one debugging pass and is recorded in `afocal4_params`.

---

## 2. Candidate (i) — field mirror near the intermediate image

### The algebra

The front end (M1, M2, their spacing) is untouched, so the paraxial state leaving M2 is
known and the intermediate image sits 1.399266 m past it. Put the field mirror at
standoff *s* before that image and let the collimator follow. Writing the marginal ray:

```
y₄ = y₂ + a·u₂                      a = a₀ − s,  the field mirror's marginal height
u₄ = u₂ − y₄·φ₄
b  = (y_out − y₄) / u₄              collimator station:  imposes M
φ₃ = u₄ / y_out                     collimator power:    imposes the afocal condition
```

**The two first-order conditions are not a solve — they are a substitution.** They
consume the collimator's station and power exactly, whatever φ₄ is. What is left over is
the chief ray, and the exit-pupil station is the one number φ₄ actually buys:

```
y_c4 = y_c2 + a·u_c2                u_c4 = u_c2 − y_c4·φ₄
y_c3 = y_c4 + b·u_c4                u_c3 = u_c4 − y_c3·φ₃
d    = −y_c3 / u_c3
```

Setting *d* = the interface spec closes φ₄ in one root-find. On this benchmark that root
is **φ₄ ≈ 3e−5 /m — a 45–78 km radius, i.e. a flat** — at every standoff, because the
parent already meets the condition. The design freedom is therefore the *family*, not
the root.

### At the exact image the mirror is free, and that is the whole idea

At *s* = 0 the marginal height y₄ is zero, so φ₄ drops out of the marginal ray
completely: `b` and `φ₃` come back exactly the parent's, and **M and the afocal
condition are preserved identically for any power**. Verified numerically to
|Δmag| < 4e−16 across the standoff scan. That is the property no other candidate has.

The cost is equally exact: a mirror *at* an image has **zero footprint**. The standoff
buys it back at 2·s·|u| = 5.74 mm at 50 mm of standoff, 22.96 mm at 200 mm — and the
price of standoff is that φ₄ now perturbs the marginal ray, so the collimator re-solves
(≈11% of R₃ per 0.5 /m). Layouts below use **s = 200 mm**.

### The family, and its leverage

At s = 200 mm (interface distance floats):

| φ₄ (1/m) | R_FM (m) | | b (m) | R₃ (m) | exit pupil (m) |
|---|---|---|---|---|---|
| −1.00 | 2.0000 | convex | 0.6130 | 0.7260 | 0.5214 |
| −0.50 | 4.0000 | convex | 0.5449 | 0.6454 | 0.4225 |
| −0.25 | 8.0000 | convex | 0.5162 | 0.6114 | 0.3808 |
| +0.25 | 8.0000 | concave | 0.4671 | 0.5532 | 0.3095 |
| +0.50 | 4.0000 | concave | 0.4458 | 0.5280 | 0.2786 |
| +1.00 | 2.0000 | concave | 0.4087 | 0.4840 | 0.2247 |

**−0.148 m of pupil station per (1/m) of field-mirror power, at constant M.**

The family is two-dimensional — interface distance (m) versus (φ₄, standoff):

| φ₄ \ s | 50 mm | 100 mm | 200 mm | 300 mm | 400 mm |
|---|---|---|---|---|---|
| 0.50 | 0.2951 | 0.2894 | 0.2786 | 0.2688 | 0.2598 |
| 1.00 | 0.2492 | 0.2403 | 0.2247 | 0.2115 | 0.2002 |
| 2.00 | 0.1636 | 0.1544 | 0.1399 | 0.1290 | 0.1206 |
| 3.00 | 0.0855 | 0.0817 | 0.0763 | 0.0726 | 0.0700 |
| 4.00 | 0.0138 | 0.0194 | 0.0269 | 0.0316 | 0.0349 |

Standoff is the **weak** axis: 10–20% over 50–400 mm, and below φ₄ ≈ 3.5 it moves the
interface the wrong way. So the interface distance is essentially the field mirror's
*power*, and the standoff is free to be set by footprint. **The two knobs are nearly
orthogonal**, which is what a joint solve wants.

> Note the sign convention: in the builder's terms the *emitted* mirror is convex for
> φ₄ > 0 in the table above — the table's "convex/concave" column is the unfolded
> thin-lens sense and the beam arrives at the field mirror travelling +z. What the
> measurements below call `field_p1/p2/p4` are the **convex-emitted** members, and they
> are the ones that help.

---

## 3. Candidate (ii) — double Mersenne

Two confocal pairs, M = m₁·m₂. Each pair is afocal by construction, so the afocal
condition and M are satisfied *identically* for every member — the only free closure is
the inter-stage gap, and the pupil condition fixes it.

**And it cannot be fixed.** Over the whole (flavour × m₁ × stage-2 f/# × gap) scan:

| second pair | exit-pupil station |
|---|---|
| Cassegrain type (convex M4, no internal focus) | **−0.038 to −0.176 m — virtual, behind M4** |
| Gregorian type (concave M4, real internal focus) | **−0.049 to +0.155 m** |

A Cassegrain second pair puts the exit pupil *behind* the last mirror, where nothing can
be interfaced to it. The Gregorian flavour makes it real but caps it at ~0.16 m — **less
than half** the 0.343 m the three-mirror delivers. A double Mersenne relays its pupil by
construction and then puts the relayed pupil in the wrong place. (The stage-2 f/# is the
lever that moves it: f/1 → 0.013 m, f/4 → 0.155 m at m₁ = 6, so the fix costs stage-2
speed and hence size.)

Layout measured: Gregorian second pair, m₁ = 6, stage-2 f/2, gap 0.5 m, all four
mirrors parabolic. Interface 0.053 m.

---

## 4. Candidate (iii) — downstream relay

The exit space of an afocal train is **collimated**, and a collimated space admits no
powered element without destroying the afocality that defines the system. So "M4
downstream of M3" has to mean re-splitting the collimation: M3 forms a **second image**
and M4 collimates it. Same substitution as §2 — M4's station and power are forced by the
two first-order conditions, and the chief ray is the residue.

M3 must be given **more** power than the parent's collimating 3.4435 /m: below it the
beam leaves M3 still diverging and never returns to the axis, and just above it M3 and
M4 stack into one mirror. Above it:

| φ₃ (1/m) | R₃ (m) | M3→M4 (m) | R₄ (m) | exit pupil (m) | train (m) |
|---|---|---|---|---|---|
| 3.600 | 0.5556 | 12.792 | 12.793 | 13.137 | 15.874 |
| 4.000 | 0.5000 | 3.595 | 3.595 | 3.939 | 6.677 |
| 5.000 | 0.4000 | 1.285 | 1.285 | 1.629 | 4.367 |
| **6.000** | **0.3333** | **0.782** | **1.126** | **1.126** | **3.865** |
| 8.000 | 0.2500 | 0.439 | 0.782 | 0.782 | 3.521 |
| 20.000 | 0.1000 | 0.121 | 0.464 | 0.464 | 3.203 |
| 40.000 | 0.0500 | 0.055 | 0.398 | 0.398 | 3.137 |

The station approaches the parent's 0.343 m **from above**, reaching it only as R₃ → 0.
φ₃ = 8 — a 250 mm tertiary, f/0.75 on the beam it carries — still lands at 0.78 m. So
the relay *can* close the pupil condition, but never at the interface distance the
three-mirror already delivers with a manufacturable tertiary. The price is a longer
interface, a 25% longer train, and a **real internal focus in air** between M3 and M4 —
a stray-light and contamination site, and equally a place to put a field stop.

Layout measured: φ₃ = 6 (R₃ = 333 mm), M4 782 mm downstream, interface 1.126 m.

---

## 5. The comparison

All layouts built through the same builder path, traced, and verified against their own
paraxial prediction on axis before any metric was taken (`traced == paraxial` for all
seven; that gate caught a builder bug — see §7). Pupil ladder from `pupil_map`, cone
aperture = the 3×3 solve set, anchored on the M1 surface. **Magnification is
chief-normal** — a footprint read on the placed plane carries that plane's own 1/cos
obliquity (rodgers2 PACKET §4 refinement). WFE is the afocal rung 2 (piston + per-field
tip/tilt), in-box max on a uniform 9×9 grid.

| candidate | blur rms µm | blur max µm | M (chief-normal) | breathing ±% | wander µm | surface P-V mm | WFE rung 2 nm | interface m | train m |
|---|---|---|---|---|---|---|---|---|---|
| parent3 (3-mirror) | 794.0 | 1708.0 | 28.6835 | 3.966 | 1415.4 | 0.0144 | **429** | 0.343 | 3.082 |
| field_m1 (φ₄ = −1) | 3077.4 | 5196.1 | 27.6315 | 7.171 | 3604.2 | 0.0622 | 5327 | 0.521 | 3.205 |
| **field_p1 (φ₄ = +1)** | **286.5** | **737.0** | 29.4692 | 1.588 | 1902.4 | 0.0137 | 5702 | 0.225 | 3.001 |
| **field_p2 (φ₄ = +2)** | 338.6 | 728.8 | **29.9165** | **0.156** | 1644.5 | 0.0307 | 14046 | 0.140 | 2.942 |
| field_p4 (φ₄ = +4) | 389.6 | 1016.4 | 29.4315 | 1.555 | 856.1 | 0.0301 | 47012 | 0.027 | 2.864 |
| relay (φ₃ = 6) | 1939.0 | 3682.1 | 27.8258 | 7.603 | 11678.8 | 0.1623 | 39020 | 1.126 | 3.865 |
| **mersenne (greg, m₁ = 6)** | **175.0** | **473.3** | 29.9427 | **0.285** | **468.4** | 0.0613 | 59375 | 0.053 | 2.285 |
| **S3 TARGET** | **47.0** | — | **30.0000** | **0.400** | **56.0** | **0.200** | **71** | — | — |

Control: `parent3` is his S1 radii at the 0.6° bias — which is his S2 variant — and it
reproduces the rodgers2 baseline (blur 794.0 vs 801.6 µm, chief-normal M 28.6835 vs
28.6848) through an entirely different pipeline: the design layer's builder instead of
the hand transcription. That agreement is what licenses the rest of the table.

AOI spread is 14.6–14.7° for every candidate — inside the 15° coronagraph preference,
and *set by the parent front end*, so the fourth mirror does not buy or cost anything
there. Shroud radius is 0.99 D for all: a 1 m primary sets it.

### Reading it

**No first-order layout meets any target except breathing** — expected, and the point.
These carry the parent's conics on a re-solved collimator and a K = 0 new mirror; the
WFE column is what an *unoptimised* four-mirror system does, not what one can do.

What the table decides is **leverage per unspent degree of freedom**:

* The field mirror moves blur 2.8× and breathing 25× in the right direction from a
  single unoptimised surface, with **all four conics, the standoff and the collimator
  re-solve still free**, and with M and the afocal condition preserved by construction
  rather than by solving. Its breathing minimum near φ₄ = +2 (0.156%) is a genuine null
  of the pupil-distortion term, not a monotone trend — there is something there for a
  solve to sit on.
* The double Mersenne is already the best pupil in the table, and its 59 µm of WFE is
  not a property of the *form* — it is the price of insisting all four mirrors be
  parabolas, which the name requires and the design does not.
* The relay is worse than doing nothing, on every pupil term.

### What would change the ranking

**Promote the double Mersenne** if a conic-relaxed version — keeping the confocal
*spacings*, which is what delivers the pupil relay, and spending the four conics on
image quality — reaches DL in-box while holding its 0.285% breathing. It would then win
outright: its pupil numbers are 2–5× better than the field mirror's and it is 26%
shorter. That is a well-posed S4 experiment: four conics against a
sixty-micron-to-seventy-nanometre problem.

**What still kills it**: the 53 mm interface distance, which is not an instrument
interface. The stage-2 f/# lever reaches only ~0.16 m, and buying more costs stage-2
speed and hence size. If the instrument's interface standoff is soft, this objection
goes away; if it is hard, the double Mersenne is out regardless of its pupil.

**Demote the field mirror** if the breathing null at φ₄ ≈ +2 turns out to be an artefact
of the carried conics rather than a property of the layout — i.e. if it moves or
disappears once the conics are re-solved. Test it by re-running the φ₄ sweep after a
conic solve at one power.

**Do not revisit the relay** unless the interface distance requirement changes to
something ≥ 0.8 m, where the relay's natural station is and the other two forms have to
strain to reach.

---

## 6. Recommendation

**Take candidate (i), the field mirror, convex, at ~200 mm standoff, into the S4 joint
solve.** Seed at φ₄ ≈ +2 /m (R = 1.0 m), which is where the breathing null sits, and
carry the interface distance as a constraint rather than a consequence: at that power it
is 140 mm, and if the instrument needs more, φ₄ comes down and the solve makes up the
pupil quality from the conics.

**Carry the double Mersenne as a live alternative** with a single, bounded experiment:
relax its four conics and see whether it reaches DL. It is the only form whose pupil
ladder already touches an S3 target, and it is the shortest train in the study.

**Drop the relay.**

The deeper finding to carry into S4, and into anything that goes back to Mike: **the
fourth mirror is not repairing a first-order deficiency — there isn't one.** His
three-mirror recollimates at exactly 30× and puts its exit pupil 0.8 mm from his
coldstop. What the fourth mirror buys is a surface that can be given power *without*
being paid for out of the first-order solution, and among the three candidates only the
field mirror at the intermediate image has that property exactly.

---

## 7. Traps recorded

1. **A paraxial prediction must be checked against an ON-AXIS trace.** The first version
   verified every candidate at the 0.6° design bias and reported the three-mirror
   control as a 5% builder failure (traced M 28.38 against a paraxial 30.00). Both
   numbers were right; the comparison was not. 28.38 is his S2 variant's magnification —
   a correct answer to the wrong question. The check now runs two passes, on axis and at
   the bias, and reports both.

2. **The builder emitted a convex third mirror as concave.** `resolve_nmirror_` forced
   `psi_z = −1` for mirrors 1–3 and applied the parity rule only from the fourth. That
   is right for every focal design in the corpus — a Korsch M3 is concave — and wrong
   for the field-mirror form, which is the first design here to put a *convex* mirror
   third. Symptom: `field_m1` traced 17 mrad off collimated with a magnification of
   19.4 against a paraxial 30.0. The parity rule now starts at k = 3; it agrees with the
   legacy −1 for every non-convex k = 3 case, so no focal design changes (fast 236/236).
   Caught only because the study verifies the trace against its own paraxial prediction
   before taking a metric — a metric on a mis-emitted deck is a number about nothing.

3. **`relay_d_` was given the wrong distance and the wrong sign.** It received the
   image→M3 distance where it needed M2→M3, and assumed the three-mirror's exit marginal
   sign — but a relay adds a *second* axis crossing, so the sign flips. Together they
   produced a plausible-looking table with the pupil trend running backwards. The sign
   is now taken from the requirement that M4 sit downstream of M3, not assumed, and a
   minimum-separation guard rejects the degenerate branch where the two mirrors stack.

4. **A rational closure needs a tight bracket.** `d(φ₄)` has poles; a wide `fzero`
   bracket straddles one and returns the pole rather than the root. ±1 /m is the
   physical window and `d` is monotone across it.
