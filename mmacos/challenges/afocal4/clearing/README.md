# afocal4 clearing — getting the collimator out of its own feed beam

Origin: `macos/BRIEF_afocal4_clear.md` (Dave, 2026-08-30), which came out of the
packaging stage's bigger finding: on the committed 343 mm family-2 design the
**M2 → field-mirror feed beam runs through the collimator's own glass** — `−79.9 mm`
with the declared body model, `−55.4 mm` against bare lit glass, over the
0.5°×0.5° field box.

**Nothing under `challenges/afocal4/` was overwritten.** Every deck this stage
writes is new and lives here. Two files beside it are *added*, not changed:
`../afocal4_union.m` (the gate the brief asks to be promoted into the afocal4
gate set) and one additive clause in `../afocal4_pack.m` that calls it.

Every number below is engine truth — traced rays and the engine's own element
getters — measured over the **whole field box**, never the deck's own single
field. No `.in` file is text-parsed for geometry.

---

## 1. The answer in four lines

1. **The interference is structural, not a layout slip.** It obeys a *field-walk
   ratio law*: the collimator's footprint and the feed beam's are both scaled
   copies of the same off-axis field box, so they separate only if their scales
   differ by more than `(bias+half)/(bias−half) = 0.85/0.35 = 2.4286`. Measured
   ratio: **1.3025**. And the collimator's own scale is *pinned* at `M × iface`
   by the interface specification — 10.290 m/rad against a traced 11.118,
   **+8.1 %**, which is this design's own pupil aberration.
2. **Leverage 1 (a flat extraction fold) cannot do it**, and the reason is exact:
   the collimator lands *on* the flat at one particular station, and the two
   failure modes are complementary about it. Best over every station and both
   turn directions: **−79.89 mm — the parent's own number, to the last digit.**
3. **Leverage 2 (the collimator station) cannot do it either.** Over **1.40 m**
   of collimator travel the ratio reaches **1.48**; the demanded footprint runs
   83.6…100.3 mm against 23.0…34.1 mm available. The curves never cross.
   **Every point of the committed trade curve fails the gate** — −29.0 / −33.4 /
   −54.2 / −67.5 / −79.9 mm at 50/90/140/220/343 mm. At 50 mm the *collimator*
   pair does clear (+36.8 mm, ratio 11.7), but its collimator sits **5.21 m**
   behind the primary and the deck then fails on a different pair.
4. **Leverage 3 (an extraction tilt on the field mirror) does it**, because a
   tilt is the one remedy the law does not bind: it separates the two bundles by
   a **field-independent** `2αd`, measured directly as a fitted offset of
   **+195.3 mm** at −10°. The gate goes from **−79.89 mm** to **+57.44 mm** raw
   and **+37.82 mm** on the delivered re-solved design, zero rays lost — and the
   price is paid in the **pupil**, not the wavefront (which improves 13.6 %).

---

## 2. The defect, and the number that hid it

Sky is at −z, so behind the primary is +z. Committed deck
`../afocal4_b2long_343mm.in`, model 256, his 3×3 field box.

| element | union footprint r | body r (1.15× + 15 mm) | beam offset from vertex | clear aperture the part needs |
|---|---|---|---|---|
| M1 | 500.1 | 574.3 | 0.3 | 515.3 |
| M2 | 89.9 | 103.1 | 11.1 | 116.0 |
| FM (field mirror) | 112.6 | 129.0 | 183.9 | 311.5 |
| **M3 (collimator)** | **87.0** | **99.6** | **111.7** | **213.6** |
| ColdStop | 20.0 | 23.0 | 0.1 | 35.1 |

(mm. The primary needs a 174.0 mm through-hole; that is a requirement, not a
collision, and is true of the unfolded design too.)

Signed clearance floors, `leg M2→FM` against the collimator's body:

| model | floor |
|---|---|
| declared body, 1.15 × union footprint + 15 mm | **−79.89 mm** |
| **bare lit glass**, 1.00 ×, 0 mm | **−55.36 mm** |
| what **one** field sees | **−5.9 … +26.8 mm** |

**That last row is the whole reason this shipped.** Per field the collimator has
tens of millimetres of daylight; the union of nine fields is 80 mm inside the
beam. A centre-field check passes it. `afocal4_pack` ran exactly such a check —
it asks where there is *daylight* and never asks whether a *part* is already in
the way. A margin is a number, not a body.

---

## 3. Why — the field-walk ratio law (`clear_law`)

On a coaxial train every chief-ray height is proportional to its field angle, so
a part's union footprint and a beam's union footprint at the same station are two
scaled copies of the **same** field box, anchored at the axis. Write each
centroid as

```
    centroid(theta) = c * theta + o
```

* `c` — the **field-proportional walk**. On a coaxial train this is all there is.
* `o` — the **field-independent offset**. Identically zero on a coaxial train,
  and the *only* quantity that can defeat the law.

Two scaled copies of the box `[B−A, B+A]` are disjoint only if
`c_beam / c_body > (B+A)/(B−A)`. Measured on the committed deck:

| quantity | value |
|---|---|
| field box in the bias direction | 0.3500 … 0.8500 deg |
| **ratio the box demands** | **2.4286** |
| walk, collimator body | 11.1179 m/rad |
| walk, feed beam at the collimator's plane | 14.4811 m/rad |
| **ratio achieved** | **1.3025** |
| field-independent offset | +4.46 mm (i.e. zero — it is coaxial) |
| centre-set gap | −72.02 mm |
| … minus the two beam radii (19.0 + 16.1 mm) | **−107.15 mm** |
| residual of the walk+offset fit | 1.2e-03 m |

And one of the two scales is **pinned by the specification**: the chief converges
from the last powered mirror to the exit pupil `iface` away at the exit angular
magnification, so

```
    c_body(collimator) = M * iface = 30 * 0.343 = 10.29 m/rad
    measured                                      11.12 m/rad   (+8.0 %)
```

The pin is paraxial and the measurement is a traced footprint centroid, so the
8 % is this design's own pupil aberration — the quantity the fourth mirror exists
to control. What matters is that the *scale* is set by the interface spec and not
by anything a layout change can reach.

That is why it is the *collimator* standing in the beam and not some other part,
and why the interference grows with the interface standoff — the same knob the S4
ruling already carries as the operating point, now pulling a third way.

**What the law rules out:** every remedy whose separation is proportional to
field. A different collimator station moves `c_beam` by the chief slope alone; a
different interface standoff moves `c_body`; a flat fold moves *neither*, because
an isometry carries both copies together. What it does **not** rule out is a
tilt, whose `o` does not shrink with the field angle.

---

## 4. Leverage 1 — one flat extraction fold. Retired. (`clear_fold`)

A fold inserted *between* the two conflict partners re-routes one of them, so the
"an isometry carries every clearance across unchanged" objection genuinely does
not apply — which is why this is the first thing to try.

Let `LEG` be the M2 → field-mirror vertex spacing and `B` the field-mirror →
collimator spacing. Put the flat `DIST` along the leg; on the folded axis

```
    fold 0,   field mirror (LEG − DIST),   collimator (LEG − DIST − B)
```

so **the collimator sits exactly on the flat when `DIST = LEG − B`** — here
2.3654 m, **0.808 of the leg**. Either side of that station:

| where the flat goes | what binds | floor |
|---|---|---|
| `DIST < 0.808 LEG` | the fold has not separated the partners — the piercing piece of feed beam travels with the collimator | **−79.89 mm, the parent's own number** |
| `DIST > 0.808 LEG` | the return beam comes back through the flat, which is union-sized (~250 mm across) because the field walk at the intermediate image dominates the beam | −76.9 … −103.4 mm |
| very early stations | the flat lands in the M1→M2 beam | −77 … −111 mm |

The two conditions are complementary and meet at a single station, so there is no
window. **Best over every station and both turn directions (+x and −y): −79.89 mm.**
A flat buys exactly nothing.

The obvious escape — put the flat far enough before the focus that the returning
beam misses it sideways — is closed by the same law: the outgoing and returning
bundles at that station are two scaled copies of the field box whose ratio only
reaches 2.43 about a metre *past* the field mirror, well beyond where the
collimator is allowed to be.

---

## 5. Leverage 2 — the station scan. Retired, with the figure. (`clear_scan`)

**Figure: `afocal4_clear_scan.png`.** Two panels, `r_demand` (the union footprint
the part must carry) against `r_available` (the largest that part could be and
still only *touch* the feed beam). They never cross.

### 5a. The collimator station, interface held

Sweeping the field-mirror standoff moves the collimator through 1.40 m of z:

| s_FM (mm) | z_collimator (m) | demand (mm) | available (mm) | collimator pair, bare | collimator pair, body | ratio |
|---|---|---|---|---|---|---|
| −600 | +2.1119 | 100.3 | 33.9 | −66.40 | −91.37 | 1.480 |
| −450 | +1.9012 | 94.8 | 34.1 | −60.78 | −85.96 | 1.465 |
| −300 | +1.6905 | 91.1 | 33.2 | −57.90 | −82.77 | 1.422 |
| −200 | +1.5501 | 89.2 | 32.3 | −56.94 | −81.66 | 1.382 |
| −100 | +1.4096 | 87.7 | 31.8 | −55.97 | −80.56 | 1.335 |
| **−38.6** | **+1.3234** | **87.0** | **31.6** | **−55.36** | **−79.89** | **1.303** |
| +50 | +1.1990 | 86.0 | 31.6 | −54.48 | −78.93 | 1.253 |
| +150 | +1.0585 | 85.2 | 24.8 | −60.37 | −85.17 | 1.193 |
| +250 | +0.9181 | 84.5 | 23.4 | −61.05 | −85.32 | 1.130 |
| +400 | +0.7074 | 83.6 | 23.0 | −60.63 | −84.84 | 1.031 |

**The station is nearly powerless**: the best ratio anywhere is 1.48 against the
2.43 the field box demands, and the floor never gets within 54 mm of zero.

### 5b. The interface standoff, front end and conics held

| iface (mm) | demand (mm) | available (mm) | collimator pair, bare | collimator pair, body | ratio |
|---|---|---|---|---|---|
| 50 | 42.7 | 50.2 | **+7.47** | −13.17 | 3.671 |
| 90 | 48.2 | 40.9 | −7.23 | −28.43 | 2.975 |
| 140 | 55.3 | 29.4 | −25.94 | −47.85 | 2.394 |
| 200 | 64.3 | 19.3 | −44.97 | −66.77 | 1.929 |
| 260 | 73.6 | 21.4 | −52.23 | −75.06 | 1.609 |
| 343 | 87.0 | 31.6 | −55.36 | −79.89 | 1.303 |
| 400 | 96.4 | 33.2 | −63.24 | −89.26 | 1.149 |

The ratio crosses 2.4286 between 140 and 90 mm, exactly as the law predicts —
but even at 50 mm the **declared body allowance still does not clear** (−13.17
mm on the collimator pair, −39.43 mm on the gate as a whole); only bare glass
does, by 7 mm.

### 5c. The committed trade curve, each deck as it was actually solved

Two columns, because on one deck they disagree in the way that matters: the
**collimator pair** is what this stage is about, the **gate** is the minimum over
every pair.

| deck | z_collimator (m) | demand (mm) | collimator pair, bare | collimator pair, body | **GATE (all pairs)** | ratio |
|---|---|---|---|---|---|---|
| `b2long_050mm` | **+5.2101** | 28.6 | +55.49 | **+36.77** | **−29.04** | 11.74 |
| `b2long_090mm` | +0.5001 | 36.4 | −13.83 | −33.40 | **−33.40** | 3.351 |
| `b2long_140mm` | +0.6829 | 46.6 | −33.43 | −54.18 | **−54.18** | 2.328 |
| `b2long_220mm` | +0.8737 | 62.6 | −45.61 | −67.51 | **−67.51** | 1.660 |
| `b2long_343mm` | +1.3234 | 87.0 | −55.36 | −79.89 | **−79.89** | 1.303 |

**Every delivered trade point fails the gate.** At 90–343 mm the collimator pair
*is* the worst pair. At 50 mm the collimator pair genuinely clears (+36.8 mm,
ratio 11.7) — but its collimator sits **5.21 m** behind the primary and its field
mirror at 5.95 m, a back end nearly six times the M1–M2 spacing and not a
package; and the deck still fails, at −29.0 mm, on its field-mirror → collimator
leg against the cold stop's body. **Retreating along the operating point is not a
way out**, and the 50 mm row is the reason to report both columns rather than the
one this stage happens to be studying.

`b2rim_343mm` (the rim-convention sibling of the 343 mm point, measured
separately) reads −55.48 / −80.00 — the same design to within the convention.

---

## 6. Leverage 3 — the extraction tilt. Delivered. (`clear_tilt`, `clear_price`, `clear_solve`)

`clear_tilt` swings one mirror about the point **where the chief ray actually
strikes it** — read from the traced ray history, never from the vertex — computes
the new outgoing chief from the *rotated local surface normal* (itself engine
truth, `N = unit(d_in − d_out)`), and re-poses every downstream element by the
rotation that carries the old outgoing chief onto the new one. Measured:

* the chief-ray path up to the swung mirror moves **4.45e-16 m**;
* the pivot stays on the surface — the chief's local incidence changes by
  **10.0000°** for a 10.0000° tilt.

### 6a. What it buys, and why the law cannot stop it

At −10° on the field mirror, `clear_law` reports the walk ratio **unchanged**
(1.30 → 1.37, still hopeless) and the fitted **offset** at **+195.25 mm**. The
gap goes `−72.02 → +124.88 mm` of centre-set separation, `−107.15 → +88.80 mm`
with the radii. That is the mechanism, isolated: the tilt does not improve the
proportional part at all — it adds a term the proportional part cannot see.

### 6b. The raw exchange rate (before any re-solve)

| tilt (deg) | bare (mm) | body (mm) | offset (mm) | WFE (nm) | blur (µm) | breathing (%) | wander (µm) | lost | binding pair |
|---|---|---|---|---|---|---|---|---|---|
| −14 | +73.20 | +56.81 | +275.4 | 10353.1 | 2122.7 | 1.8383 | 2136.4 | 0 | FM→M3 vs ColdStop |
| −12 | +73.48 | +57.15 | +234.6 | 10351.5 | 1626.2 | 1.5499 | 1638.6 | 0 | FM→M3 vs ColdStop |
| **−10** | **+73.72** | **+57.44** | **+195.2** | 10353.3 | 1199.9 | 1.2675 | 1210.8 | 0 | FM→M3 vs ColdStop |
| −9 | +66.43 | +42.25 | +176.0 | 10355.3 | 1012.9 | 1.1278 | 1022.9 | 0 | feed vs collimator |
| −8 | +47.74 | +23.34 | +156.9 | 10358.2 | 843.5 | 0.9890 | 852.5 | 0 | feed vs collimator |
| −6 | +9.96 | −14.95 | +119.0 | 10366.1 | 558.0 | 0.7137 | 565.1 | 0 | feed vs collimator |
| −4 | −28.26 | −53.31 | +81.2 | 10376.9 | 346.4 | 0.4409 | 352.0 | 0 | feed vs collimator |
| −2 | −62.26 | −86.63 | +43.2 | 10390.5 | 214.5 | 0.1701 | 219.1 | 0 | feed vs collimator |
| **0 (parent)** | **−55.36** | **−79.89** | **+4.5** | **10407.0** | **157.0** | **0.1240** | **161.2** | 0 | feed vs collimator |
| +2 | −62.42 | −86.77 | −35.3 | 10426.2 | 131.3 | 0.3929 | 135.3 | 0 | feed vs collimator |
| +4 | −52.25 | −77.31 | −76.4 | 10448.2 | 102.6 | 0.6613 | 106.5 | 0 | feed vs collimator |
| +6 | −62.14 | −86.52 | −119.5 | 10473.1 | 129.4 | 0.9296 | 131.9 | 0 | feed vs collimator |
| +8 | −23.49 | −47.98 | −164.9 | 10500.9 | 271.4 | 1.1987 | 273.2 | 0 | feed vs collimator |

Four things to read off this table.

* **The offset column is the law's own prediction, measured.** It runs
  **19.35 mm per degree** of tilt, dead linear, against the `2αd` the geometry
  says: `2 × (1° in rad) × 0.5632 m = 19.66 mm/deg`, where 0.5632 m is the
  field-mirror → collimator spacing. **1.6 % agreement**, and it is what makes
  "a tilt supplies a field-independent term" a measurement and not a story.
* **The wavefront is not the price.** A mirror at the field conjugate carries
  1.8 mm of beam per field against a 113 mm union footprint, so swinging it
  barely touches the wavefront: 10407 → 10353 nm, *better* by 0.5 %. What it
  moves is the **pupil** — the one thing the fourth mirror was added to control.
* **The clearance saturates at −10°.** Past that the binding pair stops being the
  collimator and becomes a pre-existing one (the FM→M3 leg against the cold
  stop's body, +57 mm), so more tilt buys nothing and costs more blur. −10° is
  the delivered operating point for that reason, not by preference.
* **The sign matters and the design was not at its own pupil optimum.** A
  *positive* tilt makes the clearance worse and the blur **better** (102.6 µm at
  +4°, against 157.0 at zero) — 35 % of pupil blur was available for free and
  the S4 merit did not find it, because the merit is dominated by a wavefront
  term 130× off its target.

The perpendicular axis was measured too and is worse per millimetre won: a
+10° tilt about **y** buys +27.30 mm of floor for 930.6 µm of blur, where −10°
about **x** buys +57.44 mm for 1195.6 µm — **2.1× the clearance for 1.3× the
blur**. The bias plane is the cheaper axis, which is not obvious in advance.

### 6c. The price after a re-solve

Conics, the field-mirror standoff and the front end re-solved around a fixed
tilt (427 evaluations each, ~45 min, forward differences at the study's own
3e-3 scaled step):

| tilt (deg) | bare (mm) | body (mm) | offset (mm) | WFE (nm) | blur (µm) | breathing (%) | wander (µm) | M |
|---|---|---|---|---|---|---|---|---|
| −8 | +26.59 | **+2.32** | +143.1 | 9868.0 | 741.7 | 0.9194 | 750.5 | 29.9832 |
| −9 | +24.72 | **+0.69** | +147.3 | 9715.1 | 804.8 | 0.8727 | 815.0 | 29.9826 |
| **−10** | **+61.56** | **+37.82** | +201.4 | **8992.7** | **553.3** | **0.8160** | **559.9** | 30.0148 |

**The re-solve spends clearance, because the merit cannot see it.** At −8° the
floor goes +23.34 → +2.32 mm and at −9° +42.25 → +0.69 mm: the solver, free to
move the standoff and the front end, walks the design back toward the beam it
was swung away from. That is the S4b earned rule showing up again — *a
constraint is a wall, not a merit term*, and this one is not yet a wall. It is
the reason the delivered design is the −10° point, where 57 mm of raw margin
survives as 38 mm, and it is the first thing a follow-on should fix.

At −10° the re-solve buys back **54 % of the blur** (1199.9 → 553.3 µm) and
**36 % of the breathing** (1.2675 → 0.8160 %) and takes the wavefront **13.6 %
below the parent's**.

---

## 7. The cleared design

**`afocal4_clear_343mm.in`** — the committed 343 mm family-2 design with a −10°
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
| traced M (exit beam) | 29.7326 | 29.7879 | 1.002 |
| exit beam diameter (mm) | 33.633 | 33.571 | 0.998 |
| collimation (µrad) | 1477.1 | 1265.5 | 0.857 |
| chief AOI on the field mirror (deg) | 2.11 | **10.15** | 4.806 |
| **max chief AOI, any mirror (deg)** | **12.84** | **10.86** | **0.846** |
| **union body-in-beam floor (mm)** | **−79.89** | **+37.82** | — |
| rays lost over the field box | 0 | 0 | — |

**The customer interface is held**: 30.015× at the box centre, a 33.57 mm exit
beam against the committed 33.63, and the collimation is 14 % *better*. The
interface plane is carried through the swing by the same rigid motion as the
rest of the train, so its pose relative to the exit chief is unchanged.

**The AOI reads better, not worse**, which is the opposite of the obvious
expectation. The field mirror goes from 2.1° to 10.1° — but the design's *worst*
worked mirror improves from 12.84° to 10.86°, because the re-solve moved the
front end. Everything stays under the design drivers' standing 15° rule.

**What it costs, in one line:** the fourth mirror's pupil control. Blur and
wander go up 3.5×, breathing 6.6×. The wavefront and the interface do not pay.

### 7a. And it packages itself

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

*(Two ratios, named apart on purpose. The packaging stage's headline "1.81×" is
the DEEPEST OPTIC over the M1–M2 spacing; the overhang over the same spacing is
0.81× on the same deck. Quoting one under the other's name makes an improvement
look three times bigger than it is. The radius row is the girth of the structure
**behind** the primary — the whole-train envelope is the primary's own 0.500 m on
every deck and cannot tell them apart.)*

**The overhang the packaging stage spent four ~300 mm 45° flats to remove, the
swing takes from 0.81× to 0.24× of the front span with none** — deepest optic
1.81× → 1.24×, against Path A's 0.86×. It does not reach Path A's *negative*
overhang. What it does that Path A does not:

* the back-end **girth shrinks**, 0.186 → 0.150 m, where the four folds nearly
  **tripled** it to 0.435 m;
* the back focal path shortens by **0.62 m** (Path A leaves it untouched — a fold
  is an isometry);
* and it opens **no** polarization budget, where four 45° reflections in series
  is one the packaging study explicitly did not open.

**Path A does not close on the cleared deck, and does not need to.** The route
was re-searched over all four of its stated quantities on the cleared deck's own
geometry — **96 routes tried, 15 satisfied both the route algebra and the
plane-intersection bound, and every one of those lost rays** (best: −72.74 mm and
592 rays). The reason is arithmetic: the four-fold trombone needs
`x_step + return + m3_gap + the next spacing` of leg to work in, and the swing
has already shortened the feed leg and moved the field mirror 600 mm nearer.
**The route runs out of leg because the depth it was invented to remove is
mostly gone.**

**And the rest of `afocal4_pack` passes on the cleared deck**, so this is not a
design that trades one buildability clause for another:

| clause | cleared deck |
|---|---|
| last powered mirror behind M1 | **+774 mm** (minimum 500) ✔ |
| fold daylight on the exit leg | **27.1 mm** (margin 15) ✔ |
| instrument volume | 233 mm off axis; **largest that fits, Ø421 mm** against the stated Ø300 ✔ |
| union body-in-beam | **+37.8 mm** ✔ |

---

## 8. The standing gate — `afocal4_union`

The brief's fifth deliverable: promote the packaging stage's union body-in-beam
measurement into the afocal4 gate set as a **standing** check.

* **`../afocal4_union.m`** (new, beside `afocal4_pack`) is that gate. It traces
  the deck over the field set, builds every element's union footprint as a
  **convex hull** (never a centred disk — a disk fills in the middle of a walking
  footprint and invents a 107 mm interference that is the model's, not the
  design's), grows it by a **declared** allowance, and measures every
  (leg, body) pair by exact plane-crossing and exact ternary search. Both
  measures are **sampling-free**, which is the only independent check that a fold
  really is an isometry.
* **`../afocal4_pack.m`** now runs it as part 4 and folds the verdict into `K.ok`.
  The change is additive: `'union',false` reproduces the previous verdict exactly,
  and the three sub-flags `tAfocal4` asserts (`ok_station`, `fold_pick.gap`,
  `instr.dia_max`) are untouched.
* **Leg-versus-leg is deliberately not in it.** Light passes through light, and
  on a wide-field system different fields' beams genuinely cross, so a leg-leg
  zero is not an interference. `pack_clear` reports it; the gate does not,
  because the gate's verdict has to mean one thing.

**Non-vacuity.** A gate nobody can fail is not a gate, so both halves are
asserted rather than left to inspection: at the same declared allowance it must
**fail on the committed 343 mm deck** — the design that actually shipped — and
**pass on the cleared one**. `afocal4_clearing` section 6b runs exactly that.

---

## 9. Model choices that are load-bearing

| choice | why |
|---|---|
| bodies are **hulls**, not disks | a centred disk of the union's max radius fills exactly where the feed beam passes, and reports a 107 mm interference belonging to the model |
| clearances over the **field box** | the whole defect lives in the difference: 10.8 mm of daylight per field, 79.9 mm inside the beam over nine |
| the allowance is **declared and printed** | 1.15 × footprint + 15 mm, and every table also carries bare lit glass (1.00 ×, 0 mm) so an interference that survives is the design's |
| both distance measures **sampling-free** | otherwise a fold re-samples the same geometry at a different phase and an isometry appears to move something |
| the tilt pivots on the **chief hit point** | so the chief path is preserved exactly (4.45e-16 m) and the tilt's cost is a clean aberration term, not a mis-alignment |
| the walk fit carries an **intercept** | forcing it through the origin makes a tilted design report a meaningless "walk" that is silently absorbing the offset — the offset *is* the remedy and has to be a separate number |
| `iface` recovered from **`zElt`**, not from vertex geometry | the builder poses the interface plane on the traced chief, so the last mirror's vertex is 359 mm from the interface vertex on a deck whose standoff is 343; the vertex reading rebuilds a deck that is not the committed one |

---

## 10. Files

| file | what |
|---|---|
| `afocal4_clearing.m` | the driver: defect → leverage 1 → leverage 2 → leverage 3 → nulls → packaging → the row → leverage 4 |
| `clear_law.m` | the field-walk ratio law, measured: walk, offset, the ratio the field box demands, and the `M × iface` pin |
| `clear_scan.m` | the station scan and the standoff scan — the brief's first deliverable |
| `clear_fold.m` | leverage 1, retired: the two-sided squeeze about the critical station |
| `clear_tilt.m` | the extraction tilt as an exact rigid motion of the traced chief |
| `clear_build.m` | `afocal4_build` + `clear_tilt`; bit-identical to `afocal4_build` at zero tilt |
| `clear_solve.m` | `afocal4_solve`'s outer loop on `clear_build`, with the tilt as an optional DOF |
| `clear_price.m` | the exchange rate: clearance won against pupil quality paid, raw and re-solved |
| `../afocal4_union.m` | **the standing gate** (new, in the afocal4 gate set) |
| `run_clear.m` | one-line `matlab -batch` wrapper |
| `../../../tests/tAfocal4Clear.m` | the regression class (8 tests, `SUITE_FREEFORM`) |
| `afocal4_clear_343mm.in` | **the cleared design** (a copy of the −10° point below, under the name the record quotes) |
| `afocal4_clear_t-80.in` / `t-90.in` / `t-100.in` | the three re-solved price-curve points |
| `afocal4_clear_scan.png` | demand vs available, both axes |
| `afocal4_clear_price.png` | what the tilt buys and what it costs |
| `afocal4_clear_layouts.png` | the three layouts on one scale |
| `afocal4_clearing.mat` | every struct the run produced |

Run:

```matlab
run('~/dev/MACOS_res_dev/mmacos/mmacos_setup.m')
addpath('.../challenges/afocal4/clearing')
R = afocal4_clearing();                          % everything
R = afocal4_clearing('sections',[0 1 2]);        % the defect and the two retirements
R = afocal4_clearing('sections',3:6, 'resolve','load', ...
                     'resolved_mat',{'...m10.mat'});   % off a finished solve
K = afocal4_union('x.in', 'fields',P.Fsolve);    % the gate, on any deck
L = clear_law('x.in','fields',P.Fsolve,'leg',2,'elt',4,'M',30);
```

Model size 256, one MATLAB process per model size,
`MACOS_HOME=~/dev/macos/macos_f90`.

---

## 11. Leverage 4 — a fifth mirror, priced rather than built

The brief asks for the fifth mirror in parallel if the station scan shows the
four-mirror topology is fundamentally pinched. It does, so here is the number it
has to beat, stated in the law's own terms rather than by building a design this
stage did not have time to solve.

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

**Which is exactly what the −10° tilt already supplies (201.4 mm measured).** So
a fifth mirror's case does *not* rest on being able to clear the beam — the
fourth one can, by being swung. It rests on clearing it **without spending the
pupil control**, i.e. on beating

> blur 553.3 µm, breathing 0.8160 %, wander 559.9 µm, at 8993 nm of wavefront
> and a +37.8 mm union floor.

Three architectures could do that, and the law says which lever each one pulls:

1. **A powered extraction mirror near the intermediate image** — supplies the
   offset with an element designed for it instead of by swinging the one element
   that sits at the field conjugate. Same lever, better place to pull it.
2. **A relay to a second intermediate image** — the `relay` form of
   `afocal4_close`, eliminated in S3 on pupil grounds *with four mirrors*. With
   five, the collimator stops living inside the M2 → field-mirror cone at all,
   so the ratio law never applies to it.
3. **A fifth mirror that holds the pupil station**, freeing the field mirror's
   power — which the four-mirror closure spends *entirely* on that one condition
   — to put the collimator at the internal chief crossing, where its union
   footprint collapses to the beam radius. This is the one that attacks the
   *pinned* term rather than adding an offset, and it is the only one of the
   three that could return the pupil metrics to the committed row's values.

---

## 12. Open, and for Dave

1. **The clearance is not yet a wall, and the re-solve spends it.** At −8° and
   −9° the solver walks a +23 and +42 mm margin down to +2.3 and +0.7 mm,
   because `afocal4_score` cannot see the clearance and `afocal4_build` does not
   refuse on it. The fix is the S4b pattern exactly: put the union floor into
   `afocal4_build` as a **wall** beside the `m3_behind_min` one. It is not done
   here because a wall needs a compliant seed to be a wall and not a cage, and
   seeding it properly is its own slice.
2. **The delivered tilt is −10° because the clearance saturates there, not
   because the price is right.** The exchange rate is the deliverable. If the
   pupil budget is real, −8° at +2.3 mm of margin (with the wall above, more)
   costs 741.7 µm of blur instead of 553.3 — but note the wavefront is *worse*
   there too, so −10° is not a compromise on this design, it is simply better.
3. **35 % of the pupil blur was lying around unclaimed.** A *positive* 4° tilt
   takes the blur from 157.0 to 102.6 µm with no re-solve at all. The committed
   design is not at its own pupil optimum, and the reason is visible in the
   merit: a wavefront term 130× off target owns the log-domain sum of squares.
   Worth a look independently of clearance.

   > **CORRECTED 2026-08-30 (the wall stage, `../RESULTS.md` § C.7a).** The
   > finding stands and the optimum is sharper than this — **101.3 µm at +5°**,
   > 35.5 %, on a 1° grid — but the *reason* given here is wrong. Read off the
   > residual vector, the wavefront block is **78 %** of the merit, not ~97 %:
   > `afocal4_score` divides the per-field wavefront residuals by `sqrt(K)`, so
   > nine fields contribute one term's worth. The merit actually **prefers**
   > the +5° point (29.40 against 30.22). The solve did not find it because an
   > extraction tilt **was never in its DOF set** (`{conic, standoff, front}`),
   > not because the merit was blind. And the move is not free either: what
   > pays is magnification **breathing**, 0.124 → 0.795 %, i.e. the one pupil
   > target the committed design meets.
4. **The re-solves are budget-limited, not converged** (exitflag 0 at 427
   evaluations each). They also ran on the study's default 3e-3 *forward*
   difference — the setting S4c measured as reading the gradient 17 % low on a
   merit smooth to 1e-5. A central-difference polish is the known next step and
   would move these numbers, probably in the cleared design's favour.
5. **Nothing here is pushed and nothing committed was overwritten.** Two files
   beside the stage are additive: `../afocal4_union.m` (new) and one clause in
   `../afocal4_pack.m`. `tAfocal4` stays 8/8 with that clause in place, and
   `tAfocal4Clear` (new, 8 tests, `SUITE_FREEFORM`) gates the stage.
6. **Slide 13 says "redesign queued".** It can now say what the redesign is and
   what it costs — but that is outward-facing and waits on sign-off
   (`doc/STYLE_REPORTS.md` §5).
