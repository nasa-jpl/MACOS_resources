# PACKET — rodgers2: the 30× afocal TMA benchmark

> **The four-mirror answer is delivered.**  The form study that chose it is
> `../examples/afocal4/FORM_STUDY.md`; the joint solve, the answer ladder, the
> interface-standoff trade curve and the Mersenne verdict are
> `../examples/afocal4/RESULTS.md`.  Headline: a convex field mirror at the
> intermediate image **solves the interface pupil** — on axis, every pupil target
> inside, including the wander target no first-order form could reach — and it is
> paid for in image quality, at an exchange rate set by the interface standoff.
> This packet remains the record of **his** three-mirror benchmark under our metrics.

J.M. Rodgers supplied four CODE V `.seq` decks on 2026-08-02
(`~/dev/MACOS_sandbox/Design/Rodgers2/`, plus
`260802-AfocalTMA_Offsetfield-jmr.pptx`): a coaxial three-mirror **afocal**
telescope, EPD 1000 mm, λ = 1 µm, 30× angular magnification, over a
0.5°×0.5° field box offset +0.6° in Y, delivering its collimated beam to a
tilted **coldstop** = the interface pupil.

His four-slide ladder is on-axis → offset-unoptimised → conics re-solved →
+ M2/M3 tilt/dec, with in-box max RMS WFE **15 / 430 / 160 / 119 nm** and
in-box averages **4.0 / 154 / 93 / 48 nm**.  His slides also assert that
*"with 3 mirrors the pupil quality is not very good; a 4th mirror is needed
for pupil control."*

**His deck contains no pupil metric.**  The only pupil-adjacent numbers in
it are the coldstop's DAR tilt (0 / 4.289 / 3.577 / −0.356°) and a
magnification that slips from 30×.  The pupil-quality **definition** in
this packet is therefore **ours** — the cone-convergence model Dave ruled
on 2026-08-02, implemented as `pupil_map` — and any material that goes back
to Mike must say so.

Scope of this packet: **S1 and S2 of `PLAN_AFOCAL4.md`** — the measurement
infrastructure and the scored 3-mirror baseline.  The 4-mirror form study
(S3) is a separate gate.

---

## 0. Verdict up front

1. **The transcription is exact, and the decode is a 10⁹ margin.**  His
   "recenter" coordinate break places the coldstop vertex on the *traced*
   exit chief ray to **2 × 10⁻⁷ mm**.  Under the opposite ADE sense the
   same arithmetic misses it by **211–247 mm** on a 33 mm beam.  The
   recenter ADE *is* the exit chief angle to five decimals.

2. **His CODE V afocal field-map RMS is our rung 2** — piston and per-field
   tip/tilt removed, and nothing more.  At that rung, scored on a uniform
   grid over his box, our numbers reproduce his:
   **max 0.952–1.015×, average 0.994–1.042×, on all four variants.**
   The ≤1.15× gate passes with room.

3. **His in-box averages are AREA averages**, not means of his nine solve
   points.  His 3×3 set is one-third corners; scored on it the average
   ratio is 2.11× on S1, and on a uniform 9×9 grid over the same box it is
   1.04×.  The maxima are identical either way (both sets contain the
   corners).

4. **The pupil claim is now quantitative, and it is four separate
   statements.**  Section 3 is the baseline table.  The headline: his 30×
   magnification measures **28.686×** at the box centre on the unoptimised
   offset variant — his slides say 28.7× — and is restored to **30.0015×**
   by the conic re-solve.  The pupil-image **blur** is 153 µm rms on the
   perfect on-axis system and 775–802 µm on the offset ones, against a
   2.7 µm diffraction floor; the **wander** at his placed coldstop is
   160 µm → 893–926 µm.  On a 33 mm exit pupil that is 0.5% → 2.7% of the
   pupil diameter.

5. **CALIB runs on an afocal deck, converges, and wrecks the design.**  Its
   merit there is 10³–10⁴× the wavefront error, because it is the OPL
   spread on a coldstop his design deliberately tilts off the chief.
   Section 4.

6. **`Element= Return` must not be used for the interface flat.**  It
   reverses the ray directions.  Section 4.0.

7. **The magnification breathing is real, and it has two frames** (added
   2026-08-03, §4).  A footprint read on the fixed tilted coldstop carries a
   1/cos(incidence) stretch as each field's exit chief swings up to 13.6° off
   that plane; measured normal to each field's own chief the best 3-mirror
   variant breathes **±3.63%** against ±3.84% as read.  The headline
   survives.  On the *perfect* on-axis variant, though, 35% of the apparent
   spread was the frame, and the ±0.78% that remains is the cleanest
   evidence here that the deficits are pupil-imaging aberrations rather than
   field-dependent design error.

---

## 1. The conversion audit — `.seq` ↔ `.in`, surface by surface

Transcription, not parsing (`rodgers2_seq.m` → `rodgers2_deck.m`); no `.seq`
reader exists and none is to be written.

### CODE V `.seq` — surfaces (thickness = to the NEXT surface; z = global vertex station, M1 at z = 0)

| # | label | radius (mm) | thickness (mm) | vertex z (mm) | notes |
|---|---|---|---|---|---|
| SO | object | ∞ | 367 915 496 283.038 | −3.679e11 | infinity proxy |
| S1 | dummy | ∞ | 1100.0 | −1150 | no optical effect |
| S2 | "tilt" | ∞ | 0 | −50 | placeholder, **no decenter in any file** |
| S3 | `STO` | ∞ | 50.0 | **−50** | the stop is 50 mm AHEAD of M1; `CIR EDG 0.1` |
| S4 | m1 | **−2500.0** | −1049.239293684764 | **0** | `REFL`, `CON K −1` (parabola), `CIR HOL` semi 130.0 |
| S5 | m2 | per variant | +1049.239293684764 | **−1049.239294** | `REFL`, `CON` |
| S6 | "thru" | ∞ | 350.0 | 0 | dummy |
| S7 | dummy | ∞ | per variant | +350 | **the intermediate image** |
| S8 | m3 | per variant | 0 | +640.486221 (S1: +640.415896) | `REFL`, `CON` |
| S9 | "recenter" | ∞ | per variant | = m3 | **no DAR** ⇒ a coordinate break |
| S10 | "coldstop" | ∞ | 0 | — | `DAR` + `ADE` per variant |
| S11 | dummy | ∞ | −1000.0 | — | on the recenter axis |
| SI | image | ∞ | 0 | — | `AFI −1000` |

**Why S7 exists.**  With f₁ = 1250 mm and the M1→M2 spacing 1049.239 mm the
marginal ray is at 80.30 mm on M2 and crosses the axis 1399 mm later — i.e.
at z = +350.0, exactly S7.  The remaining 290.4 mm to M3 is M3's focal
length, so M3 recollimates and the exit semi-beam is 16.67 mm ⇒ M = 30.
Verified numerically by the first-order gate (§2).

### MACOS `.in` — source

| keyword | value |
|---|---|
| `ChfRayDir` | (0, sin·YAN, cos·YAN); YAN = 0.0° (S1) or 0.6° (S2–S4) |
| `ChfRayPos` | `ApStop − stand·ChfRayDir` |
| `zSource` | 1e22 (collimated) |
| `Aperture` / `Obscratn` | 1.0 m / 0.0 |
| `GridType` / `nGridpts` | Circular / 41 → 1185 launched, 1104 pass, 81 obscured by the hole |
| `ApStop` | **(0, 0, −0.050) m** — his STO, 50 mm ahead of M1 |
| `Wavelen` / units | 1.0e−06 m / BaseUnits = WaveUnits = m |

The 50 mm stop offset is carried, not approximated away: at 0.6° it walks
the beam 0.52 mm across M1, and that walk is precisely what the
pupil-anchoring section of `pupil_map` exists to handle.

### MACOS `.in` — elements

| iElt | Name | Element / Surface | `KrElt` (m) | `KcElt` | `VptElt` (m) | `psiElt` | obscuration |
|---|---|---|---|---|---|---|---|
| 1 | M1 | Reflector / Conic | −2.5 | −1.0 | (0,0,0) | (0,0,−1) | `nObs=1 Circle r=0.130` |
| 2 | M2 | Reflector / Conic | −\|R₂\|/1000 | K₂ | (0, YDE₂, −1.049239294) | (0, −sin ADE₂, −cos ADE₂) | none |
| 3 | M3 | Reflector / Conic | −\|R₃\|/1000 | K₃ | (0, YDE₃, +0.640486) | (0, −sin ADE₃, −cos ADE₃) | none |
| 4 | ColdStop | **Reference** / Flat | −1e22 | 0 | see below | −ẑ_cs | none |

Per variant:

| | R₂ (mm) | K₁ | K₂ | R₃ (mm) | K₃ | S7→M3 (mm) | recenter t / YDE / ADE | coldstop ADE |
|---|---|---|---|---|---|---|---|---|
| S1 on-axis | −468.7799802942544 | −1 | −1.782495505768868 | −580.8105879437068 | −1.001753914266608 | 290.4158962406167 | −344.173 / 0 / 0 | 0 |
| S2 offset | −468.7799802942544 | −1 | −1.782495505768868 | −580.8105879437068 | −1.001753914266608 | 290.486221424707 | −365.779766 / 110.5639166839551 / 17.67063712218155 | 4.289 |
| S3 newconics | −468.1589934385201 | −1 | −1.778931782803361 | −558.9832733211452 | −0.9569802075823867 | 290.486221424707 | −351.03178 / 110.9415904849413 / 18.49984656262884 | 3.576783 |
| S4 tilt/dec | −468.2687654028678 | −1 | −1.777290021452283 | −564.6825868509671 | −0.9820445490283405 | 290.486221424707 | −355.257136 / 130.030367037749 / 21.99503063542316 | −0.355818 |

S4 additionally carries `DAR` on M2 (YDE 1.760802316111543 mm, ADE
0.5114423167490506°) and M3 (YDE 36.8890230448649 mm, ADE
4.023132834823657°), transcribed verbatim in CODE V sign.

### The recenter coordinate break, consumed

The recenter surface has no `DAR`, so its decenter and tilt persist:

```
coldstop vertex = (0, YDE_rec, z_M3) + t_rec · ẑ_rec ,   ẑ_rec = R_x(−ADE_rec)·ẑ
coldstop normal = ẑ_cs = R_x(−(ADE_rec + ADE_DAR))·ẑ ,   psiElt = −ẑ_cs
```

using the rodgers1 ADE decode (his ADE = −(our α), α = `atan2d(psi_y,−psi_z)`).

### Decoded here, and new since rodgers1: `CUM` / `THM`

Every reflective surface carries `CUM <c>; THM <t>`.  These are **mechanical
substrate data** — the curvature of the mounting/back surface and its
thickness.  M1 carries `CUM −0.0004` (= 1/−2500: the back face is
concentric with the optical surface) and `THM 75.0`; M2 `THM 40.136`; M3
`THM 13.379` — monotone with the beam footprint, as substrate thicknesses
are.  No prescription content; not transcribed.  (rodgers1's files repeated
one byte-identical `THM` on every surface and were flagged undecoded; these
files resolve what the datum is.)

### Still undecoded, flagged for Mike

`CIR EDG 0.1` on the stop surface.  The stop's optical semi-diameter is set
by `EPD` (500 mm), so this is a drawing/edge datum, but the qualifier is not
confirmed and nothing is assumed from it.

---

## 2. The first-order gate, and the ADE decode — witness #5

`rodgers2()` section 0, `rodgers2_deck(..., 'verify', true)`.

| variant | vertex-to-chief miss | exit beam (mm) | M (beam) | M (angular) | exit chief (deg) | coldstop tilt off chief (deg) |
|---|---|---|---|---|---|---|
| S1 on-axis | 0.00e+00 mm | 33.257 | 30.0693 | 30.0000 | −0.00000 | 0.00000 |
| S2 offset | 1.02e−07 mm | 35.097 | 28.4928 | 28.3794 | 17.67064 | 4.28900 |
| S3 newconics | 1.77e−07 mm | 33.520 | 29.8325 | 29.6974 | 18.49985 | 3.57678 |
| S4 tilt/dec | 1.47e−08 mm | 33.280 | 30.0477 | 30.0264 | 21.99503 | −0.35582 |

Three things this settles.

**The ADE sign.**  rodgers1 decoded it over 16 sign combinations with a 30×
margin.  Here the same convention places the coldstop vertex on the traced
exit chief to **10⁻⁷ mm**, and the wrong sign misses by **211–247 mm**.
That is a 10⁹ separation, and it is a *prediction* — nothing in the
transcription is fitted to the trace.

**The recenter ADE is the exit chief angle**, to five decimals
(17.67064 vs 17.67063712, 18.49985 vs 18.49984656, 21.99503 vs
21.99503064).  So the surface does what its label says.

**Therefore the coldstop DAR tilt is exactly the coldstop's tilt away from
normal-to-the-chief** — 4.289° / 3.577° / −0.356°, recovered to five
decimals as the difference column above.  That is the whole of the
pupil-adjacent evidence his deck carries, and it is now anchored to a
physical quantity rather than to a keyword.

---

## 3. The afocal ladder, and which rung his field map is

The reference for an afocal exit wavefront is a **plane** normal to the exit
chief, not a sphere (`afocal_refs` / `afocal_rungs`, design/src).  Three
rungs of increasing reference freedom:

1. **piston only** — everything the system did, including its pointing;
2. **+ tip/tilt** — the removed term is a BORESIGHT, not an error;
3. **+ power** — the removed term is the residual DIVERGENCE, i.e. the
   output is not quite collimated.  Reported in nm *and* µrad, because a
   collimation error is naturally an angle.

There is no focus rung: a focal system's detector can slide and buy back
defocus, an afocal system's collimation is a deliverable.

### The comparison (nm; ratios on a uniform 9×9 grid over his box)

| rung | | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| **his reported** | max / avg | 15 / 4.0 | 430 / 154 | 160 / 93 | 119 / 48 |
| 1 piston | max / avg | 26.47 / 7.40 | 584.54 / 222.12 | 358.68 / 154.89 | 247.48 / 103.53 |
| | ratio | 1.765 / 1.849 | 1.359 / 1.442 | 2.242 / 1.665 | 2.080 / 2.157 |
| **2 + tip/tilt** | max / avg | **14.41 / 4.17** | **436.36 / 155.46** | **152.28 / 92.47** | **114.10 / 48.39** |
| | **ratio** | **0.961 / 1.042** | **1.015 / 1.009** | **0.952 / 0.994** | **0.959 / 1.008** |
| 3 + power | max / avg | 14.41 / 4.13 | 425.16 / 149.90 | 116.05 / 83.69 | 92.67 / 39.66 |
| | ratio | 0.961 / 1.034 | 0.989 / 0.973 | 0.725 / 0.900 | 0.779 / 0.826 |

**The matched rung is 2**, chosen by minimising the worst-variant log
distance from 1.0 across all four decks — not by picking the best-looking
row.  Band on the max **0.952 … 1.015×**, on the average
**0.994 … 1.042×**.  The gate (≤1.15×) passes on all four.

Rung 1 is 1.4–2.2× high and rung 3 is 0.73–0.99×, so the result is
**bracketed, not fitted**: his convention removes tip/tilt and does not
remove power.  Quote the rung on every number that leaves this study.

### The sampling finding

His 3×3 set and a uniform 9×9 grid over the same box give **identical
maxima** — both contain the four corners, and a corner is always the worst
field.  The averages differ substantially: his 9-point set is one-third
corners, and scored on it the S1 average ratio is 2.11× where the uniform
grid reads 1.04×.  **His stated in-box averages are area averages over the
box.**  (rodgers1 found the same class of effect from his quincunx
sampling, at 8%; here it is a factor of two, because 4 of 9 points are
corners.)

---

## 4. The baseline pupil table

The definition is ours.  `pupil_map`, the cone-convergence model: an
entrance-pupil point is imaged to the exit pupil by the **cone of rays
through it, one per field angle**, so the imaging bundle's aperture IS the
used field set.  Cones anchored on the **M1 surface**; rim partial cones
kept, not repaired; the curved-object reference carried explicitly.
Anchoring residual 0.002–0.010 µm, i.e. 10⁻³ of the blur being measured.

Cone aperture = his 3×3 field set.  Lengths µm unless marked.

### (1) blur, (3) transverse map

| variant | blur rms | blur max | M (cone) | anamorph | distortion | λ/(2·NA_field) |
|---|---|---|---|---|---|---|
| S1 on-axis | 152.9 | 354.0 | 29.5201 | 1.00000 | 0.001% of R | 2.73 |
| S2 offset | 801.6 | 1707.2 | 28.2516 | 1.02162 | 0.061% of R | 2.81 |
| S3 newconics | 774.8 | 1673.8 | 29.5260 | 1.02054 | 0.048% of R | 2.68 |
| S4 tilt/dec | 468.6 | 1212.7 | 29.5184 | 1.00187 | 0.052% of R | 2.69 |

### magnification, (2) convergence surface, (4) wander

| variant | M (box centre) | M range over the field | surface tilt (mrad) | surface defocus (mm) | wander rms | after re-tuning the plane |
|---|---|---|---|---|---|---|
| S1 on-axis | **30.0000** | 29.28 – 30.00 | 0.0000 | −0.0274 | 160.4 | 154.6 |
| S2 offset | **28.6863** | 27.19 – 29.28 | 4.7621 | −0.0249 | 925.8 | 804.0 |
| S3 newconics | **30.0015** | 28.36 – 30.66 | 4.1167 | −0.0219 | 892.7 | 777.3 |
| S4 tilt/dec | **29.9988** | 28.34 – 30.64 | 1.9139 | −0.0169 | 557.0 | 471.5 |

**His 28.7× is here.**  The box-centre pupil magnification on the
unoptimised offset variant measures **28.686×**; his slides report 28.7×.
The conic re-solve restores it to 30.0015× and the tilt/dec variant holds
30.0 — exactly the story his ladder tells, now with a number attached to
the mechanism.

**And the magnification BREATHES.**  Even on the corrected variants it runs
28.36 – 30.66× across the 0.5° box: ±3.8%.  A restored centre magnification
is not a restored pupil.

### REFINEMENT (2026-08-03) — the breathing in two frames

The gate review asked whether that ±3.8% is a pupil-imaging defect or a
frame term, because the column above is a footprint read on the **placed
coldstop** — a plane held fixed while each field's exit chief swings by
M·θ, reaching **10.5–13.6°** of incidence at the box diagonal.  An oblique
read stretches the footprint by 1/cos(incidence) in the plane of incidence,
which is ~1.6% of areal stretch, i.e. ~0.8% of apparent magnification.
`pupil_map` now measures the per-field magnification in **both** frames:
`.mag_per_field` on the deck's placed plane and `.mag_per_field_chief` in
the plane through the **same station** normal to **that field's own** exit
chief.  The claim above is refined, not retracted; the numbers that
produced it are unchanged and stand in the table.

| variant | placed: centre / range / ± | chief-normal: centre / range / ± | incidence |
|---|---|---|---|
| S1 on-axis | 30.0000 / 29.28–30.00 / **±1.200%** | 30.0000 / 29.53–30.00 / **±0.783%** | 0.0–10.5° |
| S2 offset | 28.6863 / 27.19–29.28 / ±3.652% | 28.6848 / 27.27–29.54 / ±3.966% | 2.6–13.6° |
| S3 newconics | 30.0015 / 28.36–30.66 / ±3.828% | 30.0002 / 28.48–30.92 / ±4.071% | 3.6–13.5° |
| S4 tilt/dec | 29.9988 / 28.34–30.64 / **±3.837%** | 29.9988 / 28.59–30.77 / **±3.634%** | 0.4–10.6° |

**The frame term is real, it is exactly the 1/cos, and it does not explain
the headline.**  The ratio between the two columns matches
1/√(cos·incidence) to **7e−7** on the two variants whose box-centre chief is
normal to the coldstop (S1, S4) and to 1.5e−3 on the two that are tilted off
it (S2, S3, where the exit-frame projection adds a second small term).  So
the mechanism is identified, not fitted.  But on the corrected offset
variants it moves the answer by a few percent **of itself**: the S4
breathing is **±3.63%** chief-normal against ±3.84% as read, and S3 goes the
other way, **±4.07%** against ±3.83%.  His coldstop tilt happens to *mask*
part of S3's real breathing.

**Where it does matter is the perfect design.**  On S1 — a coaxial afocal
with the box on axis, where there is no design error to find — **35% of the
apparent 29.28–30.00 spread was the frame**, and the true pupil-imaging
breathing is 29.53–30.00, ±0.78%.  That residual is genuine pupil
distortion, and its survival on the *perfect* variant is the single
strongest piece of evidence in this packet for Mike's assertion: the
deficits are pupil-IMAGING aberrations, present with the image quality at
15 nm.

**Box-centre magnification is untouched** (28.6863 → 28.6848 on S2), because
the coldstop is tilted to the box-centre chief by construction and a tilt of
4.289° is only 1.4e−3 of areal stretch.  **His 28.7× stands.**

Both numbers stay in the record and both are reported.  The placed-plane
column is what a fixed coldstop actually samples — the instrument feels it,
and it is not fictitious.  The chief-normal column is the pupil-imaging
defect, and it is the one an S3 target is written against.  Gated by
`tPupilMap/test_chief_normal_mag_carries_no_obliquity`: tilting the
evaluation plane 10° must not move the chief-normal number at all, must move
the placed one, and must break the four-corner symmetry of the placed read
while leaving the chief-normal read rotationally symmetric on a coaxial
deck.

### The two references for the convergence surface

| variant | β / m² vs the ideal image of M1's sag | residual after it | vs the FLAT placed plane, rms | P-V |
|---|---|---|---|---|
| S1 on-axis | **−0.9899** | 0.10 µm | 279.6 µm | 0.0502 mm |
| S2 offset | −0.8244 | 6.30 µm | 3163.5 µm | 1.8211 mm |
| S3 newconics | −0.7896 | 5.50 µm | 2892.1 µm | 1.7354 mm |
| S4 tilt/dec | −0.6100 | 1.72 µm | 1941.2 µm | 1.8884 mm |

M1's own sag is **44.22 mm P-V** over the used annulus, which a perfect
pupil imager delivers to the exit at longitudinal magnification
m² = 1.147e−3, i.e. **50.7 µm** of curvature that is **not** an aberration.  On the perfect on-axis
system β/m² = −0.99: the sag is imaged exactly as it should be, and the
residual after removing it is **0.10 µm**.  On the offset variants β/m²
falls to −0.61…−0.82 — the sag is *not* faithfully imaged, and that
shortfall is real pupil-imaging aberration.  Charging a fast primary's
correctly-imaged sag as "pupil curvature" would have inflated every one of
these numbers by ~50 µm; that is what this reference is for.

(Note that m² is measured, not assumed: 1.147e−3 against a nominal
1/30² = 1.111e−3, the 3.2% gap being the same cone-vs-centre magnification
difference as the 29.52 vs 30.00 column above.)

Against the **flat** coldstop the convergence surface sits 1.7–1.9 mm P-V
away on the offset variants (0.05 mm on-axis).  That is the operational
"a flat coldstop cannot be conjugate everywhere" number.

### His coldstop tuning, tested

| variant | his DAR tilt | further tilt our wander fit wants | further shift | wander improvement |
|---|---|---|---|---|
| S1 on-axis | 0.0000° | 0.0000° | +0.28 mm | 1.038× |
| S2 offset | 4.2890° | 2.9294° | +3.09 mm | 1.151× |
| S3 newconics | 3.5768° | 3.0773° | +2.82 mm | 1.149× |
| S4 tilt/dec | −0.3558° | 3.6334° | +1.84 mm | 1.181× |

His coldstop tilt is **not** wander-optimal: a plane fit to minimise pupil
wander wants another ~3° and 2–3 mm of shift on every offset variant, worth
15–18% of the wander.  So the DAR tilt was tuned against some other
criterion, not against pupil-footprint stability.  Worth one line from Mike.

### The engine's own pupil surface

`macos.pupil_quality` (the XPS field-differential crossing) runs on these
afocal decks and returns astigmatism −0.0003 / −0.0281 / −0.0170 /
−0.0062 mm.  It is the **two-ray limit** of the cone model, and
`tPupilMap` gates the identity between them to 2e−4 on defocus and 1e−4 on
astigmatism — with one decode attached: **XPS's differential is the
TANGENTIAL one.**  Its `th = xGrid * 5e-6` is a rotation *vector* about x,
which tilts the chief in Y.  An x-field differential returns the sagittal
surface, and on an astigmatic pupil the two have opposite-sign astigmatism.
Match the axis or the comparison is meaningless.

### What the table says, in words

The 3-mirror afocal delivers, at its interface pupil:

* a pupil image blurred by **0.5% of the pupil diameter on-axis and 2.3–2.7%
  off-axis**, against a 2.7 µm diffraction floor — i.e. this is geometry,
  not diffraction;
* a magnification that is correct at the box centre after re-solving but
  **breathes ±3.6% across the field** measured normal to each field's own
  exit chief (±3.8% as read on the placed coldstop, which adds the plane's
  own obliquity — see the refinement above, and name the frame);
* a convergence surface **1.7–1.9 mm P-V from the flat coldstop**, of which
  only ~50 µm is the primary's correctly-imaged sag;
* **0.9 mm rms of footprint wander** at the placed plane, which no
  re-placement of that plane reduces by more than 18%.

That is the quantitative content of "with 3 mirrors the pupil quality is
not very good."  The 4-mirror targets follow from it, and are S3's business.

---

## 5. CALIB on an afocal deck — the S1d probe

`calib_afocal_probe.m`.  Short, non-gating, for the record: the sanctioned
optimisation path for this study is a MATLAB outer solve regardless
(Dave, 2026-08-02).

**5.0 The interface element must be `Element= Reference`, not `Return`.**
The plan proposed emitting the afocal terminal as a flat Return.  It must
not be: `Return` **reverses the ray directions** at that surface, so a
metric that builds its reference from the exit chief builds it backwards
and the afocal rung 1 reads 4017 µm instead of 359 nm.  The OPL itself is
unchanged (identical std to seven figures), which is exactly why this hides
from any piston-only check.  `Reference` and `FocalPlane` agree to the last
digit.

**5.1 CALIB's merit runs 1.9e3 – 2.6e4 × rung 1.**  It is the engine OPD at
`OptWFElt`, and `SUBROUTINE OPD` builds no reference — it reports the OPL
spread *on the element surface*.  On a focal deck FEX supplies the missing
reference.  Here the surface is a flat his design deliberately tilts 3.58°
off the exit chief, so the merit is dominated by (beam radius)·tan(tilt):
2.0e−3 m where the wavefront error is 1.5e−7 m.

**5.2 CALIB nevertheless loads, runs and converges** (rtn 0) on the deck.

**5.3 And the solve is destructive.**  Varying only M2's conic, CALIB moves
K from −1.7789 to **+6.9246** — a sign change — to buy 3.1% of its own
merit, and the afocal WFE goes from **152 nm to 288 µm**.  A 1900×
degradation, scored as an improvement.

**Conclusion.**  Making CALIB usable on an afocal deck needs an afocal
reference *inside the engine*: the plane analogue of what FEX supplies on a
focal deck.  That is a follow-on engine task, not part of this plan.

---

## 6. Traps recorded

Each cost a debug cycle; each is now guarded in the code that found it.

1. **A regex whose trailing `\s*` eats the newline** splices a new key onto
   the END of the `iElt=` line, and the parser never sees it.  The first
   version of the CALIB probe reported "CALIB refuses to run on an afocal
   deck" when what it had actually done was emit an unparseable `VarElt`.
   `insert_after_elt_` is line-wise now, with the reason in the comment.

2. **Loading any other Rx resets `nVarElt` / `nOptFov`.**  Taking a
   "before" score with `afocal_ladder_deck` *after* loading the Opt deck
   silently disarms CALIB, which then fails its own `nVarElt<1` pre-check.
   Order the scoring before the load.

3. **CALIB writes its optimised state back over the deck it was loaded
   from**, and that saved copy has the last probe field's chief ray baked
   in *and* — a separate save round-trip gap — **no `ApStop` line at all**.
   The Opt deck lives in a temp file; the "after" score is taken by
   rebuilding the transcription deck with the solved conic.

4. **The pupil metric's own anchoring.**  Grouping rays by source index
   across field traces inflates the measured blur by 2.9–3.6% on these
   decks (`pupil_map('anchor','index')` reproduces it deliberately).  The
   plan's ~29 µm ray-walk estimate is confirmed in magnitude; on this
   design the real blur is 5× larger, so the correction is small rather
   than dominant.  Anchor anyway — it is free.

---

## 7. Artifacts and how to reproduce

```matlab
rodgers2                       % all four sections, writes every artifact
rodgers2('sections',0:1)       % transcription + the WFE ladder only
rodgers2('map_n',15)           % denser uniform scoring grid
calib_afocal_probe             % the S1d probe (section 5)
```

| file | what |
|---|---|
| `rodgers2_seq.m` | the verbatim `.seq` transcription (data only) |
| `rodgers2_deck.m` | the `.seq` → `.in` renderer, with every convention named |
| `rodgers2.m` | the study driver |
| `calib_afocal_probe.m` | the S1d CALIB probe |
| `rodgers2_S{1..4}_*.in` | the four committed decks |
| `rodgers2_S{1..4}_*_ladder.png` | afocal-ladder field maps, three rungs each |
| `rodgers2_S{1..4}_*_pupil.png` | the four-part pupil ladder, one panel each |
| `rodgers2_results.mat` | every number in this packet |
| `rodgers2_calib_probe.mat` | section 5 |

Kernels used, all in `design/src` and gated by `tAfocalKernel` (11/11) and
`tPupilMap` (7/7): `afocal_plane_opl`, `afocal_refs`, `afocal_rungs`,
`afocal_wfe_deck`, `afocal_ladder_deck`, `afocal_score_psf`, `pupil_map`.

---

## 8. Open, and what S3 inherits

1. **One line from Mike**: what criterion set the coldstop DAR tilt?  It is
   not wander-optimal (§4).
2. **`CIR EDG 0.1`** on the stop — decode unconfirmed (§1).
3. **The 4-mirror targets** are S0's placeholders until Dave or an
   instrument spec retargets them; the working claim is "≥10× the S2
   baseline" against the table in §4.  The breathing target is written
   **chief-normal** against the S4 baseline of ±3.63% (§4 refinement) —
   quote the frame or the target means nothing.
4. **The plan's flat-Return terminal is retired** (§5.0); S3's builder must
   emit `Element= Reference`.
5. **An afocal reference inside the engine** would make CALIB usable here
   (§5) — a follow-on engine task, deliberately out of scope.

6. **S3 has since answered the "why a 4th mirror" question, and the answer
   is not the one the deck implies** — see
   `../examples/afocal4/FORM_STUDY.md`.  His 3-mirror **already closes both
   first-order conditions**: it recollimates at 30.000× and images the stop
   onto a plane **0.81 mm** from the coldstop he placed by hand.  So the 4th
   mirror repairs no first-order deficiency; what it has to buy is the pupil
   ABERRATION this packet measured in §4.  Anything that goes back to Mike
   should say so — it makes his verbal claim sharper, not weaker.
