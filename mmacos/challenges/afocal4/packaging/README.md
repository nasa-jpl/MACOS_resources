# afocal4 packaging — the 343 mm back end, measured and folded down

Origin: `macos/BRIEF_r2_packaging.md` (Dave, 2026-08-30, reviewing keysight slide 12
— *"the layout claims buildability, but the back focal length is longer than the
M1–M2 spacing. Make it compact by repeated folds, or redesign, or note it was not
driven to completion."*)

**Nothing under `challenges/afocal4/` or `challenges/rodgers2/` was modified, moved
or regenerated.** Everything here is new, in this directory, and every deck it
writes is derived from a committed one by an exact isometry.

Subject: **`../afocal4_b2long_343mm.in`** — the family-2 (image-behind-M1) winner at
the 343 mm interface standoff, which is the design the deck's slide-12 table quotes
(S4c: 10 407 nm / 157 µm / 0.124 %). The label is the paraxial standoff `P.iface`;
on the emitted deck the chief runs **372.4 mm** from the last mirror's vertex to the
interface plane, because that plane is re-posed on the traced exit chief.

Every number below is engine truth — traced rays and the engine's own element
getters. No `.in` text is parsed for geometry (the corpus-indexing lesson), and
every clearance is measured over the **whole 0.5°×0.5° field box**, not the deck's
own field.

---

## 1. The gap, in numbers (brief step 1 — delivered regardless of the rest)

Sky is at −z, so **behind the primary is +z**.

| quantity | value |
|---|---|
| M1–M2 spacing — the yardstick | **1.0420 m** |
| deepest optic behind M1 (the field mirror) | **1.8866 m** |
| **overhang** | **+0.8446 m — 1.81× the front span** |
| optics slab behind M1 | +1.3234 … +1.8866 m (depth 0.5632) |
| back focal PATH, M1 plane → interface | 2.8075 m (2.69×) |
| … plus the stated 1.000 m instrument | 3.8075 m (3.65×) |
| observatory envelope | Ø1.120 m × 3.825 m |
| through-hole the primary needs | radius ≥ **174.0 mm** |
| rays traced / lost | 10 665 / 0 |

Chief-ray legs: M1→M2 1.0421, M2→FM 2.9249, FM→M3 0.5541, M3→ColdStop 0.3724 m.

**So the complaint is exact and it is one number: the structure behind the primary
runs 1.81× as deep as the structure in front of it, and the instrument runs a
further metre beyond that.**

### 1a. And the measurement found something bigger than the depth

Body model: the **1.15-scaled convex hull** of each element's measured footprint,
grown 15 mm (`P.pack.fold_margin`). Union over the field box, because one mirror
has to carry every field.

| element | footprint r, box centre | footprint r, field box | body r | beam offset from vertex |
|---|---|---|---|---|
| M1 | 499.7 | 500.1 | 574.3 | 0.3 |
| M2 | 83.6 | 89.9 | 103.1 | 11.1 |
| FM (field mirror) | **1.8** | **112.6** | 129.0 | 183.9 |
| M3 (collimator) | **17.0** | **87.0** | 99.6 | 111.7 |
| ColdStop | 16.8 | 20.0 | 23.0 | 0.1 |

(mm. The field mirror sits ON the intermediate image, so its footprint is the image
itself: 1.8 mm for one field, 112.6 mm for the box.)

> **The M2 → field-mirror feed beam runs through the collimator's own glass.**
> Signed clearance **−79.9 mm** with the body model — and **−55.4 mm against bare
> lit glass, no 1.15, no edge allowance.** At the box centre alone it is −6.0 mm
> (i.e. it was always marginal); over the field box it is a hard obstruction.

The mechanism is not subtle once measured: the collimator sits *inside* the
converging feed cone, which passes it in an annulus 27.8–55.6 mm from its centre.
That works for one field — 10.8 mm of daylight all round. But a monolithic
collimator must cover its **union** footprint, 87.0 mm of it, and that glass is
exactly where the other fields' feed beams pass.

**This is not a packaging defect and no fold can fix it** — a fold is an isometry,
so it carries every internal clearance across unchanged (asserted below at
1.4e-12 mm). It is a property of the four-mirror form at this operating point, and it
is what `afocal4_pack`'s gate could not see: that gate checks the LAST mirror's exit
leg for fold daylight, at the box centre, and never asks whether a body is standing
in a beam.

---

## 2. What the committed single fold does — and does not do

Section 1 of the driver rebuilds the S4b demonstration's own recipe on **this**
deck (one flat after the last mirror, at the station `afocal4_pack` picks), so the
comparison is three columns of the same design instead of a comparison across
designs. Deck: `afocal4_b2long_343mm_1fold.in`.

* **It does not touch the depth.** Deepest optic **1.8866 m**, overhang **+0.8446 m**
  — identical to the unfolded deck, to the last digit. The fold sits *downstream of
  everything deep*: it moves the interface pupil sideways and leaves the collimator
  and the field mirror exactly where they were.
* **It costs the shroud.** The instrument leaves radially (axial fraction −0.00) and
  reaches **1.390 m** off the axis. Observatory envelope **Ø2.779 m × 2.626 m**,
  against Ø1.120 × 3.825 unfolded — the diameter grows 2.5×.
* **The flat does not fit where the gate put it.** `afocal4_pack` measured 17.5 mm
  of beam-to-beam daylight at that station against its own 15.0 mm margin, and
  passed. But over the field box that flat's own footprint is **103.7 mm** in radius
  (body 134.3 mm) and it clips the FM→M3 feed beam by **−73.6 mm**. The gate's
  margin is a *number*, not a body: it never sizes the part it is making room for.

---

## 3. Path A — four folds in the collimator feed leg

Deck: **`afocal4_b2pack_343mm.in`**. Driver `afocal4_packaging.m` section 2.

The depth lives in the M2 → field-mirror leg — the field mirror is the deepest
element, and it is deep *because* basin 2 buys its pupil control by pushing the
intermediate image ~900 mm behind the primary so the field mirror can sit on it. So
the folds go there, not after the last mirror.

```
    +z --F1--> +x --F2--> -z --F3--> -x --F4--> +z
```

Normals in the x–z plane so the field bias in y maps to y. Four is the minimum: a
−z leg is the only thing that buys depth back and reaching one costs a fold, and
the train must leave heading **+z** so the instrument runs aft inside the shroud
instead of radially out of it. A 2-fold dog-leg exits −z (instrument into the
primary); a 3-fold exits +x (instrument radial, the committed recipe's problem).
The fourth leg returns **toward** the axis so the lateral step that buys clearance
is not also paid as instrument girth.

Stated quantities: `x_step` 375 mm, `x_out` 190 mm, `z_front` 230 mm, `m3_gap`
100 mm. Everything else follows from the deck's own spacings:

| station | x (m) | z (m) | turns |
|---|---|---|---|
| F1 | 0.000 | +0.4467 | +z → +x |
| F2 | 0.375 | +0.4467 | +x → −z |
| F3 | 0.375 | +0.2300 | −z → −x |
| F4 | 0.190 | +0.2300 | −x → +z |
| FM | 0.190 | +0.8932 | |
| M3 | 0.190 | +0.3300 | |
| ColdStop | 0.190 | +0.6891 | |

| | as committed | S4b recipe, 1 fold | **Path A, 4 folds** |
|---|---|---|---|
| deepest optic behind M1 (m) | 1.887 | 1.887 | **0.893** |
| overhang vs the M1–M2 spacing (m) | +0.845 (1.81×) | +0.845 (1.81×) | **−0.149 (0.86×)** |
| optics slab depth (m) | 0.563 | 0.563 | 0.663 |
| optics radius (m) | 0.296 | 0.296 | 0.529 |
| instrument radial reach (m) | 0.465 | 1.390 | 0.518 |
| instrument runs | aft | **radially** | aft |
| observatory envelope | Ø1.120 × 3.825 m | Ø2.779 × 2.626 m | **Ø1.120 × 2.832 m** |
| body clearance floor, pre-existing (mm) | −79.9 | −79.9 | −79.9 |
| body clearance floor, new flats (mm) | — | **−73.6** | **+5.8** |
| back focal path (m) | 2.808 | 2.808 | 2.808 |
| rays lost over the field box | 0 | 0 | 0 |

**The depth question closes.** The optics behind the primary now occupy
z = +0.230 … +0.893 m — inside the 1.042 m the telescope already needs in front of
it — and every body stays inside r = 0.529 m, under the primary's own 0.560 m
keep-out. The observatory goes from **Ø1.120 m × 3.825 m to Ø1.120 m × 2.832 m**:
**a metre of length removed for no diameter at all.**

**What it does not close, and this is stated not smoothed over:**

* the pre-existing feed-beam-through-the-collimator interference is carried across
  unchanged (−79.9 mm), because that is what an isometry does;
* the new flats clear the beams by only **+5.8 mm** (F4→FM beam against F2's body).
  Positive, but that is a detailing margin, not a design margin;
* each flat is large and decentred: footprint radii **127–132 mm** over the field
  box, sitting **98–146 mm** off the axis the vertex is on, because the field walk
  (±76 mm of intermediate image) dominates the beam's own 35 mm radius. These are
  ~300 mm flats, not small pick-offs;
* the flats work at **45° AOI**. That is a packaging choice, not an optical
  constraint — `afocal4_pack` excludes flats from its incidence column for exactly
  this reason — but at 45° a real coating carries diattenuation and retardance, and
  four of them in series is a polarization budget this study did not open.

---

## 4. The folds are null — asserted, not assumed

Every deck re-scored on `afocal4_score`, the same kernel the trade used, and every
clearance re-measured.

| deck | WFE nm | blur µm | breathe % | wander µm | M | body floor, pre-fold mm |
|---|---|---|---|---|---|---|
| parent | 10 406.98 | 157.02 | 0.1240 | 161.23 | 30.00663 | −79.886 |
| one fold | 10 406.98 | 157.02 | 0.1240 | 161.23 | 30.00663 | −79.886 |
| four folds | 10 406.98 | 157.02 | 0.1240 | 161.23 | 30.00663 | −79.886 |

max |ΔWFE| **3.07e-8 nm**, |Δblur| 4.35e-12 µm, |Δbreathe| 4.14e-14 %,
|Δwander| 3.58e-12 µm, |ΔM| 1.42e-14.

The **pre-existing** clearance floor is a sharper test than any merit column: a
merit can agree while the geometry moved, and the clearance model reads the geometry
directly. It agrees across all three decks to **1.44e-12 mm**.

### The null test earned its keep — twice

**1. It rejected a route.** The first four-fold route scored 674 nm away from its
parent on a deck an isometry cannot have changed. The cause is a real hardware
statement, now measured and asserted in `pack_route` before any deck is written:

> A +90°/−90° fold **pair** is two flats whose planes are perpendicular, so they
> always intersect — for F1/F2 at `x = x_step/2`. A ray landing on the first flat
> beyond that line has to reflect toward a point behind it: the engine rejects the
> negative path length and loses the ray, and physically the two flats are cut into
> each other. **Each pair's step must exceed twice the beam's half-extent on its
> first flat, measured over the whole field box.**
>
> Measured here: half-extent **80.4 mm**, so each step must exceed **180.9 mm**.
> Empirically the scoring reproduced the parent exactly at 175 mm and 200 mm, was
> 1.2 % off at 150 mm and 6.5 % off at 125 mm — the bound, found the hard way and
> then derived.

**2. It exposed a bug in this study's own machinery** — worth recording because it
fails silently and plausibly. Accumulating the per-field ray-history masks with
`OK = cat(1, OK, hi.ok)` seeded from `[]` returns a **double** array of ones and
zeros, not a logical; `h.P(:,mask,j)` then indexes *by value*, handing back N copies
of ray 1 with the right size and an entirely plausible centroid. One full run's
clearances were the chief ray's. Seed logical accumulators explicitly.

---

## 5. Model choices that are load-bearing

| choice | why, and what the alternative did |
|---|---|
| bodies are **hulls**, not disks | Over a field box the collimator's footprint is nine patches walking ~70 mm; a centred disk of the union's max radius fills in the middle — exactly where the feed beam passes — and reports a 107 mm interference that is the model's, not the design's. (`oi_clear` carries the same choice.) |
| clearances over the **field box**, not the deck's field | The corners walk the beam to 80 mm off the fold axis where the centre field shows 44 mm. A centre-field table is optimistic by ~2× exactly where the folds are tightest. |
| both distance measures **sampling-free** | A fold cuts one long leg into several short ones, so a station-sampled model re-samples the same geometry at a different phase and reports a different answer — measured at 10.6 vs 5.8 mm, then 8.4 vs 4.3 mm, on pairs an isometry cannot have moved. Leg-body is now an exact ternary search (distance to a convex set is convex along a segment); leg-leg is the closed-form segment-segment distance. |
| a leg-**leg** zero is **not** an interference | Light passes through light, and on a wide-field system different fields' beams genuinely cross. A leg-leg floor says only whether a further flat could be inserted there. The leg-**body** column is the interference one. |
| through-holes are a **requirement**, not a collision | This train sends its beam back through the primary; that is true of the unfolded deck too. Reported as a hole radius, not a pierce. |
| folds emit `ApType= None` | A design-phase aperture is a body for a clearance check, not a stop — an honestly-sized flat emitted as a hard aperture turns a packaging study into a ray-loss study. The clear aperture each flat actually needs is reported from the traced beam. |

---

## 6. Files

| file | what |
|---|---|
| `afocal4_packaging.m` | the driver: measure → committed recipe → Path A → null → figures → sensitivity |
| `pack_legs.m` | engine-truth leg table, stations, footprints, envelope |
| `pack_clear.m` | signed clearance floors: leg-vs-body (hull) and leg-vs-leg, over a field set |
| `pack_fold.m` | multi-fold inserter on a **committed** deck, by the `add_fold` reflection isometry |
| `pack_route.m` | the four-fold route, with the plane-intersection bound measured and asserted |
| `pack_view.m` | x–z and y–z packaging elevations against the yardstick |
| `check_record.m` | re-measures what `RESULTS.md` §S4b.4 states about the committed folded demonstration |
| `probe_null.m` | the reproducer for §4's plane-intersection finding: the same route scored at a range of lateral steps |
| `run_pack.m` | one-line `matlab -batch` wrapper for the driver |
| `afocal4_b2pack_343mm.in` | **the packaged deck** |
| `afocal4_b2long_343mm_1fold.in` | the committed recipe rebuilt on this deck, for the comparison |
| `afocal4_pack_compare.png` | the three layouts on one scale, with the envelope and the yardstick drawn |
| `afocal4_b2pack_343mm_view_std.png` | `macos.view_std` of the packaged deck |
| `afocal4_packaging.mat` | every struct the run produced |

Run:

```matlab
run('~/dev/MACOS_res_dev/mmacos/mmacos_setup.m')
addpath('.../challenges/afocal4/packaging'); addpath('.../challenges/afocal4')
R = afocal4_packaging();                       % everything
R = afocal4_packaging('sections', 0);          % just the gap (brief step 1)
R = afocal4_packaging('sections', [0 2 3]);    % Path A + the null
```

Model size 256, one MATLAB process per model size, `MACOS_HOME=~/dev/macos/macos_f90`.

---

## 7. Open, and for Dave

1. **The collimator-in-the-feed-beam interference is the real buildability item**,
   and it is bigger than the depth the brief asked about. It is a Path-B question
   (bounded redesign), not a packaging one: no fold can move it. The cheapest
   candidates are a longer M2→FM leg (more room around the collimator, at the cost
   of the depth this study just removed) or a smaller intermediate image (a shorter
   effective focal length into the field mirror). Neither was attempted here.
2. **The four-fold package's 5.8 mm internal margin is the best this topology
   reaches**, and the window is narrow. Sweeping where the train comes back to, at a
   fixed 185 mm return step (the plane-intersection bound leaves nothing else free):

   | `x_out` m | deepest m | new-flat floor mm | pre-fold floor mm | instrument r m | shroud m | rays lost |
   |---|---|---|---|---|---|---|
   | 0.150 | 0.8932 | **−42.6** | −79.89 | 0.499 | 1.120 | 0 |
   | **0.190** | 0.8932 | **+5.8** | −79.89 | 0.518 | **1.120** | 0 |
   | 0.225 | 0.8932 | +5.8 | −79.89 | 0.537 | 1.124 | 0 |
   | 0.260 | 0.8932 | +1.8 | −79.89 | 0.559 | 1.190 | 0 |
   | 0.300 | 0.8932 | **−24.6** | −79.89 | 0.585 | 1.267 | 0 |

   The depth (0.8932 m) does not move at all — it is fixed by `z_front + L_next +
   m3_gap` alone. What `x_out` buys is clearance from the outbound axial leg; what
   it costs is girth, and past 0.26 m the return fold walks back into the
   collimator leg. **A ~70 mm-wide window, best margin ~6 mm**: a fifth station, or
   a design change, is what would open it up.
3. **A record discrepancy, flagged not fixed.** `RESULTS.md` §S4b.4 states, for the
   committed folded demonstration, an interface pupil at `[+0.304, −0.004, +0.614] m`
   and an instrument z-slab `+0.464 … +0.764 m`. Measured from the committed deck
   `afocal4_b_final_folded.in`, the fold and the interface plane are both at
   **z = +1.3782 m** with the pupil at `[+0.2483, −0.0051, +1.3782]`, so the
   instrument slab is `+1.228 … +1.528 m`. Run `check_record` to reproduce. The
   prose and the committed `.in` appear to come from different runs; which is
   authoritative is Dave's call, and nothing under `challenges/afocal4/` was touched.
4. **Slide 12.** Its footnote already says packaging was not driven to completion.
   The measured replacements are in §1 and §3 above; the slide edit is outward-facing
   and waits on sign-off (`doc/STYLE_REPORTS.md` §5).
