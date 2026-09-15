# REPORT — the reflective front end, designed on the 22.5° bench

TO for Dave, 2026-09-15. Branch `dev-candidate`. `BRIEF_to_reflective.md`: the
same TG96 bench as the lens rig with the two lenses replaced by off-axis
parabola sections, on the node angle Dave ruled. Numbers first; every claim
carries its run tag. Companions: `REPORT_oap.md` (CCMac's 7° history, kept),
`REPORT_bench_realism.md` (the node round), `REPORT_gauge_ifo.md` (the lanes).

## Status

| item | state |
|---|---|
| 1 — the drawing defect diagnosed and fixed; the lens rig unchanged | **done** — §1; `lens22g` (gate, pixel-identical), `oap22` |
| 2 — the design: fold angles and off-axis distances from a clearance solve | **done** — §2; the design is OAP1 20° / OAP2 25°, sides +1/−1, polarizer in the source leg, output optics 125 mm ahead of OAP2, collimator at its focus: **worst +33.4 mm over 8 parts**, no ray loss. Tags `fold1`–`fold4`, `loss`, `loss_src`, `loss_a2`, `conj`, `oap22d`; tail retune `oap22d_tail` |
| 3 — the layout in the recipe; the parts list | **done** — §3; auto-placed labels on the OAP rig (lens figure untouched), parts list printed by the runner (`oap22d`, `oapdraw3`) |
| 4 — the interferometer on it (rows on the 30 nm surface, servo, descent) | **running, and one row is BAD**, §4. Good: tail null 0.0223 nm, D1 100.00 % within 2 px, flat-DM null 22.3 pm. **Bad: Stage C single-actuator gain 0.0879 against 0.9927 / 0.9968 — open, §4.1.** `oapifo` in Stage D; `oapifol` and `oapdesc` queued |
| 5 — the mask sensors on it (S / V / P, bench + battery, stations figures) | **precondition measured + run queued**, §5: the ZWFS seat is 0.000 λF/D at trim 0 on this bench (`zseat`, `zseat2`) against the record's 1.1 λF/D at 6.14 mm; `oapsens22` queued (`zwfs_dm96/runs/oapsensseq.sh`) |
| 6 — the fold-angle lever: half OAP2's angle; does the pinhole recover? | **reframed**, §6: the blur is LINEAR in the angle and is 0.000 λF/D at both 20° and 25° once the conjugate is right, so the trade the item assumes does not exist. Its empirical half waits on item 5's P reading |
| 7 — the P/SRI bench through the clearance tool | **done** — recorded in `pdi_dm96/REPORT_gauge_pdi.md` (its own brief): PASS at 22.5° (+36.9 mm), FAIL at 7°; the Mach-Zehnder node clear by 227–383 mm at both. Runner `psri_clearance.m`, tag `psriclear2` |

## What is running, and how to pick it up

Detached jobs, all launched from `runs/`; each writes `runs/<tag>/<tag>_*` and
records its exit code in `runs/<tag>.log`. Read this table, then
`grep exit runs/*.log`, then continue from the first item not done.

| job | script | what it produces | state |
|---|---|---|---|
| `oap22d_tail` | `runs/tailseq.sh` | `oap22d_tail.mat` — the reflective tail re-fit on the designed geometry | **done**: null 0.0223 nm, poke recovered 150.0 of 150 nm |
| `oapifo` | `runs/ifoseq.sh` | the interferometer's rows on the 30 nm surface + the clearance table | **running** |
| `oapifol` | `runs/ifoseq.sh` | the closed-loop hold metric (hour-class) | **DEFERRED** — it would spend an hour measuring the known-broken reading (§4.2). Re-queue after the tail is fixed |
| `oapdesc` | `runs/descseq.sh` | item 4's descent: the same 60 / 150 / 300 nm ladder the record's `descent_oap` stalls on (5856 / 23557 / 53150 pm, never reaching 10 nm from 150 or 300) | **DEFERRED** — same reason |
| `tailA` / `tailB` | `runs/tailabseq.sh` | **the decisive diagnostic**: the design at model 512 with the TUNED tail vs the GEOMETRIC SEED (§4.2) | running |
| `oapsens22` | `zwfs_dm96/runs/oapsensseq.sh` | item 5's mask sensors — **independent of the tg96 tail** (the ZWFS carries its own tuned tail and `MASK_TRIM` 0), so it is unaffected by §4.2 and still worth its run | queued |
| `polA` / `polB` | `runs/polabseq.sh` | the polarizer A/B | **dropped** — redundant with `tailA`/`tailB` once the polarizer stopped being the leading suspect (§4.1). The script is kept; re-run it if the tail is exonerated |

`runs/ifoseq.sh` waits (up to 2 h) for `oap22d_tail.mat`, copies it under each
tag, and aborts loudly rather than falling back to the record's 7-degree
`oap_tail.mat` — which was fit **with** the conjugate error and would measure
the old bench.

**The design, in one line, for anything that needs to rebuild it:**

```
tg96_run('bench.optics','oap', 'bench.POL_IN','source', ...
         'bench.SRC_AT_FOCUS',true, 'bench.D_RC_L2',125, ...
         'oap.OAP1_AOI',20, 'oap.OAP2_AOI',25, ...
         'oap.OAP1_SIDE',1, 'oap.OAP2_SIDE',-1, ...
         'clear.BODY',struct('Baffle',50,'Detector',50,'TestOptic',90,'PZT',60), ...)
```

and for the sensors (item 5), the same `bench.*` knobs through `zwfs_run`'s
generic forwarding, **plus `'bench.MASK_TRIM',0`** — `zwfs_params` carries the
lens rig's −5.582.

## 1. The OAP bodies were drawn at the parent parabola's vertex

**The hypothesis in the brief is half right, and the half it gets wrong is the
half that matters.** The vertex/pole distance is real and is the size the brief
names; but `view_rx` is NOT the offender — it already draws an off-axis section
on the beam. The bars that sit off the rays in `oap_vlayout.png` are drawn by
the *recipe*, and the same confusion was doing far worse damage, silently, in
the clearance tool.

### 1.1 The measurement

Engine truth on the OAP rig at the 22.5° node with CCMac's 5° / 9° folds
(model 512, 65 rays; `VptElt` / `RptElt` read back through `macos.get_elt_vpt` /
`get_elt_rpt`, the footprint from the traced ray history), mm:

| | VptElt (parent vertex) | RptElt (pole) | ray-footprint centroid | \|fp − vertex\| | \|fp − pole\| |
|---|---|---|---|---|---|
| OAP1 (`L1`, collimator) | (850.63, 148.84, 0) | (857.14, 0, 0) | (857.37, 0, 0) | **148.99** | 0.23 |
| OAP2 (`L2`, focuser) | (1175.13, 247.63, 0) | (1261.36, 146.57, 0) | (1260.98, 146.20, 0) | **132.89** | 0.53 |

Those two distances are the OAPs' **off-axis distances**, and they are not free:
off-axis = (conjugate distance)·sin 2·AOI = 857.14·sin 10° = 148.8 mm and
428.57·sin 18° = 132.4 mm. The parent focal length is the conjugate ×cos²AOI —
850.6 and 424.4 mm for a nominal 857 / 429 — so `KrElt` = −2f = −1701.3 and
−836.2, which is what the emitted deck carries.

### 1.2 `view_rx` is correct — measured, not argued

The rim `view_rx` actually draws (pulled off the axes, rim color `[0.2 0.3 0.5]`,
one element shown at a time) is centred **1.07 mm** from OAP1's pole and
**1.51 mm** from OAP2's, spanning 83.6 / 89.8 mm — on the beam, at the beam's
own size. The reason is structural: `add_oap` deliberately leaves the element
with no declared aperture (`ap_type` 0, `ap_vec` 0, `lmon` 0 — confirmed on both
OAPs), because a Circular `ApVec` is applied about `VptElt` and would block the
whole bundle. With no aperture, `elt_geom_` falls through to the **ray-footprint
hull**, which is on the beam by construction. So the viewer needs no change, and
it did not get one — the blast radius of this item's fix is zero on every rig
that has no off-axis section.

### 1.3 What was actually wrong

**(a) The layout recipe.** `tg96_run/draw_render_` overlays a manual mirror bar
on every `Reflector` — its own comment says *"a short bar at the pole"* — and
passed `Ea(i).vpt`, the parent vertex. Those are the black bars 149 / 133 mm off
the rays in `oap_vlayout.png`. The same function's crop box (`crop_`) and label
anchors (`label_`) read `.vpt` too, so the tail panel was cropped about the
parent vertex and the "OAP2 (focuser)" leader pointed into empty space.

**(b) The clearance tool, worse and invisibly.** `dmg_bench_clearance` builds
its beam **segments** from consecutive records' `.vpt`. On the OAP rig the train
it modelled was therefore not the bench:

| segment the tool used | from | to | |
|---|---|---|---|
| `test: Baffle -> L1` | (1279.2, 74.4) | (850.6, 148.8) | ends **149 mm off the mirror** |
| `test: L1 -> PolIn` | (850.6, 148.8) | (867.1, 0.0) | a **150 mm phantom leg** that is not a beam |
| `test: Analyzer -> L2` | (1236.6, 121.8) | (1175.1, 247.6) | ends 133 mm off the mirror |
| `test: L2 -> FocalMask` | (1175.1, 247.6) | (879.5, −48.0) | starts there |

and parts were then scored against the phantom: output QWP **−28.9 mm**
"against `test: Baffle -> L1`", splitter +24.8 against the same, input polarizer
−4.4 against `ref: FLpow -> FLflat`. **No pre-fix OAP clearance number is a
measurement of the bench; none should be quoted.**

### 1.4 The fix: ask for the element's station, never its vertex

New static `macos.design.Bench.station(e)` returns the element's position **on
the beam** — its pole (`rpt`), which `Bench.push` resolves to `vpt` for every
ordinary element, so it *is* the vertex everywhere except an off-axis section.
Three consumers now ask it: `dmg_bench_clearance` (part centres **and both
endpoints of every beam segment**), and `tg96_run/draw_render_` (mirror symbols,
crop boxes, label anchors, the PZT leader). `Bench.sketch` has always drawn
mirrors at `rpt` — the builder was right all along; the downstream consumers
drifted. `add_oap`'s help and the class method list now carry the rule.

**Consumer audit** (every `.vpt` read under `templates/40_benches` and
`src/+macos/+design`), because "no consumer change needed" is vacuous unless the
consumers are classified:

| consumer | can it see an off-axis section? | action |
|---|---|---|
| `dmg_bench_clearance` (records, segments, labels) | **yes** — items 2 / 5 / 6 / 7 | fixed |
| `tg96_run/draw_render_` (`mirror_sym_`, `crop_`, `label_`, PZT) | **yes** | fixed |
| `pdi_layout_fig`, `psri_layout_fig`, `psri_bench` | not today (flat folds + lenses); **yes the moment the P/SRI sits on the OAP front end** | flagged for item 7 |
| `zwfs_vlayout`, `dmg_analyzer_maps` (re-launch a Bench at `E(iFL).vpt`) | not today (a lens) | flagged |
| `example_bench_layout`, `prop_layout`, `System` | flats / lenses; `System`'s snapshot carries both | none |
| `ctb_prop_layout:282` | `s.vpt` is a **FEX result**, not an element | none |

### 1.5 Gate: the lens rig does not move

`lens22g` — the lens rig re-emitted post-fix at `lens22`'s own settings
(model 1024, NGRID 385, stages bench + figs + clearance):

- **`lens22g_vlayout.png` is pixel-identical to `runs/lens22/lens22_vlayout.png`.**
  The files differ in 2 bytes only, in PNG's `tIME` and `Creation Time` chunks,
  which MATLAB stamps into every `print`. Decompressed IDAT: 289 610 bytes on
  both, SHA-256 `9009d2d95df8737286c87557…` on both.
- The clearance table reproduces the record cell for cell: compensator **+38.2**,
  output QWP +53.6, analyzer +63.8, L2 +99.3, input polarizer +137.5, L1 +153.1,
  test QWP +333.6, DM +361.1, reference QWP +362.8, PZT flat +390.9 —
  **worst +38.2 mm over 10 parts**, identical to `node22t`.

### 1.6 What the corrected drawing shows — and it is not buildable

`oap22` (the OAP rig on the 22.5° bench, folds 5° / 9°, model 1024) is the first
layout with the mirrors on the beam, and the first honest clearance table for
the reflective rig. **Worst −102.3 mm over 9 parts** (spec ≥ +25):

| part | clearance, mm | against |
|---|---|---|
| input polarizer | **−102.3** | `test: Baffle -> L1` (the source leg) |
| analyzer | **−93.9** | `ref: L2 -> FocalMask` (the tail) |
| output QWP | **−89.4** | `ref: L2 -> FocalMask` |
| splitter | **−49.6** | `test: Baffle -> L1` |
| OAP1 | **−1.8** | `ref: FLflat -> Detector` |
| compensator | +41.1 | `test: PolIn -> BSrefl` |
| field lens | +78.4 | `test: QWPtestOut -> Comptxfu` |
| camera | +89.1 | `test: QWPtestOut -> Comptxfu` |
| OAP2 | +99.0 | `ref: QWPrefOut -> BSbinr` |

Two distinct failures, both fold-angle driven, and item 2's problem:

1. **A near-normal fold makes the source leg double back through the node.** At
   OAP1 AOI 5° the chief turns 170°, so the source sits at x = 1701 mm — 587 mm
   *past* the splitter — and its diverging beam travels back along the whole
   node to reach the pole at x = 857. At the input polarizer, 10 mm downstream
   of the pole, the two legs are 1.7 mm apart: the polarizer is inside the
   incoming cone. No fold angle fixes that part where it stands
   (10 mm·tan 2·AOI ≥ 104 mm needs AOI ≥ 42°), so the reflective rig either
   polarizes the source ahead of OAP1 or moves the polarizer well downstream —
   a design decision for item 2, not a knob.
2. **The tail folds back over the output optics.** At OAP2 AOI 9° the converging
   leg runs from (1261, 147) to the mask at (880, −48), straight across the
   analyzer and output QWP.

`solve_fold_` could not see either: it checks only that the source / camera
**body** clears the collimated beam laterally (`F·sin 2·AOI ≥ beam + body +
margin`), which 5° / 9° satisfy by +22.4 / +6.0 mm. It never asks what the
folded leg crosses on its way.

**Figures.** `runs/oap22/oap22_vlayout.png` (train / node / tail, both arms) and
`runs/oap22/oap22_clearance.png`. The label placements in the node panel are
still the lens rig's and collide on this geometry — item 3 re-places them; the
geometry and the bodies are right.

**Run tags.** `lens22g` (gate), `oap22` (the corrected OAP rig), launched by
`runs/refseq.sh`; both exit 0.

## 2. The design

### 2.1 The instrument: solve the folds FROM the measured clearance

`oap_fold_solve.m` (+ `oap_fold_solve_batch.m`, `oap_fold_batch.sh`) sweeps the
two fold angles and the two fold sides, builds the rig at each point, and runs
`dmg_bench_clearance` on it. It reports the worst clearance, the part that
binds, and whether the trace loses rays there — a fold that clears on paper and
vignettes on the glass is not a design.

Why a sweep and not a rule: `tg96_run`'s `solve_fold_` asks only that the
source / camera **body** clear the collimated beam laterally,
`F·sin 2·AOI ≥ beam + body + margin`. It never asks what the folded leg
*crosses on its way*, which is the whole problem on this bench.

The off-axis distance is not a separate variable. For a parabola fed at
conjugate distance `r`, off-axis `= r·sin 2·AOI` and the parent focal length
`= r·cos²AOI`. Choosing the fold chooses both — and the source's own lateral
offset from the collimated axis *is* the off-axis distance, the same number.

`dmg_bench_clearance` gained one opt-in option for this, `BODY` (a struct
part-stem → physical body radius): a part's clear aperture is not its body, and
the SOURCE was not scored at all, because the builder's source-side element is
an Obscuring baffle rather than an optic. The sweep runs with the Stage-A rule's
own half-widths — source and camera 50 mm, DM 90, reference flat 60. Default
empty = the previous behavior, so §1.5's lens table does not move.

### 2.2 `fold1`: no fold angle makes the bench of record buildable

324 builds (9 x 9 angles x 4 side pairs, 3.1 s each), input polarizer in the
collimated leg as the record has it. **The input polarizer binds at every
point, on every side pair, and never comes within 80 mm of the spec.**
Worst clearance, mm, sides +1 / +1 (`*` = the trace also loses rays there):

| A1 \ A2 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | binds |
|---|---|---|---|---|---|---|---|---|---|---|
| **5** | −102 | −102 | −102 | −102 | −99 | −95 | −91 | −88* | −85* | Pol |
| **10** | −105 | −106 | −107 | −109 | −109 | −109 | −108 | −108* | −108* | **Baffle** |
| **15** | −101* | −101* | −101* | −101* | −99* | −95* | −91* | −85* | −82* | Pol |
| **20** | −98* | −98* | −98* | −98* | −98* | −98* | −93* | −87* | −81* | Pol |
| **25** | −89* | −89* | −89* | −89* | −89* | −89* | −89* | −89* | −83* | Pol |
| **30** | −80* | −80* | −80* | −80* | −80* | −80* | −80* | −80* | −80* | Pol |
| **35** | −69* | −66* | −66* | −66* | −66* | −66* | −66* | −66* | −66* | Pol |
| **40** | −62* | −55* | −48* | −39* | −31* | −31* | −31* | −31* | −31* | Pol |
| **45** | −56* | −50* | −42* | −33* | −21* | −3* | −25* | −60* | 14* | Detector |

The other three side pairs are the same table to within a few mm (full file:
`runs/fold1/fold1_fold.txt`, map `fold1_fold.png`).

**Why the polarizer cannot be cleared by an angle.** It sits `D_POL` = 10 mm
past the collimator. A lens collimator's conjugate leg is *on axis* — nothing
travels the other way — so 10 mm is free. An OAP collimator's conjugate leg
comes **back along the collimated axis**, and a part `d` past the pole sees it
at lateral `d·tan 2·AOI`: **1.7 mm at 5°, 11.9 mm at 25°**, against a
requirement of beam + part + mount + margin ≈ 130 mm. Clearing 130 mm at
d = 10 mm would need `AOI ≥ 42°`, and at 42° the rest of the bench has already
failed. The station, not the angle, is the variable.

**Two more things the map shows, both real collisions.**
- At **A1 = 10°** the *source head* binds instead, at −105 to −109 across the
  whole row. At that fold the baffle lands at (1259.85, 146.58) and OAP2's
  pole is at (1261.36, 146.57) — **1.5 mm apart**: the source assembly is
  sitting on the focuser. It is
  a coincidence of this bench — the source leg and the output leg both leave
  the splitter region at 45° and the two stations happen to coincide — but it
  is exactly the kind of thing only a measurement of the whole train finds.
- From **A1 ≥ 15° the trace loses rays** at every point. A fold angle that
  clears on paper and vignettes on the glass is not a design, so the solver
  refuses those points whatever their clearance; `oap_conj_probe` (§2.4)
  measures whether that is the optic or the node.

### 2.3 The ray loss at a large fold is the SAME part — and the mechanism is the mirror's own sag

`oap_loss_probe` (tag `loss`) traces the test arm at each fold and reads the
engine's per-ray status (`elt_mod`'s `RayStat_*` through
`macos.get_ray_status`). OAP2 fixed at 9°, sides +1/+1, polarizer in the
collimated leg:

| OAP1 AOI | rays | lost | obscured | **miss** | bracket | where |
|---|---|---|---|---|---|---|
| 5° | 3210 | 0 | 0 | 0 | 0 | — |
| 10° | 3210 | 0 | 0 | 0 | 0 | — |
| 15° | 3210 | 66 | 0 | **66** | 0 | elt 3 `PolIn` |
| 20° | 3210 | 343 | 0 | **343** | 0 | elt 3 `PolIn` |
| 25° | 3210 | 600 | 0 | **600** | 0 | elt 3 `PolIn` |
| 30° | 3210 | 769 | 0 | **769** | 0 | elt 3 `PolIn` |
| 35° | 3210 | 887 | 0 | **887** | 0 | elt 3 `PolIn` |
| 40° | 3210 | 1009 | 0 | **1009** | 0 | elt 3 `PolIn` |
| 45° | 3210 | 1070 | 0 | **1070** | 0 | elt 3 `PolIn` |

**The OAPs never lose a ray. Every loss is a surface MISS at the input
polarizer** — the same part the clearance sweep binds on, failing a second,
independent way.

The mechanism is the mirror's own **sag envelope**, and it checks out
arithmetically. An off-axis section is the parent parabola between radii
`h ± R` where `h = r·sin 2·AOI` and `R` is the footprint semi-axis, so its
surface spans a range of sag along the parent axis — which here IS the
collimated direction. At 15° the pole sits 57.4 mm of sag from the vertex and
the footprint spans **44.1 to 69.1 mm**: ±12 mm of GLASS about the pole, with
the plate 10 mm downstream. Rays landing on the far part of the mirror are
already *behind* the plate's plane when they reflect, and a forward-travelling
ray cannot reach a plane behind it. Predicted loss from that geometry: 2.8 % at
15°, measured **2.06 %**; at 45° the sag range is ±41 mm and a third of the
mirror is past the plate — predicted ≈ 33 %, measured **33.3 %**.

So the two failures are one fact stated twice: **a plate 10 mm past an
off-axis parabola's pole is inside the mirror.** It is not a tolerance to
tighten; the part collides with the optic.

This also disqualifies the otherwise attractive **90° fold**. At `AOI1 = 45°`
the source leg runs *parallel* to the polarizer's plane and never crosses it,
so the clearance constraint evaporates — sides +1/−1 at A1 45°, A2 ≥ 35° is the
only corner of `fold1` that goes positive (+33, +86, +86 mm, binding on the
compensator). But every one of those points loses a third of its rays into the
plate. The 90° fold is only available once the polarizer has moved.

**The cure, measured** (tag `loss_src`, the identical ladder with `POL_IN`
`'source'`): **0 rays lost at 5, 10, 15, 20, 25, 30, 35, 40 and 45°** — the
whole fold range opens, including the 90° fold.

### 2.4 Where the polarizer goes, and what it costs

`twyman_green` gained `POL_IN` — `'collimated'` (default, the record) or
`'source'`, `'oap'` optics only, so the lens rig is untouched by construction.
In `'source'` the input polarizer sits `D_POL` past the **baffle**, in the
diverging leg, which is where a real reflective bench polarizes anyway: the
laser is polarized before the spatial filter, not after the collimator.

**Its axis is reflected through OAP1** so the state arriving at the splitter is
the record's. A plane mirror maps a transverse vector by `a → a − 2(a·n̂)n̂`
about its normal, and that map is an involution, so the incoming axis that
*becomes* `ax_local(d_out, pol_in_deg)` after the fold is that same expression
applied to it. Without this, "45°" in the source leg is 45° about a different
local x (`ax_local` seeds from `perp(dir)`) and the fold flips the in-plane
component — a real change of input state, not a labelling one.

Two honest costs, neither a defect:
- the leg is **f/8.3**, so `add_polarizer`'s own "collimated, normal-incidence"
  precondition is broken at the `sin²(3.4°)` ≈ **0.4 %** level (the projected
  material-axis rule, `REVIEW_POL_ELEMENTS_2026-07-27`);
- a **coated** OAP1 (`coat_oap`) now acts on an already-polarized beam, so its
  diattenuation and retardance enter the input state. That is physics, not an
  artifact, and item 4 prices it.

The ZWFS / sensor rigs are unaffected — `zwfs_params` sets
`polarizing = false`, so there is no input polarizer in them at all.

### 2.5 There is no fold coma. There is a 25 mm conjugate error that the fold turns into coma.

CCMac's record reports a best-focus blur that grows with the fold angle —
0.17 / 0.31 / 0.47 / 0.65 / 0.82 λF/D at 1 / 3 / 5 / 7 / 9° — and calls it fold
coma, "scaling as the fold angle squared". **A paraboloid fed exactly at its
focus collimates perfectly at any off-axis distance**, so an on-axis
interferometer should pay nothing for the fold. Something else was being
measured.

`oap_conj_probe` (tag `conj`) isolates it on a bare two-mirror train at the
rig's own scale — source → baffle → OAP1 (collimate) → OAP2 (focus) → detector,
no node, no tail — with the source at the record's distance and at the
corrected one, best focus found per point:

| fold AOI | collimation after OAP1, µrad rms | equivalent focus | **best-focus blur, λF/D** | | |
|---|---|---|---|---|---|
| | record | corrected | record | **record** | **corrected** |
| 1° | 925.53 | **0.00** | 28.8 m | 0.128 | **0.000** |
| 3° | 925.53 | **0.00** | 28.8 m | 0.236 | **0.000** |
| 5° | 925.53 | **0.00** | 28.8 m | 0.367 | **0.000** |
| 7° | 925.54 | **0.00** | 28.8 m | 0.504 | **0.000** |
| 9° | 925.54 | **0.00** | 28.8 m | 0.645 | **0.000** |
| 15° | 925.57 | **0.00** | 28.7 m | 1.081 | **0.000** |
| 20° | 925.60 | **0.00** | 28.6 m | 1.465 | **0.000** |
| 30° | 925.72 | **0.00** | 28.5 m | 2.322 | **0.000** |
| 37° | 925.85 | **0.00** | 28.4 m | 3.028 | **0.000** |
| 45° | 926.10 | **0.00** | 28.2 m | 4.018 | **0.000** |

**And the mask-seat trim falls out with it.** The longitudinal trim the focus
needed is **+6.46 mm at every fold angle** in the record case and
**−0.00 mm at every fold angle** corrected. CCMac solved that trim on the trace
and got **6.14 mm**, calling it "the collimator's residual defocus refocused by
OAP2" — which is exactly right as a description and exactly the 25 mm conjugate
error as a cause. Fed at its focus, the reflective rig's mask seat sits at the
lens rig's own station, with **no trim at all**.

**The cause.** `Bench` emits `zSource` (25 mm) and the engine puts the real
point source at `ChfRayPos + zSource·ChfRayDir` (`sourcsub.F:38`), so the source
sits 25 mm *downstream* of the point `front_end` computes — while `add_oap`
builds the parabola for a focus **at** that point. The collimator is fed 25 mm
inside its focus. The residual convergence that predicts is
`f_par²/Δ = 850.6²/25 = 28.9 m`; the engine measures **28.8 m**.

**Why it never showed on the lens rig.** The lens collimator's figures are
*tuned* (`L1_Kr` 236.866, `L1_Kc` −0.5829, from `l2_trade`), and that tuning
absorbed the same 25 mm error as a conic adjustment. A parabola has no such
freedom, so on the reflective rig the error had nowhere to hide and came out as
an angle-dependent blur.

**It is LINEAR in the fold angle, not quadratic.** The measured slope is
0.071 λF/D per degree with a small offset — the signature of *defocus × off-axis
angle*, which is coma from a conjugate error, not an intrinsic property of
folding. Reading it as `AOI²` and concluding that halving OAP2's angle buys a
factor of 4 (at the price of a 1.33× longer leg) is therefore the wrong trade:
halving the angle halves the blur, and *fixing the conjugate removes it
entirely*.

**Builder option `SRC_AT_FOCUS`** (twyman_green, `'oap'` only, default false so
no recorded number moves): adds `zsource` to the source distance so the
effective point source lands on the parabola's focus. The pole, and therefore
the whole downstream bench, does not move — only the source does.

**What this means for the reflective rig's standing.** Every reflective-vs-lens
row in the record — cross-talk 0.18 against the lens's < 0.06, the servo that
never gets under the 2 pm walk, the vector sensor's 19.6 pm and the pinhole's
94 pm gate failures, the 6.14 mm mask-seat trim — sits downstream of a 1.1 λF/D
blur at the mask seat that need not exist. Items 4 and 5 re-measure them on a
correctly-fed collimator.

### 2.6 With the polarizer moved, the binding constraint moves to the OUTPUT optics

`fold2` (360 builds, `POL_IN 'source'`, everything else the record). The
polarizer stops binding; the source head and the output optics take over, and
**nothing in the grid reaches the 25 mm spec**. Sides +1/+1, worst clearance mm:

| A1 \ A2 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | binds |
|---|---|---|---|---|---|---|---|---|---|---|
| **6** | -99 | -93 | -86 | -77 | -63 | -40 | -33 | -31* | -29* | OutQWP |
| **8** | -99 | -93 | -86 | -80 | -80 | -80 | -80 | -80* | -80* | Baffle |
| **10** | -105 | -106 | -107 | -109 | -109 | -109 | -108 | -108* | -108* | Baffle |
| **12** | -99 | -93 | -86 | -79 | -79 | -79 | -79 | -79* | -80* | Baffle |
| **14** | -99 | -93 | -86 | -77 | -63 | -40 | -32 | -35* | -43* | Pol |
| **16** | -99 | -93 | -86 | -77 | -63 | -40 | **+2** | +3* | -0* | Pol |
| **18** | -99 | -93 | -86 | -77 | -63 | -40 | **+2** | +41* | +41* | L2 |
| **20-24** | -99 | -93 | -86 | -77 | -63 | -40 | **+2** | +54* | +56* | Comp |

`fold3` extends A1 to 26-46 deg and does no better: +12 mm at best without ray
loss. The +41...+56 mm points all carry `*`.

**The `*` at A2 >= 40 deg is the sag rule again, now on OAP2** (`loss_a2`,
A1 = 16 deg): 0 rays lost at A2 = 5...35, then **16 misses at A2 = 40 and 169
at A2 = 45, all at element 17 = `L2`**. At a 40 deg fold OAP2's surface spans
+-50 mm of sag about its pole while the analyzer sits only
`D_RC_L2 - 2*D_POL` = 35 mm ahead of it, so part of the mirror lies *behind*
the analyzer's plane. Same physics as the input polarizer, opposite end of the
bench.

**So the output optics' standoff is a clearance variable on a reflective rig,
not a packaging one**, and `oap_fold_solve` / `oap_loss_probe` now take
`D_RC_L2` (and `SRC_AT_FOCUS`) for exactly that reason.

### 2.7 The design

`fold4`: `POL_IN 'source'`, `SRC_AT_FOCUS true`, **`D_RC_L2` = 125 mm** (the
output quarter-wave plate and analyzer 125 mm ahead of OAP2 instead of 55, so
they clear its sag envelope and its returning tail). 120 builds. Worst
clearance, mm:

| | A2 = 25 | 30 | 35 | 40 | 45 | binds |
|---|---|---|---|---|---|---|
| **sides +1 / +1** | | | | | | |
| A1 = 18 | -22 | -31 | -46 | -74 | -14 | Baffle |
| A1 = 20 | +15 | +8 | -5 | -40 | +6 | Baffle |
| A1 = 22 | **+34** | **+42** | **+44** | +5 | **+60** | Comp |
| A1 = 24 | **+34** | **+42** | **+47** | **+53** | **+25** | Comp |
| **sides +1 / -1** | | | | | | |
| A1 = 18 | +4 | +5 | +6 | +6 | -26 | Baffle |
| A1 = 20 | **+33** | **+35** | **+35** | **+36** | -26 | Baffle |
| A1 = 22 | **+33** | **+41** | **+46** | **+52** | -27 | Comp |
| **sides -1 / +1** | | | | | | |
| A1 = 14 | -0 | +4 | +9 | +15 | +21 | Comp |
| A1 = 16 | -11 | -7 | -2 | +4 | +10 | Comp |
| A1 = 18-24 | worse with A1 | | | | | Comp |

No point in any of these carries a ray loss, so every positive entry is a real
design. The smallest total fold that clears is **OAP1 20 deg, OAP2 25 deg,
sides +1 / -1, at +33 mm** -- and it sits on a plateau (+33 / +35 / +35 / +36
across A2 = 25...40), so OAP2's fold is free to a wide tolerance once OAP1 is
at 20 deg.

**The parts that follow from the angles:**

| | fold AOI | off-axis distance `r*sin 2*AOI` | parent focal length `r*cos^2 AOI` | conjugate `r` |
|---|---|---|---|---|
| OAP1 (collimator) | 20 deg | **551.0 mm** | **756.9 mm** | 857.14 mm |
| OAP2 (focuser) | 25 deg | **328.3 mm** | **352.0 mm** | 428.57 mm |

Both are ordinary catalogue shapes -- a 40-deg-off-axis and a 50-deg-off-axis
section, each carrying a 103 mm beam.

**The solver's verdict and the measured clearance table** (`fold4`, model 512 /
65 rays, mount +8 mm, bodies: source and camera 50, DM 90, reference flat 60):

> SOLVED: OAP1 20 deg / OAP2 25 deg, sides +1/-1 -- worst **+33.4 mm** (Analyzer)
> OAP1 off-axis 551.0 mm, parent f 756.9 mm; OAP2 off-axis 328.3 mm, parent f 352.0 mm

| element | type | a+mount | beam r | **clearance, mm** | against |
|---|---|---|---|---|---|
| analyzer | TrPolarizer | 59.9 | 46.9 | **+33.4** | test: L2 -> FocalMask |
| source head | Obscuration | 58.0 | 21.9 | **+34.6** | test: Analyzer -> L2 |
| compensator | Refractor | 63.7 | 50.7 | **+37.0** | test: L1 -> BSrefl |
| output QWP | WavePlate | 59.9 | 46.9 | +47.2 | test: L2 -> FocalMask |
| input polarizer | TrPolarizer | 35.4 | 22.4 | +58.5 | test: Analyzer -> L2 |
| splitter | Reflector | 63.7 | 50.7 | +75.1 | test: PolIn -> L1 |
| OAP1 | Reflector | 59.4 | 50.0 | +149.3 | test: QWPtestOut -> Comptxfu |
| OAP2 | Reflector | 59.4 | 51.4 | +170.1 | ref: BStxbf -> QWPrefIn |
| field lens | Refractor | 14.2 | 1.2 | +463.3 | test: Comptxbd -> QWPtestIn |
| test QWP, DM, camera, reference QWP, PZT flat | | | | — | no other beam crosses their plane |

**Worst +33.4 mm over 9 scored parts; spec >= +25. The reflective bench is
buildable.** The source head is in the table for the first time (the `BODY`
option), at +34.6 mm with its 50 mm body; so is the camera, which no beam
crosses.

**What changed from the bench of record, and why each change is forced:**

| change | from | to | forced by |
|---|---|---|---|
| input polarizer's leg | collimated, 10 mm past the collimator | the diverging source leg, 10 mm past the baffle | it is inside the incoming cone at every angle, AND inside the mirror's sag envelope |
| output optics standoff `D_RC_L2` | 55 mm | **125 mm** | OAP2's sag envelope (+-50 mm at a 40 deg fold) and its returning tail |
| collimator conjugate `SRC_AT_FOCUS` | the record (25 mm inside focus) | at the focus | it is the whole of the "fold coma" and the whole of the 6.14 mm seat trim |
| OAP1 fold / side | 5 deg, +1 | **20 deg, +1** | clearance |
| OAP2 fold / side | 9 deg, +1 | **25 deg, -1** | clearance |
| mask-seat trim | 6.14 mm | **0.00 mm** | falls out of the conjugate fix |

Nothing else moves: the splitter stays at 22.5 deg, both arms and the recomb
plane are the lens rig's, the DM leg is 450 mm, and the tail architecture is
unchanged (it is re-tuned, not re-designed).

### 2.8 The design at full resolution

`oap22d` — the same bench built by `tg96_run` at the record's own resolution
(model 1024, NGRID 385), stages bench + figs + clearance. It reproduces the
sweep's table to the tenth of a millimetre:

> Clearance -- every physical part vs every beam it is not in (mount +8 mm):
> Analyzer **+33.4**, Comp +37.0, OutQWP +47.2, Pol +58.5, BS +75.1, L1 +149.3,
> L2 +170.2, FL +463.3; test QWP / DM / camera / reference QWP / PZT flat have
> no other beam crossing their plane.
> **worst +33.4 mm over 8 parts (spec >= 25 mm)**

Stage A's own screening rule agrees at the node — every node part ≥ +25.6 mm —
and the fold solve now prints the design's lateral clearances: OAP1 at 20°
gives 551.0 mm (margin +424.5), OAP2 at 25° gives 328.3 mm (+201.9).

Two bookkeeping fixes went in with it, both in the direction of "the screen and
the measurement should describe the same bench":

- **`P.clear.BODY`** is now forwarded from the runner into
  `dmg_bench_clearance`, so the measured table can carry physical bodies. It
  defaults to **empty** — the record — so `REPORT_bench_realism` §2's lens
  table (worst +38.2 over 10 parts) reproduces exactly; the reflective runs
  pass the Stage-A rule's own half-widths (source and camera 50, DM 90,
  reference flat 60). Worth knowing either way: **with apertures only, the
  source head is not in the table at all** (its builder element is an Obscuring
  baffle, not an optic) and the camera is scored at its pupil-image size, 4.6 mm.
- **`node_parts_` drops the input-polarizer row when `POL_IN` is `'source'`.**
  The screening rule is about parts at the splitter, measured against the
  opposing arm at the splitter angle; a polarizer in the diverging leg is not
  one of those, and leaving the row in reported a part that is not there.

**Figure:** `runs/oap22d/oap22d_vlayout.png` — the train, the node and the tail,
both arms, mirrors on the beam. The source leg now enters steeply from outside
the node and the tail drops away from it, which is the whole point of the fold
angles. The labels are still placed for the lens geometry and collide; item 3
re-places them.

## 3. The layout and the parts list

**The figure** is `tg96_run`'s own three-panel render (the `zwfs_vlayout`
recipe: `macos.view_rx` from above, both arms overlaid, passive planes hidden,
a mirror bar at every reflector) — the tool's own output file, not a
re-rendering. Design run: `runs/oap22d/oap22d_vlayout.png` at the record's
resolution; `runs/oapdraw3/` is the same figure at dev resolution.

Two changes were needed beyond §1's station fix:

- **The labels place themselves on the reflective rig.** The offsets in
  `draw_render_`'s `Ltrain` / `Lnode` / `Ltail` tables were hand-tuned for the
  lens geometry and collided once the folds moved — three labels on top of each
  other in the node panel. For `optics 'oap'`, `label_` now *places* each
  label: eight compass directions at three standoffs, scored by distance to the
  nearest element station, to every label already placed, and to the
  hand-placed PZT leader, with candidates outside the panel heavily penalised
  and a mild preference for a short leader. The separation metric is
  **anisotropic** (`diag([0.35, 1])`) because a label is wide and short, so a
  given horizontal gap buys less than the same vertical one. The lens rig
  still takes the hand offsets — `auto = oap` — so its figure does not move.
- **The node and tail panels crop wider on the OAP rig** (150 / 130 mm of pad
  against 90 / 60), because the folded legs put the parts further apart.

**The parts list** is now printed by the runner at Stage B, for either optics
(`parts_list_`). For an off-axis section it gives the four numbers that specify
the *optic* rather than the layout — and they are not free: for a parabola fed
at conjugate `r`, parent focal length `= r·cos²AOI` and off-axis distance
`= r·sin 2·AOI`, so choosing the fold chooses both. The clear radius quoted is
the **traced footprint**, since `add_oap` leaves an off-axis section with no
declared aperture on purpose (§1.2).

From `oapdraw3_report.txt` — the test arm, station measured along the chief
from the source:

| element | type | station, mm | clear r | |
|---|---|---|---|---|
| Baffle | Obscuration | 428.6 | 21.4 | source baffle |
| PolIn | TrPolarizer | 438.6 | — | input polarizer, in the diverging leg |
| **L1** | Reflector | **882.1** | 51.4 | **off-axis parabola (collimator): fold 20°, parent f 756.9 mm (Kr −1513.8), off-axis 551.0 mm, conjugate 857.1 mm** |
| BSrefl | Reflector | 1139.3 | — | plate splitter, 22.5° |
| Comptxfd/bd | Refractor | 1336.5 / 1339.2 | — | compensator |
| QWPtestIn | WavePlate | 1561.6 | — | test-arm quarter-wave plate |
| TestOptic | Reflector | 1586.6 | 51.4 | the 96×96 DM |
| QWPtestOut | WavePlate | 1611.6 | — | the same plate, second pass |
| Recomb | Reference | 2186.6 | — | recombination plane |
| OutQWP | WavePlate | 2196.6 | — | output quarter-wave plate |
| Analyzer | TrPolarizer | 2206.6 | — | analyzer |
| **L2** | Reflector | **2311.6** | 51.4 | **off-axis parabola (focuser): fold 25°, parent f 352.0 mm (Kr −704.1), off-axis 328.3 mm, conjugate 428.6 mm** |
| FocalMask | Reference | 2740.1 | — | the mask seat |
| FLpow/flat | Refractor | 2750.9 / 2755.2 | — | field lens |
| Detector | FocalPlane | 2792.9 | — | camera at the pupil image |

Two readings fall out of the table and are worth stating plainly:

- **`L2` at 2311.6 and the mask seat at 2740.1 are 428.5 mm apart — `F2`
  exactly.** The mask-seat trim on this bench is **zero**, as §2.5 predicted.
- **`L1` sits 882.1 mm from the source, not 857.1** — `F1` plus the 25 mm
  `zSource`, which is `SRC_AT_FOCUS` doing its job: the pole has not moved, the
  source has.

Both mirrors carry the same coating (`bench.coat_oap`), and each needs a
tip/tilt mount; the runner prints that, and prints which leg the input
polarizer is in and whether the source is at the collimator's focus, so a
parts list can never silently describe a different bench from the one traced.

**Gate: the lens figure still does not move.** `lens22h` re-renders it with the
auto-placer, the wider crops and the re-ordered PZT leader in place. Its
decompressed pixel stream is **21 717 704 bytes, SHA-256
`9009d2d95df8737286c87557…` — byte-for-byte the value `lens22` and `lens22g`
carry.** So the lens rig's figure is unchanged across *both* item 1's station
fix and item 3's label work.

## 4. The interferometer on it — running

The tail had to be re-fit first: the record's `oap_tail.mat` was tuned on the
7-degree bench **with the 25 mm conjugate error in it**, so reusing it would
measure the old bench. `tg96_tail` on the designed geometry (tag `oap22d_tail`,
objective `sharpness`, model 512 / NGRID 193) — and its **seed already reads**:

| | flat-DM null | single-poke peak recovered (150 nm poke) |
|---|---|---|
| the record's *tuned* OAP tail (`oap_tail.mat`) | 12.887 nm | — |
| the lens rig's *tuned* tail (`lens_tail.mat`) | 0.134 nm | — |
| this bench's geometric SEED | **0.0289 nm** | 134.8 nm (90 %) |
| **this bench, TUNED** (`oap22d_tail.mat`) | **0.0223 nm** | **150.0 nm (100 %)** |

The designed reflective bench's *untuned* tail already beats the record's tuned
reflective tail by **446×** and the lens rig's tuned tail by **4.6×**; tuned, it
is **578×** and **6.0×**, and it recovers the single-actuator poke **in full**.
Winner: `FL_F` 38.0937, `FL_Kc` −2.64620, `D_MASK_FL` 1.9836, `DET_TRIM`
45.9613 (`runs/oap22d_tail.log`; the field lens moves a long way from the lens
rig's seed, which is what one would expect once the focuser is a parabola at
its exact conjugate rather than a tuned singlet).

**First row in, and it is the record's sharpest reflective-vs-lens claim
reversed.** `oapifo`'s Stage PLACE -- the D1 window-placement gate, response
centre-of-mass against the affine-predicted pixel for every lit actuator:

| | within 2 px | median error | blobs caught |
|---|---|---|---|
| lens rig (the record's control) | 100.00 % | 0.07 px | -- |
| **reflective rig, the record** | **72.77 %** | 0.16 px | -- |
| **reflective rig, as designed here** | **100.00 %** | **0.70 px** | **466 of 466** |

The record calls its 72.77 % *"physical, not a code artifact -- the single
off-axis OAP images the pupil with astigmatism, so ~20-25 % of actuators image
dark or elongated, worst at centre"*, and cites it as the same-plane-fold
effect. It is not: on the corrected bench every one of the 466 lit actuators
lands within 2 px, exactly as on the lens rig.

**Second row: the flat-DM null, at full resolution.** The record calls this
*"the same-plane-fold arm difference"* and it is the number the reflective
rig's differential has to cancel:

| | flat-DM null |
|---|---|
| reflective rig, the record (`runs/oap`, `runs/descent_oap`) | **12.893 nm / 13.089 nm** (12 893 / 13 089 pm) |
| lens rig (`runs/lens`) | 0.1345 nm (134.5 pm) |
| **reflective rig, as designed here** (`oapifo`, model 1024) | **0.0223 nm (22.3 pm)**, and the tail predicted 0.0223 |

**578× smaller than the record's reflective null, and 6.0× smaller than the
LENS rig's.** The "same-plane-fold arm difference" was not the fold: two
parabolas fed at their conjugates have almost nothing to differ about, and what
is left is six times better than a pair of tuned singlets. The record's D4
finding — that the 13 nm null is benign because it cancels in the differential
— is still true, but on this bench there is essentially no null to cancel.

**A consequence to carry into the realism round: the pupil image is nearly
twice as big.** The ray-affine magnification, DM-mm per detector-mm:

| | DM-mm / det-mm | pupil image across |
|---|---|---|
| lens rig | 9.879 | 9.7 mm |
| reflective rig, the record | 10.704 | 9.0 mm |
| **reflective rig, as designed here** | **5.477** | **17.5 mm** |

That follows from the retuned tail (`FL_F` 38.09 against the lens rig's 42.53,
`DET_TRIM` +45.96 against −1.25), not from the front end as such. It is good
for sampling — a 6.5 µm sCMOS lays **2692 pixels** across this pupil against
1446 on the lens rig — but **it does not fit the 2048 × 6.5 µm sensor**
(13.3 mm) that `BRIEF_ccmac_bench_realism` §4 names: 17.5 mm needs a larger
format, or the tail re-tuned with the image size as a constraint rather than a
free outcome. That is realism item 5's business (the camera) and is flagged
here, not resolved.

Also worth recording because it changes what a gate means: the D1 non-vacuity
check — "the best axis-aligned shear-free map fails where the affine
succeeds" — is **vacuous on this bench**. It reports 100.00 % for the
axis-aligned map against 100.00 % for the affine, because the mapping is now a
clean 0.00 % anamorphism at +0.00° off the DM axes. On the record's reflective
rig the same check was meaningful (the fold's flip plus rotation). A gate that
discriminates only when the thing it guards against is present is not evidence
here either way.

### 4.1 STOP — Stage C regresses, and it is like-for-like

**Not every row goes the designed bench's way.** The single-actuator gain in
actuator space, the measured matrix's own estimate of a 150 nm poke:

| | Stage C gain | off-target floor |
|---|---|---|
| lens rig (`runs/lens`) | 0.9968 | 98.8 pm |
| reflective rig, the record (`runs/oap`) | 0.9927 | 22.4 pm |
| **reflective rig, as designed here** (`oapifo`) | **0.0879** | **199.8 pm** |

Same stage, same line of the same runner, same 150 nm poke: the designed bench
recovers **8.8 %** of the actuator it is asked about. That is a regression, not
a win, and it is reported here before the rows that went the other way are read
as a verdict.

**What it is not.** Not window clipping: the calibration window is 37 px here
and 37 px on the record's reflective rig (41 on the lens), and the lit-actuator
count went *up*, 3388 → 3672, so the pupil is not being cut. Not placement:
D1 is 100.00 % within 2 px. Not the null: 22.3 pm here against 13 089 pm on the
record's bench.

**What it might be, untested:** the 150 nm poke sits at 0.95 of the four-step's
unambiguous range (`|h| < λ/4 = 158.2 nm`), so any change in the local phase
gradient can fold it; and the estimator is a 64-state matrix over 3672 lit
columns whose per-column support scales with the pupil magnification, which
this bench changed by 1.95×.

**Stages D and E are worse, and they settle the "artifact?" question: no.**

Modal transfer, gain and cross-talk by spatial frequency:

| mode | cyc/pup | lens gain | record OAP gain | **designed gain** | **designed cross-talk** |
|---|---|---|---|---|---|
| 1,1 | 0.7 | 0.9877 | 0.9493 | **0.1579** | 0.3649 |
| 4,4 | 2.8 | 0.9884 | 0.6381 | **0.1078** | 0.3455 |
| 16,16 | 11.3 | 0.9886 | 0.7789 | **0.1182** | 0.4551 |
| 32,32 | 22.6 | 0.9889 | 0.7681 | **0.1025** | 0.4625 |
| 64,64 | 45.3 | — | — | **0.1465** | 0.4104 |
| 96,96 | 67.9 | — | — | **0.0292** | 0.1483 |

Differential rows (actuator space):

| base | deviation | **designed** gain / resid pm / corr | record OAP gain / resid / corr |
|---|---|---|---|
| flat | single 10 nm | **−0.4079 / 304.1 / −0.3243** | 0.9948 / 2.2 / 0.9999 |
| flat | random 10 nm | **0.1044 / 10569.0 / 0.1757** | 0.7486 / 4848.5 / 0.8706 |
| random 16 nm | single 10 nm | **0.5414 / 374.0 / 0.2367** | 0.9958 / 2.3 / 0.9999 |
| random 16 nm | random 10 nm | **0.0987 / 11605.5 / 0.1293** | 0.7486 / 4848.3 / 0.8706 |

**The reading is not attenuated, it is broken.** The gains are erratic
(−0.41, 0.10, 0.54, 0.10) rather than a constant factor, one is NEGATIVE, and
the correlations are 0.13–0.24 with one at −0.32. That is a reading carrying
almost no information about the surface it is asked about.

**The hypothesis I formed first — that this is the input polarizer's
relocation — is WEAKENED by evidence already on disk, and must not be
asserted.** `tg96_tail` drives the *same* four-step machinery
(`analyzer_basis` → `fourstep` → `meas_surface`) on the *same* geometry with
the *same* `POL_IN 'source'`, and it returned a 0.0223 nm null and recovered
**150.0 of a 150 nm poke** — gain 1.00. If the polarization state at the
splitter were wrong, that could not have happened.

What differs between the tuner and the battery is **resolution**: the tail
tuned at model 512 / NGRID 193; `oapifo` runs model 1024 / NGRID 385. Combined
with §4's measured 1.95× change in pupil magnification, the live hypothesis is
now a **sampling or registration failure at the full-resolution detector
grid**, not the polarization chain.

**The discriminating run is already queued** (`runs/polabseq.sh`, tags `polA` /
`polB`): the designed geometry at **model 512 / NGRID 193**, once with
`POL_IN 'source'` and once with `'collimated'`.
- If `polA` is clean at 512 — as the tail tuner suggests — the polarizer is
  exonerated and the defect is resolution-dependent; the next experiment is
  `oapifo`'s settings at 512 vs 1024 with everything else pinned.
- If `polA` is broken at 512 too, the polarizer returns as a suspect and `polB`
  separates it (at the cost of ~11 % of its rays into the plate, so `polB` is a
  diagnostic, never a candidate design).

### 4.2 The runner diagnoses it: the map WRAPS, and the tail is the suspect

The break ladder — the single 10 nm differential against an increasing base
surface — names the failure outright:

| base rms | lens rig | record reflective | **designed bench** |
|---|---|---|---|
| 30 nm | 1.0013 | 0.9967 | **0.2816 — BROKE (wrap: base reads 1.00 of λ/4)** |
| 60 nm | 1.0103 | 0.9987 | **0.2437 — BROKE** |
| 120 nm | 1.0258 | 0.9617 | **−0.9554 — BROKE** |
| 240 nm | 1.9271 | 0.4779 — BROKE | 0.5738 |
| 480 nm | 1.5849 | 1.0192 | **0.1995 — BROKE** |

The record's rigs read a 30 nm base cleanly and wrap only at 240 nm. **This
bench wraps at the FIRST rung**: a 30 nm base already saturates the four-step's
unambiguous range, `|h| < λ/4 = 158.2 nm` — about 8× too easily, which is the
size of the gain deficit.

And the regularization sweep shows *where* the information went:

| matrix λ | | all | bright 25 % | dark 25 % |
|---|---|---|---|---|
| lens rig | 1e-03 | 0.9893 | 0.9890 | 0.9906 |
| record reflective | 1e-03 | 0.7486 | **0.9932** | 0.0494 |
| **designed bench** | 1e-03 | 0.1044 | **−0.0048** | **0.4739** |

On both working rigs the **bright** (well-illuminated) columns carry the signal
and the dark ones do not. Here it is **inverted**: the bright columns read
essentially zero and the dark ones carry what little there is. That is not a
gauge reading a surface badly; it is a map that is not a pupil image of the DM.

**Leading hypothesis: the tail retune, and specifically its objective.**
`tg96_tail`'s `'oap'` objective is `sharpness` — the recovered single-poke peak
— which rewards a sharp poke image but **does not constrain the detector to the
DM's pupil conjugate**. Its winner moved `DET_TRIM` to **+45.96 mm** (the lens
rig's is −1.25) and §4 already measured the pupil magnification changing
**1.95×**. A detector off the conjugate carries field curvature into the
measured map: that saturates λ/4, scrambles which actuator owns which pixel,
and leaves exactly these erratic sign-flipping rows — while leaving the *null*
tiny (an arm difference, common-mode, it cancels) and the *poke peak* sharp.
**Which is why the tuner reported 0.0223 nm and 150.0 / 150 nm and looked like
a triumph.** §4's opening rows are consistent with a tail that is optically
sharp and metrologically wrong.

**The A/B, queued** (`runs/tailabseq.sh`): `tailA` = the design with the tuned
tail, `tailB` = the same with the **geometric seed** (`bench.tail_from_mat`
false), both at model 512 / NGRID 193. If `tailB` reads and `tailA` does not,
the defect is the tail objective and the fix is to constrain the retune on the
pupil conjugate — not to change the geometry, which items 1–3 measured
independently of any tail.

`bench.tail_from_mat` is new and closes a real gap: the tail lookup falls back
to `<optics>_tail.mat`, so simply not writing a per-tag mat picks up **another
bench's** tail. Seed-vs-tuned was previously unrunnable.

**And the objective has a readable weakness, independent of what the A/B
says.** `tg96_tail`'s sharpness cost is

```
peak_nm = 1e6*max(abs(hp(msk)));      frac = peak_nm / poke_nm;
r = (1 - min(frac,1.2))^2 + (null_nm/2.0)^2;
```

`max(abs(hp(msk)))` is the largest value **anywhere in the pupil**. Nothing ties
it to the actuator that was poked, and nothing asks whether the response is
*localized*. A map that is defocused, wrapped, or mis-registered can carry a
large maximum somewhere and score as "sharp"; a wrapped map in particular is
*guaranteed* a large maximum, since wrapping throws values to the ends of the
λ/4 range. So the cost is maximized, not merely tolerated, by exactly the
failure §4.2 measures — and the reported `poke-peak 150.0 nm` (frac = 1.00) is
what that looks like from inside the tuner.

The fix this implies — reward a response that is **localized** as well as tall
(peak plus the fraction of |h| energy in the blob around it), and refuse any
candidate whose map approaches λ/4 — is **not being landed until the A/B
confirms the tail is the cause.** A speculative fix to an objective, applied
before the diagnosis closes, is how the first wrong attribution got written
down.

**Until this is settled: §7's deck guidance stands with this added — do not put
the reflective rig's reading performance on a slide in either direction, and do
not read §4's geometric rows (D1, the null, the seat, the tail null) as a
verdict on the instrument. They are optical measurements and they are sound;
they do not license a claim about the gauge, and at least one of them (the tail
null) is now suspected of being sharp for the wrong reason.**

The rows, the servo and the descent follow it, queued in `runs/ifoseq.sh`:
`oapifo` (bench + battery + figs + clearance — the rows on the 30 nm surface
with the matrix measured on that surface) and `oapifol` (bench + loop + figs —
the closed-loop hold metric, hour-class). Each copies the retuned
`oap22d_tail.mat` under its own tag first, so neither can silently fall back to
the 7-degree tail.

## 5. The mask sensors on it — the precondition is measured, the runs are next

CCMac's record has the reflective rig **breaking** the two focus-critical mask
readings: the vector dimple at 19.6 pm against a 12 pm gate (G4 FAIL) and the
pinhole at 94 pm (G5 FAIL), on a marginal seat focus — "the more focus-critical
the mask feature, the worse the fold coma".

§2.5 says there is no fold coma, and the ZWFS seat is where that claim has to
pay off. `oap_focus_probe` (extended to take the design and the conjugate
switch; tags `zseat`, `zseat2`) scans `MASK_TRIM` for the best ray blur at the
seat on the **ZWFS** rig, at the design point:

| OAP1 / OAP2 fold | best blur | best `MASK_TRIM` |
|---|---|---|
| 20° / 20° | **0.00 µm = 0.000 λF/D** | **−0.00 mm** |
| 25° / 25° | **0.00 µm = 0.000 λF/D** | **−0.00 mm** |
| **20° / 25° (the design)** | **0.00 µm = 0.000 λF/D** | **−0.00 mm** |

against the record's 2.94 µm (1.1 λF/D) at `MASK_TRIM` 6.14, right on the
gauge's 0.01 peak/sum gate. **The reflective rig's mask seat is diffraction-
perfect at zero trim.** So the sensors' runs are not a re-measurement of a
marginal focus; they are a measurement of a good one, and G4 / G5 should have
no focus reason to fail. The runs themselves are the next item.

**Note for the runs:** `zwfs_params` carries `MASK_TRIM = −5.582`, which is the
*lens* rig's seed-to-focus correction. On this bench it must be **0** — pass
`'bench.MASK_TRIM',0` along with the design, or the sensors will be seated
5.6 mm from focus.

## 6. The fold-angle lever — reframed by §2.5, pending item 5

The brief asks for the same battery at half OAP2's angle, on the reasoning that
astigmatism scales as the fold angle squared, so halving it buys a factor of 4
at the price of a 1.33× longer leg. **That trade does not exist on this bench.**
The blur is linear in the angle, not quadratic (0.071 λF/D per degree,
`conj`), and it is not the fold at all — at the design point the seat blur is
**0.000 λF/D at 20° and at 25°**, so there is nothing for a smaller angle to
recover. The lever that mattered was the conjugate, and it has been pulled.

The empirical half of the item — "does the pinhole recover?" — is answered by
item 5's P reading on this bench, not by a second fold angle. It is left open
until that run lands.

## 7. For the deck: what the two slides should say

The brief builds two slides out of this report -- the existing "lenses or
off-axis mirrors" table and a new "reflective rig: layout". CCL assembles;
this section is the input, and it is deliberately blunt about which of the
record's claims survive.

### "Lenses or off-axis mirrors" -- the table needs rewriting, not updating

The record's version of this slide compares a lens rig against a reflective rig
that was **fed 25 mm inside its collimator's focus** and whose input polarizer
was **inside the mirror**. Those are not properties of choosing mirrors over
lenses; they are two build errors, and both are now fixed. Every reflective row
in `REPORT_oap.md` -- the 0.18 modal cross-talk against the lens's < 0.06, the
servo that never reaches the 2 pm walk, the 27.6 % noiseless step floor, the
19.6 pm vector and 94 pm pinhole gate failures, the 6.14 mm seat trim -- sits
downstream of a 1.1 lambda F/D blur at the mask seat that need not exist.
**Do not carry those numbers onto a slide about mirrors.** They are being
re-measured (`oapifo`, `oapifol`, `oapsens22`); until they land, the slide can
say what is already settled:

| | lens rig | reflective rig, as designed here |
|---|---|---|
| buildable at the 22.5 deg node | yes, worst +38.2 mm over 10 parts | **yes, worst +33.4 mm over 8 parts** |
| mask-seat blur at best focus | diffraction-limited | **0.000 lambda F/D** |
| mask-seat trim | -5.582 mm (thin-lens seed correction) | **0.00 mm** (the parabola's exact conjugate) |
| tail flat-DM null, tuned | 0.134 nm | **0.0223 nm** |
| single 150 nm poke recovered | -- | **150.0 nm (100 %)** |
| the optics | two tuned singlets, conic figures fit per rig | two off-axis parabolas: **f 756.9 / 352.0 mm parent, 551.0 / 328.3 mm off-axis, 40 and 50 deg off-axis catalogue shapes** |
| chromatic | no (a tuned singlet at one wavelength) | **achromatic by construction** |
| what it cost | -- | the input polarizer moves into the source leg; the output optics move 70 mm; the folds are 20 and 25 deg, not 5 and 9 |

The honest headline is not "mirrors are as good as lenses" and not the
record's "the lens is the recommended configuration" either. It is: **the
reflective front end was never given a fair test. Fed at its focus and laid
out so its parts clear its beams, its tail is 6x better than the lens rig's and
it recovers a single actuator in full.**

### "Reflective rig: layout"

Figure: `runs/oap22d/oap22d_vlayout.png` -- three panels (train, node, tail),
both arms, the mirrors drawn where the beam hits them. It is the runner's own
output file; do not re-render or re-colour it (Dave's rule). The parts list for
the slide's callouts is section 3's table.

The one sentence the layout slide needs: **the source enters from outside the
node and the tail leaves away from it** -- that is what the 20 and 25 degree
folds buy, and it is why the earlier 5 and 9 degree version put the source
587 mm past the splitter with its beam running back through every part of the
node.
