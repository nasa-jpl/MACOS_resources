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
| 2 — the design: fold angles and off-axis distances from a clearance solve | not started |
| 3 — the layout in the recipe; the parts list | not started |
| 4 — the interferometer on it (rows on the 30 nm surface, servo, descent) | not started |
| 5 — the mask sensors on it (S / V / P, bench + battery, stations figures) | not started |
| 6 — the fold-angle lever: half OAP2's angle; does the pinhole recover? | not started |
| 7 — the P/SRI bench through the clearance tool | not started |

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
