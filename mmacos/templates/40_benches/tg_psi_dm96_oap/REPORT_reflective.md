# REPORT — the reflective front end, designed on the 22.5° bench

TO for Dave, 2026-09-15. Branch `dev-candidate`. `BRIEF_to_reflective.md`: the
same TG96 bench as the lens rig with the two lenses replaced by off-axis
parabola sections, on the node angle Dave ruled. Numbers first; every claim
carries its run tag. Companions: `REPORT_oap.md` (CCMac's 7° history, kept),
`REPORT_bench_realism.md` (the node round), `REPORT_gauge_ifo.md` (the lanes).

## Status — `BRIEF_to_gauge_close` (live)

The close-out brief's seven items. CCL folds each landed item into the deck;
Dave pushes. The realism and field-servo reports link back to this table.

| item | what | report | state |
|---|---|---|---|
| 0 | commit the untracked record files | — | **done** — the tag census below |
| 1 | vector pair on the redesign: rows, overcoat, verdict | §5.1 | **done** — rows hold uncalibrated; G4 634 bare / 319 quarter-wave overcoat / 199 bench-calibrated / **0.054 pm PASS** polarimetric. The variable is the channel PHASE, not the amplitude; fold lever stays unpulled. `vqw22` `vmap22` `vfit22` `vamp22` |
| 2 | item 4's loop + descent + the 120 nm wrap explained | 4.7, **4.7(a)** | **DONE** -- (a) closed 2026-09-16: the break is the BASE READING WRAPPING and BOTH rigs do it between 60 and 120 nm, saturating at the analytic 316.4/sqrt(12) = **91.34 nm**; "the reflective rig has a smaller capture range" comes OFF the deck. The ladder's dA-vs-dD arithmetic is innocent (n_cross <= 5 px in 1e5), but each crossing is a full lambda/2 and swamps any second moment -- read n_cross, never corr. Servo: **1.7e13 photons/cycle for 3 pm under the 2 pm walk**, thermal floors at 10.0 pm and is LOW-ORDER (9.24 of it below 4 cyc/ap). Descent: both starts converged (r(K) 2.344 / 2.345 pm, rho 0.502 / 0.515, 0 recals); its exit 1 was `draw_loop_` on an empty drift list AFTER the results, now guarded. `oapuw2` `lensuw2` `oapifol2` `oapdesc2` `wrapoap` `wraplens` |
| 3 | tail tuner gated by a battery row; README | README + 4.5 + **4.8** | **MEASURE FIXED, STILL ADVISORY.** The point-sample measure INVERTED the verdict (it tracked magnification); the lattice measure (`dmg_act_fit` over the illuminated lattice, stencil from tg96_place's anchor poke, `score_`'s gain verbatim) no longer does -- objwin3 **-0.1621**, lens_tail **0.8074**, thk22_tail **0.9104** against battery 0.0338 / 0.9968 / 0.9885. Ordering right, separation wide, but the SCALE is not the battery's, so 0.95 still refuses two good tails and enforcement stays OFF. **Not the regularizer:** the act_lam sweep is flat (1.1 / 0.9 / 1.4 % from 0.05 to 0.002). Live hypothesis = the single-site stencil against the battery's per-actuator matrix; recommended fix = gate the winner against the SEED through the same estimator (scale-free), NOT a lower threshold. `gate3_win` `gate3_lens` `gate3_thk` |
| 4 | realism 3-5: thicknesses, substrates, camera | `REPORT_bench_realism.md` **3.1, 3.2** | **THICKNESSES DONE on the tuned tail** (`item4bseq`, 2026-09-16): retune reproduced 75.2422 -> **20.0938 nm** (3.7x) and the advisory gate KEPT it, so thk22's rows now describe a 20 nm-null bench. Stage C **0.9885**, floor 40.0 pm; modal gain ~1.00 to 45 cyc/pup (0.9626 at 67.9), cross-talk <=0.027; differential rows 1.002-1.009, corr >=0.9999; ladder holds to 60 nm (floor 9.3 pm) and breaks at 120 -- the SAME rung both rigs break at, so the glass has not moved the capture range. **NEW:** the 10 mm plates read **13% LOW** (26.6/30, 52.1/60, saturating at 79.6 against the analytic 91.34 -- one common factor), invisible to the old saturating meter; the matrix absorbs it, so it costs SNR and floor, not gain. `thk22_tail` `thk22` |
| 5 | realism 6: snapshot polarization at the built angles | `REPORT_bench_realism.md` | **tool in**, queued (`runs/aoiseq.sh`): `tg_aoi_ladder` gained the OAP rig and two columns -- the analyzer-sweep CORRECTION (free: the basis already spans every angle) and the RESIDUAL a measured matrix cannot absorb |
| 6 | realism 8: the interferometer's station figure, both rigs | `REPORT_bench_realism.md` | **FIGURES DONE, one OPEN defect** -- both rigs at the width the brief asks for: `stnoap` / `stnlens`, **1800 x 560 px** (the pre-patch `oapifol2` figure was 2558 x 838; `exportgraphics` at Resolution 150 does not land at the figure's pixel width, `print -dpng -r96` on this 96 dpi box does, and it is the ZWFS sibling's own mechanism, which the deck holds beside it).  OAP leg bit-identical to pre-patch `oapifol2` (626.43 pm), so the patch is inert here.  **OPEN:** the lens leg's station residual is 62 067 pm against the OAP's 626 -- zero at flat, sqrt(2)x the map with structure, i.e. a LATERAL MISREGISTRATION between the recovered map and the engine field.  First lens station figure ever made, so not a regression.  `REPORT_bench_realism.md` section 6; do NOT put the two numbers on one slide yet |
| 7 | the coronagraph field servo, three steps | `bench_ctb/REPORT_field_servo.md` | **step 0 open and probed** -- the note's 33-cycle separability prediction pairs a 0.67 mm pitch with a 42.8 mm beam, and 32 x 0.67 = 21.4; the CTB documents contradict each other on radius vs diameter, so `ctb_beam_probe.m` asks the engine. Steps 1-3 not started |

**Reading order.** Sections are appended in DISCOVERY order, not numeric order,
as this report has been maintained throughout — §4.5 and §4.6 already sat after
§7 before this round. The close-out's sections are §5.1 (item 1) and §4.7 with
its four subsections (item 2); item 3 lives in the README ("The tail of record,
and the tuner's open problem") with the evidence it rests on in §4.5, and items
4-6 in `REPORT_bench_realism.md`. The table above is the
index; within item 2 the subsections are in the order the measurements landed,
and one earlier paragraph is marked superseded rather than deleted because the
reasoning it sets up is what the control then excluded.

### Item 0 — what was committed, and what was deleted

Every run tag the three reports (`REPORT_reflective.md`,
`REPORT_bench_realism.md`, `pdi_dm96/REPORT_gauge_pdi.md`) cite now has its run
directory committed: `clear22`, `lens22h`, `node22t`, `node22v`, `nodesolve`,
`oapdraw3`, `oapifo`, `oapifo2`, `tailA`, `tailB`, `oapsens22`, `oapsens22n`,
`psriclear2`, plus the `oap22d` re-render (the auto-placed one — its
`_report.txt` carries the `parts_list_` block §3 quotes, and the emitted deck is
byte-identical, so the re-render changed the drawing and the printout, not the
bench). The tail mats §4 cites are committed as evidence: `tailA`, `objseed3`,
`objwin3`, `oapifo`, `oapifol`.

Deleted, because nothing cites them and every one is regenerable from a
committed script: run dirs `foldsmoke`, `node22s`, `oapdraw`, `oapdraw2`,
`oapfixb` (`runs/verifyseq.sh`), the aborted `oapifol` stub (killed at Stage
PLACE, no result in it — item 2 re-runs it), `pdi_dm96`'s `psriclear` and
`pfdeck_smoke` (superseded by `psriclear2` / `pfdeck_smoke2`); the seven tail
mats §4 does not cite; all `runs/*.nohup` stdout captures (the `.log` carries
the exit code, and neither is tracked in this tree); every `*_sketch*.png` (no
report shows one); the tuner's scratch decks `tail_{ref,test}.in` and
`zwfs_flat.txt`. An uncommitted artifact is an unverifiable claim; a committed
artifact nothing cites is noise.

## Status — `BRIEF_to_reflective` (closed)

| item | state |
|---|---|
| 1 — the drawing defect diagnosed and fixed; the lens rig unchanged | **done** — §1; `lens22g` (gate, pixel-identical), `oap22` |
| 2 — the design: fold angles and off-axis distances from a clearance solve | **done** — §2; the design is OAP1 20° / OAP2 25°, sides +1/−1, polarizer in the source leg, output optics 125 mm ahead of OAP2, collimator at its focus: **worst +33.4 mm over 8 parts**, no ray loss. Tags `fold1`–`fold4`, `loss`, `loss_src`, `loss_a2`, `conj`, `oap22d`; tail retune `oap22d_tail` |
| 3 — the layout in the recipe; the parts list | **done** — §3; auto-placed labels on the OAP rig (lens figure untouched), parts list printed by the runner (`oap22d`, `oapdraw3`) |
| 4 — the interferometer on it (rows on the 30 nm surface, servo, descent) | **defect found and RESOLVED to its cause**, §4.1–4.3. The reading broke (gain 0.03, erratic, wrapping); a controlled A/B shows it was **my tail retune**, not the geometry: the same bench with the geometric seed tail reads **0.9809**. Fix the tail objective, then re-run. `oapifol` / `oapdesc` deferred until then |
| 5 — the mask sensors on it (S / V / P, bench + battery, stations figures) | **done**, §5 (`oapsens22`, `oapsens22n`). Seat 0.000 λF/D at trim 0; all sandwich/reference gates 1e-15. **Pinhole RECOVERS: 94 pm FAIL → 0.269 pm PASS.** Vector pair worsens 19.6 → 634 pm; coating accounts for 3.0× of it (208 pm with `coat none`), the fold angles for the rest |
| 6 — the fold-angle lever: half OAP2's angle; does the pinhole recover? | **answered**, §6 + §5: **yes the pinhole recovers** — but by fixing the CONJUGATE, not by opening the fold (blur is 0.000 λF/D at both 20° and 25°, so the lever the item assumes does not exist). The fold angles' real cost is the VECTOR pair, split 3:10 coating:geometry |
| 7 — the P/SRI bench through the clearance tool | **done** — recorded in `pdi_dm96/REPORT_gauge_pdi.md` (its own brief): PASS at 22.5° (+36.9 mm), FAIL at 7°; the Mach-Zehnder node clear by 227–383 mm at both. Runner `psri_clearance.m`, tag `psriclear2` |

## What is running, and how to pick it up

**THE CLOSE-OUT QUEUE — DRAINED 2026-09-16 09:19.** Every job below finished
with exit 0. Kept as the record of what produced what.

| # | job | script | state |
|---|---|---|---|
| 1 | `oapuw2`, `lensuw2` | `runs/item2seq.sh` | done — item 2(b), §4.7 |
| 2 | `oapifol2` (the servo, 14 loop runs) | `runs/item2seq.sh` | done — §4.7 |
| 3 | `oapdesc2` (the descent, 2 starts) | `runs/item2dseq.sh` | done — both starts converged (r(K) 2.344 / 2.345 pm, ρ 0.502 / 0.515, 0 recals). Its `exit 1` was `draw_loop_` on an empty drift list AFTER the results; guarded since |
| 4 | `gate3_win`, `gate3_lens` | `runs/gateseq3.sh` | superseded by the 3-leg `closefinal4.sh` (§4.8) |
| 5 | `aoi_lens22`, `aoi_oap22` | `runs/aoiseq.sh` | done — item 5 |
| 6 | item 4's six runs | `runs/item4seq.sh` | done |
| 7 | `ctb_beam_probe` | `closefinal5` | done — item 7 step 0, the retraction |
| 8 | **the staged patch** | `runs/apply_tg96_pending.py` | **APPLIED 2026-09-16**, `7aac1d8` |
| 9 | `wrapoap`, `wraplens` | `runs/item2bseq.sh` ← `closefinal2` | done — **item 2(a) closed**, §4.7(a) |
| 10 | `stnoap`, `stnlens` | `runs/item2bseq.sh` ← `closefinal2` | done — item 6's figures at 1800 px |
| 11 | `thk22_tail`, `thk22` | `runs/item4bseq.sh` ← `closefinal2` | done — item 4's rows on the TUNED tail |
| 12 | `gate3_win`, `gate3_lens`, `gate3_thk` | `runs/closefinal4.sh` | done — item 3's measure, §4.8 |

`closefinal2.sh` chained steps 2 and 3 gating on each wrapper's own `] exit`
marker; `closefinal4.sh` added the third gate leg (the battery calibration) and
ran all three after `thk22`.

**THE ONE MANUAL STEP — DONE 2026-09-16 (`7aac1d8`).** `runs/apply_tg96_pending.py` (with `stage_wrap.m.txt`
beside it) patches `tg96_run.m`, `tg96_params.m` and `tg96_tail.m` — the
saturating ladder meter, the `wrap` stage, the camera line, the dropped
`MASK_SUB`, the stations figure's width. It is NOT applied automatically because
editing a file an hour-class run is executing is how a long job gets corrupted.
**When the window opens:** the patch's three targets are in use for the whole
rest of the queue — `gateseq3` and `item4seq` both run `tg96_tail`, `item4seq`
runs `tg96_run` — so the first safe moment is **after `item4seq` and the CTB
probe**, which is exactly where `closefinal.sh` stops and prints the command.
Run it from `runs/`, then `runs/item2bseq.sh`. It has been dry-run against
copies of all three files (applies clean, 0 parse issues).

It is deliberately NOT applied by the chain. Editing source inside an
unattended queue is a worse failure mode than an item left open with a written
instruction: if the patch went wrong at 3 a.m. it would take the runs after it
with no one reading the error.

**Sequencers were stopped, never edited, when they needed changing** — bash
reads a script incrementally and remembers its byte offset. Killing a sequencer
leaves its running child alive and orphaned, which is why `item2dseq.sh`
supersedes `item2seq.sh`'s remainder and writes `item2seq`'s own done-marker so
`closechain4` proceeds.

### The previous brief's jobs (all finished)

| job | script | what it produces | state |
|---|---|---|---|
| `oap22d_tail` | `runs/tailseq.sh` | `oap22d_tail.mat` — the reflective tail re-fit on the designed geometry | **done**: null 0.0223 nm, poke recovered 150.0 of 150 nm |
| `oapifo` | `runs/ifoseq.sh` | the interferometer's rows on the 30 nm surface + the clearance table | **done** (exit 0); superseded at record resolution by `oapifo2`, §4.6 |
| `oapifol` | `runs/ifoseq.sh` | the closed-loop hold metric (hour-class) | **re-queued as close-out item 2** on the geometric seed tail; the first attempt was killed at Stage PLACE and its stub deleted (item 0) |
| `oapdesc` | `runs/descseq.sh` | item 4's descent: the same 60 / 150 / 300 nm ladder the record's `descent_oap` stalls on (5856 / 23557 / 53150 pm, never reaching 10 nm from 150 or 300) | **re-queued as close-out item 2**, on the seed tail |
| `tailA` / `tailB` | `runs/tailabseq.sh` | **the decisive diagnostic**: the design at model 512 with the TUNED tail vs the GEOMETRIC SEED (§4.2) | **done** (both exit 0): 0.0338 vs 0.9809, §4.5 |
| `oapsens22` | `zwfs_dm96/runs/oapsensseq.sh` | item 5's mask sensors — **independent of the tg96 tail** (the ZWFS carries its own tuned tail and `MASK_TRIM` 0), so it is unaffected by §4.2 and still worth its run | **done** (exit 0), §5; discriminator `oapsens22n` too |
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

**The A/B** (`runs/tailabseq.sh`): `tailA` = the design with the tuned tail,
`tailB` = the same with the **geometric seed** (`bench.tail_from_mat` false),
both at model 512 / NGRID 193.

**`tailA` is in, and it settles two things at once.** Stage C gain
**0.0338**, off-target floor 467.9 pm, at model 512 / NGRID 193 — the same
pupil magnification (5.477) and the same 3672 lit actuators as the
full-resolution run.

1. **The resolution hypothesis is dead.** The failure reproduces at 512 / 193.
   It is not a full-resolution sampling effect.
2. **The tuner and the battery disagree about the SAME bench at the SAME
   resolution** — `tg96_tail` scored this configuration `poke-peak 150.0 / 150`
   (frac 1.00) and the battery reads 0.0338 of the same 150 nm poke. Two
   measurements of one thing, differing by 30×. That is not a bench property;
   it is the two metrics measuring different things, which is exactly the
   `max(abs(h))`-anywhere weakness read off the code above.

### 4.3 RESOLVED: the tail retune was the defect. The geometry reads.

`tailB` — the **identical** bench with the **geometric seed** tail — against
`tailA`, the same bench with the tuned tail. One variable.

| | `tailA` (tuned tail) | **`tailB` (seed tail)** | lens rig | record reflective |
|---|---|---|---|---|
| Stage C single-actuator gain | **0.0338** | **0.9809** | 0.9968 | 0.9927 |
| off-target floor | 467.9 pm | **53.8 pm** | 98.8 pm | 22.4 pm |
| **DM-mm / det-mm** | **5.477** | **10.437** | 9.879 | 10.704 |
| lit actuators | 3672 | 3680 | 3228 | 3388 |

**The designed reflective geometry reads at 0.98 — comparable to the lens rig's
0.997 and the record's reflective 0.993. The defect was entirely my tail
retune.**

And the magnification column is the mechanism, measured rather than argued:
the seed tail puts the detector at **10.437** DM-mm per detector-mm, in family
with both working rigs (9.879 and 10.704); the tuned tail put it at **5.477**,
nearly half. The retune walked the detector off the DM's pupil conjugate — the
pupil image doubling from 9 to 17.5 mm was the visible symptom I flagged in §4
and mis-filed as a packaging consequence — and a detector off the conjugate
cannot read actuators.

**And with the seed tail the bench does not merely read — it reads better than
the record's reflective rig on exactly the rows the record calls irreducible.**
`tailB`, Stage E:

| base → deviation | **`tailB` (designed + seed tail)** | record reflective | lens rig |
|---|---|---|---|
| flat → single 10 nm | 0.9900 / 3.0 pm / 0.9999 | 0.9948 / 2.2 / 0.9999 | 0.9916 / 2.2 |
| **flat → random 10 nm** | **0.9890 / 202.9 pm / 0.9998** | **0.7486 / 4848.5 / 0.8706** | 0.9893 |
| random 16 nm → single | 0.9937 / 2.7 / 0.9999 | 0.9958 / 2.3 / 0.9999 | — |
| **random 16 nm → random** | **0.9878 / 228.7 pm / 0.9998** | **0.7486 / 4848.3 / 0.8706** | — |

Break ladder — clean at **every** rung:

| base rms | 30 | 60 | 120 | 240 | 480 nm |
|---|---|---|---|---|---|
| **`tailB`** | 0.9951 / 2.5 pm | 0.9923 / 2.5 | 0.9702 / 5.8 | **0.9552 / 9.1** | **0.9525 / 10.9** |
| record reflective | 0.9967 / 2.5 | 0.9987 / 3.1 | 0.9617 / 489.6 | 0.4779 — BROKE | 1.0192 / 15.3 |
| lens rig | 1.0013 / 4.5 | 1.0103 / 9.3 | 1.0258 / 19.3 | 1.9271 / 394.0 | 1.5849 / 419.1 |

The dense-random rows are the headline. The record's reflective rig reads them
at **0.7486 with a 4848 pm residual** and calls the gap *"the same-plane fold's
astigmatism cross-talk … only the geometry can move it"* — a property of
choosing mirrors. The designed bench reads the same rows at **0.989 with a
203 pm residual**, a **24× smaller residual**, matching the lens rig. And at the
deep end of the ladder (240 and 480 nm) it holds 0.955 / 0.953 at single-digit
picometres where the record's reflective rig breaks and the **lens rig** runs
away to 1.93 and 1.58 with ~400 pm residuals.

**CAVEAT, and it is not a small one: these are NOT resolution-matched.**
`tailB` is model 512 / NGRID 193; every record number in the tables above is
model 1024 / NGRID 385. The comparison is indicative, not decided. Claiming
parity with the lens rig — let alone superiority — requires the
full-resolution re-run, which is queued as `oapifo2` (the design, seed tail,
model 1024 / NGRID 385). **Nothing from this block goes on a slide until that
lands.** Making a cross-configuration claim from a resolution-mismatched pair
is the same error class as the two attributions already retracted in §4.1–4.2.

**What this vindicates and what it does not.** Items 1–3 are untouched: they
were measured with no tail in the loop, and `tailB` now shows the geometry they
produced reads properly. §4's opening rows — D1 at 100 %, the 22.3 pm flat-DM
null — were measured *through* the broken tail and must be re-taken on the
fixed one before they mean anything. The tail null of 0.0223 nm in particular
is now explained: a detector off the conjugate still nulls two arms that share
the same wrong tail.

### 4.4 The fix, and the gate that caught the first version of it

`tg96_tail`'s `'oap'` objective now reads

```
r = (1 - min(frac,1.2))^2 + 4*(1 - conc)^2 + 10*max(0, wrapf - 0.8)^2 + (null_nm/200)^2
```

with `conc` = the fraction of the map's energy within ~3 actuator pitches of
its own peak (mapping-free: the pupil diameter comes from the mask and one
actuator is 1/nact of it, so it needs no DM→detector affine — the very thing a
bad tail corrupts), and `wrapf` = how close the map runs to λ/4. The tuning
poke drops **150 → 100 nm**.

**The first version of this fix FAILED its gate, and the failure found the real
driver.** Evaluating both parameter sets under it:

| | `conc` | wrap | null | cost, **v1** |
|---|---|---|---|---|
| geometric seed — reads at 0.99 | 1.000 | 0.93 | 71.07 nm | **1262.9** |
| old winner — reads at 0.03 | 0.006 | 1.00 | 0.0223 nm | **1.39** |

v1 still *preferred the broken tail*, because it kept the old `(null_nm/2)^2`
term — and **that term is the defect's driver.** Minimizing the null is free
for a misplaced detector: both arms share the tail, so a common misplacement
cancels in an arm *difference* while being fatal to the reading. The optimizer
bought a 0.0223 nm null by walking off the pupil conjugate, and the cost
thanked it. A **71 nm null reads perfectly well** (`tailB`, gain 0.99), so the
null is now *bounded* at a 200 nm scale rather than minimized.

The 150 nm poke was the second half of the problem: at 0.95 of λ/4 a **healthy**
map already reads 0.93 on the wrap meter, so the guard could not separate
health from saturation. At 100 nm (0.63 of the range) it separates cleanly.

**RETRACTED — both gate tables below are CONTAMINATED, and so is the v1 table
above.** `tg96_tail` wrote **fixed** scratch filenames (`tail_flat.txt`,
`tail_test.in`, `tail_ref.in`) into the template directory, and I ran each gate
pair **concurrently** — objseed/objwin both finished at 10:52:48, objseed2 and
objwin2 at 11:29:11 and 11:29:28. Two processes were reading and writing each
other's decks. The tell was immediate once the re-tune ran alone: the
**identical** seed parameters that the contaminated probe scored at a 71.80 nm
null score **0.0289 nm** when nothing else is running — and 0.0289 is the value
the original tune recorded for the same seed. The numbers below are kept only
so the retraction is checkable; **they are not evidence for or against the
fix**, which must be re-gated sequentially.

`tg96_tail` now builds per-process scratch names
(`tail_<pid>_<tag>_{flat,test,ref}`) and deletes them on exit —
`dmg_bench_clearance` took exactly this fix on 2026-09-15 (`b55d15a`) for
exactly this reason, and the tuner was never given it.

**The (contaminated) v2 numbers:**

| | `conc` | wrap | peak (of 100 nm) | null | **cost, v2** |
|---|---|---|---|---|---|
| geometric seed — **reads at 0.99** | 1.000 | **0.52** | 81.5 | 71.8 nm | **0.1629** |
| old winner — **reads at 0.03** | 0.006 | **1.00** (pinned at λ/4) | 158.2 | 0.0223 nm | **4.3944** |

The *shape* of the discrimination — localization 1.000 vs 0.006, the wrap meter
0.52 vs 1.00, the "peak" exposed as 158.2 nm = λ/4 exactly — is what the fix is
designed to produce, and the two configurations differ so grossly that
cross-contamination is unlikely to have manufactured it. **But "unlikely to have
manufactured it" is not a measurement.** The gate is re-run sequentially after
the tune; until it is, the fix is reasoned and unproven.

`conc` and `wrapf` now print on every `TAILEVAL` line, so a tune is auditable
rather than a single scalar.

**A caveat on reading the wrap flag as a cause.** `tailA`'s ladder reports
"base reads 1.00 of λ/4" at **every** rung — 30, 60, 120 and 240 nm alike —
whereas a genuinely surface-proportional wrap grows with the base (the record's
reflective rig is clean at 30 and 60 and only breaks at 240). A ratio pinned at
1.00 from the smallest base is equally consistent with a map that is simply
**random**: a random map saturates the range trivially. So the wrap flag may be
a *symptom* of a reading that carries no information rather than the *cause* of
one, and it should not be quoted as the mechanism until something distinguishes
the two. What it does establish, independently, is that the map is not a
faithful image of the commanded surface.

`tailB` (the geometric seed) now says whether the *tuned numbers* are the
defect or the *reading on this bench* is. If the seed reads, the fix is the
tail objective and the geometry stands — items 1–3 were measured with no tail
in the loop. If the seed fails too, the defect is deeper than the retune and
the geometry itself has to go back under the microscope.

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
**Do not carry those numbers onto a slide about mirrors.** They have been
re-measured (`oapifo2`, `oapsens22`, `oapsens22n`, §4.6 and §5), with the loop
and descent re-queued on the seed tail as close-out item 2. What is settled:

**Two numbers in an earlier version of this table came from the TUNED tail and
have been removed** — its 0.0223 nm null and its "150.0 nm poke recovered
(100 %)". §4.5 measured that tail reading a single actuator at **0.0338**; the
0.0223 null is real and is exactly what made it win the tuner's objective, which
is the finding, not a credential. The rows below are the geometric seed's, which
is what this bench actually runs.

| | lens rig | reflective rig, as designed here |
|---|---|---|
| buildable at the 22.5 deg node | yes, worst +38.2 mm over 10 parts | **yes, worst +33.4 mm over 8 parts** |
| mask-seat blur at best focus | diffraction-limited | **0.000 lambda F/D** |
| mask-seat trim | -5.582 mm (thin-lens seed correction) | **0.00 mm** (the parabola's exact conjugate) |
| tail flat-DM null | 0.134 nm (tuned) | **0.0289 nm** (the GEOMETRIC SEED — the tail of record here) |
| single actuator recovered, actuator space | — | **0.9915, 2.4 pm** (`oapifo2`, record resolution) |
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

### 4.5 The sequential gate FAILS the fix — and retracts §4.4's mechanism

Re-measured with `tg96_tail`'s per-process scratch names and strictly one run
at a time (`runs/gateseq2.sh`):

| | `conc` | wrap | null | **cost** | **actually reads at** |
|---|---|---|---|---|---|
| `objseed3` — geometric seed | 1.000 | 0.57 | 0.0289 nm | **0.0089** | **0.9809** (`tailB`) |
| `objwin3` — the old winner | **1.000** | **0.66** | 0.0223 nm | **0.0015** | **0.0338** (`tailA`) |

**The fixed objective still prefers the tail that does not read**, 0.0015 against
0.0089. The fix is **not proven; it is disproven.**

**And §4.4's mechanism goes with it.** Measured cleanly, the old winner's map is
**localized (`conc` 1.000) and unwrapped (0.66 of λ/4)**. The `conc` 0.006 /
wrap 1.00 that appeared to confirm "the detector walked off the pupil conjugate
and the map wraps" came from the contaminated concurrent probe. So:

- the **wrap** story is retracted — the old winner's map does not wrap;
- the **null-term-is-the-driver** story is retracted — the seed's true null is
  0.0289 nm, so `(null/2)²` was 0.0002, never dominant (§4.4 computed it from a
  contaminated 71.80 nm);
- the **localization** story is retracted — both configurations are localized.

**What survives, because it comes only from clean sequential runs** (`tailA` /
`tailB`, per-tag deck names, run one after the other):

> The old tuned tail reads a single actuator at **0.0338**. The geometric seed
> reads it at **0.9809**, with a clean break ladder to 480 nm. The tail
> parameters decide whether this bench reads.

**What is now open:** *why*. Every quantity `tg96_tail` computes about its own
candidate — null, peak, localization, wrap — says the old winner is healthy,
while the battery says it reads at 3 %. The tuner is therefore optimizing
something **orthogonal to readability**, and no reweighting of those four terms
can fix that; the objective needs to measure what the battery measures — the
recovered gain in **actuator space**, through the affine — rather than any
detector-space proxy. That is a real piece of work and it is not started.

**The practical answer for item 4, available now:** **use the geometric seed
tail.** `tailB` proves it reads at 0.98 with a clean ladder, and `oapifo2` has
already run the full-resolution battery on it. The reflective bench needs no
tuned tail; the tuner is the thing that is broken, and it is broken in a way
this session has not diagnosed.

### 4.6 Item 4, resolution-matched, on the seed tail — `oapifo2`

Model 1024, grid 384×0.28, NGRID 385, detector Nyquist 192.5 cyc/pup — the
**same** resolution as the record's `oap` and `lens` runs. Geometric seed tail.
Magnification 10.437 DM-mm/det-mm (lens 9.879, record reflective 10.704); 3680
lit actuators; clearance re-confirmed at worst **+33.4 mm** over 9 parts.

**Modal transfer — the record's central reflective-vs-lens claim, reversed:**

| mode | cyc/pup | **`oapifo2` gain / cross-talk** | record reflective | lens |
|---|---|---|---|---|
| 1,1 | 0.7 | **0.9978 / 0.0063** | 0.9493 / 0.2202 | 0.9877 / 0.0049 |
| 4,4 | 2.8 | **0.9893 / 0.0105** | 0.6381 / 0.4772 | 0.9884 / 0.0065 |
| 16,16 | 11.3 | **0.9916 / 0.0109** | 0.7789 / 0.4106 | 0.9886 / 0.0090 |
| 32,32 | 22.6 | **0.9939 / 0.0161** | 0.7681 / 0.4189 | 0.9889 / 0.0284 |
| 80,80 | 56.6 | **0.9660 / 0.0078** | — | — |

The record calls its 0.22–0.48 cross-talk *"the same-plane fold's astigmatism
cross-talk … only the geometry can move it"* and concludes the reflective rig is
an open-loop/differential-grade instrument. **Measured on the designed bench the
cross-talk is 0.006–0.021 — the lens rig's own figure, and ~20× below the
record's reflective rig.** Fed at its focus and laid out to clear its beams, the
fold costs essentially nothing in cross-talk.

**Differential rows, and the flat-DM null:**

| base → deviation | **`oapifo2`** | record reflective | lens |
|---|---|---|---|
| flat → single 10 nm | 0.9915 / 2.4 pm / 0.9999 | 0.9948 / 2.2 / 0.9999 | 0.9916 / 2.2 |
| **flat → random 10 nm** | **0.9905 / 161.2 pm / 0.9999** | **0.7486 / 4848.5 / 0.8706** | 0.9893 |
| random 16 nm → single | 0.9952 / 2.1 / 0.9999 | 0.9958 / 2.3 / 0.9999 | — |
| **random 16 nm → random** | **0.9892 / 189.8 pm / 0.9999** | **0.7486 / 4848.3 / 0.8706** | — |
| flat-DM null | **28.9 pm** | 12 893 / 13 089 pm | 134.5 pm |
| reg sweep all / bright / dark | **0.9905 / 0.9906 / 0.9900** | 0.7486 / 0.9932 / **0.0494** | 0.9893 / 0.9890 / 0.9906 |

Dense-random residual **161 pm against the record's 4848 — 30× smaller** — and
the regularization sweep is uniform across bright and dark columns (0.9906 /
0.9900) where the record's reflective rig collapses to 0.0494 in the dark.

**The one place the designed bench is WORSE than both, and it must not be
buried: the break ladder.**

| base rms | **`oapifo2`** | record reflective | lens |
|---|---|---|---|
| 30 nm | **0.9965 / 2.0 pm** | 0.9967 / 2.5 | 1.0013 / 4.5 |
| 60 nm | **0.9929 / 2.0 pm** | 0.9987 / 3.1 | 1.0103 / 9.3 |
| 120 nm | **0.3698 / 437.9 — BROKE** | 0.9617 / 489.6 | 1.0258 / 19.3 |
| 240 nm | **−0.9009 — BROKE** | 0.4779 — BROKE | 1.9271 / 394.0 |
| 480 nm | **−4.9780 — BROKE** | 1.0192 / 15.3 | 1.5849 / 419.1 |

It wraps from **120 nm**, where the record's reflective rig holds to 120 and
breaks at 240, and the lens rig never flags. **The designed bench has a smaller
capture range on a deep surface.** This also **corrects §4.3's model-512
indication**, which showed a clean ladder to 480 nm and led me to say it "beats
the lens rig at the top of the ladder" — that was the resolution mismatch I
flagged, and the matched run does not support it. On the 30 nm working surface
the record actually operates on, the bench is clean.

### 5. Item 5 — the mask sensors on the designed bench (`oapsens22`)

Model 1024, NGRID 193, bare-Al coating, `mask.v_arm 'engine'`, matched to the
lens gate run `gate22_193`. Sandwich and reference gates all pass at 1e-15:
G1 1.83e-15, G2 1.98e-15, G3 3.92e-16, G7 2.26e-15, G8 4.0e-15.

| reading | record (7° reflective) | **designed bench** | gate |
|---|---|---|---|
| stepped dimple S | survives | (rows in the run) | — |
| **vector pair V** | FAIL, 19.6 pm | **FAIL, 633.97 pm** | < 12 pm |
| **pinhole P** | FAIL, 94 pm | **PASS, 0.269 pm** | < 12 pm |

**The pinhole recovers, and it is not marginal: 94 pm → 0.269 pm, a 350×
improvement, from FAIL to PASS.** That answers item 6's question — *does the
pinhole recover?* — **yes**, and by fixing the conjugate rather than by opening
the fold, which is the opposite of the lever the brief proposed.

**The vector pair gets worse, 19.6 → 634 pm**, and the discriminator has now
run (`oapsens22n`: identical except `coat_oap 'none'` — ideal reflectors,
RS = −1, RP = +1, zero retardance — so the coating's polarization is removed
and the geometry is kept):

| | V rms error | gate < 12 pm |
|---|---|---|
| `bareAl`, folds **20° / 25°** | **633.97 pm** | FAIL |
| **`none`, same geometry** | **208.24 pm** | **FAIL** |
| the record: `bareAl`, folds 5° / 9° | 19.6 pm | FAIL |

**Both causes are real, and the split is about 3 : 10.** Removing the coating
recovers **3.0×** — so bare aluminium at 20° / 25° is roughly two-thirds of the
excess, and *that* part is specifiable: a protected-Al or dielectric stack can
be written against it. But **208 pm with ideal reflectors is still 17× over the
gate** and 10× worse than the record's figure, with no coating in play. The
clearance-driven fold angles themselves are what the vector pair cannot take.

The effect is polarization-only, which is what the coating hypothesis predicts
and the run confirms: between the two, the **capture range is 61 vs 60 nm** and
the scalar imaging is **identical to four figures** (center-poke raw peak gain
0.7487, corr(map, truth) 0.9879 in both). Transmittance moves 0.6513 → 0.7804,
as removing an absorbing metal should.

**Caveat on the third row.** The record's 19.6 pm was measured on the 7° bench
*with* the 25 mm conjugate error and its own tail, so it is not a clean
geometry-only control. The clean control would be this bench at 5° / 9° folds —
which is exactly what §2 shows is **not buildable**. So "the fold angles cost
the vector pair ~10×" is the right reading of the evidence available, not a
number isolated by experiment.

**For the trade, not for a fix.** The reflective front end buys the pinhole
(94 pm FAIL → 0.269 pm PASS) and the interferometer's cross-talk (0.22–0.48 →
0.006–0.021, §4.6); it costs the vector pair, two-thirds of that cost being
coating and specifiable. That is the honest shape of item 6's trade, measured
in both directions.

One sampling note from the run: **2.37 detector px per actuator** against a
minimum of 2 — the 96×96 DM is close to the floor at NGRID 193 on this bench's
larger pupil image. It passes, but it is the thinnest margin in the budget.

### 5.1 Item 1 — the vector pair on the redesigned rig: rows, overcoat, verdict

Four runs at the `oapsens22` settings (model 1024, NGRID 193, OAP1 20° / OAP2 25°,
`MASK_TRIM` 0, `mask.v_arm 'engine'`, laser 45°), each changing ONE thing:
`oapsens22` (bare Al, uncalibrated solver) is the reference, `oapsens22n` removes
the coating, `vqw22` changes the coating, `vmap22` changes the solver.

| | `oapsens22n` | `vqw22` | `oapsens22` | `vmap22` |
|---|---|---|---|---|
| coating on L1/L2 | **none** (ideal reflectors) | **`qwAl`** λ/4 MgF₂ | `bareAl` | `bareAl` |
| solver (`mask.v_cal`) | ideal | ideal | ideal | **`map`** |
| **G4, V rms error** | **208.24 pm** | **318.79 pm** | **633.97 pm** | **0.054 pm PASS** |
| channel phase difference | 6.15e-2 rad | 6.16e-2 | 6.20e-2 | 6.20e-2 |
| channel amplitude \|qL\|/\|qR\| | **1.0000** | **1.0313** | **1.0849** | 1.0849 |
| diattenuation / retardance mean | 5.46e-2 / 5.22e-2 rad | 5.39e-2 / 6.37e-2 | 4.00e-2 / 1.01e-1 | 4.00e-2 / 1.01e-1 |
| transmittance | 0.7804 | 0.5583 | 0.6513 | 0.6513 |
| row: single 10 nm | 0.9963 / 3 pm | 0.9982 / 3 | 1.0022 / 4 | 0.9940 / 3 |
| row: grid @1 nm | — | 1.0028 / 3 | 1.0100 / 5 | 0.9978 / 3 |
| **row: dense random 10 nm** | — | **1.0018 / 374 pm** | **1.0086 / 633 pm** | **0.9976 / 289 pm** |
| ladder 30 / 40 / 50 / 60 nm | .9964 .9873 .9661 .9275 | .9984 .9842 .9608 .9215 | .9928 .9784 .9491 .9037 | .9941 .9873 .9686 .9322 |
| capture range to 10 % | 61 nm | 61 nm | 60 nm | 62 nm |

**The rung the brief reads the 634 pm against is not the variable on this rig.**
The brief places G4 on the zwfs record's channel-PHASE scale (README V3: 59 / 178 /
597 / 1877 pm at 0.01 / 0.03 / 0.1 / 0.3 rad) and infers ~0.1 rad. Measured, the
channel phase difference is **6.15–6.20e-2 rad in all four configurations** — it
does not move at all — while G4 moves 208 → 319 → 634 pm. What the coating moves is
the channel AMPLITUDE imbalance, 1.0000 → 1.0313 → 1.0849. `zwfs_params` says as
much in its own comment beside `v_arm_damp`: *"the OAP rig's term"*. So the
redesigned rig sits at a FIXED 0.062 rad of channel phase — a rung whose record
value is ~370 pm — and the spread around it is amplitude, not phase. Reading the
634 against the phase ladder would have put the rig at 0.1 rad and the fold lever
in play; it is at 0.062 rad on every coating, and the lever is not what moves it.

**The rows hold, uncalibrated, on all three coatings.** Single 0.996–1.002, grid
1.003–1.010, ladder 0.98–0.99 out to 40 nm, capture 60–62 nm: bare Al, the
quarter-wave overcoat and no coating at all are indistinguishable at the row level.
The vector pair's regression on this rig is confined to exactly two numbers — the
G4 single-poke absolute and the dense-random residual — and both are *uncalibrated*
quantities.

**The overcoat at a quarter wave of 632.8 nm halves it, and the record's
"protected Al" is not at a quarter wave.** `coat_protectedAl` is 229.3 nm of MgF₂:
n·t = 1.38 × 229.3 = 316.4 nm = **0.500 λ at 632.8** — a HALF wave, which
`tg96_params` already calls it. The new `coat_qwAl` is 114.6 nm = 0.250 λ, the
quarter wave of the bench's own working wavelength. It takes G4 from 633.97 to
**318.79 pm (1.99×)** and the dense row from 633 to 374 pm, i.e. it removes **half**
of the coating's excess over the coating-free floor (3.04× → 1.53× of 208.24 pm).
That is real and it is the right specification, but it is NOT the 0.05× the engine's
overcoat rule gives (`macos_f90/CLAUDE.md`): that rule is measured on
cross-polarized POWER on a two-mirror Cassegrain, and this is the reading error of
an uncalibrated circular-channel solver on a 20°/25° fold pair. The two quantities
are not the same number and should not be expected to agree; what survives is the
SIGN and the mechanism — a quarter-wave overcoat helps, an off-quarter-wave one
costs. The engine's own caution applies: the film is fixed glass, so the condition
belongs to the pair (stack, λ), not to the stack.

**The calibrated bench removes essentially all of it.** With the true per-channel
maps and constants (`v_cal 'map'`, the polarimetric oracle) G4 is **0.054 pm —
PASS against the < 12 pm gate**, from 633.97, and the dense row falls to 289 pm.
So the vector pair on the redesigned rig is not limited by the fold angles or by
the coating. It is limited by a solver that does not know its own two channels,
and the thing it does not know is very nearly a per-channel CONSTANT.

**And it is the channel PHASES, not the amplitudes, that the bench has to
learn.** Two more legs separate the oracle from what a bench can actually
measure:

| solver (`mask.v_cal`), bare Al | what it knows | G4 | dense row |
|---|---|---|---|
| `ideal` | nothing | 633.97 pm | 633 pm |
| `fit` (`vfit22`) | per-channel CONSTANTS κ₊, κ₋, η fitted on the flat DM's two masked images — a bench calibration | **199.45 pm** | 327 pm |
| `amp` (`vamp22`) | the per-channel unmasked amplitude MAPS \|qL\|, \|qR\| — the reference frames every bench already takes | **199.39 pm** | 327 pm |
| `map` (`vmap22`) | the same maps **plus the polarization phases** | **0.054 pm** | 289 pm |

(rows for `fit` / `amp`: single 0.9962 / 0.9963, grid 0.9985 / 0.9984, capture
62 / 61 nm — the rows do not separate them either.)

`fit` and `amp` land on the same number to 0.03 %. Amplitude information —
constants or full maps, cheap or free — buys a factor 3.2 and then stops dead.
The remaining **factor of 3700 is entirely the two channels' polarization
PHASES**, which neither a flat-DM fit nor an unmasked reference frame can see.
So the redesigned rig's vector pair does not need more of the data a bench
already takes; it needs a **polarimetric** calibration, and that is a line item
with hardware behind it, not a free byproduct of the frames.

**Which puts item 6's fold lever back in the picture — as a fallback, not the
fix.** The quantity that survives every amplitude calibration is the channel
phase difference, 0.0615 rad, and that is *fold-set*: it is the same to three
figures with bare Al, with the quarter-wave overcoat and with no coating at all
(6.20 / 6.16 / 6.15e-2), so the geometry sets it and the coating does not touch
it. The 208 pm coating-free floor is that term. Halving OAP2's fold is therefore
aimed at exactly the right quantity — the brief's premise is sound — but it buys
a factor of a few where the polarimetric calibration buys 3700. **The lever
stays unpulled**: it is the answer only if a polarimetric calibration is ruled
out, and it would cost the clearance solve that §2 shows has no slack at 5°/9°.

**The line for the deck's redesigned-rig slide:**

> Vector pair on the redesigned reflective rig: the rows hold uncalibrated
> (single 1.002, grid 1.010, capture 60 nm — the lens rig's own figures), and
> the single-poke absolute is **634 pm** bare, **319 pm** with a quarter-wave
> MgF₂ overcoat, **199 pm** with the calibration a bench already performs, and
> **0.054 pm — PASS** with a polarimetric one. What the calibration has to
> supply is the two channels' PHASES; their amplitudes are free and buy only
> 3.2×.

Run tags: `oapsens22` / `oapsens22n` (item 5 of the previous brief), `vqw22`,
`vmap22`, `vfit22`, `vamp22`; launchers `zwfs_dm96/runs/vcloseseq.sh`,
`vfitseq.sh`, `vampseq.sh`. The new coating option is `bench.coat_oap 'qwAl'`
(`coat_qwAl`, MgF₂ 114.6 nm on 100 nm Al); any `P.bench.coat_<name>` struct is
now a valid choice, and the five places that had to strip `coat_*` fields by
name match them by prefix instead, so a new stack cannot be forgotten in one of
them.

### 4.7 Item 2 — the 120 nm wrap, read like with like

**The comparison the brief asks for could not be made from the committed
reports, and that is a defect in the instrument, not in the benches.** The
ladder's wrap meter — `max|h|` over the lit pupil in units of λ/4, 1.00 meaning
the base reading has saturated the four-step's unambiguous range — was printed
ONLY inside the `BROKE` note. A rig that holds therefore printed no wrap number
at all, so "the OAP seed tail wraps from 120 nm where the lens rig never flags"
compares a measured quantity against a blank. `tg96_run` now prints the wrap
fraction as a column at EVERY rung, and `runs/lensuw2` re-runs the lens rig
through today's code so there is a control measured the same way.

**The sampling difference, measured, is 7 % — and the brief's hypothesis (a)
predicts the wrong sign.** The ray affine now prints detector pixels per
actuator directly:

| | mag, DM-mm/det-mm | dxd, mm | **detector px per actuator** |
|---|---|---|---|
| OAP seed tail (`oapifo2`, `oapuw2`) | 10.4370 | 2.0309e-02 | **4.718** |
| lens rig (`lens`, the record) | 9.8793 | 2.0054e-02 | **5.047** |

The lens rig gets 7.0 % more pixels per actuator. Hypothesis (a) — *fewer
pixels per actuator → a larger phase step per pixel at the same surface* — is
arithmetically right about the step and **backwards about the consequence for
the wrap meter**: fewer pixels per actuator means MORE blur per actuator, a
LOWER `max|h|`, and therefore LESS saturation, not more. And 7 % is not a
plausible size for a difference that turns "holds at 120 nm" into "broke at
120 nm"; the ladder doubles at each rung.

So the two runs are set up to discriminate, not to confirm: `oapuw2` re-reads
the OAP ladder with `battery.unwrap` on (CCMac's lens_deck captured from 150 nm
with the unwrapper alone, so the record's own comparison was already unlike),
and `lensuw2` supplies the lens rig's wrap fractions through the same code.
**Running; the numbers land here.**

**`battery.unwrap` does not reach the battery — found while the run was in
flight.** The knob is read in exactly one place, `stage_loop_`
(`do_uw = P.battery.unwrap || descent`); neither `stage_CDE_` nor
`stage_matrix_`, which own the break ladder, consults it. So `oapuw2` is
running with NO unwrapper and is, for part (a) of this item, **vacuous** — it
reproduces `oapifo2`'s ladder. It is being left to finish because it is not
worthless: it supplies the OAP rig's wrap fraction at every rung, which is half
of what part (b) needs, and `lensuw2` supplies the other half. Part (a) needs
the knob to work first.

**And reading the ladder's code to find that turned up a better hypothesis than
either of the two the brief offers.** The ladder measures

```
hb = measr(bb);  hbd = measr(bb + d_sng);  adev = est(hbd - hb);
```

`measr` goes through `ctx.measf`, which is `angle(exp(1i*(fourstep - p_null)))`
— a **separately wrapped absolute map**. So the ladder subtracts one wrapped
absolute from another, which is precisely what this file's own comment beside
`ctx.phasef` says never to do: *"the V1 lesson -- wrap the DIFFERENCE of two
phases, never subtract two separately-wrapped absolute maps. On a base that
exceeds lambda/4 the absolute map wraps but a small differential does not, so
calibration pokes and rows must difference-then-wrap."* The 10 nm differential
never wraps; the 120 nm **base** does.

That predicts the break precisely, and it predicts it without any appeal to
sampling: the rig whose base reading reaches 1.00 of λ/4 first is the rig that
breaks first, and 7 % of pixels-per-actuator has nothing to do with it. The
wrap column now measures exactly that quantity on both rigs, so the two runs
in flight discriminate between this and the brief's hypotheses rather than
merely confirming a break. If it holds, the fix for part (a) is not the
unwrapper at all — it is to difference before wrapping in the ladder, the way
the rows already do.

**And the ladder is the ONLY place left that does it.** Everything else in this
runner already differences before wrapping:

| site | form |
|---|---|
| `build_J_` (the matrix calibration), `tg96_run.m:1557-1579` | `angle(exp(1i*(phasef(base+poke) - pbase)))` — wrapped DIFFERENCE |
| the deck stage, `:885-891` | `wdiff = angle(exp(1i*(phasef(target) - phasef(base))))` — wrapped DIFFERENCE |
| **the break ladder, `:786` (`measr`, `:716`)** | **`measr(bb + d_sng) - measr(bb)` — two wrapped ABSOLUTES** |

So the ladder is the outlier, and the quantity it reports as "the bench broke"
is measured differently from the quantity every other row reports. The
suspicion this raises is concrete: **the 120 nm break may be an artifact of the
ladder's own differencing rather than a property of the bench** — and if so the
record's reflective rig "holding to 120 and breaking at 240" and the lens rig
"never flagging" are both measurements of where each rig's base map crosses
λ/4, not of capture range.

That is a claim about the instrument, so it does not go on a slide until it is
measured. The two runs in flight give the wrap fractions; the test after them
is to run one ladder rung both ways on the same bench. If the wrapped-difference
form holds where the absolute form breaks, the ladder changes and every capture
number on the deck is re-read — including the lens rig's, which would then be
understating nothing and the OAP rig's, which would be understating a lot.

#### The OAP rig's wrap column, and why the meter I added is not good enough

`oapuw2`, the seed tail at record resolution, with the wrap fraction now printed
at every rung:

| base rms | gain | floor pm | corr | **wrap** | |
|---|---|---|---|---|---|
| 30 nm | 0.9965 | 2.0 | 0.9999 | **0.93** | |
| 60 nm | 0.9929 | 2.0 | 0.9999 | **1.00** | |
| 120 nm | 0.3698 | 437.9 | 0.1418 | **1.00** | BROKE |
| 240 nm | −0.9009 | 591.9 | −0.2836 | **1.00** | BROKE |
| 480 nm | −4.9780 | 1296.4 | −0.6978 | **1.00** | BROKE |

**The base reading saturates at 60 nm and the ladder does not break until 120.**
So saturation is necessary and nowhere near sufficient, and `max|h|/(λ/4)`
**cannot discriminate 60 nm from 480 nm** — it reads 1.00 at all of them. The
meter I promoted to a column is the wrong statistic: `max` over the pupil hits
1.00 the moment a SINGLE pixel wraps. The right one is the FRACTION of the lit
pupil beyond the fold, which is what the ZWFS battery has always reported
(`fold0 / fold`) and what the tg96 ladder does not. Changing it means editing
`tg96_run.m`, which is the file the runs in flight are executing, so it waits
for them -- it is planned, not queued.

**What the two numbers together already say about the mechanism.** For a
Gaussian base of rms σ the fraction of the pupil beyond λ/4 = 158.2 nm is
`erfc(158.2/(σ√2))`: 1.4e-7 at 30 nm, **0.8 % at 60 nm**, **19 % at 120 nm**,
51 % at 240. (The ladder's rungs are ACTUATOR commands, and the surface a
random command field makes is about 3 % larger in rms at 12 % coupling, so these
are the right numbers to a few per cent.) The ladder holds at 0.8 % and breaks
at 19 %. That is the
signature of the wrapped-absolute subtraction, not of a capture limit: the
difference `measr(bb+d) − measr(bb)` is correct at every pixel where BOTH maps
wrapped the same number of times, and wrong by a full λ/2 at the pixels the
10 nm poke pushes across a wrap boundary. The count of such pixels scales with
how much of the pupil sits near a boundary, which is what the erfc tracks. It
does not scale with pixels per actuator at all.

`lensuw2` is the control: if the lens rig's break also lands near 19 % of its
own measured map, the ladder is measuring its own arithmetic on both rigs and
the "reflective rig has a smaller capture range" line comes off the deck.

**A tension the OAP data alone does not settle, stated before the control
lands.** *(Superseded below: the record's lens ladder turned out to be a 7°
bench, so this paragraph's comparison is not like-with-like. Kept because the
reasoning it sets up — that measurement amplitude had to be tested — is what
the control then excluded.)* The lens rig's record ladder (`runs/lens`) holds at
120 nm — gain 1.0258, corr 0.9946 — and only degrades at 240. On the COMMANDED surface those
rungs are the same 19 % and 51 % beyond λ/4 for both rigs. So a purely
arithmetic account ("the subtraction fails once ~19 % of the pupil has wrapped")
predicts the lens rig should break at 120 too, and it does not.

Two candidate resolutions, and they make opposite predictions for the control:

1. **The measured maps differ in amplitude.** What wraps is the MEASURED `h`,
   not the commanded surface. If the lens rig's measurement attenuates more —
   a lower raw gain on the same true surface — fewer of its pixels reach λ/4
   and it breaks later. The OAP rig's center-poke raw peak gain is 0.7487
   (`oapsens22`); the lens rig's is the number to put beside it.
2. **Sampling**, the brief's hypothesis — which is already excluded twice over:
   the lens rig has 7 % MORE pixels per actuator (5.047 vs 4.718), so it should
   resolve more of the surface and wrap EARLIER, not later, and 7 % cannot move
   a break by a factor of two on a ladder that doubles.

`lensuw2` measures (1) directly, because the wrap column is computed on each
rig's own measured map. **This is also what makes the saturating meter
expensive:** `max|h|/(λ/4)` reads 1.00 for both rigs at 120 nm and answers
nothing. The ladder needs the MEASURED base rms and the beyond-fold FRACTION
printed per rung — both one-liners in `tg96_run.m`, both waiting on the runs in
flight to release the file.

**A third instrument point: the `BROKE` flag is loose.** *(The numbers below are
the 7° record run; the flag's looseness is real and general, the "flatters the
lens rig" reading is not — on today's bench the lens rig's 240 nm rung reads
−1.6631 and IS flagged.)* The test is
`g < 0 || g > 3 || corr < 0.3`. The lens rig's 240 nm rung reads **gain 1.9271,
corr 0.6872** — wrong by 93 % — and is not flagged, because 1.93 is under 3 and
0.687 is over 0.3. Its 480 nm rung, 1.5849 / 0.5661, likewise. So "the lens rig
never flags" is partly the flag's generosity: by the stricter criterion the
capture line already uses elsewhere (gain within 0.9–1.1), the lens rig fails at
**240 nm** and the OAP rig at **120 nm** — a factor of two, not the "holds
everywhere versus breaks at 120" the unflagged table suggests.

And the measured modal transfer says the difference is NOT in how hard each rig
attenuates the surface it is reading: `oapifo2` gives 0.9978 / 0.9893 / 0.9916 /
0.9939 at 0.7 / 2.8 / 11.3 / 22.6 cyc per pupil against the lens rig's 0.9877 /
0.9884 / 0.9886 / 0.9889 — within 1 % of each other, and if anything the LENS
rig attenuates slightly more. A 1 % difference in measured amplitude cannot move
a break by a factor of two either. So of the three candidate explanations —
sampling, measurement amplitude, and the ladder's own arithmetic — the first two
are now excluded by measurement, and the wrap stage tests the third directly.

#### The control lands, and the premise of item 2(b) does not survive it

`lensuw2` — the lens rig through today's code, on today's 22.5° bench:

| base rms | gain | floor pm | corr | wrap | |
|---|---|---|---|---|---|
| 30 nm | 0.9930 | 4.8 | 0.9996 | **0.92** | |
| 60 nm | 0.9990 | 8.3 | 0.9989 | **1.00** | |
| 120 nm | 1.0250 | 15.2 | 0.9967 | **1.00** | |
| 240 nm | −1.6631 | 613.0 | −0.5910 | 1.00 | **BROKE** |
| 480 nm | −0.1862 | 363.6 | −0.1088 | 1.00 | **BROKE** |

**Two things, and the first invalidates the comparison the item is built on.**

**1. The record's lens ladder is a DIFFERENT BENCH.** `runs/lens` reports
`binding angle 6.88 deg -> BS_AOI = 7 deg`; `lensuw2` reports `BS_AOI = 22.5
deg (pinned)`. So "the seed tail wraps at 120 nm where the lens rig never
flags" compares a 22.5° reflective rig against a **7° lens rig** — the same
class of mismatch as §4.6's resolution error, and it is why the record's 240 nm
rung reads +1.9271 where today's reads −1.6631. Measured like with like, **the
lens rig does flag** — at 240 nm. The brief's "never flags" is an artifact of
the comparison, not a property of lenses. Anything on a slide that contrasts
the two rigs' capture must use `lensuw2` and `oapuw2`, not `lens` and
`oapifo2`.

**2. The wrap column is IDENTICAL on the two rigs** — 0.92 and 0.93 at 30 nm,
1.00 at every rung above. Both rigs' measured maps reach λ/4 at the same base
rms. Combined with the modal transfer agreeing to 1 %, that **excludes
measurement amplitude** as the difference, which was the leading candidate after
sampling was excluded. Neither rig's measurement is attenuating the surface more
than the other's.

**What is left, and it is now the only candidate standing.** The real
difference between the two runs is in the ESTIMATOR's conditioning:

| | lit actuators (unknowns) | window per actuator | Stage C gain |
|---|---|---|---|
| `oapuw2` | **3680** | **37 px** (1369 px²) | 0.9810 |
| `lensuw2` | 3260 | 41 px (1681 px²) | 1.0135 |

The OAP rig solves **13 % more unknowns from 23 % fewer pixels each** — a
consequence of its larger magnification (10.437 vs 9.841 DM-mm per detector-mm),
which puts more of the DM on the same detector. A wrap-corrupted pixel is off by
λ/2 = 316 nm against a 10 nm signal, i.e. 31× the thing being measured, so the
solve's tolerance for them scales with pixels per unknown. That is a mechanism
consistent with every measurement now in hand, and it is NOT what the brief's
hypothesis (a) says: the direction is the same (the OAP rig is worse off per
unknown) but the quantity is pixels per ACTUATOR IN THE SOLVE, not the phase
step per pixel, and the size is 23 % rather than 7 %.

**It is a candidate, not a conclusion — and the experiment that would settle it
got sharper while this was being written.** The first design for the `wrap`
stage measured the beyond-fold FRACTION, which is only an input to the
mechanism. The ZWFS battery turns out to already report the right idea for its
own one-frame prior — *"pixels that cross the fold under the change"* — so the
stage now forms the SAME differential both ways and compares them directly:

```
dA = measf(base+dev) - measf(base)                        the ladder's form
dD = angle(exp(1i*(phasef(base+dev) - phasef(base))))     the correct form
```

and reports `n_cross` = the pixels where they differ by more than λ/8, plus
`rms(dA−dD)` and `corr(dA,dD)`. A pixel the 10 nm poke pushes across a wrap
boundary is wrong by λ/2 = 316 nm in `dA` — 31× the signal — while `dD` never
wraps, because the difference is small everywhere.

**This needs no matrix and no affine**, because the two forms are compared
against each other rather than against a truth map, so neither the estimator's
conditioning nor the DM→detector mapping can be blamed for the answer. That
matters: it separates the two candidates cleanly. If `n_cross` is zero at a
rung, that rung's ladder result is about the bench; if it is large, the result
is mostly arithmetic whatever the bench does — and only then does the
conditioning difference above become the thing that decides WHICH rig tolerates
it.

#### Blast radius of the wrapped-absolute form: the ROWS are not affected

The rows use the same `measr` subtraction the ladder does (`hb = measr(base)`,
`hbd = measr(base+dev)`, `est(hbd-hb)`), so the question has to be asked of them
too. They are safe, and for a reason that is measured rather than assumed:
their bases are **flat** and **random 30 nm**, and at 30 nm the wrap meter reads
0.92–0.93, i.e. the deepest pixel sits at 93 % of λ/4 = 147 nm. A 10 nm
deviation cannot push it past 158.2 nm, so no pixel crosses a boundary and the
subtraction is exact.

The ladder is the only place that goes deep enough to cross — 60 nm is where
the first pixels pass λ/4 (the meter saturates there) and the gain is still
0.993–0.999, so a few crossings are tolerable; 120 nm is where ~19 % of the
pupil is past it and the estimate collapses.

**So every row of record stands** — `oapifo2`'s 0.9915 / 2.4 pm and 0.9905 /
161 pm, the cross-talk, the reg sweep, the dense-random residuals. What is in
question is the ladder's deep rungs and therefore the CAPTURE RANGE line, not
the reading itself.

#### Two things `oapifol2` has already settled, before its last rungs land

**The thermal term is not photon-limited, and the runner's own formula predicts
it exactly.** Across three decades of photons the thermal steady state barely
moves — 12.20 → 10.24 → 10.03 pm at 1e12 / 1e13 / 1e14 — while its *bias*
converges on 10.23 → 10.03 → **10.00 pm**. The report's header gives the law:
`ramp lag = rate/(gG)`. With the sheet's 5 pm-per-cycle ramp and gain 0.5 that
is **10.00 pm at G = 1**, which is what the measurement converges to as the
noise term is driven out (sig_n 11.92 → 3.77 → 1.19).

So the thermal drift has a **hard floor set by gain and cycle rate, not by
photons**: buying light does nothing for it, and the only levers are a higher
gain or a faster cycle. The WALK term behaves oppositely — 3.21 → 2.43 pm from
1e13 to 1e14 — and is genuinely photon-limited, which is the term the brief
asks about. Worth keeping the two apart on any slide: one is a light budget,
the other is a control-bandwidth budget.

**And the descent already carries the unwrapper, which gives item 2(a) a second
route.** `do_uw = P.battery.unwrap || descent` — the knob never reaches the
battery's ladder (above), but a DESCENT sets `descent` true and therefore
unwraps. `oapdesc2` starts the loop from 100 and 200 nm rms, i.e. from
**above** the 120 nm rung where the un-unwrapped ladder breaks. So it is a
de-facto test of whether unwrapping rescues a deep surface on this bench,
arriving by a different road than the ladder re-run part (a) asks for — and it
is already queued. If the descent captures from 200 nm while the ladder breaks
at 120, the unwrapper is the difference and part (a) is answered without
touching the ladder at all.

#### Both drift floors are ANALYTIC, and the loop model predicts them to 1 %

With the 1e15 rungs in, the two drift terms have each converged on a floor, and
each floor is exactly what the runner's own header formulas give at gain 0.5
with loop transfer G = 1:

| term | formula (the report's own header) | predicted | **measured** |
|---|---|---|---|
| walk | `sig_d/sqrt(gG(2−gG))` = 2/√0.75 | **2.309 pm** | **2.33 pm** (1e15) |
| thermal | `rate/(gG)` = 5/0.5 | **10.000 pm** | **10.00 pm** (1e14, 1e15) |

Including the residual noise term at 1e15 (`sig_n` 0.38 pm) the walk formula
gives 2.320 against 2.33 measured. **So the servo's behaviour on this bench is
not an empirical curve — it is the textbook proportional-loop result, and the
engine reproduces it to 1 %.** That is worth more than the numbers themselves:
it means the design levers are the analytic ones and can be read off without
another run.

**What that says for the deck.** Light buys only the APPROACH to these floors,
never the floors:

- the **walk** floor, 2.31 pm, is set by the per-actuator walk `sig_d` and the
  loop gain. It sits *below* the 3 pm spec, so 3 pm is reachable — but with
  only a factor 1.29 of margin, so the walk is the term that decides whether
  this servo meets spec at all.
- the **thermal** floor, 10.0 pm, is set by the ramp rate and the gain, and it
  is *five times* the spec. No photon budget touches it. The levers are a
  higher gain or a faster cycle — a control-bandwidth question, not a light one.

Full series, noise-only / walk / thermal steady state in pm, by photons per
cycle:

| N per cycle | none | walk | thermal | sig_n |
|---|---|---|---|---|
| 1e12 | 6.99 | 7.37 | 12.20 | 11.92 |
| 1e13 | 2.21 | 3.21 | 10.24 | 3.77 |
| 1e14 | 0.70 | 2.43 | 10.03 | 1.19 |
| 1e15 | 0.22 | 2.33 | (run 14) | 0.38 |

The noise-only column is clean 1/√N across four decades (6.99 / 2.21 / 0.70 /
0.22), which is the check that the photon bookkeeping is right before any of
the above is believed. Log-log interpolation puts the 3 pm crossing under the
walk at **≈1.8e13 photons per cycle**; the runner prints its own interpolation
when the last rung lands, and that is the number for the slide.

#### Item 2(1), the servo: `oapifol2`, the seed tail at record resolution

14 loop runs, 854 traced states, 331.8 min. Gain 0.50, 60 cycles, steady state
over the last 30, set point = the 30 nm working surface, one measurement per
cycle (the DM traced once, four frames sharing N photons).

**The number the brief asks for — photons per cycle to hold 3.0 pm rms:**

| drift | photons per cycle |
|---|---|
| noise only | **5.4e12** |
| **random walk, 2 pm per actuator per cycle** | **1.7e13** |
| thermal ramp, 5 pm rms per cycle | **floor 10.0 pm — never reached** |

So on the redesigned reflective bench the servo holds 3 pm against the 2 pm
walk at **1.7e13 photons per cycle**, and cannot hold 3 pm against the thermal
ramp at any photon level, because that term's floor is 10.0 pm.

**And the residual's SPECTRUM says where each floor lives**, which is what
turns the thermal result from a wall into a design lever. Held residual at 1e15
photons, rms in pm by spatial frequency:

| drift | < 4 cyc/ap | 4–12 | > 12 |
|---|---|---|---|
| none | 0.00 | 0.01 | 0.22 |
| walk | 0.24 | 0.67 | **2.22** |
| thermal | **9.24** | 3.12 | 2.22 |

The walk's residual is **high-order** (2.22 of 2.33 pm above 12 cycles), as a
per-actuator random walk must be. The thermal residual is **low-order** — 9.24
of its 10.0 pm sits below 4 cycles per aperture, which is exactly what a
defocus-plus-astigmatism ramp leaves behind.

**That is the actionable part.** The 10 pm thermal floor is `rate/(gG)`, a
bandwidth limit, and it is almost entirely in the first few Zernikes. It does
not need the full 96×96 loop run faster: a low-order servo, or simply a higher
gain on the low-order modes, addresses the whole of it. Nothing about the light
budget changes. The walk, by contrast, is genuinely photon-limited down to its
own 2.31 pm floor and is the term that decides whether 3 pm is met.

Rows of record for the reflective rig are therefore `oapifo2` (§4.6) + this,
with `oapdesc2` to follow for the descent.

### 4.7(a) RESOLVED — the break is the BASE READING WRAPPING, and both rigs do it at the same rung

`wrapoap` / `wraplens`, the new `wrap` stage: the ladder differential formed
BOTH ways at every rung, with the two quantities the saturating `max|h|` meter
could not supply. λ/4 = 158.2 nm of surface; the deviation is a 10 nm
single actuator.

| base rms | OAP meas rms | lens meas rms | OAP n_cross | lens n_cross |
|---|---|---|---|---|
| 30 nm | 33.0 nm | 31.1 nm | 0 | 0 |
| 60 nm | 64.1 nm | 61.5 nm | 0 | 0 |
| 120 nm | **90.2 nm** | **88.7 nm** | 2 (0.002 %) | 0 |
| 240 nm | **92.1 nm** | **91.0 nm** | 0 | 1 (0.001 %) |
| 480 nm | **91.9 nm** | **90.9 nm** | 0 | 5 (0.004 %) |

**The measured rms SATURATES at 91 nm, and the saturation value is analytic.**
A four-step reading is the surface modulo λ/2 = 316.4 nm, so a base the sensor
cannot follow is a reading distributed uniformly across that range, whose rms
is 316.4/√12 = **91.34 nm**. Measured: 90.2 / 92.1 / 91.9 on the OAP rig and
88.7 / 91.0 / 90.9 on the lens rig. There is no fitted constant here — the
number is the width of the unambiguous range and nothing else, which is why
both rigs land on it.

**Both rigs break between 60 and 120 nm, and by 120 nm both are saturated**
(98.8 % and 97.1 % of the analytic value). The rungs at 30 and 60 nm track the
commanded surface on both (the OAP rig reads 10.0 % and 6.8 % high, the lens
rig 3.7 % and 2.5 %; the excess falls with the rung, so it is not a fixed
additive background, and it is not pursued here). **"The reflective rig has a
smaller capture range" therefore comes off the deck** — the capture range is
λ/2 of surface on both, it is set by the four-step reading and not by the
optics, and the rig-to-rig difference in which rung the battery flagged is not
a capture-range difference. With sampling already excluded twice over (7 %, and
in the wrong direction, §4.7 above), nothing rig-specific survives.

**The ladder's own arithmetic is innocent.** dA (the difference of two wrapped
absolutes) and dD (the wrapped difference of the two phases) agree pixel for
pixel at every rung but three, where they differ at 2, 1 and 5 pixels out of a
~10⁵-pixel mask.

**But a handful of crossings still poisons a second-moment meter, and that is
the caution worth carrying.** Each crossing is a FULL λ/2, so it dominates any
rms or correlation taken over the mask. Two crossed pixels in 10⁵ predict an
rms(dA−dD) of √(2·316.4²/10⁵) = 1.41 nm against 1.31 nm measured; the lens
rig's five in 1.25·10⁵ predict 2.00 nm against 2.08 nm measured. The same two
pixels take corr(dA,dD) from exactly 1.00000 to −0.098 — against a 10 nm
deviation signal, two 316 nm outliers swamp the second moment. So `n_cross` is
the meter to read, and a correlation collapse on a ladder rung should be read
as "a few pixels crossed", never as "the differential is wrong".

Tags `wrapoap`, `wraplens`. This closes item 2(a); the wrap mechanism needed a
meter that does not saturate, which is what the staged patch supplied.

### 4.8 Item 3 — the gate's measure, fixed and calibrated against the battery

The point-sample measure was replaced by **lattice deconvolution**: the bench's
own measured influence stencil, taken from `tg96_place`'s anchor poke, is
deconvolved off a five-site poked map over the illuminated lattice
(`dmg_act_fit`) and the recovered command is regressed on the commanded one
with `tg96_run`'s `score_` verbatim — the battery's own quantity, not a proxy.
Three `verify_tail` legs, one placement and one row each:

| tail | battery | gate, `act_lam` 0.05 | 0.02 | 0.01 | 0.005 | 0.002 | at 0.95 |
|---|---|---|---|---|---|---|---|
| `objwin3` (bad) | 0.0338 | **−0.1621** | −0.1608 | −0.1604 | −0.1603 | −0.1603 | REFUSED ✓ |
| `lens_tail` (good) | 0.9968 | **0.8074** | 0.8137 | 0.8147 | 0.8149 | 0.8150 | REFUSED ✗ |
| `thk22_tail` (good) | 0.9885 | **0.9104** | 0.9212 | 0.9229 | 0.9233 | 0.9234 | REFUSED ✗ |

**The inversion is fixed.** The point-sample measure read `objwin3` at 0.9804
and `lens_tail` at −0.8285 — exactly backwards. The lattice measure orders all
three correctly and separates the bad tail from the good ones by ~1.0 in gain.
The non-vacuity leg passes: the known-bad tail is refused, and refused on a
number (−0.16) that no threshold choice could confuse with a good one.

**The threshold is still wrong, so the gate does not enforce.** Two tails the
battery certifies at 0.99 read 0.81 and 0.91 here, and 0.95 refuses both. A gate
that falls back to the seed on a good tail would put a tail regression inside
item 4's substrate runs — the same failure the advisory decision was taken to
avoid, so it stands.

**It is not the regularizer.** `dmg_act_fit` is Tikhonov-weighted and ridge
shrinkage biases a recovered amplitude low, which was the leading hypothesis.
The sweep refutes it: from `act_lam` 0.05 to 0.002 the gain moves 1.1 %, 0.9 %
and 1.4 %, against a deficit of 8–19 %. At the smallest weight `thk22_tail`
reaches 0.9234 of the battery's 0.9885. The sweep is nearly free — the trace,
the placement and the poked map happen once; only the `pcg` solve repeats — so
it now prints on every gate run.

**Live hypothesis: the single-site stencil.** The kernel is measured at the
anchor only, while the battery carries a column per actuator. A field-varying
influence response is then under-fitted at the other four sites and loses
amplitude, which would show up as a systematic low bias that no regularization
change can reach. Untested.

**Recommended fix, and it is scale-free.** Gate the winner against the
GEOMETRIC SEED measured through the SAME estimator, refusing when the winner is
materially worse than the seed. The seed is what the gate falls back to, so
that is the decision the gate actually has to make; and a ratio of two
identically-estimated quantities cancels the systematic bias exactly, which no
recalibration of an absolute threshold can promise. `row_gain_` already
measures the seed on the refusal path, so the cost is one extra row per tune.
**Not to be fixed by lowering 0.95** — moving a threshold to fit a measure that
is not understood is how the first gate came to be trusted.

### 4.8.1 A claim of mine, refuted by the diagnostic built to check it

The lattice measure was routed through a new `tg96_samp` rather than the shared
`dmg_samp`, on the argument that `dmg_samp`'s 8-parity family (axis permutation
+ signs + one isotropic scale) cannot express a rotation that is not a multiple
of 90°, and that two folds at 20° and 25° must put one into this rig's mapping.

**That argument is wrong, and the measurement says so.** `tg96_samp` reports the
distance from `mag·Linv` to the nearest signed permutation on every run. On all
three legs it is **0.0 %, anisotropy 1.0000** — the OAP rig sits at an exact 90°
(a permutation), the lens rig at 0°. `dmg_samp` would have worked on both.

The physics missed: a fold mirror **reflects** the pupil, it does not rotate it
about the axis. Image rotation comes from out-of-plane fold geometry, and these
folds are coplanar; the fold angle drives aberration, not image rotation.

`tg96_samp` is kept on the weaker, honest grounds — it is `tg96_place`'s own
affine and parity evaluated on the DM grid, so the bench carries one
registration convention instead of two, and it stays correct if a layout ever
does go out of plane. It is not load-bearing for correctness on any bench
measured here. The diagnostic stays, printing every run, so the day a bench
leaves the permutation family the number will say so rather than an argument.
(Display note: it reports distance to the NEAREST multiple of 90 — `mod(th,90)`
called an exact 90° rotation "89.999…", which prints as 90.00 and reads as the
opposite of the truth.)
