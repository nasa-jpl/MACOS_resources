# SegMirMaker segmentation audit — status

Audit of the SegMirMaker (SMM) segmented-mirror corpus.
Scripts (this directory): `seg_route.m` — the routing verdict;
`seg_audit.m` — geometry / frame / localization checks;
`seg_read_rx.m` — the shared `.in` text reader.

**Round 2, 2026-08-07.**  Round 1's absolute routing verdicts were
anchored on a frame slip and are all superseded — see
*How round 1 went wrong* below.  The mechanism work from round 1
(the `PSEG`/`HSEG` divergence, and why the defect hides) survived and
is kept.

Engine tree `macos dev` @ `ecfda5c` (includes the R1 `PSEG` fix,
`1ce1f5c`), gfortran release build.

---

## Headline

1. **The corpus routes correctly at its committed headers — with one
   exception: `ff_hex2` (BOTH lineages) is 180° PERMUTED.**  Round 1
   passed it as "ok (cancellation)"; that was the inverted reading.  It
   is a stale-generation deck, not an engine problem — see below.
   Everything else in `macos` + `MACOS_res_dev` measures CORRECT.

2. **Post-R1 the verdict collapses to ONE term.**  A deck routes
   correctly **iff its header `SegXgrid` names the basis its elements
   were built in** — detected as ring-1 sitting at **+60°** (not −120°)
   in the header basis.  R1 removed the Pie/Hex asymmetry, so the old
   two-term XOR is gone.  The rule accounts for all 35 classified decks
   (36 audited; `6MST_segV3` is unclassifiable, see the table):

   | ring-1 in header basis | decks | measured |
   |---|---|---|
   | +60° | 26 | CORRECT, 26/26 |
   | −120° | 2 (`ff_hex2` ×2) | PERMUTED, 2/2 |
   | other (not an SMM walk: `SegDemo*`, `FFSegDemo*`, legacy `old_Rx`) | 8 | CORRECT, 7/7 measured — not predicted; `6MST` (Flower) unclassifiable |

   Negating a header flips that one live term, so it always flips the
   verdict — see *Header negation* below (this supersedes round 1's
   "a header flip cannot fix any deck").

3. **The only decks ever broken were the SOUTH-layout pie decks**, under
   the *old* engine, via the `PSEG` bug.  That is exactly what Dave
   originally reported ("Seg2 rotating about `RptElt(:,5)`").  The R1
   `PSEG` fix plus their original `+1` headers repairs them.

4. **`SegMirMaker` emission was already correct** (post-`11481cd`).  The
   interim "engine-frame rule" (`8a2b462`) was reverted in `a8c04a9`.
   What was kept from that work: `WriteSegBlock` now emits per-segment
   GRID frames (`pData/x/y/zData` = the clocked Mon triad; the FF frame
   stays the parent's) — Dave's directive.

5. **There is no draw-path engine bug.**  `macos.draw_rays`'s element
   labels are correct; what is 180° off is a *reader* that treats the
   2-D projection `b.U/b.V` as global X/Y.  Root-caused in T2 below.

6. **`e2e` segment rotations are not inert.**  They respond at the
   physically expected magnitude; the deck is simply in metres, so the
   same 5e-7 rad tilt produces ~2000× less OPD than on a millimetre
   deck.  See T3 below.

7. Routing does **not** depend on `nGridpts` (8-run matrix,
   63/64/127/128 × ±header, all identical).  The parity hypothesis is
   retracted.

---

## Instrument doctrine

Only physical arbiters.  Hard-won; every entry below is a rake that was
stepped on.

| # | Instrument | Use | Why it is trustworthy |
|---|---|---|---|
| **A** | **Aperture ray count** | decks with per-segment `ApType= Polygonal` | A permuted deck clips each segment against the polygon of the segment 180° across the ring and loses most of the pupil (e2e_pie 12082 → 1422; e5pie_polyap 12007 → 1453).  A scalar count: no frame, no angle, no array map. |
| **B** | **draw-3d global attribution** | everywhere | `macos.draw_rays3d` returns each crossing's true global (x,y,z) — the engine stores `Draw3DVec = RayPos` verbatim — next to its element label.  Compare each segment element's mean crossing position to its own `RptElt` and to its 180-partner's: nearest wins.  A distance comparison in the deck's own global frame. |
| C | draw-2d projection | diagnostic only | Demonstrates the round-1 trap; see T2. |
| D | tilt-pivot piston | diagnostic only | **Confounded** — see below. |

**B is validated in both directions.**  On `e5pie_polyap`, A and B both
say CORRECT; on a byte-identical copy with the header `SegXgrid`
negated, A and B both say PERMUTED, and B's own/partner distances swap
exactly (elt 4: 64.4 / 5406 → 5406 / 64.4).  A and B agree on **every**
deck that carries both.

### Never use these

- **`draw_rays` `b.U`/`b.V` as global coordinates.**  They are the
  SOURCE / RAY-GRID projection (`xDraw=xGrid`, `yDraw=yGrid` for plane
  `'XY'`, `macos_cmd_loop.inc` DRAW).  See T2.
- **Any clock measured in the header `SegXgrid` basis** as a routing
  reference.  That basis *rotates with the patch*, so it is blind to
  exactly the quantity under test.  Legitimate for one question only:
  "does the header name the basis the elements were built in" (the
  `ring1Clk` / term1 test).
- **Any OPD-array angle without a per-deck map pin.**  The array's
  `(i,j) → (X,Y)` map varies with the deck's grid vectors *and* its
  lattice: the notes 63-pt deck reads `(i=−X, j=+Y)`; `test_in` at 128
  reads `(+X, −Y)`.
- **Midpoint-threshold wedge masks on a ramp response.**  A ramp splits
  its own wedge.  Define the region from a clean PISTON poke; only then
  evaluate the ramp over it.

### Instrument D is confounded — measured, not assumed

The brief's tilt-pivot piston (rotate segment *k* about local y with the
pivot at its own `RptElt` vs at the 180-partner's; expect
`|piston| ∝ dist(pivot, responding wedge)`) is a *flat-mirror* law.  On
these decks it does not discriminate:

| elt | partner | piston @ own | piston @ 180 | difference |
|---|---|---|---|---|
| 2 | 5 | −3.5744e−3 | +1.0232e−3 | −4.598e−3 |
| 5 | 2 | +1.3503e−3 | +5.8891e−3 | −4.539e−3 |
| 7 | 4 | −1.0774e−3 | +3.4887e−3 | −4.566e−3 |

(`e5pie_polyap`, tilt 5e−7 rad, pivot separation 5333 mm.)

Two independent reasons:

1. **The pivot shift is common-mode.**  Moving the pivot by `Δ` adds a
   rigid-body displacement `θ ŷ × Δ` — a *uniform* piston over the whole
   surface.  It carries no information about where the responding wedge
   is, which is why the three differences agree to 1 % regardless of
   segment.  Only the *absolute* readings could discriminate, and:
2. **Neither absolute reading goes to zero on a fast curved parent.**
   Rotating a `Element= Segment` re-points the whole parent conic; the
   residual is dominated by surface curvature, not by pivot-to-wedge
   distance.  (`seg_audit.m`'s check-6 header has flagged this since
   round 1.)

Plus, on aperture-carrying decks `set_elt_rpt` also **moves the aperture
polygon**, so the two legs are not the same optical system.

`seg_route` keeps D behind `'pivot',true` and reports the zero-crossing
`t*` for the record.  It is not a verdict.

---

## T2 — the draw-path attribution "bug" is a frame slip in the reader

**Not an engine defect.  No fix required in the engine.**

`macos_cmd_loop.inc`'s DRAW command projects each crossing onto a
2-vector before storing it:

```fortran
ELSE IF (LCMP(ANS,'XY',2)) THEN
  k=3; ANS='XY'
  If (.not.LudDrawGrid) Then
    CALL DEQUATE(xDraw,xGrid,3)    ! <-- the RAY-GRID basis, not global x
    CALL DEQUATE(yDraw,yGrid,3)
  Else                             ! ... or a user-supplied draw grid
    xDraw(1:3)=UDxDrawGrid(1:3); yDraw(1:3)=UDyDrawGrid(1:3)
  End If
```
(`'YZ'` and `'XZ'` are the same story with `zGrid`/`yGrid` and
`zGrid`/`xGrid`.)
and in `CTRACE`:
```fortran
DrawRay(1,nDrawElt)=DDOTC(xDraw,RayPos(1,iRay))
DrawRay(2,nDrawElt)=DDOTC(yDraw,RayPos(1,iRay))
```

So `b.U = RayPos · xGrid`, `b.V = RayPos · yGrid`.  **The whole `e5mono`
family and the `e2e` family have `xGrid = (−1,0,0)`**, so `b.U` is
*minus* global X.  Reading `b.U` as a global X mirrors the pupil about
the Y axis — and on the one segment pair that straddles the X axis
(elements 4 and 7 of a 7-segment pie, the pair round 1 tabulated) a
mirror in X is indistinguishable from a 180° rotation.  That is the
whole of the "180° permutation".

Measured on every deck in the sweep, `[C]`: max residual of the mean
draw crossing against the deck's own `RptElt` —

| deck family | read in the DRAW basis (xGrid,yGrid) | read as global x,y | segment radius |
|---|---|---|---|
| e5 pie / e5hex1 / e5_seg (mm) | 70 – 94 | 5385 – 5406 | 2667 |
| e5hex2 / ff_hex2 (mm) | 63.8 | 6436 | 3200 |
| e2e pie family (m) | 0.0446 | 2.703 | 1.333 |
| e2e_hex2 (m) | 0.0280 | 3.218 | 1.600 |

The global-frame reading is off by *exactly* twice the segment radius —
a full flip — while the DRAW-basis reading matches to ~3 % of the
radius (the residual is just the fan sampling only a chord of each
segment).  The element labels themselves (`DrawEltVec`) are written
from `imin` in the non-sequential block and `iElt` in the sequential
one, both correct; the `iElt`-vs-`imin` hypothesis is **excluded**.

### Consumers

The trap is already documented once in the tree.  `Telescope.m:355-360`
records that `center_focal_plane` read the DRAW projection -- "the
2026-07-18 heritage `xGrid=(-1,0,0)` emission flipped its U axis and sent
the FP a metre off" -- and that call site was fixed by switching to
`get_ray_info` global positions.

The design layer's response to that incident was to **pin its own
emission to `xGrid = (+1,0,0)`** (`Telescope.m:2762-2771`, with a NOTE
saying so explicitly), precisely because "draw_rays plot-axis signs
follow the grid handedness and several consumers were built on it".  So
on a design-layer-built deck `b.U` really *is* global X and the sites
below are correct **by construction**.

They are not defects today; they are unguarded assumptions.  They read
`b.U/b.V` as global x,y and would be wrong on any deck with
`xGrid_x < 0` -- which means the whole heritage/SMM corpus, and, notably,
**`segment_rx` output**: `segment_rx.m:212-216` deliberately flips the
merged deck's `xGrid` to `-1.0` to meet the SegMirMaker/engine tiling
contract.

| site | what it does | when it would bite |
|---|---|---|
| `Telescope.m:1991-1998` (`realize_apertures`) | `b.U/b.V` min/max -> aperture bounds, labelled `% U=X, V=Y` | mirrored bounds; today apertures are realized BEFORE `segment_rx` flips the grid, so the ordering is what keeps it safe |
| `Telescope.m:2532-2540` | `mean(b.U)` differenced against `e.Vpt(1)` -> off-axis pole/normal | direct global-vs-grid mix |
| `Telescope.m:2310` | `b.U - e.Vpt(cU)` -> view clipping extents | cosmetic (plot extents) |

`macos.view_rx` and `ray_hist` are clean: `view_rx` uses `draw_rays3d`
(true global 3-D).

**Recommendation:** no engine change.  Sharpen the `draw_rays` docstring
to name the projection basis as `xGrid`/`yGrid` explicitly (it currently
says only "source coords"), and either move the three sites onto
`draw_rays3d`/`get_ray_info` or have them assert `xGrid == +x_hat`.
Outside this brief's scope -- flagged for Dave.

---

## T3 — `e2e` segment rotations are not inert, they are metric

`macos.perturb(seg,'rotation',[0;5e-7;0])` on `e2e_pie` elt 2:

| deck | BaseUnits | segment radius | trans local | rot local | rot global |
|---|---|---|---|---|---|
| `e2e_pie` | m | 1.333 | 4.289e−4 | **9.313e−7** | 6.056e−7 |
| `e5pie_polyap` | mm | 2667 | 4.313e−4 | **4.258e−3** | 1.217e−3 |

Both match the rigid-tilt prediction `2·θ·d` in their own units
(e2e: 2·5e−7·1.333 = 1.33e−6 vs 9.3e−7 measured; e5: 2·5e−7·2667 =
2.67e−3 vs 4.26e−3).  The 4600× ratio between the two decks is the
unit change (m vs mm), not a defect.  **The rotation path works on both
families.**

The one place a *genuine* bit-zero was reproduced is the
SegXgrid-flipped `e5pie_polyap` control, where elt 2 is inert to
translation *and* rotation alike — because on the permuted deck the
mismatched aperture polygon clips away every one of that element's
rays.  That is the signature to look for: **inert to everything =
no rays, not a broken perturbation path.**

### The step sweep settles it

`e2e_pie` elt 2, local and global rotation about y, `nnz(dOPD)` = the
full 12082-pixel pupil at every step:

| tilt (rad) | peak dOPD, local | dOPD/theta | peak dOPD, global | dOPD/theta |
|---|---|---|---|---|
| 1e-10 | 1.862706e-10 | 1.862706 | 1.211270e-10 | 1.211270 |
| 1e-09 | 1.862694e-09 | 1.862694 | 1.211277e-09 | 1.211277 |
| 1e-08 | 1.862697e-08 | 1.862697 | 1.211279e-08 | 1.211279 |
| 1e-07 | 1.862691e-07 | 1.862691 | 1.211275e-07 | 1.211275 |
| 1e-06 | 1.862645e-06 | 1.862645 | 1.211228e-06 | 1.211228 |
| 1e-05 | 1.862178e-05 | 1.862178 | 1.210758e-05 | 1.210758 |
| 1e-04 | 1.857512e-04 | 1.857512 | 1.206060e-04 | 1.206060 |

Perfectly linear over six decades, to six significant figures, with no
floor and no threshold; translations behave the same down to 1e-12 m.
The OPD array's own resolution is 1.8e-15 (relative 4.4e-13) -- double
precision, so nothing is being truncated.  **There is no inertness on
this deck.**

> Narrowed, still open: the original "bit-zero" observation is not
> reproducible through `macos.perturb`.  If it came from the CLI `PERT`
> command, from a `dw_dx` channel, or from a differently-built deck,
> that is a different path and the question should be re-scoped to it.
> The one mechanism that DOES produce a genuine bit-zero here is rays
> being clipped away entirely (see the flipped-aperture control above) --
> check the responding-pixel count before calling anything inert.

---

## The `PSEG` mechanism (round-1 work, retained)

### `PSEG` ignored `SegXgrid`; `HSEG` honoured it

`sourcsub.F:201` builds the direction cosines of the header `SegXgrid`
in the **ray-grid** basis and hands them to the grid-type predicate:

```fortran
SegX2(1) = DDOTC(xGrid, SegXgrid)
SegX2(2) = DDOTC(yGrid, SegXgrid)
```

Both predicates opened with the identical rotation — `HSEG` at
1236-1237, `PSEG` at 1280-1281:

```fortran
xt =  x*SegXgrid(1) + y*SegXgrid(2)
yt = -x*SegXgrid(2) + y*SegXgrid(1)
```

**`HSEG` then used `xt`/`yt` throughout.  `PSEG` never used them again** —
`slopex`, `bL`, `bR` and all six inequality tests used the raw `x`, `y`.
The two assignments were dead stores.  So `GridType= Hex` applied the
frame rotation and `GridType= Pie` silently did not; the two diverge by
180° whenever `xGrid · SegXgrid < 0`.  `FSEG` (Flower) was always
correct.

**R1 (landed, `1ce1f5c`)** makes `PSEG` use the `xt`/`yt` it already
computes, exactly as `HSEG` does.

A deck is immune to R1 unless `GridType` is Pie **and** `SegX2` is not
the identity — i.e. unless `SegXgrid` is non-parallel to `xGrid`.  R1
touches `PSEG` only, so no Hex deck can move.  All movers in the corpus
rotate by exactly 180°, which is a symmetry of both the 7-segment pie
and the hex lattice, so R1 permutes segment *indices* only: ray counts,
gaps, RMS OPD, spot and PSF stay bit-identical.  `seg_audit` flags
`MOVES GEOMETRY` if a deck ever appears whose rotation is not a lattice
symmetry.

### Why it hides

Rays are assigned to segment elements **by ordinal, not by geometry**:

```fortran
tracesub.F:3455   IF (RayToSegMap(iRay,i).EQ.EltToSegMap(iElt)) ifRayToSeg=.TRUE.
```
(identically `propsub.F:590`, `srtrace.F:395`)

and every segment of an SMM deck defines the **same global surface** —
shared `VptElt`, `psiElt`, `KrElt`, `KcElt` and one replicated FF/grid
frame; only `RptElt`/`pMon`/`TElt` (bookkeeping) differ.  A
mis-ordinalled ray therefore reflects off exactly the surface it would
have hit anyway.  Nominal traces, ray counts, RMS OPD, spot diagrams
and the whole "masks are clean / union = 1.000" picture are unaffected.
The error appears only when a **single** segment is perturbed — i.e.
exactly what sensitivities, `dw_dx`, MET and the s4–s7 runners compute.

**Unless the deck carries per-segment apertures.**  Then the mismatch
is loud in the nominal trace (an ~88 % ray loss), which is why
instrument A is the gold standard where it exists.

### Header negation: what it actually does

Negating the header `SegXgrid` does **not** move the elements -- their
`RptElt`s are fixed text.  It rotates the *engine's* ray->segment map by
180 deg (via `SegX2`).  So post-R1 it **flips the verdict**, in both
directions.

Measured, current engine, valid instruments: a byte-identical copy of
`e5pie_polyap` with the header negated goes CORRECT -> PERMUTED, on both
A and B, with B's own/partner distances swapping exactly.  That is why
`seg_route` uses the flip as a **probe** (instrument A) -- it is a
reliable A/B, and on a correct deck it always makes things worse.

Round 1 reported the negation as a *no-op on Hex* and concluded "a
header-only basis flip cannot fix any deck".  That measurement came from
the draw-inverted instrument, on candidate decks that no longer exist,
and the reasoning rested on the two-term XOR that has since been
retracted.  **Re-measured in round 2 on `ff_hex2` (Hex, the one
permuted deck):**

| `ff_hex2` (Hex, 19 seg) | instrument B verdict | elt 4 own / partner | elt 18 own / partner |
|---|---|---|---|
| committed header | **PERMUTED** | 3200 / 43.8 | 6400 / 53.3 |
| header negated | **CORRECT** | 43.8 / 3200 | 53.3 / 6400 |

The distances swap exactly and the verdict flips.  **A header negation
DOES repair a permuted Hex deck** — round 1's "no-op on Hex" was the
draw-inverted reading of a candidate deck that no longer exists.  So
`ff_hex2` has a one-line repair available as well as the regeneration
route; which one to take is Dave's call (regeneration also picks up the
current per-segment grid frames).

---

## Corpus routing table

Every deck at its **committed** header, current engine.  One MATLAB
process per deck.  `[A]` shows the committed / SegXgrid-flipped pupil
counts.

| deck | Grid | nSeg | ngpt | SegX2 | [A] aperture (kept/flipped) | [B] draw3d | VERDICT |
|---|---|---|---|---|---|---|---|
| `macos/docs/macos-manual/FreeFormPieAperture.in` | Pie | 7 | 128 | -180.0 | - | CORRECT | **CORRECT** |
| `MACOS_res_dev/mmacos/design/examples/e2e/e2e_hex2.in` | Hex | 19 | 128 | +0.0 | CORRECT  (9830/480) | CORRECT | **CORRECT** |
| `MACOS_res_dev/mmacos/design/examples/e2e/e2e_pie.in` | Pie | 7 | 128 | +0.0 | CORRECT  (12082/1422) | CORRECT | **CORRECT** |
| `MACOS_res_dev/mmacos/design/examples/e2e/e2e_pie_met.in` | Pie | 7 | 128 | +0.0 | CORRECT  (12082/1422) | CORRECT | **CORRECT** |
| `MACOS_res_dev/mmacos/design/examples/e2e/e2e_pie_metopt.in` | Pie | 7 | 128 | +0.0 | CORRECT  (12082/1422) | CORRECT | **CORRECT** |
| `MACOS_res_dev/mmacos/design/examples/e5_seg/e5_seg_met.in` | Hex | 7 | 128 | +0.0 | - | CORRECT | **CORRECT** |
| `MACOS_res_dev/mmacos/design/examples/e5_seg/e5_seg_metopt.in` | Hex | 7 | 128 | +0.0 | - | CORRECT | **CORRECT** |
| `MACOS_res_dev/GMI/e2e_pie_met.in` | Pie | 7 | 128 | +0.0 | CORRECT  (12082/1422) | CORRECT | **CORRECT** |
| `MACOS_res_dev/GMI/ff_pie.in` | Pie | 7 | 128 | -180.0 | - | CORRECT | **CORRECT** |
| `MACOS_res_dev/GMI/regression/Rx/e2e_pie_met.in` | Pie | 7 | 128 | +0.0 | CORRECT  (12082/1422) | CORRECT | **CORRECT** |
| `MACOS_res_dev/GMI/regression/Rx/Rx_e5hex1.in` | Hex | 7 | 256 | -180.0 | - | CORRECT | **CORRECT** |
| `MACOS_sandbox/notes_luis_opd/e5pie.in` | Pie | 7 | 63 | -180.0 | - | CORRECT | **CORRECT** |
| `MACOS_sandbox/old_Rx/btc3.in` | Hex | 37 | 65 | +90.0 | - | CORRECT | **CORRECT** |
| `MACOS_sandbox/old_Rx/dmt6seg1313dm_centered.in` | Hex | 19 | 1024 | +90.0 | - | CORRECT | **CORRECT** |
| `MACOS_sandbox/old_Rx/j18dcWithStop.in` | Hex | 38 | 1024 | -90.0 | - | CORRECT | **CORRECT** |
| `MACOS_sandbox/old_Rx/j18sa.in` | Hex | 19 | 127 | +90.0 | - | CORRECT | **CORRECT** |
| `MACOS_sandbox/old_Rx/j18sc.in` | Hex | 19 | 1024 | -90.0 | - | CORRECT | **CORRECT** |
| `MACOS_res_dev/mmacos/design/examples/e2e/s4_grid.in` | Pie | 7 | 128 | +0.0 | CORRECT  (12082/1422) | CORRECT | **CORRECT** |
| `MACOS_res_dev/mmacos/examples/sensitivities/e5hex1/e5hex1.in` | Hex | 7 | 256 | -180.0 | - | CORRECT | **CORRECT** |
| `MACOS_res_dev/mmacos/examples/sensitivities/e5hex1/e5hex1_grid.in` | Hex | 7 | 256 | -180.0 | - | CORRECT | **CORRECT** |
| `MACOS_res_dev/mmacos/sensitivities/examples/e5hex2_refzern/e5hex2.in` | Hex | 19 | 128 | -180.0 | - | CORRECT | **CORRECT** |
| `MACOS_res_dev/mmacos/sensitivities/examples/e5hex2_refzern/e5hex2grid.in` | Hex | 19 | 128 | -180.0 | - | CORRECT | **CORRECT** |
| `MACOS_res_dev/segmirmaker/test_in/e5hex2.in` | Hex | 19 | 128 | -180.0 | - | CORRECT | **CORRECT** |
| `MACOS_res_dev/segmirmaker/test_in/e5pie.in` | Pie | 7 | 128 | +0.0 | - | CORRECT | **CORRECT** |
| `MACOS_res_dev/segmirmaker/test_in/e5seg1.in` | Hex | 7 | 128 | -180.0 | - | CORRECT | **CORRECT** |
| `MACOS_res_dev/mmacos/examples/view_rx_demo/e5hex1.in` | Hex | 7 | 256 | -180.0 | - | CORRECT | **CORRECT** |
| `MACOS_res_dev/mmacos/examples/view_rx_demo/e5pie.in` | Pie | 7 | 128 | +0.0 | - | CORRECT | **CORRECT** |
| `macos/ZGD_test_files/FFSegDemoAll.in` | Pie | 7 | 256 | +0.0 | - | CORRECT | **CORRECT** |
| `macos/ZGD_test_files/SegDemo3.in` | Pie | 7 | 256 | +0.0 | - | CORRECT | **CORRECT** |
| `macos/ZGD_test_files/e2e_pie.in` | Pie | 7 | 128 | +0.0 | CORRECT  (12082/1422) | CORRECT | **CORRECT** |
| `macos/ZGD_test_files/e5hex1.in` | Hex | 7 | 256 | -180.0 | - | CORRECT | **CORRECT** |
| `macos/ZGD_test_files/e5hex2.in` | Hex | 19 | 128 | -180.0 | - | CORRECT | **CORRECT** |
| `macos/ZGD_test_files/e5pie.in` | Pie | 7 | 128 | -180.0 | - | CORRECT | **CORRECT** |
| `MACOS_sandbox/old_Rx/6MST_segV3.in` | Flower | 19 | 511 | +0.0 | - | - | **UNDETERMINED** |
| `MACOS_res_dev/segmirmaker/test_in/ff_hex2.in` | Hex | 19 | 128 | -180.0 | - | PERMUTED | **PERMUTED** |
| `macos/ZGD_test_files/ff_hex2.in` | Hex | 19 | 128 | -180.0 | - | PERMUTED | **PERMUTED** |

`[A]` = pupil pixels at the committed header / with the header `SegXgrid`
negated.  `[B]` = draw-3d global attribution.  Where both ran they agree,
on every deck.  A dash under `[A]` means the deck has no per-segment
polygon apertures, so that instrument does not exist for it.

**`6MST_segV3.in` (Flower) is not classifiable by instrument B** — a
petal layout has no 180° symmetry, so the partner map degenerates (worst
nearest-match error 7.98e3 against a 5157 radius).  `seg_route` now
refuses to vote when the partner match is worse than 5 % of the ring
radius; an earlier run without that guard printed a meaningless
"CORRECT".  Left unclassified.

`btc3.in` is classic-Mac **CR-only**; `seg_read_rx` now normalises line
terminators, without which the whole deck parses as one key and no
segments are found.

### The `[C]` frame demonstration, whole corpus

Max residual of the mean draw-2d crossing against the deck's own
`RptElt`, read **in the DRAW basis (xGrid,yGrid) / as global x,y**:

```
  docs_FFPieAperture       94.38 / 5406  (r=2667)
  e2e_hex2                 0.02801 / 3.218  (r=1.6)
  e2e_pie                  0.04455 / 2.703  (r=1.333)
  e2e_pie_met              0.04455 / 2.703  (r=1.333)
  e2e_pie_metopt           0.04455 / 2.703  (r=1.333)
  e5_seg_met               87.78 / 5395  (r=2667)
  e5_seg_metopt            87.78 / 5395  (r=2667)
  gmi_e2e_pie_met          0.04455 / 2.703  (r=1.333)
  gmi_ff_pie               94.38 / 5406  (r=2667)
  gmireg_e2e_pie_met       0.04455 / 2.703  (r=1.333)
  gmireg_e5hex1            70.49 / 5385  (r=2667)
  luis_e5pie               70.49 / 5385  (r=2667)
  old_6MST                 608.4 / 608.7  (r=5157)
  old_btc3                 465.3 / 464.6  (r=2646)
  old_dmt6seg              571.1 / 571.1  (r=2082)
  old_j18dc                630.2 / 4568  (r=2275)
  old_j18sa                567.4 / 567.3  (r=2078)
  old_j18sc                623.6 / 4520  (r=2275)
  s4_grid                  0.04455 / 2.703  (r=1.333)
  sens_e5hex1              70.49 / 5385  (r=2667)
  sens_e5hex1_grid         70.49 / 5385  (r=2667)
  sens_e5hex2              63.76 / 6436  (r=3200)
  sens_e5hex2grid          63.76 / 6436  (r=3200)
  ti_e5hex2                63.76 / 6436  (r=3200)
  ti_e5pie                 94.42 / 5406  (r=2667)
  ti_e5seg1                87.78 / 5395  (r=2667)
  vrx_e5hex1               70.49 / 5385  (r=2667)
  vrx_e5pie                94.42 / 5406  (r=2667)
  zgd_FFSegDemoAll         0.1985 / 0.1985  (r=1.225)
  zgd_SegDemo3             0.1985 / 0.1985  (r=1.225)
  zgd_e2e_pie              0.04455 / 2.703  (r=1.333)
  zgd_e5hex1               70.49 / 5385  (r=2667)
  zgd_e5hex2               63.76 / 6436  (r=3200)
  zgd_e5pie                94.38 / 5406  (r=2667)
  ti_ff_hex2               6436 / 63.81  (r=3200)
  zgd_ff_hex2              6436 / 63.74  (r=3200)
```

Three regimes, all as predicted:

- `xGrid = (−1,0,0)` (the e5mono and e2e families): grid-basis residual
  is a few % of the radius, the global reading is **2× the radius** — a
  full flip.  This is the round-1 trap.
- `xGrid = (+1,0,0)` (`SegDemo3`, `FFSegDemoAll`, `btc3`, `dmt6seg`,
  `j18sa`, `6MST`): the two readings are **identical**, because there
  `b.U` really is global X.  The control case.
- `ff_hex2`: the two readings **swap** (6436 / 63.8).  That is the
  fingerprint of a genuinely permuted deck, and it is independent
  confirmation of instrument B's verdict.


---

## The one live defect: `ff_hex2` is 180° permuted

Both lineages -- `macos/ZGD_test_files/ff_hex2.in` (`0be1fca7`) and
`MACOS_res_dev/segmirmaker/test_in/ff_hex2.in` (`78d16f36`) -- measure
**PERMUTED** by instrument B, and independently by the `[C]` residual
swap (6436 / 63.8 instead of 63.8 / 6436).

It is the only deck in the corpus whose ring-1 segment sits at **−120°**
in the header basis instead of +60° -- i.e. the only one whose header
`SegXgrid` does *not* name the basis its elements were built in.  That is
the stale-generation signature: SegMirMaker rotates its in-plane basis
180° for a back-facing parent (`zs(3) < 0`, commit `11481cd`) but older
builds emitted the *un-rotated* basis.  `ff_hex2` was generated in that
window.

Consequences and options:

- **Blast radius is small.**  A tree-wide grep finds no code reference to
  `ff_hex2` at all -- no test, no example, no runner.  It is an SMM
  regression fixture and a ZGD test file.  (Round 1's R5 listed it under
  `tFreeFormComposite` / `tReadGridFile`; that does not hold -- those
  classes do not name it.)
- **Nominal results are unaffected**, for the reason in "Why it hides":
  it has no per-segment apertures, and every segment is a piece of the
  same parent surface.  Only per-segment perturbations land wrong.
- **Two repairs are available**, both measured this round:
  *(a)* negate the header `SegXgrid` — verified to flip the verdict to
  CORRECT (see *Header negation* above); one line, but it leaves the
  deck's grid frames at their 2026-07 generation.
  *(b)* regenerate from `ffparent.in` with the current SMM — also picks
  up the per-segment grid frames from `49d0970`.
  Round 1's "a header flip is a Hex no-op" is retracted.
- Recommendation: **regenerate** (b).  Cheap, no consumers to break, and
  it brings the deck up to the current frame convention.  Dave's call;
  not done here.

---

## Geometry / frame findings (unchanged from round 1)

- **The trigger premise is wrong.**  `e5pie` does not ship with every
  segment at the parent vertex.  Only `VptElt` is shared at `(0,0,0)` —
  correct: it is the parent conic's vertex and every segment is a piece
  of that one conic.  `RptElt` **is** per-segment and lands on each
  segment's own centre (ring-1 radius = `width` exactly).  Element 1 is
  at the origin because it *is* the ring-0 centre segment.
- **`RptElt` is honoured as the rotation pivot.**  Forcing it to the
  parent vertex changes the OPD response by 30–200 % on every ring
  segment.
- **Localization is clean everywhere:** zero mask overlap, masks union
  to 1.000 of the pupil, correct piston sign and magnitude.
- **`gap` is not part of the segment pitch, and that is correct.**  SMM
  sets `dx = width/2`, `dy = 3·length/4`, giving ring-1 centre spacing =
  `width` exactly; `HSEG` uses the same convention.  So `width` is the
  **pitch** and the physical segment is `width − gap` flat-to-flat.
- **Check 4, whole corpus:** every SMM deck shares ONE `pFF/xFF/yFF/zFF`
  frame across all segments.  This is **correct for SMM** — design
  choice 4 replicates the *parent's* FF coefficients and grid into every
  segment, so they must be evaluated in the parent's frame.  It is the
  opposite of the `segment_rx` convention (per-segment grids in the
  clocked Mon frame), so the e2e localization gotcha does not apply.
  `zFF · zMon = −1` on `e5mono`-derived decks and `+1` on `e2e`-derived
  ones — a **parent-deck** convention difference, not SMM behaviour.
- **Frame convention (Dave-verified on-deck):** `xMon` radial,
  `yMon` = radial − 90° (tangential), `zMon` = `psi` = the OUTWARD
  normal; the grid frame equals the clocked Mon triad; the FF frame is
  the parent's.  Seven pie decks were re-clocked to this
  radial-bisector convention (`1890dbc`/`bcdd0cd`,
  res_dev `199ea6f`/`49d0970`).
- **Engine reload instability.**  A second `macos.load_rx` of a
  256-grid segmented deck can kill the process.  Run **one deck per
  process**.  `seg_route` picks `model_size` 1024 for `nGridpts > 128`.

---

## T5 — mmacos suite against the corrected tree

Full suite, `MACOS_BUILD_DIR=build_release_gfortran`.

| group | result |
|---|---|
| fast (size 128) | 260 pass, **2 fail** |
| masks (size 128) | 62 pass, 0 fail |
| freeform (size 256) | 99 pass, 0 fail |
| proper Cass-FF (512) | 10 pass, 0 fail |
| pupil aperture (512) | 5 pass, 0 fail |
| proper Coro (1024) | 13 pass, 0 fail |
| **total** | **449 pass, 2 fail** |

The 2 failures are `tSegMirMaker/test_pie_reproduces_committed` and
`test_hex2_reproduces_committed`, both `MATLAB:fileread:cannotOpenFile`.
**No reference moved and nothing regressed** — in particular
`tSegmentRx`, `tDwDx`, `tDwDxGroups`, `tRunSensitivities`, `tFingerprint`
and `tReadGridFile` are unchanged against the corrected tree, and the
GMI references (`0e9df7b`) were not touched.

Two test-infrastructure problems surfaced, both **pre-existing and
unrelated to the routing work**:

1. **`tSegMirMaker`'s byte-identity references are gitignored.**  The two
   `*_reproduces_committed` tests read `test_in/e5pie.presc`,
   `e5hex2.presc` and `*Hx.m` and call them "the committed reference".
   `.gitignore:21-24` says the opposite, explicitly:

   ```
   # SegMirMaker run artifacts (keep test_in/*.in inputs tracked)
   segmirmaker/test_in/*.presc
   segmirmaker/test_in/*Hx.m
   ```

   So on a worktree where nobody ran the generator by hand the tests
   ERROR; where somebody did, they compare against a local artifact of
   unknown provenance.  (`MACOS_resources` has a pair dated 2026-07-23,
   generated from a branch whose `SegMirMaker.f` differs from `dev`'s.)
   The test's premise and the ignore rule's stated intent contradict each
   other; one of them has to give.  Not resolved here — minting a fresh
   reference would bake in this week's frame re-clocking, which is Dave's
   call.

2. **Five test classes could only find an `ifx` SegMirMaker build**, so on
   a gfortran box every segmentation / MET class skipped — and a skip
   reads as green.  19 tests across `tSegmentRx`, `tEdgeSensors`, `tMet`,
   `tRunMet` and `tRunSegmentation` were "Filtered by assumption" in the
   baseline run.  `tSegMirMaker`, `tRunCompare` and `tRunSensitivities`
   already searched all four build trees; the other five hard-coded
   `build_release_ifx`.  **Fixed** here: new
   `mmacos/tests/private/segmirmaker_bin.m` does the four-tree search
   once, and the five classes call it.  With the fix those classes
   actually execute — and **all pass against the corrected tree**:

   | class | before | after |
   |---|---|---|
   | `tSegmentRx` | 11 filtered | **12 pass, 0 fail** |
   | `tMet` | 1 filtered | **7 pass, 0 fail** |
   | `tRunMet` | 3 filtered | **3 pass, 0 fail** |
   | `tEdgeSensors` | 3 filtered | **3 pass, 0 fail** |
   | `tRunSegmentation` | 1 filtered | **1 pass, 0 fail** |

   That is 26 segmentation / MET tests that had never run on a gfortran
   box, green on the re-clocked decks and the R1 engine.

### SegMirMaker emission has drifted, in exactly one respect

To find out whether the missing reference was hiding anything, a fresh
`.presc` was generated with each test's exact arguments and diffed
against the 2026-07-23 `MACOS_resources` (pol-core) copy:

| deck | differing lines | of total | what differs |
|---|---|---|---|
| `e5pie` | 56 | 370 | 7 segments × `pData/xData/yData/zData` |
| `e5hex2` | 152 | 994 | 19 segments × `pData/xData/yData/zData` |

**Nothing but the grid-frame lines** — i.e. exactly `49d0970`
("SegMirMaker + pie decks: per-segment grid frames"), Dave's directive,
and nothing incidental rode along.  With a correctly-minted reference
`tSegMirMaker` is 3 pass / 0 fail, so the generator and the test
machinery are both sound; only the reference's provenance is broken.
The generated files were **deleted again** rather than left in the
worktree — a gitignored, self-generated reference would make a
byte-identity test pass vacuously, which is the exact failure mode this
audit exists to clean up.

---

## How round 1 went wrong

Worth recording, because the failure mode is generic.

1. Round 1's absolute placement reference was `macos.draw_rays('XY')`,
   described as "convention-free … the real DRAW ray fan in **global
   X-Y**, no array indexing involved".  It is not global; it is the
   ray-grid projection, and `xGrid = (−1,0,0)` on every deck in the
   comparison.
2. The comparison was tabulated on elements 4 and 7 — the ±X pair —
   where a mirror in X and a 180° rotation give the same answer.
3. The XOR rule (`term1 ⊕ term2`) was then *fitted* to those inverted
   measurements and reported as "validated 8/8".  A rule fitted to
   inverted data reproduces inverted data perfectly.
4. The secondary reference, `RptClk`, was measured in the header
   `SegXgrid` basis — which rotates with the patch — so it could not
   catch the error either.

The lesson, in the form it generalises: **a reported position or angle
is meaningless without its reference frame, and a "convention-free"
claim is exactly where to check hardest.**  Both instruments that
finally settled it (a scalar ray count; a distance between two points
in one named frame) were chosen because they have no frame to get
wrong.

---

## Reproducing

```matlab
addpath ~/dev/MACOS_res_dev/mmacos/src
addpath ~/dev/MACOS_res_dev/segmirmaker

% routing verdict -- ONE DECK PER MATLAB PROCESS
R = seg_route('~/dev/macos/ZGD_test_files/e5pie.in');
R = seg_route(deck, 'pivot', true);       % also run instrument D (diagnostic)
R.verdict, R.by, R.A, R.B, R.C, R.T3      % everything is in the struct

% geometry / frames / localization (does NOT decide routing)
A = seg_audit({deck});
A = seg_audit(decks, 'static_only', true);   % no engine, whole corpus, fast
```

The corpus sweep in this document was driven one process per deck:

```bash
SM=~/dev/MACOS_res_dev/segmirmaker
while read -r tag rx; do
  timeout 1200 matlab -batch \
    "addpath ~/dev/MACOS_res_dev/mmacos/src; addpath $SM; seg_route('$rx'); exit(0)" \
    > "log/$tag.log" 2>&1
done < decks.txt        # lines of:  <tag> <abs path to .in>
```

A fresh process per deck is not optional: a second `load_rx` of a
256-grid segmented deck can kill the session.

Regenerating a deck from its parent:

```bash
cd ~/dev/MACOS_res_dev/segmirmaker/test_in
MACOS_HOME=~/dev/macos/macos_f90 ../build_release_gfortran/SegMirMaker < e5pie.stdin
```

---

## Open items

- Three `Telescope.m` call sites read `draw_rays` `b.U/b.V` as global
  X/Y (T2 table above).  Safe today only because the design layer pins
  its emission to `xGrid=(+1,0,0)`; unguarded against a heritage deck or
  a `segment_rx` output (which flips `xGrid` to `-1`).  Pre-existing;
  needs Dave's call because touching `realize_apertures` moves saved
  aperture numbers.
- The original `e2e` "bit-zero rotation" observation is not reproduced
  through `macos.perturb`; re-scope it to whichever path produced it.
- The `MACOS_resources` worktree is on branch `pol-core` and still
  carries the PRE-re-pin `e5pie` headers.  Checked: those files are
  untouched on `pol-ifo` since the merge base, so the pending
  `pol-ifo -> dev` merge takes `dev`'s versions cleanly -- no conflict,
  no reintroduction.  Re-check if either tip moves.
- `MACOS_sandbox/old_Rx/` holds legacy Hex decks with `SegX2 = ±90°`,
  outside the 0°/180° model.  All five (`j18sa`, `j18sc`,
  `j18dcWithStop`, `btc3`, `dmt6seg1313dm_centered`) were **measured**
  this round and route CORRECT.  R1 could not touch them (Hex), but 90°
  is not a hex-lattice symmetry, so if anyone ever changes `HSEG` they
  must be re-measured, not predicted.
- `6MST_segV3.in` (Flower) is unclassified: a petal layout has no 180°
  partner, so instrument B has nothing to compare against.  A
  Flower-specific instrument (or instrument A, if it were given
  per-segment apertures) would be needed.
- `ff_hex2` (both lineages) is 180 deg permuted and needs a decision:
  negate its header (measured to fix it) or regenerate.  No consumers, so
  nothing is blocked on it.
- Three different `e5pie.in` lineages remain in circulation
  (`de3008b8` south / `2844a5b2` north / `c3049f3a` the 63-pt notes
  deck).  They number their segments 180° apart.  Collapse them, or
  rename so the generation is visible in the filename.
