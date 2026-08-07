# SegMirMaker segmentation audit — status

Audit of the SegMirMaker (SMM) segmented-mirror corpus requested in
`macos/BRIEF_seg_audit.md`.  Script: `seg_audit.m` (this directory).
Engine tree `macos dev` @ e1d3c2b; SMM binary
`build_release_gfortran/SegMirMaker` built from `9ca9b64` (current tip).

**Nothing in the committed corpus was modified.**  Read-only audit plus a
recommendation list.

---

## Headline

1. **The trigger premise is wrong.**  e5pie does *not* ship with every
   segment at the parent vertex.  Only `VptElt` is shared at `(0,0,0)` —
   which is **correct**: it is the parent conic's vertex and every
   segment is a piece of that one conic.  `RptElt` **is** per-segment and
   lands on each segment's own centre (ring-1 radius = `width` exactly,
   clocks 60/120/180/−120/−60/0).  Element 1 is at `(0,0,0)` because it
   *is* the ring-0 centre segment.  `macos.set_elt_rpt` is not needed.

2. **`RptElt` is honoured as the rotation pivot.**  A/B against the same
   local tilt with `RptElt` forced to the parent vertex changes the OPD
   response by 30–200 % on every ring segment; forcing `RptElt` to 10×
   the radius scales it further.  The engine pivots where the deck says.

3. **e5pie audits clean** on every optics-facing check: perfect
   localization (zero mask overlap, masks union to 1.000 of the pupil),
   correct piston sign and magnitude, positions matching `RptElt` to
   0.1°, Mon frames clocked as the deck states.

4. **The real defect is a 180° ray↔element permutation, and it is two
   bugs XOR'd** — one in the engine, one in SegMirMaker.  It is the
   concrete form of the long-flagged "latent 180° frames↔tiling offset".
   Measured on 8 decks, predicted exactly on all 8.
   **Currently permuted: `e5hex1`, `e5hex2`, `e5seg1`,
   `test_in/e5pie` (== `view_rx_demo/e5pie`), `view_rx_demo/e5hex1`,
   `test_in/e5hex2`.**  `e5pie`, `ff_hex2`, `e2e_pie`, `e2e_hex2`,
   `e5pie_polyap`, `e5_seg_met` and the pre-SMM `SegDemo*` family are
   correct.

5. **Nominal traces never reveal it** — see "Why it hides".  Only
   per-segment perturbations do, i.e. exactly what sensitivities,
   `dw_dx`, MET and the s4–s7 runners compute.

---

## The 180° permutation

### Bug A — engine: `PSEG` ignores `SegXgrid`, `HSEG` honours it

`sourcsub.F:201` builds the direction cosines of the header `SegXgrid`
in the **ray-grid** basis and hands them to the grid-type predicate:

```fortran
SegX2(1) = DDOTC(xGrid, SegXgrid)
SegX2(2) = DDOTC(yGrid, SegXgrid)
```

Both predicates open with the identical rotation —
`HSEG` at 1236-1237, `PSEG` at 1280-1281:

```fortran
xt =  x*SegXgrid(1) + y*SegXgrid(2)
yt = -x*SegXgrid(2) + y*SegXgrid(1)
```

**`HSEG` then uses `xt`/`yt` throughout.  `PSEG` never uses them again** —
`slopex`, `bL`, `bR` and all six inequality tests use the raw `x`, `y`.
The two assignments are dead stores.

So `GridType= Hex` applies the frame rotation and `GridType= Pie`
silently does not.  The two diverge by 180° whenever
`xGrid · SegXgrid < 0`.

**Isolated experimentally** — two one-variable copies of `ZGD/e5hex1.in`,
element blocks byte-identical in both:

| copy | change | measured offset |
|---|---|---|
| `x_hex1_asPie.in` | header word `Hex` → `Pie`, nothing else | **+0.00°** |
| `x_hex1_128.in` | `nGridpts` 256 → 128, still `Hex` | **±180.00°** |

`GridType` is the variable; grid size is not.

### Bug B — SegMirMaker: `SegXgrid` is written in the wrong frame

SMM emits `SegXgrid` in **global** coordinates (the in-plane basis `xs`
it placed the elements in).  The engine interprets it in the **ray-grid**
frame, via `xGrid`/`yGrid`.  SMM never consults `xGrid`.  The whole
`e5mono` family has `xGrid = (−1,0,0)`, so a globally-correct
`SegXgrid = (+1,0,0)` reaches the engine as `SegX2 = (−1,0,0)` — a
spurious 180°.

A second, related generation issue: SMM rotates its in-plane basis 180°
about `zs` for a back-facing parent (`zs(3) < 0`, commit `11481cd`) but
older builds still emitted the *un-rotated* basis.  Decks from that
window carry a flipped tiling with an unflipped header, detectable as
ring-1 sitting at −120° instead of +60° in the header basis.

### The rule (validated 8/8)

```
permutation = term1  XOR  term2
  term1 = 180  if the deck's ring-1 segment reads -120 deg (not +60)
               in the header SegXgrid basis          [ SMM generation ]
  term2 = 180  if GridType is Hex AND xGrid.SegXgrid < 0   [ PSEG bug ]
```

| deck | Grid | `xGrid·SegXgrid` | ring-1 clk | predicted | **measured** |
|---|---|---|---|---|---|
| `ZGD/e5pie.in` | Pie | −1 | +60 | ok | **+0.00°** ✔ |
| `ZGD/e5hex1.in` | Hex | −1 | +60 | permuted | **−180.00°** ✘ |
| `ZGD/e5hex2.in` | Hex | −1 | +60 | permuted | **−180.00°** ✘ |
| `ZGD/ff_hex2.in` | Hex | −1 | −120 | ok | **+0.00°** ✔ |
| `ZGD/e2e_pie.in` | Pie | +1 | +60 | ok | **+0.00°** ✔ |
| `ZGD/FFSegDemoAll.in` | Pie | +1 | −60 (pre-SMM walk) | n/a | **+0.00°** ✔ |
| `test_in/e5pie.in` | Pie | −1 | −120 | permuted | **+180.00°** ✘ |
| `e2e/e2e_hex2.in` | Hex | +1 | +60 | ok | **+0.00°** ✔ |

Measurement: per-segment `+zMon` piston probe, OPD at the first
non-segment element, mask centroid vs the segment's own `RptElt`.
Offset spreads are 0.05–0.15° throughout, so each verdict is a clean
rigid 0° or 180°, never a smear.

`ff_hex2` works only because two independent 180° errors cancel.

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
and the whole `masks are clean / union = 1.000` picture are unaffected.
The error appears only when a **single** segment is perturbed.

Convention-free confirmation via `macos.draw_rays('XY',1,7)` — the real
DRAW ray fan in **global X-Y**, no array indexing involved.  `e5pie` and
`e5hex1` have byte-identical element blocks:

| deck | elt 4 rays ⟨X⟩ | elt 4 `RptElt` X | elt 7 rays ⟨X⟩ | elt 7 `RptElt` X | |
|---|---|---|---|---|---|
| `ZGD/e5pie.in` (Pie) | −2668 | −2666.67 | +2739 | +2666.67 | match |
| `ZGD/e5hex1.in` (Hex) | **+2718** | −2666.67 | **−2680** | +2666.67 | **misrouted** |

On `e5hex1` the rays reaching element 4 are physically on the opposite
side of the pupil from where element 4 sits.

### Predicted status, whole corpus

| status | decks |
|---|---|
| **180° PERMUTED** | `ZGD/e5hex1.in`, `ZGD/e5hex2.in`, `test_in/e5hex2.in`, `test_in/e5seg1.in`, `test_in/e5pie.in`, `view_rx_demo/e5pie.in`, `view_rx_demo/e5hex1.in` |
| ok | `ZGD/e5pie.in`, `ZGD/ff_hex2.in`, `test_in/ff_hex2.in`, `ZGD/e2e_pie.in`, `e2e/e2e_pie.in`, `e2e/e2e_hex2.in`, `e2e/e2e_pie_met.in`, `e2e/s4_grid.in`, `e5_pie/e5pie_polyap.in`, `e5_seg/e5_seg_met.in` |
| n/a (pre-SMM walk, `xGrid·SegXgrid = +1` so Pie/Hex agree) | `SegDemo.in`, `SegDemo3*.in`, `FFSegDemo*.in` |

The MET / sensitivity workhorses `e5pie_polyap.in` and `e5_seg_met.in`
come out **ok** — the exposure is narrower than the corpus-wide staleness
suggests.

---

## Frame-convention staleness (separate from the permutation)

`e701f87` (Dave, 2026-07-18) made *Pie* tilings clock each segment's
`xMon` along its own radial bisector so same-shape wedges get congruent
frames; *Hex* keeps the heritage convention (`xMon` 30° off the
bisector, along a hex flat normal).  A freshly regenerated
`e5pie.presc` shows `dClk ≈ 0`; so does `e2e_pie.in`.

**All three committed `e5pie` decks and `e5pie_polyap.in` still carry
`dClk ≈ −30` — hex heritage frames under a `GridType= Pie` header.**
They predate the convention.  This matters for anything that defines a
figure in the Mon frame (Zernike/monomial coefficients, per-segment
influence bases, the s5 shape-class launcher patterns): a 30° frame
rotation changes the physical figure a given coefficient produces.

**Three different `e5pie.in` files are in circulation.**
`view_rx_demo/e5pie.in` == `segmirmaker/test_in/e5pie.in` (`c659bd7a`);
`ZGD/e5pie.in` is older and different (`a821bd9c`);
`notes_luis_opd/e5pie.in` is a third variant (`nGridpts=63`).
They do not agree on tiling generation, and the ZGD one is the only
correctly-routed member of the three.

---

## Per-deck audit table (brief's numbering)

Engine probes take OPD at the **first non-segment element**, so the
placement check is absolute (an erect, demagnified image of the
segmented pupil, no intervening focus).

| deck | 1 Rpt/Vpt | 2 clocking | 3 zMon sign | 4 frames | 5 localize | 6 pivot | 7 parity | routing |
|---|---|---|---|---|---|---|---|---|
| `ZGD/e5pie.in` | OK | OK 0.10° | OK 0.9 % | OK | OK 0 / 1.000 | OK | EVEN 128 | **OK** |
| `ZGD/e5hex1.in` | OK | OK 0.13° | OK 1.5 % | OK | OK 0 / 1.000 | OK | EVEN 256 | **FLAG 180°** |
| `ZGD/e5hex2.in` | OK | OK 0.15° | OK 1.6 % | OK | OK 0 / 1.000 | OK | EVEN 128 | **FLAG 180°** |
| `ZGD/ff_hex2.in` | OK | OK 0.15° | OK 1.0 % | OK | OK 0 / 1.000 | OK | EVEN 128 | OK (cancellation) |
| `ZGD/e2e_pie.in` | OK | OK 0.04° | OK 0.2 % | OK | OK 0 / 1.000 | OK | EVEN 128 | **OK** |
| `ZGD/FFSegDemoAll.in` | OK | OK 0.05° | OK | OK | OK 0 / 1.000 | OK | EVEN 256 | **OK** |
| `test_in/e5pie.in` | OK | OK 0.10° | OK | OK | OK 0 / 1.000 | OK | EVEN 128 | **FLAG 180°** |
| `e2e/e2e_hex2.in` | OK | OK 0.15° | OK | OK | OK 0 / 1.000 | OK | EVEN 128 | **OK** |

*Checks 2 and 5 pass on the FLAG decks too:* the masks are clean,
non-overlapping and complete — they are simply attached to the wrong
element ordinal.  That is why this survived so long, and why the audit
needed an absolute placement reference rather than a consistency test.

**Check 1 (static, whole corpus):** no non-central segment sits at the
parent vertex in any deck.  `pMon == RptElt` exactly (0.0) everywhere —
SMM design choice 3 holds.  `VptElt` shared at the parent vertex
everywhere, as intended.

**Check 4 (static, whole corpus):** every SMM deck shares ONE
`pFF/xFF/yFF/zFF` frame across all segments and `pData/…/zData` equals
it.  This is **correct for SMM** — design choice 4 replicates the
*parent's* FF coefficients and grid into every segment, so they must be
evaluated in the *parent's* frame.  It is the opposite of the
`segment_rx` convention (per-segment grids in the clocked Mon frame),
so the e2e localization gotcha does not apply here.
`zFF · zMon = −1` on `e5mono`-derived decks and `+1` on `e2e`-derived
ones — a **parent-deck** convention difference (`e5mono` writes
`zFF = −psiElt`, the e2e parent writes `zFF = +psiElt`), not SMM
behaviour.

**Check 7 parity:** *every* deck in the corpus uses an even `nGridpts`
(128 or 256) — all in the half-pixel class closed by the even-grid
centre fix.

---

## Corrections to earlier notes

- **`project_opd_conventions` — "global-frame rigid pokes are inert on
  `Element=Segment`".**  Not reproduced.  Via
  `macos.perturb(e,'translation',[0;0;d],'frame','global')`
  (→ `CPERTURB_PROG`) every e5pie segment 1–7 responds, at the same
  magnitude as the local-frame poke and with the opposite sign — as
  expected, since these segments have `zMon·ẑ_global = −0.996`.  Local
  `+z` gives wedge `−4.18e-4`; global `+z` gives `+4.14e-4`.  If the
  original observation came from the CLI `PERT` command rather than the
  programmatic path, that is a different code path and the open question
  should be re-scoped to it.

- **`gap` is not part of the segment pitch, and that is correct.**  SMM
  sets `dx = width/2`, `dy = 3·length/4`, giving ring-1 centre spacing =
  `width` exactly; `HSEG` uses the same convention (centres on a lattice
  of pitch `width`, each hexagon shrunk inward by `gap/2` per side).  So
  `width` is the **pitch** and the physical segment is `width − gap`
  flat-to-flat.  Measured ring radii are `1.00000 × width` corpus-wide.

- **Engine reload instability.**  A second `macos.load_rx` of a
  256-grid segmented deck inside one MATLAB session kills the process
  (reproduced on `e5hex1`, ~41 700 rays; survived at `model_size` 1024,
  not at 512).  Same family as the known model-size-transition state
  leak.  Run `seg_audit` **one deck per process**.

---

## Recommendations (nothing applied — Dave's call)

**R1 — engine: make `PSEG` and `HSEG` agree.**  Mechanically it is three
lines in `PSEG` (use `xt`/`yt` in `slopex`, `bL`, `bR` and the six
inequality tests, exactly as `HSEG` does).  **Do not land it alone** — it
re-permutes every Pie deck whose `xGrid·SegXgrid < 0`, which today means
`ZGD/e5pie.in` would go from correct to permuted and
`test_in/e5pie.in` from permuted to correct.  Land it with R2.

**R2 — SegMirMaker: emit `SegXgrid` in the frame the engine reads it
in.**  SMM should write the in-plane basis expressed against the
parent's `xGrid`/`yGrid`, not in global coordinates — or, equivalently,
negate it when `xGrid · xs < 0`.  With R1 + R2 together, a freshly
generated deck routes correctly under either `GridType`, and the rule
above collapses to "always ok".

**R3 — deck decision.**  Options, increasing cost:
 - *(a) document only.*  Cheapest.  The permuted decks stay permuted;
   anything built on them stays wrong.
 - *(b) negate `SegXgrid` in the header of the 7 permuted decks.*  Zero
   geometry change, no regeneration, fixes routing **under today's
   engine** — but R1 would invert it again.
 - *(c) R1 + R2, then regenerate the corpus from `e5mono.in`.*  Correct
   end state; moves every pinned number that depends on segment index
   or Mon clocking.

**R4 — audit downstream products built on permuted decks.**  Any
`dw_dx`, MET influence matrix, edge-sensor matrix or shape-class result
computed on `e5hex1`/`e5hex2`/`e5seg1`/`test_in`-or-`view_rx_demo`
`e5pie` has its ring-segment columns permuted by the 180° map (2↔5,
3↔6, 4↔7 for a 1-ring tiling; ring-0 and the ring-2 edge segments of a
2-ring tiling are self-conjugate and unaffected).  `e5pie_polyap.in`
and `e5_seg_met.in` are **not** affected.

**R5 — pinned decks.**  `e5hex1.in` is pinned by the **GMI regression**
(`reference/nominal_e5hex1.mat`, `reference/zern_response_e5hex1.mat`);
`e5pie`/`e5hex2`/`ff_hex2`/`e2e_pie` appear in `tSegMirMaker`,
`tSegmentRx`, `tFingerprint`, `tDwDx`, `tDwDxGroups`,
`tRunSensitivities`, `tFreeFormComposite`, `tReadGridFile`.  Those
references are nominal-trace based, so they will neither catch the
permutation nor move under R3(b).

**R6 — collapse the three `e5pie` variants** to one, or rename so the
generation is visible in the filename.

---

## Reproducing

```matlab
addpath ~/dev/MACOS_res_dev/mmacos/src
addpath ~/dev/MACOS_res_dev/segmirmaker
R = seg_audit({'~/dev/macos/ZGD_test_files/e5pie.in'});   % ONE deck per process
seg_audit(decks, 'static_only', true);                    % no engine, whole corpus
```

Regenerating the reference output from the parent:

```bash
cd ~/dev/MACOS_resources/segmirmaker/test_in
MACOS_HOME=~/dev/macos/macos_f90 ../build_release_gfortran/SegMirMaker < e5pie.stdin
```
