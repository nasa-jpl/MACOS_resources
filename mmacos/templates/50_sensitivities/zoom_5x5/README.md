# zoom_5x5 — multi-configuration × multi-field sensitivities

The fixture for a CONFIGURATIONS axis on the `dw_d*` supervisors: a
sensitivity Jacobian evaluated per (zoom position, field point) rather
than per field alone.  Design sketch and open questions:
[`../../../design/PLAN_CONFIGURATIONS.md`](../../../design/PLAN_CONFIGURATIONS.md)
§6.

**Status:** shipped.  The `'configs'` option is on all four
`macos.dw_d*_multi` supervisors and on `run_sensitivities`; the four
`run_dwd*_5zoom_5fov.m` drivers here are thin wrappers over it.

| driver | rung | on THIS deck |
|---|---|---|
| `run_dwdx_5zoom_5fov.m` | rigid-body 6-DOF | **runs** — 132 channels (21 optics × 6 + the PM group's 6), 25 blocks |
| `run_dwdsurf_5zoom_5fov.m` | Kr / Kc | **runs** — SM (M2) + TM (M3) each in Kr & Kc = 4 channels, piston/tip/tilt removed |
| `run_dwdz_5zoom_5fov.m` | MonZernike figure | **runs** — 20 optics × MODES (segs + SM + TM) |
| `run_dwdgrid_5zoom_5fov.m` | segment + optic grid | **runs** — segments share a basis, SM/TM each own one |

Every driver is resumable: per-configuration checkpoints land in
`_resume*/` and are pruned on success.  Each writes a **flat**, channel-
named `<name>.mat`: the Jacobian at the top level under its own name
(`dwdx` / `dwdz` / `dwdsurf` / `dwdgrid`), with `indxall` / `w0_stacked` /
`channel_names` / `config_*` beside it — no `ox`/`og` wrapper struct
(`sensitivities/save_dw_flat`).

### The figure rungs run on the promoted optics

**Since 2026-08-21** the deck's 18 real segments (elts 5–22) and the SM
(elt 23) and TM (elt 24) are `Surface= FreeForm` carrying a MonZernike
figure channel (`MonZernType= BornWolf`, zero coefficients) — so the
MonZernike rung (`dw/dz`, targeting `find_freeform_elts`) and the
segment-grid rung (`dw/dgrid`, targeting `find_grid_elts`) both harvest
them.  The SM and TM additionally carry a grid channel centred and sized
to their **traced footprint** (all fields × all zoom configurations — a
vertex is not the beam centre), and get their **own full-aperture** grid
basis, while the segments share one bespoke per-segment basis.  The
figure DOFs are **zero-amplitude fixture channels with no design
authority**; the promotion from `Surface= Conic` is optically **inert** (a
zero-coefficient FreeForm computes the identical conic sag — verified to
~5e-11 mm, sub-picometer, against the Conic trace).

**Element 4 (CenterSegment) stays `Surface= Conic`.**  It is a *virtual*
element — not a real telescope segment, almost entirely obscured (it
passes only the chief-ray sliver, ~2.5 % of the beam), included to pass
the chief ray and reference the PM — so its sensitivities are ~zero.  It
carries no figure channel, and the rungs that would still list it
(`dw/dx`) drop it **number-free**: `flag_zero_norm_channels` flags any
all-zero channel group by its *response* (elt 4's dw/dx column norms are
~1e-7 vs ~0.2 for a real segment), and `drop_channels` removes it — no
element number is hard-coded in the drivers.

The promotion was applied by `macos.design.promote_segments_freeform`
(Rx in, frames + lMon derived by tracing) and is gated in
`tRunSensitivities`: the trace is inert
(`test_zoom_fixture_promotion_is_inert`), both rungs harvest live
Jacobians incl. SM/TM (`test_promoted_fixture_feeds_both_figure_rungs`),
single-mode pokes localize (`test_promoted_segment_poke_localizes`), and
the flat `.mat` layout is checked (`test_save_dw_flat_layout`).  The
rigid-body (`dw/dx`, minus the obscured elt 4) and prescription-parameter
(`dw/dsurf`) rungs are otherwise unaffected — the conic base and every
pose are unchanged.

### Element groups (rigid-body) — ON, on the `dw/dx` rung

`run_sensitivities` takes `'groups'` (a `containers.Map` name → column
vector of element ids) and `'groups_auto'` (parse `EltGrp=` out of the
deck), plus `group_coords` / `group_fp_mode` / `group_stop_mode` /
`group_stop_pos`.  They reach the **`dwdx` channel only** — a group is a
RIGID-BODY group, driven by the engine's `GPERTURB`; `dwdz` / `dwdsurf` /
`dwdgrid` are figure and surface channel kinds with no group analogue,
and grouping them is deferred.

**`run_dwdx_5zoom_5fov.m` ships with one group ON: the PM.**  The deck's
18 real segments (elts 5–22) are declared as `'PM'`, so the harvest
carries — alongside each segment's own 6 DOFs — the six columns of the
primary-mirror **backplane** moving as a single body.  That is the
sensitivity a pointing/alignment budget spends: a backplane thermal tilt
is one rigid motion of the whole PM, not 18 independent segment motions
that happen to agree.  Element 4 (CenterSegment) is deliberately **not**
a member — it is the virtual, almost-entirely-obscured element the
zero-norm flag drops anyway.

```matlab
GROUPS = containers.Map('KeyType','char','ValueType','any');
GROUPS('PM') = (5:22).';
```

A group contributes **6 more columns**, appended **after** the
per-element block, in every field's block and every configuration's
block — so the stacked column order is `[per-element] [group]` and the
supervisor's channel-identity assertion covers the group columns too.
The committed report shows `dwdxall 54585 x 138` (was 132 ungrouped);
the saved flat `.mat` has **132** channels — 126 per-element after the
elt-4 drop, plus the group's 6.  Group channels carry **no element id**:
`out.iElt` is `0` (the value source channels also carry) and `out.kind`
is `'Group'` — section on `kind`, not on `iElt`.  The per-element figure
pages do exactly that and give the group its own page,
`<name>_grpPM_center.png`; `drop_channels` never touches a group,
because a group column is a distinct rigid-body motion and not a sum of
its members' columns.

**What the PM columns say** (RMS over the 5×5 stack; rotations per rad,
translations per metre after dividing the group columns by CBM — see the
units note below):

| DOF | PM as one body | one segment (elt 5) | PM / segment |
|---|---|---|---|
| Rx | 3.0114e+00 | 1.6132e-01 | 18.6667 |
| Ry | 3.0948e+00 | 1.6289e-01 | 18.9996 |
| Rz | 2.4853e-01 | 2.1582e-05 | 11515.6161 |
| Tx | 1.9317e-01 | 1.0145e-02 | 19.0411 |
| Ty | 1.8801e-01 | 1.0118e-02 | 18.5822 |
| Tz | 1.9739e-02 | 4.6090e-01 | **0.0428** |

The table is appended to the committed `_sens_report.txt` by the driver
(`sensitivities/group_exhibit`), so every figure here is greppable in the
artifact.  Tilt and decenter come out at ≈ N = 18× a single segment,
which is what a rigid motion of N alike members must give.  **Piston is
the interesting one: the whole PM is 23× LESS piston-sensitive than one
segment.**  A single segment pistoning puts a step into the wavefront;
the whole PM
pistoning is a global despace that the exit-pupil reference
largely absorbs.  That is exactly the intra-group cancellation a per-element
budget cannot see — summing 18 large per-segment piston columns does not
reproduce it.

**Units, and why `DELTA` is a `(1,6)` vector.**  A group TRANSLATION
column is OPD per **BaseUnit** (the engine's `prb_grp` takes BaseUnits),
while a per-element translation column is OPD per **metre** — 1000×
apart on this millimetre deck.  Rotations are rad on both sides.  So a
*scalar* `delta` pokes the group 1000× smaller than the elements and
drives its columns toward the finite-difference floor: measured here
against a converged 1e-4 step, the PM group's translation columns are off
by 4.0e-03 (Tx) / 3.7e-03 (Ty) / 3.2e-02 (Tz) at `delta` 1e-8, and by
3.9e-05 / 3.9e-05 / 3.3e-04 at 1e-6.  The driver therefore uses
`[1e-8 1e-8 1e-8 1e-6 1e-6 1e-6]` — rotations 1e-8 rad, translations 1 µm
at the elements and 1 nm at the group.  The per-element translation
columns improve too (worst 1.9e-04 off at the old scalar 1e-8, 1.8e-06 at
1e-6), and the per-segment column-norm table in the report is unchanged
to three figures.

**The committed artifacts here WERE regenerated for this** — report,
PNGs, and the (gitignored) `.mat`.  The 25-block harvest takes about
165 s as shipped (measured 164 s, Linux, gfortran-built engine).  The
gated case
`tRunSensitivities/test_groups_reach_the_dwdx_channel` covers the
bookkeeping; `tDwDxGroups` covers the channel physics.

## The deck

`jwst_ote_designc.in` — an early 18-segment JWST OTE design.  Element 25
is a flat fine-steering mirror at a pupil, which is what the zoom axis
moves; element 27 (`nElt-1`) is the `ExitPupil` `Return`, which is where
`reset_xp` writes.

It is **not the flight prescription** — 15 mm segment gaps and 1313.25 mm
flat-to-flat segments, against the flight 7 mm and 1.32 m.  The three
powered mirrors do match the published JWST OTE prescription to the
published precision (McElwain et al. 2023, PASP **135**, 058001, Table 2,
open access; design source TRW); the fold mirrors do not — they were
added to unfold the train and carry no design authority.  The deck header
carries the full comparison.

**It is a load case, not a design.**  At ±1 arcmin the wavefront error is
0.64 mm, about 278 waves at the deck's 2.3 µm.  The point of the fixture
is that it traces cleanly and responds on both axes, not that the numbers
mean anything optically.

## Configuration and field grid

Five configurations — centred, then the FSM tilted 0.5 arcmin
(1.45444e-4 rad) to each corner of a square, LOCAL frame — crossed with
the stock five-field set at 1 arcmin (2.90888e-4 rad).  25 blocks.

**The canvas is tiled the way the field set is.**  Each zoom state sits
at its own position on an outer 3×3 grid — four corners and the centre —
and each of those cells holds that state's whole five-field canvas, so
`_opdall.png` is a quincunx of quincunxes and position on the page means
(zoom state, field point).  The stacked ROW order is a different walk and
deliberately so: `w` for one zoom stacks its **fields**, `w` for the run
stacks the **zooms**, so each zoom keeps a contiguous block of rows even
though its tile does not lie along a single canvas column.  Address a
block with `indxall.config == c`.

## Measured, at model 512 / `ngridpts` 63 / stop at element 25, OPD at 27

| state | rays | valid | lost | RMS WFE (mm) |
|---|---|---|---|---|
| nominal | 2301 | 2184 | 0 | 6.846e-06 |
| FSM 0.5′ corners (4) | 2301 | 2184 | 0 | 1.457e-02 … 1.460e-02 |
| field ±1′ corners (4) | 2301 | 2182–2184 | 0 | 6.381e-01 … 6.393e-01 |

No ray loss in any of the eight perturbed states.  117 rays are obscured
throughout (the central obscuration), and the chief ray runs
`LRayOK=1, LRayPass=0, RayStatus=Obscured` — so the fixture also
exercises the obscured-chief OPD reference
([`../../../doc/opd_conventions.md`](../../../doc/opd_conventions.md)
§1.2).

## Why the five blocks look alike, and why that is right

The supervisors re-find the exit pupil PER FIELD (`reset_xp`, default
true), and a tilt of a FLAT mirror AT A PUPIL is to first order exactly a
wavefront tilt — which that re-reference removes.  Measured on this
fixture with a 0.5′ FSM tilt: the configuration's effect on the nominal
wavefront collapses from **3.033e-02 mm** (pupil frozen) to **4.043e-06
mm** — a factor 7500 — and its effect on the Jacobian drops to the
second-order residual, **5.308e-06 relative** against 1.886e-04 frozen.
Both legs come from one script with the same statistic on each side:
nominal effect = max over fields and configurations of |W(cfg) − W(z0)|
at pixels valid in both; Jacobian effect = the same max, relative, over
the per-element columns.  (Earlier revisions quoted 2.7e-02 / 2.3e-07 /
1.7e-05 / 2.4e-05 from a statistic that was not recorded; the definition
above is stated so the numbers can be reproduced.)  That residual is the
quantity a compensation-state sensitivity study wants — the first-order
term is what the compensator is FOR.  The §6 feasibility table below was
measured with the pupil frozen, which is why its numbers are the larger
ones.

## Two things a driver here must do

- **Set the stop.**  The deck carries no `ApStop=`, and `reset_xp`
  requires one.  The FSM is the pupil.
- **Restore the configuration element.**  Element 25 is also one of the
  elements whose rigid-body DOFs are Jacobian channels, so a
  configuration moves an optic that is itself a variable.  That is what a
  zoom-dependent sensitivity IS, but the snapshot/restore must cover it,
  and the restore assertion must run *after* the channel loop has undone
  its own poke.
