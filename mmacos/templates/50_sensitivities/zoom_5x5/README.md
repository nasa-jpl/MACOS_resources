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
| `run_dwdx_5zoom_5fov.m` | rigid-body 6-DOF | **runs** — 132 channels, 25 blocks |
| `run_dwdsurf_5zoom_5fov.m` | Kr / Kc | **runs** — 4 channels, 25 blocks |
| `run_dwdz_5zoom_5fov.m` | MonZernike figure | **runs** — segments carry zero-amplitude MonZern channels |
| `run_dwdgrid_5zoom_5fov.m` | segment grid | **runs** — grid-augmented in each segment's clocked Mon frame |

Every driver is resumable: per-configuration checkpoints land in
`_resume*/` and are pruned on success.

### The two figure rungs run on the promoted segments

**Since 2026-08-21** the deck's 19 segments are `Surface= FreeForm`
carrying a per-segment MonZernike figure channel (`MonZernType= BornWolf`,
zero coefficients) in each segment's own clocked Mon frame — so the
MonZernike rung (`dw/dz`, which targets `find_freeform_elts`) and, after
`macos.design.grid_augment_rx`, the segment-grid rung (`dw/dgrid`, which
targets `find_grid_elts`) both harvest here.  The figure DOFs are
**zero-amplitude fixture channels with no design authority**; the
promotion from the original `Surface= Conic` is optically **inert** (a
zero-coefficient FreeForm computes the identical conic sag — verified to
~7e-11 mm, sub-picometer, against the Conic trace).

The promotion was applied by `macos.design.promote_segments_freeform`
and is gated three ways in `tRunSensitivities`: the trace is inert
(`test_zoom_fixture_promotion_is_inert`), both rungs harvest non-empty
live Jacobians (`test_promoted_fixture_feeds_both_figure_rungs`), and a
single-mode poke localizes to its segment rather than de-localizing to
the aperture centre (`test_promoted_segment_poke_localizes`).  The
rigid-body (`dw/dx`) and prescription-parameter (`dw/dsurf`) rungs are
unaffected, since the conic base and every element pose are unchanged.

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
wavefront collapses from **2.7e-02 mm** (pupil frozen) to **2.3e-07 mm**,
and its effect on the Jacobian is the second-order residual, **1.7e-05
relative** (2.4e-05 frozen).  That residual is the quantity a
compensation-state sensitivity study wants — the first-order term is what
the compensator is FOR.  The §6 feasibility table below was measured with
the pupil frozen, which is why its numbers are the larger ones.

## Two things a driver here must do

- **Set the stop.**  The deck carries no `ApStop=`, and `reset_xp`
  requires one.  The FSM is the pupil.
- **Restore the configuration element.**  Element 25 is also one of the
  elements whose rigid-body DOFs are Jacobian channels, so a
  configuration moves an optic that is itself a variable.  That is what a
  zoom-dependent sensitivity IS, but the snapshot/restore must cover it,
  and the restore assertion must run *after* the channel loop has undone
  its own poke.
