# zoom_5x5 — multi-configuration × multi-field sensitivities

The fixture for a CONFIGURATIONS axis on the `dw_d*` supervisors: a
sensitivity Jacobian evaluated per (zoom position, field point) rather
than per field alone.  Design sketch and open questions:
[`../../../design/PLAN_CONFIGURATIONS.md`](../../../design/PLAN_CONFIGURATIONS.md)
§6.

**Status:** deck staged; the `run_dwd*_5zoom_5fov.m` drivers are not
written yet — they are gated on the API question in that document's §5.1
(is a compensation state a configuration, or a DOF?).

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

## Two things a driver here must do

- **Set the stop.**  The deck carries no `ApStop=`, and `reset_xp`
  requires one.  The FSM is the pupil.
- **Restore the configuration element.**  Element 25 is also one of the
  elements whose rigid-body DOFs are Jacobian channels, so a
  configuration moves an optic that is itself a variable.  That is what a
  zoom-dependent sensitivity IS, but the snapshot/restore must cover it,
  and the restore assertion must run *after* the channel loop has undone
  its own poke.
