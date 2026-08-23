# mmacos sensitivities — multi-field wavefront-sensitivity Jacobians

Generic, ready-to-adapt drivers for the four wavefront-sensitivity
("`dW/d…`") channels MACOS exposes through mmacos. Each script runs the
corresponding `macos.dw_d*_multi` supervisor over a set of field points and
produces a canonical state-vector Jacobian plus figures.

> **Single source (2026-07-19).** Every script below is now a thin
> CONFIG wrapper over the sensitivity **stage runner**
> `design/runners/run_sensitivities.m` — the same code the design
> pipeline (design → segmentation → sensitivities → MET → …) uses.
> The CONFIG-block interfaces are unchanged. What you gain from the
> runner: a conditioning report (all-column AND segment-only),
> per-segment column norms, piston-removed plots with per-element
> pages collected in a `<name>_pages/` folder, automatic ApStop
> injection for stop-less (SMM-corpus) fixtures, and — for `dwdgrid`
> — grid augmentation in each segment's CLOCKED Mon frame with the
> span sized from the parent Aperture, replacing the stale
> parent-frame grid lines SegMirMaker replicates into segment blocks
> (poking those paints a "central dot" and rank-collapses the
> Jacobian).

| Script | Channel | DOFs | underlying driver |
|---|---|---|---|
| `run_dwdx_multi.m`      | rigid body      | Rx Ry Rz Tx Ty Tz per optic | `macos.dw_dx_multi` |
| `run_dwdx_multi_zoom.m` | rigid body      | the same, per (CONFIGURATION, field) | `macos.dw_dx_multi` |
| `run_dwdz_multi.m`      | Zernike coef    | (element, Zernike mode)      | `macos.dw_dz_zernike_multi` |
| `run_dwdsurf_multi.m`   | powered surface | Kr, Kc per powered optic     | `macos.dw_dsurf_multi` |
| `run_dwdgrid_multi.m`   | grid data       | (element, influence poke)    | `macos.dw_dgrid_multi` |

> **Self-contained examples + `mmacos_setup` (2026-06).** Per-driver copies in
> `examples/` each ship their own `.in`, set the path via the repo-root
> `mmacos_setup` (run once per MATLAB session — no `addpath` in the script), and
> also emit single-page-**per-element** *center* and *multi* plots via the generic
> `plot_dw_per_element` helper (one page per optic/segment, parula + zero-mask).

## Use it on your own system

Open any script and edit the **CONFIG block** at the top — the only line you
*must* change is:

```matlab
RX = '';   % <-- YOUR .in FILE GOES HERE (absolute path)
```

Everything below the CONFIG block is generic. Leave `RX` empty to run the
bundled demo — `e5hex1` for four of the scripts, the 18-segment
`jwst_ote_designc` for `run_dwdx_multi_zoom.m`, which needs a deck with an
element worth putting a configuration schedule on. Other knobs: `MODEL`
(≥ your aperture grid sampling; use 256+ for grid-data prescriptions),
`NGRIDPTS` (ray-grid sampling
override, e.g. `NGRIDPTS = 63;` — replaces the `.in`-file `nGridpts` /
model-limit default; `[]` keeps the `.in` value; engine-clamped to
[3, model-size limit]), `FOV` (half-field in rad for the four
corner field points), `DELTA` (finite-difference step — see the note
below), `GROUPS` (rigid-body element groups, `dw/dx` only), and the
channel-specific set (`DOFS` / `KINDS`+`ZSTART`+`ZEND` / `PARAMS` /
`ZMODES`+`INFL`).  `run_dwdx_multi_zoom.m` adds `STOP_ELT`, `CFG_ELT`,
`TILT` and the `SCHEDULE` table.
The same override is available programmatically on every driver —
`macos.dw_dx[_multi]`, `dw_dz_zernike[_multi]`, `dw_dsurf[_multi]`,
`dw_dgrid[_multi]` — as `'ngridpts', N` (applied once after the Rx load;
persists across the per-field calls).

Run with `>> run('run_dwdgrid_multi.m')` (or any of the five).

## Outputs (written next to the script; runner naming, 2026-07-19)

- `<name>_sens_report.txt` — sizes, conditioning (all-column AND
  segment-only), per-segment column norms.
- `<name>_opdall.png` — the nominal OPD at every field point (tiled
  field canvas); `<name>_svspec.png` — singular-value spectra.
- `<name>_<ch>_channels.png` — each channel's multi-field sensitivity,
  one subplot per (element, DOF), piston removed.
- `<name>_pages/` — the per-element single-page maps (center AND
  multi field; the pages are numerous, so they get their own folder).
- `<name>_sens.mat` — the supervisor outputs (`ox`/`oz`/`og`/`os`) in
  the canonical state-vector layout

  ```
  wall = dwdxall * x + w0_stacked
  ```

- `<name>_grid.in` + `flat<ng>.txt` — the grid-augmented Rx (dwdgrid
  channel; grids in each segment's CLOCKED Mon frame, span from the
  parent Aperture — the dxGrid convention).
- `<name>.mat` (`run_dwdx_multi_zoom.m`) — the FLAT, channel-named save:
  `dwdx` / `indxall` / `w0_stacked` / `channel_names` / `config_*` at the
  top level, no wrapper struct (`save_dw_flat`).

All of the above are demo scratch and `.gitignore`d here — the scripts are
what this directory ships.

## Element groups (`GROUPS`) — rigid-body assemblies

`GROUPS` (in `run_dwdx_multi.m` and `run_dwdx_multi_zoom.m`) is a
`containers.Map` of *name* → column vector of member element ids. Each
group is perturbed as ONE rigid body by the engine's `GPERTURB` and
contributes **6 more columns**, appended **after** the per-element block
in every field's (and every configuration's) block.

That is the sensitivity an assembly actually has — a bonded lens cell, a
mirror backplane, a camera. The members' responses partly **cancel** when
they move together, and summing their individual columns cannot reproduce
that: on the bundled decks a cell is **7.5× less** tilt-sensitive than its
own front surface, and an 18-segment primary is **23× less**
piston-sensitive than one segment. A per-element budget overstates both.

`GROUPS` defaults to `[]` in both scripts because a group is
system-specific; each CONFIG block carries a commented line naming the
bundled demo deck's group. `GROUPS_AUTO = true` (`run_dwdx_multi.m`)
instead parses `EltGrp=` declarations out of the Rx, which is the right
switch for a deck that carries them.

Group channels carry **no element id** — `out.iElt` is `0`, the value a
*source* channel also carries — and `out.kind` is `'Group'`. **Section on
`kind`, not on `iElt`.** The per-element pages do that and give each group
its own page. Units are the same on both sides: OPD-per-metre for
translations, OPD-per-rad for rotations, so one numeric `DELTA` is one
physical poke for either.

With a group set, the driver appends a **group-vs-member table** to
`<name>_sens_report.txt` (`group_exhibit`): the group's six column norms
beside a member's, and the ratio. A ratio **below 1** is intra-group
compensation; a ratio at ≈ N (the member count) is a rigid motion adding
up, which is equally expected.

Committed, worked examples with the numbers:
`templates/50_sensitivities/run_dwdx_multi/` (a lens cell) and
`templates/50_sensitivities/zoom_5x5/` (a PM backplane, 5 zoom states ×
5 fields).

## The configuration axis (`run_dwdx_multi_zoom.m`)

A CONFIGURATION is a named set of element setting overrides: a zoom
position, or — more often in our systems — a **compensation state**, e.g.
a steering mirror at a pupil re-pointed to cancel pointing drift. You
write them as a SCHEDULE (one row per configuration, one column per
`<elt>.<DOF>`) and `macos.design.configs_from_table` turns that into the
`configs` array the supervisors take; a row of zeros is a legal
configuration — the nominal state.

Configurations stack as extra **ROWS**, never a third array dimension: a
configuration adds observations of the same state vector `x`, exactly as a
field point does, so every downstream consumer reads the result unchanged.
`w` for one configuration stacks its fields, `w` for the run stacks the
configurations — address one block with `out.indxall.config == c`. The
canvas is tiled differently and deliberately so: each configuration sits at
its own position on an outer grid holding that configuration's whole field
canvas, so `_opdall.png` is a grid of grids.

The run is **resumable** — each configuration's block is checkpointed and
reloaded rather than recomputed after a kill (`RESUME_DIR`; `""` disables,
and the directory is pruned on success). On the bundled deck a 5×5-block
harvest is about 165 s, so the checkpoints matter mainly once `NGRIDPTS`
or `MODEL` goes up.

## A note on `DELTA`

Both rigid-body scripts default to the `(1,6)` form
`[1e-8 1e-8 1e-8 1e-6 1e-6 1e-6]` = `[Rx Ry Rz Tx Ty Tz]`, rotations in
rad and translations in SI metres. A scalar `1e-8` is *too small a
translation poke* on both decks in this repo: the per-element translation
columns land 2.5e-3 (e5hex1) and 1.9e-04 (zoom deck) away from their
converged values, while 1e-6 and 1e-5 agree with each other to ~1e-5.
Rotations show no such drift, so only the translation entries move. The
floor scales with your coordinate magnitudes — worth re-checking on your
own deck.

## Per-field exit-pupil reset (`reset_xp`)

An off-axis field tilts the chief ray, which — if the OPD keeps referencing
the on-axis exit pupil — shows up as a large linear-in-field **wavefront
tilt** that swamps the real aberration (e.g. ~265 waves at ±100 µrad on
e5hex1, vs ~0.01 waves of true residual). `dw_dz_zernike_multi`,
`dw_dsurf_multi` and `dw_dgrid_multi` therefore default `reset_xp=true`:
before differencing each field they re-find the exit pupil on that field's own
chief ray (`fex`), removing the field tilt from the nominal **while retaining a
poke's own tilt** (the reference is fixed per field, not re-fit after a poke).
This needs a STOP set and >3 elements. `dw_dx_multi` takes `reset_xp`
too, and also defaults it to `true` (the claim here that it does *not* was
stale — it gained the option with the rest of the family). Its
`FocalPlaneChannel` separately re-references the exit pupil per
perturbation (`fp_mode='track'` → `sxp`); the two compose, with the
per-field EP written *before* the channel builds its columns.

## Gotcha: `ZEND` / `n_zcoef` is the END mode

In `run_dwdz_multi.m`, `ZEND` (the driver's `n_zcoef`) is the **highest** mode,
not a count: modes `ZSTART:ZEND` are taken (`4:9` → Z4..Z9, which includes the
comas Z7/Z8/Z9).

## See also

- `templates/50_sensitivities/e5hex1/` — the concrete e5hex1 fixtures and
  `verifyall.m` display/round-trip checker.
- `help macos.dw_dgrid_multi` (and the `dw_dx`/`dw_dz_zernike`/`dw_dsurf`
  siblings) for the full name-value reference, including alternative field
  sets (`'grid','3x3'`, `'fields','list.txt'`).
