# mmacos sensitivities — multi-field wavefront-sensitivity Jacobians

Generic, ready-to-adapt drivers for the four wavefront-sensitivity
("`dW/d…`") channels MACOS exposes through mmacos. Each script runs the
corresponding `macos.dw_d*_multi` supervisor over a set of field points and
produces a canonical state-vector Jacobian plus two figures.

| Script | Channel | DOFs | underlying driver |
|---|---|---|---|
| `run_dwdx_multi.m`    | rigid body      | Rx Ry Rz Tx Ty Tz per optic | `macos.dw_dx_multi` |
| `run_dwdz_multi.m`    | Zernike coef    | (element, Zernike mode)      | `macos.dw_dz_zernike_multi` |
| `run_dwdsurf_multi.m` | powered surface | Kr, Kc per powered optic     | `macos.dw_dsurf_multi` |
| `run_dwdgrid_multi.m` | grid data       | (element, influence poke)    | `macos.dw_dgrid_multi` |

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
bundled `e5hex1` demo. Other knobs: `MODEL` (≥ your aperture grid sampling;
use 256+ for grid-data prescriptions), `NGRIDPTS` (ray-grid sampling
override, e.g. `NGRIDPTS = 63;` — replaces the `.in`-file `nGridpts` /
model-limit default; `[]` keeps the `.in` value; engine-clamped to
[3, model-size limit]), `FOV` (half-field in rad for the four
corner field points), `DELTA` (finite-difference step), and the channel-
specific set (`DOFS` / `KINDS`+`ZSTART`+`ZEND` / `PARAMS` / `ZMODES`+`INFL`).
The same override is available programmatically on every driver —
`macos.dw_dx[_multi]`, `dw_dz_zernike[_multi]`, `dw_dsurf[_multi]`,
`dw_dgrid[_multi]` — as `'ngridpts', N` (applied once after the Rx load;
persists across the per-field calls).

Run with `>> run('run_dwdgrid_multi.m')` (or any of the four).

## Outputs (written next to the script)

- `*_OPDall.png` — the nominal OPD at every field point (the tiled field
  canvas).
- `*_channels.png` — **each channel's multi-field sensitivity** `dW/d(param)`,
  one subplot per (element, DOF), reconstructed onto the field canvas.
- `*_<rx>.mat` — the canonical state-vector layout

  ```
  wall = dwdxall * x + w0_stacked
  ```

  consumable directly by the shared `examples/sensitivities/verifyall.m`.

## Per-field exit-pupil reset (`reset_xp`)

An off-axis field tilts the chief ray, which — if the OPD keeps referencing
the on-axis exit pupil — shows up as a large linear-in-field **wavefront
tilt** that swamps the real aberration (e.g. ~265 waves at ±100 µrad on
e5hex1, vs ~0.01 waves of true residual). `dw_dz_zernike_multi`,
`dw_dsurf_multi` and `dw_dgrid_multi` therefore default `reset_xp=true`:
before differencing each field they re-find the exit pupil on that field's own
chief ray (`fex`), removing the field tilt from the nominal **while retaining a
poke's own tilt** (the reference is fixed per field, not re-fit after a poke).
This needs a STOP set and >3 elements. `dw_dx_multi` does **not** take
`reset_xp` — its `FocalPlaneChannel` already re-references the exit pupil per
perturbation (`fp_mode='track'` → `sxp`).

## Gotcha: `ZEND` / `n_zcoef` is the END mode

In `run_dwdz_multi.m`, `ZEND` (the driver's `n_zcoef`) is the **highest** mode,
not a count: modes `ZSTART:ZEND` are taken (`4:9` → Z4..Z9, which includes the
comas Z7/Z8/Z9).

## See also

- `examples/sensitivities/e5hex1/` — the concrete e5hex1 fixtures and
  `verifyall.m` display/round-trip checker.
- `help macos.dw_dgrid_multi` (and the `dw_dx`/`dw_dz_zernike`/`dw_dsurf`
  siblings) for the full name-value reference, including alternative field
  sets (`'grid','3x3'`, `'fields','list.txt'`).
