# ns_griddata — a GridData (Zernike-grid) figure imaged at the exit pupil

Demonstrates a **grid-data surface figure** (`GridFile=`) on a segment of a
double-pass system (`iris_dp_ZGD.in`: segment **A1** is an `Element=NSReflector`,
`Surface=ZrnGrData` hit twice, iElt 17 & 35), and shows it in the **ExitPupil
OPD** by comparing the figure **ON** (`zern41em5z155em3.txt`) vs **OFF**
(`flat.txt`) — the difference is the grid figure's contribution to the
wavefront.

Run (after `mmacos_setup`):

```matlab
mmacos_setup
run ns_griddata      % OPD_on, OPD_off, and their difference at the ExitPupil
```

Outputs `ns_griddata_opd_on_off_diff.png` (3-panel: OPD on / off / difference)
and `ns_griddata.mat`.

## Files

| File | Role |
|---|---|
| `iris_dp_ZGD.in` | prescription; A1 grid = `zern41em5z155em3.txt` (figure ON) |
| `iris_dp_ZGD_flat.in` | same, A1 grid = `flat.txt` (figure OFF) |
| `zern41em5z155em3.txt` | the N×N grid figure (a steep z41), `GridSrfdx=0.2` |
| `flat.txt` | an all-zero N×N grid (the OFF baseline) |
| `ns_griddata.m` | the driver (uses `macos.opd_psf`) |

## The reusable utility

`ns_griddata.m` calls **`macos.opd_psf`** (in `src/+macos/`) — a small, general
front-end for any `.in`: it loads the Rx, optionally cleans up the exit pupil
with FEX, traces, and returns/displays/saves the OPD (and, optionally, a PSF via
INTENSITY):

```matlab
macos.opd_psf('iris_dp_ZGD.in', 'wf_elt', 55, 'psf', true, ...
              'save_png', true, 'save_mat', true);
```

## What this example caught

A `GridFile=` line with an inline comment kept the trailing **tabs** before the
`%` — e.g. `GridFile=  zern41em5z155em3.txt<TAB><TAB>% flat.txt`.  The parser
stored `"zern41em5z155em3.txt<TAB><TAB>"`, `GridInit` reported *"does not
exist"*, zeroed the grid, and the figure was **silently dropped** (identical OPD
with the figure on or off).  Fixed in the engine (strip trailing tabs from
`GridFile=` in `msmacosio.inc` + `GridInit`); a clean/`SAVE`'d Rx works.

> **Note on the aperture stop:** the ExitPupil OPD needs a correct `ApStop=` in
> the Rx header.  A wrong/absent stop can push the source outside the usable
> aperture so every ray is obscured — then `opd` reports a `9.9999D+36` RMS
> (and, unguarded, can segfault).  Find the stop interactively with
> `STOP <iElt>`, but note that **`SAVE` does not currently round-trip `ApStop=`
> (or comments)** — a SAVE'd Rx comes back with no `ApStop=` and no comments —
> so add the `ApStop=` line to the Rx directly.  (Both the ApStop-SAVE gap and
> the opd-segfault-on-zero-rays are tracked engine TODOs.)
